#!/usr/bin/env python3
"""Unsloth fine-tuning on a full-precision (bf16 / fp16) base: LoRA or GCMP.

For customers who train on the original 16-bit checkpoint rather than a 4-bit
one. Both adapters share every Unsloth optimization that does not depend on
LoRA, and differ only in the adapter:

  * FastModel 16-bit loading with Unsloth's patched kernels (RMSNorm, RoPE,
    fused cross-entropy) and compiled decoder blocks
  * padding-free batching (auto) and optional sequence packing (--packing)
  * Unsloth smart gradient checkpointing with activation offload
  * fused or 8-bit AdamW (--optim)
  * LoRA additionally uses Unsloth's fused LoRA MLP/QKV kernels; GCMP uses the
    licensed native my_qlinear::gcmp_* operators when cq_kernels is installed.

Entry points: train_unsloth_16bit_lora.py and train_unsloth_16bit_gcmp.py
(or this file with --adapter). Same data format as train_unsloth_nf4.py:
JSONL with a "text" field.

Memory: a 16-bit base needs ~2 bytes per parameter for weights alone
(Qwen3.8-27B ~ 54 GB), so 27B-class models need an 80 GB GPU; on a 48 GB A40
use a smaller model or the 4-bit script.
"""
import os
import sys

if '--unsloth-compile-disable' in sys.argv:  # must be set before importing unsloth
    os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'

import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
from unsloth import FastModel  # noqa: E402  (import before transformers/trl)
from trl import SFTConfig  # noqa: E402

from train_unsloth_nf4 import (TARGETS, ComparisonLogger, CountedSFTTrainer,  # noqa: E402
                               attach_gcmp, read_jsonl)

DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}


def parse_args(argv=None, adapter=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', required=True, help='16-bit HF/Unsloth checkpoint (not a 4-bit or CQ repo)')
    p.add_argument('--train-data', required=True)
    p.add_argument('--eval-data')
    p.add_argument('--output', required=True)
    if adapter is None:
        p.add_argument('--adapter', choices=['lora', 'gcmp'], required=True)
    p.add_argument('--dtype', choices=sorted(DTYPES), default=None,
                   help='default: bfloat16 if supported, else float16')
    p.add_argument('--targets', default=','.join(TARGETS))
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--optim', default='adamw_torch_fused',
                   help='adamw_torch_fused (default), adamw_8bit (Unsloth/bitsandbytes), adamw_torch')
    p.add_argument('--packing', action='store_true', help='pack sequences (changes loss averaging)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-steps', type=int, default=50)
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--save-total-limit', type=int, default=None)
    p.add_argument('--benchmark-warmup-steps', type=int, default=10)
    # LoRA
    p.add_argument('--lora-rank', type=int, default=32)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.0)
    p.add_argument('--use-rslora', action='store_true')
    # GCMP
    p.add_argument('--gcmp-scale', type=float, default=1.0)
    p.add_argument('--kernel-module', default='cq.cq_kernels',
                   help='GCMP: extension with my_qlinear::gcmp_* (torch fallback if absent)')
    p.add_argument('--unsloth-compile-disable', action='store_true',
                   help='disable Unsloth block compilation (fallback if compile fails with GCMP)')
    a = p.parse_args(argv)
    if adapter is not None:
        a.adapter = adapter
    if a.dtype is None:
        a.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    a.targets = [t for t in a.targets.split(',') if t]
    return a


def load_16bit(a):
    model, tokenizer = FastModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.max_length,
        dtype=DTYPES[a.dtype],
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=a.dtype != 'float32',
        full_finetuning=False,
    )
    quantized = [n for n, m in model.named_modules() if type(m).__name__ in ('Linear4bit', 'Linear8bitLt')]
    if quantized or getattr(model, 'is_quantized', False):
        raise RuntimeError(f'Expected a full-precision base, found quantized layers: {quantized[:3]}; '
                           'use train_unsloth_nf4.py for 4-bit checkpoints')
    dtypes = {m.weight.dtype for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in a.targets}
    gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    print(f'Loaded {a.dtype} base: target weight dtypes {sorted(map(str, dtypes))}, ~{gib:.1f} GiB of weights')
    return model, tokenizer


def attach_lora(model, a):
    return FastModel.get_peft_model(
        model,
        r=a.lora_rank,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        target_modules=a.targets,
        bias='none',
        use_gradient_checkpointing='unsloth',
        use_rslora=a.use_rslora,
        random_state=a.seed,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )


def main(argv=None, adapter=None):
    a = parse_args(argv, adapter)
    os.makedirs(a.output, exist_ok=True)
    train_ds = read_jsonl(a.train_data)
    eval_ds = read_jsonl(a.eval_data) if a.eval_data else None
    model, tokenizer = load_16bit(a)
    if a.adapter == 'lora':
        model = attach_lora(model, a)
    else:
        model = attach_gcmp(model, tokenizer, a)

    half = DTYPES[a.dtype] is torch.float16
    config = SFTConfig(
        output_dir=a.output,
        dataset_text_field='text',
        max_length=a.max_length,
        packing=a.packing,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        per_device_eval_batch_size=a.batch_size,
        max_steps=a.max_steps,
        learning_rate=a.learning_rate,
        lr_scheduler_type='linear',
        warmup_steps=a.warmup_steps,
        weight_decay=a.weight_decay,
        optim=a.optim,
        bf16=DTYPES[a.dtype] is torch.bfloat16,
        fp16=half,
        logging_steps=1,
        save_strategy='steps',
        save_steps=a.save_steps,
        save_total_limit=a.save_total_limit,
        eval_strategy='steps' if eval_ds is not None else 'no',
        eval_steps=a.eval_steps if eval_ds is not None else None,
        report_to='none',
        seed=a.seed,
        data_seed=a.seed,
        remove_unused_columns=False,
        # LoRA: Unsloth enables smart checkpointing in get_peft_model.
        # GCMP: same patch (attach_gcmp), enabled through the Trainer.
        gradient_checkpointing=(a.adapter == 'gcmp'),
        gradient_checkpointing_kwargs={'use_reentrant': False} if a.adapter == 'gcmp' else None,
    )
    fields = ({'rank': a.lora_rank, 'alpha': a.lora_alpha, 'dropout': a.lora_dropout, 'rslora': a.use_rslora}
              if a.adapter == 'lora' else {'gcmp_scale': a.gcmp_scale})
    with open(os.path.join(a.output, 'comparison_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': a.model, 'quantization': f'none ({a.dtype})', 'adapter': a.adapter,
                   'targets': a.targets, **fields, 'train_data': a.train_data, 'eval_data': a.eval_data,
                   'max_length': a.max_length, 'seed': a.seed, 'max_steps': a.max_steps,
                   'batch_size': a.batch_size, 'grad_accum': a.grad_accum, 'packing': a.packing,
                   'learning_rate': a.learning_rate, 'optimizer': a.optim, 'scheduler': 'linear',
                   'warmup_steps': a.warmup_steps, 'weight_decay': a.weight_decay,
                   'eval_steps': a.eval_steps, 'save_steps': a.save_steps,
                   'benchmark_warmup_steps': a.benchmark_warmup_steps}, f, indent=2)

    logger = ComparisonLogger(a.output, a.benchmark_warmup_steps)
    trainer_class = CountedSFTTrainer
    if a.adapter == 'gcmp':
        import gcmp
        trainer_class = gcmp.adapter_trainer_class(CountedSFTTrainer, base_model=a.model)
    trainer = trainer_class(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                            processing_class=tokenizer, callbacks=[logger])
    trainer.comparison_counter = logger.counter
    trainer.train()
    final = os.path.join(a.output, 'final_adapter')
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f'Adapter ({a.adapter}) saved to {final}')


if __name__ == '__main__':
    main()
