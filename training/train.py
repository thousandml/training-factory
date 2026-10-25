import argparse
import json
import os
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoProcessor, Trainer, TrainingArguments
from load_model import load_model
from gradient_safety import prepare_trainable_lora, install_nonfinite_gradient_guards
from unsloth_compat import configure_checkpointing, optimizer_name


def main():
    from _cq_training_license import require_valid
    require_valid()  # fail closed before loading checkpoint or allocating CUDA memory
    p = argparse.ArgumentParser()
    p.add_argument('--kernel-module', default='cq.cq_kernels')
    p.add_argument('--kernel-backend', choices=['auto','raw','registered'], default='auto')
    p.add_argument('--model', required=True, help='Local complete CQ checkpoint')
    p.add_argument('--data', required=True, help='JSONL: text or messages; multimodal messages may contain image paths')
    p.add_argument('--output', default='outputs/cq-lora')
    p.add_argument('--layer-map')
    p.add_argument('--text-only', action='store_true', help='Exclude absent vision/audio/MTP CQ records from text-only CausalLM')
    p.add_argument('--targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=16)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--benchmark', action='store_true', help='Collect CUDA-synchronized optimizer-step timing and peak memory')
    p.add_argument('--benchmark-warmup-steps', type=int, default=10)
    p.add_argument('--comparison-logs', action='store_true', help='Log actual consumed tokens, throughput and loss history')
    p.add_argument('--eval-data', help='Held-out JSONL data; same schema as training data')
    p.add_argument('--eval-steps', type=int, default=25)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpointing', choices=['native', 'unsloth'], default='unsloth',
                   help='Unsloth smart checkpointing by default; use native to disable')
    p.add_argument('--optimizer', choices=['adamw_torch', 'adamw_torch_fused'], default='adamw_torch_fused',
                   help='PyTorch AdamW; fused uses CUDA fused optimizer without bitsandbytes')
    p.add_argument('--resume')
    p.add_argument('--trust-remote-code', action='store_true')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('This initial bridge supports one GPU only')
    if a.benchmark_warmup_steps < 0:
        raise ValueError('benchmark-warmup-steps must be nonnegative')
    if not 0 <= a.lora_dropout < 1 or a.eval_steps < 1:
        raise ValueError('lora-dropout must be in [0,1) and eval-steps >= 1')
    if a.max_length < 2 or a.rank < 1:
        raise ValueError('max-length >= 2 and rank >= 1 required')
    from inspect_checkpoint import inspect
    inspect(a.model, a.layer_map, text_only=a.text_only, trust_remote_code=a.trust_remote_code)  # fail before CUDA weight allocation on layout/header errors
    from kernel_backend import configure_backend
    configure_backend(a.kernel_module, a.kernel_backend)
    model = load_model(a.model, a.layer_map, a.trust_remote_code, text_only=a.text_only)
    targets = a.targets.split(',')
    matched = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split('.')[-1] in targets]
    if not matched:
        raise ValueError('No target linears matched')
    config = LoraConfig(task_type='CAUSAL_LM', r=a.rank, lora_alpha=a.alpha,
                        lora_dropout=a.lora_dropout, target_modules=matched, bias='none',
                        base_model_name_or_path=str(Path(a.model).resolve()))
    # Configure before PEFT wrapping so the model's checkpoint hooks are patched.
    configure_checkpointing(a.checkpointing)
    selected_optimizer = optimizer_name(a.optimizer)
    load_report = model._cq_load_report
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    prepare_trainable_lora(model)
    _gradient_guards = install_nonfinite_gradient_guards(model)
    processor = None
    if not a.text_only:
        try:
            processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
        except (ValueError, OSError, KeyError) as exc:
            raise ValueError('Full-model mode requires a compatible multimodal AutoProcessor') from exc
    tokenizer = getattr(processor, 'tokenizer', None) if processor is not None else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer needs a pad or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    dataset = load_dataset('json', data_files=a.data, split='train')
    if processor is not None:
        # Keep source rows intact: multimodal processor must see structured content.
        from PIL import Image
        data_root = Path(a.data).resolve().parent
        def collate(rows):
            prompts, images = [], []
            # Multimodal checkpoints can also receive ordinary text-only batches.
            # Never send an empty images=[] list to a processor that expects images.
            for row in rows:
                messages = row.get('messages')
                if not messages:
                    if isinstance(row.get('text'), str):
                        messages = [{'role': 'user', 'content': row['text']}]
                    else:
                        raise ValueError('Each training row needs text or messages')
                row_images = []
                for msg in messages:
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'image':
                                path = part.get('path') or part.get('image')
                                if not isinstance(path, str) or path.startswith(('http:', 'https:')):
                                    raise ValueError('Image content requires a local file path')
                                image_path = (data_root / path).resolve()
                                if not image_path.is_relative_to(data_root):
                                    raise ValueError('Image path escapes dataset directory')
                                with Image.open(image_path) as im:
                                    row_images.append(im.convert('RGB'))
                            elif isinstance(part, dict) and part.get('type') not in ('text', 'image'):
                                raise ValueError(f"Unsupported modality {part.get('type')}; audio/video not implemented")
                if len(row_images) > 1:
                    raise ValueError('This initial bridge supports at most one image per training row')
                prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
                images.extend(row_images)
            if images and len(images) != len(rows):
                raise ValueError('Mixed text-only and image-text rows in the same batch are not yet supported; use batch-size 1 or group rows by modality')
            if images:
                batch = processor(text=prompts, images=images, padding=True, truncation=True,
                                  max_length=a.max_length, return_tensors='pt')
            else:
                batch = tokenizer(prompts, padding=True, truncation=True,
                                  max_length=a.max_length, return_tensors='pt')
            if 'input_ids' not in batch or 'attention_mask' not in batch:
                raise ValueError('Processor did not return token IDs and attention masks')
            labels = batch['input_ids'].clone()
            labels[batch['attention_mask'] == 0] = -100
            for token in getattr(processor, 'image_token_id', None), getattr(model.config, 'image_token_id', None):
                if isinstance(token, int):
                    labels[batch['input_ids'] == token] = -100
            batch['labels'] = labels
            return batch
        if not len(dataset):
            raise ValueError('No training examples')
    else:
        def tokenize(row):
            if row.get('messages'):
                ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False)
                ids = ids[:a.max_length]
            elif isinstance(row.get('text'), str):
                ids = tokenizer(row['text'], truncation=True, max_length=a.max_length)['input_ids']
            else:
                raise ValueError('Each row needs text or messages')
            if len(ids) < a.max_length and tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
                ids.append(tokenizer.eos_token_id)
            return {'input_ids': ids, 'attention_mask': [1] * len(ids)}
        dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda row: len(row['input_ids']) >= 2)
        if not len(dataset):
            raise ValueError('No usable training examples')
        def collate(rows):
            batch = tokenizer.pad(rows, return_tensors='pt', pad_to_multiple_of=8)
            labels = batch['input_ids'].clone()
            labels[batch['attention_mask'] == 0] = -100
            batch['labels'] = labels
            return batch
    eval_dataset = None
    if a.eval_data:
        eval_dataset = load_dataset('json', data_files=a.eval_data, split='train')
        if processor is None:
            eval_dataset = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)
            eval_dataset = eval_dataset.filter(lambda row: len(row['input_ids']) >= 2)
        if not len(eval_dataset):
            raise ValueError('No usable evaluation examples')
    if a.benchmark:
        from benchmark import TokenCountingCollator
        collate = TokenCountingCollator(collate)
    args = TrainingArguments(output_dir=a.output, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
        max_steps=a.max_steps, learning_rate=a.learning_rate, fp16=False, bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        optim=selected_optimizer, logging_steps=1, save_steps=100, save_total_limit=2,
        report_to='none', remove_unused_columns=False, seed=a.seed,
        eval_strategy='steps' if a.eval_data else 'no', eval_steps=a.eval_steps if a.eval_data else None,
        max_grad_norm=1.0)
    if a.comparison_logs:
        from comparison_metrics import CountedTrainer, ComparisonMetricsCallback
        trainer_class = CountedTrainer
    else:
        trainer_class = Trainer
    trainer = trainer_class(model=model, args=args, train_dataset=dataset, eval_dataset=eval_dataset,
                            data_collator=collate, processing_class=processor or tokenizer)
    if a.comparison_logs:
        comparison = ComparisonMetricsCallback(a.output, {
            'model': a.model, 'train_data': a.data, 'eval_data': a.eval_data,
            'quantization': 'CQ 4bit', 'targets': targets, 'rank': a.rank,
            'alpha': a.alpha, 'lora_dropout': a.lora_dropout,
            'batch_size': a.batch_size, 'grad_accum': a.grad_accum,
            'max_length': a.max_length, 'max_steps': a.max_steps,
            'learning_rate': a.learning_rate, 'seed': a.seed,
            'gradient_checkpointing': a.checkpointing, 'optimizer': selected_optimizer,
            'train_examples': len(dataset),
            'eval_examples': len(eval_dataset) if eval_dataset is not None else None,
        }, warmup_steps=a.benchmark_warmup_steps)
        comparison.trainer_ref = trainer
        trainer.add_callback(comparison)
    if a.benchmark:
        from benchmark import BenchmarkCallback
        trainer.add_callback(BenchmarkCallback(
            output_file=str(Path(a.output) / 'benchmark.json'),
            warmup_steps=a.benchmark_warmup_steps,
            token_counter=collate,
        ))
    result = trainer.train(resume_from_checkpoint=a.resume)
    model.save_pretrained(a.output, safe_serialization=True)
    tokenizer.save_pretrained(a.output)
    trainer.save_metrics('train', result.metrics)
    Path(a.output, 'cq_load_report.json').write_text(json.dumps(load_report, indent=2))
    Path(a.output, 'cq_training.json').write_text(json.dumps(vars(a), indent=2))

if __name__ == '__main__':
    main()
