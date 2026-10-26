#!/usr/bin/env python3
"""Generation-based ROUGE/BLEU evaluation for CQ or Unsloth NF4 LoRA adapters.

Input JSONL: {"instruction":"...", "input":"...", "output":"..."}
Or {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}.
Requires: pip install rouge-score sacrebleu peft transformers; Unsloth environment for --backend unsloth
"""
import argparse
import json
import random
from pathlib import Path
import sys


def read_examples(path, limit, seed):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if 'instruction' in row and 'output' in row:
                instruction = str(row['instruction']).strip()
                extra = str(row.get('input') or '').strip()
                user_text = instruction + ('\n\n' + extra if extra else '')
                messages = [{'role': 'user', 'content': user_text}]
                reference = str(row['output']).strip()
            elif isinstance(row.get('messages'), list):
                messages_all = row['messages']
                if not messages_all or messages_all[-1].get('role') != 'assistant':
                    raise ValueError(f'Line {line_number}: messages must end with assistant reference')
                reference = messages_all[-1]['content']
                messages = messages_all[:-1]
                if not messages or not isinstance(reference, str):
                    raise ValueError(f'Line {line_number}: missing prompt or text reference')
                if any(not isinstance(m.get('content'), str) for m in messages):
                    raise ValueError('Multimodal messages are not supported by this text evaluator')
                reference = reference.strip()
            else:
                raise ValueError(
                    f'Line {line_number}: expected instruction/output or messages ending with assistant. '
                    'Plain {"text": ...} rows have no separate reference response; '
                    'prepare a held-out prompt/reference JSONL first.'
                )
            if reference:
                rows.append((messages, reference))
    if not rows:
        raise ValueError('No usable prompt/reference pairs')
    if limit > 0 and len(rows) > limit:
        rng = random.Random(seed)
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), limit))]
    return rows


def build_prompt(tokenizer, messages):
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # For base checkpoints without a chat template, keep the prompt deterministic.
    prefix = '\n'.join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
    return prefix + '\nAssistant:'


def load_model(args):
    import torch
    if args.backend == 'cq':
        # Match training/train.py's CQ loader and kernel setup.
        from _cq_training_license import require_valid
        require_valid()
        from kernel_backend import configure_backend
        from load_model import load_model as load_cq_model
        configure_backend(args.kernel_module, args.kernel_backend)
        model = load_cq_model(args.model, text_only=True,
                              trust_remote_code=args.trust_remote_code)
    else:
        # Unsloth must be imported before transformers/peft.
        from unsloth import FastModel
        model, _ = FastModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_input_tokens + args.max_new_tokens,
            load_in_4bit=True,
            load_in_8bit=False,
            load_in_16bit=False,
        )
        nf4_count = 0
        for module in model.modules():
            if module.__class__.__name__ == 'Linear4bit':
                qtype = getattr(module, 'quant_type', None)
                if qtype is None:
                    qtype = getattr(getattr(module, 'weight', None), 'quant_type', None)
                if qtype != 'nf4':
                    raise RuntimeError(f'Expected NF4, got {qtype}')
                nf4_count += 1
        if nf4_count == 0:
            raise RuntimeError('No bitsandbytes NF4 Linear4bit modules found')
        print(f'Confirmed {nf4_count} NF4 modules', flush=True)

    from peft import PeftModel
    from transformers import AutoTokenizer
    if args.adapter:
        adapter_path = Path(args.adapter)
        if not (adapter_path / 'adapter_config.json').exists():
            raise FileNotFoundError(f'PEFT adapter_config.json missing: {adapter_path}')
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cq', 'unsloth'], required=True)
    p.add_argument('--model', required=True, help='Local CQ checkpoint or Unsloth prequantized NF4 model ID')
    p.add_argument('--adapter', help='LoRA adapter output directory (omit to score base model)')
    p.add_argument('--tokenizer', help='Optional tokenizer path; defaults to --model')
    p.add_argument('--data', required=True, help='Held-out Alpaca instruction/output JSONL')
    p.add_argument('--output', required=True, help='Metrics JSON output path')
    p.add_argument('--predictions', help='Optional predictions JSONL; defaults beside --output')
    p.add_argument('--limit', type=int, default=200, help='Deterministic subset; 0 = all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-input-tokens', type=int, default=1024)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--kernel-module', default='cq.cq_kernels')
    p.add_argument('--kernel-backend', choices=['auto', 'raw', 'registered'], default='auto')
    p.add_argument('--trust-remote-code', action='store_true')
    args = p.parse_args()
    if args.max_input_tokens < 1 or args.max_new_tokens < 1 or args.limit < 0:
        p.error('max-input-tokens and max-new-tokens must be positive; limit must be nonnegative')

    import torch
    from rouge_score import rouge_scorer
    import sacrebleu

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this evaluator')
    torch.manual_seed(args.seed)
    examples = read_examples(args.data, args.limit, args.seed)
    model, tokenizer = load_model(args)
    device = next((p.device for p in model.parameters() if p.device.type not in ('meta', 'cpu')), None)
    if device is None:
        raise RuntimeError('Could not find model parameters on CUDA')
    if device.type == 'meta':
        raise RuntimeError('Model parameters are on meta device; model was not fully loaded')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    sums = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    predictions, references, records = [], [], []
    for i, (messages, reference) in enumerate(examples, 1):
        prompt = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                           max_length=args.max_input_tokens).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        # Decode ONLY generated tokens; prompt echo must not inflate metrics.
        prediction = tokenizer.decode(generated[0, inputs['input_ids'].shape[-1]:],
                                      skip_special_tokens=True).strip()
        scores = scorer.score(reference, prediction)
        for key in sums:
            sums[key] += scores[key].fmeasure
        predictions.append(prediction)
        references.append(reference)
        records.append({'index': i, 'messages': messages, 'reference': reference,
                        'prediction': prediction,
                        'rouge1': scores['rouge1'].fmeasure,
                        'rouge2': scores['rouge2'].fmeasure,
                        'rougeL': scores['rougeL'].fmeasure})
        if i % 10 == 0 or i == len(examples):
            print(f'Evaluated {i}/{len(examples)}', flush=True)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    result = {
        'backend': args.backend, 'model': args.model, 'adapter': args.adapter,
        'dataset': args.data, 'examples': len(examples), 'seed': args.seed,
        'max_input_tokens': args.max_input_tokens,
        'max_new_tokens': args.max_new_tokens,
        'decoding': 'greedy',
        'rouge1_f1': sums['rouge1'] / len(examples),
        'rouge2_f1': sums['rouge2'] / len(examples),
        'rougeL_f1': sums['rougeL'] / len(examples),
        'bleu': bleu.score,
        'bleu_scale': '0-100 (SacreBLEU corpus BLEU)',
        'notes': 'ROUGE scores are mean per-example F1, on a 0-1 scale.',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    pred_path = Path(args.predictions) if args.predictions else output.with_name(output.stem + '_predictions.jsonl')
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2))
    print(f'Predictions: {pred_path}')


if __name__ == '__main__':
    main()
