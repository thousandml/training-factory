#!/usr/bin/env python3
"""Generation-based ROUGE/BLEU evaluation for CQ GCMP / LoRA or Unsloth NF4 LoRA adapters.

Input JSONL: {"instruction":"...", "input":"...", "output":"..."}
Or {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}.
Requires: pip install rouge-score sacrebleu peft transformers; Unsloth environment for --backend unsloth
"""
import argparse
import json
import random
from pathlib import Path
import sys
import hashlib
import os


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
            load_in_4bit=not args.unsloth_16bit,
            load_in_8bit=False,
            load_in_16bit=args.unsloth_16bit,
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
        if args.unsloth_16bit:
            if nf4_count:
                raise RuntimeError(f'--unsloth-16bit given but the model has {nf4_count} NF4 modules')
            print('Loaded full-precision (16-bit) Unsloth base', flush=True)
        elif nf4_count == 0:
            raise RuntimeError('No bitsandbytes NF4 Linear4bit modules found (use --unsloth-16bit for a 16-bit base)')
        else:
            print(f'Confirmed {nf4_count} NF4 modules', flush=True)

    from transformers import AutoTokenizer
    if args.adapter:
        adapter_path = Path(args.adapter)
        if not (adapter_path / 'adapter_config.json').exists():
            raise FileNotFoundError(f'adapter_config.json missing: {adapter_path}')
        adapter_type = json.loads((adapter_path / 'adapter_config.json').read_text()).get('adapter_type')
        if adapter_type == 'cq_gcmp':
            import gcmp
            cfg = json.loads((adapter_path / 'adapter_config.json').read_text())
            if args.backend == 'unsloth':
                from _cq_training_license import require_valid
                require_valid()
                import importlib
                try:
                    importlib.import_module(args.kernel_module)
                except ImportError as exc:
                    print(f'CQ-GCMP: {args.kernel_module} not importable ({exc}); using the torch implementation')
            targets = sorted({name.split('.')[-1] for name in cfg['modules']})
            gcmp.inject_gcmp(model, targets, scale=cfg['scale'], allow_dense=(args.backend == 'unsloth'),
                             include=lambda name: name in cfg['modules'])
            gcmp.load_adapter(model, adapter_path)
            model.requires_grad_(False)
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return model, tokenizer



def choose_generation_cache(model, tokenizer, device):
    """use_cache setting under which generation really applies a GCMP adapter.

    Unsloth's cached decode can call base weights directly, skipping wrapped
    modules. Uncached generation recomputes every step through the normal
    forward, so it is the reference: with the adapter perturbed, cached and
    uncached step-2 logits must agree to within their adapter-free numerical
    gap. A bypassed cached decode differs by a sizeable part of the adapter's
    effect. Falls back to use_cache=False when the cached path bypasses.
    """
    import torch
    import gcmp
    probe = tokenizer('Answer in one word: the color of the sky is', return_tensors='pt').to(device)

    def generate(use_cache):
        with torch.inference_mode():
            out = model.generate(**probe, max_new_tokens=2, min_new_tokens=2, do_sample=False,
                                 use_cache=use_cache, output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tokenizer.pad_token_id)
        return int(out.sequences[0, -2]), out.scores[0].float(), out.scores[1].float()

    def diff(x, y):  # min_new_tokens masks EOS to -inf; compare finite logits only
        ok = torch.isfinite(x) & torch.isfinite(y)
        return (x - y)[ok].abs().max().item() if ok.any() else 0.0

    with gcmp.coefficients_zeroed(model):
        z_cache, z_plain, z_plain2 = generate(True), generate(False), generate(False)
    noise = max(diff(z_plain[1], z_plain2[1]), diff(z_plain[2], z_plain2[2]))
    base_gap = diff(z_cache[2], z_plain[2])  # cached vs uncached, adapter-free
    for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        with gcmp.coefficients_perturbed(model, eps):
            p_cache, p_plain = generate(True), generate(False)
        if p_plain[0] == z_plain[0]:
            break
    if diff(p_plain[1], z_plain[1]) <= 10 * noise:
        raise RuntimeError('GCMP adapter is not applied during generation on this backend '
                           '(even uncached decoding bypasses it)')
    same_token = p_plain[0] == z_plain[0]
    effect = diff(p_plain[2], z_plain[2]) if same_token else 0.0
    gap = diff(p_cache[2], p_plain[2])
    tolerance = max(10 * base_gap, 10 * noise, 0.05 * effect)
    if p_cache[0] == p_plain[0] and gap <= tolerance:
        print(f'CQ-GCMP: adapter verified active in cached generation (gap {gap:.3g} <= {tolerance:.3g})', flush=True)
        return True
    print(f'CQ-GCMP: WARNING cached decode bypasses the adapter on this backend (gap {gap:.3g} > '
          f'{tolerance:.3g}); generating with use_cache=False (correct but slower)', flush=True)
    return False


def evaluation_signature(args, examples):
    """Hash settings and exact selected examples to reject incompatible resumes."""
    payload = {
        'backend': args.backend, 'model': args.model, 'adapter': args.adapter,
        'tokenizer': args.tokenizer, 'kernel_module': args.kernel_module,
        'kernel_backend': args.kernel_backend, 'seed': args.seed,
        'max_input_tokens': args.max_input_tokens,
        'max_new_tokens': args.max_new_tokens, 'decoding': 'greedy',
        'examples': examples,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_completed(path, examples, signature):
    """Read durable JSONL checkpoints, tolerating a truncated final write."""
    completed = {}
    if not path.exists():
        return completed
    valid_end = 0
    with path.open('rb') as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b'\n'):
                print('Discarding incomplete final JSONL line', flush=True)
                break
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f'Corrupt checkpoint line ending at byte {f.tell()}: {exc}') from exc
            i = record.get('index')
            if not isinstance(i, int) or not 1 <= i <= len(examples):
                raise ValueError(f'Invalid checkpoint example index: {i!r}')
            if i in completed:
                raise ValueError(f'Duplicate checkpoint index: {i}')
            if record.get('evaluation_signature') != signature:
                raise ValueError('Checkpoint does not match model, settings, or selected dataset. '
                                 'Use a different --predictions path or remove the old checkpoint.')
            messages, reference = examples[i-1]
            if record.get('messages') != messages or record.get('reference') != reference:
                raise ValueError(f'Checkpoint example {i} does not match input data')
            if not isinstance(record.get('prediction'), str):
                raise ValueError(f'Checkpoint example {i} lacks a prediction')
            completed[i] = record
            valid_end = f.tell()
    if valid_end != path.stat().st_size:
        with path.open('r+b') as f:
            f.truncate(valid_end)
    return completed


def write_metrics(path, args, records):
    from rouge_score import rouge_scorer
    import sacrebleu
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    sums = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    predictions, references = [], []
    for record in records:
        reference, prediction = record['reference'], record['prediction']
        scores = scorer.score(reference, prediction)
        for key in sums:
            sums[key] += scores[key].fmeasure
        predictions.append(prediction)
        references.append(reference)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    result = {
        'backend': args.backend, 'model': args.model, 'adapter': args.adapter,
        'dataset': args.data, 'examples': len(records), 'seed': args.seed,
        'max_input_tokens': args.max_input_tokens,
        'max_new_tokens': args.max_new_tokens, 'decoding': 'greedy',
        'rouge1_f1': sums['rouge1'] / len(records),
        'rouge2_f1': sums['rouge2'] / len(records),
        'rougeL_f1': sums['rougeL'] / len(records),
        'bleu': bleu.score, 'bleu_scale': '0-100 (SacreBLEU corpus BLEU)',
        'notes': 'ROUGE scores are mean per-example F1, on a 0-1 scale.',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cq', 'unsloth'], required=True)
    p.add_argument('--model', required=True, help='Local CQ checkpoint or Unsloth prequantized NF4 model ID')
    p.add_argument('--adapter', help='Adapter output directory: CQ GCMP or PEFT LoRA (omit to score base model)')
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
    p.add_argument('--unsloth-16bit', action='store_true', help='Unsloth backend: load a full-precision (bf16/fp16) base instead of NF4')
    p.add_argument('--resume', action='store_true', help='Resume from per-example predictions JSONL')
    args = p.parse_args()
    if args.max_input_tokens < 1 or args.max_new_tokens < 1 or args.limit < 0:
        p.error('max-input-tokens and max-new-tokens must be positive; limit must be nonnegative')

    examples = read_examples(args.data, args.limit, args.seed)
    signature = evaluation_signature(args, examples)
    output = Path(args.output)
    pred_path = Path(args.predictions) if args.predictions else output.with_name(output.stem + '_predictions.jsonl')
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    if pred_path.resolve() == output.resolve():
        p.error('--output and --predictions must be different paths')
    if pred_path.exists() and not args.resume:
        p.error(f'{pred_path} already exists; use --resume or choose a new --predictions path')
    completed = load_completed(pred_path, examples, signature) if args.resume else {}
    print(f'Loaded {len(completed)}/{len(examples)} completed examples', flush=True)
    if len(completed) < len(examples):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this evaluator')
        torch.manual_seed(args.seed)
        model, tokenizer = load_model(args)
        device = next((p.device for p in model.parameters() if p.device.type not in ('meta', 'cpu')), None)
        if device is None:
            raise RuntimeError('Could not find model parameters on CUDA')
        use_cache = True
        if any(type(m).__name__ == 'GCMPLinear' for m in model.modules()):
            use_cache = choose_generation_cache(model, tokenizer, device)
        with pred_path.open('a', encoding='utf-8') as checkpoint:
            for i, (messages, reference) in enumerate(examples, 1):
                if i in completed:
                    continue
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
                        use_cache=use_cache,
                    )
                prediction = tokenizer.decode(
                    generated[0, inputs['input_ids'].shape[-1]:],
                    skip_special_tokens=True,
                ).strip()
                record = {
                    'index': i, 'messages': messages, 'reference': reference,
                    'prediction': prediction, 'evaluation_signature': signature,
                }
                checkpoint.write(json.dumps(record, ensure_ascii=False) + '\n')
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
                completed[i] = record
                if len(completed) % 10 == 0 or len(completed) == len(examples):
                    print(f'Evaluated {len(completed)}/{len(examples)}', flush=True)
    records = [completed[i] for i in range(1, len(examples) + 1)]
    result = write_metrics(output, args, records)
    print(json.dumps(result, indent=2))
    print(f'Predictions: {pred_path}')


if __name__ == '__main__':
    main()
