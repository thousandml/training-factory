#!/usr/bin/env python3
"""Validation loss measured the same way for every base/adapter combination.

The training scripts report eval loss through different trainers (the CQ
trainer and Unsloth's SFTTrainer), which do not tokenize, mask or average
identically, so their numbers are not on one scale. This script loads each
model through the same loaders as eval_rouge_bleu.py and computes, with one
tokenizer and one formula:

  text_loss      mean NLL over every token of the training-format rows
                 ({"text": ...}, truncated to --max-length, EOS appended):
                 the objective both trainers optimise
  response_loss  mean NLL over the assistant answer tokens only, using the
                 chat template with thinking disabled (prompt tokens masked)

Both are token-weighted means with 95% bootstrap intervals over rows.
Per-row values are written so cases can be compared pairwise.

  python uniform_loss_eval.py --backend cq --model /path/CQ --adapter /path/adapter \
      --text-data alpaca_val.jsonl --chat-data alpaca_val_eval.jsonl --output eval/cq_lora/loss.json
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def model_args(a):
    return argparse.Namespace(
        backend=a.backend, model=a.model, adapter=a.adapter, tokenizer=a.tokenizer,
        kernel_module=a.kernel_module, kernel_backend='auto', trust_remote_code=a.trust_remote_code,
        unsloth_16bit=a.unsloth_16bit, max_input_tokens=a.max_length, max_new_tokens=8,
        text_only=a.text_only)


def read_text_rows(path, limit):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get('text'), str):
                    rows.append(row['text'])
    return rows[:limit] if limit else rows


def bootstrap(nll, ntok, iters=1000, seed=0):
    rng = random.Random(seed)
    n = len(nll)
    stats = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(sum(nll[i] for i in idx) / max(sum(ntok[i] for i in idx), 1))
    stats.sort()
    return stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=['cq', 'unsloth'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter')
    p.add_argument('--tokenizer')
    p.add_argument('--text-data', help='training-format validation JSONL ({"text": ...})')
    p.add_argument('--chat-data', help='prompt/reference JSONL (from prepare_eval_data.py)')
    p.add_argument('--output', required=True)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--limit', type=int, default=0, help='rows per dataset; 0 = all')
    p.add_argument('--kernel-module', default='cq.cq_kernels')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--unsloth-16bit', action='store_true')
    layout = p.add_mutually_exclusive_group()
    layout.add_argument('--text-only', dest='text_only', action='store_true', default=None)
    layout.add_argument('--full-model', dest='text_only', action='store_false')
    a = p.parse_args()
    if not (a.text_data or a.chat_data):
        p.error('give --text-data and/or --chat-data')

    import torch
    import eval_rouge_bleu as ev
    model, tokenizer = ev.load_model(model_args(a))
    device = next(p_.device for p_ in model.parameters() if p_.device.type == 'cuda')

    @torch.inference_mode()
    def row_nll(ids, start):
        """Sum of -log p over targets ids[start+1:], and their count."""
        x = torch.tensor([ids], device=device)
        logits = model(input_ids=x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        tgt = x[0, 1:]
        tok = -lp.gather(1, tgt[:, None])[:, 0]
        tok = tok[max(start, 0):]
        return float(tok.sum()), int(tok.numel())

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_name(out.stem + '_rows.jsonl')
    result = {'backend': a.backend, 'model': a.model, 'adapter': a.adapter or None,
              'max_length': a.max_length, 'unsloth_16bit': a.unsloth_16bit}
    with open(rows_path, 'w', encoding='utf-8') as rows_out:
        if a.text_data:
            nll, ntok = [], []
            eos = tokenizer.eos_token_id
            for i, text in enumerate(read_text_rows(a.text_data, a.limit)):
                ids = tokenizer(text, add_special_tokens=False)['input_ids'][:a.max_length]
                if len(ids) < a.max_length and eos is not None and ids[-1] != eos:
                    ids.append(eos)
                s, n = row_nll(ids, 0)
                nll.append(s); ntok.append(n)
                rows_out.write(json.dumps({'set': 'text', 'row': i, 'nll': s, 'tokens': n}) + '\n')
                if (i + 1) % 200 == 0:
                    print(f'text rows {i + 1}: loss {sum(nll) / sum(ntok):.4f}', flush=True)
            lo, hi = bootstrap(nll, ntok)
            result.update(text_loss=sum(nll) / sum(ntok), text_loss_ci95=[lo, hi],
                          text_rows=len(nll), text_tokens=sum(ntok))
        if a.chat_data:
            examples = ev.read_examples(a.chat_data, a.limit, 42)
            nll, ntok = [], []
            for i, (messages, reference) in enumerate(examples):
                prompt = ev.build_prompt(tokenizer, messages, False)
                p_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
                r_ids = tokenizer(reference, add_special_tokens=False)['input_ids']
                eos = tokenizer.eos_token_id
                ids = (p_ids + r_ids + ([eos] if eos is not None else []))[:a.max_length]
                if len(ids) <= len(p_ids):
                    continue  # prompt alone fills the window
                s, n = row_nll(ids, len(p_ids) - 1)
                nll.append(s); ntok.append(n)
                rows_out.write(json.dumps({'set': 'response', 'row': i, 'nll': s, 'tokens': n}) + '\n')
                if (i + 1) % 200 == 0:
                    print(f'response rows {i + 1}: loss {sum(nll) / sum(ntok):.4f}', flush=True)
            lo, hi = bootstrap(nll, ntok)
            result.update(response_loss=sum(nll) / sum(ntok), response_loss_ci95=[lo, hi],
                          response_rows=len(nll), response_tokens=sum(ntok))
    for k in ('text_loss', 'response_loss'):
        if k in result:
            result[k.replace('loss', 'ppl')] = math.exp(result[k])
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
