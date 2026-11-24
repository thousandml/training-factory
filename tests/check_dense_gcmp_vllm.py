#!/usr/bin/env python3
"""GPU check: a GCMP adapter trained on a full-precision model, served by vLLM.

Scores the reference answers of held-out rows with the base model and with the
adapter in one vLLM engine (multi-adapter mode, so both run in the same batch),
and prints answer loss for each. The adapter should cut the loss by about as
much as the Transformers evaluation showed (BF16 + GCMP on Qwen3.8-27B: base
1.678 -> 1.471 with uniform_loss_eval.py; this script's formula is close, not identical).

    python tests/check_dense_gcmp_vllm.py --model Qwen/Qwen3.8-27B \
        --adapter outputs/qwen38-27b-unsloth-bf16-alpaca-gcmp-v1/final_adapter \
        --data eval_downstream/alpaca_val_eval.jsonl --rows 200
"""
import argparse
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--data', required=True, help='JSONL rows with "messages" and "reference" (prepare_eval_data.py)')
    ap.add_argument('--rows', type=int, default=200)
    ap.add_argument('--max-model-len', type=int, default=2048)
    a = ap.parse_args()

    os.environ['CQ_GCMP_ADAPTERS'] = f'gcmp={a.adapter}'      # before vLLM starts its workers
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()][: a.rows]
    prompts, spans = [], []
    for r in rows:
        prompt = tok.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)['input_ids']
        full = tok(prompt + r['reference'] + (tok.eos_token or ''), add_special_tokens=False)['input_ids']
        if len(full) > a.max_model_len - 1 or len(full) <= len(p_ids):
            continue
        prompts.append({'prompt_token_ids': full})
        spans.append((len(p_ids), len(full)))

    llm = LLM(model=a.model, max_model_len=a.max_model_len, enforce_eager=False)
    results = {}
    for label, extra in (('base', {}), ('gcmp', {'gcmp_adapter': 'gcmp'})):
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, extra_args=extra or None)
        outs = llm.generate(prompts, sp)
        nll, n = 0.0, 0
        for out, (start, end), pr in zip(outs, spans, prompts):
            ids = pr['prompt_token_ids']
            for pos in range(start, end):
                lp = out.prompt_logprobs[pos]
                nll -= lp[ids[pos]].logprob
                n += 1
        results[label] = nll / n
        print(f'{label:5s} answer loss {results[label]:.4f}  ({len(prompts)} rows, {n} tokens)', flush=True)
    drop = results['base'] - results['gcmp']
    print(f'adapter lowers answer loss by {drop:.4f} '
          f'({"OK" if drop > 0.1 else "TOO SMALL: adapter may not be applied"}); '
          f'perplexity {math.exp(results["base"]):.2f} -> {math.exp(results["gcmp"]):.2f}')


if __name__ == '__main__':
    main()
