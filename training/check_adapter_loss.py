#!/usr/bin/env python3
"""Measure validation loss through a running vLLM server.

Computes the mean next-token loss over each validation row, the same way
training's eval loss does (whole chat-templated sequence, truncated to
--max-length, EOS appended). Run it once against the server WITH the adapter
and once WITHOUT; the adapted number should be clearly lower and close to
the eval loss reported at the end of training.

Usage:
  python check_adapter_loss.py \
    --model /workspace/models/Qwen3.8-27B-CQ-4bit \
    --data training/examples/alpaca_val.jsonl --limit 100
"""
import argparse
import json
import math
import urllib.request

from transformers import AutoTokenizer


def token_ids(tok, row, max_length):
    if row.get("messages"):
        ids = tok.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
    elif isinstance(row.get("text"), str):
        ids = tok(row["text"])["input_ids"]
    else:
        raise ValueError("row needs 'messages' or 'text'")
    ids = list(ids)[:max_length]
    if len(ids) < max_length and tok.eos_token_id is not None and ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    return ids


def prompt_logprobs(url, model, ids):
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1,
                       "temperature": 0, "prompt_logprobs": 0}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["prompt_logprobs"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--url", default="http://localhost:8000/v1/completions")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=100)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    total_nll, total_tok, rows = 0.0, 0, 0
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if rows >= a.limit:
                break
            if not line.strip():
                continue
            ids = token_ids(tok, json.loads(line), a.max_length)
            lps = prompt_logprobs(a.url, a.model, ids)
            for tid, entry in zip(ids[1:], lps[1:]):
                total_nll -= entry[str(tid)]["logprob"]
                total_tok += 1
            rows += 1
            if rows % 10 == 0:
                print(f"{rows} rows  running loss {total_nll / total_tok:.4f}", flush=True)
    loss = total_nll / total_tok
    print(f"\nrows={rows} tokens={total_tok} mean loss={loss:.4f} perplexity={math.exp(loss):.3f}")


if __name__ == "__main__":
    main()
