#!/usr/bin/env python3
"""Validation loss through the TRAINING-side CQ model (no vLLM).

Uses the same rows, tokenization and loss as gcmp_diagnose.py, so the numbers
can be compared directly with its vLLM results (3.159 base / 3.164 adapted).
Run from the training folder (the one containing train.py), with vLLM stopped:

  python train_side_loss.py \
    --model /workspace/models/Qwen3.8-27B-CQ-4bit \
    --adapter /workspace/training-factory-gcmp/outputs/qwen38-27b-cq-4bit-alpaca-gcmp-v1 \
    --data training/examples/alpaca_val.jsonl --kernel-module cq.cq_kernels
"""
import argparse
import json

import torch
from transformers import AutoTokenizer


def token_ids(tok, row, max_length):
    if row.get("messages"):
        ids = tok.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
    else:
        ids = tok(row["text"])["input_ids"]
    ids = list(ids)[:max_length]
    if len(ids) < max_length and tok.eos_token_id is not None and ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    return ids


@torch.no_grad()
def mean_loss(model, prompts, device):
    nll, n = 0.0, 0
    for ids in prompts:
        x = torch.tensor([ids], device=device)
        logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits[0, :-1], -1)
        tgt = x[0, 1:]
        nll -= float(lp.gather(1, tgt[:, None]).sum())
        n += tgt.numel()
    return nll / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--data", required=True)
    p.add_argument("--kernel-module", default="cq.cq_kernels")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    a = p.parse_args()

    from kernel_backend import configure_backend
    from load_model import load_model
    configure_backend(a.kernel_module, "auto")
    model = load_model(a.model)
    model.eval()
    device = model.get_input_embeddings().weight.device

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if line.strip() and len(prompts) < a.limit:
                prompts.append(token_ids(tok, json.loads(line), a.max_length))

    base = mean_loss(model, prompts, device)
    print(f"training-side loss, base model     = {base:.4f}")
    if a.adapter:
        import gcmp
        cfg = json.load(open(f"{a.adapter}/adapter_config.json"))
        targets = sorted({name.split(".")[-1] for name in cfg["modules"]})
        gcmp.inject_gcmp(model, targets, scale=float(cfg.get("scale", 1.0)))
        gcmp.load_adapter(model, a.adapter)
        model.eval()
        adapted = mean_loss(model, prompts, device)
        print(f"training-side loss, with adapter   = {adapted:.4f}")
        print(f"difference                         = {base - adapted:+.4f}")


if __name__ == "__main__":
    main()
