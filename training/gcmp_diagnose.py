#!/usr/bin/env python3
"""Diagnose whether a GCMP adapter is actually applied inside vLLM.

Stop the running vLLM server first (this loads its own copy of the model).

  python gcmp_diagnose.py \
    --model /workspace/models/Qwen3.8-27B-CQ-4bit \
    --adapter /workspace/training-factory-gcmp/outputs/qwen38-27b-cq-4bit-alpaca-gcmp-v1 \
    --data /workspace/training-factory-gcmp/training/examples/alpaca_val.jsonl

Add --eager to repeat the test without torch.compile / CUDA graphs.

Step 1 checks the saved coefficients (are they non-zero?).
Step 2 counts how many layers inside the vLLM worker have adapter data attached.
Step 3 measures validation loss with the adapter, unplugs it in place,
       measures again, and reports the difference.
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # needed for collective_rpc(callable)


def adapter_stats(path):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    t = load_file(os.path.join(path, "adapter_model.safetensors"))
    vals = [v.float() for v in t.values()]
    numel = sum(v.numel() for v in vals)
    nz = sum(int((v != 0).sum()) for v in vals)
    mx = max(float(v.abs().max()) for v in vals)
    mean = sum(float(v.abs().sum()) for v in vals) / numel
    print("== Step 1: saved adapter ==")
    print(f"projections={len(cfg['modules'])} tensors={len(t)} params={numel:,} scale={cfg.get('scale')}")
    print(f"non-zero={nz / numel:.1%}  max|c|={mx:.3e}  mean|c|={mean:.3e}")
    if nz == 0:
        print("!! All coefficients are zero: the saved adapter is untrained. Problem is on the training/save side.")
    return cfg


def worker_report(worker):
    model = worker.get_model() if hasattr(worker, "get_model") else worker.model_runner.model
    layers, coef_abs, names = 0, 0.0, []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if getattr(method, "_cq_gcmp_plan", None):
            layers += 1
            coef_abs += float(module.cq_gcmp_coef.abs().sum())
            if len(names) < 3:
                names.append(name)
    return {"layers_with_adapter": layers, "sum_abs_coef": coef_abs, "examples": names}


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


def mean_loss(llm, prompts):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
    nll, n = 0.0, 0
    for ids, out in zip(prompts, outs):
        for tid, lp in zip(ids[1:], out.prompt_logprobs[1:]):
            nll -= lp[tid].logprob
            n += 1
    return nll / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eager", action="store_true", help="disable torch.compile / CUDA graphs")
    a = p.parse_args()

    adapter_stats(a.adapter)

    os.environ["CQ_GCMP_ADAPTER"] = a.adapter
    os.environ.setdefault("CQ_MODEL_DIR", a.model)
    from transformers import AutoTokenizer
    from vllm import LLM
    from cq import swap_adapter_llm

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if line.strip() and len(prompts) < a.limit:
                prompts.append(token_ids(tok, json.loads(line), a.max_length))

    llm = LLM(model=a.model, quantization="cq", tensor_parallel_size=1, max_model_len=2048,
              max_num_seqs=8, enforce_eager=a.eager, gpu_memory_utilization=0.85)

    print("\n== Step 2: inside the vLLM worker ==")
    print(llm.collective_rpc(worker_report))

    print("\n== Step 3: validation loss ==")
    with_adapter = mean_loss(llm, prompts)
    swapped = swap_adapter_llm(llm, None)
    print(f"unplugged adapter on layers: {swapped}")
    print(llm.collective_rpc(worker_report))
    without = mean_loss(llm, prompts)
    print(f"loss with adapter    = {with_adapter:.4f}")
    print(f"loss without adapter = {without:.4f}")
    print(f"difference           = {without - with_adapter:+.4f}")
    if abs(without - with_adapter) < 1e-3:
        print("!! No measurable effect: the adapter is not reaching the forward pass in this mode.")
    else:
        print("Adapter changes the model's predictions in this mode.")


if __name__ == "__main__":
    main()
