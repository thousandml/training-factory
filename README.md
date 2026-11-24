# Training Factory

**GCMP is a drop-in replacement for LoRA. It matches LoRA's accuracy with 14–55× fewer trainable parameters, and one GPU can serve 256 GCMP adapters as fast as it serves one.**

[Website](https://www.thousandml.com) · [Models](https://huggingface.co/thousandml) · [Issues](https://github.com/thousandml/training-factory/issues)

Training Factory fine-tunes open LLMs on your own GPUs. It has three parts:

- **GCMP adapters.** 2.9M trainable parameters for a 27B model, where LoRA r=32 needs ~159M. Adapter files are ~5.8 MB for GCMP vs ~319 MB for LoRA r=32, both in bf16 (55× smaller). GCMP works on full-precision (BF16/FP16) models and on 4-bit bases.
- **CQ 4-bit quantization** (optional). It needs no calibration data. On Qwen3.8-27B, the CQ base matches the full-precision model on all 9 benchmarks we ran, in an 18 GB checkpoint.
- **vLLM serving.** One base model serves many GCMP adapters, on a CQ 4-bit base or a full-precision (BF16/FP16) one. Each request picks its adapter, and requests for different adapters share a batch.

## GCMP at a glance (Qwen3.8-27B)

| | GCMP | LoRA r=32 | LoRA r=8 |
|---|---:|---:|---:|
| Trainable parameters | **2.9M** | ~159M | ~40M |
| Adapter file (bf16)¹ | **~5.8 MB** | ~319 MB | ~80 MB |
| BF16 base: benchmark accuracy | tied | tied | tied |
| BF16 base: held-out answer loss ↓ | **1.471** | 1.480 | 1.555 |
| NF4 base: GSM8K | **83.3** | 79.3 | – |
| 256 adapters on one A40: output tok/s | **~84** (same as 1 adapter) | can't start (host RAM) | 59.5 default, 74.4 tuned |
| Swap one tenant for a new one, p50 | **0.43 s** | – | 5.4 s |
| p99 time to first token, 256 tenants, long-tail traffic | **1.2 s** | – | 91 s |

¹ The 14–55× is trainable parameters; file sizes follow it at the same dtype. Our trainers saved both GCMP and the Unsloth LoRA in fp32 (11.6 MB vs ~637 MB), and both are half that in bf16 (5.8 MB vs ~319 MB). Storing GCMP in bf16 costs nothing: held-out answer loss is identical to the fp32 file on CQ (1.5033 vs 1.5033) and NF4 (1.4582 vs 1.4582). New GCMP adapters are saved in bf16 by default (`--save-dtype`); convert older fp32 ones with `python training/to_bf16.py <adapter_dir> <output_dir>`.

Each row is explained, with significance tests, in [Results](#results). [Reproduce the benchmarks](#reproduce-the-benchmarks) has every command.

## Quick start

**Requirements:** Linux x86-64 or WSL2, an NVIDIA GPU, Conda and Python 3.12. The native kernels ship compiled and need a ThousandML license check. It's free during early access.

```bash
git clone https://github.com/thousandml/training-factory.git
cd training-factory
./install.sh

hf download thousandml/Qwen3.8-27B-CQ-4bit --local-dir ./models/Qwen3.8-27B-CQ-4bit
```

**Train a GCMP adapter** on the CQ base (fits a 48 GB GPU):

```bash
python training/train.py \
  --model ./models/Qwen3.8-27B-CQ-4bit \
  --data train.jsonl \
  --output ./outputs/my-gcmp \
  --kernel-module cq.cq_kernels \
  --adapter gcmp
```

**Serve it:**

```bash
CQ_GCMP_ADAPTER=./outputs/my-gcmp \
vllm serve ./models/Qwen3.8-27B-CQ-4bit --quantization cq --max-model-len 8192
```

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "./models/Qwen3.8-27B-CQ-4bit",
  "messages": [{"role": "user", "content": "Explain fine-tuning in one sentence."}],
  "max_tokens": 128,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

**No quantization.** GCMP doesn't need CQ. To train on the original 16-bit model (27B needs an 80 GB GPU):

```bash
python training/train_unsloth_16bit_gcmp.py \
  --model Qwen/Qwen3.8-27B --dtype bfloat16 \
  --train-data train.jsonl --output ./outputs/my-gcmp-bf16
```

Serve it with plain vLLM, no `--quantization` flag:

```bash
CQ_GCMP_ADAPTER=./outputs/my-gcmp-bf16/final_adapter \
vllm serve Qwen/Qwen3.8-27B --max-model-len 8192 --max-num-batched-tokens 4096
```

**On your own cloud account** (AWS, GCP, Azure, RunPod, Lambda, Kubernetes and others), the `thousandml` CLI trains and serves with one command. See [docs/CLOUD.md](docs/CLOUD.md).

## Results

### Setup

All accuracy results use the same model, data, evaluation code, prompts and questions. Differences come only from the base model and the adapter.

| | |
|---|---|
| Model | Qwen3.8-27B |
| Bases | BF16 (no quantization); NF4 (bitsandbytes, Unsloth's prequantized checkpoint); CQ 4-bit |
| Adapters | GCMP; LoRA r=32; LoRA r=8; VeRA r=256. All of them adapt the same 7 projections: q, k, v, o, gate, up and down |
| Training | Alpaca, 500 steps, batch 2 × gradient accumulation 4, length 512, LR 2e-5, seed 42 |
| Trainers | Unsloth for BF16 and NF4; Training Factory for CQ |
| Benchmarks | lm-evaluation-harness, greedy decoding, thinking off. GSM8K 300 questions, IFEval 200, SQuAD / ARC-Challenge / WinoGrande / BoolQ 500 each, HellaSwag 1,000, MMLU 1,140, TruthfulQA MC2 817 |
| Loss | Per-token loss on 2,600 held-out Alpaca rows, with the same tokenizer and formula for every case. *Answer loss* counts response tokens only |
| Statistics | Paired exact McNemar test per benchmark; paired bootstrap 95% CI for losses. At these sample sizes, differences of about 5 points are detectable and differences of 1 point are not |

### 1. GCMP vs LoRA and VeRA, full-precision base

On the BF16 base, GCMP, LoRA r=32, LoRA r=8 and VeRA are tied on all 9 benchmarks, and GCMP has the lowest held-out loss.

| Benchmark | BF16 base | LoRA r=32 | LoRA r=8 | VeRA | **GCMP** |
|---|---:|---:|---:|---:|---:|
| Trainable params | – | 159M | 40M | 2.89M | **2.9M** |
| GSM8K | 68.3 | 90.3 | 91.0 | 90.3 | 89.0 |
| IFEval | 80.5 | 82.5 | 79.5 | 82.0 | 81.5 |
| SQuAD completion | 74.6 | 77.6 | 76.8 | 77.8 | 76.0 |
| ARC-Challenge | 58.8 | 61.2 | 62.0 | 61.0 | 61.8 |
| HellaSwag | 73.7 | 73.7 | 73.7 | 73.4 | 73.8 |
| WinoGrande | 75.8 | 77.0 | 75.6 | 77.4 | 76.0 |
| BoolQ | 87.4 | 89.6 | 90.2 | 89.8 | 89.2 |
| MMLU | 83.9 | 84.4 | 84.6 | 84.6 | 84.6 |
| TruthfulQA MC2 | 54.2 | 51.8 | 52.8 | 52.0 | 51.4 |
| Answer loss ↓ | 1.678 | 1.480 | 1.555 | 1.489 | **1.471** |
| Final validation loss ↓ | – | 1.2434 | 1.2579 | 1.2488 | **1.2385** |

- **GCMP vs LoRA r=32:** no significant difference on any benchmark (smallest p = 0.22). GCMP's answer loss is lower by 0.009, and the 95% CI excludes zero. Its validation loss was lower at all 10 checkpoints.
- **GCMP vs LoRA r=8:** no significant difference (smallest p = 0.13). LoRA r=8 and r=32 are also tied, so the fair size comparison is with r=8: GCMP uses about 14× fewer parameters.
- **GCMP vs VeRA:** no significant difference (smallest p = 0.15). VeRA is the closest alternative, with the same size and quality, and it currently trains 2× faster. vLLM can't serve VeRA adapters, though.
- **Every adapter vs the base:** GSM8K rises by 21–23 points (p < 1e-10). TruthfulQA drops by 1.4–2.8 points for every adapter, which is an effect of Alpaca training rather than of any one adapter.

### 2. GCMP on 4-bit bases

| Benchmark | NF4 base | NF4 + LoRA r=32 | **NF4 + GCMP** | CQ base | CQ + LoRA r=32 | **CQ + GCMP** |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 56.3 | 79.3 | **83.3** | 65.0 | 58.7 | 82.7 |
| IFEval | 82.5 | 83.5 | 84.0 | 82.0 | 78.5 | 79.0 |
| SQuAD completion | 71.6 | 74.8 | 75.6 | 74.6 | 71.4 | 75.6 |
| ARC-Challenge | 58.2 | 60.8 | 61.8 | 59.8 | 62.2 | 62.0 |
| HellaSwag | 73.6 | 73.4 | 73.3 | 73.5 | 72.0 | 73.1 |
| WinoGrande | 77.0 | 76.6 | 76.8 | 78.0 | 78.6 | 77.2 |
| BoolQ | 85.2 | 88.0 | 87.8 | 88.0 | 90.2 | 90.4 |
| MMLU | 83.9 | 84.0 | 83.7 | 84.0 | 83.2 | 84.0 |
| TruthfulQA MC2 | 55.0 | 52.8 | 52.2 | 53.7 | 53.3 | 52.7 |
| Answer loss ↓ | 1.654 | 1.464 | **1.458** | 1.669 | 1.517 | 1.503 |

- **NF4 base, GCMP vs LoRA r=32:** GCMP is 4.0 points higher on GSM8K (p = 0.043), and the two are tied everywhere else (p ≥ 0.36). GCMP also has the lower answer and validation loss.
- **CQ + GCMP vs the CQ base:** GSM8K +17.7 (p = 6e-10).
- **CQ + GCMP vs NF4 + GCMP:** tied on GSM8K (82.7 vs 83.3, p = 0.89). CQ is 2.6 points higher on BoolQ (p = 0.015) and 5.0 points lower on IFEval (p = 0.053, borderline). These two use different trainers, so this compares the full stacks, not only the base formats.
- **4-bit vs full precision after GCMP:** NF4 + GCMP and CQ + GCMP are both about 6 points below BF16 + GCMP on GSM8K (p ≈ 0.01). CQ's base-model lead over NF4 does not carry over after fine-tuning in this test.
- **CQ + GCMP vs CQ + LoRA r=32 (same trainer and settings):** GCMP is 24.0 points higher on GSM8K (p = 2e-14) and 4.2 higher on SQuAD (p = 0.005), and tied on the other benchmarks. Its answer loss is also lower (−0.013, CI excludes 0).
- **Read the CQ + LoRA result with care:** CQ + LoRA scores *below* the CQ base on GSM8K (58.7 vs 65.0, p = 0.056), while NF4 + LoRA with the same settings reaches 79.3. Its validation loss also rose during training (1.29 → 1.36). Both point to a remaining problem in the CQ trainer's LoRA path rather than to LoRA itself, so we don't claim GCMP beats LoRA by 24 points in general. GCMP on CQ works out of the box; we're investigating the LoRA path.

### 3. Serving many adapters on one GPU

The setup is one A40 48 GB, vLLM, the CQ base and 32 concurrent requests. Every request picks a random adapter, so each batch mixes tenants. The adapters are perturbed copies of a trained adapter.

| Setup | Adapters | GPU adapter slots | p95 TTFT (s) | p95 latency (s) | KV cache (tokens) | Ready to serve |
|---|---:|---:|---:|---:|---:|---:|
| **GCMP** | 256 | all resident | **2.18** | **49.0** | **72,362** | **~6 min** |
| LoRA r=8, tuned | 256 | 64 | 2.38 | 53.9 | 43,349 | ~41 min |
| LoRA r=8, default | 256 | 16 | 47.5 | 93.4 | 88,064 | ~30 min |
| LoRA r=32, default | 64 | 16 | 53.1 | 105.4 | 43,349 | ~18 min |
| LoRA r=32, 32 slots | 64 | 32 | – | – | −1.42 GiB | never starts |
| LoRA r=32, default | 256 | 16 | – | – | – | runs out of host RAM |
| Base, no adapter | – | – | 1.55 | 44.0 | 111,274 | – |

- **GCMP throughput is flat** at about 84 output tok/s from 1 to 256 adapters, with no tuning.
- **Default multi-LoRA** (r=8) falls to about 60 tok/s by 64 adapters. Raising it to 64 GPU slots recovers 74 tok/s, but the slots take memory up front and the KV cache drops from 88k to 43k tokens.
- **LoRA r=32** can't run 32 slots on a 48 GB GPU. With 256 adapters it needs about 82 GB of host RAM, because vLLM keeps every registered adapter in CPU memory; 256 GCMP adapters take about 1.5 GB in bf16 (3 GB in fp32).
- **Startup:** vLLM registers LoRA adapters one at a time, about 6 s each at r=8 and about 18 s at r=32.
- **With a single adapter,** LoRA is about 9% faster than GCMP (91 vs 83 tok/s).

### 4. Tenant turnover over 45 minutes

This test uses the same GPU and base. Open-loop Poisson traffic runs at 0.45 req/s with 128 output tokens, for 45 minutes per run. LoRA r=8 gets 64 GPU slots and 64 adapters cached in host RAM.

- **Skew:** 256 adapters with Zipf(1.1) popularity: a few busy tenants and a long tail of rare ones.
- **Churn:** 64 adapters. Every 20 s, one tenant leaves and a new one takes its place, after the slot has been drained. LoRA uses vLLM's runtime unload/load API. GCMP replaces the adapter in place.

| Metric | GCMP skew | LoRA r=8 skew | GCMP churn | LoRA r=8 churn |
|---|---:|---:|---:|---:|
| Requests ok / failed | 1,219 / 0 | 1,218 / 1 | 1,386 / 0 | 1,352 / 2 |
| TTFT p50 / p99 (s) | **0.70 / 1.18** | 6.90 / 91.1 | **0.82 / 1.25** | 7.26 / 56.9 |
| TTFT p99, rarest half of tenants (s) | **1.17** | 92.8 | **1.24** | 56.3 |
| Latency p99 (s) | **48.5** | 162.7 | **50.0** | 112.1 |
| Swap time p50 / p95 (s) | – | – | **0.43 / 0.72** | 5.39 / 11.8 |
| New tenant's first request, TTFT p95 (s) | – | – | **1.21** | 41.4 |
| KV cache usage, max | 62% | 98% | 56% | 98% |
| GPU memory drift over 45 min | +3 MB | +6 MB | 0 | +3 MB |

Most of LoRA's latency comes from capacity. Its GPU slots shrink the KV cache to 43k tokens, which sat about 98% full, so busy tenants waited as long as rare ones. Neither system showed memory leaks or fragmentation.

### 5. The CQ base model

| Base model | GSM8K | IFEval | SQuAD | ARC-C | BoolQ | MMLU | HellaSwag | WinoGrande | TruthfulQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 (full precision) | 68.3 | 80.5 | 74.6 | 58.8 | 87.4 | 83.9 | 73.7 | 75.8 | 54.2 |
| **CQ 4-bit** | 65.0 | 82.0 | 74.6 | 59.8 | 88.0 | 84.0 | 73.5 | 78.0 | 53.7 |
| NF4 4-bit | 56.3 | 82.5 | 71.6 | 58.2 | 85.2 | 83.9 | 73.6 | 77.0 | 55.0 |

- **CQ vs BF16:** no significant difference on any benchmark.
- **NF4 vs BF16:** NF4 is 12.0 points lower on GSM8K (p = 3e-6) and 3.0 points lower on SQuAD (p = 0.049).
- **CQ vs NF4:** CQ is 8.7 points higher on GSM8K (p = 0.0005) and 2.8 points higher on BoolQ (p = 0.02).

| Fidelity vs BF16 (WikiText-2) | CQ 4-bit | NF4 |
|---|---:|---:|
| Checkpoint size | **18 GB** | 22.4 GB |
| Mean KL divergence ↓ | **0.0282** | 0.0330 |
| 99th-percentile KL ↓ | **0.976** | 1.181 |
| Top-1 agreement ↑ | **92.42%** | 91.95% |

### 6. Training speed and memory

All runs: 500 steps, batch 2 × 4, length 512.

| Base + adapter | Trainer | GPU | s/step ↓ | Peak memory |
|---|---|---|---:|---:|
| BF16 + LoRA r=32 | Unsloth | 80 GB | 1.37 | 53.7 GiB |
| BF16 + LoRA r=8 | Unsloth | 80 GB | **1.31** | 52.0 GiB |
| BF16 + VeRA | Unsloth | 80 GB | 1.33 | 51.4 GiB |
| BF16 + GCMP | Unsloth + GCMP | 80 GB | 2.74 | 51.7 GiB |
| NF4 + LoRA r=32 | Unsloth | A40 48 GB | **3.20** | 23.8 GiB |
| NF4 + GCMP | Unsloth + GCMP | A40 48 GB | 5.60 | **21.5 GiB** |
| CQ + LoRA r=32 | Training Factory | A40 48 GB | 7.07 | 24.4 GiB |
| CQ + GCMP | Training Factory | A40 48 GB | 9.05 | 22.0 GiB |

GCMP uses the least memory on each base but currently takes about 2× LoRA's step time. Kernel work to close that gap is in progress. The CQ trainer is slower than Unsloth.

> **About losses in training logs:** the CQ trainer and Unsloth tokenize, mask and average the loss differently, so their logged losses aren't on the same scale. To compare runs, use the same-formula losses and the benchmarks above.

## Reproduce the benchmarks

Every number above comes from the scripts in `training/`. The steps below run them in order.

| Step | What it produces | GPU | Rough time |
|---|---|---|---|
| 1. Setup | environment, models | any | 30 min |
| 2. Train 9 adapters | `outputs/` | 48 GB for 4-bit, 80 GB for BF16 | 25–75 min per adapter |
| 3. Downstream evaluation | `eval_downstream/<case>/` | 48 GB for 4-bit, 80 GB for BF16 | a few hours per case; CQ is the slowest |
| 4. Comparison report | `report/compare.md`, `report/training.md` | none | 1 min |
| 5. Serving scaling benchmark | `scaling/results/summary.md` | 48 GB | several hours (mostly LoRA startup) |
| 6. Turnover benchmark | `scaling/turnover/summary.md` | 48 GB | ~5 h |

Sections 1, 2, 5 and 6 of the results ran on RunPod A40 48 GB machines; the BF16 runs used A100/H100 80 GB GPUs.

### Step 1. Setup (on every machine)

```bash
git clone https://github.com/thousandml/training-factory.git
cd training-factory
./install.sh
bash training/setup_eval_machine.sh            # lm-evaluation-harness and eval dependencies

# The CQ base. NF4 (unsloth/Qwen3.8-27B-unsloth-bnb-4bit) and BF16 (Qwen/Qwen3.8-27B)
# are downloaded from Hugging Face automatically when first used.
hf download thousandml/Qwen3.8-27B-CQ-4bit --local-dir ./models/Qwen3.8-27B-CQ-4bit

# Every script below reads these
export ROOT=$PWD
export ADAPTERS=$PWD/outputs
export CQ_MODEL=$PWD/models/Qwen3.8-27B-CQ-4bit
```

Check that the kernels load before you start a long run:

```bash
python -c "import torch, cq.cq_kernels; print('CUDA:', torch.cuda.is_available())"
```

### Step 2. Train the adapters

All runs use `training/examples/alpaca_train.jsonl` and `training/examples/alpaca_val.jsonl`, 500 steps, batch 2 × gradient accumulation 4, length 512, LR 2e-5 and seed 42. Keep the output folder names: the evaluation script looks for adapters under these names.

**CQ base, 48 GB GPU** (Training Factory):

```bash
COMMON="--model $CQ_MODEL \
  --data training/examples/alpaca_train.jsonl --eval-data training/examples/alpaca_val.jsonl \
  --kernel-module cq.cq_kernels --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch-size 2 --grad-accum 4 --max-length 512 --max-steps 500 --learning-rate 2e-5 \
  --seed 42 --eval-steps 50 --comparison-logs --benchmark --benchmark-warmup-steps 10"

python training/train.py $COMMON --adapter gcmp --gcmp-scale 1.0 \
  --output ./outputs/qwen38-27b-cq-gcmp-v2-fix

python training/train.py $COMMON --adapter lora --rank 32 --alpha 32 --lora-dropout 0 \
  --output ./outputs/qwen38-27b-cq-r32-v8-fix
```

**NF4 base, 48 GB GPU** (Unsloth):

```bash
NF4="--model unsloth/Qwen3.8-27B-unsloth-bnb-4bit \
  --train-data training/examples/alpaca_train.jsonl --eval-data training/examples/alpaca_val.jsonl \
  --max-steps 500 --max-length 512 --eval-steps 50 --save-steps 100 --seed 42 --benchmark-warmup-steps 10"

python training/train_unsloth_nf4.py $NF4 --adapter gcmp --output ./outputs/qwen38-27b-unsloth-nf4-alpaca-gcmp-v1
python training/train_unsloth_nf4.py $NF4 --adapter lora --output ./outputs/qwen38-27b-unsloth-nf4-r32-v5
```

**BF16 base, 80 GB GPU** (Unsloth). This trains LoRA r=32, LoRA r=8, VeRA and GCMP with identical settings:

```bash
VERA_LR=<lr> bash training/run_unsloth_16bit.sh              # all four
bash training/run_unsloth_16bit.sh gcmp lora                  # or selected ones: lora | lora8 | vera | gcmp
```

Each run saves `comparison_config.json` (every setting used), `comparison_summary.json` (step time, peak memory) and validation losses by step.

If you train on one machine and evaluate on another, copy the whole adapter folders into `$ADAPTERS` on the evaluation machine with the same names.

### Step 3. Evaluate

```bash
bash training/run_downstream.sh 4bit    # 48 GB GPU: cq_base cq_lora cq_gcmp nf4_base nf4_lora nf4_gcmp
bash training/run_downstream.sh bf16    # 80 GB GPU: bf16_base bf16_lora bf16_lora_r8 bf16_vera bf16_gcmp

bash training/run_downstream.sh 4bit cq_gcmp nf4_gcmp     # selected cases only
SUITE=full bash training/run_downstream.sh 4bit           # every question instead of the subset above
```

Each case runs three stages and saves them in `eval_downstream/<case>/`:

| Stage | Output | What it is |
|---|---|---|
| `loss` | `loss.json` | same-formula text loss and answer loss on 2,600 held-out rows |
| `gen` | `metrics.json`, `predictions.jsonl` | generated answers, ROUGE/BLEU and a 90-item answer key |
| `tasks` | `tasks/<task>.json` | the lm-evaluation-harness benchmarks |

A stage that already finished is skipped, so you can rerun the same command after an interruption.

**Make sure each case is using the right adapter.** Each run writes a `== <case>  base=...  adapter=...` line to `eval_downstream/<case>/run.log`, and the script stops a case whose adapter folder doesn't exist. If you named a folder differently, point the script at it with `CQ_LORA`, `CQ_GCMP`, `NF4_LORA`, `NF4_GCMP`, `BF16_LORA`, `BF16_LORA8`, `BF16_VERA` or `BF16_GCMP`:

```bash
CQ_GCMP=/path/to/my-cq-gcmp bash training/run_downstream.sh 4bit cq_gcmp
```

To rerun a case from scratch (for example, after replacing its adapter), move its folder away first: `mv eval_downstream/cq_lora eval_downstream/cq_lora_old`.

Quick health checks while it runs:

```bash
tail -f eval_downstream/<case>/run.log
cat eval_downstream/<case>/loss.json           # adapters land at about 1.46–1.56 answer loss; bases at about 1.65–1.68
grep -c '###' eval_downstream/<case>/predictions.jsonl   # should be close to 0; a large count means broken outputs
```

### Step 4. Build the report

On one machine:

```bash
python training/compare_cases.py eval_downstream --output report
```

Across machines, pack the results on each one (no weights or checkpoints), copy the archives to one place, and combine them:

```bash
NAME=gpu40 bash training/collect_results.sh      # on the 48 GB machine -> results_gpu40_<date>.tar.gz
NAME=gpu80 bash training/collect_results.sh      # on the 80 GB machine -> results_gpu80_<date>.tar.gz

bash training/report_all.sh results_gpu40_*.tar.gz results_gpu80_*.tar.gz
```

| File | Contents | README section |
|---|---|---|
| `report/compare.md` | a table per case; paired GCMP vs LoRA/VeRA, adapter vs base and CQ vs NF4 vs BF16 tests, with McNemar p-values and bootstrap CIs | 1, 2, 5 |
| `report/training.md` | every training run: validation loss by step, s/step, peak memory | 6 |

[`training/DOWNSTREAM_EVAL.md`](training/DOWNSTREAM_EVAL.md) explains every metric.

### Step 5. Serving scaling benchmark

This needs one 48 GB GPU, the CQ base, one trained GCMP adapter and one trained CQ LoRA r=32 adapter. The script makes 256 perturbed copies of each; the r=8 copies are the r=32 adapter cut to rank 8. Speed doesn't depend on the adapter's values, so any trained adapter works.

```bash
# prompt file (also created by step 3)
mkdir -p eval_downstream
python training/prepare_eval_data.py training/examples/alpaca_val.jsonl eval_downstream/alpaca_val_eval.jsonl

export MODEL=$CQ_MODEL WORK=$PWD/scaling
export GCMP_SRC=$ADAPTERS/qwen38-27b-cq-gcmp-v2-fix
export LORA_SRC=$ADAPTERS/qwen38-27b-cq-r32-v8-fix

bash training/run_adapter_scaling.sh                          # base, GCMP, LoRA r=32 and r=8 at 1/16/64/256 adapters
CONFIGS="gcmp lora8" NS="64 256" bash training/run_adapter_scaling.sh   # a subset
```

The script sweeps vLLM's `--max-loras` (GPU slots) at the largest adapter count: `SWEEP8="32 64"` for r=8 and `SWEEP32="32"` for r=32. The results go to `scaling/results/summary.md` (tok/s, latency, TTFT and KV cache size), with one JSON per case and the server log next to each. Registering 256 LoRA adapters can take 30–40 minutes, so the script waits up to 90 minutes for the server (`WAIT_TRIES`, in 10-second steps).

**Host RAM:** 256 LoRA r=32 adapters need about 82 GB of host RAM. On a pod with a lower memory limit (`cat /sys/fs/cgroup/memory.max`), that case stalls part way through registration.

### Step 6. Turnover benchmark

Run this on the same machine and settings as step 5. It reuses the adapter copies from step 5 when they exist.

```bash
bash training/run_turnover.sh                                 # GCMP and LoRA r=8, skew and churn, 45 min each (~5 h)
KINDS="gcmp" MODES="churn" bash training/run_turnover.sh      # one case
DURATION=600 bash training/run_turnover.sh                    # 10-minute smoke test
```

| Setting | Default | Meaning |
|---|---|---|
| `DURATION` | 2700 | seconds per run |
| `RATE` | 0.45 | requests per second (open-loop Poisson) |
| `ZIPF` | 1.1 | tenant popularity skew |
| `SKEW_N` / `CHURN_N` | 256 / 64 | adapters in the skew / churn runs |
| `CHURN_EVERY` | 20 | seconds between tenant swaps |
| `LORA_CPU` | 64 | LoRA adapters cached in host RAM |
| `LORA_SLOTS8` / `LORA_SLOTS32` | 64 / 16 | LoRA GPU slots |

The results go to `scaling/turnover/summary.md`, with `result.json` and the raw per-request data for each run. The script samples GPU memory, KV cache usage and host memory every 5 s. LoRA swaps use vLLM's runtime adapter API (`VLLM_ALLOW_RUNTIME_LORA_UPDATING`). GCMP swaps go through `training/gcmp_swap_ext.py`, which the script loads as a vLLM worker extension.

## Training options

Training data is JSONL, with either plain text or chat messages:

```json
{"text": "Question: When does the warranty expire? Answer: Check the purchase date."}
{"messages": [{"role": "user", "content": "Summarize LoRA."}, {"role": "assistant", "content": "It trains small adapters on a frozen base."}]}
```

| Option | Default | Notes |
|---|---|---|
| `--adapter` | `gcmp` | `gcmp` or `lora` (PEFT LoRA) |
| `--targets` | all attention + MLP projections | comma-separated projection names |
| `--rank`, `--alpha`, `--lora-dropout` | 16, 32, 0.05 | LoRA only |
| `--gcmp-scale` | 1.0 | GCMP output scale |
| `--save-dtype` | `bf16` | GCMP: dtype of the final adapter file (`bf16` or `fp32`). Checkpoints stay fp32, so `--resume` is exact. Also in the Unsloth scripts |
| `--learning-rate` | 2e-4 | the benchmarks used 2e-5 |
| `--batch-size`, `--grad-accum` | | effective batch = product |
| `--max-length`, `--max-steps` / `--epochs` | | |
| `--eval-data`, `--eval-steps` | | validation loss during training |
| `--resume` | | continue from a checkpoint |

Run `python training/train.py --help` for every option. Adapters are saved on their own (`adapter_config.json` and `adapter_model.safetensors`). Don't merge them into the CQ base.

GCMP also trains on non-CQ bases through Unsloth: `training/train_unsloth_16bit_gcmp.py` for BF16/FP16 and `training/train_unsloth_nf4.py --adapter gcmp` for NF4. vLLM serves those full-precision adapters directly (see [Serving](#serving)). The evaluation scripts also load them in Transformers, for example `python training/downstream_eval.py --backend unsloth --model Qwen/Qwen3.8-27B --adapter ./outputs/my-gcmp-bf16/final_adapter --output-dir eval/my-gcmp-bf16`.

## Serving

**One adapter:** set `CQ_GCMP_ADAPTER`, as in the quick start. `CQ_GCMP_ADAPTER_SCALE` overrides the adapter's scale.

**Which base:** adapters trained on CQ (`training/train.py`) are served on the CQ model with `--quantization cq`. Adapters trained on a full-precision model (`train_unsloth_16bit_gcmp.py`) are served on that same model with no `--quantization` flag. Everything below works on both. All adapters on one server must be of the same kind; a mismatch stops the server at startup instead of silently serving the base model.

**Full-precision check (Qwen3.8-27B, BF16, one 80 GB GPU):** the BF16 GCMP adapter served by vLLM lowers held-out answer loss from 1.623 to 1.436 on 200 rows, the same improvement the Transformers evaluation measured (1.678 → 1.471 on all 2,600 rows, with a slightly different loss formula). The multi-adapter speed benchmarks in [Results](#results) were run on the CQ base; full-precision speed numbers are next.

**Memory on full-precision bases:** GCMP adds a small fp32 buffer per layer during prefill, so keep `--max-num-batched-tokens` at about 4096 on a model that nearly fills the GPU (the default 16384 ran out of memory with BF16 27B on 80 GB).

**Many adapters on one server:**

```bash
CQ_GCMP_ADAPTERS="support=./outputs/support,sql=./outputs/sql" \
vllm serve ./models/Qwen3.8-27B-CQ-4bit --quantization cq
```

Choose the adapter per request with the model name (`"model": "sql"`), or keep the base model name and pass `"vllm_xargs": {"gcmp_adapter": "sql"}`. `/v1/models` lists the adapters. All of them stay on the GPU, so no slot setting needs tuning.

**Swapping a tenant without a restart:** start vLLM with `VLLM_SERVER_DEV_MODE=1 PYTHONPATH=training` and `--worker-extension-cls gcmp_swap_ext.GCMPSwap`, then replace the adapter in an existing slot:

```bash
curl localhost:8000/collective_rpc -H 'Content-Type: application/json' \
  -d '{"method": "gcmp_swap", "args": ["sql", "./outputs/sql-v2"]}'
```

**LoRA adapters** use vLLM's standard flags:

```bash
vllm serve ./models/Qwen3.8-27B-CQ-4bit --quantization cq \
  --enable-lora --max-lora-rank 32 --lora-modules my-lora=./outputs/my-lora
```

**Multiple GPUs:** CQ models support pipeline parallelism in vLLM, but not tensor parallelism. GCMP adapters work across pipeline stages.

```bash
CQ_GCMP_ADAPTER=./outputs/my-gcmp \
vllm serve ./models/Qwen3.8-27B-CQ-4bit --quantization cq --pipeline-parallel-size 2
```

Qwen3.x models start in thinking mode. Pass `"chat_template_kwargs": {"enable_thinking": false}`, or start vLLM with `--reasoning-parser qwen3`.

## Models

| Model | Hugging Face |
|---|---|
| Qwen3.8-27B | [thousandml/Qwen3.8-27B-CQ-4bit](https://huggingface.co/thousandml/Qwen3.8-27B-CQ-4bit) |
| Llama-3.1-8B-Instruct | [thousandml/Llama-3.1-8B-Instruct-CQ-4bit](https://huggingface.co/thousandml/Llama-3.1-8B-Instruct-CQ-4bit) |
| Llama-3.3-70B-Instruct | [thousandml/Llama-3.3-70B-Instruct-CQ-4bit](https://huggingface.co/thousandml/Llama-3.3-70B-Instruct-CQ-4bit) |
| Qwen3-Coder-Next | [thousandml/Qwen3-Coder-Next-CQ-4bit](https://huggingface.co/thousandml/Qwen3-Coder-Next-CQ-4bit) |

[All models →](https://huggingface.co/thousandml)

To quantize your own model, run these from the repo root:

```bash
# 1. Quantize (calibration-free)
./quantize-model YOUR_ORG/MODEL --output-dir ./models/MODEL-CQ-4bit

# 2. Build the CQ checkpoint in that folder. --no-upload keeps it local.
python build_convert_upload.py \
  --model YOUR_ORG/MODEL \
  --quantized-model-dir ./models/MODEL-CQ-4bit \
  --repo-id YOUR_USERNAME/MODEL-CQ-4bit \
  --no-upload
```

The folder is then ready for `--model` in training and for `vllm serve --quantization cq`. Without `--no-upload`, step 2 also uploads the checkpoint to `--repo-id` (run `hf auth login` first). If step 2 fails after the shards are built, rerun it with `--skip-convert`.

## Limitations

- **Scope of the results:** one model (Qwen3.8-27B), one dataset (Alpaca), one seed per case. The benchmark subset detects differences of about 5 points, not 1. GCMP ran at LoRA's learning rate, with no sweep. DROP is left out because every model scored about 0.02 on it.
- **Speed:** GCMP training takes about 2× LoRA's step time. With a single adapter, serving is about 9% slower than LoRA. The CQ trainer is slower than Unsloth, and its validation losses (about 1.28–1.29) are still a little above the NF4 and BF16 runs (about 1.24).
- **Serving tests:** synthetic adapter copies, short prompts and 128-token outputs, on an A40 with the CQ base. Long contexts would put more weight on the KV cache. On a full-precision base, accuracy is verified in vLLM but the multi-adapter speed benchmarks haven't been run yet.
- **Tenant swaps:** a GCMP swap reuses an existing adapter slot, so adding a new name still needs a restart, and the swap hook uses vLLM's development endpoint. LoRA can register new names at runtime.
- **Parallelism and bases:**

  | | 1 GPU | Pipeline parallel | Tensor parallel |
  |---|:-:|:-:|:-:|
  | Training (any base) | ✓ | – | – |
  | vLLM serving, CQ base + GCMP or LoRA | ✓ | ✓ | – |
  | vLLM serving, full-precision base + GCMP | ✓ | untested | – |
  | Full-precision base + GCMP in Transformers | ✓ | – | – |

  CQ MoE models use TP=1 with expert parallelism. MoE expert weights and non-CQ dense layers are not adapted.
- **Your workload:** measure quality, speed and memory on your own data before relying on these numbers.

## Troubleshooting

```bash
python -c "import torch, cq.cq_kernels; print('CUDA:', torch.cuda.is_available())"
```

- **Kernel import:** the module is `cq.cq_kernels`, not `cq_kernels`.
- **Native binaries:** they must match your Python (3.12), CUDA and PyTorch versions.
- **License:** set `CQ_LICENSE_KEY` / `CQ_LICENSE_URL` if your distribution needs a license.
- **Windows line endings:** if you get a `bash\r` error, run `sed -i 's/\r$//' install.sh`.
- **"Stale file handle" on a network disk:** don't edit a script while it's running from that disk. Run a copy from `/tmp`.

## Citation

```bibtex
@software{elfadil2026trainingfactory,
  author       = {Elfadil, Husam},
  title        = {Training Factory: Fine-Tuning with GCMP Adapters on Full-Precision and CQ 4-bit Models},
  organization = {ThousandML},
  year         = {2026},
  url          = {https://github.com/thousandml/training-factory}
}
```

## Contributing

Bug reports, hardware measurements, model-compatibility reports and independent benchmark runs are welcome. Please open an [issue](https://github.com/thousandml/training-factory/issues).

**ThousandML: Making Every Bit Count.**
