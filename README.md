# Training Factory

**Fine-tune open LLMs on your own GPUs with GCMP, an adapter that matches LoRA with 55× fewer parameters, on a full-precision base or a CQ 4-bit base.**

[Website](https://www.thousandml.com) · [Models](https://huggingface.co/thousandml) · [Issues](https://github.com/thousandml/training-factory/issues)

- **GCMP**: a parameter-efficient adapter that replaces LoRA. It matches LoRA r=32 quality with **55× fewer trainable parameters**: 2.9M vs ~159M, and an ~11 MB adapter vs ~320 MB. It works on **full-precision (BF16/FP16) models** with no quantization, and on 4-bit bases.
- **CQ** (optional): calibration-free 4-bit quantization. On Qwen3.8-27B it matches the full-precision model on downstream benchmarks, and the checkpoint is 18 GB.
- **vLLM serving**: a CQ base model with one or many GCMP adapters, with the adapter chosen per request.

## Results (Qwen3.8-27B)

The same evaluation code, prompts and questions were used for every row. Decoding is greedy. Significance comes from paired tests: an exact McNemar test for right/wrong tasks and a bootstrap for losses.

### 4-bit base quality: CQ matches BF16, NF4 does not

| Base model | GSM8K | SQuAD | BoolQ | ARC-C | MMLU | IFEval | HellaSwag | WinoGrande | TruthfulQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 (full precision) | 0.683 | 0.746 | 0.874 | 0.588 | 0.839 | 0.805 | 0.737 | 0.758 | 0.542 |
| **CQ 4-bit** | 0.650 | **0.746** | **0.880** | **0.598** | **0.840** | **0.820** | 0.735 | **0.780** | 0.537 |
| NF4 4-bit (bitsandbytes) | 0.563 | 0.716 | 0.852 | 0.582 | 0.839 | 0.825 | 0.736 | 0.770 | 0.550 |

- **CQ vs BF16**: no significant difference on any task.
- **NF4 vs BF16**: NF4 is 12.0 points lower on GSM8K (p = 3e-6).
- **CQ vs NF4**: CQ is higher on GSM8K by 8.7 points (p = 0.0005) and on BoolQ by 2.8 points (p = 0.02).

Questions per task: IFEval 200, GSM8K 300, SQuAD/ARC/WinoGrande/BoolQ 500, HellaSwag 1000, MMLU 1140, TruthfulQA 817.

<details>
<summary>Quantization fidelity on WikiText-2 (vs the full-precision model)</summary>

| Metric | CQ 4-bit | NF4 (bitsandbytes) |
|---|---:|---:|
| Checkpoint size | **18 GB** | 22.4 GB |
| Mean KL divergence ↓ | **0.0282** | 0.0330 |
| 99th-percentile KL ↓ | **0.976** | 1.181 |
| Top-1 agreement ↑ | **92.42%** | 91.95% |

</details>

### GCMP vs LoRA r=32 on a full-precision base (no quantization)

The base is the original BF16 Qwen3.8-27B. Both adapters were trained with Unsloth, using the same settings. This comparison involves no quantization, so it shows GCMP on its own.

| | Trainable params | Answer loss ↓ | GSM8K | SQuAD | BoolQ | ARC-C | MMLU | IFEval | TruthfulQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No adapter | – | 1.678 | 0.683 | 0.746 | 0.874 | 0.588 | 0.839 | 0.805 | 0.542 |
| LoRA r=32 | ~159M | 1.480 | 0.903 | 0.776 | 0.896 | 0.612 | 0.844 | 0.825 | 0.518 |
| **GCMP** | **2.9M** | **1.471** | 0.890 | 0.760 | 0.892 | 0.618 | 0.846 | 0.815 | 0.514 |

- **GCMP vs LoRA**: no significant difference on any benchmark; McNemar p ≥ 0.2 on every right/wrong task.
- **Loss**: GCMP's answer-token loss is slightly lower (−0.009, 95% CI [−0.010, −0.008]).
- **Validation loss**: GCMP's is lower at all 10 checkpoints, on both BF16 (1.2385 vs 1.2434) and NF4 (1.2435 vs 1.2484) bases.
- **Effect of fine-tuning**: both adapters raise GSM8K by about 21 points, and both lower TruthfulQA by about 2.5 points, which is typical of Alpaca tuning.

**Pending:** benchmarks for LoRA and GCMP on CQ and NF4 bases. They will be added here when the runs finish.

### Training speed and memory

All runs use Alpaca, 500 steps, batch 2 × gradient accumulation 4, sequence length 512.

| Base + adapter | Trainer | GPU | s/step ↓ | Peak memory |
|---|---|---|---:|---:|
| CQ + LoRA r=32 | Training Factory | A40 48 GB | 8.6 | 23.5 GiB |
| CQ + GCMP | Training Factory | A40 48 GB | 12.4 | 22.0 GiB |
| NF4 + LoRA r=32 | Unsloth | A40 48 GB | **3.2** | 23.8 GiB |
| NF4 + GCMP | Unsloth + GCMP | A40 48 GB | 5.6 | **21.5 GiB** |
| BF16 + LoRA r=32 | Unsloth | 80 GB | **1.37** | 53.7 GiB |
| BF16 + GCMP | Unsloth + GCMP | 80 GB | 2.74 | 51.7 GiB |

Unsloth is still faster. GCMP currently costs about 2× the step time of LoRA, and kernel work to reduce that is in progress.

> **About training-log losses:** the CQ trainer and Unsloth tokenize, mask and average loss differently, so their logged losses are not on the same scale. An earlier version of this README compared them directly (0.84 vs 1.25). Use the same-formula losses and the benchmarks above instead.

## Quick start

**Requirements:** Linux x86-64 or WSL2, an NVIDIA GPU, Conda and Python 3.12. The native kernels ship compiled and need a ThousandML license check, which is free during early access.

```bash
git clone https://github.com/thousandml/training-factory.git
cd training-factory
./install.sh

hf download thousandml/Qwen3.8-27B-CQ-4bit --local-dir ./models/Qwen3.8-27B-CQ-4bit
```

**Train a GCMP adapter:**

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

### Full precision, no quantization

GCMP doesn't need CQ. To train it on the original 16-bit model (80 GB GPU for 27B; smaller models need less):

```bash
python training/train_unsloth_16bit_gcmp.py \
  --model Qwen/Qwen3.8-27B --dtype bfloat16 \
  --train-data train.jsonl --output ./outputs/my-gcmp-bf16
```

The evaluation scripts load the result in Transformers, for example `training/downstream_eval.py --backend unsloth --model Qwen/Qwen3.8-27B --adapter ./outputs/my-gcmp-bf16/final_adapter --output-dir eval/my-gcmp-bf16`. vLLM serving of GCMP adapters supports CQ bases only for now.

## Training

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
| `--learning-rate` | 2e-4 | |
| `--batch-size`, `--grad-accum` | | effective batch = product |
| `--max-length`, `--max-steps` / `--epochs` | | |
| `--eval-data`, `--eval-steps` | | validation loss during training |
| `--resume` | | continue from a checkpoint |

Run `python training/train.py --help` for every option. Adapters are saved on their own (`adapter_config.json` and `adapter_model.safetensors`). Don't merge them into the CQ base.

GCMP also works on non-CQ bases through Unsloth: `training/train_unsloth_16bit_gcmp.py` for full-precision BF16/FP16 models, and `training/train_unsloth_nf4.py --adapter gcmp` for NF4.

## Serving

**One adapter:** set `CQ_GCMP_ADAPTER`, as in the quick start. `CQ_GCMP_ADAPTER_SCALE` overrides the adapter's scale.

**Several adapters on one server (new; please report issues):**

```bash
CQ_GCMP_ADAPTERS="support=./outputs/support,sql=./outputs/sql" \
vllm serve ./models/Qwen3.8-27B-CQ-4bit --quantization cq
```

Pick an adapter per request by model name (`"model": "sql"`), or keep the base model name and pass `"vllm_xargs": {"gcmp_adapter": "sql"}`. Requests for different adapters run in the same batch, and `/v1/models` lists them.

**LoRA adapters** are served with vLLM's standard flags:

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

## Reproduce the results

All runs use `training/examples/alpaca_train.jsonl` and `training/examples/alpaca_val.jsonl` with seed 42, 500 steps, batch 2 × gradient accumulation 4, length 512 and learning rate 2e-5.

**1. Train the six adapters.** The 4-bit runs fit a 48 GB GPU; BF16 needs 80 GB.

```bash
COMMON="--data training/examples/alpaca_train.jsonl --eval-data training/examples/alpaca_val.jsonl \
  --kernel-module cq.cq_kernels --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch-size 2 --grad-accum 4 --max-length 512 --max-steps 500 --learning-rate 2e-5 \
  --seed 42 --eval-steps 50 --comparison-logs --benchmark --benchmark-warmup-steps 10"

# CQ + LoRA r=32
python training/train.py --model ./models/Qwen3.8-27B-CQ-4bit $COMMON \
  --adapter lora --rank 32 --alpha 32 --lora-dropout 0 \
  --output ./outputs/qwen38-27b-cq-r32-comparison-v4

# CQ + GCMP
python training/train.py --model ./models/Qwen3.8-27B-CQ-4bit $COMMON \
  --adapter gcmp --gcmp-scale 1.0 \
  --output ./outputs/qwen38-27b-cq-4bit-alpaca-gcmp-v1

# NF4 + LoRA r=32 and NF4 + GCMP (Unsloth)
NF4="--model unsloth/Qwen3.8-27B-unsloth-bnb-4bit \
  --train-data training/examples/alpaca_train.jsonl --eval-data training/examples/alpaca_val.jsonl \
  --max-steps 500 --max-length 512 --eval-steps 50 --save-steps 100 --seed 42 --benchmark-warmup-steps 10"
python training/train_unsloth_nf4.py $NF4 --adapter lora --output ./outputs/qwen38-27b-unsloth-nf4-r32-v5
python training/train_unsloth_nf4.py $NF4 --adapter gcmp --output ./outputs/qwen38-27b-unsloth-nf4-alpaca-gcmp-v1

# BF16 + LoRA r=32 and BF16 + GCMP (Unsloth, 80 GB GPU)
ROOT=$PWD bash training/run_unsloth_16bit.sh
```

**2. Evaluate.** This runs the same-formula loss, ROUGE/BLEU and the lm-evaluation-harness tasks for every case. Interrupted runs resume where they stopped.

```bash
bash training/setup_eval_machine.sh
export ROOT=$PWD ADAPTERS=$PWD/outputs CQ_MODEL=$PWD/models/Qwen3.8-27B-CQ-4bit

bash training/run_downstream.sh 4bit    # cq/nf4 × base/lora/gcmp  (48 GB GPU)
bash training/run_downstream.sh bf16    # bf16 × base/lora/gcmp     (80 GB GPU)
SUITE=full bash training/run_downstream.sh 4bit   # every question instead of the quick subset
```

**3. Compare.**

```bash
python training/compare_cases.py eval_downstream --output compare   # add more eval dirs if split across machines
```

`compare/compare.md` has a table per case and paired comparisons with confidence intervals and McNemar tests. [`training/DOWNSTREAM_EVAL.md`](training/DOWNSTREAM_EVAL.md) has the details.

## Models

| Model | Hugging Face |
|---|---|
| Qwen3.8-27B | [thousandml/Qwen3.8-27B-CQ-4bit](https://huggingface.co/thousandml/Qwen3.8-27B-CQ-4bit) |
| Llama-3.1-8B-Instruct | [thousandml/Llama-3.1-8B-Instruct-CQ-4bit](https://huggingface.co/thousandml/Llama-3.1-8B-Instruct-CQ-4bit) |
| Llama-3.3-70B-Instruct | [thousandml/Llama-3.3-70B-Instruct-CQ-4bit](https://huggingface.co/thousandml/Llama-3.3-70B-Instruct-CQ-4bit) |
| Qwen3-Coder-Next | [thousandml/Qwen3-Coder-Next-CQ-4bit](https://huggingface.co/thousandml/Qwen3-Coder-Next-CQ-4bit) |

[All models →](https://huggingface.co/thousandml)

To quantize your own model, run two scripts from the repo root:

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

`./models/MODEL-CQ-4bit` is then ready for `--model` in training and for `vllm serve --quantization cq`. Without `--no-upload`, step 2 also uploads the checkpoint to `--repo-id` on Hugging Face, which needs `hf auth login` first. If step 2 fails after the shards are built, for example during the upload, rerun it with `--skip-convert` so it doesn't rebuild them.

## Limitations

- **Parallelism:**

  | | 1 GPU | Pipeline parallel | Tensor parallel |
  |---|:-:|:-:|:-:|
  | Training (any base) | ✓ | – | – |
  | vLLM serving, CQ base + GCMP or LoRA | ✓ | ✓ | – |
  | vLLM serving, full-precision base + GCMP | not yet | not yet | not yet |
  | Full-precision base + GCMP in Transformers | ✓ | – | – |

  CQ MoE models use TP=1 with expert parallelism.
- **What gets adapted:** MoE expert weights and non-CQ dense layers are not adapted.
- **Benchmark scope:** results so far are for one model (Qwen3.8-27B) and one dataset (Alpaca), using the quick benchmark subset, which detects differences of about 5 points rather than 1. DROP is left out because its scores were invalid (about 0.02 for every model).
- **Your workload:** measure quality, speed and memory on your own data before relying on these numbers.

## Troubleshooting

```bash
python -c "import torch, cq.cq_kernels; print('CUDA:', torch.cuda.is_available())"
```

- **Kernel import:** the module is `cq.cq_kernels`, not `cq_kernels`.
- **Native binaries:** they must match your Python (3.12), CUDA and PyTorch versions.
- **License:** set `CQ_LICENSE_KEY` / `CQ_LICENSE_URL` if your distribution needs a license.
- **Windows line endings:** if you get a `bash\r` error, run `sed -i 's/\r$//' install.sh`.

## Citation

If you use Training Factory, CQ or GCMP in your work, please cite:

```bibtex
@software{elfadil2026trainingfactory,
  author       = {Elfadil, Husam},
  title        = {Training Factory: Fine-Tuning on CQ 4-bit Models with GCMP Adapters},
  organization = {ThousandML},
  year         = {2026},
  url          = {https://github.com/thousandml/training-factory}
}
```

## Contributing

Bug reports, hardware measurements, model-compatibility reports and independent benchmark runs are welcome. Open an [issue](https://github.com/thousandml/training-factory/issues).

**ThousandML: Making Every Bit Count.**
