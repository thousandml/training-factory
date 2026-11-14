#!/usr/bin/env bash
# Unsloth 16-bit (bf16) base: LoRA r=32 vs GCMP, same settings as the NF4 runs.
#
#   bash run_unsloth_16bit.sh            # both
#   bash run_unsloth_16bit.sh gcmp       # one
#
# Needs a GPU with >= 80 GB: the 27B base alone is ~54 GB in bf16.
set -euo pipefail

ROOT=${ROOT:-/workspace/training-factory-gcmp}
cd "$ROOT"

MODEL=${MODEL:-Qwen/Qwen3.8-27B}     # original 16-bit checkpoint

# Hyperparameters shared with the NF4 runs (data, steps, batch 2 x grad-accum 4,
# LR 2e-5 linear, no warmup, no weight decay, seed). Every Unsloth optimization
# is on for both adapters: 16-bit FastModel loading with Unsloth's patched
# kernels and compiled decoder blocks, padding-free batching, smart gradient
# checkpointing with activation offload, fused AdamW (same math as plain
# AdamW, faster). LoRA also gets Unsloth's fused LoRA kernels; GCMP uses the
# native GCMP operators instead.
#
# PACKING=1 also packs short Alpaca rows into full 512-token sequences. It is
# much faster but changes tokens per step and loss averaging, so results are
# no longer step-for-step comparable with the NF4 runs. Use it for both or
# neither.
PACKING=${PACKING:-0}
COMMON=(
  --model "$MODEL" --dtype bfloat16
  --train-data training/examples/alpaca_train.jsonl
  --eval-data training/examples/alpaca_val.jsonl
  --max-length 512 --max-steps 500
  --batch-size 2 --grad-accum 4
  --learning-rate 2e-5 --warmup-steps 0 --weight-decay 0.0
  --optim adamw_torch_fused
  --seed 42 --eval-steps 50 --save-steps 100
  --benchmark-warmup-steps 10
)

[[ $PACKING == 1 ]] && COMMON+=(--packing)

run() {
  local which=$1 out=$2; shift 2
  if [[ -e $out/comparison_summary.json ]]; then
    echo "!! $out already holds a finished run; choose a new output folder" >&2
    return 1
  fi
  mkdir -p "$out"
  echo "== $which -> $out"
  python "training/train_unsloth_16bit_$which.py" "${COMMON[@]}" --output "$out" "$@" 2>&1 | tee "$out/train.log"
}

CASES=("${@:-lora gcmp}")
for c in ${CASES[*]}; do
  case $c in
    lora) run lora ./outputs/qwen38-27b-unsloth-bf16-r32-v1 \
            --lora-rank 32 --lora-alpha 32 --lora-dropout 0.0 ;;
    gcmp) run gcmp ./outputs/qwen38-27b-unsloth-bf16-alpaca-gcmp-v1 \
            --gcmp-scale 1.0 --kernel-module cq.cq_kernels ;;
    *) echo "unknown case $c (lora | gcmp)" >&2; exit 2 ;;
  esac
done
