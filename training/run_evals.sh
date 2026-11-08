#!/usr/bin/env bash
# ROUGE/BLEU + answer-key evaluation for the four adapter cases.
#
#   bash run_evals.sh                 # all four cases
#   bash run_evals.sh cq_gcmp         # one case
#   bash run_evals.sh cq_base nf4_base  # optional no-adapter baselines
#
# Each case writes only into its own folder:
#   eval/<case>/metrics.json        ROUGE/BLEU + run settings
#   eval/<case>/predictions.jsonl   one line per validation row
#   eval/<case>/run.log             full console output
#   eval/<case>/answer_key.txt      short-answer accuracy (90 items)
# Interrupted runs continue where they stopped; the evaluator refuses to
# resume a predictions file written with different settings or adapter.
# Stop any vLLM server first: the evaluator needs the GPU.
set -euo pipefail

# ---------------------------------------------------------------- settings
ROOT=/workspace/training-factory-gcmp
TRAINING=$ROOT/training
VAL_TEXT=$TRAINING/examples/alpaca_val.jsonl  # training-format validation rows ({"text": ...})
DATA=$ROOT/eval/alpaca_val_eval.jsonl         # prompt/reference form, created from VAL_TEXT if missing
KEY=$ROOT/qa_extraction_indices.jsonl      # answer key (90 short-answer items)
EVAL_DIR=$ROOT/eval

CQ_MODEL=/workspace/models/Qwen3.8-27B-CQ-4bit
NF4_MODEL=unsloth/Qwen3.8-27B-unsloth-bnb-4bit

CQ_GCMP=$ROOT/outputs/qwen38-27b-cq-4bit-alpaca-gcmp-v1
NF4_GCMP=$ROOT/outputs/qwen38-27b-unsloth-nf4-alpaca-gcmp-v1/final_adapter
CQ_LORA=/workspace/training-factory/outputs/qwen38-27b-cq-r32-comparison-v4
NF4_LORA=/workspace/training-factory/outputs/qwen38-27b-unsloth-nf4-r32-v5/final_adapter

LIMIT=0              # 0 = full validation set (required for the answer key)
MAX_NEW_TOKENS=256
SEED=42
# --------------------------------------------------------------------------

run_case() {
  local name=$1 backend=$2 model=$3 adapter=${4:-}
  local out=$EVAL_DIR/$name
  mkdir -p "$out"
  if [[ -n $adapter && ! -f $adapter/adapter_config.json ]]; then
    echo "!! $name: no adapter_config.json in $adapter (set the path at the top of this script)" >&2
    return 1
  fi
  local extra=()
  [[ $backend == cq ]] && extra+=(--kernel-module cq.cq_kernels)
  [[ -n $adapter ]] && extra+=(--adapter "$adapter")
  echo "== $name  backend=$backend  adapter=${adapter:-none}"
  ( cd "$ROOT" && python "$TRAINING/eval_rouge_bleu.py" \
      --backend "$backend" --model "$model" "${extra[@]}" \
      --data "$DATA" --limit "$LIMIT" --seed "$SEED" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$out/metrics.json" --predictions "$out/predictions.jsonl" \
      --resume ) 2>&1 | tee -a "$out/run.log"
  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "!! $name: evaluator exited with $status (see $out/run.log)" >&2
    return 1
  fi
  if [[ -f $KEY && $LIMIT == 0 ]]; then
    python "$TRAINING/score_answer_key.py" "$KEY" "$out/predictions.jsonl" | tee "$out/answer_key.txt"
  fi
}

declare -A CASES=(
  [cq_gcmp]="cq $CQ_MODEL $CQ_GCMP"
  [cq_lora]="cq $CQ_MODEL $CQ_LORA"
  [nf4_gcmp]="unsloth $NF4_MODEL $NF4_GCMP"
  [nf4_lora]="unsloth $NF4_MODEL $NF4_LORA"
  [cq_base]="cq $CQ_MODEL"
  [nf4_base]="unsloth $NF4_MODEL"
)
ORDER=(cq_gcmp cq_lora nf4_gcmp nf4_lora)
[[ $# -gt 0 ]] && ORDER=("$@")

if [[ ! -f $DATA ]]; then
  mkdir -p "$(dirname "$DATA")"
  python "$TRAINING/prepare_eval_data.py" "$VAL_TEXT" "$DATA"
fi

failed=()
for name in "${ORDER[@]}"; do
  [[ -n ${CASES[$name]:-} ]] || { echo "unknown case $name (choose: ${!CASES[*]})" >&2; exit 2; }
  # shellcheck disable=SC2086
  run_case "$name" ${CASES[$name]} || failed+=("$name")
done

python "$TRAINING/summarize_evals.py" "$EVAL_DIR"
[[ ${#failed[@]} -eq 0 ]] || { echo "!! failed: ${failed[*]}" >&2; exit 1; }
