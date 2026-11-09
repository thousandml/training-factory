#!/usr/bin/env bash
# Full downstream evaluation for every base/adapter combination.
#
#   bash run_downstream.sh 4bit                 # 40 GB machine: CQ + NF4, base/LoRA/GCMP
#   bash run_downstream.sh bf16                 # 80 GB machine: BF16, base/LoRA/GCMP
#   bash run_downstream.sh 4bit cq_gcmp nf4_lora   # selected cases only
#   STAGES="loss tasks" SUITE=full bash run_downstream.sh 4bit
#
# Stages per case (each skips work already done; interrupted runs continue):
#   loss   uniform_loss_eval.py   same-formula validation loss  -> loss.json
#   gen    eval_rouge_bleu.py     ROUGE/BLEU + 90-item answer key -> metrics.json, answer_key.txt
#   tasks  downstream_eval.py     lm-eval benchmarks            -> tasks/<task>.json
# Everything for a case lives in $EVAL_DIR/<case>/. Afterwards, on any machine:
#   python compare_cases.py <eval dir 4bit> <eval dir bf16> --output <folder>
set -uo pipefail

# ------------------------------------------------------------------ settings
ROOT=${ROOT:-/workspace/training-factory-gcmp}
TRAINING=$ROOT/training
EVAL_DIR=${EVAL_DIR:-$ROOT/eval_downstream}
VAL_TEXT=$TRAINING/examples/alpaca_val.jsonl     # training-format validation rows
VAL_CHAT=$EVAL_DIR/alpaca_val_eval.jsonl         # prompt/reference form (auto-created)
KEY=$ROOT/qa_extraction_indices.jsonl            # 90-item answer key (optional)

CQ_MODEL=/workspace/models/Qwen3.8-27B-CQ-4bit
NF4_MODEL=unsloth/Qwen3.8-27B-unsloth-bnb-4bit
BF16_MODEL=Qwen/Qwen3.8-27B

ADAPTERS=/workspace/adapters                     # copy the adapter folders here
CQ_LORA=$ADAPTERS/qwen38-27b-cq-r32-comparison-v4
CQ_GCMP=$ADAPTERS/qwen38-27b-cq-4bit-alpaca-gcmp-v1
NF4_LORA=$ADAPTERS/qwen38-27b-unsloth-nf4-r32-v5/final_adapter
NF4_GCMP=$ADAPTERS/qwen38-27b-unsloth-nf4-alpaca-gcmp-v1/final_adapter
BF16_LORA=$ADAPTERS/qwen38-27b-unsloth-bf16-r32-v1/final_adapter
BF16_GCMP=$ADAPTERS/qwen38-27b-unsloth-bf16-alpaca-gcmp-v1/final_adapter

STAGES=${STAGES:-"loss gen tasks"}
SUITE=${SUITE:-quick}            # quick | full
TASKS=${TASKS:-}                 # optional comma list, e.g. gsm8k,ifeval,swde
BATCH=${BATCH:-4}                # lm-eval batch size; lower to 1 on out-of-memory
GEN_TOKENS=${GEN_TOKENS:-256}
# ---------------------------------------------------------------------------

declare -A CASE=(
  [cq_base]="cq|$CQ_MODEL|"           [cq_lora]="cq|$CQ_MODEL|$CQ_LORA"
  [cq_gcmp]="cq|$CQ_MODEL|$CQ_GCMP"
  [nf4_base]="nf4|$NF4_MODEL|"        [nf4_lora]="nf4|$NF4_MODEL|$NF4_LORA"
  [nf4_gcmp]="nf4|$NF4_MODEL|$NF4_GCMP"
  [bf16_base]="bf16|$BF16_MODEL|"     [bf16_lora]="bf16|$BF16_MODEL|$BF16_LORA"
  [bf16_gcmp]="bf16|$BF16_MODEL|$BF16_GCMP"
)
case ${1:-} in
  4bit) ORDER=(cq_base cq_lora cq_gcmp nf4_base nf4_lora nf4_gcmp) ;;
  bf16) ORDER=(bf16_base bf16_lora bf16_gcmp) ;;
  *) echo "usage: $0 4bit|bf16 [case ...]" >&2; exit 2 ;;
esac
shift
[[ $# -gt 0 ]] && ORDER=("$@")

mkdir -p "$EVAL_DIR"
if [[ ! -f $VAL_CHAT ]]; then
  python "$TRAINING/prepare_eval_data.py" "$VAL_TEXT" "$VAL_CHAT" || exit 1
fi

failed=()
for name in "${ORDER[@]}"; do
  spec=${CASE[$name]:-}
  [[ -n $spec ]] || { echo "unknown case $name" >&2; exit 2; }
  IFS='|' read -r kind model adapter <<< "$spec"
  out=$EVAL_DIR/$name
  mkdir -p "$out"
  log=$out/run.log
  if [[ -n $adapter && ! -f $adapter/adapter_config.json ]]; then
    echo "!! $name: no adapter_config.json in $adapter" | tee -a "$log" >&2
    failed+=("$name"); continue
  fi
  common=(--model "$model")
  [[ -n $adapter ]] && common+=(--adapter "$adapter")
  case $kind in
    cq)   common+=(--backend cq --kernel-module cq.cq_kernels) ;;
    nf4)  common+=(--backend unsloth --kernel-module cq.cq_kernels) ;;
    bf16) common+=(--backend unsloth --unsloth-16bit --kernel-module cq.cq_kernels) ;;
  esac
  echo "==================== $name ($kind, adapter=${adapter:-none})" | tee -a "$log"
  ok=1
  for stage in $STAGES; do
    case $stage in
      loss)
        [[ -f $out/loss.json ]] && { echo "-- loss: done" | tee -a "$log"; continue; }
        python "$TRAINING/uniform_loss_eval.py" "${common[@]}" \
          --text-data "$VAL_TEXT" --chat-data "$VAL_CHAT" --output "$out/loss.json" 2>&1 | tee -a "$log"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || ok=0 ;;
      gen)
        if [[ ! -f $out/metrics.json ]]; then
          python "$TRAINING/eval_rouge_bleu.py" "${common[@]}" --data "$VAL_CHAT" --limit 0 \
            --max-new-tokens "$GEN_TOKENS" --output "$out/metrics.json" \
            --predictions "$out/predictions.jsonl" --resume 2>&1 | tee -a "$log"
          [[ ${PIPESTATUS[0]} -eq 0 ]] || { ok=0; continue; }
        fi
        if [[ -f $KEY ]]; then
          python "$TRAINING/score_answer_key.py" "$KEY" "$out/predictions.jsonl" | tee "$out/answer_key.txt"
        fi ;;
      tasks)
        extra=()
        [[ -n $TASKS ]] && extra+=(--tasks "$TASKS")
        python "$TRAINING/downstream_eval.py" "${common[@]}" --output-dir "$out/tasks" \
          --suite "$SUITE" --batch-size "$BATCH" "${extra[@]}" 2>&1 | tee -a "$log"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || ok=0 ;;
      *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
  done
  [[ $ok -eq 1 ]] || failed+=("$name")
done

python "$TRAINING/compare_cases.py" "$EVAL_DIR" --output "$EVAL_DIR" > /dev/null && \
  echo "summary: $EVAL_DIR/compare.md"
[[ ${#failed[@]} -eq 0 ]] || { echo "!! failed: ${failed[*]} (see <case>/run.log)" >&2; exit 1; }
