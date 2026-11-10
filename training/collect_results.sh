#!/usr/bin/env bash
# Pack every result file on this machine into one small archive (no model
# weights, adapters or checkpoints), ready to copy to one place.
#
#   bash training/collect_results.sh            # -> results_<hostname>_<date>.tar.gz
#   NAME=gpu80 bash training/collect_results.sh  # choose the archive label
set -uo pipefail

NAME=${NAME:-$(hostname)}
STAMP=$(date +%Y%m%d-%H%M)
OUT=${OUT:-$PWD/results_${NAME}_${STAMP}.tar.gz}
ROOTS=(${ROOTS:-/workspace/training-factory-gcmp /workspace/training-factory})

LIST=$(mktemp)
for root in "${ROOTS[@]}"; do
  [[ -d $root ]] || continue
  # downstream suite and ROUGE/BLEU runs
  for d in "$root/eval_downstream" "$root/eval"; do
    [[ -d $d ]] && find "$d" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \
      -o -name '*.txt' -o -name '*.log' \) >> "$LIST"
  done
  # training runs: logs and benchmarks only, no checkpoints or weights
  [[ -d $root/outputs ]] && find "$root/outputs" -maxdepth 2 -type f \( \
      -name 'comparison_*.json' -o -name 'comparison_*.jsonl' -o -name 'benchmark.json' \
      -o -name 'train.log' -o -name '*results.json' -o -name 'cq_training.json' \
      -o -name 'trainer_state.json' -o -name 'vllm_serve*.txt' \) -not -path '*/checkpoint-*' >> "$LIST"
done
sort -u "$LIST" -o "$LIST"
echo "packing $(wc -l < "$LIST") files ($(xargs -a "$LIST" -d '\n' du -ch 2>/dev/null | tail -1 | cut -f1))"
tar -czf "$OUT" -T "$LIST" 2>/dev/null && echo "wrote $OUT"
rm -f "$LIST"
