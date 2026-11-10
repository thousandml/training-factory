#!/usr/bin/env bash
# Unpack the result archives from every machine and build one report.
#
#   bash training/report_all.sh results_gpu40_*.tar.gz results_gpu80_*.tar.gz
#
# Writes report/ with:
#   compare.md / compare.json   downstream suite: uniform loss, ROUGE/BLEU, answer key,
#                               benchmarks, paired GCMP-vs-LoRA / adapter-vs-base / CQ-NF4-BF16
#   evals_<machine>.md          the earlier run_evals.sh ROUGE/BLEU tables
#   training.md                 every training run: eval loss by step, step time, memory
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPORT=${REPORT:-$PWD/report}
mkdir -p "$REPORT/raw"

for archive in "$@"; do
  label=$(basename "$archive" .tar.gz)
  mkdir -p "$REPORT/raw/$label"
  tar -xzf "$archive" -C "$REPORT/raw/$label"
  echo "unpacked $archive -> $REPORT/raw/$label"
done

# 1. Downstream suite (run_downstream.sh), all machines together
mapfile -t DS < <(find "$REPORT/raw" -type d -name eval_downstream | sort)
if [[ ${#DS[@]} -gt 0 ]]; then
  python "$HERE/compare_cases.py" "${DS[@]}" --output "$REPORT" > /dev/null \
    && echo "downstream comparison -> $REPORT/compare.md"
fi

# 2. ROUGE/BLEU runs (run_evals.sh), one table per machine
for d in $(find "$REPORT/raw" -type d -name eval | sort); do
  label=$(echo "$d" | sed "s|$REPORT/raw/||; s|/.*||")
  python "$HERE/summarize_evals.py" "$d" > /dev/null && cp "$d/summary.md" "$REPORT/evals_$label.md" \
    && echo "ROUGE/BLEU table -> $REPORT/evals_$label.md"
done

# 3. Training runs (CQ, NF4 and BF16 x LoRA / GCMP)
mapfile -t RUNS < <(find "$REPORT/raw" -name comparison_logs.jsonl -printf '%h\n' | sort -u)
if [[ ${#RUNS[@]} -gt 0 ]]; then
  python "$HERE/summarize_training.py" "${RUNS[@]}" --output "$REPORT/training.md" > /dev/null \
    && echo "training runs -> $REPORT/training.md"
fi

echo
echo "Done. Start with $REPORT/compare.md, then training.md."
