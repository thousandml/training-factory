#!/usr/bin/env bash
# One-time setup of a fresh evaluation machine (40 GB for 4-bit, 80 GB for BF16).
# Run from the folder that contains the training/ directory of this repo.
#
# Before running, copy onto the machine:
#   training/                      this repo's training folder (v14 or later)
#   cq_quant-*.whl                 your rebuilt CQ wheel (v14: graph-safe GCMP ops)
#   the CQ license setup you use on the training machine
#   /workspace/models/Qwen3.8-27B-CQ-4bit         (4-bit machine only)
#   /workspace/adapters/<adapter folders>         (see run_downstream.sh)
#   qa_extraction_indices.jsonl    (optional 90-item answer key)
# NF4 and BF16 base weights download from Hugging Face on first use.
set -euo pipefail

pip install --upgrade pip
pip install "lm-eval[ifeval]>=0.4.13" rouge-score sacrebleu evaluate langdetect immutabledict nltk
pip install unsloth                       # NF4 / BF16 cases
WHEEL=$(ls cq_quant-*.whl 2>/dev/null | head -1 || true)
if [[ -n $WHEEL ]]; then pip install --force-reinstall "$WHEEL"; else echo "!! no cq_quant wheel here: CQ cases and native GCMP ops need it"; fi
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "--- checks"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), 'GiB')"
python -c "import lm_eval; print('lm-eval', lm_eval.__version__)"
python -c "import cq.cq_kernels, torch; print('cq_kernels OK; GCMP ops:', hasattr(torch.ops.my_qlinear, 'gcmp_delta'))" || \
  echo "!! cq_kernels not importable (CQ cases and native GCMP ops will fail)"
python - <<'EOF'
from lm_eval.tasks import TaskManager
tm = TaskManager()
need = ['ifeval', 'gsm8k', 'squad_completion', 'drop', 'arc_challenge', 'hellaswag',
        'winogrande', 'boolq', 'mmlu', 'truthfulqa_mc2', 'swde', 'fda']
missing = [t for t in need if t not in tm.all_tasks]
print('lm-eval tasks:', 'all present' if not missing else f'missing {missing}')
EOF
