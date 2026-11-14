# Downstream evaluation: CQ / NF4 / BF16 x base / LoRA / GCMP

## Why

Training-time eval losses come from two trainers (the CQ trainer and
Unsloth's SFTTrainer) that tokenize, mask and average differently, so their
numbers are not on one scale (for example 0.84 vs 1.25). This suite measures
every case with the same code and adds benchmarks scored right or wrong.

## Cases

| machine | cases |
|---|---|
| 40 GB (4-bit) | cq_base, cq_lora, cq_gcmp, nf4_base, nf4_lora, nf4_gcmp |
| 80 GB (16-bit) | bf16_base, bf16_lora, bf16_gcmp |

## What is measured

| stage | script | output |
|---|---|---|
| loss | uniform_loss_eval.py | same-formula validation loss: all tokens of the training rows, and answer tokens only |
| gen | eval_rouge_bleu.py + score_answer_key.py | ROUGE/BLEU on all validation rows (thinking off) + 90-item short-answer accuracy |
| tasks | downstream_eval.py (lm-evaluation-harness) | IFEval, GSM8K, SQuAD extraction, DROP, ARC-Challenge, HellaSwag, WinoGrande, BoolQ, MMLU, TruthfulQA (+ optional SWDE / FDA document extraction) |

All decoding is greedy, few-shot examples use fixed seeds, and limited runs
take the first N questions of each task, so every case answers the same
questions.

## Run

```bash
bash training/setup_eval_machine.sh            # once per machine
bash training/run_downstream.sh 4bit           # 40 GB machine
bash training/run_downstream.sh bf16           # 80 GB machine
SUITE=full bash training/run_downstream.sh 4bit   # every question (much longer)
TASKS=swde,fda bash training/run_downstream.sh 4bit   # optional document extraction
```

Copy both `eval_downstream/` folders to one place, then:

```bash
python training/compare_cases.py eval_4bit eval_bf16 --output compare
```

`compare/compare.md` has one table per case and paired comparisons
(GCMP vs LoRA per base, adapter vs base, and each adapter across CQ/NF4/BF16),
each with a 95% paired-bootstrap interval and, for right/wrong metrics, an
exact McNemar test.
