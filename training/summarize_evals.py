#!/usr/bin/env python3
"""Collect eval/<case>/ results into one table (eval/summary.md + summary.json).

  python summarize_evals.py /workspace/training-factory-gcmp/eval
"""
import json
import re
import sys
from pathlib import Path

ORDER = ["cq_base", "cq_lora", "cq_gcmp", "nf4_base", "nf4_lora", "nf4_gcmp", "bf16_base", "bf16_lora", "bf16_gcmp"]


def finished_fraction(pred_path):
    total = done = 0
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            p = json.loads(line)["prediction"]
            done += "</think>" in p or not re.match(r"^\s*(<think>|we need|the user|let me)", p, re.I)
    return done / total if total else None


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "eval")
    cases = sorted((d for d in root.iterdir() if (d / "metrics.json").exists()),
                   key=lambda d: (ORDER.index(d.name) if d.name in ORDER else 99, d.name))
    rows = []
    for d in cases:
        m = json.loads((d / "metrics.json").read_text())
        key = (d / "answer_key.txt").read_text() if (d / "answer_key.txt").exists() else ""
        acc = re.search(r"correct (\d+)/(\d+)", key)
        rows.append({
            "case": d.name,
            "adapter": m.get("adapter") or "none",
            "examples": m["examples"],
            "thinking": m.get("enable_thinking", "unrecorded"),
            "rouge1": m["rouge1_f1"], "rouge2": m["rouge2_f1"], "rougeL": m["rougeL_f1"],
            "bleu": m["bleu"],
            "answer_key": f"{acc.group(1)}/{acc.group(2)}" if acc else "-",
            "answered": finished_fraction(d / "predictions.jsonl") if (d / "predictions.jsonl").exists() else None,
        })
    if not rows:
        print(f"no eval/<case>/metrics.json under {root}")
        return
    header = "| case | examples | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | answer key | answered | adapter |"
    lines = [header, "|" + "---|" * 9]
    for r in rows:
        ans = f"{r['answered']:.0%}" if r["answered"] is not None else "-"
        lines.append(f"| {r['case']} | {r['examples']} | {r['rouge1']:.4f} | {r['rouge2']:.4f} | "
                     f"{r['rougeL']:.4f} | {r['bleu']:.2f} | {r['answer_key']} | {ans} | {r['adapter']} |")
    table = "\n".join(lines)
    print(table)
    if len({r["thinking"] for r in rows}) > 1 or len({r["examples"] for r in rows}) > 1:
        print("\n!! cases differ in thinking mode or example count; they are not directly comparable")
    (root / "summary.md").write_text(table + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
