#!/usr/bin/env python3
"""Score predictions JSONL files against the short-answer key.

Only the final answer is scored: text after the last </think> if the model
reasoned, else the whole prediction. A prediction with an unclosed
reasoning block counts as unanswered.

  python score_answer_key.py qa_extraction_indices.jsonl run_a_predictions.jsonl run_b_predictions.jsonl

Use predictions from an eval run over the full validation set (--limit 0);
with a --limit subset the indices refer to the sample, not the dataset rows.
"""
import json
import re
import sys


def norm(s):
    return re.sub(r"\s+", " ", s.lower().replace("*", "")).strip()


def contains(pred, answers):
    p = norm(pred)
    for a in map(norm, answers):
        if re.fullmatch(r"[\w.%$\-^/]+", a):
            if re.search(r"(?<![\w.])" + re.escape(a) + r"(?![\w])", p):
                return True
        elif a in p:
            return True
    return False


def final_answer(pred):
    if "</think>" in pred:
        return pred.split("</think>")[-1].strip()
    if re.match(r"^\s*(<think>|we need|the user|let me)", pred, re.I):
        return None  # reasoning never finished
    return pred.strip()


def main():
    key = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    for path in sys.argv[2:]:
        rows = {r["index"]: r for r in map(json.loads, open(path))}
        items = [k for k in key if k["index"] in rows]
        unanswered = correct = 0
        for k in items:
            ans = final_answer(rows[k["index"]]["prediction"])
            if ans is None:
                unanswered += 1
            elif contains(ans, k["answers"]):
                correct += 1
        print(f"{path}: correct {correct}/{len(items)} ({correct / max(len(items), 1):.1%}), "
              f"unanswered {unanswered}")


if __name__ == "__main__":
    main()
