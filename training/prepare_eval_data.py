#!/usr/bin/env python3
"""Turn {"text": ...} training rows into instruction/input/output rows for eval.

eval_rouge_bleu.py needs the prompt and the reference answer separately.
Understands the two usual text layouts:
  * Alpaca prompt: "### Instruction: ... [### Input: ...] ### Response: ..."
  * ChatML (Qwen chat template): "<|im_start|>user ... <|im_start|>assistant ..."
Rows already in instruction/output or messages form are passed through.
Row order is kept, so answer-key row numbers stay valid.

  python prepare_eval_data.py alpaca_val.jsonl alpaca_val_eval.jsonl
"""
import json
import re
import sys

ALPACA = re.compile(
    r"###\s*Instruction:?\s*(?P<instruction>.*?)"
    r"(?:###\s*Input:?\s*(?P<input>.*?))?"
    r"###\s*Response:?\s*(?P<output>.*)\Z", re.S)
CHATML_TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|\Z)", re.S)
THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_eos(s):
    return re.sub(r"(<\|im_end\|>|<\|endoftext\|>|</s>|<eos>)\s*\Z", "", s).strip()


def convert(row, n):
    if ("instruction" in row and "output" in row) or isinstance(row.get("messages"), list):
        return row
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError(f"line {n}: no text/instruction/messages field")
    if "<|im_start|>" in text:
        turns = [(r, THINK.sub("", c).strip()) for r, c in CHATML_TURN.findall(text)]
        turns = [(r, c) for r, c in turns if r in ("system", "user", "assistant")]
        if not turns or turns[-1][0] != "assistant":
            raise ValueError(f"line {n}: ChatML text has no final assistant turn")
        return {"messages": [{"role": r, "content": c} for r, c in turns]}
    m = ALPACA.search(text)
    if m:
        return {"instruction": m["instruction"].strip(), "input": (m["input"] or "").strip(),
                "output": strip_eos(m["output"])}
    raise ValueError(f"line {n}: unrecognised text layout; first 300 chars:\n{text[:300]!r}")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    count = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for n, line in enumerate(f, 1):
            if line.strip():
                out.write(json.dumps(convert(json.loads(line), n), ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} rows to {dst}")


if __name__ == "__main__":
    main()
