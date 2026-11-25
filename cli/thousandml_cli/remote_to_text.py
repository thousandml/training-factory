"""Unsloth's NF4/BF16 trainers read {"text": ...} rows. Render chat rows
({"messages": [...]}) with the model's own chat template; pass text rows through.

  python remote_to_text.py SRC.jsonl DST.jsonl MODEL_DIR
"""
import json
import sys

src, dst, model = sys.argv[1:4]
with open(src, encoding='utf-8') as f:
    rows = [json.loads(line) for line in f if line.strip()]

tok = None
with open(dst, 'w', encoding='utf-8') as out:
    for n, row in enumerate(rows, 1):
        if isinstance(row.get('text'), str):
            text = row['text']
        elif isinstance(row.get('messages'), list):
            if tok is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model)
            text = tok.apply_chat_template(row['messages'], tokenize=False)
        else:
            sys.exit(f'{src}:{n}: each row needs "text" or "messages"')
        out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
print(f'{len(rows)} rows -> {dst}')
