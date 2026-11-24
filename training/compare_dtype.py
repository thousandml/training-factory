#!/usr/bin/env python3
"""Compare the same adapters evaluated twice: fp32 adapter files vs bf16 adapter files.

    python training/compare_dtype.py eval_downstream eval_bf16store
    python training/compare_dtype.py eval_downstream eval_bf16store --cases cq_gcmp nf4_gcmp

For every case folder found in both roots: held-out loss (paired bootstrap CI),
each benchmark (paired McNemar for right/wrong tasks, bootstrap CI otherwise), and
how many generated answers are identical. Writes <bf16 root>/dtype_compare.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_cases import PRIMARY, TASK_ORDER, bootstrap_diff, fmt, load_case, loss_pair, mcnemar  # noqa: E402


def predictions(d):
    f = d / 'predictions.jsonl'
    if not f.exists():
        return {}
    out = {}
    for line in open(f, encoding='utf-8'):
        r = json.loads(line)
        out[r.get('index')] = r.get('prediction')
    return out


def compare(fp, bf):
    a, b = load_case(fp), load_case(bf)
    lines = [f'## {fp.name}', '', f'fp32: `{fp}`  ', f'bf16: `{bf}`', '']
    lines += ['| Metric | fp32 | bf16 | bf16 − fp32 | test |', '|---|---:|---:|---:|---|']
    for which, label in (('response', 'Answer loss'), ('text', 'Text loss')):
        lp = loss_pair(b, a, which)
        if lp:
            lines.append(f"| {label} | {lp['b']:.4f} | {lp['a']:.4f} | {lp['diff']:+.4f} | "
                         f"95% CI [{lp['ci95'][0]:+.4f}, {lp['ci95'][1]:+.4f}] |")
    for task in TASK_ORDER:
        if task not in a['samples'] or task not in b['samples']:
            continue
        keys = sorted(set(a['samples'][task]) & set(b['samples'][task]))
        if not keys:
            continue
        va = [a['samples'][task][k] for k in keys]
        vb = [b['samples'][task][k] for k in keys]
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        if PRIMARY[task][2]:
            only_b, only_a, p = mcnemar(vb, va)
            test = f'McNemar p = {p:.3g} (only bf16 right: {only_b}, only fp32 right: {only_a}, n = {len(keys)})'
        else:
            lo, hi = bootstrap_diff(vb, va)
            test = f'95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}] (n = {len(keys)})'
        lines.append(f'| {task} | {ma * 100:.1f} | {mb * 100:.1f} | {(mb - ma) * 100:+.1f} | {test} |')
    pa, pb = predictions(fp), predictions(bf)
    common = set(pa) & set(pb)
    if common:
        same = sum(1 for k in common if pa[k] == pb[k])
        lines.append(f'| Identical generated answers | – | – | {same}/{len(common)} ({same / len(common):.1%}) | greedy decoding |')
    return lines + ['']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fp32_root')
    ap.add_argument('bf16_root')
    ap.add_argument('--cases', nargs='*', help='default: every case folder present in both roots')
    a = ap.parse_args()
    fr, br = Path(a.fp32_root), Path(a.bf16_root)
    cases = a.cases or sorted(d.name for d in br.iterdir() if d.is_dir() and (fr / d.name).is_dir())
    if not cases:
        sys.exit('no case folders found in both roots')
    out = ['# GCMP adapters: fp32 files vs bf16 files', '',
           'Same base, data, prompts and decoding; only the stored adapter dtype differs.', '']
    for c in cases:
        out += compare(fr / c, br / c)
    text = '\n'.join(out) + '\n'
    (br / 'dtype_compare.md').write_text(text, encoding='utf-8')
    print(text)
    print(f'-> {br / "dtype_compare.md"}')


if __name__ == '__main__':
    main()
