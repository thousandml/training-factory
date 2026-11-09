#!/usr/bin/env python3
"""Summarise and compare evaluation cases, question by question.

Reads case folders (eval/<case>/) written by run_downstream.sh from one or
more eval roots (e.g. the 40 GB and the 80 GB machine), then writes
compare.md and compare.json with:

  * one table of every metric per case (uniform loss, ROUGE/BLEU, answer key,
    each benchmark's accuracy)
  * paired comparisons for the questions people ask: GCMP vs LoRA on the same
    base, adapter vs no adapter, and the same adapter across CQ / NF4 / BF16.
    Each comparison uses only questions both cases answered, and reports the
    difference with a 95% paired-bootstrap interval; for right/wrong metrics
    also the exact McNemar test (how many questions only one side got right).

  python compare_cases.py /data/eval_4bit /data/eval_bf16 --output /data/compare
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

PRIMARY = {  # task: (metric, filter or None, binary)
    'ifeval': ('prompt_level_strict_acc', None, True),
    'gsm8k': ('exact_match', 'flexible-extract', True),
    'squad_completion': ('contains', None, True),
    'drop': ('f1', None, False),
    'arc_challenge': ('acc_norm', None, True),
    'hellaswag': ('acc_norm', None, True),
    'winogrande': ('acc', None, True),
    'boolq': ('acc', None, True),
    'mmlu': ('acc', None, True),
    'truthfulqa_mc2': ('acc', None, False),
    'swde': ('contains', None, True),
    'fda': ('contains', None, True),
}
TASK_ORDER = list(PRIMARY)
CASE_ORDER = ['cq_base', 'cq_lora', 'cq_gcmp', 'nf4_base', 'nf4_lora', 'nf4_gcmp',
              'bf16_base', 'bf16_lora', 'bf16_gcmp']
PAIRS = [
    ('cq_gcmp', 'cq_lora'), ('nf4_gcmp', 'nf4_lora'), ('bf16_gcmp', 'bf16_lora'),
    ('cq_lora', 'cq_base'), ('cq_gcmp', 'cq_base'), ('nf4_lora', 'nf4_base'),
    ('nf4_gcmp', 'nf4_base'), ('bf16_lora', 'bf16_base'), ('bf16_gcmp', 'bf16_base'),
    ('cq_lora', 'nf4_lora'), ('cq_gcmp', 'nf4_gcmp'), ('cq_base', 'nf4_base'),
    ('cq_lora', 'bf16_lora'), ('cq_gcmp', 'bf16_gcmp'), ('cq_base', 'bf16_base'),
    ('nf4_lora', 'bf16_lora'), ('nf4_gcmp', 'bf16_gcmp'), ('nf4_base', 'bf16_base'),
]


def load_case(d):
    case = {'name': d.name, 'dir': str(d), 'tasks': {}, 'samples': {}, 'loss_rows': {}}
    if (d / 'loss.json').exists():
        case['loss'] = json.loads((d / 'loss.json').read_text())
        rows = d / 'loss_rows.jsonl'
        if rows.exists():
            for line in open(rows):
                r = json.loads(line)
                case['loss_rows'][(r['set'], r['row'])] = (r['nll'], r['tokens'])
    if (d / 'metrics.json').exists():
        case['gen'] = json.loads((d / 'metrics.json').read_text())
    if (d / 'answer_key.txt').exists():
        m = re.search(r'correct (\d+)/(\d+)', (d / 'answer_key.txt').read_text())
        if m:
            case['answer_key'] = (int(m.group(1)), int(m.group(2)))
    tdir = d / 'tasks'
    if tdir.is_dir():
        for task, (metric, filt, _) in PRIMARY.items():
            sp = tdir / f'{task}_samples.jsonl'
            if not (tdir / f'{task}.json').exists() or not sp.exists():
                continue
            vals = {}
            for line in open(sp):
                s = json.loads(line)
                if metric not in s:
                    continue
                if filt and s.get('filter') not in (filt, None):
                    continue
                v = s[metric]
                vals[(s['task'], s['doc_id'])] = v / 100.0 if v > 1.0 else v
            if vals:
                case['samples'][task] = vals
                case['tasks'][task] = sum(vals.values()) / len(vals)
    return case


def bootstrap_diff(a, b, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


def mcnemar(a, b):
    only_a = sum(1 for x, y in zip(a, b) if x >= 0.5 > y)
    only_b = sum(1 for x, y in zip(a, b) if y >= 0.5 > x)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return only_a, only_b, min(1.0, p)


def loss_pair(ca, cb, which):
    keys = sorted(k for k in ca['loss_rows'] if k[0] == which and k in cb['loss_rows'])
    if not keys:
        return None
    A = [ca['loss_rows'][k] for k in keys]
    B = [cb['loss_rows'][k] for k in keys]
    la = sum(x for x, _ in A) / sum(n for _, n in A)
    lb = sum(x for x, _ in B) / sum(n for _, n in B)
    rng = random.Random(0)
    diffs = []
    for _ in range(1000):
        idx = [rng.randrange(len(keys)) for _ in keys]
        da = sum(A[i][0] for i in idx) / sum(A[i][1] for i in idx)
        db = sum(B[i][0] for i in idx) / sum(B[i][1] for i in idx)
        diffs.append(da - db)
    diffs.sort()
    return {'rows': len(keys), 'a': la, 'b': lb, 'diff': la - lb, 'ci95': [diffs[25], diffs[974]]}


def fmt(x, digits=3):
    return '-' if x is None else f'{x:.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots', nargs='+', help='eval roots containing <case>/ folders')
    p.add_argument('--output', default='.', help='folder for compare.md / compare.json')
    a = p.parse_args()

    cases = {}
    for root in map(Path, a.roots):
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name in CASE_ORDER:
                if d.name in cases:
                    raise SystemExit(f'case {d.name} found in two roots; keep one')
                cases[d.name] = load_case(d)
    names = [n for n in CASE_ORDER if n in cases]
    if not names:
        raise SystemExit('no case folders found')
    tasks = [t for t in TASK_ORDER if any(t in cases[n]['tasks'] for n in names)]

    md = ['# Evaluation summary', '', '## Per case', '']
    head = ['case', 'text loss', 'answer loss', 'ROUGE-L', 'BLEU', 'answer key'] + tasks
    md += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    table = {}
    for n in names:
        c = cases[n]
        loss = c.get('loss', {})
        gen = c.get('gen', {})
        ak = c.get('answer_key')
        row = {'text_loss': loss.get('text_loss'), 'response_loss': loss.get('response_loss'),
               'rougeL': gen.get('rougeL_f1'), 'bleu': gen.get('bleu'),
               'answer_key': f'{ak[0]}/{ak[1]}' if ak else None,
               **{t: c['tasks'].get(t) for t in tasks}}
        table[n] = row
        cells = [n, fmt(row['text_loss']), fmt(row['response_loss']), fmt(row['rougeL']),
                 fmt(row['bleu'], 2), row['answer_key'] or '-'] + [fmt(row[t]) for t in tasks]
        md.append('| ' + ' | '.join(cells) + ' |')
    md += ['', 'Losses: mean NLL per token, same tokenizer and formula for every case '
           '(uniform_loss_eval.py). Task columns: the primary metric of each task, 0-1.', '']

    md += ['## Paired comparisons (A minus B)', '',
           'Positive = A better for accuracy metrics; negative = A better for losses. '
           '"only A / only B" = questions only that side got right; p = exact McNemar test.', '']
    comparisons = []
    for A, B in PAIRS:
        if A not in cases or B not in cases:
            continue
        ca, cb = cases[A], cases[B]
        lines = []
        for which, label in (('text', 'text loss'), ('response', 'answer loss')):
            r = loss_pair(ca, cb, which)
            if r:
                comparisons.append({'a': A, 'b': B, 'metric': label, **r})
                lines.append(f'| {label} | {r["rows"]} | {fmt(r["a"])} | {fmt(r["b"])} | {r["diff"]:+.4f} | '
                             f'[{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}] | - |')
        for t in tasks:
            sa, sb = ca['samples'].get(t), cb['samples'].get(t)
            if not sa or not sb:
                continue
            keys = sorted(set(sa) & set(sb), key=str)
            if not keys:
                continue
            va = [sa[k] for k in keys]
            vb = [sb[k] for k in keys]
            ma, mb = sum(va) / len(va), sum(vb) / len(vb)
            lo, hi = bootstrap_diff(va, vb)
            rec = {'a': A, 'b': B, 'metric': t, 'questions': len(keys), 'a_score': ma, 'b_score': mb,
                   'diff': ma - mb, 'ci95': [lo, hi]}
            test = '-'
            if PRIMARY[t][2]:
                oa, ob, pv = mcnemar(va, vb)
                rec.update(only_a=oa, only_b=ob, mcnemar_p=pv)
                test = f'only A {oa} / only B {ob}, p={pv:.3g}'
            comparisons.append(rec)
            lines.append(f'| {t} | {len(keys)} | {fmt(ma)} | {fmt(mb)} | {ma - mb:+.4f} | '
                         f'[{lo:+.4f}, {hi:+.4f}] | {test} |')
        if lines:
            md += [f'### {A} vs {B}', '',
                   '| metric | n | A | B | A-B | 95% CI | right/wrong test |', '|---|---|---|---|---|---|---|']
            md += lines + ['']
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'compare.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out / 'compare.json').write_text(json.dumps({'cases': table, 'comparisons': comparisons}, indent=2) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
