#!/usr/bin/env python3
"""One table of training runs from their output folders.

Reads what the trainers write (comparison_logs.jsonl, comparison_steps.jsonl,
benchmark.json, comparison_config.json) and prints eval loss at every eval
step, training-loss averages, median step time and peak memory.

  python summarize_training.py outputs/run-a outputs/run-b ... [--output training_summary.md]

Loss values come from each run's own trainer. Runs from different trainers
(the CQ trainer vs Unsloth's) are not on the same loss scale; compare those
with the uniform losses in eval_downstream/compare.md instead.
"""
import argparse
import json
import statistics as st
from pathlib import Path


def load_jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()] if p.exists() else []


def summarize(run):
    run = Path(run)
    logs = load_jsonl(run / 'comparison_logs.jsonl')
    steps = load_jsonl(run / 'comparison_steps.jsonl')
    cfg = json.loads((run / 'comparison_config.json').read_text()) if (run / 'comparison_config.json').exists() else {}
    bench = json.loads((run / 'benchmark.json').read_text()) if (run / 'benchmark.json').exists() else {}
    train = {d['step']: d['loss'] for d in logs if 'step' in d and isinstance(d.get('loss'), (int, float))}
    evals = [(d['step'], d['eval_loss']) for d in logs if 'step' in d and isinstance(d.get('eval_loss'), (int, float))]

    def mean(a, b):
        v = [train[s] for s in train if a <= s <= b]
        return st.mean(v) if v else None
    times = []
    for d in steps[10:]:  # trainers name the per-step time differently
        v = next((d[k] for k in ('optimizer_step_seconds', 'step_seconds', 'seconds', 'step_time', 'wall_seconds')
                  if isinstance(d.get(k), (int, float))), None)
        if v is not None:
            times.append(v)
    return {
        'run': run.name,
        'adapter': cfg.get('adapter') or ('lora' if cfg.get('rank') else None),
        'base': cfg.get('quantization') or cfg.get('model'),
        'evals': evals,
        'final_eval': evals[-1][1] if evals else None,
        'best_eval': min(v for _, v in evals) if evals else None,
        'train_1_10': mean(1, 10), 'train_401_500': mean(401, 500), 'train_all': mean(1, 10 ** 9),
        'median_step_s': bench.get('median_steady_step_seconds') or (st.median(times) if times else None),
        'peak_alloc_gib': bench.get('peak_pytorch_allocated_gib'),
        'trainable': cfg.get('trainable_params'),
    }


def fmt(v, d=4):
    return '-' if v is None else (f'{v:.{d}f}' if isinstance(v, float) else str(v))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--output')
    a = p.parse_args()
    rows = [summarize(r) for r in a.runs if Path(r).is_dir()]
    md = ['| run | adapter | base | final eval | best eval | train 1-10 | train 401-500 | train mean | s/step | peak GiB |',
          '|' + '---|' * 10]
    for r in rows:
        md.append(f"| {r['run']} | {r['adapter'] or '-'} | {r['base'] or '-'} | {fmt(r['final_eval'])} | "
                  f"{fmt(r['best_eval'])} | {fmt(r['train_1_10'])} | {fmt(r['train_401_500'])} | "
                  f"{fmt(r['train_all'])} | {fmt(r['median_step_s'], 2)} | {fmt(r['peak_alloc_gib'], 1)} |")
    md += ['', '### Eval loss by step', '']
    steps = sorted({s for r in rows for s, _ in r['evals']})
    md += ['| run | ' + ' | '.join(map(str, steps)) + ' |', '|' + '---|' * (len(steps) + 1)]
    for r in rows:
        ev = dict(r['evals'])
        md.append(f"| {r['run']} | " + ' | '.join(fmt(ev.get(s)) for s in steps) + ' |')
    text = '\n'.join(md) + '\n'
    print(text)
    if a.output:
        Path(a.output).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
