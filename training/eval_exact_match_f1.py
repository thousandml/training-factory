#!/usr/bin/env python3
"""Offline QA/extraction Exact Match + token F1 for saved eval_rouge_bleu.py predictions.

No model loading, GPU, or new generation. Supports indexed curated slices and
optional baseline predictions (e.g. base model without adapter) for regression checks.

Examples:
  python eval_exact_match_f1.py \
    --run unsloth=outputs/rouge_bleu/unsloth_full_predictions.jsonl \
    --run cq500=outputs/rouge_bleu/cq_full_predictions.jsonl \
    --run cq100=outputs/rouge_bleu/cq_full_predictions-100steps.jsonl \
    --output outputs/rouge_bleu/qa_scores.json

  python eval_exact_match_f1.py --run ... --slice qa_extraction_indices.jsonl \
    --baseline base=outputs/rouge_bleu/cq_base_predictions.jsonl

A slice is JSONL with one row per evaluated example, e.g.:
  {"index": 12, "task": "qa"}
  {"index": 37, "task": "extraction", "reference": "Paris"}
'index' is the original *1-based* prediction index. Optional 'reference'
replaces the original reference for all runs: use only for independently
verified ground truths and keep the same target for all runs. You may specify
multiple acceptable answer strings as 'answers': ["USA", "United States"].
Without --slice, scores run over all saved examples and should be treated as
lexical overlap diagnostics, NOT correctness on general open-ended Alpaca data.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import string
import unicodedata


def normalize_answer(value):
    """SQuAD-style English normalization: case, punctuation, articles, whitespace."""
    value = unicodedata.normalize('NFKC', str(value)).lower()
    value = ''.join(ch for ch in value if not unicodedata.category(ch).startswith('P'))
    value = re.sub(r'\b(a|an|the)\b', ' ', value)
    return ' '.join(value.split())


def exact_match(prediction, references):
    return max(float(normalize_answer(prediction) == normalize_answer(ref))
               for ref in references)


def token_f1_one(prediction, reference):
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    matches = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if not matches:
        return 0.0
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction, references):
    return max(token_f1_one(prediction, ref) for ref in references)


def load_predictions(path):
    result = {}
    with Path(path).open(encoding='utf-8') as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            index = entry.get('index')
            if not isinstance(index, int) or index <= 0 or index in result:
                raise ValueError(f'{path}:{line_no}: invalid or duplicate index {index!r}')
            if not isinstance(entry.get('reference'), str) or not isinstance(entry.get('prediction'), str):
                raise ValueError(f'{path}:{line_no}: requires string reference and prediction')
            result[index] = entry
    if not result:
        raise ValueError(f'Empty prediction file: {path}')
    return result


def parse_named_path(text):
    if '=' not in text:
        raise argparse.ArgumentTypeError('Expected NAME=PATH')
    name, path = text.split('=', 1)
    if not name or not path:
        raise argparse.ArgumentTypeError('Expected nonempty NAME=PATH')
    return name, path


def load_slice(path):
    rows = {}
    with Path(path).open(encoding='utf-8') as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row.get('index')
            if not isinstance(idx, int) or idx < 1 or idx in rows:
                raise ValueError(f'{path}:{line_no}: invalid/duplicate index: {idx!r}')
            task = row.get('task', 'qa')
            if task not in ('qa', 'extraction'):
                raise ValueError(f'{path}:{line_no}: task must be qa or extraction')
            if 'answers' in row:
                answers = row['answers']
                if not isinstance(answers, list) or not answers or not all(isinstance(x, str) and x for x in answers):
                    raise ValueError(f'{path}:{line_no}: answers must be a nonempty list of strings')
            elif 'reference' in row:
                if not isinstance(row['reference'], str):
                    raise ValueError(f'{path}:{line_no}: reference must be a string')
                answers = [row['reference']]
            else:
                answers = None
            rows[idx] = {'task': task, 'answers': answers}
    if not rows:
        raise ValueError('Slice is empty')
    return rows


def score_run(rows, definitions):
    groups = defaultdict(list)
    per_example = {}
    for idx, entry in rows.items():
        definition = definitions[idx]
        refs = definition['answers'] or [entry['reference']]
        em = exact_match(entry['prediction'], refs)
        f1 = token_f1(entry['prediction'], refs)
        record = {'index': idx, 'task': definition['task'], 'exact_match': em, 'token_f1': f1}
        per_example[idx] = record
        groups['all'].append(record)
        groups[definition['task']].append(record)
    summary = {}
    for group, values in sorted(groups.items()):
        summary[group] = {
            'examples': len(values),
            'exact_match': sum(v['exact_match'] for v in values) / len(values),
            'token_f1': sum(v['token_f1'] for v in values) / len(values),
        }
    return summary, per_example


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, type=parse_named_path,
                        metavar='NAME=PATH', help='Saved model predictions; repeat for multiple runs')
    parser.add_argument('--slice', help='Optional curated QA/extraction index list JSONL')
    parser.add_argument('--baseline', type=parse_named_path, metavar='NAME=PATH',
                        help='Saved predictions from untuned base model, on the SAME examples')
    parser.add_argument('--output', required=True, help='JSON summary output; does not overwrite predictions')
    args = parser.parse_args()
    entries = list(args.run) + ([args.baseline] if args.baseline else [])
    names = [name for name, _ in entries]
    if len(set(names)) != len(names):
        parser.error('Run names must be unique')
    datasets = {name: load_predictions(path) for name, path in entries}
    if args.slice:
        definitions = load_slice(args.slice)
    else:
        first = datasets[names[0]]
        definitions = {idx: {'task': 'qa', 'answers': None} for idx in first}

    indices = set(definitions)
    for name, rows in datasets.items():
        missing = indices - set(rows)
        if missing:
            parser.error(f'{name}: missing {len(missing)} requested indices (first: {sorted(missing)[:5]})')

    # Avoid comparing mismatched evaluation runs (same indices, different prompts/references).
    anchor = datasets[names[0]]
    for name, rows in datasets.items():
        for idx in indices:
            a, b = anchor[idx], rows[idx]
            if a['reference'] != b['reference'] or a.get('messages') != b.get('messages'):
                parser.error(f'{name}: example {idx} has different prompt/reference than {names[0]}')

    summaries, detailed = {}, {}
    for name, rows in datasets.items():
        summaries[name], detailed[name] = score_run({idx: rows[idx] for idx in sorted(indices)}, definitions)

    comparisons = {}
    baseline_name = args.baseline[0] if args.baseline else None
    if baseline_name:
        base_details = detailed[baseline_name]
        for name in names:
            if name == baseline_name:
                continue
            improved = regressed = same = 0
            for idx in indices:
                base_correct = base_details[idx]['exact_match'] == 1
                tuned_correct = detailed[name][idx]['exact_match'] == 1
                if tuned_correct and not base_correct:
                    improved += 1
                elif base_correct and not tuned_correct:
                    regressed += 1
                else:
                    same += 1
            comparisons[name] = {'baseline': baseline_name, 'examples': len(indices),
                                 'became_correct': improved, 'became_incorrect': regressed,
                                 'unchanged_correctness': same,
                                 'exact_match_delta': summaries[name]['all']['exact_match'] - summaries[baseline_name]['all']['exact_match'],
                                 'token_f1_delta': summaries[name]['all']['token_f1'] - summaries[baseline_name]['all']['token_f1']}

    result = {
        'mode': 'curated_qa_extraction' if args.slice else 'all_examples_lexical_diagnostic',
        'note': ('A curated QA/extraction subset with verified short-answer references is needed to interpret EM/F1 as factual correctness.'
                 if not args.slice else 'EM and token-F1 measure reference agreement; use verified, preferably short-answer QA/extraction labels.'),
        'slice': args.slice, 'metrics': summaries, 'baseline_comparisons': comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'\nSaved: {output}')


if __name__ == '__main__':
    main()
