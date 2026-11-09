#!/usr/bin/env python3
"""Count-right-or-wrong benchmarks (lm-evaluation-harness) for one base/adapter case.

Loads the model exactly as eval_rouge_bleu.py does (CQ or Unsloth NF4/16-bit,
with a GCMP or LoRA adapter, or none) and runs standard lm-eval tasks on the
already-loaded model. Every task's results and per-question outcomes are saved
separately, so cases can be compared question by question.

Suites (all deterministic: greedy decoding, fixed few-shot seeds, the first N
questions of each task when limited, so every case sees the same questions):

  task            measures                         shots  scored by
  ifeval          following explicit instructions    0    strict prompt-level pass
  gsm8k           grade-school math                  5    exact match of final number
  squad_completion  extract an answer span from a passage  0  answer contained in output
  drop            extraction + arithmetic over a passage 3    exact match / F1
  arc_challenge   science reasoning (multiple choice) 0   accuracy
  hellaswag       commonsense completion (MC)        0    accuracy
  winogrande      pronoun resolution (MC)            0    accuracy
  boolq           yes/no reading comprehension       0    accuracy
  mmlu            57-subject knowledge (MC)          0    accuracy
  truthfulqa_mc2  avoiding common falsehoods (MC)    0    probability on true answers

  optional (--tasks swde,fda): extraction of fields from long semi-structured
  documents (HTML pages, FDA filings); slow and memory-hungry, off by default.

  --suite quick   limits each task (~1-3 h per case on one GPU)
  --suite full    every question (expect 10+ h per case for a 27B model)

  python downstream_eval.py --backend cq --model /path/CQ --adapter /path/adapter \
      --output-dir eval/cq_gcmp/tasks --suite quick
"""
import argparse
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# name: (num_fewshot, chat template, quick limit)
SUITE = {
    'ifeval': (0, True, 200),
    'gsm8k': (5, False, 300),
    'squad_completion': (0, False, 500),
    'drop': (3, False, 500),
    'arc_challenge': (0, False, 500),
    'hellaswag': (0, False, 1000),
    'winogrande': (0, False, 500),
    'boolq': (0, False, 500),
    'mmlu': (0, False, 20),          # per subject (57 subjects)
    'truthfulqa_mc2': (0, False, None),
}
OPTIONAL = {
    'swde': (0, False, 300),
    'fda': (0, False, 300),
}
METRIC_KEYS = ('acc', 'acc_norm', 'exact_match', 'em', 'f1', 'contains',
               'prompt_level_strict_acc', 'prompt_level_loose_acc',
               'inst_level_strict_acc', 'inst_level_loose_acc')


def model_args(a):
    return argparse.Namespace(
        backend=a.backend, model=a.model, adapter=a.adapter, tokenizer=a.tokenizer,
        kernel_module=a.kernel_module, kernel_backend='auto', trust_remote_code=a.trust_remote_code,
        unsloth_16bit=a.unsloth_16bit, max_input_tokens=a.max_length, max_new_tokens=512,
        text_only=a.text_only)


def signature(a, task, shots, chat, limit):
    payload = {'backend': a.backend, 'model': a.model, 'adapter': a.adapter, 'task': task,
               'shots': shots, 'chat': chat, 'limit': limit, 'unsloth_16bit': a.unsloth_16bit,
               'text_only': a.text_only, 'max_length': a.max_length}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def numeric(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, (bool, int, float)) for x in v):
        return float(sum(v)) / len(v)  # ifeval instruction-level lists
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=['cq', 'unsloth'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter')
    p.add_argument('--tokenizer')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--suite', choices=['quick', 'full'], default='quick')
    p.add_argument('--tasks', help='comma-separated subset of the suite')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--kernel-module', default='cq.cq_kernels')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--unsloth-16bit', action='store_true')
    layout = p.add_mutually_exclusive_group()
    layout.add_argument('--text-only', dest='text_only', action='store_true', default=None)
    layout.add_argument('--full-model', dest='text_only', action='store_false')
    a = p.parse_args()

    tasks = [t for t in (a.tasks.split(',') if a.tasks else SUITE) if t]
    known = {**SUITE, **OPTIONAL}
    unknown = [t for t in tasks if t not in known]
    if unknown:
        p.error(f'unknown tasks {unknown}; choose from {list(known)}')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan = []
    for task in tasks:
        shots, chat, quick = known[task]
        limit = quick if a.suite == 'quick' else None
        sig = signature(a, task, shots, chat, limit)
        res_path = out / f'{task}.json'
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            if prev.get('signature') == sig:
                print(f'== {task}: done already, skipping', flush=True)
                continue
            raise SystemExit(f'{res_path} was produced with different settings; '
                             'use a new --output-dir or delete it')
        plan.append((task, shots, chat, limit, sig))
    if not plan:
        return

    import torch
    import eval_rouge_bleu as ev
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model, tokenizer = ev.load_model(model_args(a))
    lm_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    lm_model.eval()
    device = next(p_.device for p_ in lm_model.parameters() if p_.device.type == 'cuda')
    if any(type(m).__name__ == 'GCMPLinear' for m in lm_model.modules()):
        use_cache = ev.choose_generation_cache(lm_model, tokenizer, device)
        if not use_cache:
            # lm-eval always passes use_cache=True; force the reference uncached path.
            original = lm_model.generate

            @functools.wraps(original)
            def generate(*args, **kwargs):
                kwargs['use_cache'] = False
                return original(*args, **kwargs)
            lm_model.generate = generate
            print('GCMP: cached decoding bypasses the adapter here; generating without KV cache', flush=True)
    tokenizer.padding_side = 'left'
    lm = HFLM(pretrained=lm_model, tokenizer=tokenizer, backend='causal', batch_size=a.batch_size,
              max_length=a.max_length, enable_thinking=False)

    for task, shots, chat, limit, sig in plan:
        print(f'== {task}: {shots}-shot, chat template {"on" if chat else "off"}, '
              f'limit {limit or "none"}', flush=True)
        t0 = time.time()
        res = simple_evaluate(
            model=lm, tasks=[task], num_fewshot=shots, limit=limit, log_samples=True,
            apply_chat_template=chat, fewshot_as_multiturn=chat and shots > 0,
            gen_kwargs={'do_sample': False, 'temperature': 0.0},
            bootstrap_iters=1000, random_seed=0, numpy_random_seed=1234,
            torch_random_seed=1234, fewshot_random_seed=1234, confirm_run_unsafe_code=False)
        samples_path = out / f'{task}_samples.jsonl'
        with open(samples_path, 'w', encoding='utf-8') as f:
            for sub, items in res.get('samples', {}).items():
                for s in items:
                    rec = {'task': sub, 'doc_id': s.get('doc_id'), 'filter': s.get('filter')}
                    for k in METRIC_KEYS:
                        if k in s:
                            v = numeric(s[k])
                            if v is not None:
                                rec[k] = v
                    f.write(json.dumps(rec) + '\n')
        summary = {
            'signature': sig, 'task': task, 'shots': shots, 'chat_template': chat, 'limit': limit,
            'backend': a.backend, 'model': a.model, 'adapter': a.adapter or None,
            'unsloth_16bit': a.unsloth_16bit, 'seconds': round(time.time() - t0, 1),
            'results': res.get('results', {}), 'groups': res.get('groups', {}),
            'n_samples': res.get('n-samples', {}), 'versions': res.get('versions', {}),
        }
        tmp = out / f'{task}.json.tmp'
        tmp.write_text(json.dumps(summary, indent=2, default=str) + '\n', encoding='utf-8')
        tmp.replace(out / f'{task}.json')
        print(json.dumps(summary['results'].get(task, summary['results']), indent=2, default=str)[:1500], flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
