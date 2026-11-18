#!/usr/bin/env python3
"""Mixed-adapter serving throughput against a running vLLM OpenAI server.

Every request picks an adapter uniformly at random (by model name), so batches
mix tenants. Output length is fixed (ignore_eos) so every configuration does
the same amount of work. Reports output tokens/s, requests/s, latency and TTFT.

  python bench_multi_adapter.py --url http://localhost:8000 --names-file adapters.json \
      --prompts alpaca_val_eval.jsonl --concurrency 32 --num-requests 384 --output result.json
"""
import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


def load_prompts(path, n, seed):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        text = r['instruction'] + (f"\n\n{r['input']}" if r.get('input') else '')
        out.append(f"### Instruction:\n{text}\n\n### Response:\n")
    while len(out) < n:
        out.extend(out[: n - len(out)])
    return out


async def one(session, url, model, prompt, max_tokens):
    body = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0.0,
            'ignore_eos': True, 'stream': True,
            'stream_options': {'include_usage': True}}
    t0 = time.perf_counter()
    ttft, ntok = None, 0
    async with session.post(f'{url}/v1/completions', json=body) as r:
        if r.status != 200:
            raise RuntimeError(f'{model}: HTTP {r.status}: {(await r.text())[:300]}')
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if chunk.get('choices') and chunk['choices'][0].get('text'):
                if ttft is None:
                    ttft = time.perf_counter() - t0
            usage = chunk.get('usage')
            if usage:
                ntok = usage.get('completion_tokens', ntok)
    return time.perf_counter() - t0, ttft or 0.0, ntok or max_tokens


async def run(a):
    if a.names_file:
        p = Path(a.names_file)
        names = list(json.loads(p.read_text())) if p.suffix == '.json' else p.read_text().split()
        names = [n.split('=')[0] for n in names]
    else:
        names = [a.model]
    prompts = load_prompts(a.prompts, a.num_requests, a.seed)
    rng = random.Random(a.seed)
    picks = [rng.choice(names) for _ in prompts]
    sem = asyncio.Semaphore(a.concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        # warmup: touch a few adapters so first-load cost isn't in the timed window
        await asyncio.gather(*[one(s, a.url, n, prompts[0], 8) for n in names[: min(len(names), a.concurrency)]])

        async def task(m, pr):
            async with sem:
                return await one(s, a.url, m, pr, a.max_tokens)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[task(m, pr) for m, pr in zip(picks, prompts)])
        wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in res)
    ttft = sorted(r[1] for r in res)
    toks = sum(r[2] for r in res)
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
    out = {'adapters': len(names), 'requests': len(res), 'concurrency': a.concurrency,
           'max_tokens': a.max_tokens, 'wall_s': round(wall, 2),
           'output_tok_per_s': round(toks / wall, 1), 'req_per_s': round(len(res) / wall, 3),
           'latency_p50_s': round(q(lat, .5), 2), 'latency_p95_s': round(q(lat, .95), 2),
           'ttft_p50_s': round(q(ttft, .5), 3), 'ttft_p95_s': round(q(ttft, .95), 3),
           'distinct_adapters_hit': len(set(picks))}
    print(json.dumps(out, indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://localhost:8000')
    p.add_argument('--names-file', help='adapters.json or lora_modules.txt; omit for the base model')
    p.add_argument('--model', help='served model name when --names-file is omitted')
    p.add_argument('--prompts', required=True)
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--num-requests', type=int, default=384)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output')
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()
