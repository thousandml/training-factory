#!/usr/bin/env python3
"""Long-running multi-tenant serving test with skewed traffic and adapter turnover.

Unlike bench_multi_adapter.py (uniform traffic, fixed adapter set, short run), this
sends open-loop Poisson traffic for a fixed duration, picks tenants with a Zipf
(long-tail) popularity, and optionally replaces tenants while traffic runs:

  skew  : fixed adapter set, Zipf traffic. With LoRA's max_cpu_loras below the
          adapter count, tail tenants have to be paged in from disk (LoRA's
          normal turnover path); GCMP keeps every adapter resident.
  churn : every --churn-every seconds one tenant leaves and a new one arrives.
          The slot is drained (no new requests, wait for in-flight ones), then
            LoRA: POST /v1/unload_lora_adapter old + /v1/load_lora_adapter new
            GCMP: POST /collective_rpc gcmp_swap(slot, new_path)  (in-place swap)
          and the new tenant gets an immediate first request.

Every 5 s it also samples GPU memory (nvidia-smi), KV-cache usage (/metrics) and
the server's host RSS, to show drift or fragmentation over the run.

  python bench_turnover.py --url http://localhost:8011 --kind gcmp --mode churn \
     --names-file pool.json --incoming-file incoming.json --prompts val.jsonl \
     --duration 2700 --rate 0.45 --server-pid 1234 --output result.json
"""
import argparse
import asyncio
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import aiohttp

from bench_multi_adapter import load_prompts


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, math.ceil(q / 100 * len(xs)) - 1))], 3)


def summary(xs):
    return {'n': len(xs), 'p50': pct(xs, 50), 'p95': pct(xs, 95), 'p99': pct(xs, 99),
            'max': round(max(xs), 3) if xs else None}


def tree_rss_mb(pid):
    """RSS of a process and all its descendants (vLLM runs the engine in a child)."""
    if not pid:
        return None
    kids = {}
    for d in os.listdir('/proc'):
        if d.isdigit():
            try:
                ppid = int(open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except OSError:
                pass
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        try:
            for line in open(f'/proc/{p}/status'):
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1])
        except OSError:
            pass
        todo += kids.get(p, [])
    return round(total / 1024)


class Tenants:
    """Slots ordered by popularity rank; a new tenant inherits the leaving tenant's rank."""

    def __init__(self, names, zipf, seed):
        self.slots = list(names)                    # served name per rank
        self.available = [True] * len(names)
        self.inflight = [0] * len(names)
        self.rng = random.Random(seed)
        w = [1 / (i + 1) ** zipf for i in range(len(names))]
        s = sum(w)
        self.cum = []
        acc = 0.0
        for x in w:
            acc += x / s
            self.cum.append(acc)

    def pick(self):
        for _ in range(100):
            u = self.rng.random()
            lo, hi = 0, len(self.cum) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self.cum[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            if self.available[lo]:
                return lo
        return next(i for i, ok in enumerate(self.available) if ok)


async def request(session, url, model, prompt, max_tokens):
    body = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0.0,
            'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}}
    t0 = time.perf_counter()
    ttft, ntok = None, 0
    async with session.post(f'{url}/v1/completions', json=body) as r:
        if r.status != 200:
            raise RuntimeError(f'HTTP {r.status}: {(await r.text())[:200]}')
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if ttft is None and chunk.get('choices') and chunk['choices'][0].get('text'):
                ttft = time.perf_counter() - t0
            if chunk.get('usage'):
                ntok = chunk['usage'].get('completion_tokens', ntok)
    return time.perf_counter() - t0, ttft or 0.0, ntok


async def main_async(a):
    pool = json.loads(Path(a.names_file).read_text())
    incoming = list(json.loads(Path(a.incoming_file).read_text()).items()) if a.incoming_file else []
    prompts = load_prompts(a.prompts, 512, a.seed)
    rng = random.Random(a.seed + 1)
    names = list(pool)
    rng.shuffle(names)                                 # popularity rank independent of name order
    ten = Tenants(names, a.zipf, a.seed)
    records, swaps, samples = [], [], []
    t_start = time.perf_counter()
    end = t_start + a.duration
    sem = asyncio.Semaphore(a.max_inflight)
    rejected = 0
    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def one(slot, first=False):
            model = ten.slots[slot]
            ten.inflight[slot] += 1
            t0 = time.perf_counter() - t_start
            try:
                lat, ttft, ntok = await request(session, a.url, model, rng.choice(prompts), a.max_tokens)
                records.append({'t': round(t0, 2), 'rank': slot, 'model': model, 'lat': lat, 'ttft': ttft,
                                'tok': ntok, 'ok': True, 'first_after_swap': first})
            except Exception as e:  # noqa: BLE001 - record and keep going
                records.append({'t': round(t0, 2), 'rank': slot, 'model': model, 'ok': False,
                                'err': str(e)[:200], 'first_after_swap': first})
            finally:
                ten.inflight[slot] -= 1

        async def guarded(slot, first=False):
            nonlocal rejected
            if sem.locked():
                rejected += 1
                return
            async with sem:
                await one(slot, first)

        async def post(path, body):
            t0 = time.perf_counter()
            async with session.post(f'{a.url}{path}', json=body) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f'{path}: HTTP {r.status}: {text[:300]}')
            return time.perf_counter() - t0

        async def churn():
            k = 0
            while incoming and time.perf_counter() + a.churn_every < end:
                await asyncio.sleep(a.churn_every)
                lo = len(ten.slots) // 4                   # leave the top quarter alone
                slot = rng.randrange(lo, len(ten.slots))
                new_name, new_path = incoming[k % len(incoming)]
                k += 1
                ten.available[slot] = False
                t0 = time.perf_counter()
                while ten.inflight[slot] and time.perf_counter() - t0 < 120:
                    await asyncio.sleep(0.05)
                drained = time.perf_counter() - t0
                old = ten.slots[slot]
                rec = {'t': round(t0 - t_start, 2), 'rank': slot, 'old': old, 'new': new_name, 'drain_s': round(drained, 3)}
                try:
                    if a.kind == 'lora':
                        rec['unload_s'] = round(await post('/v1/unload_lora_adapter', {'lora_name': old}), 3)
                        rec['load_s'] = round(await post('/v1/load_lora_adapter',
                                                         {'lora_name': new_name, 'lora_path': new_path}), 3)
                        ten.slots[slot] = new_name
                    else:
                        rec['load_s'] = round(await post('/collective_rpc',
                                                         {'method': 'gcmp_swap', 'args': [old, new_path]}), 3)
                    rec['ok'] = True
                except Exception as e:  # noqa: BLE001
                    rec.update(ok=False, err=str(e)[:300])
                rec['unavailable_s'] = round(time.perf_counter() - t0, 3)
                ten.available[slot] = True
                swaps.append(rec)
                asyncio.create_task(guarded(slot, first=True))   # the new tenant's first request

        async def sampler():
            while time.perf_counter() < end:
                s = {'t': round(time.perf_counter() - t_start, 1), 'rss_mb': tree_rss_mb(a.server_pid)}
                try:
                    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                          f'--id={a.gpu}'], capture_output=True, text=True, timeout=10).stdout
                    s['gpu_mb'] = int(out.strip().splitlines()[0])
                except Exception:  # noqa: BLE001
                    pass
                try:
                    async with session.get(f'{a.url}/metrics') as r:
                        for line in (await r.text()).splitlines():
                            if line.startswith(('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')):
                                s['kv_usage'] = round(float(line.rsplit(' ', 1)[1]), 4)
                except Exception:  # noqa: BLE001
                    pass
                samples.append(s)
                await asyncio.sleep(5)

        bg = [asyncio.create_task(sampler())]
        if a.mode == 'churn':
            bg.append(asyncio.create_task(churn()))
        tasks = []
        nxt = time.perf_counter()
        while nxt < end:                                   # open-loop Poisson arrivals
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
            tasks.append(asyncio.create_task(guarded(ten.pick())))
            nxt += rng.expovariate(a.rate)
        await asyncio.gather(*tasks)
        for t in bg:
            t.cancel()

    ok = [r for r in records if r['ok']]
    wall = time.perf_counter() - t_start
    windows = []
    for w0 in range(0, int(a.duration), a.window):
        rs = [r for r in ok if w0 <= r['t'] < w0 + a.window]
        windows.append({'from_s': w0, 'n': len(rs), 'lat_p95': pct([r['lat'] for r in rs], 95),
                        'ttft_p50': pct([r['ttft'] for r in rs], 50), 'ttft_p99': pct([r['ttft'] for r in rs], 99)})
    head = [r for r in ok if r['rank'] < 16]
    tail = [r for r in ok if r['rank'] >= len(names) // 2]

    def drift(key):
        xs = [s[key] for s in samples if s.get(key) is not None]
        if len(xs) < 10:
            return None
        k = max(1, len(xs) // 10)
        return {'start': round(statistics.mean(xs[:k]), 1), 'end': round(statistics.mean(xs[-k:]), 1),
                'max': max(xs), 'change': round(statistics.mean(xs[-k:]) - statistics.mean(xs[:k]), 1)}

    res = {
        'kind': a.kind, 'mode': a.mode, 'adapters': len(names), 'zipf': a.zipf, 'rate_req_s': a.rate,
        'duration_s': a.duration, 'max_tokens': a.max_tokens, 'max_inflight': a.max_inflight,
        'requests_ok': len(ok), 'requests_failed': len(records) - len(ok), 'requests_rejected': rejected,
        'distinct_adapters_hit': len({r['model'] for r in ok}),
        'output_tok_per_s': round(sum(r['tok'] for r in ok) / wall, 1),
        'latency_s': summary([r['lat'] for r in ok]), 'ttft_s': summary([r['ttft'] for r in ok]),
        'ttft_head16_s': summary([r['ttft'] for r in head]), 'ttft_tail_half_s': summary([r['ttft'] for r in tail]),
        'first_request_after_swap_ttft_s': summary([r['ttft'] for r in ok if r['first_after_swap']]),
        'swaps': {'n': len(swaps), 'failed': sum(not s['ok'] for s in swaps),
                  'load_s': summary([s['load_s'] for s in swaps if s.get('load_s') is not None]),
                  'unavailable_s': summary([s['unavailable_s'] for s in swaps])},
        'gpu_mem_mb': drift('gpu_mb'), 'host_rss_mb': drift('rss_mb'), 'kv_usage': drift('kv_usage'),
        'windows': windows,
        'errors': [r['err'] for r in records if not r['ok']][:5] + [s['err'] for s in swaps if not s['ok']][:5],
    }
    Path(a.output).write_text(json.dumps(res, indent=2))
    Path(a.output).with_suffix('.raw.json').write_text(json.dumps({'requests': records, 'swaps': swaps,
                                                                   'samples': samples}))
    print(json.dumps({k: v for k, v in res.items() if k != 'windows'}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--kind', choices=['gcmp', 'lora'], required=True)
    p.add_argument('--mode', choices=['skew', 'churn'], required=True)
    p.add_argument('--names-file', required=True, help='{"name": "path"} of adapters loaded at startup')
    p.add_argument('--incoming-file', help='churn: {"name": "path"} of new tenants to bring in')
    p.add_argument('--prompts', required=True)
    p.add_argument('--duration', type=float, default=2700)
    p.add_argument('--rate', type=float, default=0.45, help='mean arrivals per second (Poisson)')
    p.add_argument('--zipf', type=float, default=1.1, help='popularity skew; 0 = uniform')
    p.add_argument('--churn-every', type=float, default=20.0)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--max-inflight', type=int, default=96, help='client-side cap; extra arrivals count as rejected')
    p.add_argument('--window', type=int, default=300)
    p.add_argument('--server-pid', type=int)
    p.add_argument('--gpu', default='0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == '__main__':
    main()
