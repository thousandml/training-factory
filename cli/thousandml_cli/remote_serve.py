"""Start vLLM on the base model with every adapter in /adapters as a named model.

Each sub-folder with an adapter_config.json is one adapter, requested with
"model": "<folder name>". "model": "base" uses no adapter.

  CQ base  : GCMP adapters trained on CQ (ThousandML plugin) or LoRA adapters
  BF16 base: GCMP adapters trained on that full-precision model, or LoRA adapters
  NF4 base : LoRA adapters only (vLLM multi-LoRA)
"""
import json
import os
import pathlib
import sys

prec = os.environ['PRECISION']
model_dir = os.environ['MODEL_DIR']
max_len = os.environ.get('MAX_MODEL_LEN', '8192')
max_loras = int(os.environ.get('MAX_LORAS', '16'))

gcmp, lora, rank = {}, {}, 8
root = pathlib.Path('/adapters')
for p in sorted(root.iterdir()) if root.is_dir() else []:
    if not (p / 'adapter_config.json').is_file():
        continue
    cfg = json.loads((p / 'adapter_config.json').read_text())
    meta_file = p / 'thousandml.json'
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    if meta.get('precision') and meta['precision'] != prec:
        print(f'skip {p.name}: trained on a {meta["precision"]} base, this service runs a {prec} base')
        continue
    if cfg.get('adapter_type') == 'cq_gcmp':
        need = {'cq': 'cq', 'bf16': 'dense'}.get(prec)
        if need is None or cfg.get('base_kind', 'cq') != need:
            print(f'skip {p.name}: a GCMP adapter with base_kind={cfg.get("base_kind", "cq")!r} '
                  f'cannot be served on a {prec} base')
            continue
        gcmp[p.name] = str(p)
    elif str(cfg.get('peft_type', '')).upper() == 'LORA':
        lora[p.name] = str(p)
        while rank < int(cfg.get('r', 8)):
            rank *= 2
    else:
        print(f'skip {p.name}: unknown adapter type')

if gcmp and lora:
    sys.exit('This folder mixes GCMP and LoRA adapters. Serve them as two services.')

cmd = ['vllm', 'serve', model_dir, '--host', '0.0.0.0', '--port', '8000',
       '--max-model-len', max_len, '--served-model-name', 'base']
if prec == 'bf16':
    # GCMP's prefill buffer: keep the batch modest on a model that nearly fills the GPU
    cmd += ['--max-num-batched-tokens', os.environ.get('MAX_BATCHED_TOKENS', '4096')]
cmd += {'cq': ['--quantization', 'cq'], 'nf4': ['--quantization', 'bitsandbytes']}.get(prec, [])
os.environ.pop('CQ_GCMP_ADAPTER', None)
if gcmp:
    with open('/tmp/adapters.json', 'w') as f:
        json.dump(gcmp, f, indent=1)
    os.environ['CQ_GCMP_ADAPTERS'] = '/tmp/adapters.json'
if lora:
    cmd += ['--enable-lora', '--max-lora-rank', str(rank),
            '--max-loras', str(min(len(lora), max_loras)), '--max-cpu-loras', str(len(lora)),
            '--lora-modules', *[f'{k}={v}' for k, v in lora.items()]]

names = list(gcmp or lora)
kind = 'GCMP' if gcmp else 'LoRA'
print(f'{prec} base, {len(names)} {kind} adapters: {", ".join(names) or "none"}', flush=True)
os.execvp(cmd[0], cmd)
