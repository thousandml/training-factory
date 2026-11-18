#!/usr/bin/env python3
"""Make N distinct copies of one trained adapter for serving-throughput benchmarks.

Throughput and memory depend on adapter shape and count, not on the trained
values, so copies of one real adapter (with a tiny per-copy perturbation so no
two are identical) are a fair stand-in for N tenants.

  # GCMP (writes adapters.json for CQ_GCMP_ADAPTERS)
  python make_adapter_copies.py --kind gcmp --src outputs/qwen38-27b-cq-4bit-alpaca-gcmp-v1 \
      --out-dir /workspace/scaling/gcmp --n 256

  # PEFT LoRA, optionally cut to a lower rank (writes lora_modules.txt for --lora-modules)
  python make_adapter_copies.py --kind lora --src outputs/qwen38-27b-cq-r32-v7/checkpoint-300 \
      --out-dir /workspace/scaling/lora8 --n 256 --rank 8
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['gcmp', 'lora'], required=True)
    p.add_argument('--src', required=True, help='adapter folder with adapter_config.json + adapter_model.safetensors')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--rank', type=int, help='LoRA only: keep the first R ranks (sets r and lora_alpha to R)')
    p.add_argument('--noise', type=float, default=1e-3, help='relative per-copy perturbation')
    p.add_argument('--prefix', default=None)
    a = p.parse_args()

    src = Path(a.src)
    cfg = json.loads((src / 'adapter_config.json').read_text())
    tensors = load_file(str(src / 'adapter_model.safetensors'))
    if a.kind == 'lora' and a.rank:
        old = int(cfg['r'])
        if a.rank > old:
            raise SystemExit(f'--rank {a.rank} > source rank {old}')
        cut = {}
        for k, t in tensors.items():
            if 'lora_A' in k:
                cut[k] = t[:a.rank].contiguous()
            elif 'lora_B' in k:
                cut[k] = t[:, :a.rank].contiguous()
            else:
                cut[k] = t
        tensors = cut
        cfg['r'] = a.rank
        cfg['lora_alpha'] = a.rank
        cfg.pop('rank_pattern', None)
        cfg.pop('alpha_pattern', None)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = a.prefix or a.kind
    names = {}
    for i in range(a.n):
        name = f'{prefix}{i:04d}'
        d = out / name
        d.mkdir(exist_ok=True)
        g = torch.Generator().manual_seed(i)
        copy = {}
        for k, t in tensors.items():
            if t.is_floating_point() and t.numel() > 0:
                scale = a.noise * float(t.float().abs().mean())
                copy[k] = (t.float() + scale * torch.randn(t.shape, generator=g)).to(t.dtype)
            else:
                copy[k] = t
        save_file(copy, str(d / 'adapter_model.safetensors'))
        (d / 'adapter_config.json').write_text(json.dumps(cfg, indent=2))
        for extra in ('tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json'):
            if (src / extra).exists():
                shutil.copy(src / extra, d / extra)
        names[name] = str(d.resolve())

    (out / 'adapters.json').write_text(json.dumps(names, indent=2))
    (out / 'lora_modules.txt').write_text(' '.join(f'{k}={v}' for k, v in names.items()))
    size = sum(f.stat().st_size for f in (out / next(iter(names))).glob('*.safetensors'))
    print(f'{a.n} {a.kind} adapters in {out} ({size / 2**20:.1f} MiB each)')


if __name__ == '__main__':
    main()
