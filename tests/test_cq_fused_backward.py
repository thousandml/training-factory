#!/usr/bin/env python3
"""Numerical parity and performance test for my_qlinear.backward_packed_fused_small.

Example:
  cd cq-kernels
  python /path/to/test_cq_fused_backward.py --packed /path/to/packed.pt --n 5120 --m 17408 --tokens 1,4,8,16

packed.pt must contain a real uint8 packed CQ tensor of shape [M*(N//1024), 520],
or a dictionary containing one under --key. Do not use random bytes for parity.
"""
import argparse
import sys
import time
from pathlib import Path

import torch


def load_packed(path, key):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict):
        if key:
            obj = obj[key]
        else:
            matches = [(k, v) for k, v in obj.items() if isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.ndim == 2 and v.shape[1] == 520]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one packed uint8 [*,520] tensor; found {[k for k,_ in matches]}. Supply --key.')
            print('Selected packed tensor:', matches[0][0])
            obj = matches[0][1]
    if not isinstance(obj, torch.Tensor):
        raise TypeError('Packed file must contain a tensor or dict of tensors')
    return obj


def timed(fn, repeats):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        result = fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / repeats
    peak = torch.cuda.max_memory_allocated() / 2**20
    return result, ms, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--packed', required=True, help='torch.save(...) file with a real packed CQ uint8 tensor')
    p.add_argument('--key', default=None, help='dictionary key, if packed file contains multiple tensors')
    p.add_argument('--n', type=int, required=True, help='padded input width N (multiple of 1024)')
    p.add_argument('--m', type=int, required=True, help='output width M')
    p.add_argument('--tokens', default='1,4,8,16')
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--atol', type=float, default=0.1, help='absolute tolerance for half-GEMM comparison')
    p.add_argument('--rtol', type=float, default=0.03)
    args = p.parse_args()

    import cq.cq_kernels  # noqa: F401 -- import registers torch.ops
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    print('Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
    ops = torch.ops.my_qlinear
    for name in ('decode_weight', 'backward_packed_fused_small'):
        if not hasattr(ops, name):
            raise RuntimeError(f'Missing registered operator: my_qlinear.{name}')
    print('Operators registered: yes')
    if args.n <= 0 or args.n % 1024 or args.m <= 0:
        raise ValueError('N must be a positive multiple of 1024 and M must be positive')
    packed = load_packed(args.packed, args.key)
    expected = (args.m * (args.n // 1024), 520)
    if packed.dtype != torch.uint8 or tuple(packed.shape) != expected:
        raise ValueError(f'Expected packed uint8 shape {expected}, got {packed.dtype} {tuple(packed.shape)}')
    packed = packed.contiguous().cuda()
    torch.manual_seed(args.seed)
    failures = 0
    print('Packed shape:', tuple(packed.shape), 'N:', args.n, 'M:', args.m)
    print('Memory numbers below are absolute allocated peaks, not additional workspace.')
    for L in [int(x) for x in args.tokens.split(',')]:
        if not (1 <= L <= 16):
            raise ValueError('This fused operator is intended for 1 <= L <= 16')
        gt = (torch.randn((L, args.m), device='cuda', dtype=torch.float32) * 0.25).half().contiguous()
        try:
            torch.cuda.synchronize()
            # Decode once for reference; release dense matrix before timing fused path.
            dense = ops.decode_weight(packed, args.n, args.m)
            reference = (gt @ dense).float()
            torch.cuda.synchronize()
            del dense
            torch.cuda.empty_cache()
            fused = ops.backward_packed_fused_small(packed, gt, args.n, args.m)
            torch.cuda.synchronize()
            if fused.shape != reference.shape:
                raise AssertionError(f'Shape mismatch: {tuple(fused.shape)} vs {tuple(reference.shape)}')
            delta = (fused.float() - reference).abs()
            max_abs = delta.max().item()
            rmse = delta.square().mean().sqrt().item()
            ok = torch.allclose(fused.float(), reference, atol=args.atol, rtol=args.rtol)
            print(f'L={L:2d} parity={"PASS" if ok else "FAIL"} max_abs={max_abs:.6g} rmse={rmse:.6g} '
                  f'finite={bool(torch.isfinite(fused).all())}')
            if not ok:
                failures += 1
            del fused, reference, delta
            torch.cuda.empty_cache()
            def baseline():
                w = ops.decode_weight(packed, args.n, args.m)
                return (gt @ w).float()
            _, base_ms, base_peak = timed(baseline, args.repeats)
            torch.cuda.empty_cache()
            _, fused_ms, fused_peak = timed(lambda: ops.backward_packed_fused_small(packed, gt, args.n, args.m), args.repeats)
            print(f'      dense={base_ms:.3f} ms, {base_peak:.1f} MiB peak | '
                  f'fused={fused_ms:.3f} ms, {fused_peak:.1f} MiB peak | '
                  f'speedup={base_ms / fused_ms:.2f}x')
        except (RuntimeError, AssertionError) as exc:
            failures += 1
            print(f'L={L:2d} ERROR: {exc}', file=sys.stderr)
    print('RESULT:', 'PASS' if not failures else f'FAIL ({failures} token cases)')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
