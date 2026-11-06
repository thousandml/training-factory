#!/usr/bin/env python3
"""GPU benchmark: GCMP adapter vs LoRA, forward + backward, per projection shape.

Measures only the adapter path (the frozen CQ layer is identical for both),
with the same dtypes as training: bf16 activations; LoRA in fp32 params under
bf16 autocast; GCMP via the licensed native ops (cuFFT inside cq_kernels).

    python benchmark_gcmp_vs_lora.py                    # A40 defaults
    python benchmark_gcmp_vs_lora.py --tokens 4096 --tf32
    python benchmark_gcmp_vs_lora.py --shapes 5120x5120,17408x5120

Prints per-shape GCMP time (split into transform vs GEMM), LoRA time per rank,
the LoRA rank at which LoRA becomes slower than GCMP, and a dense bf16 matmul
of the full frozen projection (forward + backward) for scale. Private tool.
"""
import argparse
import importlib
import json

import torch


def parse_shapes(text):
    out = []
    for item in text.split(','):
        L, M = item.lower().split('x')
        out.append((int(L), int(M)))
    return out


def timer(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    import time
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel-module', default='cq.cq_kernels')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tokens', type=int, default=1024, help='tokens per microbatch (batch * seq)')
    ap.add_argument('--shapes', default='5120x5120,5120x1024,5120x17408,17408x5120',
                    help='comma list of LxM (in x out) projection shapes')
    ap.add_argument('--ranks', default='16,32,64,128,256,512')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--tf32', action='store_true', help='allow TF32 for GCMP fp32 GEMMs')
    ap.add_argument('--json', help='write results here')
    a = ap.parse_args()

    from _cq_training_license import require_valid
    require_valid()
    importlib.import_module(a.kernel_module)  # registers my_qlinear::gcmp_*
    import gcmp
    if not gcmp.native_ops_available():
        raise SystemExit(f'{a.kernel_module} lacks my_qlinear::gcmp_* ops; rebuild cq_kernels')
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(dev)}  tokens={a.tokens}  tf32={a.tf32}')
    ranks = [int(r) for r in a.ranks.split(',')]
    T = a.tokens
    results = []
    for L, M in parse_shapes(a.shapes):
        g = gcmp.geometry(L)
        p = g['p']
        R = -(-M // p)
        x = torch.randn(T, L, device=dev, dtype=torch.bfloat16, requires_grad=True)
        coef = (torch.randn(R, p, device=dev) * 1e-3).requires_grad_(True)
        gy = torch.randn(T, M, device=dev, dtype=torch.bfloat16)

        def gcmp_step():
            x.grad = coef.grad = None
            y = gcmp.gcmp_delta(x, coef, M, native=True)
            y.backward(gy.float())

        xd = x.detach()
        Phi = torch.ops.my_qlinear.gcmp_features(xd, p)
        spec = gcmp.prepare_coef(coef.detach())

        def transforms():  # forward, backward recompute, adjoint
            P = torch.ops.my_qlinear.gcmp_features(xd, p)
            torch.ops.my_qlinear.gcmp_features(xd, p)
            torch.ops.my_qlinear.gcmp_features_adjoint(P, L)

        def gemms():  # forward Phi@spec^T, dspec, dPhi
            Y = Phi @ spec.t()
            G = Y
            G.reshape(-1, R).t() @ Phi.reshape(-1, p)
            G @ spec

        t_gcmp = timer(gcmp_step, dev, a.warmup, a.iters)
        t_tr = timer(transforms, dev, a.warmup, a.iters)
        t_mm = timer(gemms, dev, a.warmup, a.iters)

        lora = {}
        for r in ranks:
            A = (torch.randn(r, L, device=dev) / L ** 0.5).requires_grad_(True)
            B = torch.zeros(M, r, device=dev, requires_grad=True)

            def lora_step():
                x.grad = A.grad = B.grad = None
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    y = (x @ A.t()) @ B.t()
                y.backward(gy)

            lora[r] = timer(lora_step, dev, a.warmup, a.iters)
            del A, B

        Wd = torch.randn(M, L, device=dev, dtype=torch.bfloat16)

        def dense_step():  # frozen projection cost scale: fwd + recompute + dX
            y = x.detach() @ Wd.t()
            x.detach() @ Wd.t()
            gy @ Wd
        t_dense = timer(dense_step, dev, a.warmup, a.iters)

        cross = None
        rs = sorted(lora)
        for r0, r1 in zip(rs, rs[1:]):
            if lora[r0] < t_gcmp <= lora[r1]:
                cross = r0 + (r1 - r0) * (t_gcmp - lora[r0]) / (lora[r1] - lora[r0])
        if cross is None and lora[rs[0]] >= t_gcmp:
            cross = f'<={rs[0]}'
        if cross is None:
            cross = f'>{rs[-1]}'
        mac_break_even = R * p * p / (L + M)
        row = {'L': L, 'M': M, 'p': p, 'gcmp_ms': t_gcmp, 'gcmp_transform_ms': t_tr,
               'gcmp_gemm_ms': t_mm, 'lora_ms': lora, 'dense_frozen_ms': t_dense,
               'measured_break_even_rank': cross, 'mac_break_even_rank': mac_break_even,
               'gcmp_params': R * p, 'lora32_params': 32 * (L + M)}
        results.append(row)
        lr = ' '.join(f'r{r}:{t:.3f}' for r, t in lora.items())
        ce = f'{cross:.0f}' if isinstance(cross, float) else cross
        print(f'{L}x{M} p={p}: GCMP {t_gcmp:.3f} ms (transforms {t_tr:.3f}, GEMMs {t_mm:.3f}) | '
              f'LoRA {lr} | frozen dense {t_dense:.3f} ms | break-even rank: measured {ce}, '
              f'multiply-only {mac_break_even:.0f}', flush=True)
        del x, coef, gy, Phi, spec, Wd
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
