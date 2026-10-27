"""Run on licensed CUDA host after rebuilding cq.cq_kernels."""
import importlib
import torch
import torch.nn.functional as F

importlib.import_module('cq.cq_kernels')
ops = torch.ops.my_qlinear


def swizzle(x):
    k = x.shape[-1]
    return x.reshape(-1, k // 1024, 32, 4, 4, 2).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, k)


def unswizzle(x):
    k = x.shape[-1]
    return x.reshape(-1, k // 1024, 4, 32, 4, 2).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, k)


def check(n, m=2048):
    npad = ((n + 1023) // 1024) * 1024
    packed = torch.randint(0, 256, (n * m // 1024, 520), dtype=torch.uint8, device='cuda')
    weight = ops.decode_weight(packed, n, m)
    assert weight.shape == (m, npad), (n, weight.shape)
    for L in (1, 4, 8, 17):
        x = torch.randn(L, n, dtype=torch.float16, device='cuda')
        xp = swizzle(F.pad(x, (0, npad - n)))
        native = ops.forward_pass(packed, x, n, m).float()
        ref = (xp @ weight.T).float()
        assert torch.allclose(native, ref, atol=0.2, rtol=0.01), (n, L, 'forward parity', (native-ref).abs().max().item())
        gt = torch.randn(L, m, dtype=torch.float16, device='cuda')
        expected = (gt @ weight).float()
        for tile in (4, 64, 128, 256):
            actual = ops.backward_packed_tiled(packed, gt, n, m, tile)
            assert actual.shape == (L, npad), (n, L, tile, actual.shape)
            err = (actual - expected).abs()
            # Half GEMM in tiles followed by FP32 accumulation; allow normal
            # FP16 GEMM rounding, not a layout-sized error.
            assert torch.allclose(actual, expected, atol=0.35, rtol=0.02), (n, L, tile, err.max().item(), err.square().mean().sqrt().item())
            print(f'N={n} L={L} tile={tile} parity max={err.max().item():.5f} rmse={err.square().mean().sqrt().item():.5f}')
        # Independent PyTorch autograd reference for swizzle/pad/GEMM.
        xref = x.float().detach().requires_grad_(True)
        yref = swizzle(F.pad(xref, (0, npad - n))) @ weight.float().T
        (yref * gt.float()).sum().backward()
        # Independent FP32 analytical gradient.
        # Match the precision of the PyTorch autograd reference.
        expected_fp32 = gt.float() @ weight.float()

        analytic = unswizzle(expected_fp32)[:, :n]
        diff = (xref.grad - analytic).abs()

        print(
            f'N={n} L={L} autograd '
            f'max={diff.max().item():.6f} '
            f'rmse={diff.square().mean().sqrt().item():.6f}'
        )

        assert torch.allclose(
            xref.grad,
            analytic,
            atol=1e-3,
            rtol=1e-3,
        ), (n, L, diff.max().item())
    print('PASS', n)


if __name__ == '__main__':
    torch.manual_seed(42)
    for n in (1024, 4096, 5376, 6144):
        check(n)
    print('ALL PADDED BACKWARD TESTS PASSED')
