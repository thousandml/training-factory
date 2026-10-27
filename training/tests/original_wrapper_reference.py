"""Layout/matmul reference extracted from the original user-supplied wrappers.

Registration and package imports deliberately omitted. kernel_module is injected
by tests, so no real extension is loaded here.
"""
import torch
from torch.nn import functional as F
kernel_module = None

def swizzle_polar_input_uint4(x_vec: torch.Tensor) -> torch.Tensor:
    """
    Swizzle 1024-element polar blocks for kernel accesses of:

        input4[block * 128 + chunk * 32 + lane]

    Original block layout:
        [lane=32][pair=16][half=2]

    Output block layout:
        [chunk=4][lane=32][pair_in_chunk=4][half=2]

    Each [pair_in_chunk=4][half=2] region is one uint4.
    """
    if x_vec.ndim < 1:
        raise ValueError('x_vec must have at least one dimension')
    K = x_vec.shape[-1]
    if K % 1024 != 0:
        raise ValueError(f'x_vec last dimension must be divisible by 1024, got {K}')
    x_half = x_vec.to(dtype=torch.float16)
    original_shape = x_half.shape
    leading_shape = original_shape[:-1]
    num_blocks = K // 1024
    leading_dims = len(leading_shape)
    x = x_half.reshape(*leading_shape, num_blocks, 32, 4, 4, 2)
    x = x.permute(*range(leading_dims), leading_dims, leading_dims + 2, leading_dims + 1, leading_dims + 3, leading_dims + 4)
    return x.contiguous().reshape(original_shape)

def swizzle_mma_a(x: torch.Tensor):
    """
    x: [L, K], L <= 16

    Returns a 2-D tensor with the same logical shape [16, K],
    but its physical element ordering is:

        [kTile][row_half][lane][4 fp16]
    """
    if x.dim() != 2:
        raise ValueError(f'expected [L,K], got {tuple(x.shape)}')
    L, K = x.shape
    assert L <= 16
    assert K % 16 == 0
    x = x.to(torch.float16)
    if L < 16:
        padded = torch.zeros((16, K), dtype=x.dtype, device=x.device)
        padded[:L] = x
        x = padded
    kTiles = K // 16
    x = x.reshape(16, kTiles, 4, 4)
    x = x.reshape(2, 8, kTiles, 4, 4)
    x = x.permute(2, 0, 1, 3, 4)
    x = x.reshape(kTiles, 2, 32, 4)
    return x.contiguous().reshape(16, K)

def forward_pass_cuda(packed_all: torch.Tensor, x_vec: torch.Tensor, N: int, M: int) -> torch.Tensor:
    L = x_vec.shape[0]
    B = packed_all.shape[0]
    N_pad_1024 = (N + 1023) // 1024 * 1024
    assert N_pad_1024 % 1024 == 0, 'N_pad_1024 must be multiple of 1024'
    pad_n = N_pad_1024 - N
    if pad_n > 0:
        x_vec = F.pad(x_vec, (0, pad_n), value=0.0)
    if L <= 16:
        if L <= 4:
            XY_out = torch.zeros((L, M), dtype=torch.float32, device=x_vec.device)
            x_vec_swizzled = swizzle_polar_input_uint4(x_vec)
            kernel_module.fused_matmul_small_cuda_wrapper(XY_out, packed_all, x_vec_swizzled)
            return XY_out
        else:
            XY_out = torch.zeros((16, M), dtype=torch.float32, device=x_vec.device)
            x_vec_swizzled = swizzle_mma_a(x_vec)
            kernel_module.fused_matmul_cuda_wrapper(XY_out, packed_all, x_vec_swizzled)
            return XY_out[:L, :]
    else:
        x_vec_swizzled = swizzle_polar_input_uint4(x_vec)
        Y_out = torch.zeros((M, N_pad_1024), dtype=torch.float16, device=packed_all.device)
        kernel_module.polar_encode_u8_cuda_wrapper(packed_all, Y_out)
        return (x_vec_swizzled.to(Y_out.dtype) @ Y_out.T).float()
