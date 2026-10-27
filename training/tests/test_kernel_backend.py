"""Contract tests against the user's original wrapper, using CPU mock kernels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pytest
import kernel_backend as kb
import original_wrapper_reference as original


class MockRaw:
    def __init__(self,w):
        self.w=w
        self.calls=[]

    def fused_matmul_small_cuda_wrapper(self,out,packed,x):
        self.calls.append(('small',x.clone()))
        out.copy_((x @ self.w.T).float())

    def fused_matmul_cuda_wrapper(self,out,packed,x):
        self.calls.append(('mma',x.clone()))
        # Undo MMA's physical A packing; decoder W uses POLAR column order.
        k=x.shape[-1]
        logical=x.reshape(k//16,2,8,4,4).permute(1,2,0,3,4).contiguous().reshape(16,k)
        out.copy_((kb.polar_swizzle(logical) @ self.w.T).float())

    def polar_encode_u8_cuda_wrapper(self,packed,out):
        self.calls.append(('decode',None))
        out.copy_(self.w)

    def dl_fft_forward_cuda_wrapper(self,x,dl):
        self.calls.append(('fft',None))
        return x.clone(),x.float().sum(-1,keepdim=True)

    def dl_ifft_reverse_cuda_wrapper(self,*args):
        self.calls.append(('ifft',args[-1].clone()))
        return args[0].clone()


@pytest.mark.parametrize('rows',[1,4,5,8,16,17,32,129])
def test_original_dispatch_and_layout(rows,monkeypatch):
    torch.manual_seed(3)
    w=torch.randn(12,1024,dtype=torch.float16)/32
    module=MockRaw(w)
    backend=kb.OriginalKernels(module)
    monkeypatch.setattr(original,'kernel_module',module)
    packed=torch.zeros(12,520,dtype=torch.uint8)
    x=torch.randn(rows,1024,dtype=torch.float16)
    expected=original.forward_pass_cuda(packed,x,1024,12)
    reference_calls=module.calls[:]
    module.calls=[]
    actual=backend.forward_pass(packed,x,1024,12)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert [name for name,_ in module.calls]==[name for name,_ in reference_calls]
    for (_,a),(_,b) in zip(module.calls,reference_calls):
        if a is not None:
            torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_raw_no_bias_empty_tensor():
    module=MockRaw(torch.zeros(8,1024,dtype=torch.float16))
    backend=kb.OriginalKernels(module)
    x=torch.zeros(3,8)
    backend.reverse_ifft(x,torch.ones(1,8),torch.tensor(1.),torch.zeros(3,1),torch.tensor(0.),None,True)
    assert module.calls[-1][0]=='ifft' and module.calls[-1][1].numel()==0


@pytest.mark.parametrize('has_n',[False,True])
def test_registered_fft_signature(monkeypatch,has_n):
    calls=[]
    monkeypatch.setattr(torch.ops.my_qlinear,'forward_fft',lambda *args: calls.append(args),raising=False)
    backend=kb.RegisteredOps(fft_has_n=has_n)
    x=torch.zeros(1,1024);dl=torch.ones(1024)
    backend.forward_fft(x,1024,dl,True)
    assert len(calls[0])==(4 if has_n else 3)


def test_registered_required_bias(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.ops.my_qlinear,'reverse_ifft',lambda *args:calls.append(args),raising=False)
    kb.RegisteredOps(bias_optional=False).reverse_ifft(torch.zeros(1,8),torch.ones(1,8),torch.tensor(1.),torch.zeros(1,1),torch.tensor(0.),None,False)
    assert calls[0][5].shape==(8,) and calls[0][5].sum()==0
