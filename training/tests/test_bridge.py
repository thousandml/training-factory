"""CPU tests use a fake decoder; CUDA parity requires real packed records."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from cq_linear import CQLinear, swizzle, unswizzle


@pytest.fixture(autouse=True)
def fake_native_operators(monkeypatch):
    """CPU arithmetic oracle, NOT a substitute for real CUDA dispatch tests."""
    def fft(x,n,dl,flag):
        signed = (x*dl).float().reshape(-1,n//2,2).contiguous()
        t = torch.view_as_real(torch.fft.fft(torch.view_as_complex(signed),norm='ortho')).reshape(-1,n).half()
        return t,x.float().sum(-1,keepdim=True)
    def matmul(packed,x,n,m):
        return torch.nn.functional.linear(swizzle(x),torch.ops.my_qlinear.decode_weight(packed,n,m)).float()
    def reverse(y,dr,std,sums,u,bias,flag):
        m=dr.numel()
        y=torch.view_as_real(torch.fft.ifft(torch.view_as_complex(y.reshape(-1,m//2,2).contiguous()),norm='ortho')).reshape(-1,m)
        out=y*dr*std+sums*u
        return out if bias is None else out+bias
    for name,fn in [('forward_fft',fft),('forward_pass',matmul),('reverse_ifft',reverse)]:
        monkeypatch.setattr(torch.ops.my_qlinear,name,fn,raising=False)


def test_swizzle_inverse():
    x = torch.arange(2 * 2048).reshape(2, 2048)
    assert torch.equal(unswizzle(swizzle(x)), x)


def test_decoder_reused_in_backward(monkeypatch):
    calls=[]
    w=torch.randn(8,1024,dtype=torch.float16)/32
    def decode(*args):
        calls.append(1)
        return w
    monkeypatch.setattr(torch.ops.my_qlinear,'decode_weight',decode,raising=False)
    rec={'packed_all':torch.zeros(8,520,dtype=torch.uint8),'u_W':torch.tensor(.01),'std_W':torch.tensor(.8)}
    layer=CQLinear(1024,8,rec,device='cpu')
    x=torch.randn(3,1024,dtype=torch.float16,requires_grad=True)
    y=layer(x)
    assert len(calls)==1  # CPU mock of native forward needs decoder
    y.sum().backward()
    assert len(calls)==2  # backward decodes again rather than retaining weight
    assert torch.isfinite(x.grad).all()


def test_peft_adapter_gradient_and_export(monkeypatch, tmp_path):
    from peft import LoraConfig, get_peft_model
    w = torch.randn(8, 1024, dtype=torch.float16) / 32
    monkeypatch.setattr(torch.ops.my_qlinear, 'decode_weight', lambda *a: w, raising=False)
    rec = {'packed_all': torch.zeros(8, 520, dtype=torch.uint8), 'u_W': torch.tensor(.01), 'std_W': torch.tensor(.8)}
    base = CQLinear(1024, 8, rec, device='cpu')
    model = torch.nn.Sequential(base)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['0']))
    x = torch.randn(2, 1024, requires_grad=True)
    model(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    grads = [p.grad for n, p in model.named_parameters() if 'lora_B' in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)
    assert not base.packed_all.requires_grad and not base.weight.requires_grad
    model.save_pretrained(tmp_path)
    from safetensors.torch import load_file
    saved = load_file(str(tmp_path / 'adapter_model.safetensors'))
    assert saved and all('lora_' in k for k in saved)


def test_fused_output_slices_and_adapters(monkeypatch):
    from cq_linear import CQOutputSlice
    from peft import LoraConfig, get_peft_model
    w = torch.randn(12, 1024, dtype=torch.float16) / 32
    monkeypatch.setattr(torch.ops.my_qlinear, 'decode_weight', lambda *a: w, raising=False)
    rec = {'packed_all':torch.zeros(12,520,dtype=torch.uint8),'u_W':torch.tensor(.02),'std_W':torch.tensor(.7)}
    fused = CQLinear(1024,12,rec,device='cpu')
    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = CQOutputSlice(fused,0,8)
            self.k_proj = CQOutputSlice(fused,8,10)
            self.v_proj = CQOutputSlice(fused,10,12)
        def forward(self,x):
            return torch.cat([self.q_proj(x),self.k_proj(x),self.v_proj(x)],-1)
    x = torch.randn(2,1024,requires_grad=True)
    base = Projections()
    torch.testing.assert_close(base(x),fused(x))
    assert base.q_proj.base_cq is base.v_proj.base_cq
    adapted = get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj']))
    torch.testing.assert_close(adapted(x),fused(x))  # zero initialized B
    adapted(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    grads = [p.grad for n,p in adapted.named_parameters() if 'lora_B' in n]
    assert len(grads) == 3 and all(g is not None and g.abs().sum() > 0 for g in grads)


def test_pack_legacy_record():
    from load_model import pack_record
    b = 8
    rec = {k:torch.full((b,width),i,dtype=torch.uint8) for i,(k,width) in enumerate([
        ('SigRec1_select_packed',7),('SigRec2_select_packed',12),
        ('SigRec3_select_packed',29),('SigRec4_select_packed',83)],1)}
    rec.update(X567_packed=torch.arange(3*b*128).reshape(3*b,128).to(torch.uint8),u_W=torch.tensor(0.),std_W=torch.tensor(1.))
    packed = pack_record(rec,[8,1024])['packed_all']
    assert packed.shape == (8,520)
    assert packed[:,7].sum() == 0 and packed[:,49:52].sum() == 0 and packed[:,135].sum() == 0
    assert torch.equal(packed[:,136:264],rec['X567_packed'][:b])


@pytest.mark.parametrize('rows',[1,4,5,16,17,32])
def test_native_adjoint_vs_autograd(monkeypatch,rows):
    from cq_linear import reference_forward
    torch.manual_seed(12)
    w=torch.randn(12,1024,dtype=torch.float16)/32
    monkeypatch.setattr(torch.ops.my_qlinear,'decode_weight',lambda *a:w,raising=False)
    rec={'packed_all':torch.zeros(12,520,dtype=torch.uint8),'u_W':torch.tensor(.02),'std_W':torch.tensor(.7)}
    layer=CQLinear(1024,12,rec,device='cpu')
    x=torch.randn(rows,1024,dtype=torch.float16,requires_grad=True)
    x2=x.detach().clone().requires_grad_()
    g=torch.randn(rows,12,dtype=torch.float16)
    y=layer(x)
    ref=reference_forward(layer,x2)
    torch.testing.assert_close(y,ref,atol=0,rtol=0)
    y.backward(g)
    ref.backward(g)
    torch.testing.assert_close(x.grad,x2.grad,atol=.003,rtol=.03)


def test_mean_correction_gradient(monkeypatch):
    monkeypatch.setattr(torch.ops.my_qlinear,'decode_weight',lambda *a:torch.zeros(8,1024,dtype=torch.float16),raising=False)
    rec={'packed_all':torch.zeros(8,520,dtype=torch.uint8),'u_W':torch.tensor(.125),'std_W':torch.tensor(0.)}
    layer=CQLinear(1024,8,rec,device='cpu')
    x=torch.ones(2,1024,dtype=torch.float16,requires_grad=True)
    layer(x).sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x))
