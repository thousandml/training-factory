"""Offline test of GCMP on full-precision vLLM layers (no GPU, no vLLM, no kernels).

Fakes: vLLM's UnquantizedLinearMethod and a few layer classes, and the licensed
my_qlinear ops (gcmp_prepare_coef = identity, gcmp_delta = a fixed linear map
of the coefficients). Checks name mapping into fused layers, attach, apply,
multi-adapter slots, kind mismatches and swapping.

    python -m pytest tests/test_gcmp_dense.py -q
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

LIB = Path(__file__).resolve().parents[1] / "lib" / "cq"


# ---------------------------------------------------------------- fakes
_LIB = None


def _fake_ops():
    global _LIB
    if _LIB is not None:
        return _LIB
    # delta[t, j] = sum_i x[t, i] * coef_flat[j % n]  -> enough to see which coefficients land where
    lib = _LIB = torch.library.Library("my_qlinear", "FRAGMENT")   # kept alive: ops vanish with it
    lib.define("gcmp_prepare_coef(Tensor c) -> Tensor")
    lib.define("gcmp_delta(Tensor x, Tensor c) -> Tensor")
    lib.define("gcmp_delta_multi(Tensor x, Tensor ct, Tensor slots) -> Tensor")
    lib.impl("gcmp_prepare_coef", lambda c: c.clone(), "CPU")

    def delta(x, c):
        flat = c.reshape(-1)
        return x.float().sum(-1, keepdim=True) * flat.unsqueeze(0)
    lib.impl("gcmp_delta", delta, "CPU")

    def delta_multi(x, ct, slots):
        coef = ct.transpose(1, 2)                       # [A+1, R, p+1] -> back to [A+1, R, p] layout used here
        out = []
        for t in range(x.shape[0]):
            out.append(delta(x[t:t + 1], coef[int(slots[t])]))
        return torch.cat(out, 0)
    lib.impl("gcmp_delta_multi", delta_multi, "CPU")
    return lib


class UnquantizedLinearMethod:
    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        return x @ layer.weight.t()


class _Layer(torch.nn.Module):
    def __init__(self, prefix, in_f, sizes, tp=1):
        super().__init__()
        self.prefix, self.input_size, self.input_size_per_partition = prefix, in_f, in_f
        self.output_partition_sizes, self.tp_size = list(sizes), tp
        self.weight = torch.nn.Parameter(torch.zeros(sum(sizes), in_f), requires_grad=False)
        self.quant_method = UnquantizedLinearMethod()


def _install_fake_vllm():
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.linear"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.model_executor.layers.linear"].UnquantizedLinearMethod = UnquantizedLinearMethod


def _load_modules():
    pkg = types.ModuleType("cq")
    pkg.__path__ = [str(LIB)]
    sys.modules["cq"] = pkg
    mods = {}
    for name in ("gcmp_adapter", "gcmp_dense"):
        spec = importlib.util.spec_from_file_location(f"cq.{name}", LIB / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"cq.{name}"] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods["gcmp_adapter"], mods["gcmp_dense"]


P = 4   # toy p


def _adapter(tmp, name, value, kind="dense", prefix="base_model.model.model.language_model"):
    """A toy adapter on layer 0: q (8 rows), v (4 rows), up (8 rows), down (6 rows)."""
    d = Path(tmp) / name
    d.mkdir()
    mods, tensors = {}, {}
    for proj, rows, in_f in (("self_attn.q_proj", 8, 6), ("self_attn.v_proj", 4, 6),
                             ("mlp.up_proj", 8, 6), ("mlp.down_proj", 6, 10)):
        mname = f"{prefix}.layers.0.{proj}"
        shape = [-(-rows // P), P]
        key = f"{mname}.gcmp_coef"
        mods[mname] = {"key": key, "cq_record": None if kind == "dense" else f"rec.{proj}",
                       "row_start": 0, "row_end": rows, "in_features": in_f, "out_features": rows,
                       "coef_shape": shape}
        tensors[key] = torch.full(shape, float(value))
    cfg = {"adapter_type": "cq_gcmp", "format_version": 1, "base_kind": kind, "scale": 1.0, "modules": mods}
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    save_file(tensors, str(d / "adapter_model.safetensors"))
    return d


def _model(tp=1):
    """vLLM-style layers for decoder layer 0 (q 8, k 4, v 4 rows; gate/up 8 each)."""
    pre = "language_model.model.layers.0"
    return {
        "qkv": _Layer(f"{pre}.self_attn.qkv_proj", 6, [8, 4, 4], tp),
        "o": _Layer(f"{pre}.self_attn.o_proj", 6, [6], tp),
        "gate_up": _Layer(f"{pre}.mlp.gate_up_proj", 6, [8, 8], tp),
        "down": _Layer(f"{pre}.mlp.down_proj", 10, [6], tp),
        "vision": _Layer("visual.blocks.0.attn.qkv", 6, [18], tp),
    }


@pytest.fixture()
def env(monkeypatch):
    _fake_ops()
    _install_fake_vllm()
    for k in ("CQ_GCMP_ADAPTER", "CQ_GCMP_ADAPTERS", "CQ_GCMP_DEFAULT_ADAPTER", "CQ_GCMP_ADAPTER_SCALE"):
        monkeypatch.delenv(k, raising=False)
    sys.modules["vllm.model_executor.layers.linear"].UnquantizedLinearMethod = UnquantizedLinearMethod
    UnquantizedLinearMethod.process_weights_after_loading = lambda self, layer: None
    UnquantizedLinearMethod.apply = lambda self, layer, x, bias=None: x @ layer.weight.t()
    return monkeypatch


def _load_all(m):
    for layer in m.values():
        layer.quant_method.process_weights_after_loading(layer)


def test_dense_record_names(env):
    g, _ = _load_modules()
    assert g.dense_record("base_model.model.model.language_model.layers.3.self_attn.q_proj") == "layers.3.self_attn.q_proj"
    assert g.dense_record("language_model.model.layers.3.self_attn.qkv_proj") == "layers.3.self_attn.qkv_proj"
    assert g.dense_record("model.layers.12.mlp.down_proj") == "layers.12.mlp.down_proj"
    assert g.dense_record("visual.blocks.0.attn.qkv") is None


def test_single_dense_adapter(env, tmp_path):
    env.setenv("CQ_GCMP_ADAPTER", str(_adapter(tmp_path, "a", 1.0)))
    g, dense = _load_modules()
    dense.install()
    m = _model()
    _load_all(m)
    # q rows 0:8 and v rows 12:16 of qkv get a delta; k rows 8:12 stay zero
    x = torch.ones(2, 6)
    out = m["qkv"].quant_method.apply(m["qkv"], x)
    assert out.shape == (2, 16)
    assert torch.all(out[:, :8] == 6) and torch.all(out[:, 8:12] == 0) and torch.all(out[:, 12:] == 6)
    # gate_up: only up (rows 8:16)
    out = m["gate_up"].quant_method.apply(m["gate_up"], x)
    assert torch.all(out[:, :8] == 0) and torch.all(out[:, 8:] == 6)
    # down: whole layer, o_proj and vision untouched
    assert torch.all(m["down"].quant_method.apply(m["down"], torch.ones(2, 10)) == 10)
    assert torch.all(m["o"].quant_method.apply(m["o"], x) == 0)
    assert getattr(m["vision"].quant_method, "_cq_gcmp_plan", None) is None
    assert g.attached_count() == 4


def test_multi_dense_adapters_and_swap(env, tmp_path):
    a, b = _adapter(tmp_path, "a", 1.0), _adapter(tmp_path, "b", 2.0)
    env.setenv("CQ_GCMP_ADAPTERS", f"a={a},b={b}")
    g, dense = _load_modules()
    dense.install()
    m = _model()
    _load_all(m)
    layer = m["down"]
    g.fill_token_slots(["a", "b", None], [1, 1, 1])
    out = layer.quant_method.apply(layer, torch.ones(3, 10))
    assert out[0, 0] == 10 and out[1, 0] == 20 and out[2, 0] == 0      # a, b, base
    c = _adapter(tmp_path, "c", 3.0)
    assert g.swap_adapter(torch.nn.ModuleDict(m), c, name="b") == 3      # layers: qkv, gate_up, down
    g.fill_token_slots(["b"], [1])
    assert layer.quant_method.apply(layer, torch.ones(1, 10))[0, 0] == 30


def test_cq_adapter_does_not_attach_to_dense_model(env, tmp_path):
    env.setenv("CQ_GCMP_ADAPTER", str(_adapter(tmp_path, "a", 1.0, kind="cq")))
    g, dense = _load_modules()
    dense.install()
    m = _model()
    _load_all(m)
    assert g.attached_count() == 0
    assert torch.all(m["down"].quant_method.apply(m["down"], torch.ones(1, 10)) == 0)


def test_mixed_kinds_refused(env, tmp_path):
    a, b = _adapter(tmp_path, "a", 1.0), _adapter(tmp_path, "b", 1.0, kind="cq")
    env.setenv("CQ_GCMP_ADAPTERS", f"a={a},b={b}")
    g, dense = _load_modules()
    with pytest.raises(RuntimeError, match="same kind"):
        g.active_registry()


def test_tensor_parallel_refused(env, tmp_path):
    env.setenv("CQ_GCMP_ADAPTER", str(_adapter(tmp_path, "a", 1.0)))
    g, dense = _load_modules()
    dense.install()
    m = _model(tp=2)
    with pytest.raises(RuntimeError, match="tensor-parallel-size 1"):
        _load_all(m)


def test_no_adapters_leaves_vllm_untouched(env):
    _, dense = _load_modules()
    orig = UnquantizedLinearMethod.apply
    dense.install()
    assert UnquantizedLinearMethod.apply is orig
