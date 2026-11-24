"""GCMP adapters on full-precision (BF16/FP16) models in vLLM.

The CQ path attaches adapters inside the CQ linear methods. A model served
without ``--quantization`` uses vLLM's ``UnquantizedLinearMethod`` instead, so
this module adds the same two hooks there:

  * after a layer's weights are loaded, attach the adapter coefficients to it
    (same buffers, same per-token slots, same ``swap_adapter``), and
  * in ``apply``, add ``dW x`` to the frozen ``W x``.

Adapters trained on a full-precision base (``base_kind: "dense"``, e.g. with
``train_unsloth_16bit_gcmp.py``) are matched to vLLM layers by decoder-layer
name, so HF names (``...layers.3.self_attn.q_proj``) find vLLM's fused layers
(``...layers.3.self_attn.qkv_proj``) and the right row slice inside them.

    CQ_GCMP_ADAPTERS="support=./support,sql=./sql" vllm serve Qwen/Qwen3.8-27B

Limits: tensor parallel size 1 (pipeline parallel is fine); the adapter math
runs in the licensed ``my_qlinear::gcmp_*`` operators, as on CQ.
"""
from __future__ import annotations

import torch

from . import gcmp_adapter as _g

# vLLM fused layer name -> the HF projections it holds, in output order
FUSED = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

_installed = False


def _slots(layer):
    """[(record, output_offset, rows)] for a vLLM linear layer, or None."""
    record = _g.dense_record(getattr(layer, "prefix", "") or "")
    if record is None:
        return None
    head, _, last = record.rpartition(".")
    sizes = [int(s) for s in (getattr(layer, "output_partition_sizes", None)
                              or [getattr(layer, "output_size_per_partition", getattr(layer, "output_size", 0))])]
    names = FUSED.get(last)
    if names is None:
        if len(sizes) != 1:
            return None            # an unknown fused layer: leave it alone
        names = (last,)
    elif len(sizes) < len(names):
        return None
    slots, offset = [], 0
    for name, rows in zip(names, sizes):   # extra trailing sizes (padding) are ignored
        slots.append((f"{head}.{name}", offset, rows))
        offset += rows
    return slots


def _device(layer):
    for p in layer.parameters(recurse=False):
        return p.device
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def _attach_dense(method, layer) -> None:
    reg = _g.active_registry()
    if reg is None or reg.kind != "dense":
        return
    slots = _slots(layer)
    if not slots:
        return
    tp = int(getattr(layer, "tp_size", 1) or 1)
    if tp > 1:
        raise RuntimeError("CQ-GCMP: GCMP adapters on a full-precision model support "
                           "--tensor-parallel-size 1 only (pipeline parallel is fine)")
    input_size = int(getattr(layer, "input_size_per_partition", getattr(layer, "input_size")))
    _g.attach(layer, method, slots, input_size, sum(r for *_, r in slots), _device(layer), kind="dense")


def install() -> None:
    """Patch vLLM's unquantized linear method and the model loader. Idempotent."""
    global _installed
    if _installed:
        return
    specs, _, _ = _g.adapter_specs()
    if not specs:
        return                    # no adapters configured: leave vLLM untouched
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except ImportError as exc:
        print(f"[CQ-GCMP] full-precision support not installed ({exc!r})", flush=True)
        return

    orig_post = UnquantizedLinearMethod.process_weights_after_loading
    orig_apply = UnquantizedLinearMethod.apply

    def process_weights_after_loading(self, layer, _orig=orig_post):
        _orig(self, layer)
        _attach_dense(self, layer)

    def apply(self, layer, x, bias=None, _orig=orig_apply):
        out = _orig(self, layer, x, bias)
        if getattr(self, "_cq_gcmp_plan", None):
            out = _g.add_delta(self, layer, x, out)
        return out

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    _install_load_check()
    _installed = True


def _install_load_check() -> None:
    """Fail at startup, not silently, when adapters are set but attach nowhere."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        return
    if getattr(GPUModelRunner.load_model, "_cq_gcmp_check", False):
        return
    orig = GPUModelRunner.load_model

    def load_model(self, *a, _orig=orig, **k):
        result = _orig(self, *a, **k)
        reg = _g.active_registry()
        if reg is not None and _g.attached_count() == 0:
            hint = ("dense adapters need a full-precision model (no --quantization cq)"
                    if reg.kind == "dense" else
                    "CQ adapters need a CQ model served with --quantization cq")
            raise RuntimeError(f"CQ-GCMP: adapters are configured but none attached to this model; {hint}. "
                               "Check that the adapter was trained for this model.")
        if reg is not None:
            print(f"[CQ-GCMP] {_g.attached_count()} projections attached ({reg.kind} adapters)", flush=True)
        return result

    load_model._cq_gcmp_check = True
    GPUModelRunner.load_model = load_model
