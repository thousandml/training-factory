"""GCMP adapter plug-in for CQ vLLM layers.

This is the public integration layer. The adapter math runs only inside
the licensed ``my_qlinear::gcmp_delta`` and ``gcmp_prepare_coef`` operators in the
compiled ``cq_kernels`` extension. The frozen CQ weights are never modified;
each CQ linear layer computes ``W x`` exactly as before and adds ``dW x``.

Plugging an adapter
-------------------
Set ``CQ_GCMP_ADAPTER`` to an adapter directory (or Hugging Face repo id)
produced by the CQ trainer, before vLLM starts:

    CQ_GCMP_ADAPTER=/path/to/outputs/cq-gcmp vllm serve <cq-model> --quantization cq

``CQ_GCMP_ADAPTER_SCALE`` optionally overrides the trained scale.

Adapters are keyed by CQ manifest record name plus a row range inside that
record, so fused runtime layers (qkv_proj, gate_up_proj, ...) and legacy
split records both receive the right slices.

Swapping at runtime
-------------------
``swap_adapter(model, path)`` copies another adapter's coefficients into the
existing device buffers in place (``path=None`` unplugs by zeroing them).
Buffer addresses don't change, so compiled graphs and CUDA graphs stay valid.
The new adapter must cover the same projections with the same shapes. For an
offline ``vllm.LLM``, use ``swap_adapter_llm(llm, path)``.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

ADAPTER_TYPE = "cq_gcmp"
FORMAT_VERSION = 1
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"
ENV_PATH = "CQ_GCMP_ADAPTER"
ENV_SCALE = "CQ_GCMP_ADAPTER_SCALE"


@dataclass(frozen=True)
class AdapterEntry:
    module: str
    key: str
    record: str
    row_start: int
    row_end: int
    in_features: int
    coef_shape: tuple[int, int]


class AdapterStore:
    """One adapter's config and CPU coefficients, indexed by CQ record."""

    def __init__(self, path: str | os.PathLike, scale: float | None = None):
        from safetensors.torch import load_file

        self.path = _resolve_adapter_path(path)
        cfg = json.loads((self.path / CONFIG_NAME).read_text())
        if cfg.get("adapter_type") != ADAPTER_TYPE:
            raise RuntimeError(
                f"CQ-GCMP: {self.path} is not a {ADAPTER_TYPE} adapter "
                f"(adapter_type={cfg.get('adapter_type')!r})"
            )
        if int(cfg.get("format_version", 0)) != FORMAT_VERSION:
            raise RuntimeError(f"CQ-GCMP: unsupported adapter format_version {cfg.get('format_version')!r}")
        self.config = cfg
        self.scale = float(cfg.get("scale", 1.0) if scale is None else scale)
        self.tensors = load_file(str(self.path / WEIGHTS_NAME))
        self.by_record: dict[str, list[AdapterEntry]] = {}
        for name, spec in cfg["modules"].items():
            if spec.get("cq_record") is None:
                raise RuntimeError(f"CQ-GCMP: adapter module {name!r} has no cq_record")
            entry = AdapterEntry(
                module=name,
                key=spec["key"],
                record=str(spec["cq_record"]),
                row_start=int(spec["row_start"]),
                row_end=int(spec["row_end"]),
                in_features=int(spec["in_features"]),
                coef_shape=tuple(int(v) for v in spec["coef_shape"]),
            )
            t = self.tensors.get(entry.key)
            if t is None or tuple(t.shape) != entry.coef_shape:
                raise RuntimeError(f"CQ-GCMP: tensor {entry.key!r} missing or wrong shape in {self.path}")
            p = entry.coef_shape[1]
            rows = entry.row_end - entry.row_start
            if rows <= 0 or entry.coef_shape[0] != -(-rows // p):
                raise RuntimeError(f"CQ-GCMP: {name!r} coef {entry.coef_shape} inconsistent with rows {rows}")
            self.by_record.setdefault(entry.record, []).append(entry)
        for entries in self.by_record.values():
            entries.sort(key=lambda e: e.row_start)
            for a, b in zip(entries, entries[1:]):
                if b.row_start < a.row_end:
                    raise RuntimeError(f"CQ-GCMP: overlapping adapters {a.module!r} and {b.module!r}")

    def validate_against_manifest(self, manifest) -> None:
        known = getattr(manifest, "quantized_layers", None)
        if known is None:
            return
        unknown = sorted(set(self.by_record) - set(known))
        if unknown:
            raise RuntimeError(
                "CQ-GCMP: adapter targets CQ records absent from this model's manifest "
                f"(trained for a different checkpoint?): {unknown[:5]}"
            )

    def coefficient(self, entry: AdapterEntry) -> torch.Tensor:
        return self.tensors[entry.key].float() * self.scale


_store: AdapterStore | None = None
_store_loaded = False
_lock = threading.Lock()
_attached = 0


def _resolve_adapter_path(path) -> Path:
    p = Path(os.path.expanduser(str(path)))
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path), allow_patterns=[CONFIG_NAME, WEIGHTS_NAME]))


def active_store(manifest=None) -> AdapterStore | None:
    """The process-wide adapter selected by CQ_GCMP_ADAPTER (None if unset)."""
    global _store, _store_loaded
    with _lock:
        if not _store_loaded:
            path = os.environ.get(ENV_PATH, "").strip()
            if path:
                scale = os.environ.get(ENV_SCALE)
                _store = AdapterStore(path, float(scale) if scale else None)
                if manifest is not None:
                    _store.validate_against_manifest(manifest)
                print(f"[CQ-GCMP] adapter {path} ({len(_store.config['modules'])} projections, "
                      f"scale={_store.scale})", flush=True)
            _store_loaded = True
        return _store


def _plan_entries(store: AdapterStore, slots, input_size: int):
    """[(out_start, out_end, entry)] in layer-output coordinates."""
    planned = []
    for record, offset, record_rows in slots:
        for e in store.by_record.get(record, ()):
            if e.row_end > record_rows:
                raise RuntimeError(f"CQ-GCMP: {e.module!r} rows {e.row_start}:{e.row_end} exceed record {record!r} ({record_rows} rows)")
            if e.in_features != int(input_size):
                raise RuntimeError(f"CQ-GCMP: {e.module!r} in_features {e.in_features} != layer input {input_size}")
            planned.append((offset + e.row_start, offset + e.row_end, e))
    planned.sort(key=lambda t: t[0])
    return planned


def _stack(store: AdapterStore, planned, device):
    coefs, flat, R = [], [], 0
    for out_start, out_end, e in planned:
        c = store.coefficient(e)
        flat.append((out_start, out_end, R * c.shape[1]))
        coefs.append(c)
        R += c.shape[0]
    return _prepare(torch.cat(coefs, 0), device), flat


def _prepare(coef: torch.Tensor, device) -> torch.Tensor:
    """Trained coefficients -> runtime form used by gcmp_delta (licensed op)."""
    coef = coef.to(device=device, dtype=torch.float32).contiguous()
    return torch.ops.my_qlinear.gcmp_prepare_coef(coef).contiguous()


@torch.no_grad()
def attach(layer: torch.nn.Module, method, slots, input_size: int, output_size: int, device) -> None:
    """Attach adapter coefficients to a CQ layer after its weights are loaded.

    ``slots`` lists ``(cq_record, output_offset, record_rows)`` for each CQ
    record backing this runtime layer, in output order.
    """
    global _attached
    method._cq_gcmp_plan = None
    store = active_store()
    if store is None:
        return
    planned = _plan_entries(store, slots, input_size)
    if not planned:
        return
    ps = {e.coef_shape[1] for _, _, e in planned}
    if len(ps) != 1:
        raise RuntimeError(f"CQ-GCMP: inconsistent p {ps} for one layer input")
    coef, flat = _stack(store, planned, device)
    if hasattr(layer, "cq_gcmp_coef"):
        layer.cq_gcmp_coef = coef
    else:
        layer.register_buffer("cq_gcmp_coef", coef, persistent=False)
    method._cq_gcmp_plan = tuple(flat)
    method._cq_gcmp_layout = tuple((s, e_, e.record, e.row_start, e.row_end, e.coef_shape) for s, e_, e in planned)
    method._cq_gcmp_output_size = int(output_size)
    _attached += len(planned)
    if os.environ.get("CQ_DEBUG_DISPATCH", "0") == "1":
        print(f"[CQ-GCMP] attached {[e.module for _, _, e in planned]} (total {_attached})", flush=True)


def add_delta(method, layer: torch.nn.Module, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Return ``out + dW x`` for every adapter slice attached to ``layer``."""
    plan = getattr(method, "_cq_gcmp_plan", None)
    if not plan:
        return out
    flat_x = x.reshape(-1, x.shape[-1])
    d = torch.ops.my_qlinear.gcmp_delta(flat_x, layer.cq_gcmp_coef)
    total = out.shape[-1]
    pieces, cursor = [], 0
    for out_start, out_end, flat_start in plan:
        if out_start > cursor:
            pieces.append(d.new_zeros((d.shape[0], out_start - cursor)))
        pieces.append(d[:, flat_start:flat_start + (out_end - out_start)])
        cursor = out_end
    if cursor < total:
        pieces.append(d.new_zeros((d.shape[0], total - cursor)))
    delta = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
    return out + delta.reshape(out.shape).to(out.dtype)


def attached_count() -> int:
    return _attached


@torch.no_grad()
def swap_adapter(model: torch.nn.Module, path=None, scale: float | None = None) -> int:
    """In-place swap (or unplug with path=None) on an already-attached model."""
    store = None if path is None else AdapterStore(path, scale)
    swapped = 0
    used = set()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        layout = getattr(method, "_cq_gcmp_layout", None)
        if not layout:
            continue
        if store is None:
            module.cq_gcmp_coef.zero_()
            swapped += 1
            continue
        new = []
        for out_start, out_end, record, row_start, row_end, shape in layout:
            match = [e for e in store.by_record.get(record, ()) if e.row_start == row_start and e.row_end == row_end]
            if len(match) != 1 or match[0].coef_shape != shape:
                raise RuntimeError(f"CQ-GCMP swap: new adapter lacks a matching {record!r}[{row_start}:{row_end}] {shape}")
            new.append(store.coefficient(match[0]))
            used.add(match[0].module)
        module.cq_gcmp_coef.copy_(_prepare(torch.cat(new, 0), module.cq_gcmp_coef.device))
        swapped += 1
    if store is not None:
        unused = sorted(set(store.config["modules"]) - used)
        if unused and swapped:
            print(f"[CQ-GCMP] swap: {len(unused)} adapter projections have no attached layer on this worker "
                  f"(other pipeline stage, or not attached at startup): {unused[:3]}", flush=True)
    return swapped


def _worker_swap(worker, path, scale):
    model = worker.get_model() if hasattr(worker, "get_model") else worker.model_runner.model
    return swap_adapter(model, path, scale)


def swap_adapter_llm(llm, path=None, scale: float | None = None):
    """Swap on every vLLM worker of an offline ``vllm.LLM`` (best effort API)."""
    return llm.collective_rpc(_worker_swap, args=(path, scale))
