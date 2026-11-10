"""GCMP adapter plug-in for CQ vLLM layers.

This is the public integration layer. The adapter math runs only inside the
licensed ``my_qlinear::gcmp_*`` operators in the compiled ``cq_kernels``
extension. The frozen CQ weights are never modified; each CQ linear layer
computes ``W x`` exactly as before and adds ``dW x``.

One adapter for every request
-----------------------------
    CQ_GCMP_ADAPTER=/path/to/adapter vllm serve <cq-model> --quantization cq

``CQ_GCMP_ADAPTER_SCALE`` optionally overrides the trained scale.

Several adapters, chosen per request
------------------------------------
    CQ_GCMP_ADAPTERS="support=/adapters/support,sql=/adapters/sql" \
        vllm serve <cq-model> --quantization cq

(or ``CQ_GCMP_ADAPTERS=/path/adapters.json`` holding ``{"name": "path", ...}``).
Requests pick an adapter either by model name, like vLLM LoRA adapters::

    {"model": "sql", "messages": [...]}

or explicitly with ``"vllm_xargs": {"gcmp_adapter": "sql"}`` (``SamplingParams
(extra_args={"gcmp_adapter": "sql"})`` for an offline ``vllm.LLM``). Requests
that name no adapter use the base model, or ``CQ_GCMP_DEFAULT_ADAPTER`` if set.
Requests for different adapters run together in the same batch; ``/v1/models``
lists the adapters, and prefix caching keeps their KV blocks apart.

Adapters are keyed by CQ manifest record name plus a row range inside that
record, so fused runtime layers (qkv_proj, gate_up_proj, ...) and legacy split
records both receive the right slices. Adapters may cover different
projections; a projection an adapter does not train gets a zero delta.

Swapping at runtime
-------------------
``swap_adapter(model, path, name=...)`` copies another adapter's coefficients
into the existing device buffers in place (``path=None`` unplugs by zeroing).
Buffer addresses don't change, so compiled graphs and CUDA graphs stay valid.
The new adapter must fit the projections attached at startup. For an offline
``vllm.LLM``, use ``swap_adapter_llm(llm, path, name=...)``. ``name`` selects the
slot in multi-adapter mode and is ignored in single-adapter mode.
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
ENV_MULTI = "CQ_GCMP_ADAPTERS"
ENV_DEFAULT = "CQ_GCMP_DEFAULT_ADAPTER"
REQUEST_KEY = "gcmp_adapter"
MAX_TOKENS = int(os.environ.get("CQ_GCMP_MAX_TOKENS", str(1 << 17)))


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
        if cfg.get("base_kind", "cq") != "cq":
            raise RuntimeError(
                f"CQ-GCMP: {self.path} was trained on a non-CQ base (base_kind={cfg.get('base_kind')!r}, "
                "e.g. Unsloth NF4); this plug-in serves adapters trained on CQ checkpoints only"
            )
        self.config = cfg
        self.scale = float(cfg.get("scale", 1.0) if scale is None else scale)
        self.tensors = load_file(str(self.path / WEIGHTS_NAME))
        self.by_record: dict[str, list[AdapterEntry]] = {}
        for name, spec in cfg["modules"].items():
            if spec.get("cq_record") is None:
                raise RuntimeError(f"CQ-GCMP: adapter module {name!r} has no cq_record (not trained on a CQ checkpoint)")
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


def _resolve_adapter_path(path) -> Path:
    p = Path(os.path.expanduser(str(path)))
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path), allow_patterns=[CONFIG_NAME, WEIGHTS_NAME]))


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


def _prepare(coef: torch.Tensor, device) -> torch.Tensor:
    """Trained coefficients -> runtime form used by gcmp_delta (licensed op)."""
    coef = coef.to(device=device, dtype=torch.float32).contiguous()
    return torch.ops.my_qlinear.gcmp_prepare_coef(coef).contiguous()



# ---------------------------------------------------------------- registry
def adapter_specs():
    """Parse the environment without loading anything.

    Returns ``(specs, default_name, multi)`` where ``specs`` is a list of
    ``(name, path)``; ``([], None, False)`` when no adapter is configured.
    """
    single = os.environ.get(ENV_PATH, "").strip()
    multi = os.environ.get(ENV_MULTI, "").strip()
    if single and multi:
        raise RuntimeError(f"CQ-GCMP: set either {ENV_PATH} or {ENV_MULTI}, not both")
    if single:
        return [("default", single)], "default", False
    if not multi:
        return [], None, False
    if multi.endswith(".json") and Path(os.path.expanduser(multi)).is_file():
        mapping = json.loads(Path(os.path.expanduser(multi)).read_text())
        specs = [(str(k), str(v)) for k, v in mapping.items()]
    else:
        specs = []
        for item in multi.replace("\n", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise RuntimeError(f"CQ-GCMP: {ENV_MULTI} entries must be name=path, got {item!r}")
            name, path = item.split("=", 1)
            specs.append((name.strip(), path.strip()))
    names = [n for n, _ in specs]
    if not specs or any(not n for n in names) or len(set(names)) != len(names):
        raise RuntimeError(f"CQ-GCMP: {ENV_MULTI} needs unique, non-empty adapter names: {names}")
    default = os.environ.get(ENV_DEFAULT, "").strip() or None
    if default is not None and default not in names:
        raise RuntimeError(f"CQ-GCMP: {ENV_DEFAULT}={default!r} is not one of {names}")
    return specs, default, True


def adapter_names():
    specs, _, multi = adapter_specs()
    return [n for n, _ in specs] if multi else []


class AdapterRegistry:
    """Adapters in slot order. Slot 0 is always 'no adapter'."""

    def __init__(self, specs, default, multi, manifest=None):
        scale = os.environ.get(ENV_SCALE) if not multi else None
        self.multi = multi
        self.names = [n for n, _ in specs]
        self.paths = {n: p for n, p in specs}
        self.stores = [AdapterStore(p, float(scale) if scale else None) for _, p in specs]
        if manifest is not None:
            for st in self.stores:
                st.validate_against_manifest(manifest)
        self.slot_of = {n: i + 1 for i, n in enumerate(self.names)}
        self.default_slot = self.slot_of[default] if default else 0

    def slot_for(self, name) -> int:
        if name is None or name == "":
            return self.default_slot
        try:
            return self.slot_of[str(name)]
        except KeyError:
            raise ValueError(f"CQ-GCMP: unknown adapter {name!r}; available: {self.names}") from None


_registry: AdapterRegistry | None = None
_registry_loaded = False
_lock = threading.Lock()
_attached = 0
_token_slots: torch.Tensor | None = None


def active_registry(manifest=None) -> AdapterRegistry | None:
    """The process-wide adapters from the environment (None if unset)."""
    global _registry, _registry_loaded
    with _lock:
        if not _registry_loaded:
            specs, default, multi = adapter_specs()
            if specs:
                _registry = AdapterRegistry(specs, default, multi, manifest)
                for name, st in zip(_registry.names, _registry.stores):
                    label = f"{name} = {st.path}" if multi else str(st.path)
                    print(f"[CQ-GCMP] adapter {label} ({len(st.config['modules'])} projections, "
                          f"scale={st.scale})", flush=True)
                if multi:
                    default_name = _registry.names[_registry.default_slot - 1] if _registry.default_slot else "base model"
                    print(f"[CQ-GCMP] {len(_registry.names)} adapters, selected per request; "
                          f"requests without one use: {default_name}", flush=True)
            _registry_loaded = True
        return _registry


def active_store(manifest=None) -> AdapterStore | None:
    """First configured adapter (kept for callers of the single-adapter API)."""
    reg = active_registry(manifest)
    return reg.stores[0] if reg else None


def _resolve_layout(reg: AdapterRegistry, slots, input_size: int):
    """Union of every adapter's projections on this layer, in output order.

    Returns ``[(out_start, out_end, record, row_start, row_end, coef_shape)]``.
    """
    layout = {}
    for st in reg.stores:
        for out_start, out_end, e in _plan_entries(st, slots, input_size):
            key = (out_start, out_end, e.record, e.row_start, e.row_end)
            if key in layout and layout[key] != e.coef_shape:
                raise RuntimeError(f"CQ-GCMP: adapters disagree on the shape of {e.record!r}"
                                   f"[{e.row_start}:{e.row_end}]: {layout[key]} vs {e.coef_shape}")
            layout[key] = e.coef_shape
    ordered = sorted((k + (v,) for k, v in layout.items()), key=lambda t: (t[0], t[1]))
    for a, b in zip(ordered, ordered[1:]):
        if b[0] < a[1]:
            raise RuntimeError(f"CQ-GCMP: adapters split {a[2]!r} into overlapping row ranges "
                               f"{a[3]}:{a[4]} and {b[3]}:{b[4]}; adapters served together must use the same targets")
    return ordered


def _coefficients(store: AdapterStore | None, layout, used=None) -> torch.Tensor:
    """Trained coefficients of one adapter for a layer layout (zeros where absent)."""
    parts = []
    for _, _, record, row_start, row_end, shape in layout:
        match = [] if store is None else [e for e in store.by_record.get(record, ())
                                          if e.row_start == row_start and e.row_end == row_end]
        if len(match) > 1 or (match and match[0].coef_shape != shape):
            raise RuntimeError(f"CQ-GCMP: adapter does not fit {record!r}[{row_start}:{row_end}] {shape}")
        if match:
            parts.append(store.coefficient(match[0]))
            if used is not None:
                used.add(match[0].module)
        else:
            parts.append(torch.zeros(shape, dtype=torch.float32))
    return torch.cat(parts, 0)


def _flat_plan(layout):
    flat, offset = [], 0
    for out_start, out_end, _, _, _, shape in layout:
        flat.append((out_start, out_end, offset * shape[1]))
        offset += shape[0]
    return tuple(flat)


def _slot_buffer(device) -> torch.Tensor:
    """Per-token adapter slots, shared by every layer, filled before each step."""
    global _token_slots
    if _token_slots is None or _token_slots.device != torch.device(device):
        _token_slots = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=device)
    return _token_slots


@torch.no_grad()
def attach(layer: torch.nn.Module, method, slots, input_size: int, output_size: int, device) -> None:
    """Attach adapter coefficients to a CQ layer after its weights are loaded.

    ``slots`` lists ``(cq_record, output_offset, record_rows)`` for each CQ
    record backing this runtime layer, in output order.
    """
    global _attached
    method._cq_gcmp_plan = None
    reg = active_registry()
    if reg is None:
        return
    layout = _resolve_layout(reg, slots, input_size)
    if not layout:
        return
    ps = {shape[1] for *_, shape in layout}
    if len(ps) != 1:
        raise RuntimeError(f"CQ-GCMP: inconsistent p {ps} for one layer input")
    if reg.multi:
        stack = [_prepare(_coefficients(None, layout), device)]
        stack += [_prepare(_coefficients(st, layout), device) for st in reg.stores]
        coef_t = torch.stack(stack, 0).transpose(1, 2).contiguous()      # [A+1, p+1, R]
        _set_buffer(layer, "cq_gcmp_coef_t", coef_t)
        _set_buffer(layer, "cq_gcmp_slots", _slot_buffer(device))
        install_runner_hook()
    else:
        _set_buffer(layer, "cq_gcmp_coef", _prepare(_coefficients(reg.stores[0], layout), device))
    method._cq_gcmp_plan = _flat_plan(layout)
    method._cq_gcmp_layout = tuple(layout)
    method._cq_gcmp_multi = reg.multi
    method._cq_gcmp_output_size = int(output_size)
    _attached += len(layout)
    if os.environ.get("CQ_DEBUG_DISPATCH", "0") == "1":
        print(f"[CQ-GCMP] attached {[(r, a, b) for _, _, r, a, b, _ in layout]} (total {_attached})", flush=True)


def _set_buffer(layer, name, value):
    if hasattr(layer, name):
        setattr(layer, name, value)
    else:
        layer.register_buffer(name, value, persistent=False)


def add_delta(method, layer: torch.nn.Module, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Return ``out + dW x`` for every adapter slice attached to ``layer``."""
    plan = getattr(method, "_cq_gcmp_plan", None)
    if not plan:
        return out
    flat_x = x.reshape(-1, x.shape[-1])
    if getattr(method, "_cq_gcmp_multi", False):
        token_slots = layer.cq_gcmp_slots[: flat_x.shape[0]]
        d = torch.ops.my_qlinear.gcmp_delta_multi(flat_x, layer.cq_gcmp_coef_t, token_slots)
    else:
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


# ---------------------------------------------------------------- per-request routing
def request_adapter(sampling_params) -> str | None:
    extra = getattr(sampling_params, "extra_args", None) or {}
    name = extra.get(REQUEST_KEY)
    return None if name in (None, "") else str(name)


def fill_token_slots(req_adapters, num_scheduled_tokens) -> int:
    """Write each scheduled token's adapter slot into the shared device buffer.

    ``req_adapters``: adapter name (or None) per request, in batch order;
    ``num_scheduled_tokens``: tokens per request, same order. Returns the
    number of tokens written. Padding tokens after them keep stale but valid
    slots; their outputs are discarded.
    """
    import numpy as np
    reg = _registry
    if reg is None or not reg.multi or _token_slots is None:
        return 0
    per_req = np.fromiter((reg.slot_for(n) for n in req_adapters), dtype=np.int64, count=len(req_adapters))
    counts = np.asarray(num_scheduled_tokens[: len(req_adapters)], dtype=np.int64)
    tokens = np.repeat(per_req, counts)
    n = int(tokens.shape[0])
    if n > _token_slots.shape[0]:
        raise RuntimeError(f"CQ-GCMP: {n} tokens in one step exceeds CQ_GCMP_MAX_TOKENS={_token_slots.shape[0]}")
    if n:
        _token_slots[:n].copy_(torch.from_numpy(tokens))
    return n


_runner_hooked = False


def install_runner_hook() -> None:
    """Fill the per-token adapter slots right after vLLM prepares each batch."""
    global _runner_hooked
    if _runner_hooked:
        return
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:  # offline tests without vLLM
        print(f"[CQ-GCMP] vLLM model runner not found ({exc}); per-request routing disabled", flush=True)
        return
    original = GPUModelRunner._prepare_inputs

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens, *args, **kwargs):
        result = original(self, scheduler_output, num_scheduled_tokens, *args, **kwargs)
        num_reqs = self.input_batch.num_reqs
        names = []
        for req_id in self.input_batch.req_ids[:num_reqs]:
            state = self.requests.get(req_id)
            names.append(request_adapter(getattr(state, "sampling_params", None)))
        fill_token_slots(names, num_scheduled_tokens)
        return result

    GPUModelRunner._prepare_inputs = _prepare_inputs
    _runner_hooked = True


def install_api_routing() -> None:
    """API-server side: model-name routing, validation, model list, prefix-cache keys.

    Safe to call in every process; does nothing without multi-adapter config.
    """
    names = adapter_names()
    if not names:
        return
    specs, _, _ = adapter_specs()
    paths = dict(specs)

    def _route(request, params):
        if params is None or not hasattr(params, "extra_args"):
            return params
        extra = dict(params.extra_args or {})
        name = extra.get(REQUEST_KEY)
        model = getattr(request, "model", None)
        if name in (None, "") and model in paths:
            name = model
        if name not in (None, ""):
            if name not in paths:
                raise ValueError(f"Unknown GCMP adapter {name!r}; available: {names}")
            extra[REQUEST_KEY] = name
            params.extra_args = extra
        return params

    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        for cls in (ChatCompletionRequest, CompletionRequest):
            if getattr(cls.to_sampling_params, "_cq_gcmp", False):
                continue
            orig = cls.to_sampling_params

            def to_sampling_params(self, *a, _orig=orig, **k):
                return _route(self, _orig(self, *a, **k))
            to_sampling_params._cq_gcmp = True
            cls.to_sampling_params = to_sampling_params
    except Exception as exc:
        print(f"[CQ-GCMP] request routing not installed: {exc!r}", flush=True)

    try:
        from vllm.entrypoints.openai.engine.protocol import ModelCard, ModelPermission
        from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
        if not getattr(OpenAIModelRegistry.is_base_model, "_cq_gcmp", False):
            orig_is_base = OpenAIModelRegistry.is_base_model
            orig_list = OpenAIModelRegistry.show_available_models

            def is_base_model(self, model_name, _orig=orig_is_base):
                return model_name in paths or _orig(self, model_name)

            async def show_available_models(self, _orig=orig_list):
                models = await _orig(self)
                base = self.base_model_paths[0].name
                models.data.extend(ModelCard(id=n, root=str(paths[n]), parent=base,
                                             max_model_len=self.model_config.max_model_len,
                                             permission=[ModelPermission()]) for n in names)
                return models
            is_base_model._cq_gcmp = True
            OpenAIModelRegistry.is_base_model = is_base_model
            OpenAIModelRegistry.show_available_models = show_available_models
    except Exception as exc:
        print(f"[CQ-GCMP] adapter model names not registered: {exc!r}", flush=True)

    try:
        import vllm.v1.core.kv_cache_utils as kvu
        if not getattr(kvu._gen_lora_extra_hash_keys, "_cq_gcmp", False):
            orig_keys = kvu._gen_lora_extra_hash_keys
            orig_need = kvu.need_extra_keys

            def _gen_lora_extra_hash_keys(request, _orig=orig_keys):
                keys = _orig(request)
                name = request_adapter(getattr(request, "sampling_params", None))
                return keys + [f"cq-gcmp:{name}"] if name else keys

            def need_extra_keys(request, _orig=orig_need):
                return _orig(request) or request_adapter(getattr(request, "sampling_params", None)) is not None
            _gen_lora_extra_hash_keys._cq_gcmp = True
            kvu._gen_lora_extra_hash_keys = _gen_lora_extra_hash_keys
            kvu.need_extra_keys = need_extra_keys
    except Exception as exc:
        print(f"[CQ-GCMP] prefix-cache keys not extended ({exc!r}); "
              "run with --no-enable-prefix-caching when serving several adapters", flush=True)


# ---------------------------------------------------------------- swapping
@torch.no_grad()
def swap_adapter(model: torch.nn.Module, path=None, scale: float | None = None, name: str | None = None) -> int:
    """In-place swap (or unplug with path=None) on an already-attached model."""
    store = None if path is None else AdapterStore(path, scale)
    reg = _registry
    slot = None
    if reg is not None and reg.multi:
        if name is None:
            raise ValueError(f"CQ-GCMP: name= is required with several adapters; one of {reg.names}")
        slot = reg.slot_of.get(name)
        if slot is None:
            raise ValueError(f"CQ-GCMP: unknown adapter {name!r}; available: {reg.names}")
    swapped, used = 0, set()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        layout = getattr(method, "_cq_gcmp_layout", None)
        if not layout:
            continue
        coef = _coefficients(store, layout, used)
        if slot is None:
            module.cq_gcmp_coef.copy_(_prepare(coef, module.cq_gcmp_coef.device))
        else:
            module.cq_gcmp_coef_t[slot].copy_(_prepare(coef, module.cq_gcmp_coef_t.device).t())
        swapped += 1
    if store is not None:
        unused = sorted(set(store.config["modules"]) - used)
        if unused and swapped:
            print(f"[CQ-GCMP] swap: {len(unused)} adapter projections have no attached layer on this worker "
                  f"(other pipeline stage, or not attached at startup): {unused[:3]}", flush=True)
    if store is not None and slot is not None:
        reg.stores[slot - 1] = store
        reg.paths[name] = str(store.path)
    return swapped


def _worker_swap(worker, path, scale, name=None):
    model = worker.get_model() if hasattr(worker, "get_model") else worker.model_runner.model
    return swap_adapter(model, path, scale, name)


def swap_adapter_llm(llm, path=None, scale: float | None = None, name: str | None = None):
    """Swap on every vLLM worker of an offline ``vllm.LLM`` (best effort API)."""
    return llm.collective_rpc(_worker_swap, args=(path, scale, name))
