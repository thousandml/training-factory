"""ThousandML CQ vLLM quantization plugin.

The Python integration is intentionally public. Proprietary CQ execution and
authorization live in the compiled ``cq_kernels`` extension. GGUF is not
supported.
"""

from .cq import CqConfig, CqLinearMethod, configure_cq_pipeline_partition
from . import runtime_ops as _runtime_ops  # registers FakeTensor metadata
from .gcmp_adapter import swap_adapter, swap_adapter_llm  # GCMP adapter
from . import gcmp_adapter as _gcmp_adapter


def _separate_compile_cache_for_adapter() -> None:
    """Keep torch.compile caches of adapter and no-adapter runs apart.

    vLLM keys its compile cache on the model code, not on this plugin's
    state. A graph traced without a GCMP adapter (or by an older plugin)
    would otherwise be reused when an adapter is plugged, silently skipping
    the adapter. The adapter's layout (not its values, which live in
    buffers) selects a dedicated cache root.
    """
    import hashlib
    import os
    from pathlib import Path

    single = os.environ.get("CQ_GCMP_ADAPTER", "").strip()
    multi = os.environ.get("CQ_GCMP_ADAPTERS", "").strip()
    if not (single or multi) or os.environ.get("CQ_GCMP_KEEP_CACHE_ROOT") == "1":
        return
    h = hashlib.sha256(f"{single}|{multi}".encode())
    paths = [single] if single else [item.split("=", 1)[-1].strip() for item in multi.split(",") if item.strip()]
    for path in paths:
        cfg = Path(os.path.expanduser(path)) / "adapter_config.json"
        if cfg.is_file():
            h.update(cfg.read_bytes())
    h.update(b"cq-gcmp-shim-v3-multi")
    base = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "vllm")
    if os.path.basename(base).startswith("cq-gcmp-"):
        return  # already redirected (inherited by a child process)
    os.environ["VLLM_CACHE_ROOT"] = os.path.join(base, f"cq-gcmp-{h.hexdigest()[:16]}")


def register() -> None:
    """Register CQ with vLLM before EngineCore workers are spawned."""
    _separate_compile_cache_for_adapter()
    try:
        # Several GCMP adapters: model-name routing, validation, /v1/models,
        # and prefix-cache keys per adapter (no-op without CQ_GCMP_ADAPTERS).
        _gcmp_adapter.install_api_routing()
    except Exception as exc:
        print("[CQ-GCMP] multi-adapter routing setup failed:", repr(exc), flush=True)
        raise
    try:
        configure_cq_pipeline_partition()
    except Exception as exc:
        # Auto partitioning is an optimization; configuration gets another
        # chance during CqConfig initialization if early inspection is unavailable.
        print("[CQ PP] early auto-partition skipped:", repr(exc), flush=True)

    from vllm import ModelRegistry

    if "cq" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "cq",
            "vllm_add_dummy_model.cq:cq",
        )


__all__ = ["CqConfig", "CqLinearMethod", "configure_cq_pipeline_partition", "register",
           "swap_adapter", "swap_adapter_llm"]
