"""ThousandML CQ vLLM quantization plugin.

The Python integration is intentionally public. Proprietary CQ execution and
authorization live in the compiled ``cq_kernels`` extension. GGUF is not
supported.
"""

from .cq import CqConfig, CqLinearMethod, configure_cq_pipeline_partition
from . import runtime_ops as _runtime_ops  # registers FakeTensor metadata
from .gcmp_adapter import swap_adapter, swap_adapter_llm  # GCMP adapter


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

    path = os.environ.get("CQ_GCMP_ADAPTER", "").strip()
    if not path or os.environ.get("CQ_GCMP_KEEP_CACHE_ROOT") == "1":
        return
    h = hashlib.sha256(path.encode())
    cfg = Path(os.path.expanduser(path)) / "adapter_config.json"
    if cfg.is_file():
        h.update(cfg.read_bytes())
    h.update(b"cq-gcmp-shim-v2")
    base = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "vllm")
    if os.path.basename(base).startswith("cq-gcmp-"):
        return  # already redirected (inherited by a child process)
    os.environ["VLLM_CACHE_ROOT"] = os.path.join(base, f"cq-gcmp-{h.hexdigest()[:16]}")


def register() -> None:
    """Register CQ with vLLM before EngineCore workers are spawned."""
    _separate_compile_cache_for_adapter()
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
