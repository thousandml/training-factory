"""ThousandML CQ vLLM quantization plugin.

The Python integration is intentionally public. Proprietary CQ execution and
authorization live in the compiled ``cq_kernels`` extension. GGUF is not
supported.
"""

from .cq import CqConfig, CqLinearMethod, configure_cq_pipeline_partition
from . import runtime_ops as _runtime_ops  # registers FakeTensor metadata
from .gcmp_adapter import swap_adapter, swap_adapter_llm  # GCMP adapter


def register() -> None:
    """Register CQ with vLLM before EngineCore workers are spawned."""
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
