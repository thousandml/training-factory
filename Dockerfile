
FROM vllm/vllm-openai:v0.28.0-ubuntu2404

# Inference-only dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install the CQ inference plugin and packaged native extensions.
COPY inference /tmp/inference

RUN python3 -m pip install --no-cache-dir --no-deps /tmp/inference

# Validate the components that are actually needed for CQ inference.
RUN python3 - <<'PY'
import sys
import torch
import transformers
import vllm
import cq
import cq.runtime_ops
import cq.cq_kernels

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
print("CQ:", cq.__file__)
print("CQ runtime:", cq.runtime_ops.__file__)
print("CQ kernels:", cq.cq_kernels.__file__)
PY

# Check that CQ is registered with vLLM.
RUN python3 - <<'PY'
from vllm.model_executor.layers.quantization import get_quantization_config

config = get_quantization_config("cq")
print("CQ quantization registered:", config)
PY

# Enable WSL2 pinned-memory compatibility.
ENV VLLM_WSL2_ENABLE_PIN_MEMORY=1

EXPOSE 8000

# Allow: docker run IMAGE vllm serve MODEL --quantization cq
ENTRYPOINT []
