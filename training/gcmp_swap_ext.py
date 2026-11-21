"""vLLM worker extension that lets a running OpenAI server swap GCMP adapters.

Start vLLM with
    VLLM_SERVER_DEV_MODE=1 vllm serve ... --worker-extension-cls gcmp_swap_ext.GCMPSwap
(with this file's folder on PYTHONPATH), then swap an adapter slot in place:
    curl -X POST localhost:8000/collective_rpc -H 'Content-Type: application/json' \
         -d '{"method": "gcmp_swap", "args": ["tenant-a", "/adapters/new-tenant"]}'

The swap copies the new coefficients into the slot's existing GPU buffers, so no
memory is allocated and CUDA graphs stay valid. The slot keeps its name.
"""


class GCMPSwap:
    def gcmp_swap(self, name, path=None, scale=None):
        from cq.gcmp_adapter import swap_adapter
        model = self.get_model() if hasattr(self, 'get_model') else self.model_runner.model
        return swap_adapter(model, path, scale, name)
