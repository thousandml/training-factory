"""Pick GPUs that fit the selected model, precision and job (train or serve).

The weight size is the size of the checkpoint that will actually be loaded (the CQ
repo for cq, the prequantized NF4 repo for nf4, the original model for bf16),
read from Hugging Face or from a local folder. Standard library only.
"""
import json
import os
import urllib.request

# SkyPilot accelerator name -> memory (GB). Ordered small to large.
CATALOG = [
    ('L4', 24), ('A10G', 24),
    ('A100', 40),
    ('L40S', 48), ('A40', 48),
    ('A100-80GB', 80), ('H100', 80),
    ('H200', 141),
]
USABLE = 0.90                       # leave room for the CUDA context and fragmentation
BIG = ('A100-80GB', 'H100')          # used several at a time when one GPU is not enough
GB = 1e9


def _hf_size(repo):
    """Total bytes of a repo's .safetensors files, or None if unknown."""
    req = urllib.request.Request(f'https://huggingface.co/api/models/{repo}?blobs=true')
    if os.environ.get('HF_TOKEN'):
        req.add_header('Authorization', 'Bearer ' + os.environ['HF_TOKEN'])
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            info = json.load(r)
    except Exception:  # noqa: BLE001 - offline, private, renamed: fall back
        return None
    sizes = [s.get('size') or 0 for s in info.get('siblings', []) if s.get('rfilename', '').endswith('.safetensors')]
    return sum(sizes) or None


def _local_size(path):
    total = 0
    for root, _, files in os.walk(os.path.expanduser(path)):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files if f.endswith('.safetensors'))
    return total or None


def weight_bytes(prec, ckpt, base):
    """Bytes of weights on the GPU for this precision, or None if unknown."""
    src = ckpt
    size = _local_size(src) if os.path.isdir(os.path.expanduser(src)) else _hf_size(src)
    if size and not (prec == 'nf4' and ckpt == base):
        return size
    # NF4 quantized on load from a full-precision checkpoint: ~0.3x the 16-bit size
    full = _local_size(base) if os.path.isdir(os.path.expanduser(base)) else _hf_size(base)
    if full and prec == 'nf4':
        return int(full * 0.30)
    return size


def need_gb(weights, job):
    """GPU memory a job needs: weights plus working memory.

    train: activations (batch 2, length ~1k, gradient checkpointing), adapter
           gradients and optimizer state.
    serve: a KV cache large enough for useful concurrency.
    """
    # Calibrated on Qwen3.8-27B, batch 2 x length 512, gradient checkpointing:
    # BF16 (55.6 GB) peaked at 51.7-53.7 GiB in training; CQ (18 GB) at 22-24 GiB.
    w = weights / GB
    extra = max(6.0, 0.12 * w) if job == 'train' else max(8.0, 0.15 * w)
    return w + extra


def choose(prec, ckpt, base, job):
    """-> (accelerator spec, pipeline-parallel size, explanation) or raises ValueError."""
    weights = weight_bytes(prec, ckpt, base)
    if weights is None:
        return None, 1, 'model size unknown'
    need = need_gb(weights, job)
    fits = [n for n, mem in CATALOG if mem * USABLE >= need]
    why = f'{weights / GB:.1f} GB of weights, ~{need:.0f} GB needed to {job}'
    if fits:
        # Up to four of the smallest that fit; SkyPilot launches the cheapest available.
        smallest = min(mem for n, mem in CATALOG if n in fits)
        picks = [n for n, mem in CATALOG if n in fits and mem <= max(smallest, 80)][:4]
        return ','.join(f'{n}:1' for n in picks), 1, why
    if job == 'train':
        raise ValueError(f'{why}: too big for one GPU, and training uses one GPU. '
                         'Use --precision cq or nf4, a smaller model, or pass --gpus yourself.')
    for n_gpus in (2, 4, 8):                       # serve: split layers across GPUs
        if n_gpus * 80 * USABLE >= need:
            return ','.join(f'{n}:{n_gpus}' for n in BIG), n_gpus, why + f', split across {n_gpus} GPUs'
    raise ValueError(f'{why}: needs more than 8 GPUs. Use --precision cq or nf4.')
