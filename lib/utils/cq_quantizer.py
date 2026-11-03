
from __future__ import annotations

CQ_QUANTIZER_BUILD = "cq-v1"
print(
    f"[CQ] Loaded quantizer build {CQ_QUANTIZER_BUILD} from {__file__}",
    flush=True,
)


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Dict, List

import gc
import os
import pathlib
import re
import torch

from safetensors import safe_open
from huggingface_hub import snapshot_download

try:
    from . import _cq_core
except ImportError as _cq_core_import_error:
    _cq_core = None
    _CQ_CORE_IMPORT_ERROR = _cq_core_import_error
else:
    _CQ_CORE_IMPORT_ERROR = None

def _require_cq_core():
    if _cq_core is None:
        raise RuntimeError(
            "Private CQ native core is not installed. Build native/cq_core and "
            "copy _cq_core*.so into lib/utils/. Original import error: "
            f"{_CQ_CORE_IMPORT_ERROR!r}"
        )
    return _cq_core

import torch.nn as nn
from transformers import PreTrainedModel
#from cqdm import cqdm

import torch.nn.functional as F
import json



# Qwen3.8 / Qwen4Exp N-gram CQ geometry.  The source checkpoint stores
# [rows, 160] embedding shards.  CQ quantizes each group of 1024 rows as one
# linear matrix W=[160, 1024], so an embedding lookup is W @ one_hot(local_row).
CQ_NGRAM_CHUNK_ROWS = 1024
CQ_NGRAM_EMBED_DIM = 160
CQ_NGRAM_POLAR_BLOCKS_PER_CHUNK = (CQ_NGRAM_CHUNK_ROWS * CQ_NGRAM_EMBED_DIM) // 1024
assert CQ_NGRAM_POLAR_BLOCKS_PER_CHUNK == 160


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shard_path: Path
    shape: tuple[int, ...]
    dtype: str
    numel: int


def _load_state_dict_from_model(
    model_name_or_path: str,
    map_location: str = "cpu",
) -> "StreamingSafeTensorState":
    """Return the streaming safetensor index used by layer-by-layer CQ.

    ``map_location`` is retained for compatibility with the older loader API;
    StreamingSafeTensorState always indexes/checkpoints on CPU and moves only
    the currently requested tensor to the requested quantization device.
    """
    if map_location != "cpu":
        raise ValueError(
            "Streaming CQ checkpoint indexing must use map_location='cpu'"
        )
    return StreamingSafeTensorState(model_name_or_path)


class StreamingSafeTensorState:
    """
    Metadata index + one-tensor-at-a-time loader.

    This intentionally does NOT hold all tensors in memory.
    """

    def __init__(self, model_name_or_path: str):
        if os.path.exists(model_name_or_path):
            self.model_path = Path(model_name_or_path)
        else:
            self.model_path = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    ignore_patterns=[
                        "*.msgpack",
                        "*.h5",
                        "tf_model*",
                        "flax_model*",
                    ],
                    token=os.environ.get("HF_TOKEN", True),
                )
            )

        self.entries: dict[str, TensorEntry] = {}
        self._build_index()

    def _build_index(self) -> None:
        shard_paths = sorted(self.model_path.glob("*.safetensors"))

        if not shard_paths:
            raise FileNotFoundError(
                f"No .safetensors files found in {self.model_path}"
            )

        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor_slice = f.get_slice(name)
                    shape = tuple(tensor_slice.get_shape())
                    dtype = str(tensor_slice.get_dtype())

                    numel = 1
                    for dim in shape:
                        numel *= dim

                    self.entries[name] = TensorEntry(
                        name=name,
                        shard_path=shard_path,
                        shape=shape,
                        dtype=dtype,
                        numel=numel,
                    )

    def keys(self):
        return self.entries.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.get_tensor(key, device="cpu", dtype=None)

    def get_entry(self, key: str) -> TensorEntry:
        return self.entries[key]

    def iter_entries(self) -> Iterator[TensorEntry]:
        return iter(self.entries.values())

    def get_tensor_rows(
        self,
        key: str,
        row_start: int,
        row_end: int,
        *,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Read only a contiguous row range from a rank-2 safetensor."""
        entry = self.entries[key]
        if len(entry.shape) != 2:
            raise RuntimeError(
                f"Row-sliced CQ source must be rank-2: {key}, shape={entry.shape}"
            )
        if not (0 <= int(row_start) < int(row_end) <= int(entry.shape[0])):
            raise IndexError(
                f"Bad row slice [{row_start}:{row_end}] for {key} shape={entry.shape}"
            )

        with safe_open(entry.shard_path, framework="pt", device="cpu") as f:
            sl = f.get_slice(key)
            tensor = sl[int(row_start):int(row_end)]

        if dtype is not None or device != "cpu":
            tensor = tensor.to(
                device=device,
                dtype=dtype,
                non_blocking=False,
            )
        return tensor

    def get_tensor(
        self,
        key: str,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        entry = self.entries[key]

        with safe_open(entry.shard_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)

        if dtype is not None or device != "cpu":
            tensor = tensor.to(
                device=device,
                dtype=dtype,
                non_blocking=False,
            )

        return tensor
    

@dataclass(frozen=True)
class QuantizationTarget:
    target_name: str
    state_key: str
    slice_index: Optional[int]
    source_shape: tuple[int, ...]
    matrix_shape: tuple[int, int]
    numel: int
    kind: str

    # Native-vLLM packed modules may be assembled from multiple HF checkpoint
    # tensors (q/k/v -> qkv_proj, gate/up -> gate_up_proj, etc.).  Keep the
    # legacy single-source fields above for backward compatibility and record
    # all physical sources here when a target is fused.
    source_keys: tuple[str, ...] = ()
    source_shapes: tuple[tuple[int, ...], ...] = ()

    # Chunked embedding support. For cq_embedding_chunk, source rows
    # [row_start:row_end] are transposed before CQ quantization so the runtime
    # can evaluate embedding lookup as one-hot @ W_chunk.
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    embedding_row_start: Optional[int] = None
    embedding_row_end: Optional[int] = None
    transpose_source: bool = False
    embedding_group: Optional[str] = None

    def all_source_keys(self) -> tuple[str, ...]:
        return self.source_keys or (self.state_key,)

    def all_source_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self.source_shapes or (self.source_shape,)


LINEAR_NAME_TOKENS = (
    # Llama / Mistral / common HF
    "q_proj", "k_proj", "v_proj", "o_proj",
    "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
    "qkv_proj",
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",

    # Qwen Gated-DeltaNet / linear-attention projections. Some of these are
    # intentionally narrow (<128 output features), so they must be recognized
    # explicitly rather than relying on the generic matrix-size heuristic.
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
    "in_proj_qkvz", "in_proj_ba",

    # GPT-NeoX / Pythia
    "query_key_value",
    "attention.dense",
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",

    # GPT-2 / GPT-J / Conv1D-like checkpoint names
    "c_attn",
    "c_proj",
    "c_fc",

    # Falcon / RW
    "self_attention.query_key_value",
    "self_attention.dense",
    "dense_h_to_4h",
    "dense_4h_to_h",

    # MoE / generic
    "shared_experts",
    "experts",
    ".w1", ".w2", ".w3",
    "wo", "wq", "wk", "wv",
)

EXCLUDE_NAME_TOKENS = (
    "embed",
    "embedding",
    "word_embeddings",
    "embed_in",
    "embed_out",
    "wte",
    "wpe",
    "lm_head",
    "norm",
    "layernorm",
    "layer_norm",
    "ln_f",
    "ln_",
    "rotary",
    "rope",
    "inv_freq",
    "bias",
    "router",
    "gate.weight",   # router gate, not gate_proj
)

FUSED_EXPERT_TENSOR_SUFFIXES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)


# ---------------------------------------------------------------------------
# Native-vLLM packed linear layout
# ---------------------------------------------------------------------------
#
# These rules describe the *logical runtime parameters* used by native vLLM.
# Source names are the names found in ordinary Hugging Face checkpoints.
# The converter fuses them before CQ quantization, so the resulting CQ record
# is named after the native vLLM module rather than the Transformers module.
#
# The common rules below cover Llama/Mistral/Qwen dense models and Qwen3.5's
# Gated-DeltaNet projections.  The discovery code only applies a rule when all
# required source tensors exist with compatible shapes, so unrelated models are
# unaffected.
NATIVE_VLLM_PACKED_LINEAR_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("gate_up_proj", ("gate_proj", "up_proj")),
    ("in_proj_qkvz", ("in_proj_qkv", "in_proj_z")),
    ("in_proj_ba", ("in_proj_b", "in_proj_a")),
)


def _replace_last_component(name: str, old: str, new: str) -> str | None:
    suffix = "." + old
    if name.endswith(suffix):
        return name[:-len(suffix)] + "." + new
    if name == old:
        return new
    return None


def _native_fusion_candidate(
    base_name: str,
    source_component: str,
    target_component: str,
) -> str | None:
    return _replace_last_component(base_name, source_component, target_component)


def _compatible_native_sources(entries) -> tuple[int, tuple[int, ...]] | None:
    """Return concat dimension + fused shape, or None if sources are incompatible.

    Rank-2 linear weights use [out, in] and are concatenated on dim 0.
    Rank-3 expert-major weights use [experts, out, in] and are concatenated on
    dim 1, after which each expert is quantized independently.
    """
    if not entries:
        return None
    ranks = {len(e.shape) for e in entries}
    if len(ranks) != 1:
        return None
    rank = next(iter(ranks))

    if rank == 2:
        in_features = entries[0].shape[1]
        if any(e.shape[1] != in_features for e in entries):
            return None
        return 0, (sum(int(e.shape[0]) for e in entries), int(in_features))

    if rank == 3:
        experts = entries[0].shape[0]
        in_features = entries[0].shape[2]
        if any(e.shape[0] != experts or e.shape[2] != in_features for e in entries):
            return None
        return 1, (
            int(experts),
            sum(int(e.shape[1]) for e in entries),
            int(in_features),
        )

    return None


_NGRAM_SHARD_RE = re.compile(
    r"^(?P<group>.*ngram_embedding)\.shard_(?P<idx>\d+)(?:\.weight)?$",
    re.IGNORECASE,
)


_NGRAM_PHYSICAL_SHARD_RE = re.compile(
    r"^(?P<group>.+\.ngram_embedding)\.shard_(?P<idx>\d+)\.weight$",
    re.IGNORECASE,
)


def _parse_ngram_embedding_entry(
    entry: TensorEntry,
) -> tuple[str, int] | None:
    """Recognize the exact original Qwen4Exp n-gram checkpoint layout."""
    if len(entry.shape) != 2:
        return None

    m = _NGRAM_PHYSICAL_SHARD_RE.match(str(entry.name))
    if m is None:
        return None

    rows, dim = map(int, entry.shape)
    if rows <= 0 or dim <= 0:
        return None

    return m.group("group"), int(m.group("idx"))


def _discover_ngram_embedding_targets(
    entries: list[TensorEntry],
    *,
    verbose: bool = True,
) -> tuple[list[QuantizationTarget], set[str]]:
    """Discover Qwen4Exp physical N-gram shards for 1024x160 CQ chunking.

    A checkpoint shard remains one *storage record* so we do not create
    millions of files.  Internally each record is quantized as independent
    source chunks [1024, 160], transposed to CQ matrices [M=160, N=1024].

    Runtime lookup:
        global row -> physical shard -> 1024-row chunk -> local row
        embedding = CQ([160,1024]) @ one_hot(local_row, 1024)
    """
    physical: list[tuple[str, int, TensorEntry]] = []
    for entry in entries:
        parsed = _parse_ngram_embedding_entry(entry)
        if parsed is None:
            continue
        group, shard_idx = parsed
        rows, dim = map(int, entry.shape)
        if dim != CQ_NGRAM_EMBED_DIM:
            raise RuntimeError(
                f"CQ n-gram expects embedding dim {CQ_NGRAM_EMBED_DIM}, "
                f"got {entry.name} shape={entry.shape}"
            )
        physical.append((group, shard_idx, entry))

    if not physical:
        if verbose:
            print(
                "[CQ] N-gram chunk discovery: 0 matching "
                "*.ngram_embedding.shard_N.weight tensors",
                flush=True,
            )
        return [], set()

    groups: dict[str, list[tuple[int, TensorEntry]]] = {}
    for group, shard_idx, entry in physical:
        groups.setdefault(group, []).append((shard_idx, entry))

    targets: list[QuantizationTarget] = []
    consumed: set[str] = set()

    for group, members in sorted(groups.items()):
        members.sort(key=lambda x: x[0])
        ids = [idx for idx, _ in members]
        expected = list(range(ids[0], ids[-1] + 1))
        if ids != expected:
            raise RuntimeError(
                f"CQ n-gram group {group!r}: physical shard IDs are not "
                f"contiguous: first={ids[:8]} last={ids[-8:]}"
            )

        global_row = 0
        group_params = 0
        group_chunks = 0

        if verbose:
            print(f"[CQ] N-gram chunked group: {group}", flush=True)

        for shard_idx, entry in members:
            rows, dim = map(int, entry.shape)
            num_chunks = (rows + CQ_NGRAM_CHUNK_ROWS - 1) // CQ_NGRAM_CHUNK_ROWS
            group_chunks += num_chunks
            group_params += int(entry.numel)
            consumed.add(entry.name)

            # Keep one manifest/storage record per physical checkpoint shard.
            # matrix_shape describes the *runtime CQ unit*, not the whole shard.
            target_name = f"{group}.shard_{shard_idx}"
            targets.append(
                QuantizationTarget(
                    target_name=target_name,
                    state_key=entry.name,
                    slice_index=None,
                    source_shape=tuple(entry.shape),
                    matrix_shape=(CQ_NGRAM_EMBED_DIM, CQ_NGRAM_CHUNK_ROWS),
                    numel=int(entry.numel),
                    kind="cq_ngram_chunked_shard",
                    row_start=0,
                    row_end=rows,
                    embedding_row_start=global_row,
                    embedding_row_end=global_row + rows,
                    transpose_source=True,
                    embedding_group=group,
                )
            )

            if verbose and (shard_idx < 4 or shard_idx >= ids[-1] - 3):
                print(
                    f"  shard_{shard_idx}: source={tuple(entry.shape)}, "
                    f"chunks={num_chunks:,}, CQ_unit=(160,1024), "
                    f"global_rows=[{global_row}:{global_row + rows})",
                    flush=True,
                )
            global_row += rows

        if verbose:
            print(
                f"[CQ] N-gram group summary: {len(members)} physical shards, "
                f"{group_chunks:,} x [1024,160] chunks, {global_row:,} rows, "
                f"{group_params / 1e9:.3f}B params, "
                f"raw INT4={group_params / 2 / 2**30:.2f} GiB",
                flush=True,
            )

    return targets, consumed



def discover_native_vllm_quantization_targets_from_state(
    state: "StreamingSafeTensorState",
    exclude_patterns: list[str],
    verbose: bool = True,
) -> list[QuantizationTarget]:
    """Discover CQ targets using native-vLLM packed module names.

    The checkpoint is still streamed directly from HF safetensors.  What changes
    is the *logical target layout*: tensors which native vLLM packs into one
    runtime linear are fused first and then quantized as a single CQ matrix.
    Unpacked linears and already-packed checkpoint tensors fall back to the
    existing checkpoint-driven discovery path.
    """
    entries = list(state.iter_entries())

    # HARD v5 diagnostic. If this line does not appear, this exact discovery
    # function is not the one being executed.
    rank2_entries = [e for e in entries if len(e.shape) == 2]
    ngram_key_entries = [
        e for e in rank2_entries
        if (
            "ngram" in str(e.name).lower()
            or "ple_embedding" in str(e.name).lower()
            or re.search(r"(?:^|\.)shard_\d+(?:\.weight)?$", str(e.name), re.I)
        )
    ]

    print(
        f"[CQ] NGRAM-DISCOVERY-v5 ACTIVE: "
        f"checkpoint_entries={len(entries)}, rank2={len(rank2_entries)}, "
        f"ngram/ple/shardN={len(ngram_key_entries)}",
        flush=True,
    )

    for e in ngram_key_entries[:20]:
        print(
            f"[CQ] NGRAM-CANDIDATE-v5: "
            f"{e.name} shape={tuple(e.shape)} params={e.numel:,}",
            flush=True,
        )

    by_base = {
        e.name.removesuffix(".weight"): e
        for e in entries
        if _is_checkpoint_weight_name(e.name, e.shape)
    }

    embedding_targets, embedding_consumed = _discover_ngram_embedding_targets(
        entries,
        verbose=verbose,
    )
    consumed: set[str] = set(embedding_consumed)
    targets: list[QuantizationTarget] = list(embedding_targets)

    if ngram_key_entries and not embedding_targets:
        preview = "\n".join(
            f"  {e.name} shape={tuple(e.shape)}"
            for e in ngram_key_entries[:20]
        )
        raise RuntimeError(
            "Qwen4Exp n-gram/PLE checkpoint tensors were found, but the "
            "CQ embedding detector produced zero targets. Refusing to silently "
            "continue with only the ~127B linear/MoE subset.\n"
            f"First candidate keys:\n{preview}"
        )

    if verbose and embedding_targets:
        emb_params = sum(t.numel for t in embedding_targets)
        emb_groups = len({
            t.embedding_group for t in embedding_targets
            if t.embedding_group is not None
        })
        print(
            f"[CQ] N-gram embedding CQ: {len(embedding_targets)} chunks "
            f"across {emb_groups} physical table(s), "
            f"{emb_params / 1e9:.3f}B params"
        )

    # Build packed native-vLLM targets first.
    for target_component, source_components in NATIVE_VLLM_PACKED_LINEAR_RULES:
        # Every occurrence of the first source component proposes one packed
        # target.  We then require every sibling source to exist.
        first_component = source_components[0]
        for base_name, first_entry in list(by_base.items()):
            native_base = _native_fusion_candidate(
                base_name, first_component, target_component
            )
            if native_base is None:
                continue

            prefix = native_base.removesuffix("." + target_component)
            source_bases = [
                f"{prefix}.{component}" if prefix else component
                for component in source_components
            ]
            if not all(name in by_base for name in source_bases):
                continue

            source_entries = [by_base[name] for name in source_bases]

            # These tensors were matched by an explicit native-vLLM fusion rule,
            # so they are known linear projections even when one component is
            # narrow (for example Qwen GDN in_proj_a/in_proj_b). Do NOT run the
            # generic >=128 dimension heuristic here; it would incorrectly drop
            # valid packed targets such as in_proj_ba. Keep only the exclusions
            # that are semantically meaningful for explicit fusion sources.
            def _native_source_allowed(e):
                lower = e.name.lower()
                if any(pattern.lower() in lower for pattern in exclude_patterns):
                    return False
                if any(token in lower for token in EXCLUDE_NAME_TOKENS):
                    return False
                if not _is_checkpoint_weight_name(e.name, e.shape):
                    return False
                return len(e.shape) in (2, 3)

            if not all(_native_source_allowed(e) for e in source_entries):
                if verbose:
                    print(
                        f"[CQ] native fusion skipped by explicit exclusion: "
                        f"target={native_base} sources={source_bases}",
                        flush=True,
                    )
                continue

            compatible = _compatible_native_sources(source_entries)
            if compatible is None:
                continue
            _concat_dim, fused_shape = compatible

            source_keys = tuple(e.name for e in source_entries)
            source_shapes = tuple(tuple(e.shape) for e in source_entries)
            consumed.update(source_keys)

            if len(fused_shape) == 2:
                out_features, in_features = fused_shape
                targets.append(
                    QuantizationTarget(
                        target_name=native_base,
                        state_key=source_keys[0],
                        slice_index=None,
                        source_shape=source_shapes[0],
                        matrix_shape=(int(out_features), int(in_features)),
                        numel=int(out_features) * int(in_features),
                        kind=f"native_vllm_fused_{target_component}",
                        source_keys=source_keys,
                        source_shapes=source_shapes,
                    )
                )
            else:
                num_experts, out_features, in_features = fused_shape
                expert_numel = int(out_features) * int(in_features)
                for expert_idx in range(int(num_experts)):
                    targets.append(
                        QuantizationTarget(
                            target_name=f"{native_base}.expert_{expert_idx:04d}",
                            state_key=source_keys[0],
                            slice_index=expert_idx,
                            source_shape=source_shapes[0],
                            matrix_shape=(int(out_features), int(in_features)),
                            numel=expert_numel,
                            kind=f"native_vllm_fused_expert_{target_component}",
                            source_keys=source_keys,
                            source_shapes=source_shapes,
                        )
                    )

    # Everything not consumed by a native packed module is treated exactly like
    # the old quantizer: one rank-2 tensor -> one CQ record; rank-3 expert-major
    # tensors -> one CQ record per expert.
    for entry in entries:
        if entry.name in consumed:
            continue
        if not _should_quantize_checkpoint_tensor(
            entry.name, entry.shape, exclude_patterns
        ):
            continue

        base_name = entry.name.removesuffix(".weight")
        if len(entry.shape) == 2:
            out_features, in_features = entry.shape
            targets.append(
                QuantizationTarget(
                    target_name=base_name,
                    state_key=entry.name,
                    slice_index=None,
                    source_shape=entry.shape,
                    matrix_shape=(int(out_features), int(in_features)),
                    numel=int(entry.numel),
                    kind="native_vllm_linear_2d",
                )
            )
        elif len(entry.shape) == 3:
            num_experts, out_features, in_features = entry.shape
            expert_numel = int(out_features) * int(in_features)
            for expert_idx in range(int(num_experts)):
                targets.append(
                    QuantizationTarget(
                        target_name=f"{base_name}.expert_{expert_idx:04d}",
                        state_key=entry.name,
                        slice_index=expert_idx,
                        source_shape=entry.shape,
                        matrix_shape=(int(out_features), int(in_features)),
                        numel=expert_numel,
                        kind="native_vllm_expert_3d",
                    )
                )

    # Deterministic checkpoint order makes resumability and diffs predictable.
    targets.sort(key=lambda t: (t.target_name, -1 if t.slice_index is None else t.slice_index))

    if verbose:
        total_numel = sum(t.numel for t in targets)
        fused = sum(1 for t in targets if "fused" in t.kind)
        print(
            f"[CQ] NGRAM-DISCOVERY-v5 RESULT: "
            f"embedding_targets={len(embedding_targets)}, "
            f"embedding_params={sum(t.numel for t in embedding_targets) / 1e9:.3f}B",
            flush=True,
        )
        print(f"[CQ] Native-vLLM layout: {len(targets)} CQ targets")
        print(f"[CQ] Native packed/fused targets: {fused}")
        print(f"[CQ] Target parameters: {total_numel / 1e9:.3f}B")
        print(
            f"[CQ] Raw INT4 payload estimate: "
            f"{total_numel / 2 / 2**30:.2f} GiB"
        )

    return targets


def _is_checkpoint_weight_name(
    name: str,
    shape: tuple[int, ...],
) -> bool:
    lower = name.lower()

    if lower.endswith(".weight"):
        return True

    return (
        len(shape) == 3
        and any(
            lower.endswith(suffix)
            for suffix in FUSED_EXPERT_TENSOR_SUFFIXES
        )
    )

def _matches_any_token(name: str, tokens: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(token.lower() in lower for token in tokens)


def _should_quantize_checkpoint_tensor(
    name: str,
    shape: tuple[int, ...],
    exclude_patterns: list[str],
) -> bool:
    lower = name.lower()

    if any(pattern.lower() in lower for pattern in exclude_patterns):
        return False

    if any(token in lower for token in EXCLUDE_NAME_TOKENS):
        return False

    # if not name.endswith(".weight"):
    #     return False
    
    if not _is_checkpoint_weight_name(name, shape):
        return False

    if len(shape) not in (2, 3):
        return False

    # Explicit known linear names.
    if any(token in lower for token in LINEAR_NAME_TOKENS):
        return True

    # Safe fallback:
    # Most transformer linear weights are rank-2 matrices.
    # Embeddings/norms/lm_head were already excluded above.
    if len(shape) == 2:
        rows, cols = shape

        # Avoid tiny accidental matrices.
        if rows >= 128 and cols >= 128:
            return True

    # Fused expert fallback.
    if len(shape) == 3:
        num_experts, rows, cols = shape
        if num_experts >= 1 and rows >= 128 and cols >= 128:
            return True

    return False




def discover_quantization_targets_from_state(
    state: StreamingSafeTensorState,
    exclude_patterns: list[str],
    verbose: bool = True,
) -> list[QuantizationTarget]:
    entries = list(state.iter_entries())
    embedding_targets, embedding_consumed = _discover_ngram_embedding_targets(
        entries,
        verbose=verbose,
    )
    targets: list[QuantizationTarget] = list(embedding_targets)

    for entry in entries:
        if entry.name in embedding_consumed:
            continue
        if not _should_quantize_checkpoint_tensor(
            entry.name,
            entry.shape,
            exclude_patterns,
        ):
            continue

        base_name = entry.name.removesuffix(".weight")

        if len(entry.shape) == 2:
            out_features, in_features = entry.shape

            targets.append(
                QuantizationTarget(
                    target_name=base_name,
                    state_key=entry.name,
                    slice_index=None,
                    source_shape=entry.shape,
                    matrix_shape=(out_features, in_features),
                    numel=entry.numel,
                    kind="linear_2d",
                )
            )

        elif len(entry.shape) == 3:
            # Assumption: [num_experts, out_features, in_features].
            # If your model uses a different axis order, change this here.
            num_experts, out_features, in_features = entry.shape
            expert_numel = out_features * in_features

            for expert_idx in range(num_experts):
                targets.append(
                    QuantizationTarget(
                        target_name=f"{base_name}.expert_{expert_idx:04d}",
                        state_key=entry.name,
                        slice_index=expert_idx,
                        source_shape=entry.shape,
                        matrix_shape=(out_features, in_features),
                        numel=expert_numel,
                        kind="fused_expert_3d",
                    )
                )

    if verbose:
        total_numel = sum(t.numel for t in targets)

        print(f"[CQ] Discovered {len(targets)} quantization targets")
        print(f"[CQ] Target parameters: {total_numel / 1e9:.3f}B")
        print(
            f"[CQ] Raw INT4 payload estimate: "
            f"{total_numel / 2 / 2**30:.2f} GiB"
        )

        by_kind: dict[str, int] = {}
        by_kind_count: dict[str, int] = {}

        for target in targets:
            by_kind[target.kind] = by_kind.get(target.kind, 0) + target.numel
            by_kind_count[target.kind] = by_kind_count.get(target.kind, 0) + 1

        for kind in sorted(by_kind, key=by_kind.get, reverse=True):
            print(
                f"  {kind:20s}: "
                f"{by_kind_count[kind]:8d} targets, "
                f"{by_kind[kind] / 1e9:10.3f}B params"
            )

    return targets


def _get_target_weight_from_state(
    state: StreamingSafeTensorState,
    target: QuantizationTarget,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    source_keys = target.all_source_keys()

    if target.kind == "cq_ngram_chunked_shard":
        raise RuntimeError(
            "Chunked N-gram targets must be handled by "
            "_quantize_ngram_chunked_target(); they are intentionally not "
            "materialized as one huge matrix."
        )

    # Legacy / non-packed target.
    if len(source_keys) == 1:
        source = state.get_tensor(
            source_keys[0],
            device="cpu",
            dtype=None,
        )
        weight = source if target.slice_index is None else source[target.slice_index]
        if weight.ndim != 2:
            raise RuntimeError(
                f"Resolved target is not 2D: "
                f"{target.target_name}, source={source_keys[0]}, "
                f"resolved_shape={tuple(weight.shape)}"
            )
        weight = weight.to(device=device, dtype=dtype, non_blocking=False)
        del source
        return weight

    # Native-vLLM packed target. Allocate the final fused matrix directly on the
    # quantization device and stream each source tensor into its output slice.
    # This avoids simultaneously materializing a second full fused copy on CPU.
    out_features, in_features = map(int, target.matrix_shape)
    weight = torch.empty(
        (out_features, in_features),
        device=device,
        dtype=dtype,
    )
    row = 0
    for key in source_keys:
        source = state.get_tensor(key, device="cpu", dtype=None)
        part = source if target.slice_index is None else source[target.slice_index]
        if part.ndim != 2:
            raise RuntimeError(
                f"Native-vLLM source is not 2D after expert slicing: "
                f"target={target.target_name}, source={key}, "
                f"shape={tuple(part.shape)}"
            )
        if int(part.shape[1]) != in_features:
            raise RuntimeError(
                f"Native-vLLM fusion input-size mismatch for {target.target_name}: "
                f"source={key}, shape={tuple(part.shape)}, expected in={in_features}"
            )
        rows = int(part.shape[0])
        weight[row:row + rows].copy_(
            part.to(device=device, dtype=dtype, non_blocking=False)
        )
        row += rows
        del part, source

    if row != out_features:
        raise RuntimeError(
            f"Native-vLLM fusion output-size mismatch for {target.target_name}: "
            f"copied {row} rows, expected {out_features}"
        )
    return weight


def _get_target_bias_from_state(
    state: StreamingSafeTensorState,
    target: QuantizationTarget,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Optional[torch.Tensor]:
    # Expert slices normally do not have a normal per-expert bias.
    if target.slice_index is not None:
        return None

    source_keys = target.all_source_keys()
    bias_keys = [key.removesuffix(".weight") + ".bias" for key in source_keys]
    present = [key in state for key in bias_keys]

    if not any(present):
        return None
    if not all(present):
        missing = [k for k, ok in zip(bias_keys, present) if not ok]
        raise RuntimeError(
            f"Native-vLLM fused target {target.target_name} has only partial bias "
            f"coverage; missing {missing}"
        )

    parts = [state.get_tensor(k, device=device, dtype=dtype) for k in bias_keys]
    try:
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    finally:
        if len(parts) > 1:
            # torch.cat owns separate storage, so source tensors can be released.
            del parts


def _hf_config_to_dict(cfg):
    # cfg is model.config; convert to plain dict
    return cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__


try:
    from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
    HAVE_SAFE = True
except Exception:
    HAVE_SAFE = False


FIXED_SEED_D_R = 987654321
FIXED_SEED_D_L = 123456789


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


@torch.jit.script
def rand_signs_splitmix_fixed(n: int, device: torch.device, seed: int, out_dtype: torch.dtype = torch.float16 ) -> torch.Tensor:
    # SplitMix64 constants as signed int64 (same bit patterns)
    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")


#@torch.compile()
def regen_perm_signs_and_apply_fw(W: torch.Tensor, seed_D: int, dim: int = 0,
                               out_dtype: torch.dtype | None = None):

    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")

#@torch.compile()
def fft_forward_real_pack_torch_right(W: torch.Tensor) -> torch.Tensor:

    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")

#@torch.compile()
def fft_forward_real_pack_torch_left(W: torch.Tensor) -> torch.Tensor:
 
    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")




#@torch.compile(mode="reduce-overhead")  # Optimize for performance
def normal_iid_at_indices_cuda(indices: torch.Tensor, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    PyTorch equivalent of the optimized CUDA normal_iid_from_index function.
    
    Generates normal random numbers from indices using hash mixing and Box-Muller transform.
    Uses float32 precision for optimal GPU performance, matching the CUDA fast-math intrinsics.
    
    This is optimized for speed while maintaining numerical consistency with the CUDA version.
    
    Args:
        indices: Tensor of indices (any shape, dtype: int64 or uint64)
        
    Returns:
        Tensor of half-precision (float16) normal random numbers (same shape as indices)
        
    Example:
        >>> indices = torch.arange(1000, device='cuda', dtype=torch.int64)
        >>> normals = normal_iid_at_indices_optimized(indices)
        >>> print(normals.shape)  # torch.Size([1000])
        >>> print(normals.dtype)  # torch.float16
    """
    # Constants matching the CUDA version exactly
    FIXED_SEED_GAUSS = 123456789
    TWO_PI = 6.283185307179586  # float32 version: 6.283185307179586f
    MIN_U = 1.192092896e-07  # 2^-23
    MAX_U = 0.9999998807907104  # 1.0 - 2^-23
    device = indices.device
    
    # Ensure indices are int64 for consistent arithmetic
    indices = indices.to(dtype=torch.int64)
    
    # OPTIMIZED: Fast hash mixing (inlined for better performance)
    # CUDA: uint32_t x0 = uint32_t(idx * 0x9E3779B9ull) ^ FIXED_SEED_GAUSS
    x0 = (indices * 0x9E3779B9) & 0xFFFFFFFF
    x0 = x0 ^ FIXED_SEED_GAUSS
    
    # CUDA: uint32_t x1 = uint32_t(idx * 0x85EBCA6Bull) ^ (FIXED_SEED_GAUSS + 0x27D4EB2Du)
    x1 = (indices * 0x85EBCA6B) & 0xFFFFFFFF
    x1 = x1 ^ (FIXED_SEED_GAUSS + 0x27D4EB2D)
    
    # Inline hash mixing (MurmurHash3 fmix32) - fewer function calls
    # CUDA: x0 ^= x0 >> 16; x0 *= 0x85EBCA6Bu; etc.
    
    # x0 mixing
    x0_uint32 = x0 & 0xFFFFFFFF
    x0_uint32 = x0_uint32 ^ (x0_uint32 >> 16)
    x0_uint32 = (x0_uint32 * 0x85EBCA6B) & 0xFFFFFFFF
    x0_uint32 = x0_uint32 ^ (x0_uint32 >> 13)
    x0_uint32 = (x0_uint32 * 0xC2B2AE35) & 0xFFFFFFFF
    x0_uint32 = x0_uint32 ^ (x0_uint32 >> 16)
    x0_uint32 = x0_uint32 & 0xFFFFFFFF
    
    # x1 mixing
    x1_uint32 = x1 & 0xFFFFFFFF
    x1_uint32 = x1_uint32 ^ (x1_uint32 >> 16)
    x1_uint32 = (x1_uint32 * 0x85EBCA6B) & 0xFFFFFFFF
    x1_uint32 = x1_uint32 ^ (x1_uint32 >> 13)
    x1_uint32 = (x1_uint32 * 0xC2B2AE35) & 0xFFFFFFFF
    x1_uint32 = x1_uint32 ^ (x1_uint32 >> 16)
    x1_uint32 = x1_uint32 & 0xFFFFFFFF
    
    # OPTIMIZED: Direct bit manipulation for uniform generation
    # CUDA: float u0 = __uint_as_float(0x3F800000u | (x0 >> 9)) - 1.0f;
    # This creates a float in [1.0, 2.0) then subtracts 1.0 to get [0.0, 1.0)
    # 
    # PyTorch equivalent: Use bit manipulation to create the same float pattern
    # The bit pattern 0x3F800000 | (x >> 9) sets:
    #   - Sign bit: 0
    #   - Exponent: 127 (0x7F) -> value = 1.0 * 2^0 = 1.0
    #   - Mantissa: top 23 bits of x
    #
    # Mathematical equivalent: u = (x >> 9) / (2^23) gives [0.0, 1.0) directly
    # This is mathematically equivalent and avoids bit reinterpretation overhead
    
    # Extract mantissa bits (top 23 bits) and convert to float
    # x >> 9 extracts bits [31:9], which when divided by 2^23 gives [0.0, 1.0)
    u0 = (x0_uint32 >> 9).to(torch.float32) * (1.0 / 8388608.0)  # 1 / 2^23
    u1 = (x1_uint32 >> 9).to(torch.float32) * (1.0 / 8388608.0)
    
    # OPTIMIZED: Branchless clamping (better for warp execution)
    # CUDA: u0 = fmaxf(fminf(u0, MAX_U), MIN_U);
    # PyTorch: torch.clamp automatically uses optimized GPU operations
    u0 = torch.clamp(u0, min=MIN_U, max=MAX_U)
    u1 = torch.clamp(u1, min=MIN_U, max=MAX_U)
    
    # OPTIMIZED: All computations in float32 (native GPU precision)
    # CUDA uses: __logf, __fsqrt_rn, __cosf (hardware-accelerated)
    # PyTorch will automatically use optimized GPU math functions
    log_u0 = torch.log(u0)
    r = torch.sqrt(-2.0 * log_u0)
    z = r * torch.cos(TWO_PI * u1)
    
    # SINGLE conversion to half at the end - this is optimal!
    # Avoiding intermediate half conversions saves overhead
    return z.to(out_dtype)



#@torch.inference_mode()
# @torch.compile(mode='max-autotune',dynamic=True)
# @torch.compile(mode="max-autotune", fullgraph=True)
class OnTheFlyLinear(torch.nn.Module):
    """
    Still 'OnTheFly', but forward() is a fast addmm on a cached, pre-transposed weight.
    """
    def __init__(self, in_features, out_features,
                 layer_rec: dict, bias=None, dtype=torch.float16, device='cuda',
                nbits_target: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.layer_rec = layer_rec
        self.dtype = dtype

        # nbits_target: int = 4
        S: int = 65521
        S_prime: int = 65521
        match nbits_target:
            case 2:
               T_small: int = 4
            case 4:
                T_small: int = 2
            case 8:
                T_small: int = 1
            case _:  # Default case, equivalent to 'else'
                T_small: int = 2

        U: int = 1
        U_prime: int = 1
        T_U=2
        batch_size: int = int(T_U*T_small)
        N: int = int(S_prime*U_prime)

        self.batch_size=batch_size
        self.N=N
        self.U=U
        self.S=S
        self.U_prime=U_prime
        self.S_prime=S_prime
        self.device=device
        self.T_small=T_small
        self.T_U=T_U
        # self.k_full=k_full


        # SMALL buffers only (no full W)
        self.register_buffer("u_W",  layer_rec["u_W"].detach().to(device=device, dtype=torch.float16).contiguous(), persistent=True)
        self.register_buffer("std_W", layer_rec["std_W"].detach().to(device=device, dtype=torch.float16).contiguous(), persistent=True)
        self.register_buffer("SigRec1_select_packed",          layer_rec["SigRec1_select_packed"].detach().to(device=device, dtype=torch.uint8).contiguous(), persistent=True)
        self.register_buffer("SigRec2_select_packed",          layer_rec["SigRec2_select_packed"].detach().to(device=device, dtype=torch.uint8).contiguous(), persistent=True)
        self.register_buffer("SigRec3_select_packed",          layer_rec["SigRec3_select_packed"].detach().to(device=device, dtype=torch.uint8).contiguous(), persistent=True)
        self.register_buffer("SigRec4_select_packed",          layer_rec["SigRec4_select_packed"].detach().to(device=device, dtype=torch.uint8).contiguous(), persistent=True)
        self.register_buffer("X567_packed",          layer_rec["X567_packed"].detach().to(device=device, dtype=torch.uint8).contiguous(), persistent=True)
        T = int(layer_rec["T"])
        self.T=T
        #self.register_buffer( "T", torch.tensor(T, dtype=torch.int64, device=device),  persistent=False)

        self.num_batches = (T + batch_size - 1) // batch_size

        # # Bias: keep on device, matching dtype; (out_features,)
        # if layer_rec["bias"] is not None:
        #     b = layer_rec["bias"].to(self.device, dtype)
        #     assert b.numel() == out_features
        #     self.register_buffer('bias', b, persistent=False)
        # else:
        #     if bias is None:
        #         self.register_buffer('bias', None, persistent=False)
        #     else:
        #         b = bias.detach().to(self.device, dtype)
        #         assert b.numel() == out_features
        #         self.register_buffer('bias', b, persistent=False)

        # if bias is None:
        #     self.register_buffer('bias', None, persistent=False)
        # else:
        #     b = bias.detach().to(self.device, dtype)
        #     assert b.numel() == out_features
        #     self.register_buffer('bias', b, persistent=False)

        self.bias = bias
        # self.weight = weight
        # self.register_buffer('weight', torch.empty((), device=device, dtype=torch.float16, requires_grad=True), persistent=True)
        # self.register_buffer('weight', None, persistent=True)

        # Persistent caches (set in warmup)
        self.register_buffer('_W_T_shape',  torch.tensor([in_features, out_features],
                                                       device=self.device, dtype=torch.int32), persistent=False)
        # then call: unpack_uint4_lut(packed, original_len, self.uint4_lut)
  
        DL=rand_signs_splitmix_fixed(self._W_T_shape[0], device, FIXED_SEED_D_R, out_dtype=torch.float32)
        self.register_buffer("DL", DL, persistent=False)
        #self.register_buffer("DL_y", DL[:,None], persistent=False)

        DR=rand_signs_splitmix_fixed(self._W_T_shape[1], device, FIXED_SEED_D_L, out_dtype=torch.float32)
        
        self.register_buffer("DR", DR[None,:], persistent=False)
        scale_W = torch.tensor(1.0/((float(self.out_features/2)**0.5)*(float(self.in_features/2)**0.5)), device=self.device, dtype=torch.float32)
        self.register_buffer("scale_W", scale_W, persistent=False)

        N = self.in_features
        M = self.out_features
        self.N_cplx = N >> 1
        self.M_cplx = M >> 1
        

        self.ts = torch.arange(T_small, device=device, dtype=torch.uint8)

        # c=normal_iid_at_indices_cuda(torch.arange(N_codeword, device='cuda').contiguous()).contiguous() 
        # self.register_buffer('codebook', c, persistent=False)
        tlut_bits=9
        V=2
        fname = f'/tmp/kmeans_{tlut_bits}_{V}.pt'
        if not os.path.exists(fname):
            # tlut = torch.randn(2**tlut_bits, V)
            tlut=normal_iid_at_indices_cuda(torch.arange(2**tlut_bits*V, device='cpu').contiguous(), out_dtype=torch.float32).contiguous()
            tlut=tlut.reshape(2**tlut_bits, V)
            import scipy
            data = torch.randn(1 << 20, 2)
            data=normal_iid_at_indices_cuda(torch.arange(1<<21, device='cpu').contiguous(), out_dtype=torch.float32).contiguous()
            data=data.reshape(1 << 20, 2)
            clusters = scipy.cluster.vq.kmeans(data, tlut)
            tlut = torch.tensor(clusters[0])
            tlut = (tlut /
                    tlut.std(unbiased=False)) * 0.9682458365518543
            torch.save(tlut, fname)
        else:
            tlut = torch.load(fname)

        c = quantlut_sym(tlut, 16, tlut_bits)[:S_prime,:].to(dtype=torch.float16, device=device).contiguous()

        self.register_buffer('codebook', c, persistent=False)

        self.register_buffer('tlut', tlut.to(device=device, dtype=torch.float16).contiguous(), persistent=False)


    def decode_compressed(self):
        idxs=(self.ts[None,:] - self.input_seq.to(torch.int32)[:,None])
        idxs.remainder_(self.S_prime)                           # in-place mod
        idxs=idxs.view(-1)[:int(self.T/2)] 
        Y_est = self.codebook[idxs,:].view(self.out_features,self.in_features)
        return Y_est


    def apply_W_optimized(self, input: torch.Tensor) -> torch.Tensor:

        N, M = self.in_features, self.out_features
        N=int(N)
        M=int(M)

        x = input.view(-1, N).to(torch.float32) * self.DL
        bs = x.shape[0]
        N_cplx = int(N/2)
        M_cplx = int(M/2)

        x_cplx = torch.view_as_complex(x.view(bs, N_cplx, 2))
        x = torch.fft.fft(x_cplx.contiguous(), dim=-1, norm="ortho")
        x = torch.view_as_real(x).view(bs, N)

        if bs==1:
            x=torch.ops.my_qlinear.fused_matvec(self.input_seq, x.contiguous(), self.tlut, M)#.to(torch.float16)
        else:
            Y_est=self.decode_compressed()
            # Y_est=torch.ops.my_qlinear.forward(self.input_seq, self.tlut, N, M)
            x = (x.to(Y_est.dtype) @ Y_est.T).float()

        XY_cmplx = torch.view_as_complex(x.view(bs, M_cplx, 2))
        XY_ifft = torch.fft.ifft(XY_cmplx.contiguous(), dim=-1, norm="ortho")
        x = torch.view_as_real(XY_ifft).view(bs, M) * self.DR

        x = self.std_W*x + (input.sum(dim=-1, keepdim=True) * self.u_W)

        return x.view(*input.shape[:-1], M).to(input.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x=self.apply_W_optimized(input)

        if self.bias is not None:
            return x + self.bias
        return x


def _get_parent_and_attr(root: torch.nn.Module, qualified_name: str):
    parts = qualified_name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

class attrgetter:
    """
    Return a callable object that fetches the given attribute(s) from its operand.
    After f = attrgetter('name'), the call f(r) returns r.name.
    After g = attrgetter('name', 'date'), the call g(r) returns (r.name, r.date).
    After h = attrgetter('name.first', 'name.last'), the call h(r) returns
    (r.name.first, r.name.last).
    """
    __slots__ = ('_attrs', '_call')

    def __init__(self, attr, /, *attrs):
        if not attrs:
            if not isinstance(attr, str):
                raise TypeError('attribute name must be a string')
            self._attrs = (attr,)
            names = attr.split('.')
            def func(obj):
                for name in names:
                    obj = getattr(obj, name)
                return obj
            self._call = func
        else:
            self._attrs = (attr,) + attrs
            getters = tuple(map(attrgetter, self._attrs))
            def func(obj):
                return tuple(getter(obj) for getter in getters)
            self._call = func

    def __call__(self, obj, /):
        return self._call(obj)

    def __repr__(self):
        return '%s.%s(%s)' % (self.__class__.__module__,
                              self.__class__.__qualname__,
                              ', '.join(map(repr, self._attrs)))

    def __reduce__(self):
        return self.__class__, self._attrs

def replace_quantized_linears_with_onthefly(model, quantized_records, dtype=torch.float16, device='cuda'):
    replaced = 0
    for name, module in list(model.named_modules()):
        if name in quantized_records and isinstance(module, torch.nn.Linear):
            parent, attr = _get_parent_and_attr(model, name)
            # print(module.weight.dtype)
            # print(module.weight)

            # quit()
            new_lin = OnTheFlyLinear(module.in_features, module.out_features,
                                     layer_rec=quantized_records[name],
                                     bias=module.bias, dtype=dtype, device=device)
            # setattr(parent, attr, new_lin)
            split_attr = name.split('.')
            setattr(
                attrgetter('.'.join(split_attr[:-1]))(model), split_attr[-1],
                new_lin)

            replaced += 1
    return replaced







def pack_bits_uint8(bits: torch.Tensor, bitorder: str = "little") -> torch.Tensor:
    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")

def unpack_bits_uint8(packed: torch.Tensor, num_bits: int, bitorder: str = "little") -> torch.Tensor:
    packed = packed.to(torch.uint8)

    if bitorder == "little":
        shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    elif bitorder == "big":
        shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    else:
        raise ValueError("bitorder must be 'little' or 'big'")

    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    bits = bits.reshape(*packed.shape[:-1], -1)
    return bits[..., :num_bits]


def encoder4polar_torch_batched(Miu: torch.Tensor) -> torch.Tensor:
    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")

# import math
# import torch


# def compute_lr_level_torch(y, PreviousX, X, Px, lr_funcs):
#     """
#     Select the appropriate LR function according to the decoded prefix.

#     PreviousX must contain integer prefix IDs in:
#         0 .. len(lr_funcs)-1
#     """
#     y = torch.as_tensor(y)
#     previous_x = torch.as_tensor(PreviousX, device=y.device)
#     out = torch.empty_like(y, dtype=torch.float32)

#     for prefix_id, fn in enumerate(lr_funcs):
#         mask = previous_x == prefix_id
#         if mask.any():
#             out[mask] = fn(y[mask], X, Px)

#     return out


# def lr_level_func_torch(y, X, Px, level, func_id, sigma=0.31623):
#     """
#     Generic multilevel LR for a set-partition-labelled alphabet.

#     This version is configured for M = 128 and levels 1..7.

#     Parameters
#     ----------
#     y:
#         Observation tensor of any shape.
#     X:
#         Reconstruction alphabet, length 128.
#     Px:
#         Symbol probabilities, length 128.
#     level:
#         Set-partition level, 1..7.
#     func_id:
#         MATLAB-style prefix/function ID:
#             1 .. 2**(level-1)
#     sigma:
#         Gaussian standard deviation.

#     For zero-based prefix p = func_id - 1:

#         idx0 = p + k*2**level
#         idx1 = idx0 + 2**(level-1)

#     for every valid k. The returned value is

#         log p(y, bit_level=0, prefix)
#         - log p(y, bit_level=1, prefix).
#     """
#     M = 128

#     if not (1 <= level <= 7):
#         raise ValueError("level must be in 1..7")

#     num_prefixes = 1 << (level - 1)
#     if not (1 <= func_id <= num_prefixes):
#         raise ValueError(
#             f"func_id must be in 1..{num_prefixes} for level {level}"
#         )

#     if sigma <= 0:
#         raise ValueError("sigma must be positive")

#     device = y.device
#     y = y.float()
#     X = torch.as_tensor(X, device=device, dtype=torch.float32).flatten()
#     Px = torch.as_tensor(Px, device=device, dtype=torch.float32).flatten()

#     if X.numel() != M or Px.numel() != M:
#         raise ValueError("X and Px must each contain exactly 128 entries")

#     prefix = func_id - 1
#     stride = 1 << level
#     half_stride = 1 << (level - 1)

#     idx0 = torch.arange(prefix, M, stride, device=device)
#     idx1 = idx0 + half_stride

#     X0 = X[idx0]
#     X1 = X[idx1]
#     Px0 = Px[idx0]
#     Px1 = Px[idx1]

#     yy = y.unsqueeze(-1)

#     sigma2 = float(sigma) * float(sigma)
#     log_norm_const = -0.5 * math.log(2.0 * math.pi * sigma2)

#     log_pdf0 = log_norm_const - 0.5 * ((yy - X0) ** 2) / sigma2
#     log_pdf1 = log_norm_const - 0.5 * ((yy - X1) ** 2) / sigma2

#     tiny = torch.finfo(torch.float32).tiny
#     log_w0 = torch.log(Px0.clamp_min(tiny))
#     log_w1 = torch.log(Px1.clamp_min(tiny))

#     log_y0 = torch.logsumexp(log_w0 + log_pdf0, dim=-1)
#     log_y1 = torch.logsumexp(log_w1 + log_pdf1, dim=-1)

#     return log_y0 - log_y1


# def make_lr_level_func(level, func_id):
#     """Create a named wrapper compatible with the original function API."""
#     def fn(y, X, Px, sigma=0.31623):
#         return lr_level_func_torch(
#             y=y,
#             X=X,
#             Px=Px,
#             level=level,
#             func_id=func_id,
#             sigma=sigma,
#         )

#     fn.__name__ = f"lr_level{level}_func{func_id}_torch"
#     fn.__qualname__ = fn.__name__
#     fn.__doc__ = (
#         f"LR function for M=128, level={level}, "
#         f"MATLAB-style func_id={func_id}."
#     )
#     return fn


# # Create named functions:
# #   level 1:  1 function
# #   level 2:  2 functions
# #   level 3:  4 functions
# #   level 4:  8 functions
# #   level 5: 16 functions
# #   level 6: 32 functions
# #   level 7: 64 functions
# LR_FUNCS_BY_LEVEL = {}

# for _level in range(1, 8):
#     _funcs = []

#     for _func_id in range(1, (1 << (_level - 1)) + 1):
#         _fn = make_lr_level_func(_level, _func_id)
#         globals()[_fn.__name__] = _fn
#         _funcs.append(_fn)

#     LR_FUNCS_BY_LEVEL[_level] = _funcs


# # Convenient aliases.
# lr_level1_funcs = LR_FUNCS_BY_LEVEL[1]
# lr_level2_funcs = LR_FUNCS_BY_LEVEL[2]
# lr_level3_funcs = LR_FUNCS_BY_LEVEL[3]
# lr_level4_funcs = LR_FUNCS_BY_LEVEL[4]
# lr_level5_funcs = LR_FUNCS_BY_LEVEL[5]
# lr_level6_funcs = LR_FUNCS_BY_LEVEL[6]
# lr_level7_funcs = LR_FUNCS_BY_LEVEL[7]


# def compute_lr_for_level_torch(
#     y,
#     PreviousX,
#     X,
#     Px,
#     level,
#     sigma=0.31623,
# ):
#     """
#     Compute the LR for a complete level without manually passing lr_funcs.

#     PreviousX stores zero-based previous-prefix IDs:
#         level 1: always 0
#         level 2: 0..1
#         level 3: 0..3
#         ...
#         level 7: 0..63
#     """
#     if not (1 <= level <= 7):
#         raise ValueError("level must be in 1..7")

#     funcs = LR_FUNCS_BY_LEVEL[level]

#     # Bind sigma while retaining the original callable interface.
#     bound_funcs = [
#         (lambda yy, xx, pp, fn=fn: fn(yy, xx, pp, sigma))
#         for fn in funcs
#     ]

#     return compute_lr_level_torch(
#         y=y,
#         PreviousX=PreviousX,
#         X=X,
#         Px=Px,
#         lr_funcs=bound_funcs,
#     )


def build_lr_index_tables_m128(device):
    tables = {}

    for level in range(1, 8):
        num_prefixes = 1 << (level - 1)
        points_per_hypothesis = 128 >> level
        stride = 1 << level
        offset = 1 << (level - 1)

        prefix = torch.arange(
            num_prefixes,
            device=device,
            dtype=torch.long,
        ).unsqueeze(1)

        k = torch.arange(
            points_per_hypothesis,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

        idx0 = prefix + stride * k
        idx1 = idx0 + offset

        tables[level] = (idx0, idx1)

    return tables

LR_INDEX_TABLES = None  # native _cq_core owns LR index construction




@torch.no_grad()
def compute_lr_level_m128_uniform_precomputed(
    y,
    PreviousX,
    X,
    level,
    index_tables,
    # sigma=0.31623,
    inv_two_sigma2=4.9999,
):
    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")

@torch.no_grad()
def compute_lr_level7_uniform_m128_fast(
    y,
    PreviousX,
    X,
    # sigma=0.31623,
    inv_two_sigma2=4.9999,
    ):
    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")

def arithenc_manual_py(seq, counts):
    """
    Python equivalent of MATLAB arithenc_manual.

    seq: iterable of symbols 1..M
    counts: iterable of counts for symbols 1..M

    returns:
        bits: list of 0/1 ints
    """
    seq = [int(s) for s in seq]
    counts = [int(c) for c in counts]

    M = len(counts)

    if any(s < 1 or s > M for s in seq):
        raise ValueError("Symbols in seq must be between 1 and len(counts).")

    used = set(seq)
    for i, c in enumerate(counts, start=1):
        if c == 0 and i in used:
            raise ValueError("A used symbol has zero count.")

    total = sum(counts)

    cum = [0]
    for c in counts:
        cum.append(cum[-1] + c)

    PREC = 32
    TOP = (1 << PREC) - 1
    HALF = 1 << (PREC - 1)
    FIRST_QTR = 1 << (PREC - 2)
    THIRD_QTR = 3 * FIRST_QTR

    low = 0
    high = TOP

    bits = []
    pending = 0

    def output_bit(b):
        bits.append(1 if b else 0)

    for s in seq:
        range_ = high - low + 1

        new_low = low + (range_ * cum[s - 1]) // total
        new_high = low + (range_ * cum[s]) // total - 1

        low = new_low
        high = new_high

        while True:
            if high < HALF:
                output_bit(0)

                while pending > 0:
                    output_bit(1)
                    pending -= 1

            elif low >= HALF:
                output_bit(1)

                while pending > 0:
                    output_bit(0)
                    pending -= 1

                low -= HALF
                high -= HALF

            elif low >= FIRST_QTR and high < THIRD_QTR:
                pending += 1
                low -= FIRST_QTR
                high -= FIRST_QTR

            else:
                break

            low = 2 * low
            high = 2 * high + 1

    pending += 1

    if low < FIRST_QTR:
        output_bit(0)

        while pending > 0:
            output_bit(1)
            pending -= 1
    else:
        output_bit(1)

        while pending > 0:
            output_bit(0)
            pending -= 1

    return bits


def arithenc_manual_torch(seq, counts):
    """
    PyTorch tensor version of arithenc_manual.

    seq: torch tensor of symbols 1..M
    counts: torch tensor of counts for symbols 1..M

    returns:
        bits: torch tensor of 0/1 ints
    """

    seq = seq.to(dtype=torch.long)
    counts = counts.to(dtype=torch.long)

    device = seq.device
    M = counts.numel()

    if torch.any((seq < 1) | (seq > M)):
        raise ValueError("Symbols in seq must be between 1 and len(counts).")

    used_counts = counts[seq - 1]
    if torch.any(used_counts == 0):
        raise ValueError("A used symbol has zero count.")

    total = torch.sum(counts)

    cum = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(counts, dim=0)
    ])

    PREC = 32
    TOP = torch.tensor((1 << PREC) - 1, dtype=torch.long, device=device)
    HALF = torch.tensor(1 << (PREC - 1), dtype=torch.long, device=device)
    FIRST_QTR = torch.tensor(1 << (PREC - 2), dtype=torch.long, device=device)
    THIRD_QTR = torch.tensor(3 * (1 << (PREC - 2)), dtype=torch.long, device=device)

    low = torch.tensor(0, dtype=torch.long, device=device)
    high = TOP.clone()

    bits = torch.empty(0, dtype=torch.long, device=device)
    pending = torch.tensor(0, dtype=torch.long, device=device)

    def append_bit(bits, b):
        b = torch.tensor([b], dtype=torch.long, device=device)
        return torch.cat([bits, b])

    def append_pending(bits, bit, pending):
        if pending.item() > 0:
            extra = torch.full(
                (pending.item(),),
                bit,
                dtype=torch.long,
                device=device
            )
            bits = torch.cat([bits, extra])
            pending = torch.tensor(0, dtype=torch.long, device=device)
        return bits, pending

    for s in seq:
        range_ = high - low + 1

        new_low = low + (range_ * cum[s - 1]) // total
        new_high = low + (range_ * cum[s]) // total - 1

        low = new_low
        high = new_high

        while True:
            if high < HALF:
                bits = append_bit(bits, 0)
                bits, pending = append_pending(bits, 1, pending)

            elif low >= HALF:
                bits = append_bit(bits, 1)
                bits, pending = append_pending(bits, 0, pending)

                low = low - HALF
                high = high - HALF

            elif (low >= FIRST_QTR) and (high < THIRD_QTR):
                pending = pending + 1
                low = low - FIRST_QTR
                high = high - FIRST_QTR

            else:
                break

            low = 2 * low
            high = 2 * high + 1

    pending = pending + 1

    if low < FIRST_QTR:
        bits = append_bit(bits, 0)
        bits, pending = append_pending(bits, 1, pending)
    else:
        bits = append_bit(bits, 1)
        bits, pending = append_pending(bits, 0, pending)

    return bits


def arithdec_manual_py(bits, counts, nsym):
    """
    Python equivalent of MATLAB arithdec_manual.

    bits: list of 0/1 ints
    counts: counts for symbols 1..M
    nsym: number of symbols to decode

    returns:
        seq: list of decoded symbols 1..M
    """
    bits = [int(b) for b in bits]
    counts = [int(c) for c in counts]

    M = len(counts)
    total = sum(counts)

    cum = [0]
    for c in counts:
        cum.append(cum[-1] + c)

    PREC = 32
    TOP = (1 << PREC) - 1
    HALF = 1 << (PREC - 1)
    FIRST_QTR = 1 << (PREC - 2)
    THIRD_QTR = 3 * FIRST_QTR

    low = 0
    high = TOP
    code = 0
    bitptr = 0

    def read_bit():
        nonlocal bitptr
        if bitptr < len(bits):
            b = bits[bitptr]
        else:
            b = 0
        bitptr += 1
        return b

    for _ in range(PREC):
        code = 2 * code + read_bit()

    seq = []

    for _ in range(nsym):
        range_ = high - low + 1

        value = ((code - low + 1) * total - 1) // range_

        s = None
        for j in range(M):
            if cum[j + 1] > value:
                s = j + 1
                break

        if s is None:
            raise RuntimeError("Decoder failed: symbol not found.")

        seq.append(s)

        new_low = low + (range_ * cum[s - 1]) // total
        new_high = low + (range_ * cum[s]) // total - 1

        low = new_low
        high = new_high

        while True:
            if high < HALF:
                pass

            elif low >= HALF:
                low -= HALF
                high -= HALF
                code -= HALF

            elif low >= FIRST_QTR and high < THIRD_QTR:
                low -= FIRST_QTR
                high -= FIRST_QTR
                code -= FIRST_QTR

            else:
                break

            low = 2 * low
            high = 2 * high + 1
            code = 2 * code + read_bit()

    return seq


def pack_bits_list_little(bits):
    """
    bits: Python list of 0/1 values, length multiple of 8
    returns: uint8 tensor of shape [len(bits)//8]
    """
    b = torch.tensor(bits, dtype=torch.uint8)

    b = b.view(-1, 8)

    shifts = torch.arange(8, dtype=torch.uint8)
    packed = (b << shifts).sum(dim=1).to(torch.uint8)

    return packed

def unpack_bits_little(packed, num_bits):
    packed = packed.to(torch.uint8)

    shifts = torch.arange(8, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(-1)

    return bits[:num_bits]


def encoder4polar_packed_torch_batched(Miu_packed: torch.Tensor, N: int) -> torch.Tensor:
    """Implementation moved to the private _cq_core native extension."""
    raise RuntimeError("This CQ primitive is private and only available through _cq_core.so")


def restore_full_packed_from_selected_packed(
    SigRec_select_packed: torch.Tensor,
    select_index: torch.Tensor,
    N: int,
    bitorder: str = "little",
):
    """
    Restore full packed SigRec directly.

    SigRec_select_packed: [B, ceil(K/8)] uint8
    select_index: [K] long, 0-based selected positions in full length N
    N: full length, must be divisible by 8

    returns:
        SigRec_full_packed: [B, N//8] uint8
    """
    if N % 8 != 0:
        raise ValueError("N must be divisible by 8")

    device = SigRec_select_packed.device
    select_index = select_index.to(device=device, dtype=torch.long)

    B = SigRec_select_packed.shape[0]
    K = select_index.numel()

    selected_bits = unpack_bits_uint8(
        SigRec_select_packed,
        num_bits=K,
        bitorder=bitorder,
    )

    full_packed = torch.zeros(B, N // 8, device=device, dtype=torch.uint8)

    byte_idx = select_index // 8

    if bitorder == "little":
        bit_idx = select_index % 8
    elif bitorder == "big":
        bit_idx = 7 - (select_index % 8)
    else:
        raise ValueError("bitorder must be 'little' or 'big'")

    masks = (1 << bit_idx).to(torch.uint8)

    # Since select_index should not have duplicates, this is safe.
    full_packed[:, byte_idx] |= selected_bits * masks

    return full_packed

# Checkpoint naming compatibility (independent of removed legacy codecs).
def _deepseek_v4_flash_key(layer_name: str) -> list[str]:
    if layer_name.startswith("model."):
        layer_name = layer_name[len("model."):]  # layers.0....

    keys = []

    # Attention
    attn = layer_name.replace(".self_attn.", ".attn.")
    attn = attn.replace(".q_a_proj", ".wq_a")
    attn = attn.replace(".q_b_proj", ".wq_b")
    attn = attn.replace(".kv_proj", ".wkv")
    attn = attn.replace(".o_a_proj", ".wo_a")
    attn = attn.replace(".o_b_proj", ".wo_b")
    if attn != layer_name:
        keys.append(attn)

    # Shared experts: gate/up/down usually map to w1/w3/w2
    if ".mlp.shared_experts." in layer_name:
        base = layer_name.replace(".mlp.shared_experts.", ".ffn.shared_experts.")

        keys.append(base.replace(".gate_proj", ".w1"))
        keys.append(base.replace(".down_proj", ".w2"))
        keys.append(base.replace(".up_proj", ".w3"))

        # fallback variants
        keys.append(base.replace(".gate_proj", ".gate"))
        keys.append(base.replace(".down_proj", ".down"))
        keys.append(base.replace(".up_proj", ".up"))

    # Non-shared MLP fallback
    if ".mlp." in layer_name:
        base = layer_name.replace(".mlp.", ".ffn.")
        keys.append(base.replace(".gate_proj", ".w1"))
        keys.append(base.replace(".down_proj", ".w2"))
        keys.append(base.replace(".up_proj", ".w3"))

    return list(dict.fromkeys(keys))


# -----------------------------
# Model wrapper: RHT + QT
# -----------------------------
def _pack_cq_layer_data_to_packed_all(layer_data: dict) -> torch.Tensor:
    """Convert one normal CQ record to final [B,520] runtime bytes."""
    sig1 = layer_data["SigRec1_select_packed"].detach()
    sig2 = layer_data["SigRec2_select_packed"].detach()
    sig3 = layer_data["SigRec3_select_packed"].detach()
    sig4 = layer_data["SigRec4_select_packed"].detach()
    x567 = layer_data["X567_packed"].detach()
    B = int(sig1.shape[0])
    if B != CQ_NGRAM_POLAR_BLOCKS_PER_CHUNK:
        raise RuntimeError(f"N-gram CQ chunk expected B=5 polar blocks, got B={B}")
    out = torch.zeros((B, 520), dtype=torch.uint8, device=sig1.device)
    out[:, 0:7] = sig1
    out[:, 8:20] = sig2
    out[:, 20:49] = sig3
    out[:, 52:135] = sig4
    out[:, 136:264] = x567[:B]
    out[:, 264:392] = x567[B:2 * B]
    out[:, 392:520] = x567[2 * B:3 * B]
    return out.contiguous()


def _quantize_ngram_chunked_target(
    wrapper,
    state: StreamingSafeTensorState,
    target: QuantizationTarget,
    *,
    device: str,
    verbose: bool,
) -> bool:
    """Conservative one-chunk-at-a-time Qwen4Exp N-gram quantization.

    This path intentionally keeps only one [160,1024] CUDA input and one native
    quantization result live at a time. It avoids staged CUDA allocations, pinned
    shard-sized output buffers, asynchronous D2H copies, and per-chunk
    ``torch.cuda.empty_cache()`` calls.
    """
    if target.kind != "cq_ngram_chunked_shard":
        raise ValueError(target.kind)

    entry = state.get_entry(target.state_key)
    rows, dim = map(int, entry.shape)
    if dim != CQ_NGRAM_EMBED_DIM:
        raise RuntimeError(f"Bad N-gram dim for {target.state_key}: {entry.shape}")

    num_chunks = (rows + CQ_NGRAM_CHUNK_ROWS - 1) // CQ_NGRAM_CHUNK_ROWS
    packed_all = torch.empty(
        (num_chunks, CQ_NGRAM_POLAR_BLOCKS_PER_CHUNK, 520),
        dtype=torch.uint8,
        device="cpu",
    )
    u_all = torch.empty((num_chunks,), dtype=torch.float32, device="cpu")
    std_all = torch.empty((num_chunks,), dtype=torch.float32, device="cpu")

    miss_before = int(getattr(wrapper, "_ngram_threshold_miss_count", 0))
    worst_before = getattr(wrapper, "_ngram_threshold_worst_nmse", None)

    with safe_open(entry.shard_path, framework="pt", device="cpu") as f:
        source_slice = f.get_slice(target.state_key)
        for chunk_id, row0 in enumerate(range(0, rows, CQ_NGRAM_CHUNK_ROWS)):
            row1 = min(row0 + CQ_NGRAM_CHUNK_ROWS, rows)
            source = source_slice[row0:row1]
            if int(source.shape[1]) != CQ_NGRAM_EMBED_DIM:
                raise RuntimeError(
                    f"N-gram source width changed in {target.state_key}: "
                    f"{tuple(source.shape)}"
                )
            if int(source.shape[0]) < CQ_NGRAM_CHUNK_ROWS:
                source = F.pad(
                    source,
                    (0, 0, 0, CQ_NGRAM_CHUNK_ROWS - int(source.shape[0])),
                    value=0.0,
                )

            W = source.t().contiguous().to(
                device=device, dtype=torch.float16, non_blocking=False
            )
            tmp_name = f"{target.target_name}.__chunk_tmp__"
            ok = wrapper.quantize_layer(tmp_name, W, bias=None, verbose=False)
            del W, source
            if not ok:
                wrapper.quantized_layers.pop(tmp_name, None)
                raise RuntimeError(
                    f"CQ N-gram quantization failed: {target.target_name} "
                    f"chunk={chunk_id} rows=[{row0}:{row1})"
                )

            q = wrapper.quantized_layers.pop(tmp_name)
            packed_cpu = _pack_cq_layer_data_to_packed_all(q).detach().to(
                device="cpu", non_blocking=False
            )
            packed_all[chunk_id].copy_(packed_cpu)
            u_all[chunk_id] = q["u_W"].detach().float().cpu().reshape(())
            std_all[chunk_id] = q["std_W"].detach().float().cpu().reshape(())
            del packed_cpu, q

            if verbose and (chunk_id % 256 == 0 or chunk_id + 1 == num_chunks):
                print(
                    f"  [CQ] ngram {target.target_name}: "
                    f"chunk {chunk_id + 1:,}/{num_chunks:,}",
                    flush=True,
                )

    miss_after = int(getattr(wrapper, "_ngram_threshold_miss_count", 0))
    shard_misses = max(0, miss_after - miss_before)
    worst_after = getattr(wrapper, "_ngram_threshold_worst_nmse", None)
    if shard_misses:
        worst = worst_after if worst_after is not None else worst_before
        worst_text = "unknown" if worst is None else f"{float(worst):.6f} dB"
        print(
            f"[CQ][WARNING] N-gram {target.target_name}: "
            f"{shard_misses:,}/{num_chunks:,} chunks missed -22 dB; "
            f"worst={worst_text}; continuing",
            flush=True,
        )

    wrapper.quantized_layers[target.target_name] = {
        "packed_all": packed_all,
        "u_W": u_all,
        "std_W": std_all,
        "bias": None,
        "T": int(rows * dim),
        "original_shape": (rows, dim),
        "matrix_shape": (CQ_NGRAM_EMBED_DIM, CQ_NGRAM_CHUNK_ROWS),
        "target_kind": "cq_ngram_chunked_shard",
        "ngram_chunk_rows": CQ_NGRAM_CHUNK_ROWS,
        "ngram_embedding_dim": CQ_NGRAM_EMBED_DIM,
        "ngram_num_chunks": num_chunks,
        "ngram_valid_rows": rows,
    }
    return True


class CQModelWrapper(nn.Module):
    def __init__(self, 
                model: PreTrainedModel,
                nbits_target: int = 4
    ):
        super().__init__()

        self.model = model
        self.quantized_layers: Dict[str, Dict] = {}
        self.original_layers: Dict[str, torch.Tensor] = {}

        # The private native core owns all CQ codebook/polar/SC constants and
        # its CUDA-graph cache. Customer Python only selects the CUDA device.
        self.device = "cuda"

        # N-gram threshold diagnostics are aggregated per physical shard so the
        # hot loop does not print+flush thousands of multi-line warnings.
        self._ngram_threshold_miss_count = 0
        self._ngram_threshold_worst_nmse = None

    def quantize_layer(self, layer_name: str, weight_tensor: torch.Tensor, bias: Optional[torch.Tensor] = None, verbose: bool = True) -> bool:
        """Quantize one [M,N] matrix through the private native CQ core.

        Python intentionally contains no FFT/LR/polar/packing implementation in
        this execution path.  _cq_core.so validates the signed license lease
        before performing the proprietary ATen quantization.
        """
        try:
            core = _require_cq_core()
            result = core.quantize(weight_tensor)

            nmse_db_value = float(result["nmse_db"])
            mse_db_value = float(result["mse_db"])
            den_db_value = float(result["den_db"])
            dB_threshold = -22.0

            if verbose:
                print(f"MSE for layer {layer_name}:")
                print(f"nmse_db={nmse_db_value}")
                print(f"mse_db={mse_db_value}")
                print(f"den_db={den_db_value}")

            layer_name_str = str(layer_name)
            is_ngram_chunk = (
                "ngram_embedding" in layer_name_str
                and (
                    ".__chunk_" in layer_name_str
                    or ".__chunk_tmp__" in layer_name_str
                )
            )

            if nmse_db_value > dB_threshold:
                if is_ngram_chunk:
                    self._ngram_threshold_miss_count += 1
                    if (
                        self._ngram_threshold_worst_nmse is None
                        or nmse_db_value > self._ngram_threshold_worst_nmse
                    ):
                        # "Worst" means highest/least-negative NMSE dB.
                        self._ngram_threshold_worst_nmse = nmse_db_value
                else:
                    print(
                        f"[CQ] Failed to quantize {layer_name}: "
                        f"Distortion is too high = {nmse_db_value:.6f} dB "
                        f"(threshold={dB_threshold:.6f} dB)",
                        flush=True,
                    )
                    return False

            bias_to_store = None
            if bias is not None:
                bias_to_store = bias.detach().to(
                    device=weight_tensor.device, dtype=torch.float16
                ).contiguous()

            self.quantized_layers[layer_name] = {
                "u_W": result["u_W"],
                "std_W": result["std_W"],
                "SigRec1_select_packed": result["SigRec1_select_packed"],
                "SigRec2_select_packed": result["SigRec2_select_packed"],
                "SigRec3_select_packed": result["SigRec3_select_packed"],
                "SigRec4_select_packed": result["SigRec4_select_packed"],
                "X567_packed": result["X567_packed"],
                "bias": bias_to_store,
                "T": int(result["T"]),
                "original_shape": (int(result["M"]), int(result["N"])),
            }
            return True
        except Exception as e:
            message = str(e)
            # Native licensing is a process-level prerequisite, not a per-layer
            # quantization failure. Abort immediately instead of silently
            # marking every target as skipped.
            if "CQ license" in message or "license server" in message.lower():
                raise RuntimeError(message) from e
            if verbose:
                print(f"[CQ] Failed to quantize {layer_name}: {e}")
            return False


    def quantize_model_layer_by_layer(
        self,
        model_name: str,
        exclude_patterns: Optional[List[str]] = None,
        verbose: bool = True,
        device: str = "cuda",
        save_path: Optional[str] = None,
        save_incremental: bool = False,
        native_vllm_layout: bool = True,
    ):
        if exclude_patterns is None:
            exclude_patterns = [
                "lm_head",
                "embed",
                "embedding",
                "norm",
                "ln_f",
                "layernorm",
                "LayerNorm",
            ]

        if save_incremental and save_path is None:
            raise ValueError(
                "save_path must be provided when save_incremental=True"
            )

        if verbose:
            print(f"[CQ] Building streaming checkpoint index for {model_name}")

        # This is not a full state dict. It is metadata + one-tensor loader.
        state = _load_state_dict_from_model(
            model_name,
            map_location="cpu",
        )

        # Always repair/save the ORIGINAL top-level HF config, not the inner
        # CausalLM skeleton config (critical for Qwen4Exp multimodal models).
        if save_path is not None:
            original_config_path = state.model_path / "config.json"
            with open(original_config_path, "r", encoding="utf-8") as f:
                original_cfg = json.load(f)
            qcfg = original_cfg.setdefault("quantization_config", {})
            qcfg["quant_method"] = "cq_quant"
            qcfg["cq_layout"] = "native_vllm"
            original_cfg["_name_or_path"] = (
                original_cfg.get("_name_or_path") or model_name
            )
            save_dir = pathlib.Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            tmp_cfg = save_dir / "config.json.tmp"
            with open(tmp_cfg, "w", encoding="utf-8") as f:
                json.dump(original_cfg, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_cfg, save_dir / "config.json")

        # ============================================================
        # Qwen4Exp PLE n-gram physical shards — PRE-DISCOVERY
        # ============================================================
        # N-gram shards are mandatory only for Qwen4Exp. Generic models
        # (GPTNeoX/Pythia, Llama, etc.) must bypass this architecture-specific
        # pre-scan and continue through normal Linear/MoE discovery.
        architectures = getattr(self.model.config, "architectures", None) or []
        model_type = str(getattr(self.model.config, "model_type", "") or "")
        is_qwen4exp = (
            any("qwen4exp" in str(arch).lower() for arch in architectures)
            or "qwen4exp" in model_type.lower()
        )

        ngram_targets = []
        _ngram_sources = set()

        if is_qwen4exp:
            # Run this BEFORE normal Linear/MoE discovery. That native
            # discovery prints its own subtotal before returning, so putting
            # n-gram discovery afterward makes the large PLE path look absent.
            all_entries_ngram = list(state.iter_entries())

            print(
                f"[CQ] NGRAM-v8 pre-scan: "
                f"checkpoint_entries={len(all_entries_ngram)}",
                flush=True,
            )

            ngram_targets, _ngram_sources = _discover_ngram_embedding_targets(
                all_entries_ngram,
                verbose=verbose,
            )

            ngram_params = sum(int(t.numel) for t in ngram_targets)

            print(
                f"[CQ] NGRAM-v8 pre-discovery result: "
                f"{len(ngram_targets)} physical shards, "
                f"{ngram_params / 1e9:.3f}B params, "
                f"raw INT4={ngram_params / 2 / 2**30:.2f} GiB",
                flush=True,
            )

            if not ngram_targets:
                exact_candidates = [
                    e for e in all_entries_ngram
                    if ".ngram_embedding.shard_" in str(e.name)
                    and str(e.name).endswith(".weight")
                ]
                preview = "\n".join(
                    f"  {e.name} shape={tuple(e.shape)}"
                    for e in exact_candidates[:20]
                )
                raise RuntimeError(
                    "Qwen4Exp n-gram pre-discovery found zero CQ targets. "
                    f"Exact shard_N.weight candidates found={len(exact_candidates)}.\n"
                    f"{preview}"
                )
        elif verbose:
            arch_label = architectures[0] if architectures else model_type or "unknown"
            print(
                f"[CQ] N-gram pre-scan skipped for architecture {arch_label}",
                flush=True,
            )

        if native_vllm_layout:
            targets = discover_native_vllm_quantization_targets_from_state(
                state,
                exclude_patterns=exclude_patterns,
                verbose=verbose,
            )
        else:
            targets = discover_quantization_targets_from_state(
                state,
                exclude_patterns=exclude_patterns,
                verbose=verbose,
            )
        # ============================================================
        # Merge the already-discovered 128 physical n-gram targets with the
        # normal Linear/MoE target list after native discovery returns.
        existing_target_names = {str(t.target_name) for t in targets}
        added_ngram_targets = [
            t for t in ngram_targets
            if str(t.target_name) not in existing_target_names
        ]
        targets = list(targets) + added_ngram_targets

        total_params_after_ngram = sum(int(t.numel) for t in targets)

        print(
            f"[CQ] NGRAM-v8 merge: "
            f"{len(added_ngram_targets)} n-gram targets added",
            flush=True,
        )
        print(
            f"[CQ] Authoritative target total after n-gram merge: "
            f"{len(targets)} targets, "
            f"{total_params_after_ngram / 1e9:.3f}B params, "
            f"raw INT4={total_params_after_ngram / 2 / 2**30:.2f} GiB",
            flush=True,
        )

        if verbose and len(targets) == 0:
            print("\n[CQ] No targets found. Showing rank-2 .weight tensors:")
            shown = 0

            for entry in state.iter_entries():
                if not entry.name.endswith(".weight"):
                    continue
                if len(entry.shape) != 2:
                    continue

                print(
                    f"  {entry.name:80s} "
                    f"shape={entry.shape} "
                    f"dtype={entry.dtype}"
                )

                shown += 1
                if shown >= 100:
                    break
        if not targets:
            raise RuntimeError("No quantization targets were discovered")

        # Recover a missing/corrupt manifest from the actual saved storage.
        # This uses the same .pt / cq_shards.json+safetensors checks as resume.
        if save_path is not None:
            self._rebuild_layers_index_from_saved_targets(
                save_path,
                targets,
                verbose=verbose,
            )

        q = 0
        q_resumed = 0
        skipped = 0

        quantized_parameter_count = 0
        target_parameter_count = sum(t.numel for t in targets)

        # If the model is a meta model this is cheap; if not, avoid this for 405B.
        try:
            total_modules = len(list(self.model.named_modules()))
        except Exception:
            total_modules = 0

        for i, target in enumerate(targets):
            layer_name = target.target_name

            if verbose:
                print(
                    f"\n[CQ] Processing target "
                    f"{i + 1}/{len(targets)}: {layer_name}"
                )
                print(f"  source_keys:  {target.all_source_keys()}")
                print(f"  kind:         {target.kind}")
                print(f"  source_shapes:{target.all_source_shapes()}")
                print(f"  matrix_shape: {target.matrix_shape}")
                print(f"  params:       {target.numel:,}")
                print(
                    f"  GPU memory before: "
                    f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
                )

            if save_path is not None and self._is_layer_already_saved(
                save_path,
                layer_name,
            ):
                if verbose:
                    print(f"  [CQ] Already saved, skipping")

                self._ensure_layer_in_layers_index(
                    save_path,
                    layer_name,
                )

                self.quantized_layers[layer_name] = {
                    "T": 0,
                    "original_shape": target.matrix_shape,
                    "source_state_key": target.state_key,
                    "source_state_keys": list(target.all_source_keys()),
                    "slice_index": target.slice_index,
                    "source_shape": target.source_shape,
                    "source_shapes": [list(x) for x in target.all_source_shapes()],
                    "target_kind": target.kind,
                    "row_start": target.row_start,
                    "row_end": target.row_end,
                    "transpose_source": target.transpose_source,
                    "embedding_group": target.embedding_group,
                    "_saved": True,
                }

                q_resumed += 1
                quantized_parameter_count += target.numel
                continue

            weight_tensor = None
            bias_tensor = None

            try:
                if target.kind == "cq_ngram_chunked_shard":
                    success = _quantize_ngram_chunked_target(
                        self, state, target, device=device, verbose=verbose
                    )
                else:
                    weight_tensor = _get_target_weight_from_state(
                        state,
                        target,
                        device=device,
                        dtype=torch.float16,
                    )

                    bias_tensor = _get_target_bias_from_state(
                        state,
                        target,
                        device=device,
                        dtype=torch.float16,
                    )

                    success = self.quantize_layer(
                        layer_name,
                        weight_tensor,
                        bias=bias_tensor,
                        verbose=verbose,
                    )

                if not success:
                    skipped += 1
                    continue

                q += 1
                quantized_parameter_count += target.numel

                layer_data = self.quantized_layers[layer_name]

                # Store provenance for reconstruction / debugging.
                layer_data["source_state_key"] = target.state_key
                layer_data["source_state_keys"] = list(target.all_source_keys())
                layer_data["slice_index"] = target.slice_index
                layer_data["source_shape"] = target.source_shape
                layer_data["source_shapes"] = [list(x) for x in target.all_source_shapes()]
                layer_data["target_kind"] = target.kind
                layer_data["matrix_shape"] = target.matrix_shape
                layer_data["row_start"] = target.row_start
                layer_data["row_end"] = target.row_end
                layer_data["embedding_row_start"] = target.embedding_row_start
                layer_data["embedding_row_end"] = target.embedding_row_end
                layer_data["transpose_source"] = target.transpose_source
                layer_data["embedding_group"] = target.embedding_group
                if target.kind == "cq_ngram_chunked_shard":
                    layer_data["ngram_chunk_rows"] = CQ_NGRAM_CHUNK_ROWS
                    layer_data["ngram_embedding_dim"] = CQ_NGRAM_EMBED_DIM
                    layer_data["ngram_num_chunks"] = (
                        int(target.source_shape[0]) + CQ_NGRAM_CHUNK_ROWS - 1
                    ) // CQ_NGRAM_CHUNK_ROWS
                    layer_data["ngram_valid_rows"] = int(target.source_shape[0])

                if save_incremental:
                    self.save_quantized_layer(
                        save_path,
                        layer_name,
                        layer_data,
                        state_dict=None,  # do not pass a full state dict
                    )

                    # Keep only small metadata in memory.
                    self.quantized_layers[layer_name] = {
                        "T": int(layer_data["T"]),
                        "original_shape": layer_data["original_shape"],
                        "source_state_key": target.state_key,
                        "slice_index": target.slice_index,
                        "source_shape": target.source_shape,
                        "target_kind": target.kind,
                        "matrix_shape": target.matrix_shape,
                        "_saved": True,
                    }

                    for key in [
                        "packed_all",
                        "u_W",
                        "std_W",
                        "SigRec1_select_packed",
                        "SigRec2_select_packed",
                        "SigRec3_select_packed",
                        "SigRec4_select_packed",
                        "X567_packed",
                        "bias",
                    ]:
                        layer_data.pop(key, None)

                    if os.environ.get("CQ_AGGRESSIVE_CUDA_CLEANUP") == "1":
                        torch.cuda.empty_cache()

                if verbose:
                    print(f"  [CQ] Successfully quantized {layer_name}")

            # except Exception as error:
            #     skipped += 1
            #     if verbose:
            #         print(
            #             f"  [CQ] Error quantizing "
            #             f"{layer_name}: {error}"
            #         )
            except Exception as error:
                skipped += 1

                import traceback

                print(
                    f"\n[CQ] Error quantizing {layer_name}\n"
                    f"  exception type: {type(error).__name__}\n"
                    f"  exception repr: {error!r}\n"
                )

                traceback.print_exc()


            finally:
                if weight_tensor is not None:
                    del weight_tensor

                if bias_tensor is not None:
                    del bias_tensor

                if os.environ.get("CQ_AGGRESSIVE_CUDA_CLEANUP") == "1":
                    gc.collect()
                    torch.cuda.empty_cache()

                if verbose:
                    print(
                        f"  GPU memory after cleanup: "
                        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    )

        coverage = (
            quantized_parameter_count / target_parameter_count
            if target_parameter_count
            else 0.0
        )

        if verbose:
            print("\n[CQ] Layer-by-layer quantization complete!")
            print(
                f"[CQ] Quantized {q} new targets, "
                f"resumed {q_resumed}, skipped {skipped}"
            )
            print(
                f"[CQ] Represented parameters: "
                f"{quantized_parameter_count / 1e9:.3f}B / "
                f"{target_parameter_count / 1e9:.3f}B "
                f"({coverage:.2%})"
            )
            print(
                f"[CQ] Expected raw INT4 payload: "
                f"{quantized_parameter_count / 2 / 2**30:.2f} GiB"
            )

        stats = {
            "quantized_targets": q,
            "resumed_targets": q_resumed,
            "skipped_targets": skipped,
            "total_targets": len(targets),
            "quantized_parameters": quantized_parameter_count,
            "target_parameters": target_parameter_count,
            "parameter_coverage": coverage,
            "total_modules": total_modules,
        }

        # Return streaming state only if you need metadata.
        # Do not treat this as Dict[str, Tensor].
        return stats, None



    def _write_layers_index_atomic(self, save_path, layer_names) -> None:
        save_path = pathlib.Path(save_path)
        quant_dir = save_path / "quantization_data"
        quant_dir.mkdir(parents=True, exist_ok=True)

        final_path = quant_dir / "_layers.json"
        tmp_path = quant_dir / "_layers.json.tmp"

        ordered = list(dict.fromkeys(str(x) for x in layer_names))

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, final_path)

    def _read_layers_index_safe(self, save_path) -> list[str]:
        path = pathlib.Path(save_path) / "quantization_data" / "_layers.json"
        if not path.exists():
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, list):
                raise TypeError(f"expected list, got {type(obj).__name__}")
            return [str(x) for x in obj]
        except Exception as exc:
            print(f"[CQ] WARNING: unreadable _layers.json: {exc!r}")
            print("[CQ] It will be rebuilt from verified saved records.")
            return []

    def _ensure_layer_in_layers_index(self, save_path, layer_name: str) -> None:
        layers = self._read_layers_index_safe(save_path)
        layer_name = str(layer_name)
        if layer_name not in layers:
            layers.append(layer_name)
            self._write_layers_index_atomic(save_path, layers)

    def _rebuild_layers_index_from_saved_targets(
        self,
        save_path,
        targets,
        *,
        verbose: bool = True,
    ) -> list[str]:
        """Recreate _layers.json from records that actually exist on disk."""
        saved = []
        seen = set()

        if verbose:
            print(
                f"[CQ] Rebuilding _layers.json from "
                f"{len(targets)} discovered targets..."
            )

        for target in targets:
            name = str(target.target_name)
            if name in seen:
                continue
            if self._is_layer_already_saved(save_path, name):
                saved.append(name)
                seen.add(name)

        self._write_layers_index_atomic(save_path, saved)

        if verbose:
            print(
                f"[CQ] _layers.json rebuilt with "
                f"{len(saved)} verified saved records"
            )

        return saved

    def save_quantized_layer(self, save_path: str, layer_name: str, layer_data: Dict, 
                             state_dict: Optional[Dict[str, torch.Tensor]] = None):
        """
        Incrementally save a single layer's quantization data. This allows saving as we go
        for memory-efficient quantization of very large models.
        
        Each layer is saved in a separate .pt file to avoid loading the entire quantization
        data file each time a new layer is saved.
        
        Args:
            save_path: Path to save directory
            layer_name: Name of the layer being saved
            layer_data: Quantization data for this layer (from quantized_layers dict)
            state_dict: Optional state dict to get bias from (for layer-by-layer quantization)
        """
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
    # # Create quantization_data subdirectory for individual layer files
    #     quant_dir = save_path / "quantization_data"
    #     quant_dir.mkdir(parents=True, exist_ok=True)
    #     lvl6_total_compressed_bytes = int(
    #         layer_data["lvl6_bits_packed_num_bytes"]
    #         .detach()
    #         .cpu()
    #         .item()
    #     )
    #     config_path = save_path / "config.json"

    #     # Load existing config, or create from HF config
    #     if config_path.exists():
    #         with open(config_path, "r") as f:
    #             cfg_dict = json.load(f)
    #     else:
    #         cfg_dict = _hf_config_to_dict(self.model.config)

    #     # Ensure quantization_config exists
    #     qcfg = cfg_dict.setdefault("quantization_config", {})
    #     qcfg["quant_method"] = "cq_quant"
    #     # Ensure per-layer dict exists
    #     per_layer_bytes = qcfg.setdefault("lvl6_total_compressed_bytes", {})

    #     # Add/update this layer
    #     per_layer_bytes[str(layer_name)] = int(lvl6_total_compressed_bytes)

    #     # Save config back
    #     with open(config_path, "w") as f:
    #         json.dump(cfg_dict, f, indent=2)

        # Create quantization_data subdirectory for individual layer files
        quant_dir = save_path / "quantization_data"
        quant_dir.mkdir(parents=True, exist_ok=True)
        
        # Create/update config.json if it doesn't exist
        if not (save_path / "config.json").exists():
            cfg_dict = _hf_config_to_dict(self.model.config)
            qcfg = cfg_dict.setdefault("quantization_config", {})
            qcfg["quant_method"] = "cq_quant"
            qcfg["cq_layout"] = "native_vllm"
            with open(save_path / "config.json", "w") as f:
                json.dump(cfg_dict, f, indent=2)
        
        # Prepare this layer's tensors.  N-gram physical shards already contain
        # final runtime bytes for many independent [160,1024] CQ matrices.
        layer_tensors = {}
        if layer_data.get("target_kind") == "cq_ngram_chunked_shard":
            layer_tensors["packed_all"] = layer_data["packed_all"].detach().to("cpu").contiguous()
            layer_tensors["u_W"] = layer_data["u_W"].detach().to("cpu").contiguous()
            layer_tensors["std_W"] = layer_data["std_W"].detach().to("cpu").contiguous()
        else:
            layer_tensors["u_W"] = layer_data["u_W"].detach().to("cpu").contiguous()
            layer_tensors["std_W"] = layer_data["std_W"].detach().to("cpu").contiguous()
            layer_tensors["SigRec1_select_packed"] = layer_data["SigRec1_select_packed"].detach().to("cpu").contiguous()
            layer_tensors["SigRec2_select_packed"] = layer_data["SigRec2_select_packed"].detach().to("cpu").contiguous()
            layer_tensors["SigRec3_select_packed"] = layer_data["SigRec3_select_packed"].detach().to("cpu").contiguous()
            layer_tensors["SigRec4_select_packed"] = layer_data["SigRec4_select_packed"].detach().to("cpu").contiguous()
            layer_tensors["X567_packed"] = layer_data["X567_packed"].detach().to("cpu").contiguous()

        # Handle bias
        layer_tensors["bias"] = None
        if layer_data.get("bias") is not None:
            layer_tensors["bias"] = layer_data["bias"].detach().to("cpu").contiguous()
        elif state_dict is not None:
            bias_key = f"{layer_name}.bias"
            if bias_key in state_dict:
                layer_tensors["bias"] = state_dict[bias_key].to("cpu").contiguous()
        else:
            # Try to get from model module
            try:
                mod = self.model.get_submodule(layer_name)
                bias = getattr(mod, "bias", None)
                if isinstance(bias, torch.nn.Parameter) and bias is not None:
                    layer_tensors["bias"] = bias.detach().to("cpu").contiguous()
            except Exception:
                pass
        
        # # Prepare metadata for this layer
        # layer_meta = {
        #     "T": int(layer_data["T"]),
        #     "original_shape": layer_data["original_shape"]
        # }

        layer_meta = {
            "T": int(layer_data["T"]),
            "original_shape": layer_data["original_shape"],

            # New fields.
            "source_state_key": layer_data.get("source_state_key"),
            "source_state_keys": layer_data.get("source_state_keys"),
            "slice_index": layer_data.get("slice_index"),
            "source_shape": layer_data.get("source_shape"),
            "source_shapes": layer_data.get("source_shapes"),
            "target_kind": layer_data.get("target_kind"),
            "matrix_shape": layer_data.get("matrix_shape"),
            "row_start": layer_data.get("row_start"),
            "row_end": layer_data.get("row_end"),
            "embedding_row_start": layer_data.get("embedding_row_start"),
            "embedding_row_end": layer_data.get("embedding_row_end"),
            "transpose_source": layer_data.get("transpose_source"),
            "embedding_group": layer_data.get("embedding_group"),
            "ngram_chunk_rows": layer_data.get("ngram_chunk_rows"),
            "ngram_embedding_dim": layer_data.get("ngram_embedding_dim"),
            "ngram_num_chunks": layer_data.get("ngram_num_chunks"),
            "ngram_valid_rows": layer_data.get("ngram_valid_rows"),
        }
        
        # Save this layer's data in a separate file
        # Sanitize layer_name for filename (replace dots and slashes with underscores)
        safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
        layer_file = quant_dir / f"{safe_layer_name}.pt"
        torch.save({"tensors": layer_tensors, "meta": layer_meta}, layer_file)
        
        # Keep _layers.json synchronized using an atomic write.
        self._ensure_layer_in_layers_index(save_path, layer_name)

        # Free memory
        del layer_tensors, layer_meta

    def _resume_layer_name_candidates(self, layer_name: str) -> list[str]:
        """Logical aliases used by the quantizer and converted CQ manifests."""
        name = str(layer_name)
        out = [name]

        def add(x: str):
            if x and x not in out:
                out.append(x)

        short_mm = "model.language_model."
        full_mm = "model.language_model.model."

        # HF/older multimodal -> native-vLLM multimodal.
        if name.startswith(short_mm) and not name.startswith(full_mm):
            rest = name[len(short_mm):]
            if rest.startswith(("layers.", "embed_tokens.", "norm.")):
                add(full_mm + rest)

        # Native-vLLM multimodal -> HF/older multimodal + text-only aliases.
        if name.startswith(full_mm):
            rest = name[len(full_mm):]
            add(short_mm + rest)
            add("model." + rest)

        # Text-only native -> multimodal aliases.
        if name.startswith("model.") and not name.startswith("model.language_model."):
            rest = name[len("model."):]
            add(full_mm + rest)
            add(short_mm + rest)

        # lm_head aliases.
        if name == "lm_head" or name.startswith("lm_head."):
            add("model.language_model." + name)
        if name.startswith("model.language_model.lm_head"):
            add(name[len("model.language_model."):])

        return out

    def _load_cq_resume_index(self, save_path: pathlib.Path):
        """Load and cache cq_shards.json."""
        cache_key = str(save_path.resolve())

        if not hasattr(self, "_cq_resume_index_cache"):
            self._cq_resume_index_cache = {}

        if cache_key in self._cq_resume_index_cache:
            return self._cq_resume_index_cache[cache_key]

        quant_dir = save_path / "quantization_data"

        for index_path in (
            quant_dir / "cq_shards.json",
        ):
            if not index_path.exists():
                continue
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                layers = index.get("layers")
                if isinstance(layers, dict):
                    result = (index_path, layers)
                    self._cq_resume_index_cache[cache_key] = result
                    return result
            except Exception as exc:
                print(
                    f"[CQ] WARNING: could not read resume index "
                    f"{index_path}: {exc}"
                )

        result = (None, None)
        self._cq_resume_index_cache[cache_key] = result
        return result

    def _resolve_cq_resume_shard_path(
        self,
        save_path: pathlib.Path,
        filename: str,
    ) -> pathlib.Path:
        """Resolve converter index filenames like quantization_data/cq-....safetensors."""
        quant_dir = save_path / "quantization_data"
        raw = pathlib.Path(str(filename))

        if raw.is_absolute():
            return raw

        repo_relative = save_path / raw
        if repo_relative.exists():
            return repo_relative

        basename_relative = quant_dir / raw.name
        if basename_relative.exists():
            return basename_relative

        return repo_relative

    def _cq_safetensor_record_exists(
        self,
        save_path: pathlib.Path,
        layer_name: str,
    ) -> bool:
        """Check one logical layer inside a converted multi-layer CQ shard."""
        index_path, layers = self._load_cq_resume_index(save_path)
        if index_path is None or not isinstance(layers, dict):
            return False

        location = None
        matched_name = None
        for candidate in self._resume_layer_name_candidates(layer_name):
            loc = layers.get(candidate)
            if isinstance(loc, dict):
                location = loc
                matched_name = candidate
                break

        if location is None:
            return False

        filename = location.get("filename")
        tensor_keys = location.get("tensor_keys")
        if not filename or not isinstance(tensor_keys, dict) or not tensor_keys:
            return False

        shard_path = self._resolve_cq_resume_shard_path(
            save_path,
            str(filename),
        )
        if not shard_path.exists():
            return False

        if not hasattr(self, "_cq_resume_shard_keys_cache"):
            self._cq_resume_shard_keys_cache = {}

        shard_cache_key = str(shard_path.resolve())
        stored_keys = self._cq_resume_shard_keys_cache.get(shard_cache_key)

        if stored_keys is None:
            try:
                with safe_open(
                    shard_path,
                    framework="pt",
                    device="cpu",
                ) as f:
                    stored_keys = set(f.keys())
                self._cq_resume_shard_keys_cache[shard_cache_key] = stored_keys
            except Exception as exc:
                print(
                    f"[CQ] WARNING: could not inspect CQ shard "
                    f"{shard_path}: {exc}"
                )
                return False

        required = {
            str(v)
            for v in tensor_keys.values()
            if isinstance(v, str) and v
        }
        if not required:
            return False

        ok = required.issubset(stored_keys)

        if ok and matched_name != layer_name:
            if not hasattr(self, "_cq_resume_aliases_reported"):
                self._cq_resume_aliases_reported = set()
            pair = (layer_name, matched_name)
            if pair not in self._cq_resume_aliases_reported:
                self._cq_resume_aliases_reported.add(pair)
                if len(self._cq_resume_aliases_reported) <= 10:
                    print(
                        f"  [CQ] Resume alias: "
                        f"{layer_name} -> {matched_name}"
                    )

        return ok

    def _is_layer_already_saved(self, save_path: str, layer_name: str) -> bool:
        """True if already saved in converted safetensors or legacy .pt."""
        save_path = pathlib.Path(save_path)
        quant_dir = save_path / "quantization_data"
        quant_file = save_path / "quantization_data.pt"

        # FIRST: converted local CQ layout. No .pt file is required.
        if quant_dir.exists() and self._cq_safetensor_record_exists(
            save_path,
            layer_name,
        ):
            return True

        # Legacy combined file.
        if quant_file.exists():
            try:
                bundle = torch.load(
                    quant_file,
                    map_location="cpu",
                    weights_only=False,
                )
                meta = bundle.get("meta", {})
                for candidate in self._resume_layer_name_candidates(layer_name):
                    if candidate in meta:
                        return True
            except Exception:
                pass

        # Legacy individual .pt files.
        if quant_dir.exists():
            for candidate in self._resume_layer_name_candidates(layer_name):
                safe_layer_name = candidate.replace(".", "_").replace("/", "_")
                layer_file = quant_dir / f"{safe_layer_name}.pt"
                if layer_file.exists():
                    return True

        return False
