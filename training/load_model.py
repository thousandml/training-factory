"""Strict loader for local CQ safetensor checkpoints and HF linear names."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from cq_linear import CQLinear, CQOutputSlice
from model_factory import causal_lm_skeleton
from checkpoint_layout import plan_records
from residual_names import resolve_residual
from residual_placeholders import is_cq_placeholder


def load_model(path, mapping_path=None, trust_remote_code=False, *, text_only=False):
    root = Path(path)
    quant = root / 'quantization_data'
    index = json.loads((quant / 'cq_shards.json').read_text())
    if index.get('format') != 'cq_safetensors_shards_v1':
        raise ValueError('Convert to cq_safetensors_shards_v1 before training')
    names = set(json.loads((quant / '_layers.json').read_text()))
    if names != set(index['layers']):
        raise ValueError('Manifest and shard index disagree')
    mapping = json.loads(Path(mapping_path).read_text()) if mapping_path else {}
    if set(mapping) - names:
        raise ValueError('Mapping contains unknown record names')
    model, cfg = causal_lm_skeleton(root, trust_remote_code=trust_remote_code, text_only=text_only)
    linears = {name: [module.out_features, module.in_features]
               for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    plans = plan_records(index, linears, mapping, skip_unmodeled_auxiliary=text_only)
    destinations = set()
    for plan in plans:
        record_name = plan['record']
        location = index['layers'][record_name]
        filename = plan['filename']
        if not filename:
            raise ValueError(f'Missing shard filename for {record_name}')
        shard = root / filename
        if not shard.is_file():
            shard = quant / filename
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            rec = {k: f.get_tensor(v) for k, v in location['tensor_keys'].items()}
        rec = pack_record(rec, plan['shape'])
        m, n = plan['shape']
        base = CQLinear(n, m, rec, device='cuda')
        base._cq_fused_expected = len(plan['destinations'])
        for name, (start, end) in zip(plan['destinations'], plan['slices']):
            old = model.get_submodule(name)
            if len(plan['destinations']) == 1:
                new = base
                if old.bias is not None:
                    new.bias = torch.empty(old.out_features, device='meta', dtype=torch.float16)
            else:
                new = CQOutputSlice(base, start, end, has_bias=old.bias is not None)
            parent_name, _, attr = name.rpartition('.')
            setattr(model.get_submodule(parent_name) if parent_name else model, attr, new)
            destinations.add(name)
        print(f"CQ {record_name} -> {list(zip(plan['destinations'], plan['slices']))}")
    model._cq_load_report = plans
    residual_index = root / 'model.safetensors.index.json'
    if residual_index.exists():
        files = sorted(set(json.loads(residual_index.read_text())['weight_map'].values()))
    else:
        files = ['model.safetensors']
    # Assign all aliases of tied parameters to the same loaded tensor.
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    alias_lookup = {name: group for group in aliases.values() for name in group}
    residual_targets = {**dict(model.named_parameters(remove_duplicate=False)),
                        **dict(model.named_buffers(remove_duplicate=False))}
    for filename in files:
        with safe_open(str(root / filename), framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.endswith('.weight') and key[:-7] in destinations:
                    continue
                destination = resolve_residual(key, residual_targets, text_only=text_only)
                if destination is None:
                    continue
                if is_cq_placeholder(key, f.get_slice(key).get_shape(), destination, destinations):
                    continue
                parent_name, _, attr = destination.rpartition('.')
                try:
                    parent = model.get_submodule(parent_name) if parent_name else model
                except AttributeError as e:
                    raise ValueError(f'Unexpected residual key {key}') from e
                if not hasattr(parent, attr):
                    raise ValueError(f'Unexpected residual key {key}')
                tensor = f.get_tensor(key).to('cuda')
                if tensor.is_floating_point():
                    tensor = tensor.half()
                old = getattr(parent, attr)
                if tuple(old.shape) != tuple(tensor.shape):
                    raise ValueError(f'Residual shape mismatch: {key} -> {destination}; checkpoint_shape={list(tensor.shape)}, model_shape={list(old.shape)}, destination_quantized={destination.removesuffix(".weight") in destinations}')
                value = torch.nn.Parameter(tensor, requires_grad=False) if isinstance(old, torch.nn.Parameter) else tensor
                for alias in alias_lookup.get(destination, [destination]):
                    p, _, a = alias.rpartition('.')
                    setattr(model.get_submodule(p) if p else model, a, value)
    missing = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta]
    if missing:
        raise ValueError(f'Unloaded tensors: {missing[:30]}')
    if not destinations:
        raise ValueError('No CQ linear layers found')
    model.to('cuda')
    model.requires_grad_(False)
    model.config.use_cache = False
    return model


def pack_record(rec, shape):
    """Match _load_cq_record_into_packed's exact 520-byte layout."""
    m, n = shape
    b = (m * ((n + 31) // 32 * 32) + 1023) // 1024
    if 'packed_all' not in rec:
        packed = torch.zeros(b, 520, dtype=torch.uint8)
        for key, width, offset in [('SigRec1_select_packed', 7, 0),
                                   ('SigRec2_select_packed', 12, 8),
                                   ('SigRec3_select_packed', 29, 20),
                                   ('SigRec4_select_packed', 83, 52)]:
            value = rec[key]
            if value.dtype != torch.uint8 or tuple(value.shape) != (b, width):
                raise ValueError(f'Invalid {key} shape/dtype')
            packed[:, offset:offset + width].copy_(value)
        x = rec['X567_packed']
        if x.dtype != torch.uint8 or tuple(x.shape) != (3 * b, 128):
            raise ValueError('Invalid X567_packed shape/dtype')
        for i, offset in enumerate((136, 264, 392)):
            packed[:, offset:offset + 128].copy_(x[i*b:(i+1)*b])
        rec = dict(rec, packed_all=packed)
    return rec
