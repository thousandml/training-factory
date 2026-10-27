"""Audit real CQ safetensor headers and HF mapping without CUDA/kernel imports."""
import argparse
import json
from pathlib import Path
from safetensors import safe_open
import torch
from model_factory import causal_lm_skeleton
from checkpoint_layout import plan_records
from text_only_layout import text_only_index
from residual_names import resolve_residual
from residual_placeholders import is_cq_placeholder


def inspect(path, mapping_path=None, *, text_only=False, trust_remote_code=False):
    root = Path(path)
    index = json.loads((root / 'quantization_data/cq_shards.json').read_text())
    if index.get('format') != 'cq_safetensors_shards_v1':
        raise ValueError('Expected cq_safetensors_shards_v1, not PT/pickle storage')
    names = json.loads((root / 'quantization_data/_layers.json').read_text())
    if len(names) != len(set(names)) or set(names) != set(index['layers']):
        raise ValueError('Manifest duplicates or shard-index mismatch')
    model, cfg = causal_lm_skeleton(root, text_only=text_only, trust_remote_code=trust_remote_code)
    linears = {n: [m.out_features,m.in_features] for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)}
    mapping = json.loads(Path(mapping_path).read_text()) if mapping_path else {}
    plans = plan_records(text_only_index(index, linears, text_only=text_only),linears,mapping, skip_unmodeled_auxiliary=text_only)
    quantized = {n for p in plans for n in p['destinations']}
    for plan in plans:
        loc = index['layers'][plan['record']]
        if not plan['filename']:
            raise ValueError(f"Missing filename for {plan['record']}")
        shard = root / plan['filename']
        if not shard.is_file():
            shard = root / 'quantization_data' / plan['filename']
        m,n = plan['shape']
        b = (m*((n+31)//32*32)+1023)//1024
        with safe_open(str(shard),framework='pt',device='cpu') as f:
            keys = loc['tensor_keys']
            expected = {'packed_all':[b,520]} if 'packed_all' in keys else {
                'SigRec1_select_packed':[b,7], 'SigRec2_select_packed':[b,12],
                'SigRec3_select_packed':[b,29], 'SigRec4_select_packed':[b,83],
                'X567_packed':[3*b,128]}
            for logical, shape in expected.items():
                view = f.get_slice(keys[logical])
                if view.get_shape() != shape or view.get_dtype() != 'U8':
                    raise ValueError(f"{plan['record']}: invalid {logical} header")
            for scale in ('u_W','std_W'):
                view = f.get_slice(keys[scale])
                shape = view.get_shape()
                if any(d != 1 for d in shape) or view.get_dtype() not in ('F16','BF16','F32','F64'):
                    raise ValueError(f"{plan['record']}: invalid {scale} header")
        plan['verified_shard'] = str(shard)
    residual_index = root / 'model.safetensors.index.json'
    files = sorted(set(json.loads(residual_index.read_text())['weight_map'].values())) if residual_index.exists() else ['model.safetensors']
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    targets = {**parameters,**buffers}
    loaded = set()
    covered_alias_ids = set()
    for filename in files:
        with safe_open(str(root / filename),framework='pt',device='cpu') as f:
            for name in f.keys():
                if name.endswith('.weight') and name[:-7] in quantized:
                    continue
                destination = resolve_residual(name, targets, text_only=text_only)
                if destination is None:
                    continue
                shape = f.get_slice(name).get_shape()
                if is_cq_placeholder(name, shape, destination, quantized):
                    continue
                if list(targets[destination].shape) != shape:
                    raise ValueError(f'Residual shape mismatch: {name} -> {destination}; checkpoint_shape={f.get_slice(name).get_shape()}, model_shape={list(targets[destination].shape)}, destination_quantized={destination.removesuffix(".weight") in quantized}. Check whether this projection is represented by a fused CQ record or has an architecture-specific layout.')
                loaded.add(destination)
                covered_alias_ids.add(id(targets[destination]))
    missing = [n for n,t in parameters.items() if not (n.endswith('.weight') and n[:-7] in quantized)
               and n not in loaded and id(t) not in covered_alias_ids]
    if missing:
        raise ValueError(f'Missing residual parameters: {missing[:30]}')
    return {'status':'headers_and_mapping_verified', 'model_type':cfg.model_type,
            'records':len(plans),'hf_quantized_projections':len(quantized),
            'residual_tensors':len(loaded),'plans':plans,
            'note':'Header/layout audit only; CUDA numerical parity and vLLM LoRA support are separate tests.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--layer-map')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--text-only', action='store_true', help='Exclude auxiliary CQ records only when absent from the CausalLM skeleton')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(inspect(args.model,args.layer_map,text_only=args.text_only,trust_remote_code=args.trust_remote_code),indent=2)
    if args.output:
        Path(args.output).write_text(report)
    print(report)
