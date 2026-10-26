#!/usr/bin/env python3
"""Repair legacy CQ vision-tower manifest namespaces without touching tensor shards.

Dry-run by default. --apply writes both JSON files after validation, with backups.
"""
import argparse
import json
from pathlib import Path

BAD = (
    'model.language_model.model.vision_tower.',
    'model.language_model.vision_tower.',
    'model.language_model.model.visual.',
    'model.language_model.visual.',
)

def canonical(name):
    for prefix in BAD:
        if name.startswith(prefix):
            base = 'model.vision_tower.' if 'vision_tower.' in prefix else 'model.visual.'
            return base + name[len(prefix):]
    if name.startswith(('vision_tower.', 'visual.')):
        return 'model.' + name
    return name

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir', type=Path)
    ap.add_argument('--apply', action='store_true', help='Write changes; default is dry-run')
    args = ap.parse_args()
    q = args.model_dir / 'quantization_data'
    lp, ip = q / '_layers.json', q / 'cq_shards.json'
    layers = json.loads(lp.read_text())
    index = json.loads(ip.read_text())
    if not isinstance(layers, list) or not isinstance(index.get('layers'), dict):
        raise ValueError('Unexpected CQ manifest/index format')
    old_map = index['layers']
    fixed_map = {}
    for old, record in old_map.items():
        new = canonical(old)
        if new in fixed_map and fixed_map[new] != record:
            raise ValueError(f'Conflicting records after rename: {old} -> {new}')
        fixed_map[new] = record
    fixed_layers = list(dict.fromkeys(canonical(x) for x in layers))
    missing = [x for x in fixed_layers if x not in fixed_map]
    if missing:
        raise ValueError(f'{len(missing)} manifest entries missing in index: {missing[:10]}')
    changes = [(old, canonical(old)) for old in set(layers) | set(old_map) if old != canonical(old)]
    print(f'Entries: manifest={len(layers)} index={len(old_map)}; namespace renames={len(changes)}')
    for old, new in sorted(changes)[:20]:
        print(f'  {old} -> {new}')
    if len(changes) > 20:
        print(f'  ... and {len(changes)-20} more')
    if not args.apply:
        print('DRY RUN: no files changed. Use --apply to write.')
        return
    # Backups are never overwritten; rerunning is safe and idempotent.
    for path in (lp, ip):
        backup = path.with_name(path.name + '.before-vision-fix.bak')
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    index['layers'] = fixed_map
    for path, value in ((lp, fixed_layers), (ip, index)):
        temp = path.with_name(path.name + '.vision-fix.tmp')
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        temp.replace(path)
    print('Applied. Backups saved beside both JSON files. Safetensors unchanged.')

if __name__ == '__main__':
    main()
