"""Pick the base checkpoint for a model and a precision (cq / nf4 / bf16 / auto).

Used by the CLI on your laptop, and run as-is on the cloud machine when the CLI
could not reach Hugging Face. Standard library only.

  auto -> cq if thousandml/<name>-CQ-4bit exists, else nf4
  nf4  -> unsloth/<name>-unsloth-bnb-4bit or unsloth/<name>-bnb-4bit if published,
          else the full-precision model quantized to NF4 when loaded
  bf16 -> the full-precision model as-is
"""
import json
import os
import shlex
import sys
import urllib.error
import urllib.request


def hf_exists(repo):
    req = urllib.request.Request(f'https://huggingface.co/api/models/{repo}')
    if os.environ.get('HF_TOKEN'):
        req.add_header('Authorization', 'Bearer ' + os.environ['HF_TOKEN'])
    try:
        urllib.request.urlopen(req, timeout=15)
        return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return False
        raise


def local_kind(path):
    try:
        with open(os.path.join(path, 'config.json')) as f:
            qc = json.load(f).get('quantization_config') or {}
    except OSError:
        return None
    method = str(qc.get('quant_method', '')).lower()
    return 'cq' if method == 'cq' else 'nf4' if method == 'bitsandbytes' else 'bf16'


def resolve(model, precision, exists=hf_exists):
    """-> (precision, checkpoint to load, model the user named). ValueError says how to fix."""
    name = model.rstrip('/').split('/')[-1]

    if model.startswith(('/', '.', '~')) or os.path.isdir(model):        # a folder on the machine
        kind = local_kind(os.path.expanduser(model))
        if kind is None:
            if precision == 'auto':
                return 'auto', model, model                              # decided after download
            return precision, model, model
        if kind in ('cq', 'nf4'):
            if precision not in ('auto', kind):
                raise ValueError(f'{model} is a {kind} checkpoint; point --model at the full-precision '
                                 f'model to use --precision {precision}')
            return kind, model, model
        if precision == 'cq':
            raise ValueError(f'{model} is a full-precision folder; --precision cq needs a CQ checkpoint')
        return ('nf4' if precision == 'auto' else precision), model, model

    if '-CQ' in name.upper().replace('_', '-'):
        if precision not in ('auto', 'cq'):
            raise ValueError(f'{model} is a CQ checkpoint; pass the full-precision model '
                             f'(e.g. Qwen/Qwen3.8-27B) to use --precision {precision}')
        return 'cq', model, model
    if 'bnb-4bit' in name:
        if precision not in ('auto', 'nf4'):
            raise ValueError(f'{model} is an NF4 checkpoint; pass the full-precision model '
                             f'to use --precision {precision}')
        return 'nf4', model, model

    if precision in ('auto', 'cq'):
        cq = f'thousandml/{name}-CQ-4bit'
        if exists(cq):
            return 'cq', cq, model
        if precision == 'cq':
            raise ValueError(f'No CQ checkpoint for {model} (looked for {cq}). Use --precision nf4 '
                             'or bf16, or ask contact@thousandml.com to add this model.')
    if precision in ('auto', 'nf4'):
        for cand in (f'unsloth/{name}-unsloth-bnb-4bit', f'unsloth/{name}-bnb-4bit'):
            if exists(cand):
                return 'nf4', cand, model
        return 'nf4', model, model
    return 'bf16', model, model


if __name__ == '__main__':
    # On the cloud machine: print shell exports for the job script to eval.
    stage = sys.argv[1] if len(sys.argv) > 1 else 'pick'
    model, precision = os.environ['MODEL'], os.environ['PRECISION']
    if stage == 'after-download':                 # a folder whose kind we can now read
        kind = local_kind(os.environ['MODEL_DIR'])
        if precision == 'auto':
            precision = 'nf4' if kind in (None, 'bf16') else kind
        print(f'export PRECISION={shlex.quote(precision)}')
        sys.exit()
    try:
        p, ckpt, base = resolve(model, precision)
    except (ValueError, OSError) as e:          # OSError covers network failures
        print(f'echo {shlex.quote("error: " + str(e))} >&2; exit 2')
        sys.exit()
    print(f'export PRECISION={shlex.quote(p)} MODEL={shlex.quote(ckpt)} '
          f'BASE_MODEL={shlex.quote(os.environ.get("BASE_MODEL") or base)}')
