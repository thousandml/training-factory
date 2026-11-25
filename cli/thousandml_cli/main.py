"""thousandml: train and serve ThousandML adapters on your own cloud with one command.

Jobs run in YOUR cloud account (AWS, GCP, Azure, Lambda, RunPod, Kubernetes, ...)
through SkyPilot, using your own credentials. Data and adapters stay in your
buckets; ThousandML never sees them.

  thousandml check
  thousandml train support-v1 --data s3://acme-ml/support/train.jsonl \\
                   --output s3://acme-ml/adapters --adapter gcmp
  thousandml serve prod --adapters s3://acme-ml/adapters --replicas 1-3
  thousandml status
  thousandml logs support-v1
  thousandml down support-v1

Base precision (--precision): cq (ThousandML 4-bit), nf4 (bitsandbytes 4-bit) or
bf16 (full precision). The default, auto, uses CQ when a CQ checkpoint exists for
the model and NF4 otherwise.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error

import yaml

from . import gpu_select, remote_resolve

DEFAULT_IMAGE = os.environ.get('THOUSANDML_IMAGE', 'docker:thousandml/training-factory:latest')
DEFAULT_MODEL = 'Qwen/Qwen3.8-27B'
# Used only when the model's size can't be read (offline, private repo without HF_TOKEN).
# Otherwise GPUs are chosen from the checkpoint size: see gpu_select.py.
GPUS_4BIT = 'A100-80GB:1,H100:1,L40S:1,A40:1'
GPUS_BF16 = 'A100-80GB:1,H100:1,H200:1'
PRECISIONS = ('auto', 'cq', 'nf4', 'bf16')
REPO = '/opt/training-factory'                     # where the image keeps the repo
# Sent to the job only if set in your shell. While the ThousandML license server allows
# community access, no license key is needed; HF_TOKEN only for private/gated models.
SECRET_ENVS = ('CQ_LICENSE_KEY', 'CQ_LICENSE_URL', 'HF_TOKEN')
HERE = pathlib.Path(__file__).parent


def embed(script, dest):
    """Shell lines that write one of our helper scripts onto the cloud machine."""
    return f"cat > {dest} <<'THOUSANDML_EOF'\n{(HERE / script).read_text()}THOUSANDML_EOF"


def accelerators(spec):
    """'A100:1,A40:1' -> {'A100': 1, 'A40': 1} (SkyPilot picks any one of them)."""
    out = {}
    for part in spec.split(','):
        name, _, count = part.strip().partition(':')
        out[name] = int(count or 1)
    return out if len(out) > 1 else f'{next(iter(out))}:{next(iter(out.values()))}'


def is_bucket(path):
    return '://' in path and path.split('://', 1)[0] in ('s3', 'gs', 'r2', 'cos', 'oci', 'https')


def mount(task, target, source, mode='COPY'):
    if is_bucket(source):
        task.setdefault('file_mounts', {})[target] = {'source': source, 'mode': mode}
    else:
        path = os.path.abspath(os.path.expanduser(source))
        if target == '/outputs':
            os.makedirs(path, exist_ok=True)       # an output folder may not exist yet
        elif not os.path.exists(path):
            sys.exit(f'error: {source} not found (looked for {path})')
        task.setdefault('file_mounts', {})[target] = path


def resolve_locally(a):
    """Settle the base before paying for a GPU. If Hugging Face is unreachable from
    here, the job settles it on the cloud machine with the same code."""
    try:
        prec, ckpt, base = remote_resolve.resolve(a.model, a.precision)
    except ValueError as e:
        sys.exit(f'error: {e}')
    except (urllib.error.URLError, OSError) as e:
        print(f'note: could not reach Hugging Face from here ({e.__class__.__name__}); '
              'the job will choose the base when it starts.', file=sys.stderr)
        return a.precision, a.model, a.model
    why = ''
    if a.precision == 'auto' and prec != 'auto':
        why = ' (auto: CQ checkpoint found)' if prec == 'cq' else ' (auto: no CQ checkpoint for this model)'
    print(f'base: {prec} -> {ckpt}{why}', file=sys.stderr)
    return prec, ckpt, base


def base_task(name, a):
    prec, ckpt, base = resolve_locally(a)
    if prec == 'nf4' and a.adapter_kind_hint == 'gcmp-serve':
        print('note: vLLM serves GCMP adapters on CQ and full-precision (bf16) bases; this nf4 service '
              'will serve LoRA adapters and skip GCMP ones.', file=sys.stderr)
    pp = 1
    if a.gpus:
        gpus = a.gpus
    else:
        job = 'serve' if a.cmd == 'serve' else 'train'
        try:
            gpus, pp, why = gpu_select.choose(prec, ckpt, base, job)
        except ValueError as e:
            sys.exit(f'error: {e}')
        if gpus is None:
            gpus = GPUS_BF16 if prec == 'bf16' else GPUS_4BIT
            print(f'gpus: {gpus} (default; {why})', file=sys.stderr)
        else:
            print(f'gpus: {gpus} ({why})', file=sys.stderr)
    task = {
        'name': name,
        'resources': {'accelerators': accelerators(gpus), 'image_id': a.image, 'disk_size': a.disk},
        'envs': {'MODEL': ckpt, 'BASE_MODEL': base, 'PRECISION': prec, 'HF_HUB_ENABLE_HF_TRANSFER': '1',
                 'PP_SIZE': str(pp)},
    }
    secrets = {k: None for k in SECRET_ENVS if os.environ.get(k)}   # values come from your shell at launch
    if secrets:
        task['secrets'] = secrets
    if a.cloud:
        task['resources']['cloud'] = a.cloud
    if a.spot:
        task['resources']['use_spot'] = True
    return task


# On the cloud machine: settle the base (no-op if the CLI already did), then download it.
PREPARE = '\n'.join([
    embed('remote_resolve.py', '/tmp/tml_resolve.py'),
    'eval "$(python /tmp/tml_resolve.py)"',
    'mkdir -p /models',
    'if [ -d "$MODEL" ]; then MODEL_DIR="$MODEL"; '
    'else MODEL_DIR=/models/$(basename "$MODEL"); '
    '[ -f "$MODEL_DIR/config.json" ] || hf download "$MODEL" --local-dir "$MODEL_DIR"; fi',
    'export MODEL_DIR',
    'if [ "$PRECISION" = auto ]; then eval "$(python /tmp/tml_resolve.py after-download)"; fi',
    'echo "base: $PRECISION ($MODEL)"',
])


def train_task(a):
    task = base_task(a.name, a)
    mount(task, '/data/train.jsonl', a.data)
    if a.eval_data:
        mount(task, '/data/eval.jsonl', a.eval_data)
    mount(task, '/outputs', a.output, mode='MOUNT')
    task['envs'].update({'RUN': a.name, 'ADAPTER': a.adapter})

    shared = [f'--adapter {a.adapter}', f'--max-steps {a.max_steps}', f'--max-length {a.max_length}',
              f'--batch-size {a.batch_size}', f'--grad-accum {a.grad_accum}',
              f'--learning-rate {a.learning_rate}', '--kernel-module cq.cq_kernels']
    extra = [a.extra] if a.extra else []
    cq = (['python training/train.py', '--model "$MODEL_DIR"', '--data /data/train.jsonl',
           '--output "$WORK"'] + shared + ['--comparison-logs --benchmark']
          + (['--eval-data /data/eval.jsonl'] if a.eval_data else [])
          + ([f'--rank {a.rank} --alpha {a.rank} --lora-dropout 0'] if a.adapter == 'lora' else [])
          + extra)
    unsloth = (['--model "$MODEL_DIR"', '--train-data /tmp/train.jsonl', '--output "$WORK"'] + shared
               + (['--eval-data /tmp/eval.jsonl'] if a.eval_data else [])
               + ([f'--lora-rank {a.rank} --lora-alpha {a.rank}'] if a.adapter == 'lora' else [])
               + extra)
    to_text = ['python /tmp/tml_to_text.py /data/train.jsonl /tmp/train.jsonl "$MODEL_DIR"']
    if a.eval_data:
        to_text.append('python /tmp/tml_to_text.py /data/eval.jsonl /tmp/eval.jsonl "$MODEL_DIR"')
    joined = lambda parts: ' \\\n      '.join(parts)

    task['run'] = '\n'.join([
        'set -euo pipefail',
        PREPARE,
        embed('remote_to_text.py', '/tmp/tml_to_text.py'),
        f'cd {REPO}',
        'WORK=/tmp/run && mkdir -p "$WORK"',
        'case "$PRECISION" in',
        '  cq)',
        '    ' + joined(cq),
        '    SRC="$WORK" ;;',
        '  nf4)',
        *('    ' + x for x in to_text),
        '    ' + joined(['python training/train_unsloth_nf4.py'] + unsloth),
        '    SRC="$WORK/final_adapter" ;;',
        '  bf16)',
        *('    ' + x for x in to_text),
        '    ' + joined(['python training/train_unsloth_16bit.py'] + unsloth),
        '    SRC="$WORK/final_adapter" ;;',
        '  *) echo "unknown precision: $PRECISION" >&2; exit 2 ;;',
        'esac',
        # only the final adapter goes to your bucket; checkpoints stay on the machine
        'mkdir -p "/outputs/$RUN" && cp -r "$SRC"/. "/outputs/$RUN"/',
        'printf \'{"precision": "%s", "model": "%s", "base_model": "%s", "adapter": "%s"}\\n\' '
        '"$PRECISION" "$MODEL" "$BASE_MODEL" "$ADAPTER" > "/outputs/$RUN/thousandml.json"',
        'echo "Adapter ($ADAPTER on a $PRECISION base) saved to /outputs/$RUN"',
    ])
    return task


def serve_task(a):
    task = base_task(a.name, a)
    task['resources']['ports'] = 8000
    task['envs'].update({'MAX_MODEL_LEN': str(a.max_model_len), 'MAX_LORAS': str(a.max_loras)})
    if a.adapters:
        mount(task, '/adapters', a.adapters)
    lo, _, hi = a.replicas.partition('-')
    service = {'readiness_probe': {'path': '/v1/models', 'initial_delay_seconds': 1800},
               'replica_policy': {'min_replicas': int(lo), 'max_replicas': int(hi or lo)}}
    if hi and hi != lo:
        service['replica_policy']['target_qps_per_replica'] = a.target_qps
    task['service'] = service
    task['run'] = '\n'.join([
        'set -euo pipefail',
        PREPARE,
        embed('remote_serve.py', '/tmp/tml_serve.py'),
        'exec python /tmp/tml_serve.py',
    ])
    return task


def run(cmd, dry):
    print('$ ' + ' '.join(cmd), flush=True)
    if dry:
        return 0
    if not shutil.which(cmd[0]):
        sys.exit('SkyPilot is not installed: pip install "skypilot[aws]" (or [gcp], [azure], [runpod], [kubernetes])')
    return subprocess.call(cmd)


def launch(task, cmd, a):
    text = yaml.safe_dump(task, sort_keys=False)
    if a.dry_run:
        print(text)
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, prefix='thousandml-') as f:
        f.write(text)
    secrets = [x for k in SECRET_ENVS if os.environ.get(k) for x in ('--secret', k)]
    return run(cmd + [f.name] + secrets, a.dry_run)


def add_common(p):
    p.add_argument('name', help='job or service name')
    p.add_argument('--model', default=DEFAULT_MODEL,
                   help='the model, ideally its full-precision Hugging Face id (default %(default)s). '
                        'A CQ or NF4 checkpoint id, or a folder on the machine, also works')
    p.add_argument('--precision', choices=PRECISIONS, default='auto',
                   help='base the adapter trains or serves on: cq (ThousandML 4-bit), nf4 (bitsandbytes '
                        '4-bit) or bf16 (full precision). auto = cq if a CQ checkpoint exists, else nf4')
    p.add_argument('--gpus', help='acceptable GPUs, e.g. "A100-80GB:1,A40:1" (cheapest available wins). '
                                  'Default: chosen from the model size, the precision and the job')
    p.add_argument('--cloud', help='pin a cloud: aws, gcp, azure, runpod, lambda, kubernetes, ...')
    p.add_argument('--spot', action='store_true', help='use spot/preemptible GPUs (cheaper, may be interrupted)')
    p.add_argument('--disk', type=int, default=200, help='disk GB')
    p.add_argument('--image', default=DEFAULT_IMAGE)
    p.add_argument('--dry-run', action='store_true', help='print the SkyPilot task and command, launch nothing')


def main(argv=None):
    p = argparse.ArgumentParser(prog='thousandml', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='fine-tune an adapter on your cloud')
    add_common(t)
    t.add_argument('--data', required=True, help='training JSONL ({"text"} or {"messages"} rows): '
                                                 'local path or bucket URL (s3://, gs://, ...)')
    t.add_argument('--eval-data')
    t.add_argument('--output', required=True, help='bucket URL (or local dir) for the adapter')
    t.add_argument('--adapter', choices=['gcmp', 'lora'], default='gcmp')
    t.add_argument('--rank', type=int, default=16, help='LoRA rank (alpha = rank)')
    t.add_argument('--max-steps', type=int, default=500)
    t.add_argument('--max-length', type=int, default=1024)
    t.add_argument('--batch-size', type=int, default=2)
    t.add_argument('--grad-accum', type=int, default=8)
    t.add_argument('--learning-rate', default='2e-5')
    t.add_argument('--extra', help='extra flags passed to the trainer')
    t.add_argument('--keep', action='store_true', help='keep the machine after training (default: shut down)')

    s = sub.add_parser('serve', help='serve the base model + adapters behind an OpenAI-compatible endpoint')
    add_common(s)
    s.add_argument('--adapters', help='bucket URL (or local dir) whose sub-folders are adapters')
    s.add_argument('--replicas', default='1', help='"1" or a range like "1-4" for autoscaling')
    s.add_argument('--target-qps', type=float, default=2.0, help='autoscaling target requests/s per replica')
    s.add_argument('--max-model-len', type=int, default=8192)
    s.add_argument('--max-loras', type=int, default=16, help='LoRA only: adapters kept on the GPU at once')
    s.add_argument('--single', action='store_true',
                   help='run on one plain machine (no service controller or autoscaling); good for testing. '
                        'Endpoint: sky status --endpoint 8000 NAME')

    sub.add_parser('check', help='show which clouds your credentials can use')
    sub.add_parser('status', help='list jobs and services')
    lg = sub.add_parser('logs', help='stream logs of a job')
    lg.add_argument('name')
    dn = sub.add_parser('down', help='stop a job or service')
    dn.add_argument('name')
    dn.add_argument('--service', action='store_true', help='force service mode (auto-detected otherwise)')

    a = p.parse_args(argv)
    a.adapter_kind_hint = 'gcmp-serve' if a.cmd == 'serve' else None
    if a.cmd == 'train':
        if not is_bucket(a.output) and not a.keep:
            a.keep = True
            print(f'note: --output is a local folder, so the adapter is written on the cloud machine. '
                  f'The machine is kept; when training finishes copy it back and stop the machine:\n'
                  f'  rsync -Pavz {a.name}:/outputs/{a.name} {a.output.rstrip("/")}/ && sky down -y {a.name}',
                  file=sys.stderr)
        cmd = ['sky', 'launch', '-y', '-c', a.name] + ([] if a.keep else ['--down'])
        return launch(train_task(a), cmd, a)
    if a.cmd == 'serve':
        task = serve_task(a)
        if a.single:                       # one plain machine, no service controller or autoscaling
            task.pop('service')
            return launch(task, ['sky', 'launch', '-y', '-c', a.name], a)
        return launch(task, ['sky', 'serve', 'up', '-y', '-n', a.name], a)
    if a.cmd == 'check':
        return run(['sky', 'check'], False)
    if a.cmd == 'status':
        run(['sky', 'status'], False)
        return run(['sky', 'serve', 'status'], False)
    if a.cmd == 'logs':
        return run(['sky', 'logs', a.name], False)
    if a.cmd == 'down':
        st = subprocess.run(['sky', 'serve', 'status', a.name], capture_output=True, text=True)
        is_service = a.service or (st.returncode == 0 and a.name in st.stdout)
        return run(['sky', 'serve', 'down', '-y', a.name] if is_service else ['sky', 'down', '-y', a.name], False)


if __name__ == '__main__':
    sys.exit(main())
