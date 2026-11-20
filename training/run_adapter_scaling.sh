#!/usr/bin/env bash
# Mixed-adapter serving benchmark on one GPU, same CQ base and same vLLM:
#   base (no adapter) | GCMP x N | vLLM multi-LoRA r=32 x N | vLLM multi-LoRA r=8 x N
# Each request picks a random adapter, so batches mix tenants.
#
#   bash run_adapter_scaling.sh              # everything
#   CONFIGS="gcmp lora8" NS="16 256" bash run_adapter_scaling.sh
set -uo pipefail

ROOT=${ROOT:-/workspace/training-factory-gcmp}
MODEL=${MODEL:-/workspace/models/Qwen3.8-27B-CQ-4bit}
GCMP_SRC=${GCMP_SRC:-$ROOT/outputs/qwen38-27b-cq-4bit-alpaca-gcmp-v1}
LORA_SRC=${LORA_SRC:-$ROOT/outputs/qwen38-27b-cq-r32-v7/checkpoint-300}
WORK=${WORK:-/workspace/scaling}
PROMPTS=${PROMPTS:-$ROOT/eval_downstream/alpaca_val_eval.jsonl}
NS=${NS:-"1 16 64 256"}
CONFIGS=${CONFIGS:-"base gcmp lora32 lora8"}
LORA_SLOTS=${LORA_SLOTS:-16}          # vLLM --max-loras: LoRA adapters resident on GPU at once
# extra --max-loras values to sweep at the largest N (vLLM's per-batch adapter limit), per rank
SWEEP32=${SWEEP32:-"32"}; SWEEP8=${SWEEP8:-"32 64"}
CONC=${CONC:-32}; REQS=${REQS:-384}; MAXTOK=${MAXTOK:-128}
PORT=${PORT:-8011}
SERVE=(--quantization cq --max-model-len 2048 --max-num-seqs 64 --gpu-memory-utilization 0.92 --port "$PORT")
T=$ROOT/training
mkdir -p "$WORK/results"

maxn=0; for n in $NS; do (( n > maxn )) && maxn=$n; done
[[ -f $WORK/gcmp/adapters.json ]] || python "$T/make_adapter_copies.py" --kind gcmp --src "$GCMP_SRC" --out-dir "$WORK/gcmp" --n "$maxn"
[[ -f $WORK/lora32/adapters.json ]] || python "$T/make_adapter_copies.py" --kind lora --src "$LORA_SRC" --out-dir "$WORK/lora32" --n "$maxn"
[[ -f $WORK/lora8/adapters.json ]] || python "$T/make_adapter_copies.py" --kind lora --src "$LORA_SRC" --out-dir "$WORK/lora8" --n "$maxn" --rank 8

subset() {  # kind n -> writes first n entries, prints path
  local kind=$1 n=$2 f=$WORK/$1/first_$2.json
  python - "$WORK/$kind/adapters.json" "$n" "$f" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); n = int(sys.argv[2])
json.dump(dict(list(m.items())[:n]), open(sys.argv[3], 'w'), indent=1)
PY
  echo "$f"
}

wait_up() {
  for _ in $(seq 1 "${WAIT_TRIES:-540}"); do   # 10 s each; 256 LoRA modules can take a long time to register
    curl -sf "localhost:$PORT/v1/models" >/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

run_case() {  # tag names_file(or empty) env... -- vllm args...
  local tag=$1 names=$2; shift 2
  local out=$WORK/results/$tag.json log=$WORK/results/$tag.server.log
  [[ -f $out ]] && { echo "== $tag: done"; return; }
  echo "== $tag"
  local envs=() args=()
  while [[ $# -gt 0 && $1 != -- ]]; do envs+=("$1"); shift; done; shift
  args=("$@")
  env -u CQ_GCMP_ADAPTERS -u CQ_GCMP_ADAPTER CQ_MODEL_DIR="$MODEL" "${envs[@]}" vllm serve "$MODEL" "${SERVE[@]}" "${args[@]}" > "$log" 2>&1 &
  local pid=$!
  if ! wait_up $pid; then echo "!! $tag: server failed (see $log)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; return; fi
  local base_name; base_name=$(curl -s "localhost:$PORT/v1/models" | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  if [[ -n $names ]]; then
    python "$T/bench_multi_adapter.py" --url "http://localhost:$PORT" --names-file "$names" --prompts "$PROMPTS" \
      --concurrency "$CONC" --num-requests "$REQS" --max-tokens "$MAXTOK" --output "$out.tmp"
  else
    python "$T/bench_multi_adapter.py" --url "http://localhost:$PORT" --model "$base_name" --prompts "$PROMPTS" \
      --concurrency "$CONC" --num-requests "$REQS" --max-tokens "$MAXTOK" --output "$out.tmp"
  fi
  if [[ -f $out.tmp ]]; then
    kv=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$log" | tail -1 | grep -oE "[0-9,]+" | tr -d ,)
    python - "$out.tmp" "$out" "$tag" "${kv:-0}" "${CUR_SLOTS:-}" "${CUR_CPU:-}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d['config'] = sys.argv[3]; d['kv_cache_tokens'] = int(sys.argv[4])
d['max_loras'] = int(sys.argv[5]) if sys.argv[5] else None
d['max_cpu_loras'] = int(sys.argv[6]) if sys.argv[6] else None
json.dump(d, open(sys.argv[2], 'w'), indent=2)
PY
    rm -f "$out.tmp"
  fi
  kill $pid; wait $pid 2>/dev/null; sleep 5
}

for cfg in $CONFIGS; do
  if [[ $cfg == base ]]; then run_case base "" -- ; continue; fi
  for n in $NS; do
    f=$(subset "$cfg" "$n")
    case $cfg in
      gcmp)
        CUR_SLOTS=$n CUR_CPU= run_case "gcmp_n$n" "$f" CQ_GCMP_ADAPTERS="$f" -- ;;
      lora32|lora8)
        r=32; [[ $cfg == lora8 ]] && r=8
        mods=$(python -c 'import json,sys; print(" ".join(f"{k}={v}" for k,v in json.load(open(sys.argv[1])).items()))' "$f")
        sweep=$LORA_SLOTS
        if [[ $n -eq $maxn || -n ${SWEEP_ALL:-} ]]; then [[ $cfg == lora32 ]] && sweep="$LORA_SLOTS $SWEEP32" || sweep="$LORA_SLOTS $SWEEP8"; fi
        for want in $sweep; do
          slots=$(( n < want ? n : want ))
          tag="${cfg}_n${n}"; [[ $slots -ne $(( n < LORA_SLOTS ? n : LORA_SLOTS )) ]] && tag="${tag}_slots${slots}"
          CUR_SLOTS=$slots CUR_CPU=$n run_case "$tag" "$f" -- --enable-lora --max-lora-rank "$r" \
            --max-loras "$slots" --max-cpu-loras "$n" --lora-modules $mods
        done ;;
    esac
  done
done

python - "$WORK/results" <<'PY'
import json, sys, glob, os
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json')))]
order = {'base': 0, 'gcmp': 1, 'lora32': 2, 'lora8': 3}
rows.sort(key=lambda r: (order.get(r['config'].split('_n')[0], 9), r['adapters'], r.get('max_loras') or 0))
lines = ['| config | adapters | max_loras (GPU) | max_cpu_loras | output tok/s | req/s | latency p50 / p95 (s) | TTFT p50 / p95 (s) | KV cache tokens |',
         '|---|--:|--:|--:|--:|--:|--:|--:|--:|']
for r in rows:
    lines.append(f"| {r['config']} | {r['adapters']} | {r.get('max_loras') or '-'} | {r.get('max_cpu_loras') or '-'} | {r['output_tok_per_s']} | {r['req_per_s']} | "
                 f"{r['latency_p50_s']} / {r['latency_p95_s']} | {r['ttft_p50_s']} / {r['ttft_p95_s']} | {r['kv_cache_tokens']} |")
open(os.path.join(sys.argv[1], 'summary.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
