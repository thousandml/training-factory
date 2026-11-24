#!/usr/bin/env bash
# Long-run multi-tenant test with long-tail traffic and adapter turnover, one GPU,
# same CQ base. Complements run_adapter_scaling.sh (uniform traffic, fixed set).
#
#   skew : 256 adapters, Zipf traffic for 45 min. LoRA keeps at most LORA_CPU adapters
#          in host memory, so tail tenants are paged in from disk; GCMP keeps all resident.
#   churn: 64 adapters, one tenant replaced every 20 s while traffic runs (~135 swaps).
#          LoRA: runtime unload/load API. GCMP: in-place swap through gcmp_swap_ext.
#
#   bash run_turnover.sh                          # gcmp + lora8, skew + churn (~5 h)
#   KINDS="gcmp" MODES="churn" bash run_turnover.sh
#   KINDS="lora32" bash run_turnover.sh           # optional, ~3 h more
set -uo pipefail

ROOT=${ROOT:-/workspace/training-factory-gcmp}
MODEL=${MODEL:-/workspace/models/Qwen3.8-27B-CQ-4bit}
GCMP_SRC=${GCMP_SRC:-$ROOT/outputs/qwen38-27b-cq-4bit-alpaca-gcmp-v1}
LORA_SRC=${LORA_SRC:-$ROOT/outputs/qwen38-27b-cq-r32-v8-fix}
WORK=${WORK:-/workspace/scaling}
PROMPTS=${PROMPTS:-$ROOT/eval_downstream/alpaca_val_eval.jsonl}
KINDS=${KINDS:-"gcmp lora8"}
MODES=${MODES:-"skew churn"}
DURATION=${DURATION:-2700}; RATE=${RATE:-0.45}; ZIPF=${ZIPF:-1.1}; CHURN_EVERY=${CHURN_EVERY:-20}
SKEW_N=${SKEW_N:-256}; CHURN_N=${CHURN_N:-64}; INCOMING=${INCOMING:-128}
LORA_CPU=${LORA_CPU:-64}                         # --max-cpu-loras: adapters cached in host RAM
PORT=${PORT:-8011}
# BASE=cq (default): CQ 4-bit model with --quantization cq.
# BASE=bf16: a full-precision model, no --quantization; GCMP/LoRA adapters must be trained on it.
BASE=${BASE:-cq}
MAX_BATCHED=${MAX_BATCHED:-}          # --max-num-batched-tokens; bf16 27B on 80 GB needs ~4096
SERVE=(--max-model-len 2048 --max-num-seqs 64 --gpu-memory-utilization ${GPU_UTIL:-0.92} --port "$PORT")
[[ $BASE == cq ]] && SERVE=(--quantization cq "${SERVE[@]}")
[[ $BASE == bf16 ]] && SERVE+=(--limit-mm-per-prompt '{"image":0,"video":0}')
[[ -n $MAX_BATCHED ]] && SERVE+=(--max-num-batched-tokens "$MAX_BATCHED")
T=$ROOT/training
OUT=${OUT:-$WORK/turnover}
mkdir -p "$OUT"

copies() {  # kind -> pool dir (reuses run_adapter_scaling.sh copies when present)
  case $1 in
    gcmp)   [[ -f $WORK/gcmp/adapters.json ]]   || python "$T/make_adapter_copies.py" --kind gcmp --src "$GCMP_SRC" --out-dir "$WORK/gcmp" --n 256 ;;
    lora32) [[ -f $WORK/lora32/adapters.json ]] || python "$T/make_adapter_copies.py" --kind lora --src "$LORA_SRC" --out-dir "$WORK/lora32" --n 256 ;;
    lora8)  [[ -f $WORK/lora8/adapters.json ]]  || python "$T/make_adapter_copies.py" --kind lora --src "$LORA_SRC" --out-dir "$WORK/lora8" --n 256 --rank 8 ;;
  esac
  local inc=$WORK/${1}_incoming
  if [[ ! -f $inc/adapters.json ]]; then
    case $1 in
      gcmp)   python "$T/make_adapter_copies.py" --kind gcmp --src "$GCMP_SRC" --out-dir "$inc" --n "$INCOMING" --prefix new --seed-offset 100000 ;;
      lora32) python "$T/make_adapter_copies.py" --kind lora --src "$LORA_SRC" --out-dir "$inc" --n "$INCOMING" --prefix new --seed-offset 100000 ;;
      lora8)  python "$T/make_adapter_copies.py" --kind lora --src "$LORA_SRC" --out-dir "$inc" --n "$INCOMING" --prefix new --seed-offset 100000 --rank 8 ;;
    esac
  fi
}

first_n() {  # kind n -> json path with the first n adapters
  local f=$WORK/$1/first_$2.json
  python - "$WORK/$1/adapters.json" "$2" "$f" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); json.dump(dict(list(m.items())[:int(sys.argv[2])]), open(sys.argv[3], 'w'), indent=1)
PY
  echo "$f"
}

wait_up() {
  for _ in $(seq 1 "${WAIT_TRIES:-720}"); do      # 10 s each; LoRA registration is slow
    curl -sf "localhost:$PORT/v1/models" >/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

for kind in $KINDS; do
  copies "$kind"
  for mode in $MODES; do
    tag=${kind}_${mode}; res=$OUT/$tag.json; log=$OUT/$tag.server.log
    [[ -f $res ]] && { echo "== $tag: done"; continue; }
    n=$SKEW_N; [[ $mode == churn ]] && n=$CHURN_N
    pool=$(first_n "$kind" "$n")
    envs=(); args=()
    if [[ $kind == gcmp ]]; then
      envs=(CQ_GCMP_ADAPTERS="$pool")
      [[ $mode == churn ]] && { envs+=(VLLM_SERVER_DEV_MODE=1 PYTHONPATH="$T:${PYTHONPATH:-}"); args=(--worker-extension-cls gcmp_swap_ext.GCMPSwap); }
    else
      r=32; slots=${LORA_SLOTS32:-16}
      [[ $kind == lora8 ]] && { r=8; slots=${LORA_SLOTS8:-64}; }
      (( slots > n )) && slots=$n
      cpu=$LORA_CPU; (( cpu < slots )) && cpu=$slots
      mods=$(python -c 'import json,sys; print(" ".join(f"{k}={v}" for k,v in json.load(open(sys.argv[1])).items()))' "$pool")
      args=(--enable-lora --max-lora-rank "$r" --max-loras "$slots" --max-cpu-loras "$cpu" --lora-modules $mods)
      [[ $mode == churn ]] && envs=(VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)
    fi
    echo "== $tag ($n adapters) starting server"
    env -u CQ_GCMP_ADAPTERS -u CQ_GCMP_ADAPTER CQ_MODEL_DIR="$MODEL" "${envs[@]}" \
      vllm serve "$MODEL" "${SERVE[@]}" "${args[@]}" > "$log" 2>&1 &
    pid=$!
    t0=$(date +%s)
    if ! wait_up $pid; then echo "!! $tag: server failed (see $log)"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; continue; fi
    echo "   ready after $(( ($(date +%s) - t0) / 60 )) min; running $mode for $(( ${DURATION%.*} / 60 )) min"
    extra=(); [[ $mode == churn ]] && extra=(--incoming-file "$WORK/${kind}_incoming/adapters.json" --churn-every "$CHURN_EVERY")
    bkind=gcmp; [[ $kind == lora* ]] && bkind=lora
    (cd "$T" && python bench_turnover.py --url "http://localhost:$PORT" --kind $bkind --mode "$mode" \
       --names-file "$pool" --prompts "$PROMPTS" --duration "$DURATION" --rate "$RATE" --zipf "$ZIPF" \
       --server-pid $pid --output "$res" "${extra[@]}")
    python - "$res" "$(( $(date +%s) - t0 ))" "$tag" <<'PY' 2>/dev/null
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d['server_ready_and_run_s'] = int(sys.argv[2]); d['config'] = sys.argv[3]
json.dump(d, open(p, 'w'), indent=2)
PY
    kill $pid; wait $pid 2>/dev/null; sleep 5
  done
done

python - "$OUT" <<'PY'
import json, glob, os, sys
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))) if not f.endswith('.raw.json')]
g = lambda d, *ks: (lambda v: '-' if v is None else v)(__import__('functools').reduce(lambda x, k: (x or {}).get(k), ks, d))
lines = ['| run | adapters | ok / failed / rejected | tok/s | TTFT p50 / p99 (s) | TTFT tail p99 (s) | latency p99 (s) | swaps | swap load p50 / p95 (s) | 1st req after swap TTFT p95 (s) | GPU MB start → end | host RSS MB start → end |',
         '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
for d in rows:
    lines.append(f"| {d.get('config', d['kind'] + '_' + d['mode'])} | {d['adapters']} | {d['requests_ok']} / {d['requests_failed']} / {d['requests_rejected']} | {d['output_tok_per_s']} | "
                 f"{g(d,'ttft_s','p50')} / {g(d,'ttft_s','p99')} | {g(d,'ttft_tail_half_s','p99')} | {g(d,'latency_s','p99')} | {g(d,'swaps','n')} | "
                 f"{g(d,'swaps','load_s','p50')} / {g(d,'swaps','load_s','p95')} | {g(d,'first_request_after_swap_ttft_s','p95')} | "
                 f"{g(d,'gpu_mem_mb','start')} → {g(d,'gpu_mem_mb','end')} | {g(d,'host_rss_mb','start')} → {g(d,'host_rss_mb','end')} |")
open(os.path.join(sys.argv[1], 'summary.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY
