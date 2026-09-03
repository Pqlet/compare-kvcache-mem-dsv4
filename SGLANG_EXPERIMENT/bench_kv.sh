#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/sglang/python"

MODEL="${MODEL:-/models/DeepSeek-V4-Flash}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"

OUTLEN="${OUTLEN:-256}"
CONCURRENCY="${CONCURRENCY:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"

LENGTHS=(
  1024
  4096
  8192
  16384
  32768
  65536
  131072
  262144
  524288
)

ROOT="kv_bench_sglang"
mkdir -p "$ROOT"

# Fail if your existing server is absent.
curl -fsS "http://${HOST}:${PORT}/health" >/dev/null || {
    echo "No SGLang server at ${HOST}:${PORT}"
    exit 1
}

WATCH_PID=""

stop_watcher() {
    if [[ -n "${WATCH_PID}" ]]; then
        kill "${WATCH_PID}" 2>/dev/null || true
        wait "${WATCH_PID}" 2>/dev/null || true
        WATCH_PID=""
    fi
}

trap stop_watcher EXIT INT TERM

for INPUT_LEN in "${LENGTHS[@]}"; do
    RUN="${ROOT}/${INPUT_LEN}"
    mkdir -p "$RUN"

    echo "Benchmark input=${INPUT_LEN}, output=${OUTLEN}"

    # Clear KV left by prior length.
    curl -fsS -X POST \
        "http://${HOST}:${PORT}/flush_cache" \
        >"${RUN}/flush_cache.json"

    sleep 3

    # Sample the already-running server.
    (
        while true; do
            TS="$(date +%s.%N)"

            curl -fsS "http://${HOST}:${PORT}/metrics" \
                >"${RUN}/metrics_${TS}.prom" || true

            nvidia-smi \
                --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
                --format=csv,noheader,nounits \
                >"${RUN}/gpu_${TS}.csv" || true

            sleep "$SAMPLE_INTERVAL"
        done
    ) &

    WATCH_PID=$!

    python -m sglang.benchmark.serving \
        --backend sglang \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --tokenizer "$MODEL" \
        --dataset-name random \
        --tokenize-prompt \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTLEN" \
        --random-range-ratio 1.0 \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --request-rate inf \
        --output-file "${RUN}/bench.jsonl" \
        --output-details \
        2>&1 | tee "${RUN}/bench.log"

    stop_watcher

    # Capture state after request: active tokens should become evictable.
    curl -fsS "http://${HOST}:${PORT}/metrics" \
        >"${RUN}/metrics_after.prom"
done