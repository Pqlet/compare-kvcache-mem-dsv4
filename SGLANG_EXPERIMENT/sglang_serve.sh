#!/usr/bin/env bash


MODEL="${MODEL:-/models/DeepSeek-V4-Flash-0731}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MEM_FRAC="${MEM_FRAC:-0.85}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"

export SGLANG_DSV4_COMPRESS_STATE_DTYPE="${SGLANG_DSV4_COMPRESS_STATE_DTYPE:-bf16}"

sglang serve \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp "$TP" \
  --kv-cache-dtype "$KV_DTYPE" \
  --mem-fraction-static "$MEM_FRAC" \
  --context-length 1048576 \
  --chunked-prefill-size 8192 \
  --max-running-requests 64 \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --host 0.0.0.0 \
  --port "$PORT" \
  2>&1 | tee sglang_dsv4.log