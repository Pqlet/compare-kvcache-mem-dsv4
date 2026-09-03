import glob
import json
import os
import re

root = "./kv_bench_sglang"

WANTED = [
    "sglang:token_usage",
    "sglang:full_token_usage",
    "sglang:swa_token_usage",

    "sglang:kv_used_tokens",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",

    "sglang:swa_used_tokens",
    "sglang:swa_available_tokens",
    "sglang:swa_evictable_tokens",

    "sglang:max_total_num_tokens",

    "sglang:cache_hit_rate",
    "sglang:decode_sum_seq_lens",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
]


def read_prom(path):
    out = {}

    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            for metric in WANTED:
                if not line.startswith(metric):
                    continue

                try:
                    value = float(line.rsplit(None, 1)[1])
                except Exception:
                    continue

                # Multiple TP scheduler labels may exist.
                out.setdefault(metric, []).append(value)

    return out


print(
    f"{'context':>10} "
    f"{'used':>12} "
    f"{'evict':>12} "
    f"{'free':>12} "
    f"{'full_use':>10} "
    f"{'swa_use':>10}"
)

for d in sorted(glob.glob(root + "/*"), key=lambda x: int(os.path.basename(x))):
    context = int(os.path.basename(d))

    peak_used = 0
    peak_evict = 0
    min_free = float("inf")
    peak_full_usage = 0
    peak_swa_usage = 0

    for fn in glob.glob(d + "/metrics_*.prom"):
        x = read_prom(fn)

        # use max, not sum:
        # TP ranks describe shards of the same logical token sequence
        peak_used = max(
            peak_used,
            max(x.get("sglang:kv_used_tokens", [0]))
        )

        peak_evict = max(
            peak_evict,
            max(x.get("sglang:kv_evictable_tokens", [0]))
        )

        if x.get("sglang:kv_available_tokens"):
            min_free = min(
                min_free,
                min(x["sglang:kv_available_tokens"])
            )

        peak_full_usage = max(
            peak_full_usage,
            max(x.get("sglang:full_token_usage", [0]))
        )

        peak_swa_usage = max(
            peak_swa_usage,
            max(x.get("sglang:swa_token_usage", [0]))
        )

    print(
        f"{context:10d} "
        f"{peak_used:12.0f} "
        f"{peak_evict:12.0f} "
        f"{min_free:12.0f} "
        f"{peak_full_usage:10.4f} "
        f"{peak_swa_usage:10.4f}"
    )
