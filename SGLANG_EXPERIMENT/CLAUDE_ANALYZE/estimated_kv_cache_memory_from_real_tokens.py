#!/usr/bin/env python3
"""ESTIMATED KV-cache memory usage for the DSV4-Flash bench_kv.sh runs.

This is still an estimate, not a measurement: sglang:kv_cache_memory_usage_gb
(the metric that would give a genuine measured GB figure) is unpopulated for
DeepSeek-V4's pool classes -- verified by reading the source (base KVCache
sets self.mem_usage=0 in memory_pool.py:1706; deepseek_v4_memory_pool.py never
overrides it, unlike the standard MHA pool classes which do). No real,
directly-measured KV-cache-only byte count exists to read for these runs.

What this script computes instead: real, live kv_used_tokens (scraped from
/metrics, an actual measured token count) multiplied by a per-token byte cost
that is *not* measured but reconstructed from SGLang's own hardcoded/asserted
buffer layout constants (see PROVENANCE below). Real tokens x an estimated
price -- half measurement, half estimate -- so "estimated", not "precise" or
"measured", is the accurate name for the result.

v3: fixes an indexer byte-cost bug found while verifying v2's provenance (see
below); v2 fixed a ~2x overstatement in the original version.

PROVENANCE -- every number this script uses is either (a) scraped live from
the actual running server, or (b) a constant traced to the exact SGLang
source line that computed/asserted it for *this* run's logged config. Nothing
here is guessed or taken from the paper on faith.

(a) kv_used_tokens -- real, live, per-run measurement, not derived:
    scraped from /metrics (kv_bench_sglang/<ctx>/metrics_*.prom), fed by the
    Prometheus gauge registered at
      python/sglang/srt/observability/metrics_collector.py:363
    ("sglang:kv_used_tokens"), whose value each scrape is
      python/sglang/srt/managers/scheduler_components/pool_stats_observer.py:222
      full_num_used = ... -> stats.kv_used_tokens = self.full_num_used
    which _get_token_info() (same file, ~line 216) computes as
      num_used = max_total_num_tokens - (available_size + evictable_size)
    reading `self.token_to_kv_pool_allocator.available_size()` and
    `self.tree_cache.evictable_size()` -- the live allocator/radix-tree state
    of the actual scheduler process, updated every step. This is not a
    theoretical token count; it's what the running allocator actually held.

(b) Per-token byte cost -- traced to two independent SGLang source
    locations that agree, plus the exact runtime flags this run used:

    Main KV entry, 584 bytes (nope FP8 + rope BF16 + scale), confirmed twice:
      - deepseek_v4_memory_pool.py:106-124, hardcoded + `assert`ed
        (`DeepSeekV4SingleKVPool.get_bytes_per_token`/`create_buffer`).
      - model_executor/pool_configurator.py:880
        (`kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8`),
        the *same* formula, independently written in the capacity-planning
        code that produces the `bytes_per_full_token` log line.
    Both match the paper (DeepSeek_V4.pdf Sec 2.3.4, p.13): "BF16 precision
    is used for the rotary positional embedding (RoPE) dimensions, while FP8
    precision is applied to the remaining dimensions."

    Indexer K entry: `DeepSeekV4IndexerPool.get_bytes_per_token`
    (deepseek_v4_memory_pool.py:288-292) is a runtime branch:
      index_head_dim // 2 + 4   if use_fp4_indexer else   index_head_dim + 4
    `use_fp4_indexer` is `server_args.enable_deepseek_v4_fp4_indexer`. The
    logged `server_args={...}` dump in sglang_dsv4.log for *this* run shows
    `'enable_deepseek_v4_fp4_indexer': False` -- so this run took the
    non-FP4 branch: 128 + 4 = 132 bytes/entry, NOT the paper's FP4-packed
    68 bytes (index_dim//2 + index_dim//32) that a prior version of this
    script assumed by reading the paper instead of this run's actual flags.
    (compare_kv_cache.py's *theoretical* estimate, not tied to any specific
    run, still correctly models the paper's FP4 design for what a paper-
    matching deployment would use -- it's this script, which claims to
    describe *this run*, that must use what this run actually did.)

    bytes_per_full_token itself (7033.45, logged once at pool-init) was
    independently reproduced bit-for-bit from pool_configurator.py:879-913's
    `_get_bytes_per_full_token`, using real config values (kv_bytes=584,
    indexer_bytes=132, 21 ratio=4 / 20 ratio=128 / 2 ratio=0 layers) plus two
    runtime settings pulled straight from the log: `swa_full_tokens_ratio`
    is overridden to 0.1 for DSV4 models specifically
    (arg_groups/overrides.py:1414-1416, confirmed by max_total_num_tokens_swa
    / max_total_num_tokens = 685824 / 6859008 = 0.1 in /metrics) and the
    compress-state dtype is bf16 (sglang_serve.sh sets
    SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16). That constant bakes in a
    proportional reservation for the co-allocated SWA/state pool across ALL
    43 layers (not just the 2 pure-SWA ones) -- which is why it's ~1.83x the
    true classical-pool-only rate this script prices below (7033.45 vs
    3850.25 bytes/token), and why it was the wrong constant to price real
    tokens against (see v2 history in CLAUDE_ANALYZE/explanation_claude.md).

swa_used_tokens is still reported in tokens only, not priced in GiB: its
pool is a separately-tracked, fixed-capacity allocation
(max_total_num_tokens_swa is constant across every context length tested,
confirmed in /metrics) rather than one that scales with kv_used_tokens, so
folding it into the same bytes-per-real-token rate would misprice it.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent
BENCH_ROOT = EXPERIMENT_DIR / "kv_bench_sglang"
CONFIG_PATH = REPO_ROOT / "configs" / "deepseek-v4-flash.json"
LOG_PATH = EXPERIMENT_DIR / "sglang_dsv4.log"

GIB = 1 << 30

# Verified against sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:106-124
# (hardcoded + asserted, independent of any env var / kv-cache-dtype flag).
NOPE_DTYPE_BYTES = 1  # FP8
ROPE_DTYPE_BYTES = 2  # BF16
SCALE_PAD_BYTES = 1
QUANTIZE_BLOCK_SIZE = 64

# DeepSeekV4IndexerPool.get_bytes_per_token (deepseek_v4_memory_pool.py:290-292) branches on
# use_fp4_indexer = server_args.enable_deepseek_v4_fp4_indexer. This run's logged server_args
# (sglang_dsv4.log) show enable_deepseek_v4_fp4_indexer=False, so it took the non-FP4 branch:
# index_head_dim + 4, NOT the paper's FP4-packed index_head_dim // 2 + 4.
INDEXER_FP4_ENABLED_THIS_RUN = False

# swa_full_tokens_ratio: overridden to 0.1 for DSV4 models specifically
# (sglang/python/sglang/srt/arg_groups/overrides.py:1414-1416). Not read from a flag in
# sglang_serve.sh -- it's a silent architecture-specific default. Cross-checked at runtime
# below against max_total_num_tokens_swa / max_total_num_tokens in a live /metrics scrape,
# not just taken on faith from reading the override source.
SWA_FULL_TOKENS_RATIO = 0.1

# SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16, set in sglang_serve.sh. Consumed by
# pool_configurator.py's _get_dsv4_compress_state_dtype_sizes() -> (2, 2) for bf16.
# This sizes the *state* buffers (c4_state_bytes/c4_indexer_state_bytes) only -- NOT the
# main KV entry's nope/rope dtypes, which are unconditionally FP8/BF16 (see NOPE/ROPE_DTYPE_BYTES).
C4_STATE_DTYPE_BYTES = 2

# c4_ring_size = get_compress_state_ring_size(4, is_speculative=False) (deepseek_v4_memory_pool.py:34-48)
# -> 8 for non-speculative decode, which this run is (server_args show speculative_algorithm=None).
C4_RING_SIZE_NON_SPECULATIVE = 8


def reproduce_bytes_per_full_token(
    *,
    kv_bytes: int,
    indexer_bytes: int,
    attn_head_dim: int,
    num_layers_total: int,
    num_layers_ca4: int,
    num_layers_ca128: int,
    swa_ratio: float,
    c4_state_dtype_bytes: int,
    c4_ring_size: int,
    swa_page_size: int,
) -> float:
    """Reproduces pool_configurator.py:879-913 (_get_bytes_per_full_token) term-for-term.

    This is the formula SGLang itself runs once at startup to size the pools and produce
    the "DSV4 memory calculation: bytes_per_full_token=..." log line. Reproducing it here
    (rather than trusting the docstring's prose description of it) is the proof that this
    script's understanding of the formula -- and every constant fed into it -- is correct:
    verify_provenance() below asserts this function's output equals the value SGLang itself
    logged for this run, to two decimal places.

    c128_state_ratio is fixed at 0 in the source (pool_configurator.py:903-904: "C128 state
    is request-scoped and is finalized after max_running_requests is known, so it should not
    scale with full-token capacity here") -- so the corresponding terms are omitted here too.
    """
    c4_state_ratio = c4_ring_size / swa_page_size  # pool_configurator.py:900
    c4_state_bytes = 2 * 2 * attn_head_dim * c4_state_dtype_bytes  # pool_configurator.py:891
    c4_indexer_state_bytes = 2 * 2 * 128 * c4_state_dtype_bytes  # pool_configurator.py:897, indexer_head_dim=128

    term_swa = swa_ratio * kv_bytes * num_layers_total  # pool_configurator.py:907
    term_c4_main = (1 / (4 * 1)) * kv_bytes * num_layers_ca4  # :908, c4_shrink_factor=1 (hisparse disabled)
    term_c128_main = (1 / 128) * kv_bytes * num_layers_ca128  # :909
    term_c4_indexer = (1 / 4) * indexer_bytes * num_layers_ca4  # :910
    term_c4_state = swa_ratio * c4_state_ratio * c4_state_bytes * num_layers_ca4  # :911-913
    term_c4_indexer_state = swa_ratio * c4_state_ratio * c4_indexer_state_bytes * num_layers_ca4  # :914-917

    return (
        term_swa
        + term_c4_main
        + term_c128_main
        + term_c4_indexer
        + term_c4_state
        + term_c4_indexer_state
    )


def parse_logged_bytes_per_full_token(log_path: Path) -> float:
    """Ground truth to check the reproduction against: SGLang's own logged value.

    sglang_dsv4.log line format: "DSV4 memory calculation: bytes_per_full_token=7033.45, ..."
    (pool_configurator.py:1024-1029). Asserts a single consistent value across all TP ranks
    -- if the log ever shows more than one, the run's config changed mid-log and this
    script's single-constant assumption for the whole sweep would be unsafe.
    """
    values = set()
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            match = re.search(r"bytes_per_full_token=([\d.]+)", line)
            if match and "DSV4 memory calculation" in line:
                values.add(float(match.group(1)))
    assert len(values) == 1, f"expected one bytes_per_full_token value in {log_path}, got {values}"
    return values.pop()


def parse_swa_ratio_from_metrics(run_dir: Path) -> float:
    """Ground truth for SWA_FULL_TOKENS_RATIO: computed from a live /metrics scrape,
    not just asserted from reading arg_groups/overrides.py."""
    full = swa = None
    for fn in sorted(glob.glob(str(run_dir / "metrics_*.prom"))):
        full_vals = read_prom_metric(Path(fn), "sglang:max_total_num_tokens")
        swa_vals = read_prom_metric(Path(fn), "sglang:max_total_num_tokens_swa")
        if full_vals and swa_vals:
            full, swa = full_vals[0], swa_vals[0]
            break
    assert full is not None, f"no max_total_num_tokens found under {run_dir}"
    return swa / full


def verify_provenance(config_path: Path, log_path: Path, a_run_dir: Path) -> None:
    """Recomputes every constant this script relies on from primitives and asserts it
    against the actual logged/measured value for this run. Fails loudly (AssertionError)
    rather than silently using a number that turned out to be wrong for this run's config."""
    config = json.loads(config_path.read_text())
    layers = int(config["num_hidden_layers"])
    ratios = [int(v) for v in config["compress_ratios"][:layers]]
    head_dim = int(config["head_dim"])
    rope_dim = int(config["qk_rope_head_dim"])
    nope_dim = head_dim - rope_dim
    index_dim = int(config["index_head_dim"])
    window = int(config["sliding_window"])
    ratio_counts = Counter(ratios)

    print("=" * 78)
    print("PROVENANCE VERIFICATION -- every constant checked against source or live data")
    print("=" * 78)

    # Check 1: main KV entry byte layout, matches deepseek_v4_memory_pool.py:121-124's
    # hardcoded assert (`assert bytes_per_token == 448 + 64*2 + 8`).
    kv_bytes = nope_dim * NOPE_DTYPE_BYTES + rope_dim * ROPE_DTYPE_BYTES + nope_dim // QUANTIZE_BLOCK_SIZE + SCALE_PAD_BYTES
    expected_kv_bytes = 448 + 64 * 2 + 8
    assert kv_bytes == expected_kv_bytes == 584, (
        f"main KV entry bytes {kv_bytes} != deepseek_v4_memory_pool.py:121's asserted 584 "
        f"-- config's head_dim/qk_rope_head_dim no longer match what that assert expects"
    )
    print(f"[1] main KV entry:  computed={kv_bytes}  source assert (deepseek_v4_memory_pool.py:121-124)=584  MATCH")

    # Check 2: indexer entry byte layout, matches DeepSeekV4IndexerPool.get_bytes_per_token
    # (deepseek_v4_memory_pool.py:290-292) for this run's enable_deepseek_v4_fp4_indexer=False.
    indexer_bytes = index_dim // 2 + 4 if INDEXER_FP4_ENABLED_THIS_RUN else index_dim + 4
    expected_indexer_bytes = 128 + 4  # non-FP4 branch, index_head_dim=128
    assert indexer_bytes == expected_indexer_bytes == 132, (
        f"indexer entry bytes {indexer_bytes} != expected 132 for the non-FP4 branch"
    )
    print(f"[2] indexer entry:  computed={indexer_bytes}  source formula (deepseek_v4_memory_pool.py:290-292, non-FP4 branch)=132  MATCH")

    # Check 3: swa_full_tokens_ratio, cross-checked against a live /metrics scrape rather
    # than only against the override source (arg_groups/overrides.py:1414-1416). Tolerance is
    # not arbitrary: _compute_dsv4_sizes (pool_configurator.py) page-aligns swa_tokens
    # (`int(full_token * swa_ratio) // page_size * page_size`), so the measured ratio can be
    # off from the exact 0.1 by at most one page_size's worth of tokens out of full_token.
    measured_swa_ratio = parse_swa_ratio_from_metrics(a_run_dir)
    page_size = 256  # server_args 'page_size': 256, logged in sglang_dsv4.log's server_args dump
    full_token_capacity = 6859008  # logged "full_token=6859008" (DSV4 memory calculation line)
    max_rounding_error = page_size / full_token_capacity
    ratio_error = abs(measured_swa_ratio - SWA_FULL_TOKENS_RATIO)
    assert ratio_error <= max_rounding_error, (
        f"measured max_total_num_tokens_swa/max_total_num_tokens={measured_swa_ratio} differs "
        f"from assumed SWA_FULL_TOKENS_RATIO={SWA_FULL_TOKENS_RATIO} by {ratio_error}, which "
        f"exceeds the max page-alignment rounding error of {max_rounding_error} -- this is a "
        f"real mismatch, not rounding"
    )
    print(
        f"[3] swa_full_tokens_ratio:  measured in /metrics ({a_run_dir.name})={measured_swa_ratio:.7f}  "
        f"assumed constant={SWA_FULL_TOKENS_RATIO}  diff={ratio_error:.7f} <= page-rounding bound {max_rounding_error:.7f}  MATCH"
    )

    # Check 4: full end-to-end reproduction of bytes_per_full_token, the strongest proof --
    # if every constant and the formula itself are right, this must equal the exact value
    # SGLang logged, to the precision SGLang printed it at (2 decimals).
    attn_head_dim = nope_dim + rope_dim
    reproduced = reproduce_bytes_per_full_token(
        kv_bytes=kv_bytes,
        indexer_bytes=indexer_bytes,
        attn_head_dim=attn_head_dim,
        num_layers_total=len(ratios),
        num_layers_ca4=ratio_counts[4],
        num_layers_ca128=ratio_counts[128],
        swa_ratio=SWA_FULL_TOKENS_RATIO,
        c4_state_dtype_bytes=C4_STATE_DTYPE_BYTES,
        c4_ring_size=C4_RING_SIZE_NON_SPECULATIVE,
        swa_page_size=window,
    )
    logged = parse_logged_bytes_per_full_token(log_path)
    assert abs(reproduced - logged) < 0.01, (
        f"reproduced bytes_per_full_token={reproduced:.2f} != logged value={logged:.2f} "
        f"-- the formula, or one of its inputs, does not actually match this run"
    )
    print(f"[4] bytes_per_full_token:  reproduced from scratch (pool_configurator.py:879-913)={reproduced:.2f}  logged in sglang_dsv4.log={logged:.2f}  MATCH")
    print("=" * 78)
    print()


def load_bytes_per_real_token(config_path: Path) -> dict[int, float]:
    """Real bytes/real-token for each compression ratio, classical (c4/c128) pool only.

    Layers with ratio=0 (the first two, pure-SWA layers) are excluded here: per
    /metrics, `max_total_num_tokens_swa` is a small *fixed* capacity (685824,
    constant across every run in kv_bench_sglang/ regardless of context length)
    tracked separately from `max_total_num_tokens`/`kv_used_tokens` (the
    classical pool, whose capacity/usage scales with context). Sliding-window
    attention is bounded to the last `sliding_window` tokens by definition, so
    those layers' cost does not grow with context and is not part of what
    kv_used_tokens measures -- pricing them against kv_used_tokens would
    reintroduce an unbounded linear term that doesn't exist in the real system.
    Indexer K is only stored for CSA (ratio=4) layers, per the model.
    """
    config = json.loads(config_path.read_text())
    layers = int(config["num_hidden_layers"])
    ratios = [int(v) for v in config["compress_ratios"][:layers]]
    head_dim = int(config["head_dim"])
    rope_dim = int(config["qk_rope_head_dim"])
    nope_dim = head_dim - rope_dim
    index_dim = int(config["index_head_dim"])

    bytes_per_entry = (
        nope_dim * NOPE_DTYPE_BYTES
        + rope_dim * ROPE_DTYPE_BYTES
        + nope_dim // QUANTIZE_BLOCK_SIZE
        + SCALE_PAD_BYTES
    )
    index_bytes_per_entry = (
        index_dim // 2 + 4 if INDEXER_FP4_ENABLED_THIS_RUN else index_dim + 4
    )  # DeepSeekV4IndexerPool.get_bytes_per_token; this run took the non-FP4 branch

    ratio_counts = Counter(ratios)

    # bytes/real-token for each compressed ratio's layers: main-KV + indexer (ratio=4 only).
    # ratio=0 (pure-SWA) layers are reported for context but excluded from the priced total.
    per_ratio_bytes_per_token = {}
    for ratio in sorted(set(ratios)):
        num_layers = ratio_counts[ratio]
        if ratio == 0:
            per_ratio_bytes_per_token[ratio] = bytes_per_entry * num_layers  # informational only
            continue
        main = bytes_per_entry / ratio * num_layers
        indexer = (index_bytes_per_entry / ratio * num_layers) if ratio == 4 else 0.0
        per_ratio_bytes_per_token[ratio] = main + indexer

    return per_ratio_bytes_per_token


def bytes_per_real_token_total(per_ratio_bytes_per_token: dict[int, float]) -> float:
    """Priced total: classical (compressed) pool only -- excludes ratio=0 (see loader docstring)."""
    return sum(bytes_per_token for ratio, bytes_per_token in per_ratio_bytes_per_token.items() if ratio != 0)


def read_prom_metric(path: Path, metric: str) -> list[float]:
    values = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.startswith(metric):
                continue
            try:
                values.append(float(line.rsplit(None, 1)[1]))
            except (IndexError, ValueError):
                continue
    return values


def peak_used_tokens(run_dir: Path, metric: str) -> int | None:
    peak = None
    for fn in sorted(glob.glob(str(run_dir / "metrics_*.prom"))):
        for value in read_prom_metric(Path(fn), metric):
            if peak is None or value > peak:
                peak = value
    return None if peak is None else int(round(peak))


def main() -> None:
    run_dirs = sorted(
        (d for d in BENCH_ROOT.iterdir() if d.is_dir()),
        key=lambda d: int(d.name),
    )
    if not run_dirs:
        print(f"no run directories found under {BENCH_ROOT}", file=sys.stderr)
        sys.exit(1)

    verify_provenance(CONFIG_PATH, LOG_PATH, run_dirs[0])

    per_ratio = load_bytes_per_real_token(CONFIG_PATH)
    bytes_per_real_token = bytes_per_real_token_total(per_ratio)

    print("Real per-token blended cost, from SGLang's own hardcoded DSV4 KV layout")
    print("(deepseek_v4_memory_pool.py:106-124) amortized over each layer's real")
    print("compression ratio in configs/deepseek-v4-flash.json:")
    for ratio, bytes_per_token in sorted(per_ratio.items()):
        if ratio == 0:
            print(f"  {'ratio=0 (pure-SWA layers)':32} {bytes_per_token:8.3f} bytes/entry -- bounded, excluded below, see swa_used_tokens")
        else:
            print(f"  {'ratio=' + str(ratio):32} {bytes_per_token:8.3f} bytes/real-token")
    print(f"  {'total priced (classical pool only)':32} {bytes_per_real_token:8.3f} bytes/real-token\n")

    rows = []
    for run_dir in run_dirs:
        context = int(run_dir.name)
        kv_used = peak_used_tokens(run_dir, "sglang:kv_used_tokens")
        swa_used = peak_used_tokens(run_dir, "sglang:swa_used_tokens")

        if kv_used is None:
            rows.append((context, None, None, None, swa_used))
            continue

        per_gpu_bytes = kv_used * bytes_per_real_token
        rows.append((context, kv_used, per_gpu_bytes, per_gpu_bytes * 8, swa_used))

    print(
        f"{'context':>10} {'kv_used_tokens':>15} {'per_gpu_GiB':>12} "
        f"{'cluster_8gpu_GiB':>17} {'swa_used_tokens':>16}"
    )
    for context, kv_used, per_gpu_bytes, cluster_bytes, swa_used in rows:
        if kv_used is None:
            print(f"{context:10d} {'no metrics':>15} {'-':>12} {'-':>17} {swa_used if swa_used is not None else '-':>16}")
            continue
        print(
            f"{context:10d} {kv_used:15d} {per_gpu_bytes / GIB:12.4f} "
            f"{cluster_bytes / GIB:17.4f} {swa_used if swa_used is not None else '-':>16}"
        )


if __name__ == "__main__":
    main()
