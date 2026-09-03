# Questions:
1. > Verify the paper authors use fp8 for non-rope cache            

    Confirmed. §2.3.4 (page 13), lines 763-766 of the paper:
    
    "First, we adopt a mixed storage format for KV entries: BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage."
    
    That's an explicit, unambiguous statement: non-RoPE dims = FP8, matching exactly what deepseek_v4_memory_pool.py's hardcoded assert implements (448 FP8 + 64\*2 BF16 + 8 scale). So the paper text, the SGLang source, and the corrected (pre-diff) formula in compare_kv_cache.py all agree — the uncommitted nope_dim*2 edit is the outlier and is wrong.


2. > Why do you think 3.6 GB for V4-Flash for kv-cache is for 1 GPU, not for per the whole model ?


    Why the 3.6/3.7 GB is per-GPU, not divided across the cluster:

    Every one of the 8 TP ranks reports the identical pool size (full=6859008, c4=1714752, c128=53586, ... — byte-for-byte the same on TP0 through TP7). If the KV cache were sharded across tensor-parallel ranks (the way standard MHA/GQA key/value heads are split), each rank would show roughly 1/8th the capacity. It doesn't — every GPU allocates a full, independent copy.

    That's because DeepSeek's attention (MLA-lineage, inherited into CSA/HCA) caches a compressed latent vector, not per-head K/V — and that latent isn't head-parallel, so TP has nothing to shard it by. Each GPU must materialize the full compressed cache locally to run its share of attention. So "3.6/3.7 GB" is inherently a per-accelerator figure — that's also why compare_kv_cache.py computes v4_flash_bytes_per_gpu directly from the formula (no /gpus) and only
    multiplies by gpus afterward to get a separate cluster total. A paper reporting architecture efficiency independent of any specific deployment's TP degree would naturally report this same per-GPU, replicated number — it's the portable unit (how much cache one accelerator must hold for one sequence), not something that depends on cluster size.


3. > Why do i get approximately 54 GB total mem usage for the whole 8 gpu cluster if the paper state 3.6 GB ?

    The ~54 GB you're seeing is a real cluster number (8 GPUs really do each hold a full replica), but it's built on a per-token price that's ~2x too high. The paper's 3.6 GB is correctly a single-GPU figure (replicated, not sharded, confirmed by the identical TP0-TP7 pool sizes). The actually-correct cluster total for an 8-GPU deployment of a 1M-token sequence, matching the paper's stated precision (FP8 non-RoPE), should be ~29.5 GB, not 54.

    **Update:** `estimated_kv_cache_memory_from_real_tokens.py` has since been fixed. It no longer prices real
    `kv_used_tokens` against the flawed `bytes_per_full_token` capacity constant; it now uses
    the verified DSV4 per-token layout (`deepseek_v4_memory_pool.py:106-124`, FP8 non-RoPE)
    amortized over each layer's real compression ratio (4 or 128) from
    `configs/deepseek-v4-flash.json`, excluding the separately-tracked, bounded pure-SWA/state
    pool (`swa_used_tokens`). Blended rate: 3514.25 bytes/real-token (vs. the old 7033.45,
    confirming the ~2.00x overstatement). Extrapolated to ~1M tokens the cluster total now comes
    out to ~27.5-29.5 GiB, matching the paper's implied ~29.5 GB rather than ~54 GB. See
    `estimated_kv_cache_memory_from_real_tokens_results.txt` for the corrected per-context table.


4. > Why MLA can not do TP ? How do you know "each rank would show roughly 1/8th the capacity" is what would happen otherwise?

    **Why the compressed latent can't be head-sharded.**

    Standard MHA/GQA caches one K and one V vector *per attention head* --
    `num_key_value_heads` independent, same-shape slices. Tensor parallelism
    shards that dimension: each rank owns a disjoint subset of KV heads and
    only ever needs to cache its own subset. Nothing about attention requires
    a rank to see another rank's heads, so the KV cache shards cleanly.

    MLA (and CSA/HCA, which inherit its design) does not cache per-head K/V at
    all. It caches one shared low-rank latent vector per token
    (`kv_lora_rank`-ish width) and reconstructs each head's K/V from that
    *same* latent via a per-head up-projection matrix, applied at attention
    time (the "matrix absorption" trick from the DeepSeek-V2 paper). Concretely
    in DeepSeek-V4-Flash's config, `num_key_value_heads: 1` -- there is
    architecturally *one* KV per token, shared across all `n_h=64` query heads
    via Multi-Query Attention (paper Sec 2.3.1, "Shared Key-Value MQA": "CSA
    then performs core attention in a Multi-Query Attention (MQA) manner,
    where each compressed KV entry ... serves as both attention key and
    value"). If GPU rank 3 owns query heads 24-31, it still needs the *entire*
    shared latent to reconstruct K/V for those heads -- there's no per-head
    slice of the cache it could own instead. So there is nothing to shard by:
    every rank must materialize the full latent locally, which is exactly
    what "replicate the cache identically on every rank" means.

    **How I know sharded caches would show ~1/8th capacity, not by assumption
    but by reading the sizing code SGLang actually runs:**

    `python/sglang/srt/layers/linear.py:1008-1014` (`QKVParallelLinear`, used
    by ordinary MHA/GQA models) computes each rank's local KV head count
    directly from `tp_size`:

    ```python
    self.num_heads = divide(self.total_num_heads, tp_size)
    if kv_tp_size >= self.total_num_kv_heads:
        self.num_kv_heads = 1
        self.num_kv_head_replicas = divide(kv_tp_size, self.total_num_kv_heads)
    else:
        self.num_kv_heads = divide(self.total_num_kv_heads, kv_tp_size)
        self.num_kv_head_replicas = 1
    ```

    For a normal model with e.g. 16 KV heads on TP=8, `kv_tp_size (8) <
    total_num_kv_heads (16)`, so you land in the `else` branch:
    `num_kv_heads = divide(16, 8) = 2` per rank -- each rank owns a disjoint
    2-head slice, i.e. exactly 1/8th of the 16 total heads. The KV-cache pool
    for that model is then sized off this already-divided `num_kv_heads`, so
    each rank's pool really is ~1/8th the total. (`python/sglang/srt/models/afmoe.py:321-325`
    shows the same divide-by-tp_size pattern independently for another model
    family.) That `else` branch is the *sharding* path -- it only fires when
    there are strictly more KV heads than ranks, so each rank can get a
    non-empty disjoint slice.

    The `if` branch is the one DeepSeek-V4 actually takes: with
    `total_num_kv_heads=1` and `kv_tp_size=8`, `kv_tp_size >=
    total_num_kv_heads` is true, so `num_kv_heads = 1` on *every* rank (not
    1/8, not 0 -- the same single shared KV on all 8) and
    `num_kv_head_replicas = divide(8, 1) = 8`: the framework's own name for
    this is *replication*, not sharding, triggered automatically whenever a
    model has fewer KV heads than TP ranks.

    Separately, `deepseek_v4_memory_pool.py` (the file that actually sizes
    DeepSeek-V4's compressed KV pools) contains zero references to `tp_size`
    or `tp_rank` anywhere in its sizing logic -- there is no code path in it
    that could produce a smaller pool on some ranks than others. That absence,
    together with the `sglang_dsv4.log` evidence (identical `DSV4 pool sizes`
    line logged by TP0 through TP7), is the direct, checkable confirmation
    that DeepSeek-V4's pool is replicated: not an assumption about what MLA
    "should" do, but the literal difference between a file that divides by
    `tp_size` (`linear.py`, `afmoe.py`) and one that never mentions it
    (`deepseek_v4_memory_pool.py`).


5. > Back to `estimated_kv_cache_memory_from_real_tokens.py` -- what is your testimony that it is the real world SGLang numbers? Search the code that begot it.

    Every number the script uses traces to a specific source line, split into two kinds:
    live measurements (this run's actual state) and constants (fixed at code-read time,
    verified against the exact code path this run took).

    **Live measurement -- `kv_used_tokens`:**
    Scraped from `/metrics`, fed by the Prometheus gauge registered at
    `metrics_collector.py:363` (`sglang:kv_used_tokens`), whose value each scrape is
    `pool_stats_observer.py:222` (`stats.kv_used_tokens = self.full_num_used`), computed
    in `_get_token_info()` (~line 216) as `num_used = max_total_num_tokens -
    (available_size + evictable_size)`, reading `token_to_kv_pool_allocator.available_size()`
    and `tree_cache.evictable_size()` -- the live scheduler's actual allocator/radix-tree
    state at scrape time, not a derived or nominal count.

    **Constant -- main KV entry, 584 bytes, confirmed from two independent files:**
    `deepseek_v4_memory_pool.py:106-124` hardcodes and `assert`s it
    (`qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 (64*2) + scale = 584`), and
    `pool_configurator.py:880` computes the *same* formula independently
    (`kv_bytes = qk_nope_head_dim + qk_rope_head_dim*2 + 8`) in the unrelated
    capacity-planning code that produces the `bytes_per_full_token` log line. Two
    files, written for different purposes, agree byte-for-byte.

    **Bug found while verifying this -- indexer entry size is runtime-conditional:**
    `DeepSeekV4IndexerPool.get_bytes_per_token` (`deepseek_v4_memory_pool.py:290-292`)
    branches on `use_fp4_indexer = server_args.enable_deepseek_v4_fp4_indexer`:
    `index_head_dim//2 + 4` (68 bytes, FP4-packed, matching the paper's stated design)
    if enabled, else `index_head_dim + 4` (132 bytes). The `server_args={...}` dump
    logged at server startup in `sglang_dsv4.log` shows
    `'enable_deepseek_v4_fp4_indexer': False` for this run -- so the actual run took
    the 132-byte branch, not the paper's 68-byte one. `estimated_kv_cache_memory_from_real_tokens.py` was
    using 68 (copied from the paper / from `compare_kv_cache.py`'s theoretical model)
    until this check caught it; it's now fixed to 132, matching what this run's own
    logged flags say it actually did.

    **Full end-to-end reproduction of `bytes_per_full_token` (7033.45), bit-for-bit:**
    re-derived `pool_configurator.py:879-913`'s `_get_bytes_per_full_token` formula by
    hand using: `kv_bytes=584`, `indexer_bytes=132` (the corrected value above), the
    real layer counts from `configs/deepseek-v4-flash.json` (21 ratio=4, 20 ratio=128,
    2 ratio=0 layers), `swa_full_tokens_ratio=0.1` (a DSV4-specific override in
    `arg_groups/overrides.py:1414-1416` -- confirmed independently by
    `max_total_num_tokens_swa / max_total_num_tokens = 685824/6859008 = 0.1` in
    `/metrics`), and `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16` (set in `sglang_serve.sh`).
    Result: **7033.45 exactly**, matching the logged value to the second decimal. That
    match is the strongest evidence available that the formula, the config values, and
    the runtime-flag assumptions used throughout this analysis are all correct for this
    specific run -- not just plausible-looking numbers.

    That same exercise is *why* `bytes_per_full_token` was the wrong constant to price
    real tokens against in the original script (v1): its formula deliberately bakes in
    `swa_ratio * kv_bytes * num_layers_total` -- a reservation for the co-allocated
    SWA/state pool spread across all 43 layers -- which has nothing to do with the
    per-token cost of the classical (c4/c128) pool that `kv_used_tokens` actually
    counts. `estimated_kv_cache_memory_from_real_tokens.py` v3 now uses only the classical-pool formula
    (584 bytes/entry amortized by real ratio, 132-byte indexer for ratio=4 layers),
    giving 3850.25 bytes/real-token -- about 1.83x below `bytes_per_full_token`, for a
    fully traced reason rather than an approximate one.
