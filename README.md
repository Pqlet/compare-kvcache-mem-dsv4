# DeepSeek KV-cache memory comparison

Compares inference KV-cache payload for DeepSeek-V4-Flash and DeepSeek-V3.2. Reproduces paper Figure 1 result at 1024K tokens: V4-Flash uses about 7.3% of V3.2 cache, or 13.7x less.

## Estimate

No GPU or package install needed:

```bash
python3 compare_kv_cache.py --details
python3 compare_kv_cache.py --lengths 128Ki 256Ki 512Ki 1Mi --format csv > results.csv
```

Reported `GiB/GPU` is KV payload on each tensor-parallel rank. MLA/MQA KV is replicated across TP ranks. `GiB/8GPU` is aggregate HBM consumed, not extra context capacity.

At 1,048,576 tokens:

- DeepSeek-V3.2: 46.94 GiB/GPU
- DeepSeek-V4-Flash: 3.43 GiB/GPU
- Ratio: 7.32%; 13.67x less

V3.2 released config has 163,840-token native limit. Larger V3.2 points are theoretical extrapolations used by paper comparison.

## Validate allocation on 8 H100s

Requires CUDA PyTorch. Does not download or load model weights.

```bash
python3 materialize_cuda.py --dry-run --tokens 1Mi --world-size 8
torchrun --standalone --nproc-per-node=8 materialize_cuda.py --tokens 1Mi
```

Materializer allocates and touches expected cache bytes on every GPU, then reports `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`. Run on otherwise idle GPUs. V3.2 1Mi case needs about 49 GiB free per GPU including default 2 GiB reserve.

Full SGLang inference not used. SGLang preallocates KV pools at server startup; post-request `nvidia-smi` stays near constant and cannot isolate cache payload. This tool follows SGLang Hopper layouts directly.

Full-checkpoint warning: official V3.2 FP8 safetensors total about 642 GiB. They do not fit on 8x80GB H100 before KV cache and workspaces. Use this cache-only materializer, more/larger GPUs, CPU offload, or a smaller quantized checkpoint.

## Formula

DeepSeek-V3.2, each token and 61 layers:

- MLA latent: 512 FP8 bytes + 16 scale bytes + 64 BF16 RoPE dims = 656 bytes
- indexer K: 128 FP8 bytes + 4 scale bytes = 132 bytes
- total: `tokens * 61 * (656 + 132)`

DeepSeek-V4-Flash has 43 layers: 21 CSA (`m=4`), 20 HCA (`m'=128`), 2 SWA-only layers. Each compressed or sliding entry is 584 bytes. Each CSA index entry is 68 bytes in FP4. All 43 layers retain a 128-token sliding window.

Estimator covers persistent per-sequence payload. Excluded: page padding, allocator metadata, compression state, CUDA workspaces, model weights, activations, speculative MTP cache, and concurrency pool slack. Those are framework/config dependent.

## Sources

- [DeepSeek-V4 paper](https://arxiv.org/abs/2606.19348), Figure 1 and Sections 2.3, 3.5
- [DeepSeek-V4-Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json), snapshot `60d8d70770c6776ff598c94bb586a859a38244f1`
- [DeepSeek-V3.2 config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json), snapshot `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`
- SGLang cache layouts: [V4](https://github.com/sgl-project/sglang/blob/f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py), [V3.2](https://github.com/sgl-project/sglang/blob/f8cbf000f4a5bfd86d3fb7c1e2d6c8fb12339d0e/python/sglang/srt/mem_cache/memory_pool.py)

## Test

```bash
python3 -m unittest -v
```
# compare-kvcache-mem-dsv4
