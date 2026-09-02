#!/usr/bin/env python3
"""Materialize estimated KV-cache payloads on each CUDA rank and measure HBM."""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Iterable

from compare_kv_cache import (
    GIB,
    estimate_v32,
    estimate_v4_flash,
    load_model_configs,
    parse_token_count,
)


MODEL_NAMES = ("v3.2", "v4-flash")


def expected_components(model: str, tokens: int) -> dict[str, int]:
    v32_config, v4_config = load_model_configs()
    if model == "v3.2":
        return estimate_v32(tokens, v32_config).components()
    if model == "v4-flash":
        return estimate_v4_flash(tokens, v4_config).components()
    raise ValueError(f"unknown model: {model}")


def chunk_sizes(size: int, maximum: int) -> Iterable[int]:
    while size:
        current = min(size, maximum)
        yield current
        size -= current


def dry_run(models: list[str], tokens: int, world_size: int) -> None:
    result = []
    for model in models:
        components = expected_components(model, tokens)
        total = sum(components.values())
        result.append(
            {
                "model": model,
                "tokens": tokens,
                "expected_bytes_per_gpu": total,
                "expected_gib_per_gpu": total / GIB,
                "expected_gib_cluster": total * world_size / GIB,
                "components": components,
            }
        )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=parse_token_count, default=1 << 20)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument(
        "--no-touch", action="store_true", help="allocate without zeroing every byte"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print expected bytes without importing torch",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "8")),
        help="cluster aggregate used by --dry-run",
    )
    args = parser.parse_args()
    if args.chunk_mib <= 0 or args.reserve_gib < 0 or args.world_size <= 0:
        parser.error("chunk-mib and world-size must be positive; reserve-gib must be non-negative")
    if args.dry_run:
        dry_run(args.models, args.tokens, args.world_size)
        return

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use --dry-run on non-GPU hosts")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    chunk_bytes = args.chunk_mib << 20

    for model in args.models:
        if distributed:
            dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        components = expected_components(model, args.tokens)
        expected = sum(components.values())
        required = expected + int(args.reserve_gib * GIB)
        if free_bytes < required:
            raise RuntimeError(
                f"rank {local_rank}: {model} needs {required / GIB:.2f} GiB free "
                f"including reserve; only {free_bytes / GIB:.2f} GiB free"
            )

        buffers = []
        for size in components.values():
            for part in chunk_sizes(size, chunk_bytes):
                buffer = torch.empty(part, dtype=torch.uint8, device=device)
                if not args.no_touch:
                    buffer.zero_()
                buffers.append(buffer)
        torch.cuda.synchronize(device)
        allocated_delta = torch.cuda.memory_allocated(device) - before_allocated
        reserved_delta = torch.cuda.memory_reserved(device) - before_reserved
        local = {
            "rank": dist.get_rank() if distributed else 0,
            "device": torch.cuda.get_device_name(device),
            "model": model,
            "tokens": args.tokens,
            "expected_bytes": expected,
            "allocated_delta_bytes": allocated_delta,
            "reserved_delta_bytes": reserved_delta,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "device_total_bytes": total_bytes,
            "components": components,
        }
        records = [None] * world_size if local["rank"] == 0 else None
        if distributed:
            dist.gather_object(local, records, dst=0)
        else:
            records = [local]
        if local["rank"] == 0:
            assert records is not None
            print(
                json.dumps(
                    {
                        "model": model,
                        "tokens": args.tokens,
                        "world_size": world_size,
                        "expected_gib_per_gpu": expected / GIB,
                        "expected_gib_cluster": expected * world_size / GIB,
                        "allocated_gib_per_gpu": [
                            record["allocated_delta_bytes"] / GIB for record in records
                        ],
                        "reserved_gib_per_gpu": [
                            record["reserved_delta_bytes"] / GIB for record in records
                        ],
                        "records": records,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        del buffers
        gc.collect()
        torch.cuda.empty_cache()
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
