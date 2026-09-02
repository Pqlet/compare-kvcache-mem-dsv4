#!/usr/bin/env python3
"""Compare DeepSeek-V3.2 and DeepSeek-V4-Flash inference KV-cache payloads."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, TextIO


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
V32_CONFIG = CONFIG_DIR / "deepseek-v3.2.json"
V4_FLASH_CONFIG = CONFIG_DIR / "deepseek-v4-flash.json"
GIB = 1 << 30


@dataclass(frozen=True)
class CacheBreakdown:
    """Logical KV-cache payload for one sequence on one tensor-parallel rank."""

    main_kv_bytes: int
    indexer_k_bytes: int
    sliding_window_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.main_kv_bytes + self.indexer_k_bytes + self.sliding_window_bytes

    def components(self) -> dict[str, int]:
        return {
            "main_kv": self.main_kv_bytes,
            "indexer_k": self.indexer_k_bytes,
            "sliding_window": self.sliding_window_bytes,
        }


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_model_configs() -> tuple[dict, dict]:
    return load_json(V32_CONFIG), load_json(V4_FLASH_CONFIG)


def estimate_v32(
    tokens: int,
    config: Mapping[str, object],
    *,
    indexer_layers: int | None = None,
) -> CacheBreakdown:
    """Estimate V3.2 DSA cache using SGLang's SM90 FP8 cache layout."""

    _validate_tokens(tokens)
    layers = int(config["num_hidden_layers"])
    indexer_layers = layers if indexer_layers is None else indexer_layers
    if not 0 <= indexer_layers <= layers:
        raise ValueError(f"indexer_layers must be in [0, {layers}]")

    latent_dim = int(config["kv_lora_rank"])
    rope_dim = int(config["qk_rope_head_dim"])
    index_dim = int(config["index_head_dim"])

    # SGLang DSATokenToKVPool on Hopper:
    # latent FP8 + one FP32 scale per 128 dims + RoPE BF16.
    main_bytes_per_token = latent_dim + (latent_dim // 128) * 4 + rope_dim * 2
    # Lightning-indexer K: FP8 + one FP32 scale per 128 dims.
    index_bytes_per_token = index_dim + (index_dim // 128) * 4

    return CacheBreakdown(
        main_kv_bytes=tokens * layers * main_bytes_per_token,
        indexer_k_bytes=tokens * indexer_layers * index_bytes_per_token,
    )


def estimate_v4_flash(
    tokens: int, config: Mapping[str, object]
) -> CacheBreakdown:
    """Estimate V4-Flash CSA/HCA cache using paper and SGLang SM90 layout."""

    _validate_tokens(tokens)
    layers = int(config["num_hidden_layers"])
    ratios = [int(value) for value in config["compress_ratios"][:layers]]  # type: ignore[index]
    if len(ratios) != layers or any(value not in (0, 4, 128) for value in ratios):
        raise ValueError("invalid DeepSeek-V4 compress_ratios")

    head_dim = int(config["head_dim"])
    rope_dim = int(config["qk_rope_head_dim"])
    nope_dim = head_dim - rope_dim
    index_dim = int(config["index_head_dim"])
    window = int(config["sliding_window"])

    # SGLang DeepSeekV4SingleKVPool:
    # non-RoPE FP8 + RoPE BF16 + one byte per 64-dim FP8 scale block,
    # padded to eight scale bytes for DeepSeek-V4's 448 non-RoPE dims.
    main_bytes_per_entry = nope_dim + rope_dim * 2 + nope_dim // 64 + 1
    # Paper uses FP4 indexer QK. SGLang stores two FP4 values per byte and
    # one byte scale per 32 dims.
    index_bytes_per_entry = index_dim // 2 + index_dim // 32

    ratio_counts = Counter(ratios)
    c4_entries = tokens // 4
    c128_entries = tokens // 128
    compressed_main = main_bytes_per_entry * (
        ratio_counts[4] * c4_entries + ratio_counts[128] * c128_entries
    )
    compressed_index = index_bytes_per_entry * ratio_counts[4] * c4_entries
    sliding_window = main_bytes_per_entry * layers * min(tokens, window)

    return CacheBreakdown(
        main_kv_bytes=compressed_main,
        indexer_k_bytes=compressed_index,
        sliding_window_bytes=sliding_window,
    )


def build_rows(
    lengths: Iterable[int],
    *,
    gpus: int = 8,
    v32_indexer_layers: int | None = None,
) -> list[dict[str, int | float | bool]]:
    if gpus < 1:
        raise ValueError("gpus must be positive")
    v32_config, v4_config = load_model_configs()
    rows = []
    for tokens in lengths:
        v32 = estimate_v32(tokens, v32_config, indexer_layers=v32_indexer_layers)
        v4 = estimate_v4_flash(tokens, v4_config)
        ratio = v4.total_bytes / v32.total_bytes if v32.total_bytes else 0.0
        rows.append(
            {
                "tokens": tokens,
                "v32_bytes_per_gpu": v32.total_bytes,
                "v4_flash_bytes_per_gpu": v4.total_bytes,
                "v4_over_v32_percent": ratio * 100,
                "v32_over_v4_times": 1 / ratio if ratio else 0.0,
                "v32_bytes_cluster": v32.total_bytes * gpus,
                "v4_flash_bytes_cluster": v4.total_bytes * gpus,
                "v32_within_native_context": tokens
                <= int(v32_config["max_position_embeddings"]),
            }
        )
    return rows


def parse_token_count(text: str) -> int:
    raw = text.strip().replace("_", "").lower()
    multiplier = 1
    for suffix, value in (("ki", 1 << 10), ("mi", 1 << 20), ("k", 10**3), ("m", 10**6)):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            multiplier = value
            break
    try:
        result = int(raw) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid token count: {text}") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("token count must be positive")
    return result


def _validate_tokens(tokens: int) -> None:
    if tokens < 0:
        raise ValueError("tokens must be non-negative")


def _format_gib(value: int | float) -> str:
    return f"{float(value) / GIB:.3f}"


def print_table(rows: list[dict[str, int | float | bool]], gpus: int) -> None:
    headers = (
        "tokens",
        "V3.2 GiB/GPU",
        "V4-Flash GiB/GPU",
        "V4/V3.2",
        "saving",
        f"V3.2 GiB/{gpus}GPU",
        f"V4 GiB/{gpus}GPU",
    )
    table = [headers]
    for row in rows:
        table.append(
            (
                f"{int(row['tokens']):,}",
                _format_gib(row["v32_bytes_per_gpu"]),
                _format_gib(row["v4_flash_bytes_per_gpu"]),
                f"{float(row['v4_over_v32_percent']):.2f}%",
                f"{float(row['v32_over_v4_times']):.2f}x",
                _format_gib(row["v32_bytes_cluster"]),
                _format_gib(row["v4_flash_bytes_cluster"]),
            )
        )
    widths = [max(len(str(line[column])) for line in table) for column in range(len(headers))]
    for line_number, line in enumerate(table):
        print("  ".join(str(value).rjust(widths[i]) for i, value in enumerate(line)))
        if line_number == 0:
            print("  ".join("-" * width for width in widths))


def print_csv(rows: list[dict[str, int | float | bool]], output: TextIO = sys.stdout) -> None:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def details(tokens: int, v32_indexer_layers: int | None = None) -> dict:
    v32_config, v4_config = load_model_configs()
    v32 = estimate_v32(tokens, v32_config, indexer_layers=v32_indexer_layers)
    v4 = estimate_v4_flash(tokens, v4_config)
    return {
        "tokens": tokens,
        "deepseek_v3_2": {**asdict(v32), "total_bytes": v32.total_bytes},
        "deepseek_v4_flash": {**asdict(v4), "total_bytes": v4.total_bytes},
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=parse_token_count,
        default=[4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20],
        help="token counts; suffixes K, M, Ki, Mi accepted",
    )
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--v32-indexer-layers",
        type=int,
        default=None,
        help="override V3.2 indexer-K layer count; paper-equivalent default is all 61",
    )
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument(
        "--details", action="store_true", help="show component bytes for last length"
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    rows = build_rows(
        args.lengths,
        gpus=args.gpus,
        v32_indexer_layers=args.v32_indexer_layers,
    )
    if args.format == "table":
        print_table(rows, args.gpus)
        if any(not bool(row["v32_within_native_context"]) for row in rows):
            print(
                "\nNote: V3.2 points above 163,840 tokens are paper-style extrapolations; "
                "released config native limit is 163,840."
            )
        if args.details:
            print("\nLast-length payload breakdown (GiB/GPU):")
            for model, values in details(args.lengths[-1], args.v32_indexer_layers).items():
                if model == "tokens":
                    continue
                components = ", ".join(
                    f"{name.removesuffix('_bytes')}={_format_gib(value)}"
                    for name, value in values.items()
                )
                print(f"  {model}: {components}")
    elif args.format == "json":
        print(json.dumps({"gpus": args.gpus, "rows": rows}, indent=2))
    else:
        print_csv(rows)


if __name__ == "__main__":
    main()
