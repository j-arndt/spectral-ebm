"""Benchmark block-circulant layers and persistent Langevin execution."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from spectral_ebm.layers import BlockCirculantLinear
from spectral_ebm.models import SpectralEBM, parameter_count
from spectral_ebm.sampler import langevin_sample, vectorized_langevin_chain


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device: torch.device, repeats: int) -> dict[str, float | list[float]]:
    values: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        values.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[max(0, int(0.95 * len(values)) - 1)],
        "all_ms": values,
    }


def benchmark_block_layer(
    channels: int,
    dim: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
) -> dict:
    x = torch.randn(batch_size, channels, dim, device=device)
    block = BlockCirculantLinear(channels, channels, dim).to(device)
    dense = nn.Linear(channels * dim, channels * dim).to(device)

    def block_forward() -> None:
        block(x)

    def dense_forward() -> None:
        dense(x.flatten(start_dim=-2)).reshape(batch_size, channels, dim)

    for _ in range(3):
        block_forward()
        dense_forward()
    return {
        "channels": channels,
        "dim": dim,
        "batch_size": batch_size,
        "block_parameters": parameter_count(block),
        "dense_parameters": parameter_count(dense),
        "block_forward": measure(block_forward, device, repeats),
        "dense_forward": measure(dense_forward, device, repeats),
    }


def benchmark_chain(
    dim: int,
    batch_size: int,
    steps: int,
    repeats: int,
    device: torch.device,
) -> dict:
    model = SpectralEBM(dim).to(device)
    x = torch.randn(batch_size, dim, device=device)

    def standard() -> None:
        langevin_sample(model, x, steps=steps, step_size=0.01, noise_scale=0.0)

    def persistent() -> None:
        vectorized_langevin_chain(model, x, steps=steps, step_size=0.01, noise_scale=0.0)

    for _ in range(2):
        standard()
        persistent()
    return {
        "dim": dim,
        "batch_size": batch_size,
        "steps": steps,
        "model_parameters": parameter_count(model),
        "standard_chain": measure(standard, device, repeats),
        "persistent_state_chain": measure(persistent, device, repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chain-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/extensions.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    result = {
        "config": vars(args) | {"output": str(args.output)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
        },
        "block_layer_runs": [
            benchmark_block_layer(args.channels, dim, args.batch_size, args.repeats, device)
            for dim in args.dims
        ],
        "chain_runs": [
            benchmark_chain(dim, args.batch_size, args.chain_steps, args.repeats, device)
            for dim in args.dims
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()