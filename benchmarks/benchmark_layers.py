"""Reproducible CPU/CUDA benchmark for matched energy networks.

Example:
    python benchmarks/benchmark_layers.py --dims 128 256 512 --batch-size 64 --steps 30
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from spectral_ebm.models import DenseEBM, SpectralEBM, parameter_count
from spectral_ebm.sampler import ula_step


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def benchmark_one(
    dim: int, batch_size: int, repeats: int, warmup: int, device: torch.device
) -> dict:
    dense = DenseEBM(dim).to(device)
    spectral = SpectralEBM(dim).to(device)
    x = torch.randn(batch_size, dim, device=device)

    for _ in range(warmup):
        for model in (dense, spectral):
            state = x.detach().requires_grad_(True)
            ula_step(model, state, step_size=0.01, noise_scale=0.0)
    _sync(device)

    def dense_step() -> None:
        ula_step(dense, x, step_size=0.01, noise_scale=0.0)

    def spectral_step() -> None:
        ula_step(spectral, x, step_size=0.01, noise_scale=0.0)

    dense_times = _timed(dense_step, device, repeats)
    spectral_times = _timed(spectral_step, device, repeats)
    return {
        "dim": dim,
        "batch_size": batch_size,
        "device": str(device),
        "dense_params": parameter_count(dense),
        "spectral_params": parameter_count(spectral),
        "dense_ula_ms_median": statistics.median(dense_times),
        "spectral_ula_ms_median": statistics.median(spectral_times),
        "dense_ula_ms_all": dense_times,
        "spectral_ula_ms_all": spectral_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
        },
        "runs": [
            benchmark_one(dim, args.batch_size, args.repeats, args.warmup, device)
            for dim in args.dims
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
