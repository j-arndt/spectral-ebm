"""Measure forward, input-gradient, and one ULA step for matched models."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from spectral_ebm.models import DenseEBM, SpectralEBM, parameter_count
from spectral_ebm.sampler import ula_step


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device: torch.device, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[max(0, int(0.95 * len(values)) - 1)],
        "all_ms": values,
    }


def run_model(model: nn.Module, x: torch.Tensor, device: torch.device, repeats: int) -> dict:
    def forward() -> None:
        model(x)

    def input_gradient() -> None:
        state = x.detach().requires_grad_(True)
        energy = model(state).sum()
        torch.autograd.grad(energy, state)

    def ula() -> None:
        ula_step(model, x, step_size=0.01, noise_scale=0.0)

    for _ in range(5):
        forward()
        input_gradient()
        ula()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = {
        "parameters": parameter_count(model),
        "parameter_bytes_fp32": parameter_count(model) * 4,
        "forward": summarize(measure(forward, device, repeats)),
        "input_gradient": summarize(measure(input_gradient, device, repeats)),
        "ula_step": summarize(measure(ula, device, repeats)),
    }
    if device.type == "cuda":
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/full.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    runs = []
    for dim in args.dims:
        x = torch.randn(args.batch_size, dim, device=device)
        dense = DenseEBM(dim).to(device)
        spectral = SpectralEBM(dim).to(device)
        runs.append(
            {
                "dim": dim,
                "batch_size": args.batch_size,
                "device": str(device),
                "dense": run_model(dense, x, device, args.repeats),
                "spectral": run_model(spectral, x, device, args.repeats),
            }
        )
    result = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
