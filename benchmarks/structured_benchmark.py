"""Compare dense, diagonal, and circulant energy networks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spectral_ebm.baselines import DiagonalEBM
from spectral_ebm.models import DenseEBM, SpectralEBM
from spectral_ebm.sampler import ula_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/structured.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    x = torch.randn(args.batch_size, args.dim, device=device)
    models = {
        "dense": DenseEBM(args.dim).to(device),
        "spectral": SpectralEBM(args.dim).to(device),
        "diagonal": DiagonalEBM(args.dim).to(device),
    }
    result = {"config": vars(args) | {"output": str(args.output)}, "runs": {}}
    for name, model in models.items():
        for _ in range(3):
            ula_step(model, x, step_size=0.01, noise_scale=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start is None:
                import time

                wall_start = time.perf_counter()
            else:
                start.record()
            ula_step(model, x, step_size=0.01, noise_scale=0.0)
            if end is None:
                timings.append((time.perf_counter() - wall_start) * 1000.0)
            else:
                end.record()
                torch.cuda.synchronize(device)
                timings.append(start.elapsed_time(end))
        result["runs"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "ula_ms": timings,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
