"""Run the production-hardening memory and spherical-stability audit."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch.nn import functional as F

from spectral_ebm import (
    AmortizedHouseholderPermutation,
    BlockSpectralEBM,
    DifferentiablePermutation,
    explicit_spherical_langevin_step,
)


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--reflections", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/hardening-audit.json"),
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    sinkhorn = DifferentiablePermutation(args.dim).to(device)
    householder = AmortizedHouseholderPermutation(
        args.dim, reflections=args.reflections
    ).to(device)
    model = BlockSpectralEBM(
        channels=args.channels,
        dim=args.dim,
        hidden_layers=1,
        hidden_channels=args.channels,
    ).to(device)
    state = F.normalize(
        torch.randn(args.batch_size, args.channels, args.dim, device=device),
        dim=-1,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    norm_history: list[dict[str, float]] = []
    for step in range(args.steps):
        state = explicit_spherical_langevin_step(
            model,
            state,
            step_size=args.step_size,
            noise_scale=1.0,
            radius=1.0,
        )
        norms = state.norm(dim=-1)
        norm_history.append(
            {
                "step": step,
                "min_norm": float(norms.min().item()),
                "max_norm": float(norms.max().item()),
                "mean_norm": float(norms.mean().item()),
            }
        )

    payload = {
        "status": "completed",
        "config": vars(args) | {"output": str(args.output), "device": str(device)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
            ),
        },
        "memory_profile": {
            "sinkhorn_parameter_mb": _parameter_bytes(sinkhorn) / 1024**2,
            "householder_parameter_mb": _parameter_bytes(householder) / 1024**2,
            "reduction_factor": _parameter_bytes(sinkhorn)
            / max(_parameter_bytes(householder), 1),
            "cuda_peak_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else None
            ),
        },
        "spherical_stability": {
            "norm_history": norm_history,
            "max_abs_norm_error": max(
                abs(item["max_norm"] - 1.0) for item in norm_history
            )
            if norm_history
            else 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
