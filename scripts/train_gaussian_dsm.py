"""Train spectral and dense EBMs on a known standard-normal score."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from torch import nn

from spectral_ebm.models import DenseEBM, SpectralEBM, parameter_count
from spectral_ebm.training import denoising_score_matching_loss


def score_error(model: nn.Module, x: torch.Tensor) -> float:
    state = x.detach().requires_grad_(True)
    energy = model(state).sum()
    grad = torch.autograd.grad(energy, state)[0]
    score = -grad
    return (score + x).square().mean().item()


def train(model: nn.Module, *, dim: int, steps: int, batch_size: int, noise_std: float, lr: float) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        clean = torch.randn(batch_size, dim)
        loss = denoising_score_matching_loss(model, clean, noise_std=noise_std)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    evaluation = torch.randn(2048, dim)
    return {
        "parameters": parameter_count(model),
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "score_mse_standard_normal": score_error(model, evaluation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/gaussian-dsm.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    results = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "config": vars(args) | {"output": str(args.output)},
        "spectral": train(
            SpectralEBM(args.dim),
            dim=args.dim,
            steps=args.steps,
            batch_size=args.batch_size,
            noise_std=args.noise_std,
            lr=args.lr,
        ),
        "dense": train(
            DenseEBM(args.dim),
            dim=args.dim,
            steps=args.steps,
            batch_size=args.batch_size,
            noise_std=args.noise_std,
            lr=args.lr,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
