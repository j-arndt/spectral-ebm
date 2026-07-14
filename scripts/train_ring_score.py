"""Train energy networks against the analytic score of a noisy ring target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from spectral_ebm.models import DenseEBM, SpectralEBM
from spectral_ebm.sampler import langevin_sample


def sample_ring(batch_size: int, radius: float, radial_std: float) -> torch.Tensor:
    angle = 2.0 * torch.pi * torch.rand(batch_size)
    radial = radius + radial_std * torch.randn(batch_size)
    return radial[:, None] * torch.stack((angle.cos(), angle.sin()), dim=1)


def ring_score(x: torch.Tensor, radius: float, radial_std: float) -> torch.Tensor:
    norm = x.norm(dim=1, keepdim=True).clamp_min(1e-6)
    radial_score = -(norm - radius) / (radial_std**2 * norm)
    jacobian_score = -1.0 / norm.square()
    return (radial_score + jacobian_score) * x


def score_matching_loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    state = x.detach().requires_grad_(True)
    score = -torch.autograd.grad(model(state).sum(), state, create_graph=True)[0]
    return (score - target).square().mean()


def train(model: nn.Module, *, steps: int, batch_size: int, radius: float, radial_std: float, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        clean = sample_ring(batch_size, radius, radial_std)
        loss = score_matching_loss(model, clean, ring_score(clean, radius, radial_std))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, *, radius: float, radial_std: float, sample_count: int, sample_steps: int) -> dict:
    evaluation = sample_ring(2048, radius, radial_std)
    target = ring_score(evaluation, radius, radial_std)
    state = evaluation.detach().requires_grad_(True)
    score = -torch.autograd.grad(model(state).sum(), state)[0]
    initial = torch.empty(sample_count, 2).uniform_(-2.0, 2.0)
    samples = langevin_sample(model, initial, steps=sample_steps, step_size=0.01, bounds=(-3.0, 3.0))
    sample_radius = samples.norm(dim=1)
    return {
        "score_mse": (score - target).square().mean().item(),
        "sample_radius_mean": sample_radius.mean().item(),
        "sample_radius_std": sample_radius.std().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--radial-std", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/ring-score.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    results = {"config": vars(args) | {"output": str(args.output)}, "models": {}}
    for name, model in (("spectral", SpectralEBM(2)), ("dense", DenseEBM(2))):
        train(
            model,
            steps=args.steps,
            batch_size=args.batch_size,
            radius=args.radius,
            radial_std=args.radial_std,
            lr=args.lr,
        )
        results["models"][name] = evaluate(
            model,
            radius=args.radius,
            radial_std=args.radial_std,
            sample_count=args.sample_count,
            sample_steps=args.sample_steps,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
