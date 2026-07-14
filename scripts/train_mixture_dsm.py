"""Train EBMs on a known four-mode Gaussian mixture and report mode coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from spectral_ebm.models import DenseEBM, SpectralEBM
from spectral_ebm.sampler import langevin_sample
from spectral_ebm.training import denoising_score_matching_loss

CENTERS = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])


def sample_mixture(batch_size: int, std: float) -> torch.Tensor:
    indices = torch.randint(0, len(CENTERS), (batch_size,))
    return CENTERS[indices] + std * torch.randn(batch_size, 2)


def mixture_score(x: torch.Tensor, variance: float) -> torch.Tensor:
    centers = CENTERS.to(device=x.device, dtype=x.dtype)
    log_prob = -0.5 * (x[:, None, :] - centers[None, :, :]).square().sum(dim=-1) / variance
    weights = torch.softmax(log_prob, dim=1)
    component_scores = (centers[None, :, :] - x[:, None, :]) / variance
    return (weights[..., None] * component_scores).sum(dim=1)


def evaluate(model: nn.Module, *, sample_count: int, steps: int, step_size: float, mixture_std: float, noise_std: float) -> dict:
    evaluation = sample_mixture(2048, mixture_std)
    noisy_variance = mixture_std**2 + noise_std**2
    state = evaluation + noise_std * torch.randn_like(evaluation)
    state = state.detach().requires_grad_(True)
    score = -torch.autograd.grad(model(state).sum(), state)[0]
    score_mse = (score - mixture_score(state.detach(), noisy_variance)).square().mean().item()
    initial = torch.empty(sample_count, 2).uniform_(-2.0, 2.0)
    samples = langevin_sample(model, initial, steps=steps, step_size=step_size, bounds=(-3.0, 3.0))
    distances = (samples[:, None, :] - CENTERS[None, :, :]).square().sum(dim=-1).sqrt()
    assignments = distances.argmin(dim=1)
    counts = torch.bincount(assignments, minlength=len(CENTERS)).tolist()
    return {"score_mse": score_mse, "mode_counts": counts, "sample_mean": samples.mean(dim=0).tolist()}


def train(model: nn.Module, *, steps: int, batch_size: int, mixture_std: float, noise_std: float, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = denoising_score_matching_loss(
            model,
            sample_mixture(batch_size, mixture_std),
            noise_std=noise_std,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mixture-std", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-step-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/mixture-dsm.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    results = {"config": vars(args) | {"output": str(args.output)}, "models": {}}
    for name, model in (("spectral", SpectralEBM(2)), ("dense", DenseEBM(2))):
        train(
            model,
            steps=args.steps,
            batch_size=args.batch_size,
            mixture_std=args.mixture_std,
            noise_std=args.noise_std,
            lr=args.lr,
        )
        results["models"][name] = evaluate(
            model,
            sample_count=args.sample_count,
            steps=args.sample_steps,
            step_size=args.sample_step_size,
            mixture_std=args.mixture_std,
            noise_std=args.noise_std,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
