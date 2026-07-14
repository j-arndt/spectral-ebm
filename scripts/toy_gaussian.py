"""Small deterministic smoke experiment on a known Gaussian target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spectral_ebm import SpectralEBM, langevin_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--chains", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/toy_gaussian.json"))
    args = parser.parse_args()
    torch.manual_seed(0)
    model = SpectralEBM(args.dim)
    initial = torch.randn(args.chains, args.dim)
    final = langevin_sample(model, initial, steps=args.steps, step_size=0.01)
    result = {
        "dim": args.dim,
        "chains": args.chains,
        "steps": args.steps,
        "mean_l2": final.mean(dim=0).norm().item(),
        "mean_variance": final.var(dim=0).mean().item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
