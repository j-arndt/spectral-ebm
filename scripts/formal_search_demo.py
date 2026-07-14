"""Run the parser-agnostic HRR-to-EBM formal-state integration smoke demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_ebm import FormalProofSearchAdapter, HRREncoder, SpectralEBM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    states = [
        ["by", "intro", "h", "exact", "h"],
        ["by", "simp", "at", "h"],
    ]
    vocabulary = sorted({token for state in states for token in state} | {"<unk>"})
    adapter = FormalProofSearchAdapter(
        HRREncoder(vocabulary, args.dim, max_length=32, seed=7),
        SpectralEBM(args.dim, hidden_layers=2),
    )
    result = adapter.refine(
        states,
        steps=args.steps,
        step_size=args.step_size,
        noise_scale=0.0,
    )
    payload = {
        "states": result.states,
        "initial_energy": result.initial_energy.tolist(),
        "refined_energy": result.refined_energy.tolist(),
        "embedding_shape": list(result.refined_embeddings.shape),
        "trust_boundary": "continuous candidate refinement only; no parser or proof checker",
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")


if __name__ == "__main__":
    main()
