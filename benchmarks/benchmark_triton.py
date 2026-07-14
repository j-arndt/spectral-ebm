"""Benchmark the optional Triton block-circulant backend when its toolchain exists."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from spectral_ebm.layers import BlockCirculantLinear
from spectral_ebm.triton_backend import triton_available, triton_runtime_available


def synchronize() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/triton.json"))
    args = parser.parse_args()
    result: dict = {
        "config": vars(args) | {"output": str(args.output)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "triton_available": triton_available(),
        "triton_runtime_available": triton_runtime_available(),
    }
    if not result["triton_runtime_available"]:
        result["status"] = "unavailable"
        result["reason"] = (
            "CUDA/Triton runtime or a visible C compiler is unavailable; "
            "no Triton timing claim was recorded."
        )
    else:
        x = torch.randn(args.batch_size, args.channels, args.dim, device="cuda")
        torch_layer = BlockCirculantLinear(args.channels, args.channels, args.dim).cuda()
        triton_layer = BlockCirculantLinear(
            args.channels, args.channels, args.dim, backend="triton"
        ).cuda()
        triton_layer.load_state_dict(torch_layer.state_dict())

        def measure(layer: BlockCirculantLinear) -> dict[str, float | list[float]]:
            for _ in range(5):
                layer(x)
            values = []
            for _ in range(args.repeats):
                synchronize()
                start = time.perf_counter()
                layer(x)
                synchronize()
                values.append((time.perf_counter() - start) * 1000.0)
            return {
                "median_ms": statistics.median(values),
                "p95_ms": sorted(values)[max(0, int(0.95 * len(values)) - 1)],
                "all_ms": values,
            }

        result["status"] = "measured"
        result["torch"] = measure(torch_layer)
        result["triton"] = measure(triton_layer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
