"""Build publication-style figures from committed benchmark artifacts.

The figures are intentionally honest: parameter counts are structural, while
runtime and toy scores are labelled as measurements from specific protocols.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
ASSETS = ROOT / "docs" / "assets"

NAVY = "#0B1220"
PANEL = "#111B2E"
GRID = "#2A3852"
TEXT = "#E8EEF7"
MUTED = "#9FB0C7"
CYAN = "#55D6BE"
ORANGE = "#FFB86B"
BLUE = "#6EA8FE"
PINK = "#FF7E9D"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def style(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.55, linewidth=0.7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_axisbelow(True)


def legend(ax: plt.Axes) -> None:
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(TEXT)


def small_runs() -> list[dict]:
    return load("2026-07-14-cuda.json")["runs"]


def large_runs() -> list[dict]:
    return load("2026-07-14-full-cuda-large.json")["runs"]


def parameter_scaling(ax: plt.Axes) -> None:
    runs = small_runs()
    dims = [r["dim"] for r in runs]
    dense = [r["dense_params"] for r in runs]
    spectral = [r["spectral_params"] for r in runs]
    for r in large_runs():
        dims.append(r["dim"])
        dense.append(r["dense"]["parameters"])
        spectral.append(r["spectral"]["parameters"])
    ax.loglog(dims, dense, "o-", color=BLUE, linewidth=2.4, markersize=6, label="Dense EBM")
    ax.loglog(dims, spectral, "o-", color=CYAN, linewidth=2.4, markersize=6, label="Spectral EBM")
    ax.set_xlabel("Hidden dimension D")
    ax.set_ylabel("Trainable parameters")
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_title("Parameter scaling", color=TEXT, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.text(0.03, 0.07, "O(D²)  →  O(D) per structured layer", transform=ax.transAxes, color=MUTED, fontsize=9)
    style(ax, "Parameter scaling")
    legend(ax)


def runtime(ax: plt.Axes) -> None:
    runs = small_runs() + large_runs()
    dims = [r["dim"] for r in runs]
    dense = [
        r["dense_ula_ms_median"] if "dense_ula_ms_median" in r else r["dense"]["ula_step"]["median_ms"]
        for r in runs
    ]
    spectral = [
        r["spectral_ula_ms_median"] if "spectral_ula_ms_median" in r else r["spectral"]["ula_step"]["median_ms"]
        for r in runs
    ]
    ax.plot(dims, dense, "o-", color=BLUE, linewidth=2.4, markersize=6, label="Dense ULA")
    ax.plot(dims, spectral, "o-", color=ORANGE, linewidth=2.4, markersize=6, label="Spectral ULA")
    ax.set_xlabel("Hidden dimension D")
    ax.set_ylabel("Median ULA step (ms)")
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.text(0.03, 0.08, "RTX 4060 Laptop GPU · batch 64", transform=ax.transAxes, color=MUTED, fontsize=9)
    style(ax, "Runtime is hardware- and kernel-dependent")
    legend(ax)


def structure(ax: plt.Axes) -> None:
    generator = np.linspace(-1.0, 1.0, 10)
    matrix = np.array([np.roll(generator, i) for i in range(10)])
    im = ax.imshow(matrix, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("One generator, a full circulant map", color=TEXT, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.text(0.03, 0.06, r"W[i,j] = c[(i−j) mod D]", transform=ax.transAxes, color=TEXT, fontsize=10, family="monospace")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID)


def toy_scores(ax: plt.Axes) -> None:
    gaussian = load("2026-07-14-gaussian-dsm.json")
    mixture = load("2026-07-14-mixture-dsm.json")["models"]
    ring = load("2026-07-14-ring-score.json")["models"]
    labels = ["Gaussian\nDSM", "4-mode\nmixture", "Noisy\nring"]
    dense = [gaussian["dense"]["score_mse_standard_normal"], mixture["dense"]["score_mse"], ring["dense"]["score_mse"]]
    spectral = [gaussian["spectral"]["score_mse_standard_normal"], mixture["spectral"]["score_mse"], ring["spectral"]["score_mse"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, dense, width, color=BLUE, label="Dense")
    ax.bar(x + width / 2, spectral, width, color=CYAN, label="Spectral")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score MSE (lower is better)")
    ax.text(0.03, 0.93, "300 DSM steps · 1,024 Langevin samples", transform=ax.transAxes, color=MUTED, fontsize=9, va="top")
    style(ax, "Toy distribution checks")
    legend(ax)


def extension_comparison(axes: tuple[plt.Axes, plt.Axes]) -> None:
    data = load("2026-07-14-extensions-cpu.json")
    layer_runs = data["block_layer_runs"]
    dims = [run["dim"] for run in layer_runs]
    block_params = [run["block_parameters"] for run in layer_runs]
    dense_params = [run["dense_parameters"] for run in layer_runs]
    axes[0].loglog(dims, dense_params, "o-", color=BLUE, linewidth=2.4, label="Dense channel map")
    axes[0].loglog(dims, block_params, "o-", color=CYAN, linewidth=2.4, label="Block-circulant")
    axes[0].set_xlabel("Feature dimension D")
    axes[0].set_ylabel("Parameters")
    axes[0].set_xticks(dims, [str(dim) for dim in dims])
    style(axes[0], "Multi-channel parameter budget")
    legend(axes[0])

    chain_runs = data["chain_runs"]
    chain_dims = [run["dim"] for run in chain_runs]
    standard = [run["standard_chain"]["median_ms"] for run in chain_runs]
    persistent = [run["persistent_state_chain"]["median_ms"] for run in chain_runs]
    axes[1].plot(chain_dims, standard, "o-", color=ORANGE, linewidth=2.4, label="Repeated ULA")
    axes[1].plot(chain_dims, persistent, "o-", color=PINK, linewidth=2.4, label="Persistent state")
    axes[1].set_xlabel("Feature dimension D")
    axes[1].set_ylabel("Median chain time (ms)")
    axes[1].set_xticks(chain_dims, [str(dim) for dim in chain_dims])
    axes[1].text(0.03, 0.08, "CPU smoke benchmark · 3 ULA steps", transform=axes[1].transAxes, color=MUTED, fontsize=9)
    style(axes[1], "Persistent execution is an optimization, not a new sampler")
    legend(axes[1])

def save(fig: plt.Figure, name: str, dpi: int = 180) -> None:
    fig.savefig(ASSETS / name, dpi=dpi, facecolor=NAVY, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": NAVY, "savefig.facecolor": NAVY})

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=NAVY)
    fig.suptitle("spectral-ebm", color=TEXT, fontsize=24, fontweight="bold", x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.945, "Correctness-first structured energy models in PyTorch", color=MUTED, fontsize=12, ha="left")
    parameter_scaling(axes[0, 0])
    runtime(axes[0, 1])
    structure(axes[1, 0])
    toy_scores(axes[1, 1])
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.0, w_pad=1.6)
    save(fig, "hero.png")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=NAVY)
    parameter_scaling(ax)
    fig.tight_layout()
    save(fig, "parameter-scaling.png")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=NAVY)
    runtime(ax)
    fig.tight_layout()
    save(fig, "runtime-tradeoff.png")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=NAVY)
    toy_scores(ax)
    fig.tight_layout()
    save(fig, "toy-results.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=NAVY)
    extension_comparison(axes)
    fig.tight_layout(w_pad=2.0)
    save(fig, "extensions.png")


if __name__ == "__main__":
    main()
