# Benchmark artifacts

These JSON files are the source of truth for the figures and tables in the repository README. Each artifact records the protocol configuration, package versions, device, dimensions, every timing repetition where applicable, and the reported median or summary metrics.

## Artifact index

| Artifact | What it contains |
|---|---|
| [`2026-07-14-cpu.json`](2026-07-14-cpu.json) | CPU layer and ULA timings |
| [`2026-07-14-cuda.json`](2026-07-14-cuda.json) | CUDA layer and ULA timings for `D = 128, 256, 512` |
| [`2026-07-14-full-cuda-large.json`](2026-07-14-full-cuda-large.json) | End-to-end CUDA measurements for `D = 1,024, 2,048` |
| [`2026-07-14-structured-cuda.json`](2026-07-14-structured-cuda.json) | Dense, spectral, and diagonal ULA comparison |
| [`2026-07-14-gaussian-dsm.json`](2026-07-14-gaussian-dsm.json) | Standard Gaussian denoising score matching |
| [`2026-07-14-mixture-dsm.json`](2026-07-14-mixture-dsm.json) | Four-mode Gaussian mixture score matching and mode counts |
| [`2026-07-14-ring-score.json`](2026-07-14-ring-score.json) | Noisy-ring score matching and sampled-radius summary |

The published plots are generated with [`scripts/make_plots.py`](../scripts/make_plots.py):

- [`parameter-scaling.png`](../docs/assets/parameter-scaling.png)
- [`runtime-tradeoff.png`](../docs/assets/runtime-tradeoff.png)
- [`toy-results.png`](../docs/assets/toy-results.png)
- [`hero.png`](../docs/assets/hero.png)

## Protocol

The initial layer artifact was produced with:

```powershell
python -m benchmarks.benchmark_layers --device cuda --dims 128 256 512 --batch-size 64 --repeats 10 --warmup 5 --output benchmark_results/2026-07-14-cuda.json
```

The larger artifact uses synchronized CUDA measurements and includes forward, input-gradient, ULA-step, p95, and peak-memory fields:

```powershell
python -m benchmarks.full_benchmark --device cuda --dims 1024 2048 --batch-size 64 --repeats 10 --output benchmark_results/2026-07-14-full-cuda-large.json
```

The benchmark measures a synchronized ULA step including the input-energy gradient. It does not measure a full training epoch, distributed throughput, or formal proof search. See [`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md) for the complete methodology and limitations.