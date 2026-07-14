# Benchmark protocol

The benchmark compares `DenseEBM` and `SpectralEBM` with the same input dimension, hidden depth, activation, output projection, batch size, dtype, and device.

It reports three separate quantities:

1. forward energy evaluation;
2. forward plus input-gradient evaluation;
3. one deterministic ULA step, including the input gradient.

Each timing run is warmed up, synchronized on CUDA before and after timing, repeated, and reported with median, p95, and all observations. The JSON artifact also records parameter count, fp32 parameter bytes, peak CUDA memory, Python, PyTorch, operating system, device, dimension, and batch size.

Run the full harness from the repository root:

```powershell
python -m benchmarks.full_benchmark --device cuda --dims 128 256 512 --batch-size 64 --repeats 20 --output benchmark_results/full-cuda.json
```

The benchmark is descriptive. A theoretical `O(D log D)` layer can still lose in wall-clock latency to a dense GEMM at small or moderate dimensions because FFT launch and memory overhead dominate. Results must therefore be reported per hardware configuration.

## Extension benchmark

The extension harness measures two separate claims:

```powershell
python -m benchmarks.benchmark_extensions --device cpu --dims 64 128 --channels 4 --batch-size 16 --chain-steps 5 --repeats 10 --output benchmark_results/extensions.json
```

- `BlockCirculantLinear` is compared with a dense flattened channel map at matched input/output shape. The comparison reports parameters and forward latency separately; no universal speed claim follows.
- `vectorized_langevin_chain()` is compared with repeated `langevin_sample()` calls at the same model, steps, step size, temperature, and zero-noise deterministic setting. The comparison is a memory/allocation optimization smoke test, not evidence of different Markov-chain behavior.
## Accelerator and formal-search smoke paths

The Triton path is benchmarkable only when triton_runtime_available() is true. The repository includes a CUDA correctness test for forward and parameter/input gradients; it skips with an explicit prerequisite message when the host lacks a C compiler or compatible Triton runtime. The capability probe records unavailable environments, while the RTX 4060 Laptop GPU artifact records an actual measured comparison after the local compiler path was configured. No universal speedup is inferred.

The parser-agnostic formal-search smoke path is not a performance benchmark. Run `python scripts/formal_search_demo.py --dim 64 --steps 4` to verify token normalization, HRR encoding, persistent-chain refinement, and structured energy output.
The Triton probe also records capability without fabricating a timing result:

```powershell
python -m benchmarks.benchmark_triton --output benchmark_results/local-triton.json
```## Production-hardening audit

The v0.4.0 audit compares the quadratic parameter memory of a Sinkhorn relaxation with the linear K*D storage of the Householder mixer and runs projected spherical Langevin steps through a block-spectral energy model:

    python -m benchmarks.hardening_audit --device cuda --dim 4096 --batch-size 64 --channels 8 --output benchmark_results/hardening-cuda.json

The default scale is intentionally hardware-dependent. The committed CPU smoke artifact uses dim 64, batch size 4, and two channels. It reports parameter bytes, CUDA peak allocation when available, and per-step sphere-norm envelopes. The spherical sampler is projected ULA with tangent-space drift/noise and radial retraction; it is not advertised as an exact geodesic or stationary sampler.
