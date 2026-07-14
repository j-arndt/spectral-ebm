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

The Triton path is benchmarkable only when `triton_runtime_available()` is true. The repository includes a CUDA correctness test for forward and parameter/input gradients; it is skipped with an explicit prerequisite message when the host lacks a C compiler or compatible Triton runtime. The current Windows development host has CUDA and Triton installed but no compiler visible to Triton, so it does not publish a Triton timing claim.

The parser-agnostic formal-search smoke path is not a performance benchmark. Run `python scripts/formal_search_demo.py --dim 64 --steps 4` to verify token normalization, HRR encoding, persistent-chain refinement, and structured energy output.
The Triton probe also records capability without fabricating a timing result:

```powershell
python -m benchmarks.benchmark_triton --output benchmark_results/local-triton.json
```