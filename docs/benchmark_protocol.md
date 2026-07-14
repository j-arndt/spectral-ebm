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
