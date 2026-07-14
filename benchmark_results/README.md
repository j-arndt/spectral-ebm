# Benchmark artifacts

Each JSON file records the exact command output, package versions, device, dimensions, batch size, every timing repetition, and median timing. These results are illustrative measurements on the named hardware, not universal performance claims.

The initial artifact was produced with:

```powershell
python -m benchmarks.benchmark_layers --device cuda --dims 128 256 512 --batch-size 64 --repeats 10 --warmup 5 --output benchmark_results/2026-07-14-cuda.json
```

The benchmark measures one synchronized ULA step including the input-energy gradient. It does not measure a full training epoch or formal proof search.
