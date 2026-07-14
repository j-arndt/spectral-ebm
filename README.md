# spectral-ebm

Correctness-first experiments with FFT-parameterized circulant energy-based models (EBMs) in PyTorch.

This repository is a reproducible proof of concept. It measures the parameter and runtime trade-offs of circulant energy networks against matched dense baselines; it does not claim that circulant layers are universally faster, equally expressive, or suitable for formal proof verification without a separate verifier integration.

## Current status

The first implementation provides:

- a real-valued `CirculantLinear` layer with a documented first-column convention;
- matched `SpectralEBM`, `DenseEBM`, and `DiagonalEBM` scalar-energy baselines;
- mathematically specified Unadjusted Langevin Algorithm (ULA) steps;
- short-run contrastive-divergence and denoising-score-matching losses;
- persistent chains plus reference-matrix, gradient, sampler, serialization, and parameter-count tests.

The local POC milestone is implemented and tested. The remaining public-release gate is external publication and repository metadata. The plan is tracked in [PLAN.md](PLAN.md), with proof details in [docs/proof.md](docs/proof.md), training conventions in [docs/training.md](docs/training.md), and novelty boundaries in [docs/novelty.md](docs/novelty.md).

## Mathematical convention

For a generator `c` in `R^D`, the materialized matrix is

```text
W[i, j] = c[(i - j) mod D].
```

Its first column is `c`, and its action is circular convolution:

```text
W x = irfft(rfft(x) * rfft(c), n=D).
```

The layer has `D` generator parameters, plus `D` bias parameters when enabled. It is a structured restriction of a dense map, not an arbitrary dense projection. The derivation and normalization details are in [docs/proof.md](docs/proof.md).

For an energy `E_theta(x)` and temperature `T`, ULA uses

```text
x_next = x - h/(2T) grad_x E_theta(x) + sqrt(h) epsilon,
epsilon ~ Normal(0, I).
```

Projected bounds are exposed as an explicit approximation and are not claimed to preserve the unconstrained target exactly.

## Install and test

Python 3.10+ and PyTorch 2.1+ are supported. From a checkout:

```powershell
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pytest
ruff check .
```

If a local Python installation has unrelated globally registered pytest plugins, use the clean-project command used by CI:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest
```

## Minimal example

```python
import torch

from spectral_ebm import SpectralEBM, langevin_sample

model = SpectralEBM(dim=32, hidden_layers=3)
initial = torch.randn(16, 32)
samples = langevin_sample(model, initial, steps=10, step_size=0.01)
print(samples.shape)
```

## Reproduce experiments

Run the full benchmark harness from the repository root:

```powershell
python -m benchmarks.full_benchmark --device cuda --dims 128 256 512 1024 2048 --batch-size 64 --repeats 20 --output benchmark_results/full-cuda.json
```

The committed artifacts in [benchmark_results](benchmark_results) include CPU/CUDA timing, parameter counts, large-dimension measurements, Gaussian score matching, a four-mode Gaussian mixture, and a noisy-ring experiment. The benchmark protocol is documented in [docs/benchmark_protocol.md](docs/benchmark_protocol.md). Results are hardware-specific and include mixed outcomes; they are not universal speed claims.
## Claims and prior art

FFT/circulant projections and their asymptotic parameter and arithmetic reductions are established prior art, including [Cheng et al. (ICCV 2015)](https://openaccess.thecvf.com/content_iccv_2015/html/Cheng_An_Exploration_of_ICCV_2015_paper.html) and [CirCNN](https://arxiv.org/abs/1708.08917). EBM training with Langevin sampling is also established. The repository therefore does not describe the basic combination as a new invention. Any future research contribution must be stated narrowly, compared against the closest references, and supported by a theorem or reproducible evidence.

## License

Source code is licensed under the [Apache License 2.0](LICENSE). See [CITATION.cff](CITATION.cff) for citation metadata. Third-party dependencies and datasets retain their own licenses.
