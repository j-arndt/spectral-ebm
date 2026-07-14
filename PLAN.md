# Spectral EBM: Correctness-First Proof-of-Concept Plan

Status: public v0.2.0 architectural extension published and verified; longer-term research and documentation improvements remain on the roadmap.

## Release audit (2026-07-14)

- Public repository: https://github.com/j-arndt/spectral-ebm
- Current main commit: `62b1bde`
- Current release target: `v0.2.0` with a GitHub Release and Apache-2.0 license
- Local verification: 24 tests passed, Ruff passed, and a 0.2.0 wheel was built successfully
- Extension verification: block-circulant reference/norm checks, permutation serialization checks, fixed-noise chain equivalence, and a 10-repeat CPU artifact
- Remote verification: GitHub Actions passed on Python 3.10 and 3.12
- Public presentation: README hero plots, extension plots, result tables, raw JSON artifacts, About description, homepage, and discovery topics

## Final goal

Deliver a finished, polished public repository containing a mathematically correct spectral/circulant energy-based-model proof of concept, reproducible tests and benchmarks, an auditable proof note, a clear README, proper licensing and attribution, CI, release metadata, and version tags.

The released result must be honest about scope: the first milestone is a continuous EBM POC. It is not a formal-proof verifier until a separately specified encoder, state-transition interface, and trusted Lean/Rust/formal checker are implemented and evaluated.

## Correctness decisions that gate implementation

1. **Fix one DFT convention and test it.** Define whether the learned vector is the first column or first row of the circulant matrix, and derive the exact PyTorch `rfft`/`irfft` identity under its normalization. The implementation must agree with an explicitly materialized reference matrix for odd and even dimensions, including the sign/shift convention.

2. **State the representation trade-off.** A single square circulant map has only `D` learned coefficients and is not an arbitrary `D x D` projection. It is translation-equivariant on the cyclic feature index and can lose information relevant to unordered semantic embeddings. The POC must measure this trade-off rather than imply equivalence to a dense layer.

3. **Separate relaxation from sampling.** A noisy gradient step is only an Unadjusted Langevin Algorithm (ULA) step for a specified target temperature when drift and noise are tied consistently. Bounded domains require a stated projection, reflection, or another valid boundary treatment. We will expose temperature, step size, noise convention, and domain handling in the API.

4. **Define the EBM objective precisely.** We will distinguish maximum-likelihood/contrastive-divergence training, score matching, and deterministic energy relaxation. Negative chains will be detached or differentiated through deliberately, with the choice documented and tested. Energy offsets and regularization will be handled without accidentally changing the intended density.

5. **Do not claim universal guarantees.** Circulant structure gives parameter and asymptotic arithmetic reductions. It does not by itself guarantee faster wall-clock execution on every device, perfect Hessian conditioning, faster mixing, formal-proof validity, or superiority to dense networks. Those are empirical or theorem-specific claims and must earn their place through evidence.

## Milestone and task list

### Phase 0 — Specification and repository foundation

- [ ] Choose supported Python and PyTorch versions; add `pyproject.toml` with pinned development/test tooling.
- [ ] Add a minimal package layout, importable API, type hints, formatting/linting, and deterministic seed utilities.
- [ ] Write the notation and assumptions for the feature domain, temperature, DFT normalization, matrix orientation, and energy density.
- [ ] Define the POC hypothesis and the null hypothesis. At minimum: parameter compression and asymptotic scaling are expected; task quality and wall-clock speed are measured, not assumed.
- [ ] Add contribution, citation, security, and reproducibility guidance before accepting external contributions.

### Phase 1 — Algebraic core and proof artifacts

- [ ] Implement `CirculantLinear` with a spatial-generator parameterization.
- [ ] Implement an optional frequency-parameterized form only if Hermitian constraints and real-valued reconstruction are specified explicitly.
- [ ] Add a slow reference implementation that materializes the matrix for small `D`.
- [ ] Prove in documentation that the FFT implementation equals circular convolution and equals the reference matrix under the chosen convention.
- [ ] Verify forward values, input gradients, parameter gradients, dtype behavior, device behavior, and serialization against the reference.
- [ ] Test odd/even dimensions, `D=1`, non-power-of-two dimensions, batch and extra leading dimensions, zero and impulse inputs, and non-contiguous tensors.
- [ ] Add spectral norm/eigenvalue diagnostics for a single layer; do not generalize these diagnostics to the nonlinear network without proof.

### Phase 2 — Continuous EBM and sampler

- [ ] Implement a scalar-energy network with matched spectral and dense variants.
- [ ] Implement ULA with a mathematically explicit target `p(x) ∝ exp(-E(x)/T)` and a separate deterministic relaxation mode.
- [ ] Add optional persistent chains, burn-in, thinning, chain diagnostics, and bounded-domain handling.
- [ ] Implement at least two training objectives: a documented short-run contrastive-divergence baseline and a sampler-independent score-matching or denoising-score-matching baseline for controlled comparison.
- [ ] Add checks for finite energy, finite gradients, exploding chains, collapsed chains, acceptance/diagnostic metrics where applicable, and energy-shift invariance where applicable.
- [ ] Make data normalization and the meaning of the learned energy explicit in every experiment.

### Phase 3 — Proof-of-concept experiments

- [ ] Start with distributions whose density is known: a 2-D Gaussian, a Gaussian mixture, and a ring or other multimodal distribution.
- [ ] Evaluate density shape, sample quality, mode coverage, calibration/energy ranking, and mixing diagnostics. For low dimensions, numerically estimate the partition function on a grid so likelihood-related claims are checkable.
- [ ] Add a higher-dimensional synthetic dataset to test scaling independently of semantic-embedding quality.
- [ ] Compare spectral and dense models at matched depth, width, activation, initialization, optimizer, training budget, and parameter count where possible.
- [ ] Add ablations for one versus multiple circulant maps, bias, normalization, temperature, sampler step count, persistence, and spectral versus spatial parameterization.
- [ ] Record all seeds, configs, package versions, device details, checkpoints, and raw metrics in machine-readable files.

### Phase 4 — Benchmarking

- [ ] Correct the baseline accounting: parameter totals must include every bias and output projection actually instantiated.
- [ ] Benchmark forward inference, forward-plus-input-gradient, one ULA step, and complete short chains separately.
- [ ] Sweep dimensions such as 128, 256, 512, 1024, 2048, and larger values only when the hardware can run both models fairly.
- [ ] Sweep batch sizes and report CPU and CUDA separately when available; include device name, PyTorch version, dtype, precision mode, thread counts, and whether compilation is enabled.
- [ ] Use synchronized, warmed-up timing with repeated trials and report median plus dispersion. Do not report a single `time.time()` result as a benchmark.
- [ ] Report parameter count, serialized size, peak memory, FLOPs or operation estimates, throughput, and latency. Label theoretical complexity separately from observed wall-clock performance.
- [ ] Include a dense baseline, the proposed spectral model, and at least one additional structured baseline when practical so the comparison is not only against an intentionally weak strawman.
- [ ] Publish the exact benchmark command and raw results; generate figures from those raw results in a reproducible script.

### Phase 5 — Novelty, provenance, and licensing gate

- [ ] Maintain a novelty ledger with each proposed contribution, closest prior art, distinguishing limitation, evidence, and confidence level.
- [ ] Search scholarly literature, arXiv, Google Patents, USPTO Patent Public Search, and relevant formal-methods work before making novelty or superiority claims.
- [ ] Treat the basic FFT/circulant layer and its `O(D^2)` to `O(D log D)` / `O(D)` reductions as established prior art. Candidate contribution areas must therefore be narrower and evidence-backed: for example, a specific EBM objective/sampler coupling, a formally analyzed architecture, or a demonstrated proof-state interface.
- [ ] Add citations for all borrowed equations, algorithms, datasets, code, and benchmark conventions. Preserve license notices for dependencies and copied snippets.
- [ ] Use **Apache-2.0 for source code** as the default recommendation: it is OSI-approved, permissive for research and commercial adoption, and includes an express contributor patent license. Use a separate permissive documentation/data license only when the relevant artifact and its upstream terms allow it; never relabel third-party data.
- [ ] If the priority is to require distributed derivatives to remain open, evaluate GPLv3/AGPLv3 instead before release. Do not create a custom license.
- [ ] Add `LICENSE`, `NOTICE` if needed, `CITATION.cff`, copyright headers where appropriate, dependency/license inventory, and a clear third-party attribution section.
- [ ] If patent protection is a real objective, pause public release of enabling details until patent counsel evaluates filing strategy. Public disclosure can destroy novelty in many jurisdictions; a U.S. grace period is not a global safe harbor.

### Phase 6 — Public-repository polish and release

- [ ] Write a README that leads with verified results, states limitations, explains installation, runs a tiny example, links the proof note, and reproduces the benchmark.
- [ ] Add API documentation and a mathematical appendix covering the DFT convention, circular convolution identity, parameter counts, complexity, and sampler assumptions.
- [ ] Add CI for unit tests, gradient checks, linting, documentation build, and a small CPU smoke benchmark.
- [ ] Add GitHub issue templates, pull-request checklist, Code of Conduct, security policy, and funding/citation metadata only if wanted.
- [ ] Create a first release only after all acceptance criteria below pass; tag a semantic version and archive benchmark artifacts.
- [ ] Add repository topics/tags that describe the implementation accurately, such as `energy-based-model`, `circulant-matrix`, `fourier`, `pytorch`, `mcmc`, and `benchmark`, avoiding unsupported claims such as `formal-verification` until that integration exists.

## Acceptance criteria for the first public release

- The FFT layer matches the materialized circulant reference within documented floating-point tolerances for all tested shapes and dtypes.
- Autograd checks pass for inputs and parameters; the implementation is deterministic when the seed and deterministic settings are fixed.
- The sampler update is dimensionally and probabilistically specified, and diagnostics show that it behaves correctly on at least one analytically understood target.
- Spectral and dense baselines are matched and their parameter totals are independently recomputed by the benchmark.
- Every reported result can be regenerated from a clean checkout with one documented command, subject to hardware availability.
- Raw benchmark data, configs, environment metadata, and figure-generation code are committed or released as artifacts.
- The README contains no unsupported claims about perfect conditioning, universal speedups, or formal-proof correctness.
- License, attribution, and dependency checks pass, and any patent/novelty decision is explicitly recorded.

## Initial prior-art boundary

The core idea is not novel in isolation. Earlier work already describes circulant projections in fully connected networks and FFT multiplication with the same asymptotic storage and arithmetic reductions, including [Cheng et al., ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Cheng_An_Exploration_of_ICCV_2015_paper.html) and [CirCNN](https://arxiv.org/abs/1708.08917). More recent work also covers block-circulant adapters and spectral/circulant layers, including [Block Circulant Adapter for LLMs](https://www.ijcai.org/proceedings/2025/560), [Parameter-Efficient Fine-Tuning with Circulant and Diagonal Vectors](https://www.ijcai.org/proceedings/2025/1021), and [Compact Circulant Layers with Spectral Priors](https://arxiv.org/abs/2602.21965).

The repository should therefore present novelty, if any, as a narrowly defined and experimentally supported combination or theorem—not as the invention of FFT-based circulant neural layers. The EBM/Langevin components also have substantial prior art; for example, [Improved Contrastive Divergence Training of Energy Based Models](https://proceedings.mlr.press/v139/du21b/du21b.pdf) discusses stability and omitted gradient terms.

## Legal and licensing references

- [GitHub guidance on licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- [OSI license list](https://opensource.org/licenses)
- [Apache License 2.0, including the patent grant](https://www.apache.org/licenses/LICENSE-2.0)
- [WIPO patent FAQ on public disclosure and novelty](https://www.wipo.int/en/web/patents/faq_patents)
- [USPTO provisional-application guidance](https://www.uspto.gov/patents/basics/apply/provisional-application)

This plan is engineering and research guidance, not legal advice. Patentability, ownership, contributor rights, and the final license should be confirmed with qualified counsel if they matter commercially.
