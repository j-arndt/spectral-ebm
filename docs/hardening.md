# Production hardening note

v0.4.0 adds opt-in controls for high-dimensional structured inference and formal-state refinement. The controls preserve the correctness-first defaults: the torch block-circulant path remains portable, fixed permutations remain backward compatible, and every accelerator or manifold claim is bounded by an explicit test or diagnostic.

## Tiled Triton contraction

The Triton kernel maps a three-dimensional grid over row, output-channel, and frequency tiles. Each program accumulates a small register tile and loops over input channels in BLOCK_I chunks. This avoids a full power-of-two register vector for the complete input-channel dimension and reuses loaded values across several output channels and frequency bins.

The kernel receives the real and imaginary views of complex64 FFT spectra. It computes the pointwise complex contraction and writes the output spectrum. The forward layer still calls torch.fft.rfft and torch.fft.irfft, which are backed by the platform FFT library. Therefore this is a fused frequency-mixing kernel, not an all-in-one FFT implementation, and performance must be measured on the target GPU and compiler stack.

The optional test compares forward values and input/weight gradients against the torch einsum reference. When Triton cannot initialize its compiler/runtime, the test skips and the benchmark records an unavailable status rather than a fabricated timing.

## Householder coordinate mixing

A Householder reflection with unit vector v is

    H(v) x = x - 2 v (v^T x).

Each reflection is orthogonal. A product of K reflections is therefore orthogonal and preserves the Euclidean norm. AmortizedHouseholderPermutation stores only K*D vectors and applies the product without materializing a D by D matrix. The matrix method exists only for small diagnostics.

This layer is deliberately described as an orthogonal mixer rather than a strict coordinate permutation. It removes the quadratic Sinkhorn parameter footprint while retaining a differentiable, norm-preserving transformation. The existing Sinkhorn relaxation remains available for workloads where a doubly-stochastic coordinate assignment is specifically desired.

## Spherical projected Langevin

For a state x on a radius-r sphere, define u = x/r. The tangent projection of a vector g is

    Pi_x(g) = g - (g^T u) u.

The spherical step uses the projected energy gradient and projected Gaussian noise:

    x_tilde = x - h/(2T) Pi_x(grad E(x)) + sqrt(h) Pi_x(epsilon),
    x_next = r x_tilde / ||x_tilde||.

This is a projected ULA/retraction method. It preserves the chosen norm and is useful for HRR state refinement, but it is not claimed to be an exact geodesic sampler or an exact stationary sampler for every constrained target. FormalProofSearchAdapter uses radius 1 and spherical refinement by default; its spherical=False option selects the existing Euclidean persistent-chain behavior.

## Audit protocol

The hardening harness reports:

- parameter memory for a D by D Sinkhorn matrix versus K*D Householder vectors;
- CUDA peak allocation when the run uses CUDA;
- minimum, maximum, and mean per-vector norms after each spherical step.

The scale command is:

    python -m benchmarks.hardening_audit --device cuda --dim 4096 --batch-size 64 --channels 8 --output benchmark_results/hardening-cuda.json

For ordinary CPU development, use the committed small smoke command:

    python -m benchmarks.hardening_audit --device cpu --dim 64 --batch-size 4 --channels 2 --output benchmark_results/hardening-cpu.json
