# Algebraic proof note

## Circulant convention

Let `c = (c_0, ..., c_{D-1})` and define

```text
W[i, j] = c[(i - j) mod D].
```

The first column is `W[:, 0] = c`. For any `x`,

```text
(W x)_i = sum_j c[(i - j) mod D] x_j.
```

Setting `m = i - j mod D` gives

```text
(W x)_i = sum_m c_m x[(i - m) mod D],
```

which is circular convolution. PyTorch's default `rfft`/`irfft` normalization implements this convolution as

```text
irfft(rfft(x) * rfft(c), n=D).
```

The test suite compares this expression to the materialized matrix for dimensions including both odd and even values.

## Fourier diagonalization

Let `F` be the unnormalized DFT with entries `F[k, j] = exp(-2 pi i k j / D)`. The Fourier basis vectors diagonalize `W` and the eigenvalues are

```text
lambda_k = sum_m c_m exp(-2 pi i k m / D) = (F c)_k.
```

Thus

```text
W = F^{-1} diag(F c) F,
```

with the usual inverse normalization. Since `W` is normal, its operator 2-norm is `max_k |lambda_k|`; `CirculantLinear.spectral_norm()` computes that value exactly.

## Complexity and limitation

Materializing or multiplying by an arbitrary dense `D x D` matrix requires quadratic storage and arithmetic. The FFT representation stores `D` generator values and computes the convolution in `O(D log D)` arithmetic. These are asymptotic statements; actual wall-clock speed depends on device, batch size, FFT kernels, dtype, and memory traffic.

The restriction is substantive: one circulant map has only `D` degrees of freedom and is equivariant to cyclic shifts of the feature index. It is not an arbitrary dense projection and may be a poor inductive bias for unordered semantic embeddings.

## ULA convention

For target density

```text
p(x) = Z^(-1) exp(-E(x) / T),
```

the score is `grad log p(x) = -grad E(x) / T`. The Euler-Maruyama discretization of the Langevin diffusion with unit diffusion is

```text
x_next = x + h/2 grad log p(x) + sqrt(h) epsilon
        = x - h/(2T) grad E(x) + sqrt(h) epsilon.
```

The implementation exposes a separate deterministic `relax()` mode by setting the noise scale to zero. Short chains are approximations; they are not asserted to be equilibrated samples.

## Multi-channel block-circulant extension

Let `x` have shape `(..., C_in, D)` and let `c[o, i, :]` be one generator for each output/input channel pair. The block-circulant layer computes

```text
y[o] = sum_i circ(c[o, i]) x[i] + b[o]
```

where every block follows the same convention `W[i, j] = c[(i - j) mod D]`. In the Fourier domain, for each frequency `k`, this becomes a dense channel mixing matrix:

```text
ŷ[o, k] = sum_i ĉ[o, i, k] x̂[i, k].
```

The layer therefore costs `O(C_out C_in D log D)` arithmetic and stores `C_out C_in D + C_out D` parameters with bias. Its exact operator norm is

```text
max_k ||Ĉ[k]||_2,
```

where `Ĉ[k]` is the `C_out × C_in` Fourier symbol at frequency `k`. `BlockCirculantLinear.materialize()` and `block_circulant_matrix()` provide a deliberately slow dense oracle for small dimensions.

This extension adds channel mixing but does not remove the cyclic structure inside each feature-axis block. It is not equivalent to an unconstrained dense map.

## Permutation interleaving

`PermutedSpectralEBM` inserts fixed coordinate permutations between nonlinear spectral layers. If `S` is the cyclic shift operator and `P` is a fixed permutation, the interleaved map is generally not equivariant to `S` because `PS != SP`. The permutations are not learned: they are deterministic buffers generated from a local seed and serialized with the model. This is an architectural expressivity heuristic, not a theorem that the resulting nonlinear scalar energy is universally expressive.

## Persistent Langevin execution

`vectorized_langevin_chain()` reuses one detached state buffer and performs the drift/noise/bound update under `no_grad` after extracting the current input gradient. It is an allocation optimization, not graph pre-caching and not a different sampler. With a fixed `noise_sequence`, tests verify exact agreement with repeated `ula_step()` calls. The unavoidable input-gradient autograd work is still performed at every step.