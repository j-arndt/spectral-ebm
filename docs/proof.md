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
