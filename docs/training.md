# Training conventions

## Short-run contrastive divergence

Given positive data `x+` and a negative chain state `x-`, the implemented baseline minimizes

```text
L_CD = mean(E_theta(x+)) - mean(E_theta(stop_gradient(x-))).
```

This lowers positive energy and raises negative energy. Detaching the chain is deliberate: it is the common short-run contrastive-divergence approximation, not exact differentiation through an equilibrated Markov chain.

## Denoising score matching

For noisy observations `y = x + sigma epsilon`, the conditional denoising score target is `-epsilon / sigma^2`. Since the model score is `-grad(E_theta(y)) / T`, the loss compares those two scores. Energy offsets and additive biases do not affect the score and are therefore not identifiable from this objective; tests do not require a bias parameter to receive a gradient under score matching.

## What is not claimed

Short-run CD and ULA chains can be biased and can mix poorly. The repository reports them as approximations and exposes the relevant step size, temperature, chain length, and boundary assumptions. No convergence or likelihood guarantee is inferred from a decreasing training loss alone.
