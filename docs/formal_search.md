# Formal-state search interface

The formal-search adapter is a parser-agnostic bridge between discrete syntax states and the continuous EBM runtime. It is designed for integration with Lean 4 tactic token streams, code AST nodes, or another trusted parser owned by the host system.

## HRR encoding

For token `t_i` at position `i`, the encoder stores a fixed token vector `v(t_i)` and fixed role vector `r_i`. Binding is circular convolution:

```text
b_i = v(t_i) * r_i
```

The state representation is a bundled superposition:

```text
h(tokens) = normalize(1 / sqrt(n) * sum_i b_i)
```

The vectors are generated from a local seed and registered as buffers. The encoder therefore has deterministic serialization and does not silently call an external language model.

## Continuous refinement

`FormalProofSearchAdapter.refine()` performs:

1. token or AST normalization;
2. HRR encoding into a batch of `D`-dimensional vectors;
3. energy evaluation;
4. persistent-state Langevin refinement through `sample_from()`;
5. final energy evaluation and a structured `FormalSearchResult`.

`noise_scale=0` gives deterministic gradient relaxation; positive noise uses the documented ULA update. The result retains the original token tuples so an external decoder/checker can associate continuous candidates with source states.

## Trust boundary

This module intentionally does not claim formal verification. It does not parse Lean, synthesize tactic syntax, enforce typing, execute tactics, or check theorem kernels. `nearest_tokens()` is only a cosine-similarity diagnostic over the vocabulary and is not a syntax decoder. A production integration must place a trusted parser, action decoder, sandboxed tactic executor, and proof checker around the adapter.

## Minimal AST example

```python
from spectral_ebm import ASTNode, FormalProofSearchAdapter, HRREncoder, SpectralEBM

state = ASTNode("app", (ASTNode("identifier", ("Nat.succ",)), "n"))
encoder = HRREncoder(["app", "identifier", "Nat.succ", "n"], dim=64)
adapter = FormalProofSearchAdapter(encoder, SpectralEBM(64, hidden_layers=2))
result = adapter.refine([state], steps=4, step_size=0.01, noise_scale=0.0)
```
## Spherical HRR refinement

FormalProofSearchAdapter uses radius-one spherical projected ULA by default. The energy gradient and noise are projected to the tangent plane, then the updated embedding is retracted to unit norm. Pass spherical=False to select the Euclidean persistent-chain path.

This preserves the HRR encoder's normalized representation surface but does not make continuous candidates into valid Lean syntax or proofs. The parser, decoder, tactic executor, and kernel checker remain outside the adapter trust boundary.
