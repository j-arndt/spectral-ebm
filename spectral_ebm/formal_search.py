"""HRR token/AST encoding and continuous formal-state search adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .chains import PersistentLangevin


@dataclass(frozen=True)
class ASTNode:
    """Small parser-agnostic AST container for code or proof-state nodes."""

    kind: str
    children: tuple[ASTNode | str, ...] = ()

    def tokens(self) -> tuple[str, ...]:
        values = [self.kind]
        for child in self.children:
            values.extend(child.tokens() if isinstance(child, ASTNode) else (str(child),))
        return tuple(values)


def state_tokens(state: Sequence[str] | ASTNode) -> tuple[str, ...]:
    """Normalize a token sequence or AST node into a flat token tuple."""

    return state.tokens() if isinstance(state, ASTNode) else tuple(str(token) for token in state)


def hrr_bind(left: Tensor, right: Tensor) -> Tensor:
    """Bind HRR vectors with circular convolution along the final dimension."""

    if left.shape != right.shape:
        raise ValueError("HRR operands must have the same shape")
    return torch.fft.irfft(
        torch.fft.rfft(left, dim=-1) * torch.fft.rfft(right, dim=-1),
        n=left.shape[-1],
        dim=-1,
    )


class HRREncoder(nn.Module):
    """Encode tokenized syntax states as deterministic HRR vectors.

    Each token is bound to a fixed positional role by circular convolution and
    the bound role/value vectors are bundled by summation. The vocabulary and
    role vectors are buffers, not trainable language-model embeddings. This is
    a compact interface for Lean tactic tokens or flattened code AST nodes; it
    does not parse Lean, prove a theorem, or decode a refined vector back into
    a valid syntax state.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        dim: int,
        *,
        max_length: int = 128,
        seed: int = 0,
        unknown_token: str = "<unk>",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1 or max_length < 1:
            raise ValueError("dim and max_length must be positive")
        unique_vocabulary = list(dict.fromkeys(str(token) for token in vocabulary))
        if unknown_token not in unique_vocabulary:
            unique_vocabulary.append(unknown_token)
        self.vocabulary = tuple(unique_vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.unknown_token = unknown_token
        self.dim = int(dim)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        token_vectors = torch.randn(len(self.vocabulary), self.dim, generator=generator)
        position_vectors = torch.randn(self.max_length, self.dim, generator=generator)
        self.register_buffer("token_vectors", F.normalize(token_vectors, dim=-1), persistent=True)
        self.register_buffer("position_vectors", F.normalize(position_vectors, dim=-1), persistent=True)

    def _token_ids(self, tokens: Sequence[str]) -> Tensor:
        unknown_id = self.token_to_id[self.unknown_token]
        return torch.tensor(
            [self.token_to_id.get(str(token), unknown_id) for token in tokens],
            device=self.token_vectors.device,
            dtype=torch.long,
        )

    def encode_state(self, state: Sequence[str] | ASTNode) -> Tensor:
        """Encode one token sequence or AST node into a vector of shape ``(D,)``."""

        tokens = state_tokens(state)
        if len(tokens) > self.max_length:
            raise ValueError(f"state has {len(tokens)} tokens; max_length is {self.max_length}")
        if not tokens:
            return torch.zeros(self.dim, device=self.token_vectors.device, dtype=self.token_vectors.dtype)
        ids = self._token_ids(tokens)
        bound = hrr_bind(self.token_vectors[ids], self.position_vectors[: len(tokens)])
        encoded = bound.sum(dim=0) / len(tokens) ** 0.5
        return F.normalize(encoded, dim=-1) if self.normalize else encoded

    def encode(self, states: Sequence[Sequence[str] | ASTNode]) -> Tensor:
        """Encode a batch of tokenized states with shape ``(batch, D)``."""

        return torch.stack([self.encode_state(state) for state in states])

    def nearest_tokens(self, embeddings: Tensor, top_k: int = 5) -> list[list[str]]:
        """Return diagnostic nearest vocabulary tokens by cosine similarity.

        This is intentionally a diagnostic projection, not a syntax decoder or
        a proof verifier.
        """

        if embeddings.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {embeddings.shape[-1]}")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        normalized = F.normalize(embeddings, dim=-1)
        scores = normalized @ self.token_vectors.transpose(-1, -2)
        indices = scores.topk(min(top_k, len(self.vocabulary)), dim=-1).indices
        return [[self.vocabulary[index] for index in row.tolist()] for row in indices]


@dataclass(frozen=True)
class FormalSearchResult:
    """Continuous refinement result with the original syntax-state tokens."""

    states: tuple[tuple[str, ...], ...]
    initial_embeddings: Tensor
    refined_embeddings: Tensor
    initial_energy: Tensor
    refined_energy: Tensor


class FormalProofSearchAdapter(nn.Module):
    """Bridge tokenized formal states to a persistent continuous EBM chain.

    The adapter is deliberately parser-agnostic: callers may supply Lean 4
    tactic tokens, parser-produced AST nodes, or another code-state token
    sequence. It returns continuous candidates and energy scores. A trusted
    parser, proof checker, and action decoder must be added by an integration
    before any candidate can be called a verified proof.
    """

    def __init__(
        self,
        encoder: HRREncoder,
        energy_model: nn.Module,
        *,
        chain: PersistentLangevin | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.energy_model = energy_model
        self.chain = chain if chain is not None else PersistentLangevin()

    def encode(self, states: Sequence[Sequence[str] | ASTNode]) -> Tensor:
        return self.encoder.encode(states)

    def forward(self, states: Sequence[Sequence[str] | ASTNode]) -> Tensor:
        return self.energy_model(self.encode(states))

    def refine(
        self,
        states: Sequence[Sequence[str] | ASTNode],
        *,
        steps: int,
        step_size: float,
        temperature: float = 1.0,
        bounds: tuple[float, float] | None = None,
        noise_scale: float = 0.0,
    ) -> FormalSearchResult:
        """Run persistent Langevin refinement from HRR-encoded syntax states."""

        normalized_states = tuple(state_tokens(state) for state in states)
        initial = self.encoder.encode(normalized_states)
        initial_energy = self.energy_model(initial).detach()
        refined = self.chain.sample_from(
            self.energy_model,
            initial,
            steps=steps,
            step_size=step_size,
            temperature=temperature,
            bounds=bounds,
            noise_scale=noise_scale,
        )
        refined_energy = self.energy_model(refined).detach()
        return FormalSearchResult(
            states=normalized_states,
            initial_embeddings=initial,
            refined_embeddings=refined,
            initial_energy=initial_energy,
            refined_energy=refined_energy,
        )
