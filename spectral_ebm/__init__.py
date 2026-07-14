"""Correctness-first spectral/circulant energy-based models."""

from .formal_search import (
    ASTNode,
    FormalProofSearchAdapter,
    FormalSearchResult,
    HRREncoder,
    hrr_bind,
)
from .layers import BlockCirculantLinear, CirculantLinear, block_circulant_matrix, circulant_matrix
from .models import BlockSpectralEBM, DenseEBM, PermutedSpectralEBM, SpectralEBM
from .permutations import DifferentiablePermutation
from .sampler import langevin_sample, ula_step, vectorized_langevin_chain
from .training import contrastive_divergence_loss, denoising_score_matching_loss

__all__ = [
    "ASTNode",
    "BlockCirculantLinear",
    "BlockSpectralEBM",
    "CirculantLinear",
    "DenseEBM",
    "FormalProofSearchAdapter",
    "FormalSearchResult",
    "DifferentiablePermutation",
    "PermutedSpectralEBM",
    "SpectralEBM",
    "HRREncoder",
    "block_circulant_matrix",
    "circulant_matrix",
    "contrastive_divergence_loss",
    "denoising_score_matching_loss",
    "hrr_bind",
    "langevin_sample",
    "ula_step",
    "vectorized_langevin_chain",
]
