"""Correctness-first spectral/circulant energy-based models."""

from .layers import BlockCirculantLinear, CirculantLinear, block_circulant_matrix, circulant_matrix
from .models import BlockSpectralEBM, DenseEBM, PermutedSpectralEBM, SpectralEBM
from .sampler import langevin_sample, ula_step, vectorized_langevin_chain
from .training import contrastive_divergence_loss, denoising_score_matching_loss

__all__ = [
    "BlockCirculantLinear",
    "BlockSpectralEBM",
    "CirculantLinear",
    "DenseEBM",
    "PermutedSpectralEBM",
    "SpectralEBM",
    "block_circulant_matrix",
    "circulant_matrix",
    "contrastive_divergence_loss",
    "denoising_score_matching_loss",
    "langevin_sample",
    "ula_step",
    "vectorized_langevin_chain",
]