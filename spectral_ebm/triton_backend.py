"""Optional tiled Triton acceleration for block-circulant frequency mixing.

The FFT transforms remain explicit torch.fft operations backed by cuFFT. The
Triton kernel tiles rows, output channels, and frequency bins while processing
input channels in bounded chunks. It fuses the complex frequency contraction;
it does not replace cuFFT internal FFT stages.
"""

from __future__ import annotations

import os
import shutil

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORTED = True
except (ImportError, RuntimeError):
    _TRITON_IMPORTED = False


def triton_available() -> bool:
    """Return whether Triton is importable and CUDA is available."""

    return _TRITON_IMPORTED and torch.cuda.is_available()


def triton_runtime_available() -> bool:
    """Return whether Triton can initialize its CUDA compiler/runtime."""

    if not triton_available():
        return False
    compiler = (
        os.environ.get("CC")
        or shutil.which("cl")
        or shutil.which("gcc")
        or shutil.which("clang")
    )
    if compiler is None:
        return False
    try:
        triton.runtime.driver.active.get_current_device()
    except (RuntimeError, OSError):
        return False
    return True


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


if _TRITON_IMPORTED:

    @triton.jit
    def _fused_warp_parallel_mix_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        rows,
        in_channels,
        out_channels,
        frequencies,
        BLOCK_R: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_o = tl.program_id(1)
        pid_f = tl.program_id(2)
        offsets_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offsets_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        offsets_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        mask_r = offsets_r < rows
        mask_o = offsets_o < out_channels
        mask_f = offsets_f < frequencies
        acc_real = tl.zeros((BLOCK_R, BLOCK_O, BLOCK_F), dtype=tl.float32)
        acc_imag = tl.zeros((BLOCK_R, BLOCK_O, BLOCK_F), dtype=tl.float32)

        for input_start in range(0, in_channels, BLOCK_I):
            offsets_i = input_start + tl.arange(0, BLOCK_I)
            mask_i = offsets_i < in_channels
            x_offsets = (
                (
                    offsets_r[:, None, None] * in_channels
                    + offsets_i[None, :, None]
                )
                * frequencies
                + offsets_f[None, None, :]
            ) * 2
            x_mask = (
                mask_r[:, None, None]
                & mask_i[None, :, None]
                & mask_f[None, None, :]
            )
            x_real = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
            x_imag = tl.load(x_ptr + x_offsets + 1, mask=x_mask, other=0.0).to(tl.float32)

            weight_offsets = (
                (
                    offsets_o[:, None, None] * in_channels
                    + offsets_i[None, :, None]
                )
                * frequencies
                + offsets_f[None, None, :]
            ) * 2
            weight_mask = (
                mask_o[:, None, None]
                & mask_i[None, :, None]
                & mask_f[None, None, :]
            )
            weight_real = tl.load(
                weight_ptr + weight_offsets, mask=weight_mask, other=0.0
            ).to(tl.float32)
            weight_imag = tl.load(
                weight_ptr + weight_offsets + 1, mask=weight_mask, other=0.0
            ).to(tl.float32)

            product_real = (
                x_real[:, None, :, :] * weight_real[None, :, :, :]
                - x_imag[:, None, :, :] * weight_imag[None, :, :, :]
            )
            product_imag = (
                x_real[:, None, :, :] * weight_imag[None, :, :, :]
                + x_imag[:, None, :, :] * weight_real[None, :, :, :]
            )
            acc_real += tl.sum(product_real, axis=2)
            acc_imag += tl.sum(product_imag, axis=2)

        output_offsets = (
            (
                offsets_r[:, None, None] * out_channels
                + offsets_o[None, :, None]
            )
            * frequencies
            + offsets_f[None, None, :]
        ) * 2
        output_mask = mask_r[:, None, None] & mask_o[None, :, None] & mask_f[None, None, :]
        tl.store(output_ptr + output_offsets, acc_real, mask=output_mask)
        tl.store(output_ptr + output_offsets + 1, acc_imag, mask=output_mask)


def _tile_config(
    in_channels: int, out_channels: int, frequencies: int
) -> tuple[int, int, int, int]:
    block_r = 1
    block_o = 8 if out_channels >= 8 else _next_power_of_two(out_channels)
    block_f = 32 if frequencies >= 32 else _next_power_of_two(frequencies)
    block_i = 32 if in_channels >= 32 else _next_power_of_two(in_channels)
    return block_r, block_o, block_f, block_i


def _forward(x_fft: Tensor, weight_fft: Tensor) -> Tensor:
    if not triton_runtime_available():
        raise RuntimeError(
            "Triton requires CUDA, Triton, and a working C compiler/runtime; "
            'use backend="torch" otherwise'
        )
    if not x_fft.is_cuda or not weight_fft.is_cuda or x_fft.device != weight_fft.device:
        raise ValueError("Triton inputs must be CUDA tensors on the same device")
    if x_fft.dtype != torch.complex64 or weight_fft.dtype != torch.complex64:
        raise TypeError("Triton backend currently supports float32 block-circulant weights only")
    if x_fft.ndim < 3 or weight_fft.ndim != 3:
        raise ValueError("expected x_fft (..., C_in, F) and weight_fft (C_out, C_in, F)")
    if x_fft.shape[-2] != weight_fft.shape[-2] or x_fft.shape[-1] != weight_fft.shape[-1]:
        raise ValueError("input and weight Fourier shapes are incompatible")
    rows = x_fft.numel() // (x_fft.shape[-2] * x_fft.shape[-1])
    in_channels = x_fft.shape[-2]
    out_channels = weight_fft.shape[0]
    frequencies = x_fft.shape[-1]
    x_flat = x_fft.contiguous().reshape(rows, in_channels, frequencies)
    weight_flat = weight_fft.contiguous()
    output_real = torch.empty(
        (rows, out_channels, frequencies, 2),
        device=x_fft.device,
        dtype=torch.float32,
    )
    block_r, block_o, block_f, block_i = _tile_config(
        in_channels, out_channels, frequencies
    )
    grid = (
        (rows + block_r - 1) // block_r,
        (out_channels + block_o - 1) // block_o,
        (frequencies + block_f - 1) // block_f,
    )
    _fused_warp_parallel_mix_kernel[grid](
        torch.view_as_real(x_flat),
        torch.view_as_real(weight_flat),
        output_real,
        rows,
        in_channels,
        out_channels,
        frequencies,
        BLOCK_R=block_r,
        BLOCK_O=block_o,
        BLOCK_F=block_f,
        BLOCK_I=block_i,
    )
    return torch.view_as_complex(output_real).reshape(*x_fft.shape[:-2], out_channels, frequencies)


class _TritonFrequencyMix(torch.autograd.Function):
    """Autograd wrapper: tiled Triton forward, exact torch complex backward."""

    @staticmethod
    def forward(ctx, x_fft: Tensor, weight_fft: Tensor) -> Tensor:
        ctx.save_for_backward(x_fft, weight_fft)
        return _forward(x_fft, weight_fft)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        x_fft, weight_fft = ctx.saved_tensors
        grad_x = torch.einsum("...of,oif->...if", grad_output, weight_fft.conj())
        grad_weight = torch.einsum("...if,...of->oif", x_fft.conj(), grad_output)
        return grad_x, grad_weight


def block_circulant_frequency_mix(x_fft: Tensor, weight_fft: Tensor) -> Tensor:
    """Fuse the complex pointwise channel contraction with Triton autograd."""

    return _TritonFrequencyMix.apply(x_fft, weight_fft)
