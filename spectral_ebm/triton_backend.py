"""Optional Triton acceleration for block-circulant frequency mixing.

The FFT transforms remain explicit torch.fft operations backed by cuFFT. The
Triton kernel fuses the complex frequency-bin channel contraction so the
intermediate unsqueeze/einsum tensor is never materialized in global memory.
"""

from __future__ import annotations

import os
import shutil

import torch
from torch import Tensor

try:  # Triton is an optional accelerator dependency.
    import triton
    import triton.language as tl

    _TRITON_IMPORTED = True
except (ImportError, RuntimeError):  # pragma: no cover - depends on environment.
    _TRITON_IMPORTED = False


def triton_available() -> bool:
    """Return whether the optional Triton backend can be requested."""

    return _TRITON_IMPORTED and torch.cuda.is_available()


def triton_runtime_available() -> bool:
    """Return whether Triton can initialize its CUDA compiler/runtime."""

    if not triton_available():
        return False
    compiler = os.environ.get("CC") or shutil.which("cl") or shutil.which("gcc") or shutil.which("clang")
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
    def _frequency_mix_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        rows,
        in_channels,
        out_channels,
        frequencies,
        BLOCK_IN: tl.constexpr,
    ):
        program = tl.program_id(0)
        total = rows * out_channels * frequencies
        if program >= total:
            return
        frequency = program % frequencies
        quotient = program // frequencies
        out_channel = quotient % out_channels
        row = quotient // out_channels
        input_channels = tl.arange(0, BLOCK_IN)
        mask = input_channels < in_channels
        x_offset = ((row * in_channels + input_channels) * frequencies + frequency) * 2
        weight_offset = (
            (out_channel * in_channels + input_channels) * frequencies + frequency
        ) * 2
        x_real = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        x_imag = tl.load(x_ptr + x_offset + 1, mask=mask, other=0.0).to(tl.float32)
        weight_real = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0).to(tl.float32)
        weight_imag = tl.load(weight_ptr + weight_offset + 1, mask=mask, other=0.0).to(tl.float32)
        output_real = tl.sum(x_real * weight_real - x_imag * weight_imag, axis=0)
        output_imag = tl.sum(x_real * weight_imag + x_imag * weight_real, axis=0)
        output_offset = ((row * out_channels + out_channel) * frequencies + frequency) * 2
        tl.store(output_ptr + output_offset, output_real)
        tl.store(output_ptr + output_offset + 1, output_imag)


def _forward(x_fft: Tensor, weight_fft: Tensor) -> Tensor:
    if not triton_runtime_available():
        raise RuntimeError("Triton requires CUDA, Triton, and a working C compiler/runtime; use backend=\"torch\" otherwise")
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
    block_in = _next_power_of_two(in_channels)
    grid = (rows * out_channels * frequencies,)
    _frequency_mix_kernel[grid](
        torch.view_as_real(x_flat),
        torch.view_as_real(weight_flat),
        output_real,
        rows,
        in_channels,
        out_channels,
        frequencies,
        BLOCK_IN=block_in,
    )
    return torch.view_as_complex(output_real).reshape(*x_fft.shape[:-2], out_channels, frequencies)


class _TritonFrequencyMix(torch.autograd.Function):
    """Autograd wrapper: Triton forward, exact torch complex backward."""

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
