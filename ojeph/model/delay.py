"""Acoustic delay pattern.

Codebook 0 (semantic) is emitted on time; codebooks 1..K-1 (acoustic detail)
are shifted `delay` frames later. The model decides *what* to say first and
fills in *how it sounds* one frame later. Delayed slots are filled with pad.
"""

from __future__ import annotations

import torch


def apply_delay(codes: torch.Tensor, delay: int, pad: int) -> torch.Tensor:
    """codes: [..., K, T] -> same shape, acoustic codebooks shifted right."""
    if delay == 0:
        return codes.clone()
    out = torch.full_like(codes, pad)
    out[..., 0, :] = codes[..., 0, :]
    out[..., 1:, delay:] = codes[..., 1:, :-delay]
    return out


def undo_delay(codes: torch.Tensor, delay: int) -> torch.Tensor:
    """Inverse of apply_delay; drops the last `delay` frames (incomplete)."""
    if delay == 0:
        return codes.clone()
    T = codes.shape[-1]
    out = codes[..., :T - delay].clone()
    out[..., 1:, :] = codes[..., 1:, delay:]
    return out
