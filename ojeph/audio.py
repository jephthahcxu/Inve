"""Audio helpers: PCM conversion and stateful 8 kHz <-> 24 kHz resampling.

Phone calls are 8 kHz narrowband; the codec runs at 24 kHz. Resamplers keep
their filter history between chunks so there are no clicks at 20 ms edges.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def pcm16_to_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _lowpass(taps: int, cutoff: float) -> torch.Tensor:
    """Windowed-sinc lowpass; cutoff as fraction of Nyquist."""
    n = torch.arange(taps) - (taps - 1) / 2
    h = cutoff * torch.sinc(cutoff * n)
    w = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(taps) / (taps - 1))
    h = h * w
    return h / h.sum()


class Resampler:
    """Integer-ratio streaming resampler (up or down)."""

    def __init__(self, up: int = 1, down: int = 1, taps: int = 63):
        assert (up == 1) != (down == 1), "use pure up- or down-sampling"
        self.up, self.down = up, down
        ratio = max(up, down)
        self.h = (_lowpass(taps, 0.9 / ratio) * (up if up > 1 else 1)).view(1, 1, -1)
        self.hist = torch.zeros(taps - 1)
        self.phase = 0  # for decimation alignment across chunks

    def __call__(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if self.up > 1:
            z = torch.zeros(t.numel() * self.up)
            z[::self.up] = t
            t = z
        buf = torch.cat([self.hist, t])
        self.hist = buf[-(self.h.shape[-1] - 1):].clone()
        y = F.conv1d(buf.view(1, 1, -1), self.h).view(-1)
        if self.down > 1:
            y = y[self.phase::self.down]
            self.phase = (self.phase - t.numel()) % self.down
        return y.numpy()


def phone_to_codec(rate_in: int = 8000, rate_out: int = 24000) -> Resampler:
    return Resampler(up=rate_out // rate_in)


def codec_to_phone(rate_in: int = 24000, rate_out: int = 8000) -> Resampler:
    return Resampler(down=rate_in // rate_out)
