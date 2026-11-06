"""Neural audio codec adapters.

OJEPH never converts speech to text or text to speech. It only converts
waveform <-> discrete audio codes with a neural codec, the same way a phone
converts sound <-> bits. The codec is frozen; OJEPH models the codes.

Default: Kyutai Mimi (24 kHz, 12.5 Hz frames, up to 32 codebooks, streaming,
CC-BY weights). `pip install moshi` to enable.
"""

from __future__ import annotations

import torch


class Codec:
    sample_rate: int
    frame_samples: int
    num_codebooks: int

    def encode(self, wav: torch.Tensor) -> torch.Tensor:  # [B,1,N] -> [B,K,T]
        raise NotImplementedError

    def decode(self, codes: torch.Tensor) -> torch.Tensor:  # [B,K,T] -> [B,1,N]
        raise NotImplementedError

    def streaming(self, batch: int):
        """Context manager for frame-by-frame streaming (80 ms per call)."""
        raise NotImplementedError


class MimiCodec(Codec):
    def __init__(self, num_codebooks: int = 8, device: str = "cpu"):
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.model = loaders.get_mimi(path, device=device)
        self.model.set_num_codebooks(num_codebooks)
        self.model.eval()
        self.sample_rate = int(self.model.sample_rate)
        self.frame_samples = int(self.model.sample_rate / self.model.frame_rate)
        self.num_codebooks = num_codebooks
        self.device = device

    @torch.no_grad()
    def encode(self, wav):
        return self.model.encode(wav.to(self.device))

    @torch.no_grad()
    def decode(self, codes):
        return self.model.decode(codes.to(self.device))

    def streaming(self, batch: int):
        return self.model.streaming(batch)


class DummyCodec(Codec):
    """Deterministic toy codec for tests and CPU dry runs (not for audio quality)."""

    def __init__(self, num_codebooks: int = 4, codebook_size: int = 256,
                 sample_rate: int = 24_000, frame_rate: float = 12.5):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate / frame_rate)
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    def encode(self, wav):
        B, _, N = wav.shape
        T = N // self.frame_samples
        fr = wav[:, 0, :T * self.frame_samples].reshape(B, T, self.frame_samples)
        feats = torch.stack([fr[..., k::self.num_codebooks].mean(-1)
                             for k in range(self.num_codebooks)], dim=1)
        q = ((feats.clamp(-1, 1) + 1) / 2 * (self.codebook_size - 1)).round().long()
        return q

    def decode(self, codes):
        B, K, T = codes.shape
        lvl = codes.float() / (self.codebook_size - 1) * 2 - 1
        return lvl.mean(1).repeat_interleave(self.frame_samples, dim=-1)[:, None]

    def streaming(self, batch: int):
        import contextlib
        return contextlib.nullcontext()


def load_codec(name: str, num_codebooks: int, device: str = "cpu", **kw) -> Codec:
    if name == "mimi":
        return MimiCodec(num_codebooks, device)
    if name == "dummy":
        return DummyCodec(num_codebooks, **kw)
    raise ValueError(name)
