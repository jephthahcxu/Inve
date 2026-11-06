"""Causal transformer with RoPE and a streaming KV cache.

Used twice in OJEPH:
  * temporal transformer — one step per 80 ms frame, long context, KV cached
  * depth transformer    — K tiny steps inside one frame, reset every frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).type_as(x) * self.weight


def rope_cache(seq_len: int, head_dim: int, device, base: float = 10_000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


@dataclass
class KVCache:
    """Fixed-size ring-free cache. Grows until max_len, then slides."""

    k: torch.Tensor  # [B, H, L, D]
    v: torch.Tensor
    length: int = 0
    offset: int = 0  # absolute position of the first cached item (for RoPE)

    @classmethod
    def empty(cls, batch: int, heads: int, max_len: int, head_dim: int, device, dtype):
        z = torch.zeros(batch, heads, max_len, head_dim, device=device, dtype=dtype)
        return cls(k=z, v=z.clone())

    def append(self, k: torch.Tensor, v: torch.Tensor):
        t = k.shape[2]
        cap = self.k.shape[2]
        if self.length + t > cap:  # slide window: drop oldest frames
            drop = self.length + t - cap
            self.k = torch.roll(self.k, -drop, dims=2)
            self.v = torch.roll(self.v, -drop, dims=2)
            self.length -= drop
            self.offset += drop
        self.k[:, :, self.length:self.length + t] = k
        self.v[:, :, self.length:self.length + t] = v
        self.length += t
        return self.k[:, :, :self.length], self.v[:, :, :self.length]


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(k, v)
            # causal mask: query i (absolute past+i) sees keys <= past+i
            L = k.shape[2]
            qpos = torch.arange(T, device=x.device)[:, None] + (L - T)
            kpos = torch.arange(L, device=x.device)[None, :]
            mask = kpos <= qpos
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: float):
        super().__init__()
        hidden = int(d * mult * 2 / 3)
        hidden = 64 * math.ceil(hidden / 64)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # SwiGLU


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, heads, dropout)
        self.ffn = FeedForward(d, ffn_mult)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.ffn(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, ffn_mult: float = 4.0,
                 dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.d, self.heads, self.max_len = d, heads, max_len
        self.blocks = nn.ModuleList(Block(d, heads, ffn_mult, dropout) for _ in range(layers))
        self.norm = RMSNorm(d)
        self._rope = None

    def _rope_for(self, start: int, T: int, device):
        need = start + T
        if self._rope is None or self._rope[0].shape[0] < need or self._rope[0].device != device:
            self._rope = rope_cache(max(need, self.max_len), self.d // self.heads, device)
        cos, sin = self._rope
        return cos[start:start + T], sin[start:start + T]

    def new_cache(self, batch: int, device, dtype, max_len: int | None = None):
        L = max_len or self.max_len
        return [KVCache.empty(batch, self.heads, L, self.d // self.heads, device, dtype)
                for _ in self.blocks]

    def forward(self, x, caches: list[KVCache] | None = None):
        T = x.shape[1]
        if caches is None:
            cos, sin = self._rope_for(0, T, x.device)
            for b in self.blocks:
                x = b(x, cos, sin)
        else:
            # absolute positions: cached keys keep their rotation when the
            # window slides, and RoPE only depends on relative distance
            start = caches[0].offset + caches[0].length
            cos, sin = self._rope_for(start, T, x.device)
            for b, c in zip(self.blocks, caches):
                x = b(x, cos, sin, c)
        return self.norm(x)
