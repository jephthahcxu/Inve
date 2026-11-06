"""Full-duplex streaming engine: one object per live call.

Every 80 ms:
    caller audio (24 kHz, 1920 samples) --codec.encode--> caller codes [K]
    OJEPH.step(previous frame)           --> inner text + OJEPH codes [K]
    OJEPH codes (un-delayed)             --codec.decode--> 1920 samples out

Listening and speaking happen in the same step, every step. When the caller
starts talking over OJEPH, the very next step already sees it — that is how
interruption works, with no VAD or turn-taking logic bolted on.

Latency budget (theoretical, before network):
    80 ms frame  +  80 ms acoustic delay  +  compute (< 80 ms required for real time)
"""

from __future__ import annotations

import contextlib
import time
from collections import deque

import numpy as np
import torch

from ojeph.model.ojeph import Ojeph
from ojeph.model.codec import Codec


class CallStream:
    def __init__(self, model: Ojeph, codec: Codec, speaker: int = 0, lang: int = 0,
                 temperature: float = 0.8, text_temperature: float = 0.7, top_k: int = 250,
                 device: str | None = None):
        self.model, self.codec = model, codec
        self.cfg = model.cfg
        self.K = self.cfg.codec.num_codebooks
        self.d = self.cfg.acoustic_delay
        self.device = device or next(model.parameters()).device
        self.temps = dict(temperature=temperature, text_temperature=text_temperature, top_k=top_k)

        self.state = model.init_state(1, torch.tensor([speaker]), torch.tensor([lang]),
                                      device=self.device)
        self.prev_frame: torch.Tensor | None = None
        self.user_hist: deque = deque(maxlen=self.d + 1)   # raw caller codes
        self.ojeph_sem_hist: deque = deque(maxlen=self.d + 1)  # OJEPH semantic codes
        self.inbuf = np.zeros(0, dtype=np.float32)
        self.text_log: list[int] = []
        self.step_ms: deque = deque(maxlen=200)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(codec.streaming(1))

    # ----------------------------------------------------------------- public
    def push(self, pcm24k: np.ndarray) -> np.ndarray:
        """Feed caller audio (any length). Returns OJEPH audio produced so far."""
        self.inbuf = np.concatenate([self.inbuf, pcm24k.astype(np.float32)])
        F = self.codec.frame_samples
        out = []
        while self.inbuf.size >= F:
            frame, self.inbuf = self.inbuf[:F], self.inbuf[F:]
            out.append(self._tick(frame))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def prime_text(self, token_ids: list[int], silence_codes: torch.Tensor | None = None):
        """Inject memory / context as inner-monologue text before the caller speaks.

        Runs forced steps: OJEPH 'thinks' the tokens while staying silent.
        `silence_codes` [K] = codec codes of silence (encode zeros once at boot).
        """
        sil = silence_codes if silence_codes is not None else \
            torch.full((self.K,), self.cfg.audio_pad, dtype=torch.long)
        sil = sil.to(self.device)
        for tok in token_ids:
            t = torch.tensor([tok], device=self.device)
            text, audio = self.model.step(self.state, self.prev_frame, force_text=t,
                                          force_audio=sil[None], **self.temps)
            self.prev_frame = self._frame(text, audio, sil[None])

    def close(self):
        self._stack.close()

    @property
    def mean_step_ms(self) -> float:
        return float(np.mean(self.step_ms)) if self.step_ms else 0.0

    # --------------------------------------------------------------- internal
    def _frame(self, text, ojeph_delayed, user_delayed) -> torch.Tensor:
        return torch.cat([text.view(1, 1), ojeph_delayed.view(1, -1),
                          user_delayed.view(1, -1)], dim=1)

    def _delay_user(self, codes: torch.Tensor) -> torch.Tensor:
        """Apply the acoustic delay to the caller stream online."""
        self.user_hist.append(codes)
        out = torch.empty_like(codes)
        out[0] = codes[0]
        if len(self.user_hist) > self.d:
            out[1:] = self.user_hist[-1 - self.d][1:]
        else:
            out[1:] = self.cfg.audio_pad
        return out

    @torch.no_grad()
    def _tick(self, frame24k: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        wav = torch.from_numpy(frame24k).view(1, 1, -1)
        user_codes = self.codec.encode(wav)[0, :, 0].to(self.device)      # [K]
        user_delayed = self._delay_user(user_codes)

        text, ojeph_delayed = self.model.step(self.state, self.prev_frame, **self.temps)
        self.prev_frame = self._frame(text, ojeph_delayed[0], user_delayed)
        self.text_log.append(int(text[0]))

        # un-delay OJEPH: semantic from d frames ago + acoustic from now
        self.ojeph_sem_hist.append(int(ojeph_delayed[0, 0]))
        if len(self.ojeph_sem_hist) <= self.d:
            out = np.zeros(self.codec.frame_samples, dtype=np.float32)
        else:
            full = ojeph_delayed[0].clone()
            full[0] = self.ojeph_sem_hist[-1 - self.d]
            full = full.clamp_max(self.cfg.codec.codebook_size - 1)
            out = self.codec.decode(full.view(1, -1, 1).cpu())[0, 0].numpy()
        self.step_ms.append((time.perf_counter() - t0) * 1000)
        return out
