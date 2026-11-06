"""What answers the phone.

* EchoBackend  — repeats the caller back with a short delay. Use it on day 1 to
                 prove the number, trunk and Asterisk work end to end.
* OjephBackend — the real thing: OJEPH full-duplex speech-to-speech.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from ojeph.audio import codec_to_phone, float_to_pcm16, pcm16_to_float, phone_to_codec

log = logging.getLogger("ojeph.backend")


class Session:
    def push(self, pcm8k: bytes) -> bytes:  # 16-bit LE mono 8 kHz in -> out
        raise NotImplementedError

    def dtmf(self, digit: str):
        pass

    def close(self):
        pass


class Backend:
    def open(self, caller_id: str) -> Session:
        raise NotImplementedError


# ----------------------------------------------------------------- echo test
class _EchoSession(Session):
    def __init__(self, delay_frames: int = 25):
        self.buf: list[bytes] = [b"\x00" * 320] * delay_frames

    def push(self, pcm8k):
        self.buf.append(pcm8k)
        return self.buf.pop(0)


class EchoBackend(Backend):
    def open(self, caller_id):
        return _EchoSession()


# --------------------------------------------------------------------- OJEPH
class _OjephSession(Session):
    def __init__(self, backend: "OjephBackend", caller_id: str):
        from ojeph.inference.streamer import CallStream

        self.b = backend
        self.cid = caller_id
        caller = backend.memory.get_or_create(caller_id) if backend.memory else None
        lang = backend.default_lang
        speaker = backend.default_speaker
        if caller and caller.speaker in ("0", "1"):
            speaker = int(caller.speaker)
        self.uid = caller.uid if caller else None
        self.call_id = backend.memory.start_call(self.uid) if caller else None

        with backend.lock:  # model weights are shared; state is per call
            # each call owns its codec: streaming codec state must not be shared
            self.stream = CallStream(backend.model, backend.codec_factory(),
                                     speaker=speaker, lang=lang)
            if backend.memory and backend.tokenizer:
                prompt = backend.memory.memory_prompt(caller_id)
                ids = backend.tokenizer.encode(prompt)
                self.stream.prime_text(ids, backend.silence_codes)
        self.up, self.down = phone_to_codec(), codec_to_phone()

    def push(self, pcm8k):
        x24 = self.up(pcm16_to_float(pcm8k))
        with self.b.lock:
            y24 = self.stream.push(x24)
        return float_to_pcm16(self.down(y24)) if y24.size else b""

    def close(self):
        self.stream.close()
        if self.b.memory and self.call_id is not None:
            transcript = ""
            if self.b.tokenizer:
                cfg = self.b.model.cfg
                ids = [t for t in self.stream.text_log if t < cfg.text_vocab]
                transcript = self.b.tokenizer.decode(ids)
            self.b.memory.end_call(self.call_id, summary=transcript[-400:], transcript=transcript)
        log.info("call closed, mean step %.1f ms", self.stream.mean_step_ms)


class OjephBackend(Backend):
    def __init__(self, model, codec_factory, memory=None, tokenizer=None,
                 default_lang: int = 0, default_speaker: int = 0):
        """codec_factory() -> a fresh Codec per call (e.g. a deepcopy of a loaded Mimi)."""
        import torch

        self.model, self.codec_factory = model, codec_factory
        self.memory, self.tokenizer = memory, tokenizer
        self.default_lang, self.default_speaker = default_lang, default_speaker
        self.lock = threading.Lock()
        codec = codec_factory()
        with codec.streaming(1):
            sil = codec.encode(torch.zeros(1, 1, codec.frame_samples))
        self.silence_codes = sil[0, :, 0]

    def open(self, caller_id):
        return _OjephSession(self, caller_id)
