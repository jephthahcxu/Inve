"""OJEPH model configuration.

OJEPH is a native speech-to-speech model: audio tokens in, audio tokens out,
in ONE network. There is no STT -> LLM -> TTS pipeline.

Each 80 ms frame (12.5 Hz, Mimi codec) carries three parallel streams:
  * user audio   : K codebooks of what the caller is saying (always listening)
  * ojeph audio  : K codebooks of what OJEPH is saying (its own voice)
  * inner text   : 1 token of OJEPH's "inner monologue" (predicted in the same
                   forward pass; never spoken through a TTS, it only helps the
                   model reason — the Moshi trick)

Because both audio streams run all the time, OJEPH is full duplex: it can be
interrupted, it can back-channel ("mm-hm", "ehen"), laugh and talk over you.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class CodecConfig:
    sample_rate: int = 24_000      # Mimi native rate
    frame_rate: float = 12.5       # 80 ms frames
    num_codebooks: int = 8         # K
    codebook_size: int = 2048

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate / self.frame_rate)  # 1920


@dataclass
class OjephConfig:
    name: str = "ojeph-small"
    codec: CodecConfig = field(default_factory=CodecConfig)

    # inner-monologue text vocabulary (SentencePiece trained on Yoruba, Hausa,
    # Igbo, Igala, Pidgin, English). Two extra specials are reserved at the end.
    text_vocab: int = 32_000

    # temporal transformer: one step per 80 ms frame
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    ffn_mult: float = 4.0
    max_context: int = 3000        # frames = 4 minutes of conversation

    # depth transformer: predicts the K codebooks inside one frame
    depth_d_model: int = 512
    depth_layers: int = 4
    depth_heads: int = 8

    # acoustic codebooks (k >= 1) lag the semantic codebook (k = 0) by this many
    # frames. 1 frame = 80 ms. This is what gives ~160 ms theoretical latency.
    acoustic_delay: int = 1

    # conditioning
    num_speakers: int = 2          # 0 = OJEPH male (Jephthah voice), 1 = OJEPH female
    num_languages: int = 64        # language+dialect ids, see ojeph/data/languages.py

    # loss weights (semantic codebook matters most for intelligibility)
    text_loss_weight: float = 1.0
    semantic_loss_weight: float = 100.0
    acoustic_loss_weight: float = 1.0

    dropout: float = 0.0

    # ---- derived token ids -------------------------------------------------
    @property
    def text_pad(self) -> int:           # "no word this frame"
        return self.text_vocab

    @property
    def text_epad(self) -> int:          # "a word starts next frame"
        return self.text_vocab + 1

    @property
    def text_vocab_total(self) -> int:
        return self.text_vocab + 2

    @property
    def audio_pad(self) -> int:          # used for delayed / missing codes
        return self.codec.codebook_size

    @property
    def audio_vocab_total(self) -> int:
        return self.codec.codebook_size + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OjephConfig":
        d = dict(d)
        codec = CodecConfig(**d.pop("codec", {}))
        return cls(codec=codec, **d)


def preset(name: str) -> OjephConfig:
    """Model sizes. Parameter counts are approximate.

    tiny  ~  25M  -> unit tests
    mini  ~  45M  -> CPU live lab (real Mimi codes)
    small ~ 300M  -> mini-PC demo target (fits 8 GB GPU in bf16 with KV cache)
    base  ~ 1.1B  -> quality target once data + compute are available
    """
    if name == "tiny":
        return OjephConfig(
            name="ojeph-tiny", text_vocab=1000, d_model=256, n_layers=4, n_heads=4,
            depth_d_model=128, depth_layers=2, depth_heads=4, max_context=512,
            codec=CodecConfig(num_codebooks=4, codebook_size=256),
        )
    if name == "mini":  # ~45M, real Mimi codes, trains on a CPU for the live lab
        return OjephConfig(
            name="ojeph-mini", text_vocab=800, d_model=512, n_layers=8, n_heads=8,
            depth_d_model=256, depth_layers=2, depth_heads=4, max_context=1500,
        )
    if name == "small":
        return OjephConfig()
    if name == "base":
        return OjephConfig(
            name="ojeph-base", d_model=2048, n_layers=22, n_heads=16,
            depth_d_model=1024, depth_layers=6, depth_heads=16,
        )
    raise ValueError(f"unknown preset {name!r}")
