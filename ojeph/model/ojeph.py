"""OJEPH: full-duplex native speech-to-speech model.

Token layout per frame t (S = 1 + 2K streams):
    stream 0          : inner-monologue text token
    streams 1..K      : OJEPH audio codebooks (k=0 semantic, k>=1 acoustic)
    streams K+1..2K   : caller audio codebooks

Training / inference recurrence (Moshi-style RQ-Transformer):
    h_t        = Temporal( sum_s Emb_s(tok[s, t-1]) + spk + lang )
    text_t     ~ TextHead(h_t)
    a_t[0]     ~ Depth_0(h_t, text_t)
    a_t[k]     ~ Depth_k(h_t, a_t[k-1])        k = 1..K-1

The caller stream is never predicted, only listened to: every 80 ms the
caller's real codes are fed in. That is what makes OJEPH interruptible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ojeph.config import OjephConfig
from ojeph.model.transformer import Transformer


class Ojeph(nn.Module):
    def __init__(self, cfg: OjephConfig):
        super().__init__()
        self.cfg = cfg
        K, D, Dd = cfg.codec.num_codebooks, cfg.d_model, cfg.depth_d_model
        self.K = K

        # temporal input embeddings
        self.text_emb = nn.Embedding(cfg.text_vocab_total, D)
        self.ojeph_emb = nn.ModuleList(nn.Embedding(cfg.audio_vocab_total, D) for _ in range(K))
        self.user_emb = nn.ModuleList(nn.Embedding(cfg.audio_vocab_total, D) for _ in range(K))
        self.speaker_emb = nn.Embedding(cfg.num_speakers, D)
        self.lang_emb = nn.Embedding(cfg.num_languages, D)
        self.persona_emb = nn.Embedding(cfg.num_personas, D)
        self.emotion_emb = nn.Embedding(cfg.num_emotions, D)   # previous frame's emotion
        self.bos = nn.Parameter(torch.zeros(D))

        self.temporal = Transformer(D, cfg.n_layers, cfg.n_heads, cfg.ffn_mult,
                                    cfg.dropout, cfg.max_context)
        self.text_head = nn.Linear(D, cfg.text_vocab_total, bias=False)
        self.emotion_head = nn.Linear(D, cfg.num_emotions, bias=False)  # OJEPH's instinct

        # depth transformer (one position per OJEPH codebook)
        self.depth_in = nn.ModuleList(nn.Linear(D, Dd, bias=False) for _ in range(K))
        self.depth_text_emb = nn.Embedding(cfg.text_vocab_total, Dd)
        self.depth_emotion_emb = nn.Embedding(cfg.num_emotions, Dd)  # voice follows the emotion
        self.depth_audio_emb = nn.ModuleList(
            nn.Embedding(cfg.audio_vocab_total, Dd) for _ in range(K - 1))
        self.depth = Transformer(Dd, cfg.depth_layers, cfg.depth_heads, cfg.ffn_mult,
                                 cfg.dropout, max_len=K)
        self.audio_heads = nn.ModuleList(
            nn.Linear(Dd, cfg.codec.codebook_size, bias=False) for _ in range(K))

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------ utils
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _frame_embed(self, tok: torch.Tensor) -> torch.Tensor:
        """tok: [B, S, T] -> summed embedding [B, T, D]."""
        K = self.K
        x = self.text_emb(tok[:, 0])
        for k in range(K):
            x = x + self.ojeph_emb[k](tok[:, 1 + k]) + self.user_emb[k](tok[:, 1 + K + k])
        return x

    def _cond(self, speaker: torch.Tensor, lang: torch.Tensor,
              persona: torch.Tensor | None = None) -> torch.Tensor:
        # speaker: [B]; lang: [B] or [B, T] (language can switch mid-call); persona: [B]
        c = self.speaker_emb(speaker)[:, None]
        if persona is not None:
            c = c + self.persona_emb(persona)[:, None]
        l = self.lang_emb(lang)
        return c + (l[:, None] if lang.dim() == 1 else l)

    def _depth_inputs(self, h: torch.Tensor, text: torch.Tensor, audio: torch.Tensor,
                      emotion: torch.Tensor):
        """h: [N, D], text/emotion: [N], audio: [N, K] (teacher forced) -> [N, K, Dd]."""
        cols = [self.depth_in[0](h) + self.depth_text_emb(text) + self.depth_emotion_emb(emotion)]
        for k in range(1, self.K):
            cols.append(self.depth_in[k](h) + self.depth_audio_emb[k - 1](audio[:, k - 1]))
        return torch.stack(cols, dim=1)

    # --------------------------------------------------------------- training
    def forward(self, tokens: torch.Tensor, speaker: torch.Tensor, lang: torch.Tensor,
                emotion: torch.Tensor | None = None, persona: torch.Tensor | None = None):
        """tokens: [B, S, T] (already delayed); emotion: [B, T] labels (0 = unknown,
        no loss); persona: [B]. Returns dict of losses + logits."""
        cfg, K = self.cfg, self.K
        B, S, T = tokens.shape
        assert S == 1 + 2 * K, f"expected {1 + 2 * K} streams, got {S}"
        if emotion is None:
            emotion = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        if persona is not None and self.training and cfg.cond_dropout > 0:
            drop = torch.rand(B, device=persona.device) < cfg.cond_dropout
            persona = persona.masked_fill(drop, 0)  # learn the "auto" persona too

        # input at t is frame t-1 (tokens + emotion); frame -1 is a learned BOS
        x = self._frame_embed(tokens) + self.emotion_emb(emotion)
        x = torch.cat([self.bos.expand(B, 1, -1), x[:, :-1]], dim=1)
        h = self.temporal(x + self._cond(speaker, lang, persona))   # [B, T, D]

        text_logits = self.text_head(h)                           # [B, T, Vt]
        emotion_logits = self.emotion_head(h)                     # [B, T, E]

        hf = h.reshape(B * T, -1)
        text_tgt = tokens[:, 0].reshape(B * T)
        audio_tgt = tokens[:, 1:1 + K].permute(0, 2, 1).reshape(B * T, K)
        dz = self.depth(self._depth_inputs(hf, text_tgt, audio_tgt,
                                           emotion.reshape(B * T)))      # [N, K, Dd]
        audio_logits = torch.stack(
            [self.audio_heads[k](dz[:, k]) for k in range(K)], dim=1)  # [N, K, V]

        text_loss = F.cross_entropy(text_logits.reshape(B * T, -1), text_tgt)
        # padded (delayed) audio positions are ignored
        ign = cfg.audio_pad
        sem_loss = F.cross_entropy(audio_logits[:, 0], audio_tgt[:, 0], ignore_index=ign)
        if K > 1:
            ac_loss = F.cross_entropy(audio_logits[:, 1:].reshape(-1, audio_logits.shape[-1]),
                                      audio_tgt[:, 1:].reshape(-1), ignore_index=ign)
        else:
            ac_loss = torch.zeros((), device=tokens.device)
        if (emotion > 0).any():
            emo_loss = F.cross_entropy(emotion_logits.reshape(B * T, -1), emotion.reshape(B * T),
                                       ignore_index=0)
        else:
            emo_loss = torch.zeros((), device=tokens.device)
        w = (cfg.text_loss_weight + cfg.semantic_loss_weight
             + cfg.acoustic_loss_weight * (K - 1) + cfg.emotion_loss_weight)
        loss = (cfg.text_loss_weight * text_loss + cfg.semantic_loss_weight * sem_loss
                + cfg.acoustic_loss_weight * (K - 1) * ac_loss
                + cfg.emotion_loss_weight * emo_loss) / w
        return {"loss": loss, "text_loss": text_loss, "semantic_loss": sem_loss,
                "acoustic_loss": ac_loss, "emotion_loss": emo_loss,
                "text_logits": text_logits, "emotion_logits": emotion_logits,
                "audio_logits": audio_logits.view(B, T, K, -1)}

    # -------------------------------------------------------------- streaming
    @torch.no_grad()
    def init_state(self, batch: int, speaker: torch.Tensor, lang: torch.Tensor,
                   device=None, dtype=None, persona: torch.Tensor | None = None):
        p = next(self.parameters())
        device, dtype = device or p.device, dtype or p.dtype
        return {
            "caches": self.temporal.new_cache(batch, device, dtype, self.cfg.max_context),
            "speaker": speaker.to(device), "lang": lang.to(device),
            "persona": persona.to(device) if persona is not None else None,
            "emotion": torch.zeros(batch, dtype=torch.long, device=device),  # last frame's
        }

    @torch.no_grad()
    def step(self, state: dict, prev_frame: torch.Tensor | None,
             temperature: float = 0.8, top_k: int = 250, text_temperature: float = 0.7,
             force_text: torch.Tensor | None = None, force_audio: torch.Tensor | None = None,
             force_emotion: torch.Tensor | None = None, emotion_temperature: float = 0.6):
        """One 80 ms step.

        prev_frame: [B, S] tokens of frame t-1 (OJEPH's own last output + the
                    caller's real codes for t-1), or None at call start.
        Returns (text_t [B], audio_t [B, K]) for OJEPH at frame t; the emotion
        OJEPH chose for this frame is left in state["emotion"].
        force_emotion [B] steers it (e.g. make it laugh, or keep it calm).
        """
        B = state["speaker"].shape[0]
        if prev_frame is None:
            x = self.bos.expand(B, 1, -1)
        else:
            x = self._frame_embed(prev_frame[:, :, None]) + self.emotion_emb(state["emotion"])[:, None]
        x = x + self._cond(state["speaker"], state["lang"], state.get("persona"))
        h = self.temporal(x, state["caches"])[:, -1]               # [B, D]

        if force_emotion is not None:
            emotion = force_emotion
        else:
            el = self.emotion_head(h)
            el[:, 0] = float("-inf")                                # never pick "auto"
            emotion = sample(el, emotion_temperature, 0)
        state["emotion"] = emotion

        text = force_text if force_text is not None else \
            sample(self.text_head(h), text_temperature, top_k)

        # depth loop: K tiny steps, own short cache
        dcaches = self.depth.new_cache(B, h.device, h.dtype, max_len=self.K)
        audio = torch.zeros(B, self.K, dtype=torch.long, device=h.device)
        prev = None
        for k in range(self.K):
            if k == 0:
                inp = self.depth_in[0](h) + self.depth_text_emb(text) + self.depth_emotion_emb(emotion)
            else:
                inp = self.depth_in[k](h) + self.depth_audio_emb[k - 1](prev)
            z = self.depth(inp[:, None], dcaches)[:, -1]
            if force_audio is not None:
                tok = force_audio[:, k]
            else:
                tok = sample(self.audio_heads[k](z), temperature, top_k)
            audio[:, k] = tok
            prev = tok
        return text, audio


def sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k and top_k < logits.shape[-1]:
        v, _ = logits.topk(top_k, dim=-1)
        logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))
    return torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)
