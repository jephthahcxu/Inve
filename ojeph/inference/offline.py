"""Offline generation: let OJEPH answer a recorded caller channel.

Used by the lab to make a listening clip from every checkpoint. It runs the
same step() as a live call, with the caller's codes fed in frame by frame.
"""

from __future__ import annotations

import torch

from ojeph.model.delay import apply_delay, undo_delay
from ojeph.model.ojeph import Ojeph


@torch.no_grad()
def respond(model: Ojeph, user_codes: torch.Tensor, speaker: int, lang: int,
            temperature: float = 0.8, top_k: int = 100,
            prompt: tuple[torch.Tensor, torch.Tensor] | None = None,
            persona: int = 0, return_emotions: bool = False):
    """user_codes: [K, T] caller codes -> OJEPH codes [K, T - delay]
    (plus the emotion OJEPH chose each frame if return_emotions).

    prompt: optional (text [P], ojeph_codes [K, P]) teacher-forced prefix, e.g.
            the first seconds of a real OJEPH turn, so generation continues it.
    """
    cfg = model.cfg
    d, pad = cfg.acoustic_delay, cfg.audio_pad
    K, T = user_codes.shape
    dev = next(model.parameters()).device
    user_d = apply_delay(user_codes, d, pad).to(dev)

    st = model.init_state(1, torch.tensor([speaker]), torch.tensor([lang]), device=dev,
                          persona=torch.tensor([persona]))
    emotions = []
    P = 0
    if prompt is not None:
        ptext, pcodes = prompt
        pcodes_d = apply_delay(pcodes, d, pad).to(dev)
        P = pcodes.shape[-1]
    out = torch.full((K, T), pad, dtype=torch.long, device=dev)
    prev = None
    for t in range(T):
        if t < P:
            text, audio = model.step(st, prev, force_text=ptext[t:t + 1].to(dev),
                                     force_audio=pcodes_d[:, t][None])
        else:
            text, audio = model.step(st, prev, temperature=temperature, top_k=top_k)
        out[:, t] = audio[0]
        emotions.append(int(st["emotion"][0]))
        prev = torch.cat([text.view(1, 1), audio.view(1, -1), user_d[:, t].view(1, -1)], 1)
    full = undo_delay(out, d).clamp_max(cfg.codec.codebook_size - 1).cpu()
    return (full, emotions) if return_emotions else full
