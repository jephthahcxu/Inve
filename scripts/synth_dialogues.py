"""Render scripted conversations into two-channel training audio with ElevenLabs.

Channel 0 = OJEPH (Jephthah's voice, or the female voice), channel 1 = caller.
ElevenLabs returns character timestamps, so the inner-monologue text is aligned
to the audio for free — no speech recognition needed.

Turns may overlap (negative gap) to teach OJEPH interruptions, and may carry
v3 audio tags such as [laughs], [whispers], [sighs] for natural laughter.

    export ELEVENLABS_API_KEY=...   OJEPH_VOICE_MALE=...   CALLER_VOICES=id1,id2
    python scripts/synth_dialogues.py dialogues/pidgin_seed.jsonl out/synth/

Input JSONL, one conversation per line:
    {"lang": "pidgin:lagos", "ojeph": "male", "persona": "friend",
     "turns": [{"who": "caller", "text": "..."},
               {"who": "ojeph", "text": "[laughs] ...", "emotion": "playful", "gap_ms": -300}]}

OJEPH also back-channels automatically ("Ehen", "Mm-hm", a small laugh) during
long caller turns, so it learns to listen actively, the way people do on the phone.

Check ElevenLabs' terms of service for your plan before using its output to
train a model; the clean long-term source is your own studio recordings.
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps?output_format=pcm_24000"
SR = 24_000


def tts(text: str, voice: str, model: str, key: str, retries: int = 4) -> tuple[np.ndarray, list]:
    body = json.dumps({"text": text, "model_id": model,
                       "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}}).encode()
    req = urllib.request.Request(API.format(voice=voice), data=body, method="POST",
                                 headers={"xi-api-key": key, "Content-Type": "application/json"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            pcm = np.frombuffer(base64.b64decode(d["audio_base64"]), dtype="<i2")
            al = d.get("alignment") or {}
            chars = list(zip(al.get("characters", []), al.get("character_start_times_seconds", [])))
            return pcm.astype(np.float32) / 32768.0, chars
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry {i + 1}: {e}", file=sys.stderr)
            time.sleep(2 ** (i + 1))
    raise RuntimeError("unreachable")


def words_from_chars(chars: list, offset: float):
    """Group character timestamps into (start_sec, word) and pull out audio-tag
    events such as [laughs] as (time_sec, tag)."""
    words, events, cur, start, in_tag, tag, tag_t = [], [], "", None, False, "", 0.0
    for ch, t in chars:
        if ch == "[":
            in_tag, tag, tag_t = True, "", t
            continue
        if ch == "]":
            in_tag = False
            events.append((tag_t + offset, tag.strip()))
            continue
        if in_tag:
            tag += ch
            continue
        if ch.isspace():
            if cur:
                words.append((start + offset, cur))
            cur, start = "", None
        else:
            if start is None:
                start = t
            cur += ch
    if cur:
        words.append((start + offset, cur))
    return words, events


BACKCHANNELS = ["Ehen.", "Mm-hm.", "I dey hear you.", "Okay.", "Hmm.", "Oya.", "[chuckles] Ehen."]
_bc_cache: dict[str, tuple[np.ndarray, list]] = {}


def backchannel(voice: str, model: str, key: str, rng: random.Random):
    text = rng.choice(BACKCHANNELS)
    k = f"{voice}|{text}"
    if k not in _bc_cache:
        _bc_cache[k] = tts(text, voice, model, key)
    return text, _bc_cache[k]


def _place(ch, who, audio, start):
    s = int(start * SR)
    need = s + audio.size
    for c in range(2):
        if ch[c].size < need:
            ch[c] = np.pad(ch[c], (0, need - ch[c].size))
    ch[who][s:s + audio.size] += audio


def render(conv: dict, voices: dict, key: str, model: str, rng: random.Random,
           backchannel_prob: float = 0.6):
    """Returns stereo audio, OJEPH words, OJEPH turns (with emotion) and tag events."""
    ch = [np.zeros(0, np.float32), np.zeros(0, np.float32)]
    words, turns, events = [], [], []
    cursor = 0.0
    for turn in conv["turns"]:
        who = 0 if turn["who"] == "ojeph" else 1
        voice = voices["ojeph"] if who == 0 else voices["caller"]
        audio, chars = tts(turn["text"], voice, model, key)
        start = max(0.0, cursor + turn.get("gap_ms", rng.randint(150, 600)) / 1000)
        _place(ch, who, audio, start)
        dur = audio.size / SR
        if who == 0:
            w, ev = words_from_chars(chars, start)
            words += w
            events += ev
            turns.append({"start": start, "end": start + dur,
                          "emotion": turn.get("emotion", "neutral")})
        elif dur > 3.5 and rng.random() < backchannel_prob:
            # OJEPH listens actively: a short "ehen" while the caller is talking
            text, (bc, bchars) = backchannel(voices["ojeph"], model, key, rng)
            t = start + rng.uniform(1.5, dur - 1.0)
            _place(ch, 0, bc * 0.8, t)
            w, ev = words_from_chars(bchars, t)
            words += w
            events += ev
            turns.append({"start": t, "end": t + bc.size / SR, "emotion": "listening"})
        cursor = start + dur
    n = max(c.size for c in ch)
    stereo = np.stack([np.pad(c, (0, n - c.size)) for c in ch])
    words.sort()
    return stereo, words, turns, events


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ["ELEVENLABS_API_KEY"]
    model = os.environ.get("ELEVENLABS_MODEL", "eleven_v3")
    male = os.environ["OJEPH_VOICE_MALE"]
    female = os.environ.get("OJEPH_VOICE_FEMALE", male)
    callers = os.environ.get("CALLER_VOICES", "").split(",")
    callers = [c for c in callers if c] or [female]

    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        conv = json.loads(line)
        voices = {"ojeph": female if conv.get("ojeph") == "female" else male,
                  "caller": random.choice(callers)}
        stem = out / f"{src.stem}-{i:05d}"
        if stem.with_suffix(".json").exists():
            continue
        print(f"[{i}] {conv['lang']} ({len(conv['turns'])} turns)")
        stereo, words, turns, events = render(conv, voices, key, model, random.Random(i))
        np.save(stem.with_suffix(".npy"), stereo.astype(np.float16))
        stem.with_suffix(".json").write_text(json.dumps({
            "lang": conv["lang"], "speaker": 1 if conv.get("ojeph") == "female" else 0,
            "persona": conv.get("persona", "auto"), "words": words, "turns": turns,
            "events": events, "sample_rate": SR, "source": "synthetic"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
