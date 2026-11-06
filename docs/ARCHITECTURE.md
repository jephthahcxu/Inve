# OJEPH architecture

## Native speech-to-speech, not a pipeline

A normal voice assistant: `speech → STT → text LLM → TTS → speech`. Three models,
each waiting for the previous one, 1–3 s of delay, flat robotic turn-taking, no
laughing, no interruptions.

OJEPH: `audio codes → ONE transformer → audio codes`. The only conversion is the
neural codec (waveform ↔ codes), the same role a phone's audio codec plays.

## Streams (per 80 ms frame)

| Stream | Size | Role |
|---|---|---|
| caller audio | K codebooks | always listened to, never predicted |
| OJEPH audio | K codebooks | predicted; codebook 0 = meaning, 1..K-1 = sound detail |
| inner monologue | 1 text token | predicted in the same pass, never spoken by a TTS; makes OJEPH smarter (Moshi's "inner monologue") |

Conditioning added every frame: **speaker** (male/female OJEPH) and
**language+dialect** (can switch mid-call: code-switching is normal in Nigeria).

## Emotion instinct and characters

* **Emotion head**: every frame, OJEPH predicts its own emotion (listening,
  warm, happy, laughing, playful, empathetic, calm, serious, urgent, surprised,
  whisper, thinking). The chosen emotion is fed into the depth transformer, so
  the voice for that frame is generated in that emotion. Next frame, it's fed
  back as input, so emotions carry over and change naturally.
* **Steering**: `CallStream.steer("calm")` forces an emotion (e.g. an
  emergency line that must never laugh). `None` hands control back to instinct.
* **Persona** (character mode) is a per-call condition: friend, teacher,
  health, emergency, legal, farmer, business, storyteller, guardian. It's
  randomly hidden 15% of the time in training, so the model also has a
  sensible "auto" character.
* **Labels**: per-turn emotions plus timed audio tags ([laughs], [whispers]…)
  from the synthetic data, so a laugh is learned at the exact frames it happens.
* **Active listening**: the renderer drops short OJEPH back-channels ("ehen",
  "mm-hm", a small laugh) into long caller turns, so OJEPH learns to react
  while you talk.
* **Phone realism**: caller audio gets degraded (8 kHz band-limit, crowd or
  generator noise, level jumps, dropped packets) so OJEPH can understand
  real GSM calls.
* Character bible: [CHARACTER.md](CHARACTER.md).

## Network

* **Temporal transformer** (`small`: 1024-d, 16 layers): one step per frame, KV
  cache, sliding window of 3000 frames (4 min), RoPE with absolute positions.
* **Depth transformer** (512-d, 4 layers): K tiny steps inside the frame:
  text → codebook 0 → codebook 1 → … (RQ-Transformer).
* **Acoustic delay = 1 frame**: decide *what* to say, then *how it sounds* 80 ms later.
* Loss: text CE + 100× semantic CE + acoustic CE (intelligibility first).

## Full duplex = interruptions for free

Caller audio enters every step, even while OJEPH is talking. Training data
contains overlaps (`gap_ms < 0` in the synthetic dialogues, plus real recorded
calls), so the model learns to stop, yield, or say "ehen, go on". There's no
voice-activity detector and no turn-taking state machine.

## Latency

There is no such thing as zero latency: sound has to be heard before anyone can
answer it. The aim is to be as fast as a human, and humans leave about 200 ms
between turns.

| Stage | Time |
|---|---|
| frame size | 80 ms |
| acoustic delay | 80 ms |
| compute per step | must be < 80 ms (measured: 56 ms, 350M model, 4-core CPU, fp32) |
| **model total** | **~160–200 ms** |
| phone network (GSM/SIP), both ways | +150–300 ms, outside our control |

A 2–3 s cascaded assistant gets beaten by about 10×.

## Model sizes

| Preset | Params | Use |
|---|---|---|
| tiny | ~25M | tests |
| small | ~350M | mini-PC demo |
| base | ~1.1B | quality target |

## Memory at call start

`MemoryStore.memory_prompt()` builds a short context ("Returning caller, name
Musa, asked about maize last time") which `CallStream.prime_text()` feeds as
forced inner-monologue tokens while OJEPH stays silent. The model "remembers"
before the caller says a word.
