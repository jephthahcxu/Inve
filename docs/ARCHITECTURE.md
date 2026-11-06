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
