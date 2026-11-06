# OJEPH

**Africa's native voice-to-voice AI. Call a number, from any phone, and talk.**

OJEPH listens and speaks at the same time, in one neural network: no speech-to-text,
no text-to-speech, no pipeline. Voice in, voice out. It can be interrupted, it
back-channels ("ehen", "mm-hm"), laughs, tells jokes, and remembers you the next
time you call from the same number.

Launch languages: **Yoruba, Hausa, Igbo, Igala, Nigerian Pidgin** (+ Nigerian
English), each with its dialects ([`ojeph/data/languages.py`](ojeph/data/languages.py)).
Two voices: **OJEPH male** (Jephthah's voice) and **OJEPH female**.

Built by **Jephthah Ameh**.

---

## How it works

```
 any phone ──► MTN/Airtel/Glo ──► Nigerian number (SIP trunk or GSM gateway)
     ──► Asterisk ──► AudioSocket ──► OJEPH server (mini PC)
                                         │
       every 80 ms:  caller audio ─► Mimi codec ─► caller codes ─┐
                                                                 ▼
                                  ┌──────── OJEPH transformer ────────┐
                                  │ listens to caller + its own voice │
                                  │ thinks (inner monologue)          │
                                  │ predicts its next voice codes     │
                                  └───────────────────────────────────┘
                                                                 │
       OJEPH voice ◄─ Mimi codec ◄─ OJEPH codes ◄────────────────┘
```

* [`ojeph/model/ojeph.py`](ojeph/model/ojeph.py) — the model: temporal transformer
  (one step per 80 ms) + depth transformer (audio codebooks), full duplex.
* [`ojeph/inference/streamer.py`](ojeph/inference/streamer.py) — live call engine.
* [`ojeph/telephony/`](ojeph/telephony) — phone number bridge (Asterisk AudioSocket).
* [`ojeph/memory/store.py`](ojeph/memory/store.py) — caller UID memory + incident reports.
* [`ojeph/learning/flywheel.py`](ojeph/learning/flywheel.py) — self-learning loop.
* [`scripts/synth_dialogues.py`](scripts/synth_dialogues.py) — generate training
  conversations in Jephthah's voice (ElevenLabs), with laughs and interruptions.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/PHONE.md](docs/PHONE.md) · [docs/DATA.md](docs/DATA.md) ·
[docs/HARDWARE.md](docs/HARDWARE.md) · [docs/ROADMAP.md](docs/ROADMAP.md) ·
[docs/SAFETY.md](docs/SAFETY.md)

## Status

| Part | State |
|---|---|
| Model architecture (full duplex, streaming, KV cache) | done, tested |
| Real-time check: 350M model, 1 step | **56 ms on a 4-core CPU** (budget 80 ms) |
| Phone bridge (AudioSocket + caller ID + pacing) | done, tested with echo backend |
| Caller memory (UID, summaries, incidents) | done, tested |
| Synthetic conversation generator (Jephthah voice) | done, verified against ElevenLabs |
| Training / fine-tuning / flywheel scripts | done |
| **Trained weights** | **not yet: needs the data + a rented GPU (see docs/DATA.md)** |

## Quick start

```bash
pip install -e ".[dev]"          # add ".[mimi]" for the real codec
pytest -q                        # 12 tests

# 1. prove the phone number works today (caller hears themselves back)
python scripts/serve_phone.py --echo

# 2. make training conversations in Jephthah's voice
export ELEVENLABS_API_KEY=...  OJEPH_VOICE_MALE=<voice id>
python scripts/synth_dialogues.py dialogues/pidgin_seed.jsonl out/synth

# 3. tokenise -> shards -> train (GPU)
python scripts/build_shards.py out/synth data/shards --tokenizer tok/ojeph.model
python scripts/train.py --preset small --data data/shards --out ckpt

# 4. serve OJEPH on the number
export OJEPH_UID_SECRET=$(openssl rand -hex 32)
python scripts/serve_phone.py --ckpt ckpt/last.pt --tokenizer tok/ojeph.model
```

Secrets (`ELEVENLABS_API_KEY`, SIP passwords, `OJEPH_UID_SECRET`) live in
environment variables or a local `.env`, never in git.
