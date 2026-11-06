# Data plan

A model is only as good as its data. Most of the real work on OJEPH is here.

## Three sources

1. **Pretraining audio** (teaches the sounds of each language): 5k–20k hours
   - NaijaVoices (~1,800 h Hausa/Igbo/Yoruba), AfriSpeech-200 (Nigerian-accented
     English), Mozilla Common Voice (Hausa, Yoruba, Igbo), BibleTTS (Hausa, Yoruba)
   - licensed radio / podcast archives (get written permission; don't scrape)
   - **Igala** has almost no public data: record it (churches, radio Ankpa/Idah,
     university linguistics departments, paid speakers)
2. **Synthetic conversations** (teaches conversation, laughing, interruptions)
   - `scripts/synth_dialogues.py`, Jephthah voice as OJEPH male, ElevenLabs
     timestamps give the text alignment for free
   - dialogue scripts must be written or checked by **native speakers** of each
     dialect. `dialogues/pidgin_seed.jsonl` is the format example
   - ElevenLabs quality for Yoruba, Igbo, Hausa tones is unverified, and Igala
     isn't supported. Use it mainly for Pidgin and English, and use real
     recordings for the others
   - check ElevenLabs' terms for your plan on using its output for training
3. **Real conversations** (removes the robotic feel): 200–1,000 h
   - pay pairs of speakers to talk on the phone, recorded on two channels
   - record **Jephthah** for 20–50 clean hours: that's the long-term male voice,
     owned outright, with no third-party terms
   - pick a female voice actor and record 20–50 h under contract

## Labels

Inner-monologue text is needed only for OJEPH's channel. Synthetic data has it
from ElevenLabs timestamps. For recorded data, a forced aligner or ASR is run
**once, offline**. At runtime OJEPH never transcribes.

## Tokenizer

SentencePiece unigram, 32k vocab, trained on text from all six languages, with
Yoruba/Igbo tone marks kept (`ẹ ọ ṣ à á è é`).

## Compute (training)

Rough maths for `small` (350M): ~2.4 GFLOP per training frame, and
2,000 h of audio = 90M frames. 10 passes ≈ 2.4e18 FLOPs, which is under a day
on one RTX 4090 at realistic utilisation. Compute is cheap at this size. Data is the bottleneck.

| GPU | Fits `small` training? | 2 weeks |
|---|---|---|
| RTX 4060 8 GB | tight (batch 1–2 + checkpointing) | own it |
| RTX 3060 12 GB / 4060 Ti 16 GB | yes | own it |
| RTX 3090 / 4090 24 GB | comfortable | rent about $0.3–0.7/h, so roughly $100–250 for 2 weeks |

These are estimates; measure tokens/sec on the first run and re-plan.
Training longer on the same data stops helping (the model memorises it). It
improves when new data arrives, so keep the GPU fed with new recordings
and flywheel calls.
Free compute to apply for: Google for Startups Africa, NVIDIA Inception,
Nigeria's national AI programmes (NITDA / N-ATLAS initiatives).
