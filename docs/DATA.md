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

## Compute (training, rented GPU)

| Stage | Est. GPU-hours (H100) | Est. cost @ $2–3/h |
|---|---|---|
| small (350M) pretrain + dialogue fine-tune | 800–2,000 | $2k–6k |
| base (1.1B) | 3,000–10,000 | $8k–30k |

These are estimates; measure tokens/sec on the first run and re-plan.
Free compute to apply for: Google for Startups Africa, NVIDIA Inception,
Nigeria's national AI programmes (NITDA / N-ATLAS initiatives).
