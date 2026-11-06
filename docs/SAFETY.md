# Safety rules (built into training data and the product)

These rules keep OJEPH on the air. One harmful answer covered in the press
undoes every good headline.

## Health
- OJEPH **triages and refers**. It doesn't prescribe prescription-only drugs or doses.
- Safe home care advice and danger signs (high fever in a child with shaking,
  bleeding in pregnancy, difficulty breathing) → "go to the health centre now".
- Training dialogues must follow this. `dialogues/pidgin_seed.jsonl` shows the tone.

## Emergencies
- Danger to life → tell the caller to get safe first, give the national
  emergency number, and record the incident.

## Data (Nigeria Data Protection Act 2023)
- Phone numbers stored only as HMAC UIDs (`OJEPH_UID_SECRET`).
- Calls are recorded for training **only with consent** (an audio prompt plus a
  DTMF or voice "yes"); `CallRecorder.add` drops calls without consent.
- Callers can ask OJEPH to forget them → delete their UID rows.

## Self-learning
- No live weight updates. Nightly fine-tune → per-language evaluation gate →
  promote only if nothing gets worse (`gate_and_promote`).

## Voice
- Jephthah's voice is used with his consent. Never clone a caller's voice.
