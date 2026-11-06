# Roadmap

## Phase 1: the line is live (week 1)
- Buy a DIDHub/Kata Lagos number, set up Asterisk, run `serve_phone.py --echo`
- Greeting in Jephthah's voice plays on answer

## Phase 2: OJEPH v0 in Pidgin (weeks 2–8)
- 300–1,000 synthetic Pidgin conversations (Jephthah voice), 50–100 h real calls
- Train `small`, put it on the number, film real callers
- **Demo**: a live call from a ₦5,000 phone, interrupted mid-sentence, laughing.
  Pitch TechCabal, Techpoint, BBC Pidgin, Rest of World

## Phase 3: five languages (months 3–6)
- Yoruba, Hausa, Igbo, Igala data collection with native linguists
- Female voice, dialect conditioning, flywheel running nightly

## Phase 4: features
- **Incident line**: structured reports (kind, location, description) stored and
  pushed to a dashboard / partner agency (`MemoryStore.report_incident`)
- **Lost / directions**: landmark-based guidance. A GSM call doesn't carry GPS;
  exact location needs a telco partner (cell location) or the app/device
- **Health triage**: symptoms → urgency → nearest primary health centre (see SAFETY.md)
- Batched multi-call inference (100+ calls per box)

## Phase 5: OJEPH Guard (child safety device)
- Small device: GPS + 4G/LTE-M + mic/speaker + SOS button, running a distilled
  OJEPH. Children talk to it and parents check location and safety by voice.
- Candidate hardware: ESP32-S3 / Quectel 4G module for the device, with model
  inference on the server (the device streams audio like a phone call)
- Needs parental consent flows and child data protection; design that before any hardware
