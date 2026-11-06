# Getting a phone number anyone can call

Prices researched November 2026. Confirm with each provider before paying.

## Options, cheapest first

| Provider | What | Price | Notes |
|---|---|---|---|
| **DIDHub** | Lagos / city geographic number (+234 1 …) | **from $3.50/month** | self-serve, 2–5 days activation |
| **Africa's Talking** | Nigerian voice number | ~₦7,000/month + VAT (₦10,000 for MTN) | figures from a competitor's comparison page, verify |
| **Kata Telecom** | local DID, forwards to SIP | from $19.30/month + $5 setup | same-day activation |
| Telnyx | Nigeria local number | from $25/month | big platform, good SIP |
| Plivo (reference) | inbound on local number | $0.04/min | no Nigerian toll-free |
| **GSM gateway + SIMs** | your own MTN/Airtel SIM | gateway ₦80k–250k one-off, SIM ~₦1k | fully offline from the internet |

**Recommended for the demo: DIDHub (or Kata) Lagos number → SIP → Asterisk on the mini PC.**
It's cheap, it's legal, anyone in Nigeria can call it, and it can be live this week.

Use the GSM gateway when there's no internet at the site. The NCC is strict about
SIM boxes, so keep the gateway for receiving calls only and move to a licensed
number for the public launch.

For the public launch, get a toll-free 0800 number through a telco (MTN/Airtel/Glo)
so callers pay nothing, then a short code. That's the path to nationwide scale.

## Wiring

1. Buy the number; get SIP host, username, password from the provider.
2. Install Asterisk 18+ on the mini PC (`apt install asterisk`), copy
   `deploy/asterisk/pjsip.conf` and `extensions.conf`, fill in the CAPS values.
3. Forward UDP 5060 + RTP 10000–20000 on the router to the mini PC (or ask the
   provider for IP-based auth).
4. `python scripts/serve_phone.py --echo`, call the number, hear yourself →
   the line works.
5. Swap `--echo` for `--ckpt ...` once OJEPH is trained.

The greeting in Jephthah's voice is ready: `assets/ojeph_greeting.wav`
(8 kHz, for Asterisk) and `assets/ojeph_greeting.mp3`.

## Sources
- DIDHub Nigeria: https://didhub.io/buy/nigeria
- Kata Telecom Nigeria: https://katatelecom.com/services/nigeria/
- Telnyx Nigeria: https://telnyx.com/phone-numbers/nigeria
- Africa's Talking figures (via Signals): https://signals.pressone.africa/compare/africas-talking
- Plivo Nigeria: https://www.plivo.com/sip-trunking/coverage/ng/
