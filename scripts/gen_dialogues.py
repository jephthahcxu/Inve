"""Write training conversations at scale with Claude, following docs/CHARACTER.md.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python scripts/gen_dialogues.py --lang pidgin:lagos --persona friend --n 20 \
        --out dialogues/gen/pidgin_friend.jsonl

Output lines feed straight into scripts/synth_dialogues.py. Yoruba, Hausa, Igbo
and Igala output MUST be reviewed by native speakers before training; mark a
line reviewed by adding "reviewed": true.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ojeph.data.expressive import EMOTIONS, PERSONAS  # noqa: E402
from ojeph.data.languages import lang_id  # noqa: E402

MODEL = "claude-opus-5-5"
TAGS = "[laughs], [chuckles], [sighs], [whispers], [excited], [surprised], [curious]"

SYSTEM = """You write natural phone conversations between a caller and OJEPH, \
Africa's voice AI, for training a speech model. Follow the character bible exactly.

<character_bible>
{bible}
</character_bible>

Output rules:
- Write in the requested language and dialect the way real people speak on the phone, \
including slang and code-switching. No translations, no stage directions, no narration.
- OJEPH turns may start with or contain audio tags from this list only: {tags}. Callers may use them too.
- Every OJEPH turn has an "emotion" from: {emotions}.
- Use "gap_ms" (negative, -400 to -100) on a turn when that speaker interrupts or overlaps.
- 4 to 10 turns per conversation, short turns, realistic topics for the persona.
- Vary callers: age, gender, region, mood, background noise implied by speech ("abeg hold on").
- Each conversation is one JSON object on ONE line:
  {{"lang": ..., "ojeph": "male"|"female", "persona": ..., "turns": [{{"who": "caller"|"ojeph", "text": ..., "emotion": ..., "gap_ms": ...}}]}}
  ("emotion" only on ojeph turns, "gap_ms" only when overlapping)
- Put all conversations inside <dialogues></dialogues>, one per line, nothing else inside."""


def validate(obj: dict) -> dict:
    lang_id(obj["lang"])
    if obj.get("persona", "auto") not in PERSONAS:
        obj["persona"] = "auto"
    if obj.get("ojeph") not in ("male", "female"):
        obj["ojeph"] = "male"
    turns = obj["turns"]
    if not (2 <= len(turns) <= 14):
        raise ValueError("turn count")
    for t in turns:
        if t["who"] not in ("caller", "ojeph") or not t["text"].strip():
            raise ValueError("bad turn")
        if t["who"] == "ojeph" and t.get("emotion") not in EMOTIONS:
            t["emotion"] = "neutral"
        if "gap_ms" in t:
            t["gap_ms"] = max(-600, min(1500, int(t["gap_ms"])))
    return obj


def generate(client: anthropic.Anthropic, lang: str, persona: str, n: int, voice: str,
             topics: str) -> list[dict]:
    bible = (ROOT / "docs" / "CHARACTER.md").read_text(encoding="utf-8")
    system = SYSTEM.format(bible=bible, tags=TAGS, emotions=", ".join(EMOTIONS[2:]))
    ask = (f"Write {n} different conversations. Language/dialect: {lang}. Persona: {persona}. "
           f"OJEPH voice: {voice}. Include at least two with laughter, one with an interruption, "
           f"and one where the mood switches from playful to serious.")
    if topics:
        ask += f" Topics to cover: {topics}."
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
        output_config={"effort": "high"},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": ask}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"refused: {msg.stop_details}")
    text = "".join(b.text for b in msg.content if b.type == "text")
    body = text.split("<dialogues>", 1)[-1].split("</dialogues>", 1)[0]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(validate(json.loads(line)))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"  skipped a line: {e}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--persona", default="friend", choices=PERSONAS)
    ap.add_argument("--voice", default="male", choices=["male", "female"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--topics", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    client = anthropic.Anthropic()
    try:
        convs = generate(client, a.lang, a.persona, a.n, a.voice, a.topics)
    except anthropic.RateLimitError as e:
        sys.exit(f"rate limited, retry later: {e.message}")
    except anthropic.APIStatusError as e:
        sys.exit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("network error reaching the API")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(convs)} conversations to {out}")


if __name__ == "__main__":
    main()
