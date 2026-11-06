"""Self-learning flywheel: OJEPH gets better the more people call it.

Live weights are NEVER updated mid-call. Online learning from raw calls lets
anyone poison the model (insults, scams, wrong medical advice) and makes it
drift. Instead:

    1. record   — consented calls are saved as codec codes (both channels)
    2. score    — each call gets a quality score (caller rating via DTMF at the
                  end, human review, call length / hang-up signals)
    3. finetune — nightly: new good calls + replay of base data (stops forgetting)
    4. gate     — the new checkpoint must beat the current one on a fixed
                  held-out set per language, or it is thrown away
    5. promote  — atomic swap of the served checkpoint

So it learns every day from real Nigerians, in their real accents and slang,
without a single bad call being able to break it.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch


class CallRecorder:
    def __init__(self, out_dir: str | Path, shard_size: int = 200):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buf: list[dict] = []

    def add(self, ojeph_codes: torch.Tensor, user_codes: torch.Tensor, text: torch.Tensor,
            speaker: int, lang: int, consent: bool, quality: float | None = None):
        if not consent:
            return  # no consent, nothing kept
        self.buf.append({
            "ojeph_codes": ojeph_codes.cpu(), "user_codes": user_codes.cpu(),
            "text": text.cpu(), "speaker": speaker, "lang": lang,
            "source": "flywheel", "quality": -1.0 if quality is None else quality,
            "time": time.time(),
        })
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return None
        path = self.dir / f"flywheel-{int(time.time())}.pt"
        torch.save(self.buf, path)
        self.buf = []
        return path


def gate_and_promote(candidate: Path, current: Path, served: Path,
                     eval_fn, margin: float = 0.0, log_path: Path | None = None) -> bool:
    """eval_fn(ckpt_path) -> {lang: loss}. Promote only if no language gets worse."""
    new, old = eval_fn(candidate), eval_fn(current)
    worse = {k: (old[k], new[k]) for k in old if new.get(k, float("inf")) > old[k] + margin}
    ok = not worse
    if ok:
        tmp = served.with_suffix(".tmp")
        shutil.copy(candidate, tmp)
        tmp.replace(served)  # atomic on POSIX
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps({"time": time.time(), "candidate": str(candidate),
                                "promoted": ok, "old": old, "new": new,
                                "regressions": worse}) + "\n")
    return ok
