"""Turn rendered/recorded two-channel conversations into training shards.

Input : <stem>.npy  (2, N) float audio @24 kHz   ch0 = OJEPH, ch1 = caller
        <stem>.json {"lang", "speaker", "words": [[start_sec, word], ...]}
Output: shard-XXXX.pt (see ojeph/data/dataset.py)

    python scripts/build_shards.py out/synth data/shards --tokenizer tok/ojeph.model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ojeph.config import preset  # noqa: E402
from ojeph.data.dataset import align_text  # noqa: E402
from ojeph.data.languages import lang_id  # noqa: E402
from ojeph.model.codec import load_codec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tokenizer", required=True, help="SentencePiece .model")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--codec", default="mimi")
    ap.add_argument("--per-shard", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    cfg = preset(a.preset)
    codec = load_codec(a.codec, cfg.codec.num_codebooks, a.device)
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    buf, n = [], 0
    for meta_path in sorted(Path(a.src).glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        audio = torch.from_numpy(np.load(meta_path.with_suffix(".npy")).astype(np.float32))
        codes = codec.encode(audio[:, None])                       # [2, K, T]
        T = codes.shape[-1]
        words = [(s, sp.encode(" " + w)) for s, w in meta["words"]]
        text = align_text(words, T, cfg.codec.frame_rate, cfg.text_pad, cfg.text_epad)
        buf.append({"ojeph_codes": codes[0].cpu(), "user_codes": codes[1].cpu(), "text": text,
                    "speaker": int(meta["speaker"]), "lang": lang_id(meta["lang"]),
                    "source": meta.get("source", "recorded"), "quality": meta.get("quality", 1.0)})
        if len(buf) >= a.per_shard:
            torch.save(buf, dst / f"shard-{n:04d}.pt")
            buf, n = [], n + 1
    if buf:
        torch.save(buf, dst / f"shard-{n:04d}.pt")
    print("done")


if __name__ == "__main__":
    main()
