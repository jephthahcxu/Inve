"""Run the phone line.

Day 1 (prove the number works, no model needed):
    python scripts/serve_phone.py --echo

With OJEPH:
    export OJEPH_UID_SECRET=$(openssl rand -hex 32)
    python scripts/serve_phone.py --ckpt ckpt/last.pt --tokenizer tok/ojeph.model
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ojeph.telephony.audiosocket import AudioSocketServer  # noqa: E402
from ojeph.telephony.backends import EchoBackend, OjephBackend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--echo", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--tokenizer")
    ap.add_argument("--codec", default="mimi")
    ap.add_argument("--db", default="ojeph_memory.sqlite")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--voice", choices=["male", "female"], default="male")
    ap.add_argument("--port", type=int, default=9092)
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if a.echo:
        backend = EchoBackend()
    else:
        import torch

        from ojeph.data.languages import lang_id
        from ojeph.memory.store import MemoryStore
        from ojeph.model.codec import load_codec
        from scripts.train import load

        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, _ = load(a.ckpt, device)
        model.to(device).eval()
        if device.startswith("cuda"):
            model.half()
        base_codec = load_codec(a.codec, cfg.codec.num_codebooks, device)
        tok = None
        if a.tokenizer:
            import sentencepiece as spm
            tok = spm.SentencePieceProcessor(model_file=a.tokenizer)
        memory = MemoryStore(a.db, os.environ.get("OJEPH_UID_SECRET", "").encode())
        backend = OjephBackend(model, lambda: copy.deepcopy(base_codec), memory, tok,
                               default_lang=lang_id(a.lang),
                               default_speaker=0 if a.voice == "male" else 1)

    asyncio.run(AudioSocketServer(backend, port=a.port, max_calls=a.max_calls).serve())


if __name__ == "__main__":
    main()
