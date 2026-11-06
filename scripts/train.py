"""Train or fine-tune OJEPH.

    python scripts/train.py --preset small --data data/shards --out ckpt/
    python scripts/train.py --resume ckpt/last.pt --data data/flywheel --lr 2e-5 --steps 2000

Runs on a rented GPU (training). The mini PC only runs inference.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ojeph.config import OjephConfig, preset  # noqa: E402
from ojeph.data.dataset import ConversationShards  # noqa: E402
from ojeph.model.ojeph import Ojeph  # noqa: E402


def save(path: Path, model, opt, step, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None,
                "step": step, "config": cfg.to_dict()}, tmp)
    tmp.replace(path)


def load(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = OjephConfig.from_dict(ck["config"])
    model = Ojeph(cfg)
    model.load_state_dict(ck["model"])
    return model, cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--resume")
    ap.add_argument("--data", required=True, nargs="+", help="shard dirs (mixed evenly)")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--seq", type=int, default=375, help="frames per sample (375 = 30 s)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-quality", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.resume:
        model, cfg, ck = load(a.resume, a.device)
        start = ck["step"]
    else:
        cfg = preset(a.preset)
        model, start, ck = Ojeph(cfg), 0, None
    model.to(a.device).train()
    print(f"{cfg.name}: {model.num_params() / 1e6:.1f}M params on {a.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    if ck and ck.get("opt"):
        opt.load_state_dict(ck["opt"])

    iters = [iter(DataLoader(ConversationShards(d, cfg, a.seq, a.min_quality, seed=i),
                             batch_size=a.batch, num_workers=2)) for i, d in enumerate(a.data)]
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if a.device.startswith("cuda") \
        else torch.autocast("cpu", enabled=False)

    out = Path(a.out)
    t0 = time.time()
    for step in range(start, a.steps):
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * \
            0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        logs = {}
        for micro in range(a.accum):
            tok, spk, lang = next(iters[(step * a.accum + micro) % len(iters)])
            with amp:
                r = model(tok.to(a.device), spk.to(a.device), lang.to(a.device))
            (r["loss"] / a.accum).backward()
            for k in ("loss", "text_loss", "semantic_loss", "acoustic_loss"):
                logs[k] = logs.get(k, 0.0) + r[k].item() / a.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            el = time.time() - t0
            print(f"step {step} lr {lr:.2e} " + " ".join(f"{k} {v:.3f}" for k, v in logs.items())
                  + f" ({el:.0f}s)", flush=True)
        if (step + 1) % a.save_every == 0:
            save(out / "last.pt", model, opt, step + 1, cfg)
    save(out / "last.pt", model, opt, a.steps, cfg)


if __name__ == "__main__":
    main()
