"""Live lab: train OJEPH on the rendered conversations and publish progress.

    python scripts/lab_run.py --synth runs/synth --run runs/live

Writes, for the web console:
    runs/live/train_log.jsonl    one line per logged step
    runs/live/status.json        current state
    runs/live/samples/*.wav      OJEPH answering a caller, from each checkpoint

On a CPU this trains the `mini` preset: a small run that shows the model
learning. Quality comes from the same script on a GPU with `--preset small`
and far more data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ojeph.config import preset  # noqa: E402
from ojeph.data.dataset import align_text, build_tokens  # noqa: E402
from ojeph.data.languages import lang_id  # noqa: E402
from ojeph.inference.offline import respond  # noqa: E402
from ojeph.model.codec import load_codec  # noqa: E402
from ojeph.model.ojeph import Ojeph  # noqa: E402


def status(run: Path, **kw):
    (run / "status.json").write_text(json.dumps({"time": time.time(), **kw}))


def write_wav(path: Path, x: np.ndarray, sr: int):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[0] if pcm.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.T.tobytes() if pcm.ndim == 2 else pcm.tobytes())


def train_tokenizer(synth: Path, out: Path, vocab: int) -> "spm.SentencePieceProcessor":
    import sentencepiece as spm

    text = []
    for m in synth.glob("*.json"):
        text.append(" ".join(w for _, w in json.loads(m.read_text())["words"]))
    corpus = out / "corpus.txt"
    corpus.write_text("\n".join(text), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(out / "ojeph_tok"),
                                   vocab_size=vocab, model_type="unigram", character_coverage=1.0,
                                   hard_vocab_limit=False, bos_id=-1, eos_id=-1)
    return spm.SentencePieceProcessor(model_file=str(out / "ojeph_tok.model"))


def encode_all(synth: Path, codec, sp, cfg) -> list[dict]:
    convs = []
    for m in sorted(synth.glob("*.json")):
        meta = json.loads(m.read_text())
        audio = torch.from_numpy(np.load(m.with_suffix(".npy")).astype(np.float32))
        codes = codec.encode(audio[:, None])           # [2, K, T]
        T = codes.shape[-1]
        words = [(s, sp.encode(" " + w)) for s, w in meta["words"]]
        text = align_text(words, T, cfg.codec.frame_rate, cfg.text_pad, cfg.text_epad)
        convs.append({"ojeph_codes": codes[0], "user_codes": codes[1], "text": text,
                      "speaker": int(meta["speaker"]), "lang": lang_id(meta["lang"]),
                      "name": m.stem})
    return convs


def batch(convs, cfg, seq, bs, rng):
    toks, spk, lang = [], [], []
    for _ in range(bs):
        c = rng.choice(convs)
        tok = build_tokens(c, cfg)
        T = tok.shape[-1]
        s = rng.randrange(0, max(1, T - seq))
        tok = tok[:, s:s + seq]
        if tok.shape[-1] < seq:
            fill = torch.full((tok.shape[0], seq - tok.shape[-1]), cfg.audio_pad, dtype=torch.long)
            fill[0] = cfg.text_pad
            tok = torch.cat([tok, fill], 1)
        toks.append(tok)
        spk.append(c["speaker"])
        lang.append(c["lang"])
    return torch.stack(toks), torch.tensor(spk), torch.tensor(lang)


def make_sample(model, codec, conv, cfg, step, loss, run: Path, seconds: float, prompt_s: float):
    """OJEPH listens to the caller channel and talks back, primed with a short real prefix."""
    model.eval()
    fr = cfg.codec.frame_rate
    # start where the caller speaks, so OJEPH has something to answer
    T = int(seconds * fr)
    P = int(prompt_s * fr)
    user = conv["user_codes"][:, :T]
    prompt = (conv["text"][:P], conv["ojeph_codes"][:, :P])
    oj = respond(model, user, conv["speaker"], conv["lang"], temperature=0.7, top_k=50,
                 prompt=prompt)
    with torch.no_grad():
        y_oj = codec.decode(oj[None])[0, 0].numpy()
        y_us = codec.decode(user[None, :, :oj.shape[-1]])[0, 0].numpy()
    n = min(y_oj.size, y_us.size)
    stereo = np.stack([y_oj[:n], 0.6 * y_us[:n]])         # L = OJEPH, R = caller
    name = f"step{step:06d}"
    write_wav(run / "samples" / f"{name}.wav", stereo, codec.sample_rate)
    (run / "samples" / f"{name}.json").write_text(json.dumps(
        {"step": step, "loss": loss, "voice": "male" if conv["speaker"] == 0 else "female",
         "conversation": conv["name"], "seconds": n / codec.sample_rate}))
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="runs/synth")
    ap.add_argument("--run", default="runs/live")
    ap.add_argument("--preset", default="mini")
    ap.add_argument("--seq", type=int, default=125)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    run, synth = Path(a.run), Path(a.synth)
    (run / "samples").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    cfg = preset(a.preset)

    status(run, state="tokenizer")
    sp = train_tokenizer(synth, run, cfg.text_vocab)
    status(run, state="loading codec")
    codec = load_codec("mimi", cfg.codec.num_codebooks)
    status(run, state="encoding audio")
    convs = encode_all(synth, codec, sp, cfg)
    print(f"{len(convs)} conversations, {sum(c['user_codes'].shape[-1] for c in convs) / 12.5 / 60:.1f} min")

    model = Ojeph(cfg)
    print(f"{cfg.name}: {model.num_params() / 1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    log = open(run / "train_log.jsonl", "a")
    ema = None
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr
        tok, spk, lang = batch(convs, cfg, a.seq, a.batch, rng)
        r = model(tok, spk, lang)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        L = r["loss"].item()
        ema = L if ema is None else 0.95 * ema + 0.05 * L
        if step % 10 == 0:
            log.write(json.dumps({"step": step, "loss": ema, "text": r["text_loss"].item(),
                                  "semantic": r["semantic_loss"].item(),
                                  "acoustic": r["acoustic_loss"].item(),
                                  "sec": time.time() - t0}) + "\n")
            log.flush()
            status(run, state="training", step=step, loss=ema, params=model.num_params())
        if step % a.sample_every == 0:
            status(run, state="speaking", step=step, loss=ema)
            make_sample(model, codec, rng.choice(convs), cfg, step, ema, run, 14.0, 3.0)
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step},
                       run / "last.pt")
    status(run, state="done", step=a.steps, loss=ema)


if __name__ == "__main__":
    main()
