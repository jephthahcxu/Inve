import asyncio
import struct
import uuid

import numpy as np
import pytest
import torch

from ojeph.audio import codec_to_phone, phone_to_codec
from ojeph.config import preset
from ojeph.data.dataset import align_text, build_tokens
from ojeph.data.languages import LANG_IDS, lang_id
from ojeph.inference.streamer import CallStream
from ojeph.memory.store import MemoryStore
from ojeph.model.codec import DummyCodec
from ojeph.model.delay import apply_delay, undo_delay
from ojeph.model.ojeph import Ojeph
from ojeph.telephony.audiosocket import (T_AUDIO, T_HANGUP, T_UUID, AudioSocketServer, pack)
from ojeph.telephony.backends import EchoBackend


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = preset("tiny")
    return Ojeph(cfg).eval(), cfg


def rand_tokens(cfg, B=2, T=24):
    K = cfg.codec.num_codebooks
    text = torch.randint(0, cfg.text_vocab_total, (B, 1, T))
    audio = torch.randint(0, cfg.codec.codebook_size, (B, 2 * K, T))
    return torch.cat([text, audio], 1)


def test_delay_roundtrip():
    c = torch.randint(0, 100, (2, 4, 30))
    d = apply_delay(c, 2, pad=999)
    assert (d[:, 1:, :2] == 999).all()
    assert torch.equal(undo_delay(d, 2), c[..., :28])


def test_forward_loss(tiny):
    model, cfg = tiny
    tok = rand_tokens(cfg)
    r = model(tok, torch.tensor([0, 1]), torch.tensor([lang_id("yoruba"), lang_id("pidgin")]))
    assert r["loss"].isfinite()
    assert r["audio_logits"].shape == (2, 24, cfg.codec.num_codebooks, cfg.codec.codebook_size)


def test_streaming_matches_training_forward(tiny):
    """KV-cached step-by-step inference must equal the parallel training pass."""
    model, cfg = tiny
    K = cfg.codec.num_codebooks
    tok = rand_tokens(cfg, B=1, T=16)
    spk, lang = torch.tensor([0]), torch.tensor([3])
    full = model(tok, spk, lang)

    st = model.init_state(1, spk, lang)
    prev = None
    for t in range(16):
        # teacher-force the step and compare the text logits through the head
        B = 1
        x = model.bos.expand(B, 1, -1) if prev is None else model._frame_embed(prev[:, :, None])
        h = model.temporal(x + model._cond(st["speaker"], st["lang"]), st["caches"])[:, -1]
        assert torch.allclose(model.text_head(h), full["text_logits"][:, t], atol=1e-4)
        prev = tok[:, :, t]
    assert K > 0


def test_step_shapes(tiny):
    model, cfg = tiny
    st = model.init_state(1, torch.tensor([1]), torch.tensor([0]))
    text, audio = model.step(st, None)
    assert text.shape == (1,) and audio.shape == (1, cfg.codec.num_codebooks)
    assert int(audio.max()) < cfg.codec.codebook_size


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    cfg = preset("tiny")
    model = Ojeph(cfg)
    tok = rand_tokens(cfg, B=2, T=16)
    spk, lang = torch.tensor([0, 1]), torch.tensor([1, 2])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = model(tok, spk, lang)["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_callstream_realtime_loop(tiny):
    model, cfg = tiny
    codec = DummyCodec(cfg.codec.num_codebooks, cfg.codec.codebook_size)
    cs = CallStream(model, codec, speaker=0, lang=lang_id("hausa"))
    cs.prime_text([5, 6, 7])
    out = cs.push(np.random.uniform(-0.3, 0.3, codec.frame_samples * 5).astype(np.float32))
    assert out.size == codec.frame_samples * 5
    cs.close()


def test_align_text():
    t = align_text([(0.0, [1, 2]), (1.0, [3])], 20, 12.5, pad=0, epad=9)
    assert t[0] == 1 and t[1] == 2 and t[11] == 9 and t[12] == 3


def test_build_tokens_shape():
    cfg = preset("tiny")
    K = cfg.codec.num_codebooks
    conv = {"ojeph_codes": torch.zeros(K, 10, dtype=torch.long),
            "user_codes": torch.zeros(K, 10, dtype=torch.long),
            "text": torch.zeros(10, dtype=torch.long)}
    assert build_tokens(conv, cfg).shape == (1 + 2 * K, 10)


def test_languages_registry():
    for lang in ("yoruba", "hausa", "igbo", "igala", "pidgin"):
        assert lang_id(lang) > 0
    assert len(set(LANG_IDS.values())) == len(LANG_IDS)
    assert max(LANG_IDS.values()) < preset("small").num_languages


def test_resampler_length_and_tone():
    up, down = phone_to_codec(), codec_to_phone()
    x = np.sin(2 * np.pi * 440 * np.arange(1600) / 8000).astype(np.float32)
    y = np.concatenate([up(x[i:i + 160]) for i in range(0, 1600, 160)])
    assert y.size == 4800
    z = np.concatenate([down(y[i:i + 480]) for i in range(0, 4800, 480)])
    assert z.size == 1600
    # tone survives the round trip (search the filters' group delay)
    best = max(np.corrcoef(z[300:1500], x[300 - lag:1500 - lag])[0, 1] for lag in range(80))
    assert best > 0.99


def test_memory_store(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite", b"secret")
    assert m.uid_for("08031234567") == m.uid_for("+234 803 123 4567")
    c = m.get_or_create("08031234567")
    assert "New caller" in m.memory_prompt("08031234567")
    cid = m.start_call(c.uid)
    m.update_caller(c.uid, name="Musa", lang="hausa:standard_kano")
    m.end_call(cid, summary="Asked about maize fertilizer.")
    p = m.memory_prompt("08031234567")
    assert "Musa" in p and "maize" in p
    m.report_incident(c.uid, "security", "Lokoja-Abuja road km 40", "Road block")
    assert m.open_incidents()[0]["kind"] == "security"
    raw = (tmp_path / "m.sqlite").read_bytes()
    assert b"08031234567" not in raw and b"2348031234567" not in raw


def test_audiosocket_echo_roundtrip():
    async def run():
        srv = AudioSocketServer(EchoBackend(), host="127.0.0.1", port=0, http_port=0)
        server = await asyncio.start_server(srv.handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(pack(T_UUID, uuid.uuid4().bytes))
        frame = struct.pack("<160h", *([1000] * 160))
        for _ in range(40):
            w.write(pack(T_AUDIO, frame))
        await w.drain()
        got = 0
        while got < 10:
            head = await asyncio.wait_for(r.readexactly(3), 5)
            kind, n = struct.unpack(">BH", head)
            await r.readexactly(n)
            assert kind == T_AUDIO and n == 320
            got += 1
        w.write(pack(T_HANGUP))
        await w.drain()
        w.close()
        server.close()
    asyncio.run(run())
