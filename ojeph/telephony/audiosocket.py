"""Asterisk AudioSocket server — the bridge between a real phone number and OJEPH.

    Caller (any phone) -> MTN/Airtel/Glo -> SIP trunk or GSM gateway -> Asterisk
        -> AudioSocket (TCP, 8 kHz 16-bit mono, 20 ms frames) -> this server -> OJEPH

AudioSocket wire format: [type:1][length:2 big-endian][payload]
    0x00 hangup   0x01 uuid (16 bytes)   0x03 DTMF   0x10 audio   0xff error

Asterisk does not send caller ID over AudioSocket, so the dialplan first calls
GET /register?uuid=..&cid=.. on the small HTTP side-channel below.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
import uuid as uuidlib
from urllib.parse import parse_qs, urlparse

from ojeph.telephony.backends import Backend

log = logging.getLogger("ojeph.audiosocket")

T_HANGUP, T_UUID, T_DTMF, T_AUDIO, T_ERROR = 0x00, 0x01, 0x03, 0x10, 0xFF
FRAME_BYTES = 320          # 20 ms @ 8 kHz, 16-bit
FRAME_SEC = 0.020


def pack(kind: int, payload: bytes = b"") -> bytes:
    return struct.pack(">BH", kind, len(payload)) + payload


async def read_msg(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    head = await reader.readexactly(3)
    kind, n = struct.unpack(">BH", head)
    payload = await reader.readexactly(n) if n else b""
    return kind, payload


class CallerRegistry:
    """uuid -> caller id, filled by the dialplan before AudioSocket starts."""

    def __init__(self, ttl: float = 120.0):
        self._m: dict[str, tuple[str, float]] = {}
        self.ttl = ttl

    def put(self, call_uuid: str, cid: str):
        now = time.time()
        self._m = {k: v for k, v in self._m.items() if now - v[1] < self.ttl}
        self._m[call_uuid.lower()] = (cid, now)

    def pop(self, call_uuid: str) -> str:
        return self._m.pop(call_uuid.lower(), ("unknown", 0))[0]


class AudioSocketServer:
    def __init__(self, backend: Backend, host: str = "0.0.0.0", port: int = 9092,
                 http_port: int = 8090, max_calls: int = 20):
        self.backend, self.host, self.port, self.http_port = backend, host, port, http_port
        self.registry = CallerRegistry()
        self.sem = asyncio.Semaphore(max_calls)
        self.active = 0

    # ---------------------------------------------------------------- calls
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        if self.sem.locked():
            log.warning("line busy: max calls reached")
            writer.write(pack(T_HANGUP))
            await writer.drain()
            writer.close()
            return
        async with self.sem:
            self.active += 1
            try:
                await self._call(reader, writer)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                pass
            finally:
                self.active -= 1
                writer.close()

    async def _call(self, reader, writer):
        kind, payload = await read_msg(reader)
        if kind != T_UUID or len(payload) != 16:
            log.error("expected uuid first, got type=%s", kind)
            return
        call_uuid = str(uuidlib.UUID(bytes=payload))
        cid = self.registry.pop(call_uuid)
        log.info("call %s from %s (active=%d)", call_uuid, cid[-4:].rjust(len(cid), "*"), self.active)

        loop = asyncio.get_running_loop()
        session = await loop.run_in_executor(None, self.backend.open, cid)
        out_q: asyncio.Queue[bytes] = asyncio.Queue()
        sender = asyncio.create_task(self._pace(writer, out_q))
        pending = b""
        try:
            while True:
                kind, payload = await read_msg(reader)
                if kind == T_HANGUP:
                    break
                if kind == T_DTMF:
                    session.dtmf(payload.decode(errors="ignore"))
                    continue
                if kind == T_ERROR:
                    log.warning("asterisk error %r", payload)
                    break
                if kind != T_AUDIO:
                    continue
                # model work runs off the event loop so many calls can share it
                out = await loop.run_in_executor(None, session.push, payload)
                pending += out
                while len(pending) >= FRAME_BYTES:
                    out_q.put_nowait(pending[:FRAME_BYTES])
                    pending = pending[FRAME_BYTES:]
        finally:
            sender.cancel()
            await loop.run_in_executor(None, session.close)
            log.info("call %s ended", call_uuid)

    @staticmethod
    async def _pace(writer: asyncio.StreamWriter, q: asyncio.Queue):
        """Send audio back in real time (20 ms per frame) so Asterisk never drops it."""
        nxt = time.perf_counter()
        while True:
            frame = await q.get()
            writer.write(pack(T_AUDIO, frame))
            await writer.drain()
            nxt = max(nxt + FRAME_SEC, time.perf_counter())
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

    # ----------------------------------------------------------- side http
    async def http(self, reader, writer):
        try:
            line = (await reader.readline()).decode(errors="ignore")
            while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                pass
            parts = line.split()
            path = parts[1] if len(parts) > 1 else "/"
            u = urlparse(path)
            q = parse_qs(u.query)
            if u.path == "/register" and "uuid" in q:
                self.registry.put(q["uuid"][0], q.get("cid", ["unknown"])[0])
                body, code = b"ok", "200 OK"
            elif u.path == "/health":
                body, code = f'{{"active_calls": {self.active}}}'.encode(), "200 OK"
            else:
                body, code = b"not found", "404 Not Found"
            writer.write(f"HTTP/1.1 {code}\r\nContent-Length: {len(body)}\r\n"
                         f"Connection: close\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()

    async def serve(self):
        a = await asyncio.start_server(self.handle, self.host, self.port)
        h = await asyncio.start_server(self.http, "127.0.0.1", self.http_port)
        log.info("AudioSocket on %s:%d, register/health on 127.0.0.1:%d",
                 self.host, self.port, self.http_port)
        async with a, h:
            await asyncio.gather(a.serve_forever(), h.serve_forever())
