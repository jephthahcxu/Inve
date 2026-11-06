"""Caller memory: every phone number gets a UID; OJEPH remembers past calls.

* The raw phone number is never stored — UID = HMAC-SHA256(secret, number).
  (Nigeria Data Protection Act 2023: minimise and protect personal data.)
* Each call stores a short summary + language + consent flag.
* Incidents (security, accident, lost person, health emergency) are stored as
  structured reports that a human operator / authority dashboard can pull.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS callers (
    uid TEXT PRIMARY KEY,
    created REAL NOT NULL,
    last_call REAL,
    calls INTEGER NOT NULL DEFAULT 0,
    lang TEXT,
    speaker TEXT,
    name TEXT,
    profile TEXT NOT NULL DEFAULT '{}',
    consent_record INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL REFERENCES callers(uid),
    started REAL NOT NULL,
    ended REAL,
    lang TEXT,
    summary TEXT,
    transcript TEXT
);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL REFERENCES callers(uid),
    created REAL NOT NULL,
    kind TEXT NOT NULL,
    location TEXT,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'open'
);
"""

INCIDENT_KINDS = {"security", "accident", "fire", "medical", "lost", "abuse", "other"}


@dataclass
class Caller:
    uid: str
    calls: int
    last_call: float | None
    lang: str | None
    speaker: str | None
    name: str | None
    profile: dict
    consent_record: bool


class MemoryStore:
    def __init__(self, path: str | Path, secret: bytes):
        if not secret:
            raise ValueError("OJEPH_UID_SECRET must be set")
        self.secret = secret
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ------------------------------------------------------------------ ids
    def uid_for(self, phone: str) -> str:
        digits = "".join(c for c in phone if c.isdigit())
        if digits.startswith("0") and len(digits) == 11:      # 0803... -> 234803...
            digits = "234" + digits[1:]
        return hmac.new(self.secret, digits.encode(), hashlib.sha256).hexdigest()[:24]

    # -------------------------------------------------------------- callers
    def get_or_create(self, phone: str) -> Caller:
        uid = self.uid_for(phone)
        row = self.db.execute("SELECT uid, calls, last_call, lang, speaker, name, profile, "
                              "consent_record FROM callers WHERE uid=?", (uid,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO callers(uid, created) VALUES(?, ?)", (uid, time.time()))
            self.db.commit()
            return Caller(uid, 0, None, None, None, None, {}, False)
        return Caller(row[0], row[1], row[2], row[3], row[4], row[5], json.loads(row[6]),
                      bool(row[7]))

    def update_caller(self, uid: str, **fields):
        allowed = {"lang", "speaker", "name", "consent_record"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if "profile" in fields:
            sets["profile"] = json.dumps(fields["profile"], ensure_ascii=False)
        if not sets:
            return
        cols = ", ".join(f"{k}=?" for k in sets)
        self.db.execute(f"UPDATE callers SET {cols} WHERE uid=?", (*sets.values(), uid))
        self.db.commit()

    # ---------------------------------------------------------------- calls
    def start_call(self, uid: str, lang: str | None = None) -> int:
        now = time.time()
        cur = self.db.execute("INSERT INTO calls(uid, started, lang) VALUES(?,?,?)",
                              (uid, now, lang))
        self.db.execute("UPDATE callers SET calls=calls+1, last_call=? WHERE uid=?", (now, uid))
        self.db.commit()
        return int(cur.lastrowid)

    def end_call(self, call_id: int, summary: str = "", transcript: str = ""):
        self.db.execute("UPDATE calls SET ended=?, summary=?, transcript=? WHERE id=?",
                        (time.time(), summary, transcript, call_id))
        self.db.commit()

    def recent_summaries(self, uid: str, n: int = 3) -> list[str]:
        rows = self.db.execute("SELECT summary FROM calls WHERE uid=? AND summary != '' "
                               "ORDER BY started DESC LIMIT ?", (uid, n)).fetchall()
        return [r[0] for r in rows]

    def memory_prompt(self, phone: str) -> str:
        """Text OJEPH 'thinks' silently at call start (see CallStream.prime_text)."""
        c = self.get_or_create(phone)
        if c.calls == 0:
            return "New caller. Greet warmly, ask their name and how you can help."
        parts = [f"Returning caller ({c.calls} previous calls)."]
        if c.name:
            parts.append(f"Name: {c.name}.")
        if c.lang:
            parts.append(f"Speaks: {c.lang}.")
        for s in self.recent_summaries(c.uid):
            parts.append(f"Before: {s}")
        return " ".join(parts)

    # ------------------------------------------------------------ incidents
    def report_incident(self, uid: str, kind: str, location: str = "",
                        description: str = "") -> int:
        if kind not in INCIDENT_KINDS:
            kind = "other"
        cur = self.db.execute("INSERT INTO incidents(uid, created, kind, location, description) "
                              "VALUES(?,?,?,?,?)", (uid, time.time(), kind, location, description))
        self.db.commit()
        return int(cur.lastrowid)

    def open_incidents(self) -> list[dict]:
        rows = self.db.execute("SELECT id, uid, created, kind, location, description FROM incidents "
                               "WHERE status='open' ORDER BY created DESC").fetchall()
        keys = ("id", "uid", "created", "kind", "location", "description")
        return [dict(zip(keys, r)) for r in rows]
