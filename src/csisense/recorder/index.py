"""SQLite index of recordings and labels.

The Parquet files hold the signal; this holds everything you need to *find* the
signal. Listing sessions, filtering by class or day, and checking protocol
coverage all have to be fast enough to drive a UI, and opening a hundred Parquet
footers to answer "how many sit examples do I have from Tuesday" is not.

Labels are marked ``rejected`` rather than deleted. A fumbled repetition is
evidence that the capture protocol is too fast or the prompt is unclear, and
throwing it away throws away that signal along with it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    source_kind  TEXT NOT NULL,
    n_sub        INTEGER NOT NULL,
    rate_hz      REAL NOT NULL,
    started_ns   INTEGER NOT NULL,
    ended_ns     INTEGER NOT NULL,
    n_frames     INTEGER NOT NULL,
    duration_s   REAL NOT NULL,
    day          TEXT NOT NULL,
    position     TEXT NOT NULL DEFAULT '',
    subject      TEXT NOT NULL DEFAULT '',
    notes        TEXT NOT NULL DEFAULT '',
    dsp_fingerprint TEXT NOT NULL DEFAULT '',
    gain_locked  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    t_start_ns  INTEGER NOT NULL,
    t_end_ns    INTEGER NOT NULL,
    class       TEXT NOT NULL,
    tier        TEXT NOT NULL DEFAULT 'coarse',
    annotator   TEXT NOT NULL DEFAULT '',
    created_ns  INTEGER NOT NULL DEFAULT 0,
    rejected    INTEGER NOT NULL DEFAULT 0,
    uncertain   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_labels_session ON labels(session_id);
CREATE INDEX IF NOT EXISTS idx_labels_class ON labels(class, tier, rejected);
CREATE INDEX IF NOT EXISTS idx_sessions_day ON sessions(day);
"""


@dataclass(slots=True)
class SessionRow:
    session_id: str
    path: str
    source_kind: str
    n_sub: int
    rate_hz: float
    started_ns: int
    ended_ns: int
    n_frames: int
    duration_s: float
    day: str
    position: str = ""
    subject: str = ""
    notes: str = ""
    dsp_fingerprint: str = ""
    gain_locked: bool = False


@dataclass(slots=True)
class LabelRow:
    id: int
    session_id: str
    t_start_ns: int
    t_end_ns: int
    cls: str
    tier: str = "coarse"
    annotator: str = ""
    created_ns: int = 0
    rejected: bool = False
    uncertain: bool = False

    @property
    def duration_s(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / 1e9


class SessionIndex:
    """Catalogue of recordings and their labels."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            # WAL so the dashboard can read while a recording is being written.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- sessions ----------------------------------------------------------

    def add_session(self, row: SessionRow) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, path, source_kind, n_sub, rate_hz, started_ns, ended_ns,
                    n_frames, duration_s, day, position, subject, notes, dsp_fingerprint,
                    gain_locked)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row.session_id,
                    row.path,
                    row.source_kind,
                    row.n_sub,
                    row.rate_hz,
                    row.started_ns,
                    row.ended_ns,
                    row.n_frames,
                    row.duration_s,
                    row.day,
                    row.position,
                    row.subject,
                    row.notes,
                    row.dsp_fingerprint,
                    int(row.gain_locked),
                ),
            )

    def sessions(self, *, day: str | None = None) -> list[SessionRow]:
        query = "SELECT * FROM sessions"
        args: tuple[object, ...] = ()
        if day is not None:
            query += " WHERE day = ?"
            args = (day,)
        query += " ORDER BY started_ns"
        with self._conn() as conn:
            return [_session_row(r) for r in conn.execute(query, args)]

    def session(self, session_id: str) -> SessionRow | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return _session_row(row) if row else None

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM labels WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # -- labels ------------------------------------------------------------

    def add_label(
        self,
        session_id: str,
        *,
        t_start_ns: int,
        t_end_ns: int,
        cls: str,
        tier: str = "coarse",
        annotator: str = "",
        created_ns: int = 0,
        uncertain: bool = False,
    ) -> int:
        if t_end_ns <= t_start_ns:
            raise ValueError(f"label ends at or before it starts ({t_start_ns}..{t_end_ns})")
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO labels
                   (session_id, t_start_ns, t_end_ns, class, tier, annotator, created_ns,
                    uncertain)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    session_id,
                    t_start_ns,
                    t_end_ns,
                    cls,
                    tier,
                    annotator,
                    created_ns,
                    int(uncertain),
                ),
            )
            return int(cur.lastrowid or 0)

    def reject_label(self, label_id: int, rejected: bool = True) -> None:
        """Mark a repetition as fumbled without deleting it.

        Kept because a pile of rejections is diagnostic: it means the metronome
        is too fast or the prompt is ambiguous.
        """
        with self._conn() as conn:
            conn.execute("UPDATE labels SET rejected = ? WHERE id = ?", (int(rejected), label_id))

    def labels(
        self,
        *,
        session_id: str | None = None,
        tier: str | None = None,
        include_rejected: bool = False,
    ) -> list[LabelRow]:
        query = "SELECT * FROM labels WHERE 1=1"
        args: list[object] = []
        if session_id is not None:
            query += " AND session_id = ?"
            args.append(session_id)
        if tier is not None:
            query += " AND tier = ?"
            args.append(tier)
        if not include_rejected:
            query += " AND rejected = 0"
        query += " ORDER BY t_start_ns"
        with self._conn() as conn:
            return [_label_row(r) for r in conn.execute(query, tuple(args))]

    # -- coverage ----------------------------------------------------------

    def class_counts(self, *, tier: str = "coarse") -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT class, COUNT(*) AS n FROM labels
                   WHERE tier = ? AND rejected = 0 GROUP BY class""",
                (tier,),
            ).fetchall()
        return {r["class"]: r["n"] for r in rows}

    def coverage(self, *, tier: str = "coarse") -> dict[tuple[str, str, str], int]:
        """Counts keyed by ``(class, position, day)``.

        This is what the Train view renders and what the protocol validates
        against: a model trained on one position on one day scores beautifully
        under leave-one-session-out and then fails on Tuesday.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT l.class AS cls, s.position AS position, s.day AS day,
                          COUNT(*) AS n
                   FROM labels l JOIN sessions s ON s.session_id = l.session_id
                   WHERE l.tier = ? AND l.rejected = 0
                   GROUP BY l.class, s.position, s.day""",
                (tier,),
            ).fetchall()
        return {(r["cls"], r["position"], r["day"]): r["n"] for r in rows}

    def rejection_rate(self, *, tier: str = "coarse") -> float:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT SUM(rejected) AS rejected, COUNT(*) AS total
                   FROM labels WHERE tier = ?""",
                (tier,),
            ).fetchone()
        total = row["total"] or 0
        return (row["rejected"] or 0) / total if total else 0.0


def _session_row(r: sqlite3.Row) -> SessionRow:
    return SessionRow(
        session_id=r["session_id"],
        path=r["path"],
        source_kind=r["source_kind"],
        n_sub=r["n_sub"],
        rate_hz=r["rate_hz"],
        started_ns=r["started_ns"],
        ended_ns=r["ended_ns"],
        n_frames=r["n_frames"],
        duration_s=r["duration_s"],
        day=r["day"],
        position=r["position"],
        subject=r["subject"],
        notes=r["notes"],
        dsp_fingerprint=r["dsp_fingerprint"],
        gain_locked=bool(r["gain_locked"]),
    )


def _label_row(r: sqlite3.Row) -> LabelRow:
    return LabelRow(
        id=r["id"],
        session_id=r["session_id"],
        t_start_ns=r["t_start_ns"],
        t_end_ns=r["t_end_ns"],
        cls=r["class"],
        tier=r["tier"],
        annotator=r["annotator"],
        created_ns=r["created_ns"],
        rejected=bool(r["rejected"]),
        uncertain=bool(r["uncertain"]),
    )


__all__ = ["LabelRow", "SessionIndex", "SessionRow"]
