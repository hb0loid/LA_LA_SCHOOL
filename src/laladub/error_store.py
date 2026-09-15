from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class ErrorGroup:
    fingerprint: str
    source: str
    count: int
    first_seen: float
    last_seen: float
    details: str
    traceback_text: str
    stage: str | None
    job_numbers: tuple[str, ...]
    user_ids: tuple[int, ...]


class ErrorStore:
    """Persistent error inbox shared by the bot's failure paths."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS error_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    source TEXT NOT NULL,
                    job_number TEXT,
                    user_id INTEGER,
                    chat_id INTEGER,
                    stage TEXT,
                    details TEXT NOT NULL,
                    traceback_text TEXT NOT NULL,
                    user_message TEXT,
                    notified_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_error_events_pending
                    ON error_events(notified_at, created_at);
                CREATE INDEX IF NOT EXISTS idx_error_events_fingerprint
                    ON error_events(fingerprint, created_at);
                CREATE TABLE IF NOT EXISTS error_notification_state (
                    fingerprint TEXT PRIMARY KEY,
                    last_notified_at REAL NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
        finally:
            connection.close()

    def record(
        self,
        *,
        source: str,
        details: str,
        traceback_text: str = "",
        job_number: str | None = None,
        user_id: int | None = None,
        chat_id: int | None = None,
        stage: str | None = None,
        user_message: str | None = None,
        created_at: float | None = None,
    ) -> str:
        fingerprint = error_fingerprint(source, details, traceback_text)
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO error_events (
                        fingerprint, created_at, source, job_number, user_id,
                        chat_id, stage, details, traceback_text, user_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fingerprint,
                        float(created_at or time.time()),
                        str(source),
                        str(job_number) if job_number else None,
                        int(user_id) if user_id is not None else None,
                        int(chat_id) if chat_id is not None else None,
                        str(stage) if stage else None,
                        str(details),
                        str(traceback_text),
                        str(user_message) if user_message else None,
                    ),
                )
        return fingerprint

    def pending_groups(
        self,
        *,
        settle_seconds: float = 10.0,
        cooldown_seconds: float = 15 * 60.0,
        limit: int = 10,
        now: float | None = None,
    ) -> list[ErrorGroup]:
        now = float(now or time.time())
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.fingerprint
                FROM error_events e
                LEFT JOIN error_notification_state s ON s.fingerprint = e.fingerprint
                WHERE e.notified_at IS NULL
                GROUP BY e.fingerprint
                HAVING MIN(e.created_at) <= ?
                   AND (MAX(s.last_notified_at) IS NULL OR MAX(s.last_notified_at) <= ?)
                ORDER BY MIN(e.created_at)
                LIMIT ?
                """,
                (now - settle_seconds, now - cooldown_seconds, max(1, int(limit))),
            ).fetchall()
            return [self._load_group(connection, str(row["fingerprint"]), pending_only=True) for row in rows]

    def recent_groups(self, *, since: float, limit: int = 10) -> list[ErrorGroup]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT fingerprint, MAX(created_at) AS latest
                FROM error_events
                WHERE created_at >= ?
                GROUP BY fingerprint
                ORDER BY latest DESC
                LIMIT ?
                """,
                (float(since), max(1, int(limit))),
            ).fetchall()
            return [self._load_group(connection, str(row["fingerprint"]), since=since) for row in rows]

    def mark_notified(self, fingerprint: str, *, notified_at: float | None = None) -> None:
        notified_at = float(notified_at or time.time())
        with self._connect() as connection:
            with connection:
                connection.execute(
                    "UPDATE error_events SET notified_at = ? WHERE fingerprint = ? AND notified_at IS NULL",
                    (notified_at, fingerprint),
                )
                connection.execute(
                    """
                    INSERT INTO error_notification_state(fingerprint, last_notified_at)
                    VALUES (?, ?)
                    ON CONFLICT(fingerprint) DO UPDATE SET last_notified_at = excluded.last_notified_at
                    """,
                    (fingerprint, notified_at),
                )

    def _load_group(
        self,
        connection: sqlite3.Connection,
        fingerprint: str,
        *,
        pending_only: bool = False,
        since: float | None = None,
    ) -> ErrorGroup:
        conditions = ["fingerprint = ?"]
        values: list[object] = [fingerprint]
        if pending_only:
            conditions.append("notified_at IS NULL")
        if since is not None:
            conditions.append("created_at >= ?")
            values.append(float(since))
        rows = connection.execute(
            f"SELECT * FROM error_events WHERE {' AND '.join(conditions)} ORDER BY created_at", values
        ).fetchall()
        last = rows[-1]
        jobs = tuple(dict.fromkeys(str(row["job_number"]) for row in rows if row["job_number"]))
        users = tuple(dict.fromkeys(int(row["user_id"]) for row in rows if row["user_id"] is not None))
        return ErrorGroup(
            fingerprint=fingerprint,
            source=str(last["source"]),
            count=len(rows),
            first_seen=float(rows[0]["created_at"]),
            last_seen=float(last["created_at"]),
            details=str(last["details"]),
            traceback_text=str(last["traceback_text"]),
            stage=str(last["stage"]) if last["stage"] else None,
            job_numbers=jobs,
            user_ids=users,
        )


def error_fingerprint(source: str, details: str, traceback_text: str = "") -> str:
    lines = [line.strip() for line in str(traceback_text).splitlines() if line.strip()]
    signature = lines[-1] if lines else str(details).strip()
    signature = re.sub(r"[A-Za-z]:\\[^\r\n'\"]+", "<path>", signature)
    signature = re.sub(r"\b\d{4,}\b", "<n>", signature)
    signature = re.sub(r"\s+", " ", signature).strip().lower()
    return hashlib.sha256(f"{source}|{signature}".encode("utf-8", errors="replace")).hexdigest()[:24]
