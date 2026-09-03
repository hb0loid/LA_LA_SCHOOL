from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Records what people did with a proposed dub text before it was voiced, so the
# choices can be studied later: which variants get taken, which get thrown back.


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    id: int
    job_number: str
    user_id: int
    attempt: int
    decision: str  # approved | rejected | cancelled
    text: str
    source_lang: str
    target_lang: str
    created_at: float


def _record_from_row(row: sqlite3.Row) -> ReviewRecord:
    return ReviewRecord(
        id=int(row["id"]),
        job_number=str(row["job_number"]),
        user_id=int(row["user_id"]),
        attempt=int(row["attempt"]),
        decision=str(row["decision"]),
        text=str(row["text"]),
        source_lang=str(row["source_lang"] or ""),
        target_lang=str(row["target_lang"] or ""),
        created_at=float(row["created_at"]),
    )


class TextReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS text_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_number TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_lang TEXT,
                    target_lang TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_text_reviews_decision
                    ON text_reviews(decision, created_at);
                CREATE INDEX IF NOT EXISTS idx_text_reviews_job
                    ON text_reviews(job_number);
                """
            )

    def record(
        self,
        *,
        job_number: str,
        user_id: int,
        attempt: int,
        decision: str,
        text: str,
        source_lang: str = "",
        target_lang: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO text_reviews (
                    job_number, user_id, attempt, decision, text,
                    source_lang, target_lang, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_number, user_id, attempt, decision, text, source_lang, target_lang, time.time()),
            )

    def summary(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT decision, COUNT(*) AS n FROM text_reviews GROUP BY decision"
            ).fetchall()
        return {str(row["decision"]): int(row["n"]) for row in rows}

    def recent(self, decision: str | None = None, limit: int = 20) -> list[ReviewRecord]:
        query = "SELECT * FROM text_reviews"
        params: list[object] = []
        if decision:
            query += " WHERE decision = ?"
            params.append(decision)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_record_from_row(row) for row in rows]

    def attempts_for_job(self, job_number: str) -> int:
        """How many variants this job has already shown - the retry counter."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt), 0) AS last FROM text_reviews WHERE job_number = ?",
                (job_number,),
            ).fetchone()
        return int(row["last"]) if row is not None else 0
