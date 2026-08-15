from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class Submission:
    id: int
    job_number: str
    user_id: int
    chat_id: int
    author_name: str
    author_username: str | None
    video_path: str
    output_filename: str
    status: str
    destination: str | None
    karma_award: int
    publication_chat_id: int | None
    publication_message_id: int | None
    created_at: float
    updated_at: float


class ProposalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
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
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_number TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    author_name TEXT NOT NULL,
                    author_username TEXT,
                    video_path TEXT NOT NULL,
                    output_filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    destination TEXT,
                    karma_award INTEGER NOT NULL DEFAULT 0,
                    publication_chat_id INTEGER,
                    publication_message_id INTEGER,
                    processing_by INTEGER,
                    processing_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, job_number)
                );

                CREATE TABLE IF NOT EXISTS moderator_messages (
                    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
                    moderator_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(submission_id, moderator_id)
                );

                CREATE TABLE IF NOT EXISTS karma_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL,
                    delta INTEGER NOT NULL,
                    old_award INTEGER NOT NULL,
                    new_award INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    moderator_id INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS author_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    moderator_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error TEXT,
                    created_at REAL NOT NULL,
                    delivered_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_submissions_status
                    ON submissions(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_karma_events_user
                    ON karma_events(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_author_outbox_status
                    ON author_outbox(status, created_at);
                """
            )

    def create_submission(
        self,
        *,
        job_number: str,
        user_id: int,
        chat_id: int,
        author_name: str,
        author_username: str | None,
        video_path: Path,
        output_filename: str,
    ) -> tuple[Submission, bool]:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO submissions (
                    job_number, user_id, chat_id, author_name, author_username,
                    video_path, output_filename, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_number,
                    user_id,
                    chat_id,
                    author_name,
                    author_username,
                    str(video_path),
                    output_filename,
                    now,
                    now,
                ),
            )
            created = cursor.rowcount > 0
            if not created:
                connection.execute(
                    """
                    UPDATE submissions
                    SET author_name = ?, author_username = ?, video_path = ?,
                        output_filename = ?, updated_at = ?
                    WHERE user_id = ? AND job_number = ?
                    """,
                    (
                        author_name,
                        author_username,
                        str(video_path),
                        output_filename,
                        now,
                        user_id,
                        job_number,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM submissions WHERE user_id = ? AND job_number = ?",
                (user_id, job_number),
            ).fetchone()
        if row is None:
            raise RuntimeError("Could not create proposal submission")
        return _submission_from_row(row), created

    def get_submission(self, submission_id: int) -> Submission | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        return _submission_from_row(row) if row is not None else None

    def pending_for_moderator(self, moderator_id: int, *, limit: int = 20) -> list[Submission]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.*
                FROM submissions AS s
                LEFT JOIN moderator_messages AS m
                    ON m.submission_id = s.id AND m.moderator_id = ?
                WHERE s.status = 'pending' AND m.submission_id IS NULL
                ORDER BY s.created_at ASC
                LIMIT ?
                """,
                (moderator_id, max(1, limit)),
            ).fetchall()
        return [_submission_from_row(row) for row in rows]

    def record_moderator_message(
        self,
        submission_id: int,
        moderator_id: int,
        chat_id: int,
        message_id: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO moderator_messages (
                    submission_id, moderator_id, chat_id, message_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(submission_id, moderator_id) DO UPDATE SET
                    chat_id = excluded.chat_id,
                    message_id = excluded.message_id,
                    created_at = excluded.created_at
                """,
                (submission_id, moderator_id, chat_id, message_id, time.time()),
            )

    def moderator_messages(self, submission_id: int) -> list[tuple[int, int]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chat_id, message_id FROM moderator_messages WHERE submission_id = ?",
                (submission_id,),
            ).fetchall()
        return [(int(row["chat_id"]), int(row["message_id"])) for row in rows]

    def try_claim(self, submission_id: int, moderator_id: int) -> Submission | None:
        now = time.time()
        stale_before = now - 600.0
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE submissions
                SET processing_by = ?, processing_at = ?
                WHERE id = ?
                  AND (processing_at IS NULL OR processing_at < ?)
                """,
                (moderator_id, now, submission_id, stale_before),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        return _submission_from_row(row) if row is not None else None

    def release_claim(self, submission_id: int, moderator_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE submissions
                SET processing_by = NULL, processing_at = NULL
                WHERE id = ? AND processing_by = ?
                """,
                (submission_id, moderator_id),
            )

    def finish_decision(
        self,
        *,
        submission_id: int,
        moderator_id: int,
        destination: str,
        award: int,
        publication_chat_id: int | None,
        publication_message_id: int | None,
    ) -> tuple[Submission, int]:
        now = time.time()
        status = "rejected" if destination == "rejected" else "published"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None:
                raise KeyError(submission_id)
            if row["processing_by"] != moderator_id:
                raise RuntimeError("Submission decision lock was lost")
            old_award = int(row["karma_award"] or 0)
            delta = int(award) - old_award
            connection.execute(
                """
                UPDATE submissions
                SET status = ?, destination = ?, karma_award = ?,
                    publication_chat_id = ?, publication_message_id = ?,
                    processing_by = NULL, processing_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    destination,
                    int(award),
                    publication_chat_id,
                    publication_message_id,
                    now,
                    submission_id,
                ),
            )
            if delta:
                connection.execute(
                    """
                    INSERT INTO karma_events (
                        submission_id, user_id, delta, old_award, new_award,
                        reason, moderator_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        submission_id,
                        int(row["user_id"]),
                        delta,
                        old_award,
                        int(award),
                        destination,
                        moderator_id,
                        now,
                    ),
                )
            updated = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        if updated is None:
            raise RuntimeError("Submission vanished after decision")
        return _submission_from_row(updated), delta

    def enqueue_author_message(self, submission_id: int, moderator_id: int, text: str) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT user_id FROM submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None:
                raise KeyError(submission_id)
            cursor = connection.execute(
                """
                INSERT INTO author_outbox (
                    submission_id, user_id, text, moderator_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (submission_id, int(row["user_id"]), text, moderator_id, time.time()),
            )
            return int(cursor.lastrowid)

    def pending_author_messages(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT o.*, s.job_number
                FROM author_outbox AS o
                JOIN submissions AS s ON s.id = o.submission_id
                WHERE o.status = 'pending'
                ORDER BY o.created_at ASC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def finish_author_message(self, outbox_id: int, *, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE author_outbox
                SET status = ?, error = ?, delivered_at = ?
                WHERE id = ?
                """,
                ("failed" if error else "delivered", error, time.time(), outbox_id),
            )

    def karma_total(self, user_id: int) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(delta), 0) AS total FROM karma_events WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["total"] if row is not None else 0)


def _submission_from_row(row: sqlite3.Row) -> Submission:
    return Submission(
        id=int(row["id"]),
        job_number=str(row["job_number"]),
        user_id=int(row["user_id"]),
        chat_id=int(row["chat_id"]),
        author_name=str(row["author_name"]),
        author_username=str(row["author_username"]) if row["author_username"] else None,
        video_path=str(row["video_path"]),
        output_filename=str(row["output_filename"]),
        status=str(row["status"]),
        destination=str(row["destination"]) if row["destination"] else None,
        karma_award=int(row["karma_award"] or 0),
        publication_chat_id=int(row["publication_chat_id"]) if row["publication_chat_id"] is not None else None,
        publication_message_id=(
            int(row["publication_message_id"]) if row["publication_message_id"] is not None else None
        ),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )
