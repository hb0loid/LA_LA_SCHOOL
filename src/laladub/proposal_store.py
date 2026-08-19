from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class ScheduledPost:
    id: int
    submission_id: int
    destination: str
    target_chat: str
    moderator_id: int
    scheduled_for: float
    status: str
    created_at: float
    sent_at: float | None


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
    karma_milli: int
    karma_before_milli: int
    duration_ms: int
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
                    karma_before_milli INTEGER NOT NULL DEFAULT 0,
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    publication_chat_id INTEGER,
                    publication_message_id INTEGER,
                    comment_posted_at REAL,
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

                CREATE TABLE IF NOT EXISTS daily_usage (
                    user_id INTEGER NOT NULL,
                    job_number TEXT NOT NULL,
                    day_key TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(user_id, job_number)
                );

                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduled_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
                    destination TEXT NOT NULL,
                    target_chat TEXT NOT NULL,
                    moderator_id INTEGER NOT NULL,
                    scheduled_for REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    sent_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_scheduled_posts_status
                    ON scheduled_posts(status, scheduled_for);
                CREATE INDEX IF NOT EXISTS idx_scheduled_posts_submission
                    ON scheduled_posts(submission_id, status);

                CREATE INDEX IF NOT EXISTS idx_submissions_status
                    ON submissions(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_karma_events_user
                    ON karma_events(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_author_outbox_status
                    ON author_outbox(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_daily_usage_user_day
                    ON daily_usage(user_id, day_key);
                CREATE INDEX IF NOT EXISTS idx_submissions_publication
                    ON submissions(publication_chat_id, publication_message_id);
                """
            )
            columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(submissions)")}
            if "duration_ms" not in columns:
                connection.execute("ALTER TABLE submissions ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0")
            if "karma_before_milli" not in columns:
                connection.execute(
                    "ALTER TABLE submissions ADD COLUMN karma_before_milli INTEGER NOT NULL DEFAULT 0"
                )
            if "comment_posted_at" not in columns:
                connection.execute("ALTER TABLE submissions ADD COLUMN comment_posted_at REAL")
            migrated = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'karma_milli_v1'"
            ).fetchone()
            if migrated is None:
                connection.execute("UPDATE submissions SET karma_award = karma_award * 1000")
                connection.execute(
                    "UPDATE karma_events SET delta = delta * 1000, old_award = old_award * 1000, new_award = new_award * 1000"
                )
                connection.execute(
                    "INSERT INTO store_metadata(key, value) VALUES ('karma_milli_v1', '1')"
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
        duration_ms: int = 0,
        karma_before_milli: int = 0,
    ) -> tuple[Submission, bool]:
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO submissions (
                    job_number, user_id, chat_id, author_name, author_username,
                    video_path, output_filename, duration_ms, karma_before_milli, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_number,
                    user_id,
                    chat_id,
                    author_name,
                    author_username,
                    str(video_path),
                    output_filename,
                    max(0, int(duration_ms)),
                    int(karma_before_milli),
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
                        output_filename = ?,
                        duration_ms = CASE WHEN ? > 0 THEN ? ELSE duration_ms END,
                        updated_at = ?
                    WHERE user_id = ? AND job_number = ?
                    """,
                    (
                        author_name,
                        author_username,
                        str(video_path),
                        output_filename,
                        max(0, int(duration_ms)),
                        max(0, int(duration_ms)),
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
        award_milli: int,
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
            delta = int(award_milli) - old_award
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
                    int(award_milli),
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
                        int(award_milli),
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
        total, _event_count = self.karma_summary(user_id)
        return total

    def set_submission_duration(self, submission_id: int, duration_ms: int) -> Submission:
        with self._connect() as connection:
            connection.execute(
                "UPDATE submissions SET duration_ms = ?, updated_at = ? WHERE id = ?",
                (max(0, int(duration_ms)), time.time(), submission_id),
            )
            row = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        if row is None:
            raise KeyError(submission_id)
        return _submission_from_row(row)

    def revalue_submission(self, submission_id: int, *, duration_ms: int, award_milli: int) -> int:
        """Adjust an existing decision without reposting it (used by migrations)."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None:
                raise KeyError(submission_id)
            old_award = int(row["karma_award"] or 0)
            new_award = max(0, int(award_milli))
            delta = new_award - old_award
            connection.execute(
                "UPDATE submissions SET duration_ms = ?, karma_award = ?, updated_at = ? WHERE id = ?",
                (max(0, int(duration_ms)), new_award, now, submission_id),
            )
            if delta:
                connection.execute(
                    """
                    INSERT INTO karma_events (
                        submission_id, user_id, delta, old_award, new_award,
                        reason, moderator_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'duration_migration', 0, ?)
                    """,
                    (submission_id, int(row["user_id"]), delta, old_award, new_award, now),
                )
        return delta

    def reserve_daily_usage(
        self,
        *,
        user_id: int,
        job_number: str,
        duration_ms: int,
        limit_ms: int,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Rolling 24h quota: usage counts against the limit for exactly 24h after it
        happened, then ages out on its own - there is no shared reset moment, each
        user's window empties out at their own pace as their own old usage expires."""
        duration_ms = max(0, int(duration_ms))
        now = time.time() if now is None else now
        cutoff = now - 86400.0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT duration_ms FROM daily_usage WHERE user_id = ? AND job_number = ?",
                (user_id, job_number),
            ).fetchone()
            used_row = connection.execute(
                "SELECT COALESCE(SUM(duration_ms), 0) AS used FROM daily_usage WHERE user_id = ? AND created_at >= ?",
                (user_id, cutoff),
            ).fetchone()
            used_ms = int(used_row["used"] or 0) if used_row is not None else 0
            if existing is not None:
                return True, used_ms
            if used_ms + duration_ms > max(0, int(limit_ms)):
                return False, used_ms
            connection.execute(
                "INSERT INTO daily_usage(user_id, job_number, day_key, duration_ms, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, str(job_number), time.strftime("%Y-%m-%d", time.localtime(now)), duration_ms, now),
            )
            return True, used_ms + duration_ms

    def daily_usage_ms(self, user_id: int, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(duration_ms), 0) AS used FROM daily_usage WHERE user_id = ? AND created_at >= ?",
                (user_id, now - 86400.0),
            ).fetchone()
        return int(row["used"] or 0) if row is not None else 0

    def next_usage_free_at(self, user_id: int, now: float | None = None) -> float | None:
        """When the oldest usage still counted against the rolling window will age out
        (freeing up whatever it was holding). None if the user has no usage in-window."""
        now = time.time() if now is None else now
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MIN(created_at) AS oldest FROM daily_usage WHERE user_id = ? AND created_at >= ?",
                (user_id, now - 86400.0),
            ).fetchone()
        oldest = row["oldest"] if row is not None else None
        return float(oldest) + 86400.0 if oldest is not None else None

    def users_with_recent_usage(self, cutoff: float) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT user_id FROM daily_usage WHERE created_at >= ?",
                (cutoff,),
            ).fetchall()
        return [int(row["user_id"]) for row in rows]

    def release_daily_usage(self, user_id: int, job_number: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM daily_usage WHERE user_id = ? AND job_number = ?",
                (user_id, str(job_number)),
            )

    def publication_summary(self, user_id: int) -> dict[str, tuple[int, int]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT destination, COUNT(*) AS item_count,
                       COALESCE(SUM(duration_ms), 0) AS duration_ms
                FROM submissions
                WHERE user_id = ? AND status = 'published'
                GROUP BY destination
                """,
                (user_id,),
            ).fetchall()
        return {
            str(row["destination"]): (int(row["item_count"]), int(row["duration_ms"] or 0))
            for row in rows
        }

    def karma_summary(self, user_id: int) -> tuple[int, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(delta), 0) AS total, COUNT(*) AS event_count
                FROM karma_events
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return 0, 0
        return int(row["total"] or 0), int(row["event_count"] or 0)

    def karma_users(self) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT user_id FROM karma_events ORDER BY user_id"
            ).fetchall()
        return [int(row["user_id"]) for row in rows]

    def find_by_publication(self, chat_id: int, message_id: int) -> Submission | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM submissions WHERE publication_chat_id = ? AND publication_message_id = ?",
                (chat_id, message_id),
            ).fetchone()
        return _submission_from_row(row) if row is not None else None

    def clear_comment_posted(self, submission_id: int) -> None:
        """Give the claim back when nothing could actually be sent, so a later
        attempt is not blocked by a reservation that produced no comment."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE submissions SET comment_posted_at = NULL WHERE id = ?",
                (submission_id,),
            )

    def mark_comment_posted(self, submission_id: int) -> bool:
        """Claims the right to post the discussion-group comment for this submission.
        Returns False if it was already claimed - guards against duplicate/retried
        forward events triggering a second comment."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE submissions SET comment_posted_at = ? WHERE id = ? AND comment_posted_at IS NULL",
                (time.time(), submission_id),
            )
        return cursor.rowcount > 0

    def delayed_posting_enabled(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'delayed_posting_enabled'"
            ).fetchone()
        return row is not None and row["value"] == "1"

    def set_delayed_posting_enabled(self, enabled: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO store_metadata(key, value) VALUES ('delayed_posting_enabled', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("1" if enabled else "0",),
            )

    def next_available_slot(self, destination: str, *, interval_seconds: float, now: float | None = None) -> float:
        """The earliest time a new post to this destination may go out: `now`
        if enough time has passed since the last one, otherwise last + interval.
        "Last" covers both already-sent posts and ones still waiting in the
        queue, so a burst of clicks chains into consecutive slots instead of
        piling up on the same one."""
        now = now if now is not None else time.time()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT MAX(t) AS last_t FROM (
                    SELECT updated_at AS t FROM submissions
                        WHERE destination = ? AND publication_message_id IS NOT NULL
                    UNION ALL
                    SELECT scheduled_for AS t FROM scheduled_posts WHERE destination = ?
                )
                """,
                (destination, destination),
            ).fetchone()
        last_t = row["last_t"] if row is not None else None
        if last_t is None:
            return now
        return max(now, float(last_t) + interval_seconds)

    def pending_schedule_for_submission(self, submission_id: int) -> ScheduledPost | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scheduled_posts WHERE submission_id = ? AND status = 'pending' ORDER BY id DESC LIMIT 1",
                (submission_id,),
            ).fetchone()
        return _scheduled_post_from_row(row) if row is not None else None

    def schedule_post(
        self, *, submission_id: int, destination: str, target_chat: str, moderator_id: int, scheduled_for: float
    ) -> ScheduledPost:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scheduled_posts (
                    submission_id, destination, target_chat, moderator_id, scheduled_for, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (submission_id, destination, target_chat, moderator_id, scheduled_for, time.time()),
            )
            row = connection.execute(
                "SELECT * FROM scheduled_posts WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        if row is None:
            raise RuntimeError("Could not create scheduled post")
        return _scheduled_post_from_row(row)

    def pending_scheduled_posts(self) -> list[ScheduledPost]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scheduled_posts WHERE status = 'pending' ORDER BY scheduled_for ASC"
            ).fetchall()
        return [_scheduled_post_from_row(row) for row in rows]

    def due_scheduled_posts(self, now: float | None = None) -> list[ScheduledPost]:
        now = now if now is not None else time.time()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scheduled_posts WHERE status = 'pending' AND scheduled_for <= ? ORDER BY scheduled_for ASC",
                (now,),
            ).fetchall()
        return [_scheduled_post_from_row(row) for row in rows]

    def get_scheduled_post(self, scheduled_id: int) -> ScheduledPost | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM scheduled_posts WHERE id = ?", (scheduled_id,)).fetchone()
        return _scheduled_post_from_row(row) if row is not None else None

    def mark_scheduled_sent(self, scheduled_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE scheduled_posts SET status = 'sent', sent_at = ? WHERE id = ? AND status = 'pending'",
                (time.time(), scheduled_id),
            )

    def cancel_scheduled_post(self, scheduled_id: int) -> ScheduledPost | None:
        """Marks a still-pending scheduled post cancelled. Returns None (a
        no-op) if it was already sent or cancelled by someone else."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scheduled_posts WHERE id = ? AND status = 'pending'", (scheduled_id,)
            ).fetchone()
            if row is None:
                return None
            connection.execute("UPDATE scheduled_posts SET status = 'cancelled' WHERE id = ?", (scheduled_id,))
        return _scheduled_post_from_row(row)


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
        karma_milli=int(row["karma_award"] or 0),
        karma_before_milli=int(row["karma_before_milli"] or 0),
        duration_ms=int(row["duration_ms"] or 0),
        publication_chat_id=int(row["publication_chat_id"]) if row["publication_chat_id"] is not None else None,
        publication_message_id=(
            int(row["publication_message_id"]) if row["publication_message_id"] is not None else None
        ),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _scheduled_post_from_row(row: sqlite3.Row) -> ScheduledPost:
    return ScheduledPost(
        id=int(row["id"]),
        submission_id=int(row["submission_id"]),
        destination=str(row["destination"]),
        target_chat=str(row["target_chat"]),
        moderator_id=int(row["moderator_id"]),
        scheduled_for=float(row["scheduled_for"]),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        sent_at=float(row["sent_at"]) if row["sent_at"] is not None else None,
    )
