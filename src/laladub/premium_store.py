from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class UserSettings:
    user_id: int
    watermark_enabled: bool
    censor_percent: int | None


@dataclass(frozen=True, slots=True)
class Subscription:
    id: int
    user_id: int
    telegram_payment_charge_id: str
    stars_amount: int
    purchased_at: float
    expires_at: float
    status: str
    granted_by: str | None
    created_at: float


def _subscription_from_row(row: sqlite3.Row) -> Subscription:
    return Subscription(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        telegram_payment_charge_id=str(row["telegram_payment_charge_id"]),
        stars_amount=int(row["stars_amount"]),
        purchased_at=float(row["purchased_at"]),
        expires_at=float(row["expires_at"]),
        status=str(row["status"]),
        granted_by=row["granted_by"],
        created_at=float(row["created_at"]),
    )


class PremiumStore:
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
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    telegram_payment_charge_id TEXT NOT NULL UNIQUE,
                    stars_amount INTEGER NOT NULL,
                    purchased_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    granted_by TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_subscriptions_user
                    ON subscriptions(user_id, expires_at);

                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    watermark_enabled INTEGER NOT NULL DEFAULT 1,
                    censor_percent INTEGER,
                    updated_at REAL NOT NULL
                );
                """
            )

    def _active_expiry(self, connection: sqlite3.Connection, user_id: int, now: float) -> float:
        row = connection.execute(
            """
            SELECT MAX(expires_at) AS expiry FROM subscriptions
            WHERE user_id = ? AND status = 'active' AND expires_at > ?
            """,
            (user_id, now),
        ).fetchone()
        expiry = row["expiry"] if row is not None else None
        return float(expiry) if expiry is not None else now

    def record_payment(
        self,
        *,
        user_id: int,
        telegram_payment_charge_id: str,
        stars_amount: int,
        days: int,
    ) -> Subscription | None:
        """Extends the user's active subscription by `days`. Returns None (a no-op) if this
        charge_id was already recorded - Telegram may redeliver the successful_payment update."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id FROM subscriptions WHERE telegram_payment_charge_id = ?",
                (telegram_payment_charge_id,),
            ).fetchone()
            if existing is not None:
                return None
            start_from = max(now, self._active_expiry(connection, user_id, now))
            expires_at = start_from + days * 86400.0
            cursor = connection.execute(
                """
                INSERT INTO subscriptions (
                    user_id, telegram_payment_charge_id, stars_amount,
                    purchased_at, expires_at, status, granted_by, created_at
                ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?)
                """,
                (user_id, telegram_payment_charge_id, stars_amount, now, expires_at, now),
            )
            row = connection.execute(
                "SELECT * FROM subscriptions WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return _subscription_from_row(row) if row is not None else None

    def grant_manual(self, *, user_id: int, days: int, admin_id: int) -> Subscription:
        now = time.time()
        charge_id = f"manual-{admin_id}-{now}"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            start_from = max(now, self._active_expiry(connection, user_id, now))
            expires_at = start_from + days * 86400.0
            cursor = connection.execute(
                """
                INSERT INTO subscriptions (
                    user_id, telegram_payment_charge_id, stars_amount,
                    purchased_at, expires_at, status, granted_by, created_at
                ) VALUES (?, ?, 0, ?, ?, 'active', ?, ?)
                """,
                (user_id, charge_id, now, expires_at, str(admin_id), now),
            )
            row = connection.execute(
                "SELECT * FROM subscriptions WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        if row is None:
            raise RuntimeError("Failed to grant manual subscription")
        return _subscription_from_row(row)

    def active_subscription(self, user_id: int) -> Subscription | None:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM subscriptions
                WHERE user_id = ? AND status = 'active' AND expires_at > ?
                ORDER BY expires_at DESC LIMIT 1
                """,
                (user_id, now),
            ).fetchone()
        return _subscription_from_row(row) if row is not None else None

    def latest_charge_id(self, user_id: int) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT telegram_payment_charge_id FROM subscriptions
                WHERE user_id = ? AND status = 'active'
                ORDER BY expires_at DESC LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return str(row["telegram_payment_charge_id"]) if row is not None else None

    def revoke(self, user_id: int, *, status: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE subscriptions SET status = ? WHERE user_id = ? AND status = 'active'",
                (status, user_id),
            )
        return cursor.rowcount > 0

    def get_user_settings(self, user_id: int) -> UserSettings:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT watermark_enabled, censor_percent FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return UserSettings(user_id=user_id, watermark_enabled=True, censor_percent=None)
        censor_percent = row["censor_percent"]
        return UserSettings(
            user_id=user_id,
            watermark_enabled=bool(row["watermark_enabled"]),
            censor_percent=int(censor_percent) if censor_percent is not None else None,
        )

    def set_watermark_enabled(self, user_id: int, enabled: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_settings (user_id, watermark_enabled, censor_percent, updated_at)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(user_id) DO UPDATE SET watermark_enabled = excluded.watermark_enabled,
                    updated_at = excluded.updated_at
                """,
                (user_id, 1 if enabled else 0, time.time()),
            )

    def set_censor_percent(self, user_id: int, percent: int | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_settings (user_id, watermark_enabled, censor_percent, updated_at)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET censor_percent = excluded.censor_percent,
                    updated_at = excluded.updated_at
                """,
                (user_id, percent, time.time()),
            )
