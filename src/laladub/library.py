from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Shared between the main dub bot and the proposal bot - both point at the
# same files by default so /show works no matter which one receives it.

# /show is open to any user and now works in group chats too, so it needs its
# own spam guard. Per-process and in-memory is enough: the two bots run as
# separate processes anyway, so a shared cooldown would need its own storage
# for no real benefit - this already stops the repeated-tap pattern it's for.
_SHOW_COOLDOWN_SECONDS = 30.0
_last_show_call: dict[int, float] = {}


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    job_number: str
    user_id: int
    source_title: str
    target_lang: str
    video_path: str
    output_filename: str
    created_at: float


class LibraryStore:
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
                CREATE TABLE IF NOT EXISTS library (
                    job_number TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    source_title TEXT NOT NULL DEFAULT '',
                    target_lang TEXT NOT NULL DEFAULT '',
                    video_path TEXT NOT NULL,
                    output_filename TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def add(
        self,
        *,
        job_number: str,
        user_id: int,
        source_title: str,
        target_lang: str,
        video_path: str,
        output_filename: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO library (
                    job_number, user_id, source_title, target_lang, video_path, output_filename, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_number) DO UPDATE SET
                    video_path = excluded.video_path,
                    output_filename = excluded.output_filename,
                    source_title = excluded.source_title,
                    target_lang = excluded.target_lang
                """,
                (job_number, user_id, source_title, target_lang, video_path, output_filename, time.time()),
            )

    def get(self, job_number: str) -> LibraryEntry | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM library WHERE job_number = ?", (job_number,)).fetchone()
        if row is None:
            return None
        return LibraryEntry(
            job_number=str(row["job_number"]),
            user_id=int(row["user_id"]),
            source_title=str(row["source_title"] or ""),
            target_lang=str(row["target_lang"] or ""),
            video_path=str(row["video_path"]),
            output_filename=str(row["output_filename"]),
            created_at=float(row["created_at"]),
        )


async def show_command(update: Any, context: Any) -> None:
    library_store: LibraryStore | None = context.application.bot_data.get("library_store")
    if library_store is None:
        return

    user = update.effective_user
    user_id = int(user.id) if user is not None else None
    if user_id is not None:
        now = time.time()
        elapsed = now - _last_show_call.get(user_id, 0.0)
        if elapsed < _SHOW_COOLDOWN_SECONDS:
            await update.effective_message.reply_text(
                f"Слишком часто — подожди ещё {round(_SHOW_COOLDOWN_SECONDS - elapsed)} сек."
            )
            return
        _last_show_call[user_id] = now

    args = context.args or []
    job_number = str(args[0]).strip() if args else ""
    if not job_number:
        await update.effective_message.reply_text("Использование: /show номер_работы")
        return

    entry = await asyncio.to_thread(library_store.get, job_number)
    if entry is None:
        await update.effective_message.reply_text(f"Работа №{job_number} не найдена в библиотеке.")
        return

    video_path = Path(entry.video_path)
    if not video_path.is_file():
        await update.effective_message.reply_text(f"Файл работы №{job_number} утерян.")
        return

    from .bot import _telegram_sendable_video_path, video_upload_metadata

    send_path = await _telegram_sendable_video_path(video_path)
    metadata = await video_upload_metadata(send_path)
    caption = f"Работа №{entry.job_number}"
    if entry.source_title:
        caption += f" — {entry.source_title}"
    with send_path.open("rb") as file_obj:
        await context.bot.send_video(
            chat_id=update.effective_chat.id,
            video=file_obj,
            filename=entry.output_filename,
            caption=caption,
            supports_streaming=True,
            read_timeout=300,
            write_timeout=300,
            connect_timeout=60,
            pool_timeout=60,
            **metadata,
        )
