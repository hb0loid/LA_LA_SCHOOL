from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# NULL in every column means "ask every time" - both for a field the user
# explicitly left on ask in the /preset wizard, and for a field they never
# reached (no row, or a partially-filled row from an interrupted wizard).
PRESET_FIELDS = ("visual_mode", "source_lang", "speaker_count", "target_lang", "tts_provider", "review_mode")


@dataclass(frozen=True, slots=True)
class UserPreset:
    user_id: int
    visual_mode: str | None
    source_lang: str | None
    speaker_count: str | None
    target_lang: str | None
    tts_provider: str | None
    review_mode: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {field: getattr(self, field) for field in PRESET_FIELDS}


def _empty_preset(user_id: int) -> UserPreset:
    return UserPreset(
        user_id=user_id, visual_mode=None, source_lang=None, speaker_count=None,
        target_lang=None, tts_provider=None, review_mode=None,
    )


class PresetStore:
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
                CREATE TABLE IF NOT EXISTS presets (
                    user_id INTEGER PRIMARY KEY,
                    visual_mode TEXT,
                    source_lang TEXT,
                    speaker_count TEXT,
                    target_lang TEXT,
                    tts_provider TEXT,
                    review_mode TEXT
                );
                """
            )

    def get_preset(self, user_id: int) -> UserPreset:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM presets WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return _empty_preset(user_id)
        return UserPreset(user_id=user_id, **{field: row[field] for field in PRESET_FIELDS})

    def set_preset_field(self, user_id: int, field: str, value: str | None) -> None:
        if field not in PRESET_FIELDS:
            raise ValueError(field)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO presets (user_id, {field}) VALUES (?, ?) "
                f"ON CONFLICT(user_id) DO UPDATE SET {field} = excluded.{field}",
                (user_id, value),
            )

    def clear_preset(self, user_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM presets WHERE user_id = ?", (user_id,))
