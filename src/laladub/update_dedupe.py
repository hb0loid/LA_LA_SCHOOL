"""Refuses to act on a Telegram update twice - without losing one.

Telegram redelivers any update whose offset was never confirmed. The bots are
stopped with a hard kill, so an update being handled at that moment is never
acknowledged and comes back on the next start: one /show call turned into
seventeen videos in a chat, one per restart.

Recording the id *before* handling would block those replays, but at a price
that turned out to be worse: an update killed mid-handling is then dropped for
good. For a video someone sent to be dubbed that means the job silently never
happens. So the id is recorded only once handling has finished, and a replay
of something that never finished is allowed through - repeating work is
recoverable, losing it is not.

Repeats that are actually expensive - re-sending the same video into the same
chat - are held back where the effect lives instead: see recent_action.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class UpdateDeduplicator:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_seen = self._load()

    def _load(self) -> int:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        try:
            return int(data.get("last_update_id") or 0)
        except (TypeError, ValueError):
            return 0

    def _save(self) -> None:
        # Written next to the target and moved into place, so a kill during the
        # write cannot leave a truncated file that reads back as "seen nothing".
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps({"last_update_id": self._last_seen}), encoding="utf-8")
        temp.replace(self.path)

    def is_new(self, update_id: int | None) -> bool:
        return update_id is None or int(update_id) > self._last_seen

    def mark_done(self, update_id: int | None) -> None:
        if update_id is None:
            return
        update_id = int(update_id)
        if update_id > self._last_seen:
            self._last_seen = update_id
            self._save()


def build_replay_guard(deduplicator: UpdateDeduplicator) -> Any:
    """A handler for group=-100 that stops updates already handled to the end."""

    async def guard(update: Any, context: Any) -> None:
        from telegram.ext import ApplicationHandlerStop

        update_id = getattr(update, "update_id", None)
        if deduplicator.is_new(update_id):
            return
        print(f"Ignoring replayed update {update_id}", flush=True)
        raise ApplicationHandlerStop

    return guard


def build_completion_marker(deduplicator: UpdateDeduplicator) -> Any:
    """A handler for the last group, reached only once every other handler has
    run, which is what makes the update count as handled."""

    async def mark(update: Any, context: Any) -> None:
        deduplicator.mark_done(getattr(update, "update_id", None))

    return mark


class RecentActions:
    """Remembers recent (key -> when) pairs across restarts.

    Used to hold back a repeat whose effect is expensive and visible - the same
    work re-sent into the same chat - independently of whether the update
    carrying it was new.
    """

    def __init__(self, path: Path, *, window_seconds: float = 600.0, keep: int = 200) -> None:
        self.path = path
        self.window_seconds = window_seconds
        self.keep = keep
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, float] = self._load()

    def _load(self) -> dict[str, float]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, float] = {}
        for key, when in data.items():
            try:
                out[str(key)] = float(when)
            except (TypeError, ValueError):
                continue
        return out

    def _save(self) -> None:
        if len(self._entries) > self.keep:
            newest = sorted(self._entries.items(), key=lambda item: item[1], reverse=True)
            self._entries = dict(newest[: self.keep])
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self._entries), encoding="utf-8")
        temp.replace(self.path)

    def seconds_since(self, key: str, *, now: float | None = None) -> float | None:
        when = self._entries.get(key)
        if when is None:
            return None
        elapsed = (time.time() if now is None else now) - when
        return elapsed if elapsed < self.window_seconds else None

    def record(self, key: str, *, now: float | None = None) -> None:
        self._entries[key] = time.time() if now is None else now
        self._save()
