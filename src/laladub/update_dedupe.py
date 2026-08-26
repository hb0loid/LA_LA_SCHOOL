"""Refuses to act on a Telegram update twice.

Telegram redelivers any update whose offset was never confirmed. The bot is
stopped with a hard kill, so an update being handled at that moment - a video
send can take minutes - is never acknowledged and comes back on the next
start. With drop_pending_updates=False that replay is silent and looks exactly
like a fresh command: one /show call turned into seventeen videos in a chat,
one per restart.

The id is recorded the moment the update arrives, before any handler runs, so
a kill mid-handling still blocks the replay. That trades a lost update on a
crash for never repeating a side effect - the right way round when the side
effects are posting videos and publishing to channels.
"""

from __future__ import annotations

import json
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
        if update_id is None:
            return True
        update_id = int(update_id)
        if update_id <= self._last_seen:
            return False
        self._last_seen = update_id
        self._save()
        return True


def build_replay_guard(deduplicator: UpdateDeduplicator) -> Any:
    """A handler for group=-100 that stops replayed updates before anything else."""

    async def guard(update: Any, context: Any) -> None:
        from telegram.ext import ApplicationHandlerStop

        if deduplicator.is_new(getattr(update, "update_id", None)):
            return
        print(f"Ignoring replayed update {getattr(update, 'update_id', None)}", flush=True)
        raise ApplicationHandlerStop

    return guard
