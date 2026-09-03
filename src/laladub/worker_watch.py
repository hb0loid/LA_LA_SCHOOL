"""Tells the admins when the laptop worker disappears, and when it comes back.

The bot already knew: workers check in, and anything past the TTL counts as
offline. But nothing said so out loud - the status only appeared in /queue, if
somebody thought to look. Across 76 restarts of the bot the worker was absent
for 15 of them, and every time it was noticed by accident, hours later. It
carries 43% of the jobs, so those hours are not free.

State is kept on disk so restarting the bot does not re-announce an absence it
has already reported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# A worker restarting, or one slow moment on the network, should not raise an
# alarm - only an absence that lasts should.
DEFAULT_GRACE_SECONDS = 180.0

OFFLINE_MESSAGE = (
    "⚠️ Воркер не выходит на связь {minutes} мин.\n"
    "Задачи считаются только на основном ПК — это медленнее.\n"
    "На ноутбуке: запустить Start-Worker.cmd."
)
ONLINE_MESSAGE = "📡 Воркер снова на связи."


class WorkerPresence:
    def __init__(self, path: Path, *, grace_seconds: float = DEFAULT_GRACE_SECONDS) -> None:
        self.path = path
        self.grace_seconds = grace_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = self._load()
        self.online: bool = bool(raw.get("online", True))
        self.missing_since: float = float(raw.get("missing_since") or 0.0)
        self.reported_offline: bool = bool(raw.get("reported_offline"))

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self) -> None:
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(
                {
                    "online": self.online,
                    "missing_since": self.missing_since,
                    "reported_offline": self.reported_offline,
                }
            ),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def observe(self, online_count: int, *, now: float | None = None) -> str | None:
        """Called on a timer with how many workers are currently online.

        Returns the message to send, or None when there is nothing to say.
        """
        now = time.time() if now is None else now
        is_online = online_count > 0

        if is_online:
            # Only worth announcing a return if the absence was announced.
            message = ONLINE_MESSAGE if self.reported_offline else None
            self.online = True
            self.missing_since = 0.0
            self.reported_offline = False
            self._save()
            return message

        if self.online:
            # First tick of an absence: start the clock, say nothing yet.
            self.online = False
            self.missing_since = now
            self.reported_offline = False
            self._save()
            return None

        if self.reported_offline or now - self.missing_since < self.grace_seconds:
            return None

        self.reported_offline = True
        self._save()
        minutes = max(1, round((now - self.missing_since) / 60))
        return OFFLINE_MESSAGE.format(minutes=minutes)
