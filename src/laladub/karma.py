from __future__ import annotations

from dataclasses import dataclass


KARMA_SCALE = 1000


@dataclass(frozen=True, slots=True)
class KarmaLevel:
    minimum: int
    name: str
    daily_minutes: int
    priority_bonus: int
    queue_limit: int


KARMA_LEVELS = (
    KarmaLevel(0, "Участник", 1, 0, 1),
    KarmaLevel(6, "Автор", 5, 1, 1),
    KarmaLevel(60, "Проверенный автор", 10, 2, 2),
    KarmaLevel(360, "Любимчик редакции", 15, 3, 2),
    KarmaLevel(720, "Ветеран", 20, 4, 3),
    KarmaLevel(1440, "Алмазный подписчик", 25, 5, 3),
    KarmaLevel(2880, "Легенда La La School", 30, 6, 3),
)

# Granted by an active Telegram Stars subscription, independent of the karma
# ladder above - `minimum` is unused here since it's applied by subscription
# state rather than a karma threshold, but keeping the same dataclass shape
# lets every existing karma-level consumer accept it without a new branch.
PREMIUM_LEVEL = KarmaLevel(0, "Премиум подписка", 60, 10, 5)


def karma_milli_for_duration(duration_ms: int, destination: str) -> int:
    """Return thousandths of karma for a published video.

    Main channel awards one point per 10 seconds; the shame channel awards
    one point per 50 seconds. Millisecond input keeps sub-second durations.
    """
    duration_ms = max(0, int(duration_ms))
    divisor = 10 if destination == "main" else 50 if destination == "shame" else 0
    if not divisor:
        return 0
    return max(0, (duration_ms + divisor // 2) // divisor)


def visible_karma(karma_milli: int) -> int:
    return max(0, int(karma_milli)) // KARMA_SCALE


def level_for_karma(karma_milli: int) -> KarmaLevel:
    visible = visible_karma(karma_milli)
    current = KARMA_LEVELS[0]
    for level in KARMA_LEVELS:
        if visible < level.minimum:
            break
        current = level
    return current


def next_level_for_karma(karma_milli: int) -> KarmaLevel | None:
    visible = visible_karma(karma_milli)
    for level in KARMA_LEVELS:
        if visible < level.minimum:
            return level
    return None


def format_karma_milli(karma_milli: int, *, signed: bool = False) -> str:
    value = int(karma_milli) / KARMA_SCALE
    text = f"{abs(value):.3f}".rstrip("0").rstrip(".").replace(".", ",")
    if value < 0:
        return f"-{text}"
    if signed and value > 0:
        return f"+{text}"
    return text
