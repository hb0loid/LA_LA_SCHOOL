from __future__ import annotations

from dataclasses import dataclass


KARMA_SCALE = 1000


@dataclass(frozen=True, slots=True)
class KarmaLevel:
    minimum: int
    name: str
    daily_minutes: int
    priority_bonus: int


KARMA_LEVELS = (
    KarmaLevel(0, "Новичок", 3, 0),
    KarmaLevel(5, "Участник", 5, 0),
    KarmaLevel(25, "Автор", 10, 1),
    KarmaLevel(75, "Проверенный автор", 15, 2),
    KarmaLevel(150, "Любимчик редакции", 20, 3),
    KarmaLevel(300, "Ветеран", 30, 4),
    KarmaLevel(500, "Легенда", 40, 5),
    KarmaLevel(1000, "Классика La La School", 50, 6),
)


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
