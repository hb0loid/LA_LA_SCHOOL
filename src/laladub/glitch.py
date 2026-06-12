from __future__ import annotations

import re

from .models import Segment


ARTIFACT_PATTERNS = [
    re.compile(r"^\s*(subtitles?|captions?)\s+(by|made by|created by|provided by)\b", re.I),
    re.compile(r"^\s*субтитры\s+(сделал|создал|добавил|предоставил)\b", re.I),
    re.compile(r"\bdimatorzok\b", re.I),
    re.compile(r"\bamara\.org\b", re.I),
]

GHOST_LINES_RU = [
    "Субтитры сделал DimaTorzok",
    "Субтитры создавал DimaTorzok",
    "Перевод и субтитры DimaTorzok",
]

GHOST_LINES_EN = [
    "Subtitles by DimaTorzok",
    "Captions created by DimaTorzok",
    "Translation and subtitles by DimaTorzok",
]


def clean_segments(segments: list[Segment]) -> list[Segment]:
    cleaned: list[Segment] = []
    for segment in segments:
        source = segment.text.strip()
        translated = (segment.translated_text or "").strip()
        if _is_artifact(source) or _is_artifact(translated):
            continue
        segment.text = source
        if segment.translated_text is not None:
            segment.translated_text = translated
        if segment.text or segment.translated_text:
            cleaned.append(segment)
    return cleaned


def apply_glitch_profile(
    segments: list[Segment],
    profile: str,
    target_lang: str,
    min_gap_seconds: float,
) -> list[Segment]:
    if profile == "clean":
        return clean_segments(segments)
    if profile == "ghost":
        return inject_ghosts(segments, target_lang=target_lang, min_gap_seconds=min_gap_seconds)
    if profile == "faithful":
        return segments
    raise ValueError(f"Unknown glitch profile: {profile}")


def inject_ghosts(segments: list[Segment], target_lang: str, min_gap_seconds: float) -> list[Segment]:
    if not segments:
        return segments

    ghost_lines = GHOST_LINES_RU if target_lang.lower().startswith("ru") else GHOST_LINES_EN
    result: list[Segment] = []
    ghost_index = 0

    for index, segment in enumerate(segments):
        result.append(segment)
        if index == len(segments) - 1:
            continue

        next_segment = segments[index + 1]
        gap = next_segment.start - segment.end
        if gap < min_gap_seconds:
            continue

        start = segment.end + min(0.35, gap / 4)
        end = min(next_segment.start - 0.15, start + min(2.2, gap - 0.3))
        if end <= start:
            continue

        line = ghost_lines[ghost_index % len(ghost_lines)]
        ghost_index += 1
        result.append(Segment(start=start, end=end, text=line, translated_text=line))

    return sorted(result, key=lambda item: (item.start, item.end))


def _is_artifact(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in ARTIFACT_PATTERNS)
