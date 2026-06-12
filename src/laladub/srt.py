from __future__ import annotations

from pathlib import Path

from .models import Segment


def read_srt(path: Path, translated: bool) -> list[Segment]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]
    segments: list[Segment] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_text, end_text = [item.strip() for item in lines[1].split("-->", 1)]
        body = " ".join(lines[2:]).strip()
        if not body:
            continue
        start = _parse_ts(start_text)
        end = _parse_ts(end_text)
        if translated:
            segments.append(Segment(start=start, end=end, text=body, translated_text=body))
        else:
            segments.append(Segment(start=start, end=end, text=body))
    return segments


def write_srt(path: Path, segments: list[Segment], translated: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    counter = 1
    for segment in segments:
        text = (segment.translated_text if translated else segment.text) or ""
        text = text.strip()
        if not text:
            continue
        rows.append(str(counter))
        rows.append(f"{_format_ts(segment.start)} --> {_format_ts(segment.end)}")
        rows.append(text)
        rows.append("")
        counter += 1
    path.write_text("\n".join(rows), encoding="utf-8")


def write_txt(path: Path, segments: list[Segment], translated: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for segment in segments:
        text = (segment.translated_text if translated else segment.text) or ""
        text = text.strip()
        if text:
            lines.append(text)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _format_ts(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_ts(value: str) -> float:
    hours_text, minutes_text, rest = value.strip().split(":", 2)
    seconds_text, millis_text = rest.split(",", 1)
    return (
        int(hours_text) * 3600
        + int(minutes_text) * 60
        + int(seconds_text)
        + int(millis_text[:3].ljust(3, "0")) / 1000.0
    )
