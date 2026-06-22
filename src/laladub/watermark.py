from __future__ import annotations

import random
from pathlib import Path

from .ffmpeg import require_tool, run


def add_watermark(
    input_path: Path,
    output_path: Path,
    *,
    text: str,
    image_path: Path | None = None,
) -> None:
    selected_image = _select_watermark_image(image_path)
    if selected_image is not None:
        print(f"      Watermark image: {selected_image}", flush=True)
        add_image_watermark(input_path, output_path, selected_image)
        return
    print("      Watermark image: none, using text fallback", flush=True)
    add_text_watermark(input_path, output_path, text)


def _select_watermark_image(image_path: Path | None) -> Path | None:
    if image_path is None or not image_path.exists():
        return None
    if image_path.is_file():
        return image_path if image_path.suffix.casefold() == ".png" else None

    candidates = sorted(
        path
        for path in image_path.iterdir()
        if path.is_file() and path.suffix.casefold() == ".png"
    )
    if not candidates:
        return None
    return random.choice(candidates)


def add_image_watermark(
    input_path: Path,
    output_path: Path,
    image_path: Path,
    *,
    width: int = 220,
    opacity: float = 0.88,
    margin: int = 24,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = max(0.0, min(1.0, opacity))
    filter_value = (
        f"[1:v]format=rgba,scale={width}:-1,colorchannelmixer=aa={alpha:.2f}[wm];"
        f"[0:v][wm]overlay=W-w-{margin}:H-h-{margin}:format=auto,format=yuv420p[v]"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(image_path),
            "-filter_complex",
            filter_value,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_path),
        ]
    )


def add_text_watermark(
    input_path: Path,
    output_path: Path,
    text: str,
    *,
    font_file: Path | None = None,
    font_size: int = 28,
    opacity: float = 0.82,
    margin: int = 24,
) -> None:
    """Add a small bottom-right text watermark without re-encoding audio."""
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    safe_text = _escape_drawtext(text)
    font_file = font_file or _find_default_font()
    font_part = f"fontfile='{_escape_drawtext(font_file.as_posix())}':" if font_file else ""
    filter_value = (
        "drawtext="
        f"{font_part}"
        f"text='{safe_text}':"
        f"x=w-tw-{margin}:"
        f"y=h-th-{margin}:"
        f"fontsize={font_size}:"
        f"fontcolor=white@{max(0.0, min(1.0, opacity)):.2f}:"
        "box=1:"
        "boxcolor=black@0.38:"
        "boxborderw=10"
    )

    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            filter_value,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_path),
        ]
    )


def _find_default_font() -> Path | None:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", " ")
        .replace("\r", " ")
    )
