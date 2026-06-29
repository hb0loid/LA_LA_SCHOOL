from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .ffmpeg import probe_duration, require_tool


_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}

_UNSAFE_LABEL_HINTS = (
    "nsfw",
    "unsafe",
    "porn",
    "hentai",
    "sexy",
    "sexual",
    "nude",
    "nudity",
)
_SAFE_LABEL_HINTS = ("safe", "sfw", "normal")


def is_video_safe_for_visual_source(
    path: Path,
    *,
    cache_dir: Path,
    temp_dir: Path,
    model_name: str = "Falconsai/nsfw_image_detection",
    threshold: float = 0.72,
    frame_count: int = 5,
    device: str = "cpu",
) -> bool:
    """Return True only when sampled frames look safe enough for random visual reuse."""
    threshold = max(0.01, min(0.99, float(threshold)))
    frame_count = max(1, min(12, int(frame_count)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{_cache_key(path, model_name, frame_count)}.json"
    cached = _read_cached_score(cache_path, path, model_name, frame_count)
    if cached is not None:
        return cached < threshold

    score = 1.0
    status = "scan_failed"
    frame_paths: list[Path] = []
    try:
        frame_paths = _extract_sample_frames(path, temp_dir / _cache_key(path, "frames", frame_count), frame_count)
        if not frame_paths:
            status = "no_frames"
        else:
            classifier = _load_classifier(model_name, device)
            score = max(_unsafe_score(classifier(str(frame_path))) for frame_path in frame_paths)
            status = "scanned"
    except Exception as exc:
        print(f"      Visual safety scan failed for {path}: {type(exc).__name__}: {exc}", flush=True)
        score = 1.0
        status = f"error:{type(exc).__name__}"
    finally:
        for frame_path in frame_paths:
            frame_path.unlink(missing_ok=True)
        if frame_paths:
            shutil.rmtree(frame_paths[0].parent, ignore_errors=True)

    _write_cached_score(cache_path, path, model_name, frame_count, score, status)
    if score >= threshold:
        print(f"      Visual source rejected by safety scan: score={score:.3f} path={path}", flush=True)
    return score < threshold


def _load_classifier(model_name: str, device: str) -> Any:
    key = (model_name, device)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    from transformers import pipeline

    device_arg = -1
    normalized_device = (device or "cpu").strip().lower()
    if normalized_device in {"cuda", "gpu", "0"}:
        device_arg = 0
    elif normalized_device == "auto":
        try:
            import torch

            device_arg = 0 if torch.cuda.is_available() else -1
        except Exception:
            device_arg = -1

    classifier = pipeline("image-classification", model=model_name, device=device_arg)
    _PIPELINE_CACHE[key] = classifier
    return classifier


def _unsafe_score(result: Any) -> float:
    if isinstance(result, dict):
        items = [result]
    elif isinstance(result, list):
        items = [item for item in result if isinstance(item, dict)]
    else:
        return 1.0

    unsafe_scores: list[float] = []
    safe_scores: list[float] = []
    for item in items:
        label = str(item.get("label") or "").strip().lower()
        try:
            score = float(item.get("score") or 0.0)
        except Exception:
            score = 0.0
        if any(hint in label for hint in _UNSAFE_LABEL_HINTS):
            unsafe_scores.append(score)
        if any(hint in label for hint in _SAFE_LABEL_HINTS):
            safe_scores.append(score)

    if unsafe_scores:
        return max(unsafe_scores)
    if safe_scores:
        return 1.0 - max(safe_scores)
    return max((float(item.get("score") or 0.0) for item in items), default=1.0)


def _extract_sample_frames(video_path: Path, output_dir: Path, frame_count: int) -> list[Path]:
    ffmpeg = require_tool("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        duration = max(0.1, probe_duration(video_path))
    except Exception:
        duration = 0.0

    if duration <= 0:
        timestamps = [0.0]
    else:
        timestamps = [duration * (index + 1) / (frame_count + 1) for index in range(frame_count)]
        if duration > 1.0:
            timestamps[0] = min(0.5, max(0.0, duration - 0.1))
            timestamps[-1] = max(0.0, duration - 0.5)
    unique_timestamps = []
    for timestamp in timestamps:
        rounded = round(max(0.0, timestamp), 2)
        if rounded not in unique_timestamps:
            unique_timestamps.append(rounded)

    frames: list[Path] = []
    for index, timestamp in enumerate(unique_timestamps):
        frame_path = output_dir / f"frame_{index:02d}.jpg"
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    f"{timestamp:.2f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=384:384:force_original_aspect_ratio=increase,crop=384:384",
                    "-q:v",
                    "3",
                    str(frame_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            frame_path.unlink(missing_ok=True)
            continue
        if frame_path.exists() and frame_path.stat().st_size > 512:
            frames.append(frame_path)
    return frames


def _cache_key(path: Path, model_name: str, frame_count: int) -> str:
    resolved = str(path.resolve()).lower()
    raw = f"{resolved}|{model_name}|frames={frame_count}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:32]


def _read_cached_score(cache_path: Path, path: Path, model_name: str, frame_count: int) -> float | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        stat = path.stat()
        if data.get("path") != str(path.resolve()):
            return None
        if int(data.get("size") or -1) != int(stat.st_size):
            return None
        if int(data.get("mtime_ns") or -1) != int(stat.st_mtime_ns):
            return None
        if data.get("model") != model_name:
            return None
        if int(data.get("frame_count") or -1) != int(frame_count):
            return None
        return float(data.get("unsafe_score"))
    except Exception:
        return None


def _write_cached_score(
    cache_path: Path,
    path: Path,
    model_name: str,
    frame_count: int,
    unsafe_score: float,
    status: str,
) -> None:
    try:
        stat = path.stat()
        data = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "model": model_name,
            "frame_count": frame_count,
            "unsafe_score": float(unsafe_score),
            "status": status,
            "scanned_at": time.time(),
        }
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return
