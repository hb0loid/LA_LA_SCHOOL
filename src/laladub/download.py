from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .ffmpeg import require_tool


class DownloadError(RuntimeError):
    pass


_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def extract_url(text: str) -> str | None:
    match = _URL_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip(").,!?;]")


def download_video_url(url: str, output_dir: Path, max_file_mb: int) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError("Это не похоже на корректную ссылку.")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = max_file_mb * 1024 * 1024
    cached_path = _restore_cached_download(url, output_dir, max_bytes)
    if cached_path is not None:
        print(f"Download cache hit: {url} -> {cached_path}", flush=True)
        return cached_path

    try:
        import yt_dlp
    except ImportError as exc:
        raise DownloadError("Для скачивания ссылок нужен yt-dlp: python -m pip install yt-dlp") from exc

    output_template = str(output_dir / "input.%(ext)s")
    options: dict[str, object] = {
        "format": "bv*[vcodec!=none][height<=720]+ba/b[vcodec!=none][height<=720]/bv*[vcodec!=none]+ba/b[vcodec!=none]",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": max_bytes,
        "windowsfilenames": True,
        "js_runtimes": {"node": {}, "deno": {}},
        "remote_components": ["ejs:github"],
    }
    browser_cookies = os.environ.get("LALADUB_YTDLP_BROWSER_COOKIES", "").strip()
    if browser_cookies:
        options["cookiesfrombrowser"] = (browser_cookies, None, None, None)

    errors: list[str] = []
    info = None
    video_path: Path | None = None
    for profile_name, profile_options in _download_profiles(parsed.netloc):
        for candidate in output_dir.glob("input.*"):
            if candidate.is_file():
                candidate.unlink(missing_ok=True)
        merged_options = _merge_options(options, profile_options)
        try:
            with yt_dlp.YoutubeDL(merged_options) as downloader:
                info = downloader.extract_info(url, download=True)
            video_path = _find_downloaded_video(output_dir)
            if video_path is not None:
                break
            errors.append(f"{profile_name}: скачивание завершилось, но итоговый файл не содержит видео и аудио")
        except Exception as exc:
            errors.append(f"{profile_name}: {exc}")
    if video_path is None:
        version = getattr(getattr(yt_dlp, "version", None), "__version__", "unknown")
        error_text = "; ".join(errors[-3:]) if errors else "unknown error"
        hint = ""
        if "tiktok" in parsed.netloc.lower():
            hint = (
                " TikTok часто ломает публичное скачивание; если ссылка всё ещё не качается, "
                "можно попробовать LALADUB_YTDLP_BROWSER_COOKIES=chrome для cookies из браузера."
            )
        raise DownloadError(
            f"Не получилось скачать видео по ссылке через yt-dlp {version}: {error_text}.{hint}"
        )

    if isinstance(info, dict):
        meta = {
            "title": info.get("title"),
            "id": info.get("id"),
            "extractor": info.get("extractor_key") or info.get("extractor"),
            "webpage_url": info.get("webpage_url") or url,
        }
        (output_dir / "download_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if video_path.stat().st_size > max_bytes:
        size_mb = video_path.stat().st_size / 1024 / 1024
        video_path.unlink(missing_ok=True)
        raise DownloadError(f"Видео получилось слишком большим: {size_mb:.1f} МБ. Лимит: {max_file_mb} МБ.")

    _store_cached_download(url, output_dir, video_path)
    return video_path


def _restore_cached_download(url: str, output_dir: Path, max_bytes: int) -> Path | None:
    cache_dir = _download_cache_entry(url)
    if cache_dir is None:
        return None
    cached_video = _find_downloaded_video(cache_dir)
    if cached_video is None:
        return None
    if cached_video.stat().st_size > max_bytes:
        return None

    for candidate in output_dir.glob("input.*"):
        if candidate.is_file():
            candidate.unlink(missing_ok=True)

    output_path = output_dir / f"input{cached_video.suffix.lower() or '.mp4'}"
    try:
        shutil.copy2(cached_video, output_path)
        cached_meta = cache_dir / "download_meta.json"
        if cached_meta.exists():
            meta = json.loads(cached_meta.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                meta["cache_hit"] = True
                meta["cache_key"] = cache_dir.name
                (output_dir / "download_meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    except Exception as exc:
        print(f"Download cache restore skipped: {type(exc).__name__}: {exc}", flush=True)
        return None

    return output_path if has_video_and_audio(output_path) else None


def _store_cached_download(url: str, output_dir: Path, video_path: Path) -> None:
    cache_dir = _download_cache_entry(url)
    if cache_dir is None or not has_video_and_audio(video_path):
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"input{video_path.suffix.lower() or '.mp4'}"
    if not output_path.exists():
        temp_path = output_path.with_name(output_path.name + ".tmp")
        try:
            shutil.copy2(video_path, temp_path)
            temp_path.replace(output_path)
        except Exception as exc:
            print(f"Download cache store skipped: {type(exc).__name__}: {exc}", flush=True)
            temp_path.unlink(missing_ok=True)
            return

    meta_path = output_dir / "download_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                meta["cache_key"] = cache_dir.name
                meta["normalized_url"] = _normalize_download_cache_url(url)
                (cache_dir / "download_meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass


def _download_cache_entry(url: str) -> Path | None:
    cache_root = os.environ.get("LALADUB_DOWNLOAD_CACHE_DIR", "runs/cache/downloads").strip()
    if not cache_root:
        return None
    root = Path(cache_root)
    key = hashlib.sha256(_normalize_download_cache_url(url).encode("utf-8", errors="ignore")).hexdigest()
    return root / key[:2] / key


def _normalize_download_cache_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or parsed.path
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "si", "feature"}
    ]

    if netloc in {"youtu.be", "www.youtu.be"}:
        video_id = path.strip("/").split("/", 1)[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    if netloc.endswith("youtube.com"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return f"https://www.youtube.com/watch?v={parts[1]}"
        video_id = next((value for key, value in query_items if key == "v" and value), "")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    query = urlencode(sorted(query_items))
    return urlunparse((scheme, netloc, path, "", query, ""))


def _find_downloaded_video(output_dir: Path) -> Path | None:
    candidates = [
        path
        for path in output_dir.glob("input.*")
        if path.suffix.lower() in _MEDIA_SUFFIXES and not path.name.endswith(".part")
    ]
    candidates = [path for path in candidates if has_video_and_audio(path)]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def has_video_and_audio(path: Path) -> bool:
    ffprobe = require_tool("ffprobe")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return False
    stream_types = {str(stream.get("codec_type") or "") for stream in streams}
    return "video" in stream_types and "audio" in stream_types


def _download_profiles(netloc: str) -> list[tuple[str, dict[str, object]]]:
    if "tiktok.com" not in netloc.lower():
        return [
            ("standard-720", {}),
            (
                "fallback-480",
                {
                    "format": (
                        "bv*[vcodec!=none][height<=480]+ba/"
                        "b[vcodec!=none][height<=480]/"
                        "bv*[vcodec!=none][height<=360]+ba/"
                        "b[vcodec!=none][height<=360]"
                    )
                },
            ),
            (
                "fallback-any-video",
                {"format": "b[vcodec!=none]/bv*[vcodec!=none]+ba"},
            ),
        ]

    return [
        ("tiktok-standard", {}),
        (
            "tiktok-api-alisg",
            {"extractor_args": {"tiktok": {"api_hostname": ["api16-normal-c-alisg.tiktokv.com"]}}},
        ),
        (
            "tiktok-api-useast",
            {"extractor_args": {"tiktok": {"api_hostname": ["api16-normal-c-useast1a.tiktokv.com"]}}},
        ),
        (
            "tiktok-mobile-format",
            {
                "format": "b[vcodec!=none]/bv*[vcodec!=none]+ba",
                "extractor_args": {"tiktok": {"api_hostname": ["api16-normal-c-useast1a.tiktokv.com"]}},
            },
        ),
    ]


def _merge_options(base: dict[str, object], extra: dict[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            nested = dict(result[key])  # type: ignore[arg-type]
            nested.update(value)
            result[key] = nested
        else:
            result[key] = value
    return result
