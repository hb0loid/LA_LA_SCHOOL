from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse


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

    try:
        import yt_dlp
    except ImportError as exc:
        raise DownloadError("Для скачивания ссылок нужен yt-dlp: python -m pip install yt-dlp") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = max_file_mb * 1024 * 1024
    output_template = str(output_dir / "input.%(ext)s")
    options: dict[str, object] = {
        "format": "bv*+ba/b",
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
    for profile_name, profile_options in _download_profiles(parsed.netloc):
        for candidate in output_dir.glob("input.*"):
            if candidate.is_file():
                candidate.unlink(missing_ok=True)
        merged_options = _merge_options(options, profile_options)
        try:
            with yt_dlp.YoutubeDL(merged_options) as downloader:
                info = downloader.extract_info(url, download=True)
            break
        except Exception as exc:
            errors.append(f"{profile_name}: {exc}")
    else:
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

    candidates = [
        path
        for path in output_dir.glob("input.*")
        if path.suffix.lower() in _MEDIA_SUFFIXES and not path.name.endswith(".part")
    ]
    if not candidates:
        raise DownloadError("Скачивание завершилось, но видеофайл не найден.")

    video_path = max(candidates, key=lambda path: path.stat().st_mtime)
    if video_path.stat().st_size > max_bytes:
        size_mb = video_path.stat().st_size / 1024 / 1024
        video_path.unlink(missing_ok=True)
        raise DownloadError(f"Видео получилось слишком большим: {size_mb:.1f} МБ. Лимит: {max_file_mb} МБ.")

    return video_path


def _download_profiles(netloc: str) -> list[tuple[str, dict[str, object]]]:
    if "tiktok.com" not in netloc.lower():
        return [("standard", {})]

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
                "format": "b/bv*+ba",
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
