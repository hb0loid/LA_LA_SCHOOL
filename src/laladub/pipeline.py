from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
from typing import Any
import wave

from .asr import clear_openai_whisper_cache, transcribe
from .censor import apply_censor_to_segments
from .ffmpeg import (
    combine_video_audio,
    combine_video_audio_multitrack,
    concat_wavs,
    delayed_mix,
    extract_audio,
    extract_audio_track,
    extract_wav_slice,
    fit_wav_to_duration,
    make_whisper_chaos_audio,
    normalize_wav,
    prepare_voice_reference,
    probe_duration,
)
from .glitch import apply_glitch_profile, clean_segments
from .models import DubConfig, Segment
from .quality import (
    clamp_obvious_word_repeats_in_segments,
    collapse_repetitions,
    collapse_repetitions_in_segments,
    limit_repeated_segment_phrases,
    suppress_pathological_segment_loops,
    is_repetitive_loop,
)
from .separation import SeparationResult, separate_audio
from .srt import read_srt, write_srt, write_txt
from .translation import translate_segments, translate_text_chain
from .tts import synthesize_cosyvoice_batch, synthesize_moss_batch, synthesize_qwen3_batch, synthesize_segment


_DEFAULT_META_HALLUCINATION_TERMS = [
    "subtitle",
    "subtitles",
    "caption",
    "captions",
    "amara.org",
    "solanya",
    "dimatorzok",
    "dima torzok",
    "игорь негода",
    "с вами был игорь негода",
    "девушки отдыхают",
    "девушка отдыхает",
    "девочки отдыхают",
    "các cô gái thư giãn",
    "subscribe",
    "subscribed",
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "description",
    "\u0441\u0443\u0431\u0442\u0438\u0442\u0440",
    "\u043f\u043e\u0434\u043f\u0438\u0441",
    "\u0441\u043f\u0430\u0441\u0438\u0431\u043e \u0437\u0430 \u043f\u0440\u043e\u0441\u043c\u043e\u0442\u0440",
    "\u0441\u043f\u0430\u0441\u0438\u0431\u043e \u0437\u0430 \u0441\u0443\u0431\u0442\u0438\u0442\u0440",
    "\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0435\u043d\u0438\u0435 \u0441\u043b\u0435\u0434\u0443\u0435\u0442",
    "\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440 \u0441\u0443\u0431\u0442\u0438\u0442\u0440",
    "\u043a\u043e\u0440\u0440\u0435\u043a\u0442\u043e\u0440",
    "\u043f\u043e\u0434\u043e\u0433\u043d\u0430\u043b",
    "\u0434\u0443\u0431\u0440\u043e\u0432\u0441\u043a",
    "\u0441\u0435\u043c\u043a\u0438\u043d",
    "\u0435\u0433\u043e\u0440\u043e\u0432",
    "\u043b\u0430\u0439\u043a",
    "\u043e\u043f\u0438\u0441\u0430\u043d\u0438",
    "ph\u1ee5 \u0111\u1ec1",
    "\u0111\u0103ng k\u00fd",
    "c\u1ea3m \u01a1n",
    "c\u00e1m \u01a1n",
    "\u0e04\u0e33\u0e1a\u0e23\u0e23\u0e22\u0e32\u0e22",
    "\u0e15\u0e34\u0e14\u0e15\u0e32\u0e21",
    "\u0e02\u0e2d\u0e1a\u0e04\u0e38\u0e13",
    "\u5b57\u5e55",
    "\u30c1\u30e3\u30f3\u30cd\u30eb\u767b\u9332",
    "\u3054\u8996\u8074",
    "\uc790\ub9c9",
    "\uad6c\ub3c5",
    "\uac10\uc0ac",
    "\u8ba2\u9605",
    "\u611f\u8c22\u89c2\u770b",
    "sous-titres",
    "abonnez",
    "subt\u00edtulos",
    "suscr",
    "untertitel",
    "abonnieren",
]


def _literal_regex(term: str) -> str:
    return re.escape(term).replace(r"\ ", r"\s+")


_DEFAULT_RU_ARTIFACT_PROMPT_SEEDS = [
    "Субтитры сделал DimaTorzok.",
    "Субтитры создавал DimaTorzok.",
    "Перевод и субтитры DimaTorzok.",
    "С вами был Игорь Негода.",
    "Субтитры подогнал Игорь Негода.",
    "Девушки отдыхают.",
    "Девушка отдыхает.",
    "Девочки отдыхают.",
    "Спасибо за субтитры Алексею Дубровскому!",
    "Субтитры подогнал Симон.",
    "Продолжение следует...",
    "Редактор субтитров А.Семкин. Корректор А.Егорова.",
    "Subtitles by DimaTorzok.",
    "Captions created by DimaTorzok.",
    "amara.org.",
]

_ARTIFACT_TEXT_CONFIG_CACHE: dict[str, tuple[tuple[str | None, float | None], Any]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_text_file_path(filename: str, env_name: str) -> Path:
    configured = os.environ.get(env_name, "").strip()
    if configured:
        return Path(configured)
    return _repo_root() / "assets" / filename


def _read_artifact_text_file(filename: str, env_name: str, defaults: list[str]) -> list[str]:
    path = _artifact_text_file_path(filename, env_name)
    try:
        stat = path.stat()
        stamp: tuple[str | None, float | None] = (str(path), stat.st_mtime)
    except OSError:
        stamp = (str(path), None)

    cache_key = f"{env_name}:{filename}"
    cached = _ARTIFACT_TEXT_CONFIG_CACHE.get(cache_key)
    if cached and cached[0] == stamp:
        return cached[1]

    lines: list[str] = []
    if stamp[1] is not None:
        try:
            for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(line)
        except Exception as exc:
            print(f"      Artifact text config ignored {path}: {type(exc).__name__}: {exc}")
            lines = []

    source = lines or defaults
    result: list[str] = []
    seen: set[str] = set()
    for item in source:
        normalized = item.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    _ARTIFACT_TEXT_CONFIG_CACHE[cache_key] = (stamp, result)
    return result


def _meta_hallucination_terms() -> list[str]:
    return _read_artifact_text_file(
        "artifact_triggers.txt",
        "LALADUB_ARTIFACT_TRIGGERS_FILE",
        _DEFAULT_META_HALLUCINATION_TERMS,
    )


def _meta_hallucination_re() -> re.Pattern[str]:
    terms = _meta_hallucination_terms()
    stamp = ("|".join(terms), float(len(terms)))
    cached = _ARTIFACT_TEXT_CONFIG_CACHE.get("compiled-meta-re")
    if cached and cached[0] == stamp:
        return cached[1]
    pattern = "(" + "|".join(_literal_regex(term) for term in terms) + ")"
    compiled = re.compile(pattern, re.IGNORECASE)
    _ARTIFACT_TEXT_CONFIG_CACHE["compiled-meta-re"] = (stamp, compiled)
    return compiled


def _ru_artifact_prompt_seeds() -> list[str]:
    return _read_artifact_text_file(
        "artifact_prompt_seeds_ru.txt",
        "LALADUB_ARTIFACT_PROMPT_SEEDS_RU_FILE",
        _DEFAULT_RU_ARTIFACT_PROMPT_SEEDS,
    )


def _legacy_artifact_inserts_ru() -> list[str]:
    return _read_artifact_text_file(
        "artifact_legacy_inserts_ru.txt",
        "LALADUB_ARTIFACT_LEGACY_INSERTS_RU_FILE",
        _DEFAULT_RU_ARTIFACT_PROMPT_SEEDS,
    )


_ROOT_DEBUG_SRT_FILES = [
    "artifact_injected.srt",
    "artifact_source.srt",
    "artifact_source_raw.srt",
    "artifact_translated.srt",
    "artifact_whisper_translated.srt",
    "artifact_whisper_translated_pivot.srt",
    "translated_clean.srt",
    "translated_pivot.srt",
    "translated_raw.srt",
]

_ROOT_OUTPUT_PATTERNS = [
    "forced_*_transcript.srt",
    "input_*.srt",
]


@dataclass(slots=True)
class _SpeakerCandidate:
    segment_index: int
    segment: Segment
    ref_path: Path
    raw_ref_path: Path
    embedding: object
    quality: float
    midpoint: float
    cluster_id: int | None = None


def _report_progress(
    config: DubConfig,
    stage: str,
    current: int | None = None,
    total: int | None = None,
    detail: str | None = None,
) -> None:
    if config.progress_callback is None:
        return
    try:
        config.progress_callback(stage, current, total, detail)
    except Exception as exc:
        print(f"      Progress callback skipped: {type(exc).__name__}: {exc}")


def _format_video_position(position: float, duration: float) -> str:
    def stamp(seconds: float) -> str:
        total = max(0, round(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

    ratio = max(0.0, min(1.0, position / max(0.001, duration)))
    filled = round(12 * ratio)
    bar = "[" + "#" * filled + "-" * (12 - filled) + "]"
    return f"видео {bar} {stamp(position)} / {stamp(duration)}"


def _prepare_workdir_outputs(config: DubConfig) -> None:
    for filename in _ROOT_DEBUG_SRT_FILES:
        (config.workdir / filename).unlink(missing_ok=True)
    for pattern in _ROOT_OUTPUT_PATTERNS:
        for path in config.workdir.glob(pattern):
            path.unlink(missing_ok=True)


def _clear_downstream_dub_outputs(config: DubConfig) -> None:
    removed = 0
    for directory_name in ("tts_raw", "tts_fit"):
        directory = config.workdir / directory_name
        if not directory.exists():
            continue
        for path in directory.glob("*.wav"):
            path.unlink(missing_ok=True)
            removed += 1
    for filename in ("dub_track.wav",):
        path = config.workdir / filename
        if path.exists():
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"      Cleared downstream dub outputs after transcript change: {removed}")


def _asr_is_too_sparse(segments: list[Segment], source_duration: float) -> bool:
    text_chars = sum(len(segment.text.strip()) for segment in segments)
    if not segments or text_chars < 24:
        return True
    if source_duration >= 20.0 and text_chars / max(1.0, source_duration) < 0.8:
        return True
    # A long hole between otherwise valid phrases is easy to miss when one
    # verbose segment makes the character count look healthy. This was the
    # failure mode in job 26929 (roughly 23 seconds missing in the middle).
    spoken = sorted(
        (segment for segment in segments if segment.text.strip() and segment.end > segment.start),
        key=lambda item: (item.start, item.end),
    )
    if source_duration >= 20.0 and len(spoken) >= 2:
        largest_inner_gap = max(
            (right.start - left.end for left, right in zip(spoken, spoken[1:])),
            default=0.0,
        )
        if largest_inner_gap >= max(10.0, source_duration * 0.25):
            return True
    return False


def _dub_is_too_sparse(segments: list[Segment], source_duration: float) -> bool:
    spoken = [segment for segment in segments if segment.spoken_text]
    if not spoken:
        return True

    text_chars = sum(len(segment.spoken_text) for segment in spoken)
    coverage = _segment_coverage_seconds(spoken)
    duration = max(1.0, source_duration)
    if text_chars < 32:
        return True
    if source_duration >= 30.0 and len(spoken) < 4:
        return True
    if source_duration >= 45.0 and coverage / duration < 0.08 and text_chars / duration < 1.2:
        return True
    if source_duration >= 90.0 and len(spoken) < 6 and text_chars / duration < 1.6:
        return True
    # Plenty of phrases overall can still leave a long stretch saying nothing -
    # in job 40814 the opening 27 seconds were silent while every count above
    # looked healthy, because one segment claimed the whole 30-second window.
    if source_duration >= 30.0 and _largest_unspoken_stretch(spoken, source_duration) >= 12.0:
        return True
    return False


def _plausible_speech_seconds(text: str) -> float:
    """How long a phrase could plausibly take to say, erring on the slow side.

    Real segments run about 10 characters per second; assuming half that keeps
    this from mistaking an ordinary phrase for a hole.
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    return max(0.8, len(stripped) / 5.0)


def _tighten_overlong_segments(
    segments: list[Segment],
    min_hole_seconds: float = 5.0,
) -> list[Segment]:
    """Shrink segments that claim far more time than their words can fill.

    Whisper answers an unintelligible stretch with one segment spanning its whole
    30-second window and a handful of words in it. Everything downstream reads a
    segment's span as time that is already spoken for, so those dead seconds are
    invisible: the sparse-fill check sees full coverage, and fill candidates
    landing inside the span are rejected as conflicts. Trimming the segment back
    to what it can actually say turns that dead time into an ordinary gap, which
    the existing gap handling already knows how to fill.
    """
    tightened: list[Segment] = []
    changed = 0
    for segment in segments:
        needed = _plausible_speech_seconds(segment.spoken_text)
        if needed > 0.0 and segment.duration - needed >= min_hole_seconds:
            tightened.append(
                Segment(
                    start=segment.start,
                    end=segment.start + needed,
                    text=segment.text,
                    translated_text=segment.translated_text,
                    speaker_wav=segment.speaker_wav,
                    speaker_id=segment.speaker_id,
                    speaker_ref_text=segment.speaker_ref_text,
                )
            )
            changed += 1
            continue
        tightened.append(segment)

    if not changed:
        return segments
    print(f"      Tightened over-long ASR segments: {changed}/{len(segments)}")
    return tightened


def _largest_unspoken_stretch(segments: list[Segment], source_duration: float) -> float:
    """Longest run of the video with nothing being said, measured from the time
    each segment actually speaks rather than the span it claims."""
    if source_duration <= 0.0:
        return 0.0
    spoken: list[tuple[float, float]] = []
    for segment in segments:
        needed = _plausible_speech_seconds(segment.spoken_text)
        if needed <= 0.0:
            continue
        # Audio runs from the segment's start for as long as the words take; the
        # span it claims is bookkeeping and is exactly what hides these holes.
        start = max(0.0, segment.start)
        spoken.append((start, min(source_duration, start + needed)))
    if not spoken:
        return source_duration

    largest = 0.0
    cursor = 0.0
    for start, end in sorted(spoken):
        if start - cursor > largest:
            largest = start - cursor
        cursor = max(cursor, end)
    return max(largest, source_duration - cursor)


def _segment_coverage_seconds(segments: list[Segment]) -> float:
    ranges = sorted((segment.start, segment.end) for segment in segments if segment.end > segment.start)
    if not ranges:
        return 0.0
    total = 0.0
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += max(0.0, current_end - current_start)
        current_start, current_end = start, end
    total += max(0.0, current_end - current_start)
    return total


def _clamp_segments_to_duration(segments: list[Segment], source_duration: float) -> list[Segment]:
    if source_duration <= 0.0:
        return segments

    result: list[Segment] = []
    adjusted = 0
    for segment in segments:
        start = max(0.0, min(segment.start, source_duration))
        end = max(start, min(segment.end, source_duration))
        if end - start < 0.05:
            adjusted += 1
            continue
        if start != segment.start or end != segment.end:
            adjusted += 1
        result.append(
            Segment(
                start=start,
                end=end,
                text=segment.text,
                translated_text=segment.translated_text,
                speaker_wav=segment.speaker_wav,
                speaker_id=segment.speaker_id,
                speaker_ref_text=segment.speaker_ref_text,
            )
        )

    if adjusted:
        print(f"      ASR timing clamp: adjusted_or_dropped={adjusted}/{len(segments)}")
    return result


def _debug_path(config: DubConfig, filename: str) -> Path:
    debug_dir = config.workdir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir / filename


def _safe_label(value: str | None, fallback: str = "auto") -> str:
    value = (value or fallback).strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    return value.strip("_") or fallback


def _resume_state_path(config: DubConfig) -> Path:
    return config.workdir / "resume_state.json"


def _load_resume_state(config: DubConfig) -> dict[str, object]:
    path = _resume_state_path(config)
    if not config.resume or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"      Resume state ignored: {type(exc).__name__}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _save_resume_state(config: DubConfig, **updates: object) -> None:
    if not config.resume:
        return
    state = _load_resume_state(config)
    state.update(updates)
    path = _resume_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_asr_signature(config: DubConfig) -> str:
    """Identify source transcripts that are safe to reuse after pipeline changes."""
    values = (
        "content-chaos-v3",
        config.asr_backend,
        config.whisper_model,
        config.source_lang or "auto",
        str(bool(config.force_source_language)),
        config.whisper_task,
        config.glitch_profile,
        str(bool(config.artifact_chaos_mode)),
        str(bool(config.content_chaos_backbone)),
        str(bool(config.reference_timing_asr)),
    )
    return "|".join(values)


def _file_ready(path: Path, min_size: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_size


def _media_cache_entry(video_path: Path, config: DubConfig) -> Path | None:
    cache_root = config.media_cache_dir
    if not cache_root:
        return None
    if not cache_root.is_absolute():
        cache_root = _repo_root() / cache_root
    try:
        digest = _sha256_file(video_path)
    except Exception as exc:
        print(f"      Media cache disabled for this file: {type(exc).__name__}: {exc}")
        return None
    entry = cache_root / digest[:2] / digest
    entry.mkdir(parents=True, exist_ok=True)
    metadata_path = entry / "media.json"
    if not metadata_path.exists():
        try:
            metadata_path.write_text(
                json.dumps(
                    {
                        "sha256": digest,
                        "source_name": video_path.name,
                        "source_size": video_path.stat().st_size,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    return entry


def _media_cache_metadata(cache_entry: Path | None) -> dict[str, object]:
    if cache_entry is None:
        return {}
    metadata_path = cache_entry / "media.json"
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_cached_file(cache_entry: Path | None, cache_name: str, destination: Path, min_size: int = 1024) -> bool:
    if cache_entry is None:
        return False
    source = cache_entry / cache_name
    if not _file_ready(source, min_size=min_size):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, destination)
    except Exception as exc:
        print(f"      Media cache restore skipped {cache_name}: {type(exc).__name__}: {exc}")
        return False
    return _file_ready(destination, min_size=min_size)


def _store_cached_file(cache_entry: Path | None, source: Path, cache_name: str, min_size: int = 1024) -> None:
    if cache_entry is None or not _file_ready(source, min_size=min_size):
        return
    destination = cache_entry / cache_name
    if _file_ready(destination, min_size=min_size):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copy2(source, temp_path)
        temp_path.replace(destination)
    except Exception as exc:
        print(f"      Media cache store skipped {destination.name}: {type(exc).__name__}: {exc}")
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _restore_cached_separation(
    cache_entry: Path | None,
    mix_audio: Path,
    config: DubConfig,
) -> SeparationResult | None:
    if cache_entry is None:
        return None
    cache_dir = cache_entry / "separation" / _safe_label(config.demucs_model, fallback="demucs")
    vocals_cache = cache_dir / "vocals.wav"
    no_vocals_cache = cache_dir / "no_vocals.wav"
    if not (_file_ready(vocals_cache) and _file_ready(no_vocals_cache)):
        return None

    stem_dir = config.workdir / "separated" / config.demucs_model / mix_audio.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(vocals_cache, stem_dir / "vocals.wav")
        shutil.copy2(no_vocals_cache, stem_dir / "no_vocals.wav")
    except Exception as exc:
        print(f"      Media cache separation restore skipped: {type(exc).__name__}: {exc}")
        return None
    return _existing_separation_result(mix_audio, config)


def _store_cached_separation(cache_entry: Path | None, result: SeparationResult | None, config: DubConfig) -> None:
    if cache_entry is None or result is None:
        return
    cache_dir = cache_entry / "separation" / _safe_label(config.demucs_model, fallback="demucs")
    _store_cached_file(cache_dir, result.vocals_path, "vocals.wav")
    _store_cached_file(cache_dir, result.instrumental_path, "no_vocals.wav")


def _seed_media_cache_from_legacy_jobs(video_path: Path, cache_entry: Path | None, config: DubConfig) -> bool:
    if cache_entry is None:
        return False

    cache_has_audio = _file_ready(cache_entry / "source_16k.wav") and _file_ready(cache_entry / "source_mix.wav")
    cache_has_separation = _restore_cached_separation_ready(cache_entry, config)
    if cache_has_audio and cache_has_separation:
        return False

    metadata = _media_cache_metadata(cache_entry)
    expected_sha = str(metadata.get("sha256") or cache_entry.name)
    expected_size = metadata.get("source_size")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        return False

    runs_root = _runs_root_from_workdir(config.workdir)
    if runs_root is None or not runs_root.exists():
        return False

    current_job_dir = config.workdir.parent.resolve()
    job_files = sorted(
        runs_root.glob("*/*/job.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    video_suffixes = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
    matched_without_work = False
    for job_file in job_files[:300]:
        job_dir = job_file.parent
        try:
            if job_dir.resolve() == current_job_dir:
                continue
        except Exception:
            pass

        candidates: list[Path] = []
        for name in ("input_trimmed.mp4", "input.mp4"):
            candidate = job_dir / name
            if candidate.exists():
                candidates.append(candidate)
        candidates.extend(
            path
            for path in job_dir.glob("input*")
            if path.is_file() and path.suffix.lower() in video_suffixes and path not in candidates
        )
        for candidate in candidates:
            try:
                if expected_size is not None and candidate.stat().st_size != int(expected_size):
                    continue
                if _sha256_file(candidate) != expected_sha:
                    continue
            except Exception:
                continue
            if _store_legacy_media_cache(cache_entry, job_dir / "work", config):
                print(f"      Media cache: seeded from previous job {job_dir}")
                return True
            matched_without_work = True
    if matched_without_work:
        print("      Media cache: previous matching job found, but reusable work audio was not stored")
    return False


def _restore_cached_separation_ready(cache_entry: Path | None, config: DubConfig) -> bool:
    if cache_entry is None:
        return False
    cache_dir = cache_entry / "separation" / _safe_label(config.demucs_model, fallback="demucs")
    return _file_ready(cache_dir / "vocals.wav") and _file_ready(cache_dir / "no_vocals.wav")


def _store_legacy_media_cache(cache_entry: Path, legacy_workdir: Path, config: DubConfig) -> bool:
    stored = False
    source_audio = legacy_workdir / "source_16k.wav"
    if _file_ready(source_audio) and not _file_ready(cache_entry / "source_16k.wav"):
        _store_cached_file(cache_entry, source_audio, "source_16k.wav")
        stored = True

    mix_audio = legacy_workdir / "source_mix.wav"
    if _file_ready(mix_audio) and not _file_ready(cache_entry / "source_mix.wav"):
        _store_cached_file(cache_entry, mix_audio, "source_mix.wav")
        stored = True

    stem_dir = legacy_workdir / "separated" / config.demucs_model / "source_mix"
    separation_result = SeparationResult(
        vocals_path=stem_dir / "vocals.wav",
        instrumental_path=stem_dir / "no_vocals.wav",
    )
    if _file_ready(separation_result.vocals_path) and _file_ready(separation_result.instrumental_path):
        _store_cached_separation(cache_entry, separation_result, config)
        stored = True
    return stored


def _srt_ready(path: Path) -> bool:
    return path.exists() and bool(read_srt(path, translated=True))


def _existing_separation_result(mix_audio: Path, config: DubConfig) -> SeparationResult | None:
    stem_dir = config.workdir / "separated" / config.demucs_model / mix_audio.stem
    vocals_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"
    if _file_ready(vocals_path) and _file_ready(instrumental_path):
        return SeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)
    return None


def run_dub(video_path: Path, config: DubConfig) -> Path:
    video_path = video_path.resolve()
    config.workdir.mkdir(parents=True, exist_ok=True)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    resume_state = _load_resume_state(config)
    media_cache = _media_cache_entry(video_path, config)
    _seed_media_cache_from_legacy_jobs(video_path, media_cache, config)
    if not config.resume:
        _prepare_workdir_outputs(config)

    if config.resume and _file_ready(config.output, min_size=4096):
        print(f"[resume] Final video already exists: {config.output}")
        _report_progress(config, "Дубляж уже готов", 100, 100, config.output.name)
        return config.output

    _report_progress(config, "Извлекаю аудио", 3, 100, video_path.name)
    print(f"[1/7] Extracting audio: {video_path}")
    source_audio = config.workdir / "source_16k.wav"
    if config.resume and _file_ready(source_audio):
        print(f"      Resume: using existing audio {source_audio}")
    elif _restore_cached_file(media_cache, "source_16k.wav", source_audio):
        print("      Media cache: restored source_16k.wav")
    else:
        extract_audio(video_path, source_audio)
        _store_cached_file(media_cache, source_audio, "source_16k.wav")
    mix_audio = config.workdir / "source_mix.wav"
    separation_result = None
    bed_path = None
    # multitrack needs the untouched original mix as its own audio track even
    # when the user didn't ask for Demucs separation.
    needs_mix_audio = config.separation != "none" or config.audio_bed in {"instrumental", "multitrack"}
    needs_separation = config.separation != "none" or config.audio_bed == "instrumental"
    if needs_mix_audio:
        _report_progress(config, "Разделяю голос и фон", 10, 100, f"provider={config.separation}")
        if config.resume and _file_ready(mix_audio):
            print(f"      Resume: using existing mix audio {mix_audio}")
        elif _restore_cached_file(media_cache, "source_mix.wav", mix_audio):
            print("      Media cache: restored source_mix.wav")
        else:
            extract_audio_track(video_path, mix_audio)
            _store_cached_file(media_cache, mix_audio, "source_mix.wav")
        if needs_separation:
            separation_result = _existing_separation_result(mix_audio, config) if config.resume else None
            if separation_result is not None:
                print(f"      Resume: using existing separation {separation_result.vocals_path.parent}")
            else:
                separation_result = _restore_cached_separation(media_cache, mix_audio, config)
                if separation_result is not None:
                    print(f"      Media cache: restored separation {separation_result.vocals_path.parent}")
                else:
                    print(f"      Separating audio provider={config.separation}")
                    separation_result = separate_audio(mix_audio, config.workdir / "separated", config)
                    _store_cached_separation(media_cache, separation_result, config)
            if separation_result and config.audio_bed in {"instrumental", "multitrack"}:
                bed_path = separation_result.instrumental_path
            elif config.audio_bed == "instrumental":
                raise RuntimeError("audio_bed=instrumental needs --separation demucs")
    _save_resume_state(config, audio=True, separation=separation_result is not None)
    source_duration = probe_duration(source_audio)
    _report_progress(config, "Аудио подготовлено", 20, 100, None)

    _report_progress(
        config,
        "Распознаю основную речь",
        24,
        100,
        f"{config.asr_backend} {config.whisper_model}, вход={config.source_lang or 'auto'}",
    )
    print(
        "[2/7] Transcribing "
        f"backend={config.asr_backend} "
        f"model={config.whisper_model} "
        f"source={config.source_lang or 'auto'} "
        f"task={config.whisper_task} "
        f"force_source={config.force_source_language} "
        f"suppress_ascii={config.suppress_plain_ascii_tokens}"
    )
    source_srt_path = config.workdir / "source.srt"
    requested_source_signature = _source_asr_signature(config)
    stored_source_signature = resume_state.get("source_asr_signature")
    source_resume_compatible = stored_source_signature == requested_source_signature
    if stored_source_signature is None and not config.force_source_language:
        # Preserve old clean/auto resumes. Old jobs with a selected input
        # language must regenerate because that language now drives main ASR.
        source_resume_compatible = True
    source_file_ready = _file_ready(source_srt_path, min_size=16)
    source_asr_changed = bool(config.resume and source_file_ready and not source_resume_compatible)
    if source_asr_changed:
        print("      Resume: source ASR settings changed; regenerating the main transcript")
    if config.resume and source_file_ready and source_resume_compatible:
        resumed_lang = resume_state.get("source_lang")
        if not config.force_source_language and isinstance(resumed_lang, str) and resumed_lang:
            config.source_lang = resumed_lang
        segments = read_srt(source_srt_path, translated=False)
        print(f"      Resume: loaded source ASR segments={len(segments)}")
    else:
        requested_source_lang = config.source_lang
        main_asr_audio = _whisper_chaos_audio(source_audio, config, purpose="main")
        segments = transcribe(main_asr_audio, config)
        segments = _clamp_segments_to_duration(segments, source_duration)
        if (
            config.artifact_chaos_mode
            and config.force_source_language
            and config.source_lang
            and not config.inject_artifacts
        ):
            chunk_segments = _harvest_chunked_forced_artifacts(main_asr_audio, config, config, source_duration)
            if chunk_segments:
                segments = [*segments, *chunk_segments]
        if config.asr_retry_on_repetition and requested_source_lang is not None and is_repetitive_loop(segments):
            print(
                "      ASR looks like a repetition loop; "
                f"retrying with auto language instead of {requested_source_lang}"
            )
            config.source_lang = None
            retry_segments = transcribe(source_audio, config)
            retry_segments = _clamp_segments_to_duration(retry_segments, source_duration)
            if retry_segments and not is_repetitive_loop(retry_segments):
                segments = retry_segments
            else:
                config.source_lang = requested_source_lang
        if (
            config.asr_fallback_on_sparse
            and requested_source_lang is not None
            and _asr_is_too_sparse(segments, probe_duration(source_audio))
        ):
            print(
                "      Forced ASR is too sparse; "
                "falling back to auto ASR plus forced artifact hunt"
            )
            write_srt(_debug_path(config, "forced_sparse_source.srt"), segments, translated=False)
            config.source_lang = None
            config.force_source_language = False
            config.glitch_profile = "clean"
            config.input_pivot_lang = requested_source_lang
            config.artifact_source_lang = requested_source_lang
            config.inject_artifacts = True
            config.asr_retry_on_repetition = True
            config.collapse_repetitions = True
            segments = transcribe(source_audio, config)
            segments = _clamp_segments_to_duration(segments, source_duration)
        if config.reference_timing_asr and config.force_source_language:
            _report_progress(config, "Уточняю тайминги речи", 29, 100, "быстрый авто-ASR")
            timing_segments, timing_lang = _reference_timing_source_asr(source_audio, config, source_duration)
            if timing_segments:
                write_srt(_debug_path(config, "timing_reference.srt"), timing_segments, translated=False)
                aligned_segments = _align_forced_text_to_reference_timings(segments, timing_segments)
                if aligned_segments:
                    print(
                        "      Forced text aligned to reference timings: "
                        f"text={len(segments)} timing={len(timing_segments)} aligned={len(aligned_segments)} "
                        f"detected={timing_lang or 'auto'}"
                    )
                    write_srt(
                        _debug_path(config, "forced_text_timing_aligned.srt"),
                        aligned_segments,
                        translated=False,
                    )
                    if config.content_chaos_backbone:
                        forced_lang = config.source_lang
                        if forced_lang:
                            write_srt(
                                config.workdir / f"forced_{_safe_label(forced_lang)}_transcript.srt",
                                aligned_segments,
                                translated=False,
                            )
                        artifact_lang = _artifact_corruption_language(
                            forced_lang,
                            timing_lang,
                            config,
                        )
                        config.artifact_source_lang = artifact_lang
                        config.input_pivot_lang = forced_lang
                        config.source_lang = timing_lang
                        config.force_source_language = False
                        config.inject_artifacts = bool(artifact_lang)
                        config.artifact_ratio = 0.20
                        segments = [
                            Segment(start=item.start, end=item.end, text=item.text)
                            for item in timing_segments
                            if item.text.strip()
                        ]
                        print(
                            "      Content chaos backbone: "
                            f"content={len(segments)} forced_artifacts={len(aligned_segments)} "
                            f"source={config.source_lang or 'auto'} "
                            f"selected={forced_lang or 'none'} corruption={artifact_lang or 'none'}"
                        )
                    else:
                        segments = aligned_segments
        if config.glitch_profile == "clean":
            segments = clean_segments(segments)
        write_srt(source_srt_path, segments, translated=False)
        if config.force_source_language and config.source_lang:
            write_srt(config.workdir / f"forced_{_safe_label(config.source_lang)}_transcript.srt", segments, translated=False)
        _save_resume_state(
            config,
            source_asr=True,
            source_lang=config.source_lang,
            source_asr_signature=requested_source_signature,
        )
    retry_segments = _retry_sparse_source_asr(source_audio, config, segments, source_duration)
    if retry_segments is not segments:
        segments = retry_segments
        source_asr_changed = True
        write_srt(source_srt_path, segments, translated=False)
        _save_resume_state(config, source_asr=True, source_lang=config.source_lang, sparse_source_fallback=True)
    loop_cleaned_source = suppress_pathological_segment_loops(segments)
    if len(loop_cleaned_source) != len(segments):
        write_srt(_debug_path(config, "source_before_loop_cleanup.srt"), segments, translated=False)
        segments = loop_cleaned_source
        source_asr_changed = True
        write_srt(source_srt_path, segments, translated=False)
        _save_resume_state(
            config,
            source_asr=True,
            source_lang=config.source_lang,
            source_segment_loop_cleanup=True,
            segment_count=len(segments),
        )
    segments, pre_censor_changed = _apply_experimental_censor(segments, config, stage="pre")
    if pre_censor_changed:
        source_asr_changed = True
        write_srt(_debug_path(config, "source_censored.srt"), segments, translated=False)
        _save_resume_state(config, source_asr=True, source_lang=config.source_lang, pre_censor=True)
    print(f"      ASR segments: {len(segments)}")
    _report_progress(config, "Основная речь распознана", 32, 100, f"сегментов: {len(segments)}")

    artifact_srt_path = _debug_path(config, "artifact_translated.srt")
    if config.resume and not source_asr_changed and _file_ready(artifact_srt_path, min_size=16):
        artifact_segments = read_srt(artifact_srt_path, translated=True)
        print(f"      Resume: loaded artifact candidates={len(artifact_segments)}")
    else:
        artifact_segments = _build_artifact_segments(source_audio, config, segments)
        clear_openai_whisper_cache()
        _save_resume_state(config, artifacts=bool(artifact_segments))

    translation_detail = _translation_progress_detail(config)
    _report_progress(config, "Перевожу текст", 48, 100, translation_detail)
    print(f"[3/7] Translating {translation_detail}")
    translated_srt_path = config.workdir / "translated.srt"
    translation_changed = source_asr_changed
    if config.resume and not source_asr_changed and _file_ready(translated_srt_path, min_size=16):
        segments = read_srt(translated_srt_path, translated=True)
        print(f"      Resume: loaded translated segments={len(segments)}")
        segments, repeat_clamp_changed = clamp_obvious_word_repeats_in_segments(
            segments,
            max_word_repeats=3,
        )
        loop_cleaned_segments = suppress_pathological_segment_loops(segments)
        segment_loops_changed = len(loop_cleaned_segments) != len(segments)
        segments = loop_cleaned_segments
        if repeat_clamp_changed or segment_loops_changed:
            write_srt(translated_srt_path, segments, translated=True)
            _save_resume_state(
                config,
                translated=True,
                segment_count=len(segments),
                repeat_clamp=bool(repeat_clamp_changed),
                segment_loop_cleanup=segment_loops_changed,
            )
            translation_changed = True
    else:
        segments = _translate_dub_segments(segments, config)
        write_srt(_debug_path(config, "translated_clean.srt"), segments, translated=True)
        segments = _maybe_distort_translations(
            segments,
            config,
            output_path=_debug_path(config, "translated_pivot.srt"),
            force=_should_distort_main_translation(config),
        )
        segments = apply_glitch_profile(
            segments,
            profile=config.glitch_profile,
            target_lang=config.target_lang,
            min_gap_seconds=config.ghost_gap_seconds,
        )
        write_srt(_debug_path(config, "translated_raw.srt"), segments, translated=True)
        if config.collapse_repetitions:
            segments = collapse_repetitions_in_segments(
                segments,
                max_phrase_repeats=config.max_phrase_repeats,
                max_word_repeats=config.max_word_repeats,
            )
        # Before anything looks for room to place content: a segment that claims a
        # whole silent Whisper window makes that time look occupied, so neither
        # artifacts nor the sparse fill can land in it.
        segments = _tighten_overlong_segments(segments)
        if artifact_segments:
            segments = _inject_artifact_segments(
                segments,
                artifact_segments,
                config,
                source_duration=source_duration,
            )
        segments = _fill_sparse_dub_segments(segments, config, source_audio, source_duration)
        if config.tts.lower() not in {"moss", "moss-tts", "mosstts", "moss-v1.5"}:
            segments = _fit_segment_text_budgets(segments, config)
        segments, repeat_clamp_changed = clamp_obvious_word_repeats_in_segments(
            segments,
            max_word_repeats=3,
        )
        segments = suppress_pathological_segment_loops(segments)
        write_srt(translated_srt_path, segments, translated=True)
        _save_resume_state(config, translated=True, segment_count=len(segments))
        translation_changed = True
    filled_segments = _fill_sparse_dub_segments(segments, config, source_audio, source_duration)
    if filled_segments is not segments:
        segments = filled_segments
        if config.tts.lower() not in {"moss", "moss-tts", "mosstts", "moss-v1.5"}:
            segments = _fit_segment_text_budgets(segments, config)
        segments, repeat_clamp_changed = clamp_obvious_word_repeats_in_segments(
            segments,
            max_word_repeats=3,
        )
        segments = suppress_pathological_segment_loops(segments)
        write_srt(translated_srt_path, segments, translated=True)
        _save_resume_state(config, translated=True, segment_count=len(segments), sparse_fill=True)
        translation_changed = True
    if translation_changed:
        _clear_downstream_dub_outputs(config)
    print(f"      Dub segments: {len(segments)}")
    _report_progress(config, "Перевод подготовлен", 62, 100, f"реплик: {len(segments)}")

    if getattr(config, "preprocess_only", False):
        _save_resume_state(
            config,
            preprocess_complete=True,
            translated=True,
            segment_count=len(segments),
        )
        _report_progress(config, "Подготовка на ноутбуке завершена", 64, 100, f"реплик: {len(segments)}")
        print("[preprocess] Translation package is ready; TTS remains on the coordinator")
        return translated_srt_path

    tts_already_fit = config.resume and _tts_fit_complete(segments, config)

    # The old Demucs vocals had metallic/warbling artifacts that clone TTS
    # would copy, so speaker identity used to come from the full-band mix
    # instead. BS-Roformer's separated vocals are clean enough to use
    # directly - prefer them when separation actually ran, and fall back to
    # the raw mix otherwise (e.g. --separation none).
    clean_vocals_path = (
        separation_result.vocals_path
        if separation_result and _file_ready(separation_result.vocals_path)
        else None
    )
    if not tts_already_fit and _needs_speaker_references(config) and config.speaker_wav is None:
        config.speaker_wav = clean_vocals_path or (mix_audio if _file_ready(mix_audio) else source_audio)
    if not tts_already_fit and _needs_speaker_references(config) and config.speaker_wav is not None:
        config.speaker_wav = _prepare_xtts_reference(config.speaker_wav, config.workdir / "speaker_refs" / "global_clean.wav")
        print(f"      Clone TTS speaker reference: {config.speaker_wav}")

    if not tts_already_fit and _needs_speaker_references(config) and config.multi_speaker:
        _report_progress(config, "Готовлю голосовые референсы", 64, 100, "multi-speaker")
        reference_audio = clean_vocals_path or (mix_audio if _file_ready(mix_audio) else source_audio)
        _assign_segment_speaker_refs(segments, reference_audio, config, diarization_audio=source_audio)

    _report_progress(config, "Озвучиваю реплики", 70, 100, f"provider={config.tts}, реплик: {len(segments)}")
    print(f"[4/7] Synthesizing speech provider={config.tts}")
    raw_dir = config.workdir / "tts_raw"
    fit_dir = config.workdir / "tts_fit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)

    tts_provider = config.tts.lower()
    batch_provider = tts_provider in {
        "qwen3",
        "qwen3-tts",
        "qwen3tts",
        "cosyvoice",
        "cosyvoice-tts",
        "cosyvoicetts",
        "cosy",
        "moss",
        "moss-tts",
        "mosstts",
        "moss-v1.5",
    }
    batch_items: list[tuple[int, Segment, Path]] = []
    if batch_provider:
        for index, segment in enumerate(segments, start=1):
            fitted_path = fit_dir / f"{index:05d}.wav"
            if segment.spoken_text and not (config.resume and _file_ready(fitted_path)):
                batch_items.append((index, segment, raw_dir / f"{index:05d}.wav"))
        if batch_items:
            try:
                if tts_provider in {"qwen3", "qwen3-tts", "qwen3tts"}:
                    synthesize_qwen3_batch(batch_items, config)
                elif tts_provider in {"cosyvoice", "cosyvoice-tts", "cosyvoicetts", "cosy"}:
                    synthesize_cosyvoice_batch(batch_items, config)
                elif tts_provider in {"moss", "moss-tts", "mosstts", "moss-v1.5"}:
                    synthesize_moss_batch(batch_items, config)
            except Exception as exc:
                print(f"      {config.tts} batch fallback to F5: {type(exc).__name__}: {exc}")
                fallback_config = copy(config)
                fallback_config.tts = "f5"
                for _index, segment, raw_path in batch_items:
                    if _file_ready(raw_path):
                        continue
                    raw_path.unlink(missing_ok=True)
                    synthesize_segment(segment, raw_path, fallback_config)

    fitted_items: list[tuple[Segment, Path]] = []
    for index, segment in enumerate(segments, start=1):
        if not segment.spoken_text:
            continue
        raw_path = raw_dir / f"{index:05d}.wav"
        fitted_path = fit_dir / f"{index:05d}.wav"
        if config.resume and _file_ready(fitted_path):
            print(f"      Resume: using fitted TTS segment {index}/{len(segments)}")
        else:
            if not batch_provider:
                synthesize_segment(segment, raw_path, config)
            if tts_provider in {"moss", "moss-tts", "mosstts", "moss-v1.5"}:
                # MOSS controls its own delivery and must be allowed to finish
                # the phrase instead of being time-stretched or trimmed.
                normalize_wav(raw_path, fitted_path)
                if config.trim_tts_silence:
                    _trim_tts_silence(fitted_path, config.tts_max_pause_seconds)
            elif config.fit_to_segments:
                fit_wav_to_duration(raw_path, fitted_path, max(0.1, segment.duration))
            else:
                normalize_wav(raw_path, fitted_path)
        fitted_items.append((segment, fitted_path))
        if index == len(segments) or index % 5 == 0:
            percent = 70 + round(20 * index / max(1, len(segments)))
            position = _format_video_position(segment.end, source_duration)
            _report_progress(
                config,
                "Озвучиваю реплики",
                percent,
                100,
                f"сегмент {index}/{len(segments)} · {position}",
            )
        if index % 25 == 0:
            print(f"      synthesized {index}/{len(segments)}")
    _save_resume_state(config, tts_fit=True)

    mix_items = _plan_overlap_aware_mix(fitted_items, source_audio, config)

    _report_progress(config, "Собираю аудиодорожку", 92, 100, None)
    print("[5/7] Mixing dub track")
    video_duration = probe_duration(video_path)
    dub_track = config.workdir / "dub_track.wav"
    if config.resume and _file_ready(dub_track):
        print(f"      Resume: using existing dub track {dub_track}")
    else:
        delayed_mix(mix_items, video_duration, dub_track, config.workdir)
    _save_resume_state(config, mixed=True)

    _report_progress(config, "Собираю финальное видео", 96, 100, None)
    print("[6/7] Combining video and dub audio")
    final_original_volume = 0.0 if config.audio_bed == "dub-only" else config.original_volume
    if config.resume and _file_ready(config.output, min_size=4096):
        print(f"      Resume: using existing final video {config.output}")
    elif config.audio_bed == "multitrack":
        combine_video_audio_multitrack(
            video_path=video_path,
            original_audio_path=mix_audio,
            dub_path=dub_track,
            output_path=config.output,
            dub_volume=config.dub_volume,
            bed_path=bed_path,
            original_lang=config.source_lang,
            dub_lang=config.target_lang,
        )
    else:
        combine_video_audio(
            video_path=video_path,
            dub_path=dub_track,
            output_path=config.output,
            original_volume=final_original_volume,
            dub_volume=config.dub_volume,
            bed_path=bed_path if config.audio_bed == "instrumental" else None,
        )
    _save_resume_state(config, final=True)

    print(f"[7/7] Done: {config.output}")
    _report_progress(config, "Дубляж готов", 97, 100, config.output.name)
    return config.output


def _build_artifact_segments(
    source_audio: Path,
    config: DubConfig,
    base_segments: list[Segment],
) -> list[Segment]:
    artifact_lang = config.artifact_source_lang
    if not config.inject_artifacts or not artifact_lang or artifact_lang == "auto":
        return []
    same_language_hunt = bool(config.source_lang and artifact_lang == config.source_lang)
    ru_same_language_hunt = same_language_hunt and artifact_lang == "ru"
    if same_language_hunt and not ru_same_language_hunt:
        return []

    artifact_model = _artifact_whisper_model(config, artifact_lang)
    _report_progress(
        config,
        "Ищу Whisper-артефакты",
        34,
        100,
        f"openai-whisper {artifact_model}, вход={artifact_lang}",
    )
    print(
        "      Building artifact layer "
        f"source={artifact_lang} detected={config.source_lang or 'auto'} model={artifact_model}"
    )
    artifact_config = copy(config)
    artifact_config.source_lang = artifact_lang
    artifact_config.asr_backend = "openai-whisper"
    artifact_config.whisper_model = artifact_model
    artifact_config.whisper_device = config.artifact_whisper_device
    artifact_config.whisper_compute_type = "auto"
    artifact_config.glitch_profile = "faithful"
    artifact_config.force_source_language = True
    artifact_config.suppress_plain_ascii_tokens = False
    artifact_config.condition_on_previous_text = True
    # Only accept hallucinations produced from this video's audio. Supplying
    # known phrases here made prompted text look like a genuine Whisper find.
    artifact_config.initial_prompt = None
    artifact_config.hallucination_silence_threshold = None
    artifact_config.collapse_repetitions = True
    try:
        source_duration = probe_duration(source_audio)
    except Exception:
        source_duration = max((segment.end for segment in base_segments), default=0.0)

    whole_artifacts = _load_resumable_artifact_source(config, artifact_lang) if config.resume else []
    if whole_artifacts:
        if ru_same_language_hunt:
            before = len(whole_artifacts)
            whole_artifacts = _filter_same_language_ru_artifacts(whole_artifacts)
            print(
                "      Same-language RU artifact filter: "
                f"kept={len(whole_artifacts)} dropped={before - len(whole_artifacts)}"
            )
        _report_progress(
            config,
            "Перевожу Whisper-артефакты",
            42,
            100,
            f"resume: кандидатов {len(whole_artifacts)}",
        )
        whole_artifacts = _limit_chaos_artifacts_before_translation(
            whole_artifacts, base_segments, config
        )
        whole_artifacts = _translate_and_clean_artifacts(
            whole_artifacts,
            artifact_config,
            _debug_path(config, "artifact_whisper_translated.srt"),
        )
    else:
        try:
            artifact_audio = _whisper_chaos_audio(source_audio, artifact_config, purpose="artifact")
            full_artifacts = transcribe(artifact_audio, artifact_config)
            full_artifacts = _clamp_segments_to_duration(full_artifacts, source_duration)
            chunk_artifacts = (
                _harvest_chunked_forced_artifacts(artifact_audio, artifact_config, config, source_duration)
                if config.artifact_chaos_mode
                else []
            )
            whole_artifacts = [*chunk_artifacts, *full_artifacts] if config.artifact_chaos_mode else full_artifacts
            if whole_artifacts:
                write_srt(_debug_path(config, "artifact_source_raw.srt"), whole_artifacts, translated=False)
                if config.artifact_chaos_mode:
                    whole_artifacts = _clean_chaos_artifact_source_segments(whole_artifacts)
                else:
                    whole_artifacts = _clean_artifact_source_segments(whole_artifacts)
                if ru_same_language_hunt:
                    before = len(whole_artifacts)
                    whole_artifacts = _filter_same_language_ru_artifacts(whole_artifacts)
                    print(
                        "      Same-language RU artifact filter: "
                        f"kept={len(whole_artifacts)} dropped={before - len(whole_artifacts)}"
                    )
                forced_source_path = config.workdir / f"forced_{_safe_label(artifact_lang)}_transcript.srt"
                write_srt(forced_source_path, whole_artifacts, translated=False)
                write_srt(_debug_path(config, "artifact_source.srt"), whole_artifacts, translated=False)
                if whole_artifacts:
                    whole_artifacts = _limit_chaos_artifacts_before_translation(
                        whole_artifacts, base_segments, config
                    )
                    whole_artifacts = _translate_and_clean_artifacts(
                        whole_artifacts,
                        artifact_config,
                        _debug_path(config, "artifact_whisper_translated.srt"),
                    )
        except Exception as exc:
            print(f"      Whisper artifact layer skipped: {type(exc).__name__}: {exc}")

    artifacts = whole_artifacts if config.artifact_chaos_mode else _dedupe_artifact_segments(whole_artifacts)
    if artifacts:
        write_srt(_debug_path(config, "artifact_translated.srt"), artifacts, translated=True)
        print(f"      Artifact candidates: {len(artifacts)}")
        _report_progress(config, "Whisper-артефакты найдены", 44, 100, f"кандидатов: {len(artifacts)}")
        return artifacts

    print("      Artifact layer skipped: no candidates")
    _report_progress(config, "Whisper-артефактов нет", 44, 100, "дальше только основной перевод")
    return []


def _retry_sparse_source_asr(
    source_audio: Path,
    config: DubConfig,
    segments: list[Segment],
    source_duration: float,
) -> list[Segment]:
    if config.force_source_language or not _asr_is_too_sparse(segments, source_duration):
        return segments

    fallback_segments, detected_lang = _stable_fallback_source_asr(source_audio, config, source_duration)
    if _asr_is_too_sparse(fallback_segments, source_duration):
        return segments

    merged_segments = _merge_sparse_fill_segments(segments, fallback_segments)
    if merged_segments is segments:
        return segments
    print(
        "      Source ASR is sparse; merging stable fallback "
        f"segments={len(fallback_segments)} added={len(merged_segments) - len(segments)} "
        f"lang={detected_lang or 'auto'}"
    )
    write_srt(_debug_path(config, "sparse_source_fallback.srt"), fallback_segments, translated=False)
    if detected_lang:
        config.source_lang = detected_lang
    return merged_segments


def _fill_sparse_dub_segments(
    segments: list[Segment],
    config: DubConfig,
    source_audio: Path,
    source_duration: float,
) -> list[Segment]:
    # Do this first: a segment claiming a whole silent window would otherwise both
    # hide the hole from the check below and reject any fill landing inside it.
    segments = _tighten_overlong_segments(segments)
    if not _should_sparse_fill(config) or not _dub_is_too_sparse(segments, source_duration):
        return segments

    print("      Dub output is too sparse; trying stable ASR fill")
    _report_progress(config, "Добираю пустые места", 58, 100, "fallback ASR")
    fallback_source, detected_lang = _stable_fallback_source_asr(source_audio, config, source_duration)
    if _asr_is_too_sparse(fallback_source, source_duration):
        print("      Sparse fill skipped: fallback ASR is also sparse")
        return segments

    write_srt(_debug_path(config, "sparse_fill_source.srt"), fallback_source, translated=False)
    fill_config = copy(config)
    fill_config.source_lang = detected_lang or config.source_lang
    fill_config.force_source_language = False
    fill_config.artifact_source_lang = None
    fill_config.inject_artifacts = False
    if fill_config.input_pivot_lang and not fill_config.source_lang:
        fill_config.input_pivot_lang = None

    fallback_segments = [
        Segment(start=segment.start, end=segment.end, text=segment.text)
        for segment in fallback_source
    ]
    fallback_segments = _translate_dub_segments(fallback_segments, fill_config)
    fallback_segments = _maybe_distort_translations(
        fallback_segments,
        fill_config,
        output_path=_debug_path(config, "sparse_fill_translated_pivot.srt"),
        force=_should_distort_main_translation(config),
    )
    if config.collapse_repetitions:
        fallback_segments = collapse_repetitions_in_segments(
            fallback_segments,
            max_phrase_repeats=config.max_phrase_repeats,
            max_word_repeats=config.max_word_repeats,
        )
    write_srt(_debug_path(config, "sparse_fill_translated.srt"), fallback_segments, translated=True)

    merged = _merge_sparse_fill_segments(segments, fallback_segments)
    if merged is segments:
        print("      Sparse fill skipped: no non-overlapping fallback segments")
        return segments
    write_srt(_debug_path(config, "sparse_fill_merged.srt"), merged, translated=True)
    print(f"      Sparse fill added segments: {len(merged) - len(segments)}")
    _report_progress(config, "Пустые места заполнены", 60, 100, f"реплик: {len(merged)}")
    return merged


def _should_sparse_fill(config: DubConfig) -> bool:
    if config.tts.lower() == "none":
        return False
    if config.artifact_chaos_mode and config.inject_artifacts:
        return True
    return bool(config.input_pivot_lang and not config.force_source_language)


def _stable_fallback_source_asr(
    source_audio: Path,
    config: DubConfig,
    source_duration: float,
) -> tuple[list[Segment], str | None]:
    fallback = copy(config)
    fallback.asr_backend = "faster-whisper"
    fallback.whisper_model = "small"
    fallback.whisper_device = "cpu" if fallback.whisper_device == "auto" else fallback.whisper_device
    fallback.whisper_compute_type = "int8" if fallback.whisper_compute_type == "auto" else fallback.whisper_compute_type
    fallback.source_lang = None
    fallback.force_source_language = False
    fallback.glitch_profile = "clean"
    fallback.condition_on_previous_text = True
    fallback.initial_prompt = None
    fallback.hallucination_silence_threshold = None
    fallback.vad_filter = False
    fallback.inject_artifacts = False
    try:
        segments = transcribe(source_audio, fallback)
    except Exception as exc:
        print(f"      Stable fallback ASR skipped: {type(exc).__name__}: {exc}")
        return [], None

    segments = _clamp_segments_to_duration(segments, source_duration)
    segments = clean_segments(segments)
    return segments, fallback.source_lang


def _reference_timing_source_asr(
    source_audio: Path,
    config: DubConfig,
    source_duration: float,
) -> tuple[list[Segment], str | None]:
    """Run a quick clean auto-language pass and use only its speech intervals."""
    timing = copy(config)
    timing.source_lang = None
    timing.force_source_language = False
    timing.reference_timing_asr = False
    timing.glitch_profile = "clean"
    timing.initial_prompt = None
    timing.hallucination_silence_threshold = None
    timing.inject_artifacts = False
    timing.artifact_chaos_mode = False
    timing.suppress_plain_ascii_tokens = False
    try:
        segments = transcribe(source_audio, timing)
        segments = _clamp_segments_to_duration(segments, source_duration)
        segments = clean_segments(segments)
        if segments:
            return segments, timing.source_lang
    except Exception as exc:
        print(f"      Fast timing ASR failed; trying stable fallback: {type(exc).__name__}: {exc}")
    return _stable_fallback_source_asr(source_audio, config, source_duration)


def _merge_sparse_fill_segments(segments: list[Segment], fallback_segments: list[Segment]) -> list[Segment]:
    candidates = [segment for segment in fallback_segments if segment.spoken_text]
    if not candidates:
        return segments
    if not segments:
        return sorted(candidates, key=lambda item: (item.start, item.end))

    result = list(segments)
    added = 0
    for candidate in candidates:
        if candidate.duration < 0.15:
            continue
        if any(_fill_segment_conflicts(candidate, existing) for existing in result):
            continue
        result.append(candidate)
        added += 1
    if not added:
        return segments
    return sorted(result, key=lambda item: (item.start, item.end))


def _align_forced_text_to_reference_timings(
    forced_segments: list[Segment],
    timing_segments: list[Segment],
) -> list[Segment]:
    """Keep forced-ASR wording while borrowing only speech intervals from clean ASR."""
    references = sorted(
        (
            segment
            for segment in timing_segments
            if segment.end - segment.start >= 0.15 and segment.text.strip()
        ),
        key=lambda item: (item.start, item.end),
    )
    if not references:
        return forced_segments

    text_parts: list[str] = []
    previous = ""
    for segment in sorted(forced_segments, key=lambda item: (item.start, item.end)):
        text = " ".join(segment.text.split()).strip()
        signature = re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()
        if not text or (signature and signature == previous):
            continue
        text_parts.append(text)
        previous = signature
    if not text_parts:
        return []

    text = " ".join(text_parts)
    units = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]{1,4}|\S+", text)
    if not units:
        return []

    durations = [max(0.15, segment.duration) for segment in references]
    total_duration = sum(durations)
    buckets: list[list[str]] = [[] for _ in references]
    cumulative: list[float] = []
    elapsed = 0.0
    for duration in durations:
        elapsed += duration
        cumulative.append(elapsed / total_duration)

    reference_index = 0
    for unit_index, unit in enumerate(units):
        position = (unit_index + 0.5) / len(units)
        while reference_index < len(cumulative) - 1 and position > cumulative[reference_index]:
            reference_index += 1
        buckets[reference_index].append(unit)

    result: list[Segment] = []
    for reference, bucket in zip(references, buckets):
        if not bucket:
            continue
        result.append(Segment(start=reference.start, end=reference.end, text=" ".join(bucket)))
    return result


def _fill_segment_conflicts(candidate: Segment, existing: Segment) -> bool:
    if not existing.spoken_text:
        return False
    overlap = _overlap_seconds(candidate.start, candidate.end, existing.start, existing.end)
    if overlap <= 0.12:
        return False
    shortest = max(0.05, min(candidate.duration, existing.duration))
    return overlap / shortest >= 0.35


def _load_resumable_artifact_source(config: DubConfig, _artifact_lang: str) -> list[Segment]:
    # `forced_<lang>_transcript.srt` is also written by the main forced-ASR
    # pass before the artifact layer starts. Treating it as a resumable
    # artifact source makes every fresh job skip the short-window harvest.
    # Only this layer's own debug output is unambiguous and safe to resume.
    candidates = [_debug_path(config, "artifact_source.srt")]
    for path in candidates:
        if not _file_ready(path, min_size=16):
            continue
        try:
            segments = read_srt(path, translated=False)
        except Exception as exc:
            print(f"      Resume: could not load artifact source {path}: {type(exc).__name__}: {exc}")
            continue
        if segments:
            print(f"      Resume: loaded artifact source {path} segments={len(segments)}")
            return segments
    return []


def _translate_and_clean_artifacts(
    artifacts: list[Segment],
    artifact_config: DubConfig,
    output_path: Path,
) -> list[Segment]:
    artifacts = _translate_artifact_segments(artifacts, artifact_config)
    artifacts = _maybe_distort_translations(
        artifacts,
        artifact_config,
        output_path=output_path.with_name(f"{output_path.stem}_pivot.srt"),
        force=artifact_config.distort_translation,
    )
    if not artifact_config.artifact_chaos_mode:
        artifacts = collapse_repetitions_in_segments(
            artifacts,
            max_phrase_repeats=2,
            max_word_repeats=2,
        )
    write_srt(output_path, artifacts, translated=True)
    return artifacts


def _translate_dub_segments(segments: list[Segment], config: DubConfig) -> list[Segment]:
    if not segments:
        return segments

    pivot_lang = _normalize_lang(config.input_pivot_lang)
    if not pivot_lang:
        source_lang = _ensure_translation_source_lang(segments, config)
        if not source_lang and config.translator.lower() != "identity":
            print("      Direct translation skipped: source language is unknown; keeping source text")
            write_srt(_debug_path(config, "translation_unknown_source.srt"), segments, translated=False)
            for segment in segments:
                segment.translated_text = segment.text
            return segments
        return translate_segments(segments, config)

    source_lang = _ensure_translation_source_lang(segments, config)
    if not source_lang:
        print("      Input-pivot translation skipped: source language is unknown; keeping source text")
        write_srt(_debug_path(config, "input_pivot_unknown_source.srt"), segments, translated=False)
        for segment in segments:
            segment.translated_text = segment.text
        return segments

    print(f"      Input-pivot translation chain: {source_lang} -> {pivot_lang} -> {config.target_lang}")
    pivot_segments = [
        Segment(start=segment.start, end=segment.end, text=segment.text)
        for segment in segments
    ]
    if source_lang == pivot_lang:
        for segment in pivot_segments:
            segment.translated_text = segment.text
    else:
        pivot_config = copy(config)
        pivot_config.source_lang = source_lang
        pivot_config.target_lang = pivot_lang
        pivot_config.input_pivot_lang = None
        translate_segments(pivot_segments, pivot_config)

    input_path = config.workdir / f"input_{_safe_label(pivot_lang)}.srt"
    write_srt(input_path, pivot_segments, translated=True)
    write_srt(_debug_path(config, f"input_{_safe_label(pivot_lang)}_source_aligned.srt"), pivot_segments, translated=True)

    if pivot_lang == config.target_lang:
        for source_segment, pivot_segment in zip(segments, pivot_segments):
            source_segment.translated_text = pivot_segment.spoken_text
        return segments

    _report_progress(config, "Перевожу текст", 52, 100, f"цепочка {pivot_lang}->{config.target_lang}")
    target_segments = [
        Segment(start=segment.start, end=segment.end, text=segment.spoken_text)
        for segment in pivot_segments
    ]
    target_config = copy(config)
    target_config.source_lang = pivot_lang
    target_config.target_lang = config.target_lang
    target_config.input_pivot_lang = None
    translate_segments(target_segments, target_config)
    for source_segment, target_segment in zip(segments, target_segments):
        source_segment.translated_text = target_segment.spoken_text
    return segments


def _ensure_translation_source_lang(segments: list[Segment], config: DubConfig) -> str | None:
    source_lang = _normalize_lang(config.source_lang)
    if source_lang:
        return source_lang

    inferred = _infer_source_lang_from_segments(segments)
    if not inferred:
        return None

    config.source_lang = inferred
    _save_resume_state(config, source_lang=inferred)
    print(f"      Inferred source language from text: {inferred}")
    return inferred


def _infer_source_lang_from_segments(segments: list[Segment]) -> str | None:
    text = " ".join(segment.text for segment in segments if segment.text).strip()
    if not text:
        return None
    sample = text[:8000]

    script_counts = {
        "ru": _count_chars_in_ranges(sample, ((0x0400, 0x052F),)),
        "ko": _count_chars_in_ranges(sample, ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
        "ja": _count_chars_in_ranges(sample, ((0x3040, 0x30FF),)),
        "zh": _count_chars_in_ranges(sample, ((0x4E00, 0x9FFF),)),
        "th": _count_chars_in_ranges(sample, ((0x0E00, 0x0E7F),)),
        "ar": _count_chars_in_ranges(sample, ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))),
        "hi": _count_chars_in_ranges(sample, ((0x0900, 0x097F),)),
    }
    best_lang, best_count = max(script_counts.items(), key=lambda item: item[1])
    if best_count >= 3:
        return best_lang

    if re.search(r"[ăâêôơưđĂÂÊÔƠƯĐàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", sample):
        return "vi"
    if re.search(r"[ğüşöçıİĞÜŞÖÇ]", sample):
        return "tr"

    latin_letters = sum(1 for char in sample if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    if latin_letters >= 12:
        return "en"
    return None


def _count_chars_in_ranges(text: str, ranges: tuple[tuple[int, int], ...]) -> int:
    count = 0
    for char in text:
        codepoint = ord(char)
        if any(start <= codepoint <= end for start, end in ranges):
            count += 1
    return count


def _translation_progress_detail(config: DubConfig) -> str:
    pivot_lang = _normalize_lang(config.input_pivot_lang)
    if pivot_lang:
        return (
            f"provider={config.translator}, "
            f"цепочка={config.source_lang or 'auto'}->{pivot_lang}->{config.target_lang}"
        )
    return f"provider={config.translator}, цель={config.target_lang}"


def _normalize_lang(language: str | None) -> str | None:
    language = (language or "").strip()
    if not language or language == "auto":
        return None
    return language


def _artifact_corruption_language(
    selected_lang: str | None,
    detected_lang: str | None,
    config: DubConfig,
) -> str | None:
    selected = _normalize_lang(selected_lang)
    detected = _normalize_lang(detected_lang)
    if not selected or selected != detected or selected == "ru":
        return selected

    # A correctly selected non-Russian language used to disable the artifact
    # layer completely. Keep the stable automatic transcript as the content
    # backbone, but harvest the configured 20% artifact share with a
    # deliberately wrong Whisper language. Vietnamese is the most reliable
    # source of the familiar subtitle/subscribe hallucinations in our runs;
    # English is the fallback when the real speech is Vietnamese.
    artifact_lang = "en" if detected == "vi" else "vi"
    print(
        "      Same-language chaos fallback: "
        f"selected={selected} detected={detected} artifact={artifact_lang}"
    )
    return artifact_lang


def _artifact_initial_prompt(
    language: str | None,
    *,
    same_language: bool,
    seed_material: str | None = None,
) -> str | None:
    if language != "ru":
        return None
    seeds = list(_ru_artifact_prompt_seeds())
    if seed_material:
        seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8", errors="ignore")).digest()[:8], "big")
        random.Random(seed).shuffle(seeds)
    if same_language:
        return " ".join(seeds[:6])
    return " ".join(seeds[:3])


def _artifact_whisper_model(config: DubConfig, artifact_lang: str | None) -> str:
    return config.artifact_whisper_model or "turbo"


def _translate_artifact_segments(artifacts: list[Segment], artifact_config: DubConfig) -> list[Segment]:
    if not artifacts:
        return artifacts

    groups: dict[str, list[Segment]] = {}
    for segment in artifacts:
        source_lang = _artifact_translation_source_lang(segment.text, artifact_config.source_lang)
        groups.setdefault(source_lang, []).append(segment)

    if len(groups) > 1:
        summary = ", ".join(f"{lang}:{len(items)}" for lang, items in sorted(groups.items()))
        print(f"      Artifact translation language split: {summary}")

    for source_lang, group in groups.items():
        group_config = copy(artifact_config)
        group_config.source_lang = source_lang
        translate_segments(group, group_config)
    return artifacts


def _artifact_translation_source_lang(text: str, fallback_lang: str | None) -> str:
    if _looks_vietnamese_text(text):
        return "vi"
    if _looks_mostly_ascii(text) and _looks_like_meta_hallucination(text):
        return "en"
    return fallback_lang or "en"


def _looks_vietnamese_text(text: str) -> bool:
    normalized = text.casefold()
    if any(
        phrase in normalized
        for phrase in ("cảm ơn", "cám ơn", "đăng ký", "các bạn", "theo dõi", "ghiền mì gõ")
    ):
        return True
    return bool(re.search(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", normalized))


def _looks_mostly_ascii(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return True
    ascii_letters = sum(1 for char in letters if ord(char) < 128)
    return ascii_letters / len(letters) >= 0.85


def _clean_artifact_source_segments(segments: list[Segment]) -> list[Segment]:
    cleaned: list[Segment] = []
    dropped_loops = 0
    for segment in segments:
        segment.text = " ".join(segment.text.split()).strip()
        if not segment.text:
            continue
        if _is_text_repetition_loop(segment.text):
            dropped_loops += 1
            continue
        cleaned.append(segment)

    cleaned = collapse_repetitions_in_segments(
        cleaned,
        max_phrase_repeats=2,
        max_word_repeats=2,
        max_ngram_words=8,
    )
    if is_repetitive_loop(cleaned, min_segments=4, repeated_share=0.65):
        before = len(cleaned)
        cleaned = _dedupe_artifact_segments(cleaned)
        print(f"      Artifact loop cleanup: deduped={before - len(cleaned)}")
    if dropped_loops:
        print(f"      Artifact loop cleanup: dropped_loop_segments={dropped_loops}")
    return cleaned


def _clean_chaos_artifact_source_segments(segments: list[Segment]) -> list[Segment]:
    cleaned: list[Segment] = []
    for segment in segments:
        segment.text = " ".join(segment.text.split()).strip()
        if segment.text:
            cleaned.append(segment)
    return cleaned


def _filter_same_language_ru_artifacts(segments: list[Segment]) -> list[Segment]:
    result: list[Segment] = []
    seen_signatures: set[str] = set()
    seen_segment_keys: set[tuple[int, int, str]] = set()
    used_signature_counts: dict[str, int] = {}
    seen_per_family: dict[str, int] = {}
    candidates = [
        segment
        for segment in sorted(segments, key=lambda item: (item.start, item.end))
        if _looks_like_meta_hallucination(segment.text)
    ]
    max_total = 8
    min_total = min(6, len(candidates))

    for segment in candidates:
        if not _looks_like_meta_hallucination(segment.text):
            continue
        signature = _artifact_signature(segment.text)
        if signature in seen_signatures:
            continue
        family = _artifact_family(segment.text)
        if seen_per_family.get(family, 0) >= 2:
            continue
        seen_signatures.add(signature)
        seen_segment_keys.add(_artifact_segment_key(segment))
        used_signature_counts[signature] = used_signature_counts.get(signature, 0) + 1
        seen_per_family[family] = seen_per_family.get(family, 0) + 1
        result.append(segment)
        if len(result) >= max_total:
            break

    if len(result) >= min_total:
        return result

    for segment in candidates:
        signature = _artifact_signature(segment.text)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        seen_segment_keys.add(_artifact_segment_key(segment))
        used_signature_counts[signature] = used_signature_counts.get(signature, 0) + 1
        result.append(segment)
        if len(result) >= min_total or len(result) >= max_total:
            break

    if len(result) >= min_total:
        return result

    for segment in candidates:
        segment_key = _artifact_segment_key(segment)
        if segment_key in seen_segment_keys:
            continue
        signature = _artifact_signature(segment.text)
        if used_signature_counts.get(signature, 0) >= 2:
            continue
        seen_segment_keys.add(segment_key)
        used_signature_counts[signature] = used_signature_counts.get(signature, 0) + 1
        result.append(segment)
        if len(result) >= min_total or len(result) >= max_total:
            break
    return result


def _fill_sparse_artifacts_from_history(
    artifacts: list[Segment],
    config: DubConfig,
    artifact_lang: str | None,
    base_segments: list[Segment],
    source_duration: float,
) -> list[Segment]:
    if config.target_lang != "ru":
        return artifacts

    max_total = min(max(0, config.artifact_max_segments), 8)
    if max_total <= 0:
        return artifacts
    min_total = min(5, max_total)
    ratio_limit = round(len(base_segments) * max(0.0, min(1.0, config.artifact_ratio)))
    if config.artifact_ratio > 0.0 and ratio_limit <= 0 and base_segments:
        ratio_limit = 1
    injection_slots = min(max_total, ratio_limit)
    # Always reserve part of the existing artifact allowance for the familiar
    # legacy corpus. This does not raise the 20% cap: the final injector still
    # chooses at most injection_slots candidates in total.
    legacy_quota = min(2, injection_slots)
    sparse_needed = max(0, min_total - len(artifacts))
    needed = max(legacy_quota, sparse_needed)
    if needed <= 0:
        return artifacts
    borrowed = _borrow_historical_artifacts(config, artifacts, base_segments, source_duration, needed)
    if not borrowed:
        return artifacts

    print(f"      Legacy/historical artifact mix: added={len(borrowed)} quota={legacy_quota}")
    write_srt(_debug_path(config, "artifact_borrowed.srt"), borrowed, translated=True)
    return [*artifacts, *borrowed]


def _borrow_historical_artifacts(
    config: DubConfig,
    existing: list[Segment],
    base_segments: list[Segment],
    source_duration: float,
    needed: int,
) -> list[Segment]:
    source_duration = max(source_duration, max((segment.end for segment in base_segments), default=0.0))
    if needed <= 0 or source_duration <= 0.5:
        return []

    existing_signatures = {_artifact_signature(segment.spoken_text or segment.text) for segment in existing}
    texts = _historical_artifact_texts(config, existing_signatures, limit=needed * 2)
    if not texts:
        return []

    slots = _historical_artifact_slots(base_segments, existing, source_duration, len(texts))
    borrowed: list[Segment] = []
    for text, (slot_start, slot_end) in zip(texts, slots):
        duration = min(max(1.2, _estimated_chaos_spoken_duration(text)), max(0.4, slot_end - slot_start))
        start = max(0.0, min(source_duration, slot_start))
        end = min(source_duration, start + duration)
        if end - start < 0.4:
            continue
        borrowed.append(Segment(start=start, end=end, text=text, translated_text=text))
        if len(borrowed) >= needed:
            break
    return borrowed


def _historical_artifact_texts(
    config: DubConfig,
    existing_signatures: set[str],
    *,
    limit: int,
) -> list[str]:
    seed_material = f"{_translation_seed_base(config)}|historical-artifacts"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8", errors="ignore")).digest()[:8], "big")
    rng = random.Random(seed)

    legacy_pool = [
        text
        for text in _legacy_artifact_inserts_ru()
        if _artifact_signature(text) not in existing_signatures
    ]
    rng.shuffle(legacy_pool)

    result: list[str] = []
    family_counts: dict[str, int] = {}
    for text in legacy_pool:
        family = _artifact_family(text)
        if family_counts.get(family, 0) >= 2:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        result.append(text)
        if len(result) >= limit:
            return result

    # The editable legacy bank normally satisfies the requested quota without
    # scanning thousands of old runs on F:. Use real historical artifacts only
    # as a fallback when the bank is too small or heavily deduplicated.
    runs_root = _runs_root_from_workdir(config.workdir)
    if runs_root is None or not runs_root.exists():
        return result

    paths = [
        path
        for pattern in (
            "*/**/work/debug/artifact_translated.srt",
            "*/**/work/debug/artifact_injected.srt",
            "*/**/work/debug/artifact_source.srt",
        )
        for path in runs_root.glob(pattern)
        if path.is_file() and config.workdir not in path.parents
    ]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    history_pool: list[str] = []
    seen = set(existing_signatures)
    seen.update(_artifact_signature(text) for text in legacy_pool)
    for path in paths[:100]:
        try:
            segments = read_srt(path, translated=True)
            if not segments:
                segments = read_srt(path, translated=False)
        except Exception:
            continue
        for segment in segments:
            text = " ".join((segment.spoken_text or segment.text).split()).strip()
            if not text or not _looks_like_meta_hallucination(text):
                continue
            if not _artifact_text_matches_target_language(text, config.target_lang):
                continue
            signature = _artifact_signature(text)
            if signature in seen:
                continue
            seen.add(signature)
            history_pool.append(text)
    rng.shuffle(history_pool)
    for text in history_pool:
        family = _artifact_family(text)
        if family_counts.get(family, 0) >= 2:
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _artifact_text_matches_target_language(text: str, target_lang: str | None) -> bool:
    """Keep historical inserts in the language the user selected for output."""
    if target_lang != "ru":
        return True
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    cyrillic = sum(1 for char in letters if "\u0400" <= char <= "\u052f")
    return cyrillic / len(letters) >= 0.35


def _runs_root_from_workdir(workdir: Path) -> Path | None:
    for parent in workdir.parents:
        if parent.name in {"bot", "bot-release"}:
            return parent
    return None


def _historical_artifact_slots(
    base_segments: list[Segment],
    existing_artifacts: list[Segment],
    source_duration: float,
    needed: int,
) -> list[tuple[float, float]]:
    occupied = sorted(
        [
            (max(0.0, segment.start), min(source_duration, segment.end))
            for segment in [*base_segments, *existing_artifacts]
            if segment.end > segment.start
        ]
    )
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in occupied:
        if start - cursor >= 1.0:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if source_duration - cursor >= 1.0:
        gaps.append((cursor, source_duration))

    slots: list[tuple[float, float]] = []
    for start, end in sorted(gaps, key=lambda item: item[1] - item[0], reverse=True):
        slots.append((start + 0.1, end - 0.1))
        if len(slots) >= needed:
            return slots

    step = source_duration / max(1, needed + 1)
    for index in range(needed):
        center = min(source_duration - 0.7, max(0.7, step * (index + 1)))
        slots.append((center - 0.6, center + 0.8))
    return slots[:needed]


def _artifact_segment_key(segment: Segment) -> tuple[int, int, str]:
    return (round(segment.start * 100), round(segment.end * 100), _artifact_signature(segment.text))


def _artifact_signature(text: str) -> str:
    normalized = re.sub(r"[^\w\s]+", " ", text.casefold(), flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.replace("dima torzok", "dimatorzok")
    return normalized


def _artifact_family(text: str) -> str:
    normalized = _artifact_signature(text)
    if "lalaschool" in normalized or "la la school" in normalized or "ла ла скул" in normalized or "лалашкол" in normalized:
        return "lalaschool"
    if "большая семья" in normalized:
        return "big_family"
    if "девушки отдыхают" in normalized or "девушка отдыхает" in normalized or "девочки отдыхают" in normalized:
        return "girls_resting"
    if "продолжение следует" in normalized:
        return "continued"
    if "редактор" in normalized or "корректор" in normalized or "семкин" in normalized or "егорова" in normalized:
        return "editor_credit"
    if "спасибо" in normalized and "субтитр" in normalized:
        return "thanks_subtitles"
    if "подогнал" in normalized:
        return "subtitle_sync"
    if "подпис" in normalized or "subscribe" in normalized or "abonnieren" in normalized:
        return "subscribe"
    if "amara" in normalized:
        return "amara"
    if (
        "субтитр" in normalized
        or "subtitle" in normalized
        or "caption" in normalized
        or "dimatorzok" in normalized
        or "дубровск" in normalized
    ):
        return "subtitle_credit"
    return "meta"


def _whisper_chaos_audio(source_audio: Path, config: DubConfig, *, purpose: str) -> Path:
    if not (config.artifact_chaos_mode and config.force_source_language and config.source_lang):
        return source_audio

    output_path = config.workdir / f"{purpose}_whisper_loud.wav"
    if output_path.exists() and output_path.stat().st_size > 1024:
        return output_path

    try:
        print(f"      Preparing loudness-boosted Whisper audio: {purpose}")
        make_whisper_chaos_audio(source_audio, output_path, gain_db=50.0)
        return output_path
    except Exception as exc:
        print(f"      Loudness-boosted Whisper audio skipped: {type(exc).__name__}: {exc}")
        return source_audio


def _apply_experimental_censor(
    segments: list[Segment],
    config: DubConfig,
    *,
    stage: str,
) -> tuple[list[Segment], bool]:
    percent = max(0, min(100, int(config.censor_percent or 0)))
    if percent <= 0:
        return segments, False
    result = apply_censor_to_segments(
        segments,
        percent=percent,
        seed=f"{_translation_seed_base(config)}|{stage}",
    )
    return result.segments, result.replacements > 0


def _harvest_chunked_forced_artifacts(
    source_audio: Path,
    artifact_config: DubConfig,
    config: DubConfig,
    source_duration: float,
) -> list[Segment]:
    if source_duration <= 4.0:
        return []

    chunk_dir = config.workdir / "artifact_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    windows = _artifact_harvest_windows(source_duration)
    if not windows:
        return []

    print(f"      Chaos artifact chunk harvest: windows={len(windows)}")
    chunk_config = copy(artifact_config)
    chunk_config.condition_on_previous_text = True
    result: list[Segment] = []
    for index, (start, duration) in enumerate(windows, start=1):
        chunk_config.initial_prompt = None
        chunk_path = chunk_dir / f"{index:04d}_{start:.2f}_{duration:.2f}.wav"
        try:
            extract_wav_slice(source_audio, chunk_path, start, duration)
            chunk_segments = transcribe(chunk_path, chunk_config)
            chunk_segments = _clamp_segments_to_duration(chunk_segments, duration)
            # Short independent decodes are deliberately prone to useful
            # hallucinations, but occasionally end in hundreds of copies of
            # one word or phrase. Keep that flavour without letting the tail
            # dominate the translated subtitle and TTS duration.
            chunk_segments = collapse_repetitions_in_segments(
                chunk_segments,
                max_phrase_repeats=4,
                max_word_repeats=4,
                max_ngram_words=8,
            )
            result.extend(_offset_segments(chunk_segments, start, source_duration))
        except Exception as exc:
            print(f"      Chaos artifact chunk skipped {index}/{len(windows)}: {type(exc).__name__}: {exc}")

    if result:
        write_srt(_debug_path(config, "artifact_chunk_source_raw.srt"), result, translated=False)
        print(f"      Chaos artifact chunk candidates: {len(result)}")
    return result


def _artifact_harvest_windows(source_duration: float) -> list[tuple[float, float]]:
    chunk_seconds = 5.0
    max_windows = 24
    if source_duration < 0.5:
        return []

    starts = [index * chunk_seconds for index in range(int(source_duration // chunk_seconds) + 1)]
    windows = [
        (start, min(chunk_seconds, source_duration - start))
        for start in starts
        if source_duration - start >= 0.5
    ]
    if len(windows) <= max_windows:
        return windows

    # Do not make long videos spend minutes decoding hundreds of tiny files.
    # Sample the same 5-second windows evenly over the complete timeline.
    last_start = max(0.0, source_duration - chunk_seconds)
    step = last_start / (max_windows - 1)
    return [
        (min(last_start, round(index * step, 3)), chunk_seconds)
        for index in range(max_windows)
    ]


def _offset_segments(segments: list[Segment], offset: float, source_duration: float) -> list[Segment]:
    result: list[Segment] = []
    for segment in segments:
        start = max(0.0, min(source_duration, segment.start + offset))
        end = max(start, min(source_duration, segment.end + offset))
        if end - start < 0.05:
            continue
        result.append(
            Segment(
                start=start,
                end=end,
                text=segment.text,
                translated_text=segment.translated_text,
                speaker_wav=segment.speaker_wav,
                speaker_id=segment.speaker_id,
                speaker_ref_text=segment.speaker_ref_text,
            )
        )
    return result


def _is_text_repetition_loop(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < 24:
        return False

    collapsed = collapse_repetitions(
        compact,
        max_phrase_repeats=2,
        max_word_repeats=2,
        max_ngram_words=8,
    )
    if len(compact) >= 48 and len(collapsed) <= len(compact) * 0.45:
        return True

    normalized = re.sub(r"[\W_]+", "", compact, flags=re.UNICODE).casefold()
    if len(normalized) < 16:
        return False
    max_unit = min(24, len(normalized) // 3)
    for unit_size in range(2, max_unit + 1):
        unit = normalized[:unit_size]
        repeats = len(normalized) // unit_size
        if repeats < 4:
            continue
        repeated = (unit * (repeats + 1))[: len(normalized)]
        matches = sum(1 for left, right in zip(normalized, repeated) if left == right)
        if matches / len(normalized) >= 0.88:
            return True
    return False


def _should_distort_main_translation(config: DubConfig) -> bool:
    if not config.distort_translation:
        return False
    if config.distort_main_translation:
        return True
    artifact_lang = config.artifact_source_lang
    if not artifact_lang or artifact_lang == "auto":
        return False
    return not (config.source_lang and artifact_lang == config.source_lang)


def _maybe_distort_translations(
    segments: list[Segment],
    config: DubConfig,
    *,
    output_path: Path | None = None,
    force: bool = False,
) -> list[Segment]:
    chains = _translation_distortion_chains(config)
    if not force or not chains:
        return segments

    summary = " | ".join(" -> ".join(chain) for chain in chains)
    print(f"      Distorting translation through {len(chains)} chain variant(s): {summary}")
    changed = 0
    fallback_count = 0
    second_pass_count = 0
    second_pass_failures = 0
    for index, segment in enumerate(segments):
        text = (segment.translated_text or segment.text).strip()
        if not text:
            continue
        if _looks_like_meta_hallucination(segment.text) or _looks_like_meta_hallucination(text):
            continue

        chain = _select_translation_distortion_chain(config, chains, index, text, pass_index=0)
        try:
            distorted = translate_text_chain(text, chain, config).strip()
            if _bad_pivot_result(distorted, text):
                fallback_count += 1
                if fallback_count <= 3:
                    print("      Pivot output is unreadable; using local telephone distortion fallback")
                distorted = _telephone_distort_text(text)
        except Exception as exc:
            fallback_count += 1
            if fallback_count <= 3:
                print(
                    "      Pivot chain failed; using local telephone distortion fallback: "
                    f"{' -> '.join(chain)}: {type(exc).__name__}: {exc}"
                )
            distorted = _telephone_distort_text(text)

        if distorted and _should_apply_translation_second_pass(config, index, text):
            second_chain = _select_translation_distortion_chain(config, chains, index, distorted, pass_index=1)
            try:
                second_distorted = translate_text_chain(distorted, second_chain, config).strip()
                if not _bad_pivot_result(second_distorted, distorted):
                    distorted = second_distorted
                    second_pass_count += 1
            except Exception as exc:
                second_pass_failures += 1
                if second_pass_failures <= 3:
                    print(
                        "      Second pivot pass skipped after failure: "
                        f"{' -> '.join(second_chain)}: {type(exc).__name__}: {exc}"
                    )

        if distorted:
            segment.translated_text = distorted
            changed += int(distorted != text)

    if output_path is not None:
        write_srt(output_path, segments, translated=True)
    fallback_detail = f", fallback={fallback_count}" if fallback_count else ""
    second_pass_detail = f", pass2={second_pass_count}" if second_pass_count else ""
    print(f"      Distorted translated segments: {changed}/{len(segments)}{fallback_detail}{second_pass_detail}")
    return segments


def _bad_pivot_result(distorted: str, original: str) -> bool:
    distorted = distorted.strip()
    if not distorted:
        return True
    question_marks = distorted.count("?")
    if question_marks >= 6 and question_marks / max(1, len(distorted)) > 0.2:
        return True
    letters = sum(char.isalpha() for char in distorted)
    if len(original) > 12 and letters < 3:
        return True
    return False


def _telephone_distort_text(text: str) -> str:
    text = " ".join(text.split()).strip()
    if not text:
        return text

    replacements = [
        (r"\bговорит\b", "комментирует"),
        (r"\bсказал\b", "сделал сказать"),
        (r"\bскажи\b", "сделай сказать"),
        (r"\bпойду\b", "буду идти"),
        (r"\bпошел\b", "пошел быть"),
        (r"\bпришел\b", "прибыл"),
        (r"\bприходят\b", "прибывают"),
        (r"\bкупили\b", "приобрели"),
        (r"\bпринеси\b", "принеси для меня"),
        (r"\bвозьми\b", "возьми это"),
        (r"\bванную\b", "ванную комнату"),
        (r"\bванной\b", "ванной комнате"),
        (r"\bбыстренько\b", "быстро быстро"),
        (r"\bзнаешь\b", "вы знаете"),
        (r"\bтебе\b", "для тебя"),
        (r"\bмне\b", "для меня"),
        (r"\bсейчас\b", "теперь сейчас"),
        (r"\bпотом\b", "после этого"),
        (r"\bдомой\b", "в дом"),
        (r"\bмилый\b", "сладкий"),
    ]
    distorted = text
    for pattern, replacement in replacements:
        distorted = re.sub(pattern, replacement, distorted, flags=re.IGNORECASE)

    parts = [part.strip() for part in re.split(r"([,.;:!?])", distorted) if part.strip()]
    rebuilt: list[str] = []
    clause_index = 0
    for part in parts:
        if re.fullmatch(r"[,.;:!?]", part):
            rebuilt.append(part)
            continue
        clause_index += 1
        if clause_index % 3 == 0 and len(part.split()) >= 3:
            part = "то есть " + part
        elif clause_index % 4 == 0 and len(part.split()) >= 3:
            part = part + ", значит"
        rebuilt.append(part)

    result = ""
    for part in rebuilt:
        if re.fullmatch(r"[,.;:!?]", part):
            result = result.rstrip() + part + " "
        else:
            result += part + " "
    result = re.sub(r"\s+", " ", result).strip()
    return result or text


_DEEP_PHONETIC_VARIANTS = {
    "гена": ("Джена", "Генай", "Генна"),
    "геннадий": ("Генадия", "Дженнадий", "Генандий"),
    "чебурашка": ("Чевлашка", "Чебураха", "Чеворшка"),
    "полотенце": ("полотенза", "потенциг", "полотенсия"),
    "полотенца": ("полотензы", "потенцига", "полотенсии"),
    "крокодил": ("кракодил", "крокодиль", "крокадила"),
    "ванную": ("вандай", "ваннуй", "ванную комнату"),
    "ванной": ("ванней", "вандальной", "ванной комнате"),
    "косяк": ("косак", "косячек", "космик"),
    "косячок": ("косяжок", "косачок", "косячер"),
    "телевизор": ("телевизион", "телевизер", "телевизорный"),
    "шкаф": ("шкаб", "шкафий", "шкав"),
}


def _apply_deep_phonetic_chaos(segments: list[Segment], config: DubConfig) -> list[Segment]:
    changed = 0
    for index, segment in enumerate(segments):
        text = segment.spoken_text
        if not text:
            continue
        seed_material = f"{_translation_seed_base(config)}|deep-phonetic|{index}|{text[:96]}"
        rng = random.Random(
            int.from_bytes(hashlib.sha256(seed_material.encode("utf-8", errors="ignore")).digest()[:8], "big")
        )

        def replace_known(match: re.Match[str]) -> str:
            word = match.group(0)
            variants = _DEEP_PHONETIC_VARIANTS.get(word.casefold())
            if not variants or rng.random() > 0.68:
                return word
            replacement = rng.choice(variants)
            return replacement if word[:1].isupper() else replacement.casefold()

        names = "|".join(
            sorted((re.escape(item) for item in _DEEP_PHONETIC_VARIANTS), key=len, reverse=True)
        )
        damaged = re.sub(rf"\b(?:{names})\b", replace_known, text, flags=re.IGNORECASE)
        damaged = re.sub(r"\bпотому что\b", "патамушта", damaged, flags=re.IGNORECASE)
        damaged = re.sub(r"\bчто\b", "што", damaged, flags=re.IGNORECASE)
        damaged = re.sub(r"\bсейчас\b", "щас", damaged, flags=re.IGNORECASE)
        words = damaged.split()
        for position, word in enumerate(words):
            bare = re.sub(r"[^А-Яа-яЁё]", "", word)
            if len(bare) < 6 or rng.random() > 0.26:
                continue
            vowels = [offset for offset, char in enumerate(word) if char.casefold() in "аеёиоуыэюя"]
            if len(vowels) < 2:
                continue
            offset = rng.choice(vowels[1:])
            words[position] = word[:offset] + rng.choice("аеоиуы") + word[offset + 1 :]
        damaged = " ".join(words)
        damaged = re.sub(r"\s+", " ", damaged).strip()
        if damaged and damaged != text:
            segment.translated_text = damaged
            changed += 1
    print(f"      Deep phonetic chaos: changed={changed}/{len(segments)}")
    return segments


def _translation_distortion_chains(config: DubConfig) -> list[list[str]]:
    target = config.target_lang.strip()
    if not target:
        return []

    chains: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_variant in config.translation_pivots.split("|"):
        pivots = [
            _normalize_pivot_token(item.strip(), config)
            for item in raw_variant.replace(";", ",").replace(" ", ",").split(",")
            if item.strip()
        ]
        chain = [target]
        for pivot in pivots:
            if pivot and pivot != chain[-1]:
                chain.append(pivot)
        if chain[-1] != target:
            chain.append(target)
        key = tuple(chain)
        if len(chain) >= 3 and key not in seen:
            seen.add(key)
            chains.append(chain)
    return chains


def _normalize_pivot_token(token: str, config: DubConfig) -> str | None:
    normalized = token.casefold()
    if normalized in {"input", "{input}", "artifact", "{artifact}"}:
        return config.artifact_source_lang or config.input_pivot_lang or None
    if normalized in {"source", "{source}"}:
        return config.source_lang or None
    return token


def _translation_seed_base(config: DubConfig) -> str:
    return str(config.translation_seed or config.workdir)


def _select_translation_distortion_chain(
    config: DubConfig,
    chains: list[list[str]],
    segment_index: int,
    text: str,
    *,
    pass_index: int = 0,
) -> list[str]:
    if len(chains) == 1:
        return chains[0]
    seed_material = (
        f"{_translation_seed_base(config)}|{config.translation_chaos}|"
        f"{segment_index}|{pass_index}|{text[:96]}|{config.translation_pivots}"
    )
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8", errors="ignore")).digest()[:8], "big")
    return random.Random(seed).choice(chains)


def _should_apply_translation_second_pass(config: DubConfig, segment_index: int, text: str) -> bool:
    ratio = max(0.0, min(1.0, config.translation_second_pass_ratio))
    if ratio <= 0.0:
        return False
    seed_material = (
        f"{_translation_seed_base(config)}|{config.translation_chaos}|"
        f"second-pass|{segment_index}|{text[:96]}|{config.translation_pivots}"
    )
    bucket = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8", errors="ignore")).digest()[:8], "big")
    return (bucket / float(1 << 64)) < ratio


def _looks_like_meta_hallucination(text: str) -> bool:
    return bool(text and _meta_hallucination_re().search(text))


def _dedupe_artifact_segments(segments: list[Segment]) -> list[Segment]:
    result: list[Segment] = []
    seen: set[str] = set()
    for segment in segments:
        key = " ".join(segment.spoken_text.casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(segment)
    return result


def _artifact_injection_limit(
    base_segments: list[Segment],
    candidates: list[Segment],
    config: DubConfig,
) -> int:
    if not base_segments or not candidates:
        return 0
    source_count = len(base_segments)
    if source_count < max(0, config.artifact_min_source_segments):
        print(
            "      Artifact injection limited: "
            f"source_segments={source_count} < min={config.artifact_min_source_segments}"
        )
        return 0

    absolute_max = max(0, int(config.artifact_max_segments))
    if absolute_max <= 0:
        return 0

    ratio = max(0.0, min(1.0, float(config.artifact_ratio)))
    ratio_limit = round(source_count * ratio)
    if ratio > 0.0 and ratio_limit <= 0:
        ratio_limit = 1
    limit = min(absolute_max, ratio_limit, len(candidates))
    print(
        "      Artifact injection limit: "
        f"source_segments={source_count}, ratio={ratio:.2f}, "
        f"ratio_limit={ratio_limit}, absolute_max={absolute_max}, candidates={len(candidates)}, limit={limit}"
    )
    return max(0, limit)


def _limit_chaos_artifacts_before_translation(
    candidates: list[Segment],
    base_segments: list[Segment],
    config: DubConfig,
) -> list[Segment]:
    if not config.artifact_chaos_mode or not candidates:
        return candidates
    needed = _artifact_injection_limit(base_segments, candidates, config)
    if needed <= 0:
        return []
    # Keep a small reserve for timing conflicts or failed translations, but do
    # not run expensive multilingual chains over a bank that cannot be used.
    limit = min(len(candidates), max(needed, needed * 2))
    if limit >= len(candidates):
        return candidates
    ranked = sorted(candidates, key=_chaos_artifact_rank, reverse=True)
    print(f"      Artifact pre-translation limit: {len(candidates)} -> {limit} (needed={needed})")
    return ranked[:limit]


def _inject_artifact_segments(
    segments: list[Segment],
    artifacts: list[Segment],
    config: DubConfig,
    source_duration: float,
) -> list[Segment]:
    candidates = suppress_pathological_segment_loops(
        [artifact for artifact in artifacts if artifact.spoken_text]
    )
    candidates = limit_repeated_segment_phrases(candidates)
    if not candidates:
        return segments
    if config.artifact_chaos_mode:
        return _inject_chaos_artifact_segments(segments, candidates, config, source_duration)
    if not segments:
        return segments

    max_count = _artifact_injection_limit(segments, candidates, config)
    if max_count <= 0:
        return segments
    replacements: list[Segment] = []
    used_segment_indices: set[int] = set()
    used_ranges: list[tuple[float, float]] = []
    for artifact in candidates[:max_count]:
        text = _shorten_artifact_text(artifact.spoken_text)
        if not text:
            continue

        start, end = _artifact_replacement_interval(artifact, text, source_duration)
        if end <= start or any(_ranges_overlap(start, end, used_start, used_end) for used_start, used_end in used_ranges):
            continue

        replaced_indices = _artifact_replaced_segment_indices(
            segments,
            start,
            end,
            used_segment_indices,
        )
        if not replaced_indices:
            continue

        used_segment_indices.update(replaced_indices)
        used_ranges.append((start, end))
        replacements.append(
            Segment(
                start=start,
                end=end,
                text=artifact.text,
                translated_text=text,
            )
        )

    if not replacements:
        return segments

    print(f"      Replaced segments with artifacts: {len(replacements)}")
    result = sorted(
        [
            *(segment for index, segment in enumerate(segments) if index not in used_segment_indices),
            *replacements,
        ],
        key=lambda item: (item.start, item.end),
    )
    write_srt(
        _debug_path(config, "artifact_injected.srt"),
        sorted(replacements, key=lambda item: (item.start, item.end)),
        translated=True,
    )
    return result


def _inject_chaos_artifact_segments(
    segments: list[Segment],
    candidates: list[Segment],
    config: DubConfig,
    source_duration: float,
) -> list[Segment]:
    max_count = _artifact_injection_limit(segments, candidates, config)
    if max_count <= 0:
        return segments
    ranked = sorted(candidates, key=_chaos_artifact_rank, reverse=True)
    replacements: list[Segment] = []
    used_ranges: list[tuple[float, float]] = []

    for artifact in ranked[:max_count]:
        text = _shorten_artifact_text(artifact.spoken_text, max_words=32, max_chars=240)
        if not text:
            continue

        start, end = _chaos_artifact_replacement_interval(artifact, text, source_duration)
        if end <= start:
            continue
        if any(_chaos_ranges_conflict(start, end, used_start, used_end) for used_start, used_end in used_ranges):
            continue

        used_ranges.append((start, end))
        replacements.append(
            Segment(
                start=start,
                end=end,
                text=artifact.text,
                translated_text=text,
            )
        )

    if not replacements:
        return segments

    replacement_ranges = [(segment.start, segment.end) for segment in replacements]
    kept_segments = [
        segment
        for segment in segments
        if not _base_segment_replaced_by_chaos_artifact(segment, replacement_ranges)
    ]
    removed_count = len(segments) - len(kept_segments)
    print(f"      Chaos artifact replacements: {len(replacements)}, base_removed={removed_count}")

    sorted_replacements = sorted(replacements, key=lambda item: (item.start, item.end))
    write_srt(_debug_path(config, "artifact_injected.srt"), sorted_replacements, translated=True)
    return sorted([*kept_segments, *sorted_replacements], key=lambda item: (item.start, item.end))


def _chaos_artifact_rank(artifact: Segment) -> tuple[int, int, float, int, float]:
    text = artifact.spoken_text
    is_meta = int(_looks_like_meta_hallucination(text) or _looks_like_meta_hallucination(artifact.text))
    words = len(text.split())
    has_body = int(words >= 3)
    return (is_meta, has_body, min(artifact.duration, 30.0), words, -artifact.start)


def _chaos_artifact_replacement_interval(artifact: Segment, text: str, source_duration: float) -> tuple[float, float]:
    source_duration = max(0.0, source_duration)
    if source_duration <= 0.0:
        return 0.0, 0.0

    start = max(0.0, min(source_duration, artifact.start))
    end = max(start, min(source_duration, artifact.end))
    current_duration = end - start
    spoken_duration = _estimated_chaos_spoken_duration(text)
    if current_duration <= spoken_duration * 1.35:
        target_duration = max(current_duration, spoken_duration)
    else:
        target_duration = spoken_duration
    target_duration = min(source_duration, target_duration)
    if current_duration >= target_duration * 0.85 and current_duration <= target_duration * 1.35:
        return start, end

    hint = _artifact_timing_hint(artifact, source_duration)
    if hint is None:
        hint = start + current_duration / 2.0
    if current_duration > target_duration * 1.35:
        start = start + min(0.25, max(0.0, current_duration - target_duration) * 0.1)
    else:
        start = hint - target_duration / 2.0
    end = start + target_duration
    if start < 0.0:
        start = 0.0
        end = target_duration
    if end > source_duration:
        end = source_duration
        start = max(0.0, end - target_duration)
    return start, end


def _estimated_chaos_spoken_duration(text: str) -> float:
    words = max(1, len(text.split()))
    chars = len(text)
    word_estimate = words * 0.42
    char_estimate = chars / 18.0
    return max(1.15, min(7.5, max(word_estimate, char_estimate)))


def _chaos_ranges_conflict(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    overlap = _overlap_seconds(left_start, left_end, right_start, right_end)
    if overlap <= 0.12:
        return False
    shortest = max(0.05, min(left_end - left_start, right_end - right_start))
    return overlap / shortest >= 0.35


def _base_segment_replaced_by_chaos_artifact(segment: Segment, ranges: list[tuple[float, float]]) -> bool:
    if segment.duration <= 0.0:
        return False
    for start, end in ranges:
        overlap = _overlap_seconds(segment.start, segment.end, start, end)
        if overlap >= 0.6 or overlap / segment.duration >= 0.35:
            return True
    return False


def _artifact_replacement_interval(artifact: Segment, text: str, source_duration: float) -> tuple[float, float]:
    duration = _artifact_replacement_duration(text)
    hint = _artifact_timing_hint(artifact, source_duration)
    if hint is None:
        hint = max(0.0, min(source_duration, artifact.start))

    start = hint - duration / 2.0
    end = start + duration
    if start < 0.0:
        start = 0.0
        end = min(source_duration, duration)
    if end > source_duration:
        end = source_duration
        start = max(0.0, end - duration)
    return start, end


def _artifact_replacement_duration(text: str) -> float:
    words = max(1, len(text.split()))
    return max(1.4, min(7.5, words * 0.38))


def _artifact_replaced_segment_indices(
    segments: list[Segment],
    replacement_start: float,
    replacement_end: float,
    used_indices: set[int],
) -> list[int]:
    overlapping = [
        index
        for index, segment in enumerate(segments)
        if index not in used_indices and _overlap_seconds(segment.start, segment.end, replacement_start, replacement_end) >= 0.05
    ]
    if overlapping:
        return overlapping

    available = [index for index in range(len(segments)) if index not in used_indices]
    if not available:
        return []

    replacement_midpoint = (replacement_start + replacement_end) / 2.0
    nearest = min(
        available,
        key=lambda index: abs(((segments[index].start + segments[index].end) / 2.0) - replacement_midpoint),
    )
    return [nearest]


def _overlap_seconds(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _ranges_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return _overlap_seconds(left_start, left_end, right_start, right_end) > 0.05


def _fit_segment_text_budgets(segments: list[Segment], config: DubConfig) -> list[Segment]:
    if config.tts.lower() == "none":
        return segments

    changed = 0
    for segment in segments:
        text = segment.spoken_text
        shortened = _shorten_text_to_duration(text, segment.duration, chaos=config.artifact_chaos_mode)
        if shortened and shortened != text:
            if segment.translated_text is not None:
                segment.translated_text = shortened
            else:
                segment.text = shortened
            changed += 1

    if changed:
        print(f"      TTS text budget: shortened={changed}/{len(segments)}")
    return segments


def _shorten_text_to_duration(text: str, duration: float, *, chaos: bool = False) -> str:
    text = " ".join(text.split()).strip()
    if not text or duration <= 0.0:
        return text

    if chaos:
        max_words = max(2, int(duration * 6.2 + 1.5))
        max_chars = max(28, int(duration * 42.0))
    else:
        max_words = max(2, int(duration * 4.8 + 1.5))
        max_chars = max(24, int(duration * 34.0))
    if duration < 0.65:
        max_chars = min(max_chars, 32 if chaos else 26)

    shortened = False
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        shortened = True

    if len(text) > max_chars:
        trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
        text = trimmed or text[:max_chars].strip()
        shortened = True

    text = text.rstrip(" ,;:")
    if shortened and text and not text.endswith(("...", "…")):
        return f"{text}..."
    return text


def _artifact_slots(
    segments: list[Segment],
    source_duration: float,
    config: DubConfig,
) -> list[tuple[float, float, bool, str]]:
    slots: list[tuple[float, float, bool, str]] = []
    for current, following in zip(segments, segments[1:]):
        gap = following.start - current.end
        if gap >= config.artifact_min_gap_seconds:
            slots.append((current.end, following.start, False, "gap"))

    if segments:
        tail_start = segments[-1].end
        if source_duration - tail_start >= 0.35:
            slots.append((tail_start, source_duration, False, "tail"))

    for current, following in zip(segments, segments[1:]):
        gap = following.start - current.end
        if 0.15 <= gap < config.artifact_min_gap_seconds:
            slots.append((current.end, following.start, False, "short_gap"))

    for segment in segments:
        if segment.duration < 1.0:
            continue
        duration = min(1.15, max(0.35, segment.duration * 0.4))
        end = segment.end - min(0.05, segment.duration * 0.05)
        start = max(segment.start, end - duration)
        if end - start >= 0.25:
            slots.append((start, end, True, "overlay"))

    return slots


def _select_artifact_slot(
    artifact: Segment,
    slots: list[tuple[float, float, bool, str]],
    used_slots: set[int],
    source_duration: float,
) -> int | None:
    available = [index for index in range(len(slots)) if index not in used_slots]
    if not available:
        return None

    hint = _artifact_timing_hint(artifact, source_duration)
    if hint is None:
        return available[0]

    def score(index: int) -> tuple[float, int]:
        slot_start, slot_end, can_overlay, kind = slots[index]
        midpoint = (slot_start + slot_end) / 2.0
        kind_penalty = 0 if not can_overlay else 1
        if kind == "tail" and hint >= source_duration - 4.0:
            kind_penalty = -1
        return (abs(midpoint - hint), kind_penalty)

    return min(available, key=score)


def _artifact_timing_hint(artifact: Segment, source_duration: float) -> float | None:
    if artifact.end <= artifact.start:
        return None
    center = (artifact.start + artifact.end) / 2.0
    if 0.0 <= center <= source_duration:
        return center
    return None


def _place_artifact_in_slot(
    artifact: Segment,
    text: str,
    slot_start: float,
    slot_end: float,
    can_overlay: bool,
    kind: str,
    source_duration: float,
) -> tuple[float, float]:
    slot_duration = max(0.0, slot_end - slot_start)
    if slot_duration <= 0.05:
        return slot_start, slot_start

    target_duration = _artifact_spoken_duration(text, artifact.duration, can_overlay)
    padding = 0.05 if not can_overlay else 0.0
    duration = min(target_duration, max(0.2, slot_duration - padding))
    hint = _artifact_timing_hint(artifact, source_duration)

    if hint is not None:
        start = hint - duration / 2.0
    elif kind == "tail":
        start = slot_end - duration - 0.15
    elif can_overlay:
        start = slot_end - duration
    else:
        start = slot_start + min(0.2, slot_duration / 4.0)

    start = max(slot_start + padding, min(start, slot_end - duration - padding))
    end = min(slot_end - padding, start + duration)
    return start, end


def _artifact_spoken_duration(text: str, source_duration: float, can_overlay: bool) -> float:
    words = max(1, len(text.split()))
    estimated = max(0.65, min(2.2, words * 0.22))
    duration = min(max(0.55, source_duration), estimated)
    if can_overlay:
        duration = min(duration, 1.15)
    return duration


def _shorten_artifact_text(text: str, *, max_words: int = 18, max_chars: int = 140) -> str:
    text = " ".join(text.split()).strip()
    if not text:
        return ""

    shortened = False
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        shortened = True
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
        shortened = True
    if shortened:
        return text.rstrip(" ,.;:") + "..."
    return text


def _tts_fit_complete(segments: list[Segment], config: DubConfig) -> bool:
    fit_dir = config.workdir / "tts_fit"
    spoken_indices = [
        index
        for index, segment in enumerate(segments, start=1)
        if segment.spoken_text
    ]
    return bool(spoken_indices) and all(_file_ready(fit_dir / f"{index:05d}.wav") for index in spoken_indices)


def _plan_overlap_aware_mix(
    fitted_items: list[tuple[Segment, Path]],
    source_audio: Path,
    config: DubConfig,
) -> list[tuple[Path, int, float, float | None]]:
    """Keep useful background overlap while ducking competing foreground lines."""
    if not fitted_items:
        return []

    try:
        import numpy as np

        source_samples, sample_rate = _read_wav_mono(source_audio)
    except Exception as exc:
        print(f"      Overlap loudness analysis skipped: {type(exc).__name__}: {exc}")
        source_samples, sample_rate = None, 0

    rms_levels: list[float] = []
    for segment, _path in fitted_items:
        rms = 0.0
        if source_samples is not None and sample_rate > 0:
            start_sample = max(0, round(segment.start * sample_rate))
            end_sample = min(source_samples.size, round(segment.end * sample_rate))
            excerpt = source_samples[start_sample:end_sample]
            if excerpt.size:
                rms = float(np.sqrt(np.mean(excerpt * excerpt)))
        rms_levels.append(rms)

    positive_levels = sorted(level for level in rms_levels if level > 1e-5)
    median_rms = positive_levels[len(positive_levels) // 2] if positive_levels else 0.0
    background_cutoff = median_rms * 0.42
    is_background = [
        bool(median_rms > 0.0 and level > 0.0 and level < background_cutoff)
        for level in rms_levels
    ]

    artifact_segments: list[Segment] = []
    artifact_path = _debug_path(config, "artifact_injected.srt")
    if artifact_path.is_file():
        try:
            artifact_segments = read_srt(artifact_path, translated=True)
        except Exception:
            artifact_segments = []

    def is_artifact(segment: Segment) -> bool:
        signature = _artifact_signature(segment.spoken_text)
        return any(
            abs(segment.start - artifact.start) <= 0.12
            and (
                signature == _artifact_signature(artifact.spoken_text)
                or abs(segment.end - artifact.end) <= 0.12
            )
            for artifact in artifact_segments
        )

    artifact_flags = [is_artifact(segment) for segment, _path in fitted_items]
    gains = [0.52 if background else 1.0 for background in is_background]
    duck_after: list[float | None] = [None] * len(fitted_items)
    durations: list[float] = []
    for _segment, path in fitted_items:
        try:
            durations.append(probe_duration(path))
        except Exception:
            durations.append(0.0)

    actions: list[dict[str, object]] = []
    for index in range(len(fitted_items) - 1):
        segment, _path = fitted_items[index]
        next_segment, _next_path = fitted_items[index + 1]
        generated_end = segment.start + durations[index]
        overlap = generated_end - next_segment.start
        if overlap <= 0.35:
            continue

        current_artifact = artifact_flags[index]
        next_artifact = artifact_flags[index + 1]
        different_speakers = bool(
            segment.speaker_id
            and next_segment.speaker_id
            and segment.speaker_id != next_segment.speaker_id
        )
        reason = ""
        if current_artifact and not next_artifact:
            duck_after[index] = max(0.0, next_segment.start - segment.start)
            reason = "artifact_yields_to_main"
        elif next_artifact and not current_artifact:
            gains[index + 1] = min(gains[index + 1], 0.55)
            reason = "artifact_under_main_tail"
        elif different_speakers and (is_background[index] or is_background[index + 1]):
            # A quiet second voice is a useful part of the scene. Its base gain
            # already follows the source RMS, so retain the overlap untouched.
            reason = "different_speaker_background_allowed"
        elif different_speakers and overlap <= 1.25:
            reason = "different_speaker_short_overlap_allowed"
        elif different_speakers and rms_levels[index + 1] < rms_levels[index]:
            gains[index + 1] = min(gains[index + 1], 0.58)
            reason = "quieter_next_speaker_lowered"
        else:
            duck_after[index] = max(0.0, next_segment.start - segment.start)
            reason = "foreground_conflict_duck_previous"

        actions.append(
            {
                "segment": index + 1,
                "next_segment": index + 2,
                "overlap_seconds": round(overlap, 3),
                "reason": reason,
            }
        )

    manifest = {
        "median_source_rms": round(median_rms, 6),
        "background_cutoff_rms": round(background_cutoff, 6),
        "segments": [
            {
                "segment": index + 1,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "generated_seconds": round(durations[index], 3),
                "source_rms": round(rms_levels[index], 6),
                "background": is_background[index],
                "artifact": artifact_flags[index],
                "speaker": segment.speaker_id or "",
                "gain": round(gains[index], 3),
                "duck_after_seconds": None if duck_after[index] is None else round(duck_after[index], 3),
            }
            for index, (segment, _path) in enumerate(fitted_items)
        ],
        "overlap_actions": actions,
    }
    (config.workdir / "overlap_mix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "      Overlap-aware mix: "
        f"background={sum(is_background)}/{len(fitted_items)}, conflicts={len(actions)}, "
        f"ducked={sum(item is not None for item in duck_after)}"
    )
    return [
        (path, int(segment.start * 1000), gains[index], duck_after[index])
        for index, (segment, path) in enumerate(fitted_items)
    ]


def _needs_speaker_references(config: DubConfig) -> bool:
    return config.tts.lower() in {
        "xtts",
        "f5",
        "f5tts",
        "qwen3",
        "qwen3-tts",
        "qwen3tts",
        "cosyvoice",
        "cosyvoice-tts",
        "cosyvoicetts",
        "cosy",
        "moss",
        "moss-tts",
        "mosstts",
        "moss-v1.5",
    }


def _assign_segment_speaker_refs(
    segments: list[Segment],
    reference_audio: Path,
    config: DubConfig,
    *,
    diarization_audio: Path | None = None,
) -> None:
    if config.speaker_clustering and diarization_audio is not None:
        try:
            if _assign_pyannote_speaker_refs(segments, diarization_audio, reference_audio, config):
                return
        except Exception as exc:
            print(f"      Pyannote diarization fallback to legacy clustering: {type(exc).__name__}: {exc}")

    refs_dir = config.workdir / "speaker_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    source_duration = probe_duration(reference_audio)
    target_window = max(2.5, config.speaker_reference_seconds)
    candidates: list[_SpeakerCandidate] = []
    extracted = 0

    for index, segment in enumerate(segments, start=1):
        if not segment.spoken_text:
            continue
        segment_duration = max(0.15, segment.duration)
        window = max(target_window, min(8.0, segment_duration + 0.8))
        midpoint = max(0.0, (segment.start + segment.end) / 2.0)
        start = max(0.0, midpoint - window / 2.0)
        if start + window > source_duration:
            start = max(0.0, source_duration - window)
        duration = min(window, max(0.15, source_duration - start))
        raw_ref_path = refs_dir / f"{index:05d}_raw.wav"
        ref_path = refs_dir / f"{index:05d}.wav"
        try:
            extract_wav_slice(reference_audio, raw_ref_path, start, duration)
            ref_path = _prepare_xtts_reference(raw_ref_path, ref_path)
        except Exception as exc:
            print(f"      Speaker reference skipped for segment {index}: {type(exc).__name__}: {exc}")
            continue
        segment.speaker_wav = ref_path
        segment.speaker_ref_text = _speaker_reference_text(segment)
        extracted += 1

        embedding, quality = _speaker_embedding(ref_path)
        if embedding is None:
            continue
        candidates.append(
            _SpeakerCandidate(
                segment_index=index,
                segment=segment,
                ref_path=ref_path,
                raw_ref_path=raw_ref_path,
                embedding=embedding,
                quality=quality,
                midpoint=midpoint,
            )
        )

    if not extracted:
        print(f"      Clone TTS multi-speaker refs: 0 from {reference_audio}")
        return

    if not config.speaker_clustering or len(candidates) < 2:
        for candidate in candidates:
            candidate.cluster_id = candidate.segment_index
            candidate.segment.speaker_id = f"segment_{candidate.segment_index:05d}"
        _write_speaker_map(segments, candidates, config, clustered=False)
        print(f"      Clone TTS multi-speaker refs: {extracted} individual refs from {reference_audio}")
        return

    try:
        clusters = _cluster_speaker_candidates(candidates, config)
        bank_paths = _build_speaker_bank_refs(clusters, config)
    except Exception as exc:
        print(f"      Speaker clustering skipped: {type(exc).__name__}: {exc}")
        for candidate in candidates:
            candidate.cluster_id = candidate.segment_index
            candidate.segment.speaker_id = f"segment_{candidate.segment_index:05d}"
        _write_speaker_map(segments, candidates, config, clustered=False)
        print(f"      Clone TTS multi-speaker refs: {extracted} individual refs from {reference_audio}")
        return

    if not bank_paths:
        _write_speaker_map(segments, candidates, config, clustered=False)
        print(f"      Clone TTS multi-speaker refs: {extracted} individual refs from {reference_audio}")
        return

    for candidate in candidates:
        if candidate.cluster_id is None:
            continue
        bank_path = bank_paths.get(candidate.cluster_id)
        if bank_path is None:
            continue
        candidate.segment.speaker_wav = bank_path
        candidate.segment.speaker_id = f"speaker_{candidate.cluster_id:02d}"

    _assign_missing_cluster_refs(segments, candidates, bank_paths, config)
    _write_speaker_map(segments, candidates, config, clustered=True)
    print(
        "      XTTS speaker clustering: "
        f"{extracted} refs, {len(candidates)} usable, {len(bank_paths)} speaker banks from {reference_audio}"
    )


def _assign_pyannote_speaker_refs(
    segments: list[Segment],
    diarization_audio: Path,
    reference_audio: Path,
    config: DubConfig,
) -> bool:
    python_path = config.diarization_python
    if python_path is None:
        return False
    if not python_path.is_absolute():
        python_path = _repo_root() / python_path
    runner_path = _repo_root() / "tools" / "diarization_runner.py"
    if not python_path.is_file() or not runner_path.is_file():
        print(f"      Pyannote diarization unavailable: {python_path}")
        return False

    cache_dir = config.diarization_cache_dir
    if not cache_dir.is_absolute():
        cache_dir = _repo_root() / cache_dir
    token_file = config.diarization_token_file
    if token_file is not None and not token_file.is_absolute():
        token_file = _repo_root() / token_file
    output_path = config.workdir / "speaker_bank" / "diarization.json"
    command = [
        str(python_path), str(runner_path),
        "--audio", str(diarization_audio),
        "--output", str(output_path),
        "--model", config.diarization_model,
        "--device", config.diarization_device,
        "--cache-dir", str(cache_dir),
    ]
    if token_file is not None:
        command.extend(["--token-file", str(token_file)])
    if config.speaker_count_exact is not None and config.speaker_count_exact > 1:
        command.extend(["--num-speakers", str(config.speaker_count_exact)])
    else:
        command.extend(["--max-speakers", str(max(1, config.max_speaker_clusters))])

    print(
        "      Running pyannote diarization: "
        f"exact={config.speaker_count_exact or 'auto'} max={config.max_speaker_clusters}"
    )
    completed = subprocess.run(
        command,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(60, int(config.diarization_timeout_seconds)),
        check=False,
    )
    if completed.returncode != 0:
        details = " ".join((completed.stderr or completed.stdout or "").split())[-1200:]
        raise RuntimeError(f"diarization runner failed ({completed.returncode}): {details}")
    if not output_path.is_file():
        raise RuntimeError("diarization runner did not create diarization.json")
    data = json.loads(output_path.read_text(encoding="utf-8"))
    turns: list[tuple[float, float, str]] = []
    for item in data.get("turns", []):
        if not isinstance(item, dict):
            continue
        start = max(0.0, float(item.get("start", 0.0)))
        end = max(start, float(item.get("end", start)))
        speaker = str(item.get("speaker", "")).strip()
        if speaker and end - start >= 0.1:
            turns.append((start, end, speaker))
    if not turns:
        return False

    speaker_order = list(dict.fromkeys(speaker for _start, _end, speaker in turns))
    bank_paths = _build_pyannote_speaker_bank(turns, speaker_order, reference_audio, config)
    if not bank_paths:
        return False
    # pyannote sometimes reports a "speaker" whose turns are all too short to
    # build a voice reference from (typically noise or a stray interjection,
    # not a real extra character). Drop those turns before picking the
    # dominant speaker per utterance below - otherwise a line that mostly
    # overlaps a bank-less phantom speaker loses the vote to nobody and
    # falls all the way back to the raw full-mix reference instead of to the
    # closest real, voiced speaker.
    turns = [turn for turn in turns if turn[2] in bank_paths]
    if not turns:
        return False
    # A translated or injected subtitle is already one TTS utterance. Splitting
    # its words at every diarization boundary made a single sentence jump
    # between voices, especially when pyannote produced sub-second jitter.
    # Assign the dominant overlapping speaker to the complete utterance.
    assigned = 0
    for segment in segments:
        if not segment.spoken_text:
            continue
        speaker = _speaker_for_interval(segment.start, segment.end, turns)
        bank_path = bank_paths.get(speaker)
        if bank_path is None:
            continue
        segment.speaker_id = speaker
        segment.speaker_wav = bank_path
        segment.speaker_ref_text = ""
        assigned += 1
    if config.speaker_wav is not None:
        for segment in segments:
            if segment.spoken_text and segment.speaker_wav is None:
                segment.speaker_id = "global"
                segment.speaker_wav = config.speaker_wav
    _write_pyannote_speaker_map(segments, turns, bank_paths, data, config)
    print(
        "      Pyannote speaker diarization: "
        f"speakers={len(bank_paths)} turns={len(turns)} assigned={assigned}/{len(segments)}"
    )
    return assigned > 0


def _split_segments_at_speaker_changes(
    segments: list[Segment],
    turns: list[tuple[float, float, str]],
    bank_paths: dict[str, Path],
) -> list[Segment]:
    result: list[Segment] = []
    for segment in segments:
        text = segment.spoken_text
        if not text:
            result.append(segment)
            continue
        runs = _speaker_runs_for_segment(segment.start, segment.end, turns)
        words = text.split()
        while len(runs) > len(words) and len(runs) > 1:
            shortest = min(range(len(runs)), key=lambda index: runs[index][1] - runs[index][0])
            runs = _merge_speaker_run_at(runs, shortest)
        if len(runs) <= 1 or len(words) < 2:
            result.append(segment)
            continue

        durations = [max(0.05, end - start) for start, end, _speaker in runs]
        word_counts = _allocate_words_by_duration(len(words), durations)
        offset = 0
        for (start, end, speaker), count in zip(runs, word_counts):
            piece = " ".join(words[offset : offset + count]).strip()
            offset += count
            if not piece:
                continue
            result.append(
                Segment(
                    start=start,
                    end=end,
                    text=piece,
                    translated_text=piece,
                    speaker_wav=bank_paths.get(speaker),
                    speaker_id=speaker,
                    speaker_ref_text="",
                )
            )
    return result


def _speaker_runs_for_segment(
    start: float,
    end: float,
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    raw = [
        (max(start, turn_start), min(end, turn_end), speaker)
        for turn_start, turn_end, speaker in turns
        if _overlap_seconds(start, end, turn_start, turn_end) >= 0.18
    ]
    if not raw:
        return []
    raw.sort(key=lambda item: (item[0], item[1]))
    runs: list[tuple[float, float, str]] = []
    for run_start, run_end, speaker in raw:
        if runs and runs[-1][2] == speaker and run_start - runs[-1][1] <= 0.45:
            previous = runs[-1]
            runs[-1] = (previous[0], max(previous[1], run_end), speaker)
        else:
            runs.append((run_start, run_end, speaker))

    # Tiny alternating islands are usually boundary jitter, not a real turn.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for index, run in enumerate(runs):
            if run[1] - run[0] >= 0.35:
                continue
            runs = _merge_speaker_run_at(runs, index)
            changed = True
            break

    while len(runs) > 8:
        shortest = min(range(len(runs)), key=lambda index: runs[index][1] - runs[index][0])
        runs = _merge_speaker_run_at(runs, shortest)

    if len(runs) <= 1:
        speaker = runs[0][2]
        return [(start, end, speaker)]

    bounded: list[tuple[float, float, str]] = []
    boundaries = [start]
    for current, following in zip(runs, runs[1:]):
        boundaries.append(max(boundaries[-1], min(end, (current[1] + following[0]) / 2.0)))
    boundaries.append(end)
    for index, (_run_start, _run_end, speaker) in enumerate(runs):
        bounded_start = boundaries[index]
        bounded_end = boundaries[index + 1]
        if bounded_end - bounded_start >= 0.12:
            bounded.append((bounded_start, bounded_end, speaker))
    return bounded


def _merge_speaker_run_at(
    runs: list[tuple[float, float, str]],
    index: int,
) -> list[tuple[float, float, str]]:
    if len(runs) <= 1:
        return runs
    if index <= 0:
        target = 1
    elif index >= len(runs) - 1:
        target = index - 1
    else:
        previous_duration = runs[index - 1][1] - runs[index - 1][0]
        following_duration = runs[index + 1][1] - runs[index + 1][0]
        target = index - 1 if previous_duration >= following_duration else index + 1
    merged_start = min(runs[index][0], runs[target][0])
    merged_end = max(runs[index][1], runs[target][1])
    merged_speaker = runs[target][2]
    kept = [item for position, item in enumerate(runs) if position not in {index, target}]
    kept.append((merged_start, merged_end, merged_speaker))
    kept.sort(key=lambda item: (item[0], item[1]))
    collapsed: list[tuple[float, float, str]] = []
    for item in kept:
        if collapsed and collapsed[-1][2] == item[2]:
            previous = collapsed[-1]
            collapsed[-1] = (previous[0], max(previous[1], item[1]), item[2])
        else:
            collapsed.append(item)
    return collapsed


def _allocate_words_by_duration(word_count: int, durations: list[float]) -> list[int]:
    if not durations:
        return []
    remaining_words = word_count
    remaining_duration = sum(durations)
    counts: list[int] = []
    for index, duration in enumerate(durations):
        remaining_runs = len(durations) - index
        if remaining_runs == 1:
            count = remaining_words
        else:
            proportional = round(remaining_words * duration / max(0.05, remaining_duration))
            count = max(1, min(proportional, remaining_words - (remaining_runs - 1)))
        counts.append(count)
        remaining_words -= count
        remaining_duration -= duration
    return counts


def _speaker_for_interval(start: float, end: float, turns: list[tuple[float, float, str]]) -> str:
    overlaps: dict[str, float] = {}
    for turn_start, turn_end, speaker in turns:
        overlap = _overlap_seconds(start, end, turn_start, turn_end)
        if overlap > 0.0:
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
    if overlaps:
        return max(overlaps, key=overlaps.get)
    midpoint = (start + end) / 2.0
    return min(turns, key=lambda item: abs((item[0] + item[1]) / 2.0 - midpoint))[2]


def _build_pyannote_speaker_bank(
    turns: list[tuple[float, float, str]],
    speaker_order: list[str],
    reference_audio: Path,
    config: DubConfig,
) -> dict[str, Path]:
    bank_dir = config.workdir / "speaker_bank"
    clips_dir = bank_dir / "diarization_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for speaker_index, speaker in enumerate(speaker_order, start=1):
        candidates = sorted(
            ((start, end) for start, end, label in turns if label == speaker and end - start >= 0.85),
            key=lambda item: item[1] - item[0],
            reverse=True,
        )
        selected: list[Path] = []
        selected_seconds = 0.0
        for clip_index, (start, end) in enumerate(candidates[:8], start=1):
            margin = min(0.18, (end - start) * 0.08)
            start += margin
            duration = min(5.5, max(0.45, end - start - margin))
            clip_path = clips_dir / f"speaker_{speaker_index:02d}_{clip_index:02d}.wav"
            try:
                extract_wav_slice(reference_audio, clip_path, start, duration)
                if _file_ready(clip_path):
                    selected.append(clip_path)
                    selected_seconds += duration
            except Exception as exc:
                print(f"      Diarization ref skipped {speaker}/{clip_index}: {type(exc).__name__}: {exc}")
            if selected_seconds >= 10.0:
                break
        if not selected:
            continue
        raw_path = bank_dir / f"speaker_{speaker_index:02d}_raw.wav"
        clean_path = bank_dir / f"speaker_{speaker_index:02d}.wav"
        try:
            concat_wavs(selected, raw_path)
            result[speaker] = _prepare_xtts_reference(raw_path, clean_path)
        except Exception as exc:
            print(f"      Diarization bank fallback {speaker}: {type(exc).__name__}: {exc}")
            result[speaker] = selected[0]
    return result


def _write_pyannote_speaker_map(
    segments: list[Segment],
    turns: list[tuple[float, float, str]],
    bank_paths: dict[str, Path],
    runner_data: dict[str, object],
    config: DubConfig,
) -> None:
    bank_dir = config.workdir / "speaker_bank"
    rows = [
        {
            "segment": index,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": segment.speaker_id or "",
            "ref": str(segment.speaker_wav or ""),
            "text": re.sub(r"\s+", " ", segment.spoken_text).strip()[:180],
        }
        for index, segment in enumerate(segments, start=1)
        if segment.spoken_text
    ]
    summary = {
        "provider": "pyannote",
        "model": config.diarization_model,
        "exact_speakers": config.speaker_count_exact,
        "max_speakers": config.max_speaker_clusters,
        "speakers": {speaker: str(path) for speaker, path in bank_paths.items()},
        "turns": len(turns),
        "segments": rows,
        "runner": runner_data,
    }
    (bank_dir / "speaker_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _speaker_reference_text(segment: Segment) -> str:
    text = re.sub(r"\s+", " ", segment.spoken_text).strip()
    if len(text) > 260:
        text = text[:260].rsplit(" ", 1)[0].strip() or text[:260].strip()
    return text


def _speaker_embedding(path: Path) -> tuple[object | None, float]:
    try:
        import numpy as np
    except Exception:
        return None, 0.0

    try:
        samples, sample_rate = _read_wav_mono(path)
    except Exception as exc:
        print(f"      Speaker embedding skipped for {path.name}: {type(exc).__name__}: {exc}")
        return None, 0.0

    if samples is None or sample_rate <= 0:
        return None, 0.0
    min_samples = int(sample_rate * 0.35)
    if samples.size < min_samples:
        return None, 0.0

    samples = samples.astype(np.float64, copy=False)
    samples = samples - float(np.mean(samples))
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    if rms < 0.003 or peak < 0.01:
        return None, 0.0

    max_samples = int(sample_rate * 4.5)
    if samples.size > max_samples:
        center = samples.size // 2
        half = max_samples // 2
        samples = samples[max(0, center - half) : center + half]

    features = np.asarray(_speaker_feature_vector(samples, sample_rate, rms), dtype=np.float64)
    if features.size == 0:
        return None, 0.0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    clipping = float(np.mean(np.abs(samples) > 0.98))
    features = features - float(np.mean(features))
    std = float(np.std(features))
    if std > 1e-8:
        features = features / std
    norm = float(np.linalg.norm(features))
    if norm <= 1e-8:
        return None, 0.0
    features = features / norm

    duration_score = min(1.0, samples.size / float(sample_rate * 3.0))
    loudness_score = max(0.05, min(1.0, rms * 18.0))
    clipping_penalty = max(0.2, 1.0 - min(0.8, clipping * 5.0))
    quality = duration_score * loudness_score * clipping_penalty
    return features, quality


def _speaker_feature_vector(samples, sample_rate: int, rms: float):
    try:
        return _librosa_speaker_feature_vector(samples, sample_rate, rms)
    except Exception:
        return _basic_speaker_feature_vector(samples, sample_rate, rms)


def _librosa_speaker_feature_vector(samples, sample_rate: int, rms: float):
    import librosa
    import numpy as np

    audio = samples.astype(np.float32, copy=False)
    n_fft = min(2048, max(512, 2 ** int(math.floor(math.log2(max(512, audio.size))))))
    hop_length = max(160, min(512, n_fft // 4))
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop_length)

    cepstral = np.concatenate(
        [
            np.mean(mfcc[1:], axis=1),
            np.std(mfcc[1:], axis=1),
            np.mean(delta[1:10], axis=1),
            np.std(delta[1:10], axis=1),
        ]
    )
    spectral = np.array(
        [
            float(np.mean(centroid)) / max(1.0, sample_rate / 2.0),
            float(np.std(centroid)) / max(1.0, sample_rate / 2.0),
            float(np.mean(bandwidth)) / max(1.0, sample_rate / 2.0),
            float(np.mean(rolloff)) / max(1.0, sample_rate / 2.0),
            float(np.mean(flatness)),
            float(np.mean(zcr)),
            math.log(max(rms, 1e-8)),
        ],
        dtype=np.float64,
    )
    return np.concatenate([cepstral.astype(np.float64), spectral])


def _basic_speaker_feature_vector(samples, sample_rate: int, rms: float):
    import numpy as np

    window = np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(samples * window)) + 1e-12
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return np.zeros(16, dtype=np.float64)

    bands = [
        (80, 160),
        (160, 250),
        (250, 400),
        (400, 650),
        (650, 1000),
        (1000, 1600),
        (1600, 2600),
        (2600, 4200),
        (4200, 7200),
    ]
    band_features: list[float] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energy = float(np.sum(spectrum[mask])) if np.any(mask) else 0.0
        band_features.append(math.log(max(energy / total, 1e-9)))

    centroid = float(np.sum(freqs * spectrum) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total))
    cumulative = np.cumsum(spectrum)
    rolloff_index = int(np.searchsorted(cumulative, total * 0.85))
    rolloff = float(freqs[min(rolloff_index, freqs.size - 1)])
    flatness = float(math.exp(float(np.mean(np.log(spectrum)))) / max(float(np.mean(spectrum)), 1e-12))
    zcr = float(np.mean(samples[:-1] * samples[1:] < 0.0)) if samples.size > 1 else 0.0

    return np.array(
        [
            *band_features,
            centroid / max(1.0, sample_rate / 2.0),
            bandwidth / max(1.0, sample_rate / 2.0),
            rolloff / max(1.0, sample_rate / 2.0),
            flatness,
            zcr,
            math.log(max(rms, 1e-8)),
        ],
        dtype=np.float64,
    )


def _trim_tts_silence(path: Path, max_pause_seconds: float) -> bool:
    """Tighten a synthesized phrase in place: drop the silence MOSS pads around the
    phrase and clamp the pauses it leaves inside one.

    MOSS is deliberately never time-stretched (it owns its own delivery), so its
    padding is the one thing that can be reclaimed without touching the voice.
    On real jobs a quarter of the generated audio is silence, which is what pushes
    phrases into each other's slots - a phrase of two words can otherwise occupy
    seven seconds. Only silence is removed here, never speech, so the delivery
    itself comes out unchanged.
    """
    import numpy as np

    samples, sample_rate = _read_wav_mono(path)
    if samples is None or sample_rate <= 0 or samples.size == 0:
        return False

    frame = max(1, int(sample_rate * 0.02))
    frame_count = samples.size // frame
    if frame_count <= 1:
        return False
    frames = samples[: frame_count * frame].reshape(frame_count, frame)
    rms = np.sqrt(np.maximum(0.0, (frames * frames).mean(axis=1)))
    peak = float(rms.max())
    if peak <= 0.0:
        return False
    voiced = rms > max(peak * 0.02, 0.003)
    voiced_index = np.flatnonzero(voiced)
    if voiced_index.size == 0:
        return False

    # A little air on both sides keeps the phrase from sounding clipped.
    pad = max(1, int(0.05 / 0.02))
    max_pause = max(0, int(max(0.0, max_pause_seconds) / 0.02))
    first, last = int(voiced_index[0]), int(voiced_index[-1])
    keep = np.zeros(frame_count, dtype=bool)
    keep[max(0, first - pad) : min(frame_count, last + pad + 1)] = True

    # Clamp each run of silence between two voiced frames to max_pause.
    run_start: int | None = None
    for index in range(first, last + 2):
        silent = index <= last and not voiced[index]
        if silent and run_start is None:
            run_start = index
        elif not silent and run_start is not None:
            run_length = index - run_start
            if run_length > max_pause:
                drop_from = run_start + max_pause // 2
                drop_to = index - (max_pause - max_pause // 2)
                keep[drop_from:drop_to] = False
            run_start = None

    if keep.all():
        return False
    trimmed = frames[keep].reshape(-1)
    if trimmed.size == 0:
        return False

    data = np.clip(trimmed, -1.0, 1.0)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((data * 32767.0).astype("<i2").tobytes())
    return True


def _read_wav_mono(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if not raw or sample_rate <= 0:
        return None, sample_rate
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        return None, sample_rate

    if channels > 1:
        usable = (data.size // channels) * channels
        if usable <= 0:
            return None, sample_rate
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def _cluster_speaker_candidates(
    candidates: list[_SpeakerCandidate],
    config: DubConfig,
) -> list[list[_SpeakerCandidate]]:
    import numpy as np

    max_clusters = max(1, config.max_speaker_clusters)
    threshold = max(0.02, min(0.75, config.speaker_cluster_threshold))
    clusters: list[list[_SpeakerCandidate]] = []
    centroids: list[object] = []

    for candidate in sorted(candidates, key=lambda item: (item.midpoint, item.segment_index)):
        if not clusters:
            clusters.append([candidate])
            centroids.append(candidate.embedding)
            continue

        similarities = [float(np.dot(candidate.embedding, centroid)) for centroid in centroids]
        best_index = int(np.argmax(similarities))
        best_distance = 1.0 - similarities[best_index]
        if best_distance <= threshold or len(clusters) >= max_clusters:
            clusters[best_index].append(candidate)
            centroids[best_index] = _speaker_centroid(clusters[best_index])
        else:
            clusters.append([candidate])
            centroids.append(candidate.embedding)

    ordered = sorted(clusters, key=lambda items: min(item.midpoint for item in items))
    for cluster_id, items in enumerate(ordered, start=1):
        for item in items:
            item.cluster_id = cluster_id
    return ordered


def _speaker_centroid(candidates: list[_SpeakerCandidate]):
    import numpy as np

    weights = np.array([max(0.05, candidate.quality) for candidate in candidates], dtype=np.float64)
    matrix = np.stack([candidate.embedding for candidate in candidates], axis=0)
    centroid = np.average(matrix, axis=0, weights=weights)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1e-8:
        return matrix[-1]
    return centroid / norm


def _build_speaker_bank_refs(
    clusters: list[list[_SpeakerCandidate]],
    config: DubConfig,
) -> dict[int, Path]:
    bank_dir = config.workdir / "speaker_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}

    for cluster_id, items in enumerate(clusters, start=1):
        selected = sorted(items, key=lambda item: item.quality, reverse=True)[:3]
        if not selected:
            continue
        raw_bank_path = bank_dir / f"speaker_{cluster_id:02d}_raw.wav"
        bank_path = bank_dir / f"speaker_{cluster_id:02d}.wav"
        try:
            concat_wavs([item.ref_path for item in selected], raw_bank_path)
            result[cluster_id] = _prepare_xtts_reference(raw_bank_path, bank_path)
        except Exception as exc:
            print(f"      Speaker bank {cluster_id:02d} fallback: {type(exc).__name__}: {exc}")
            result[cluster_id] = selected[0].ref_path
    return result


def _assign_missing_cluster_refs(
    segments: list[Segment],
    candidates: list[_SpeakerCandidate],
    bank_paths: dict[int, Path],
    config: DubConfig,
) -> None:
    if not candidates or not bank_paths:
        return

    for segment in segments:
        if not segment.spoken_text or segment.speaker_id:
            continue
        midpoint = (segment.start + segment.end) / 2.0
        nearest = min(candidates, key=lambda item: abs(item.midpoint - midpoint))
        if nearest.cluster_id is None:
            continue
        bank_path = bank_paths.get(nearest.cluster_id)
        if bank_path is None:
            continue
        segment.speaker_wav = bank_path
        segment.speaker_id = f"speaker_{nearest.cluster_id:02d}"
    if config.speaker_wav is not None:
        for segment in segments:
            if segment.spoken_text and segment.speaker_wav is None:
                segment.speaker_wav = config.speaker_wav
                segment.speaker_id = "global"


def _write_speaker_map(
    segments: list[Segment],
    candidates: list[_SpeakerCandidate],
    config: DubConfig,
    *,
    clustered: bool,
) -> None:
    bank_dir = config.workdir / "speaker_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    candidate_by_index = {candidate.segment_index: candidate for candidate in candidates}
    tsv_path = bank_dir / "speaker_map.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as file:
        file.write("segment\tstart\tend\tspeaker\tquality\tref\ttext\n")
        for index, segment in enumerate(segments, start=1):
            if not segment.spoken_text:
                continue
            candidate = candidate_by_index.get(index)
            quality = candidate.quality if candidate is not None else 0.0
            speaker = segment.speaker_id or ""
            ref = str(segment.speaker_wav or "")
            text = re.sub(r"\s+", " ", segment.spoken_text).replace("\t", " ").strip()
            file.write(
                f"{index}\t{segment.start:.3f}\t{segment.end:.3f}\t{speaker}\t"
                f"{quality:.4f}\t{ref}\t{text[:180]}\n"
            )

    summary: dict[str, object] = {
        "clustered": clustered,
        "usable_candidates": len(candidates),
        "speaker_clustering": config.speaker_clustering,
        "max_speaker_clusters": config.max_speaker_clusters,
        "speaker_cluster_threshold": config.speaker_cluster_threshold,
        "clusters": [],
    }
    clusters: dict[int, list[_SpeakerCandidate]] = {}
    for candidate in candidates:
        if candidate.cluster_id is None:
            continue
        clusters.setdefault(candidate.cluster_id, []).append(candidate)
    cluster_rows: list[dict[str, object]] = []
    for cluster_id, items in sorted(clusters.items()):
        cluster_rows.append(
            {
                "speaker": f"speaker_{cluster_id:02d}",
                "segments": [item.segment_index for item in items],
                "quality": round(sum(item.quality for item in items) / max(1, len(items)), 4),
                "ref": str(items[0].segment.speaker_wav or ""),
            }
        )
    summary["clusters"] = cluster_rows
    (bank_dir / "speaker_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prepare_xtts_reference(input_path: Path, output_path: Path) -> Path:
    try:
        prepare_voice_reference(input_path, output_path)
        if output_path.exists() and output_path.stat().st_size > 1024:
            return output_path
    except Exception as exc:
        print(f"      XTTS reference cleanup skipped: {type(exc).__name__}: {exc}")
    return input_path


def run_transcript(video_path: Path, config: DubConfig) -> tuple[Path, Path]:
    video_path = video_path.resolve()
    config.workdir.mkdir(parents=True, exist_ok=True)

    _report_progress(config, "Извлекаю аудио", 10, 100, video_path.name)
    print(f"[1/3] Extracting audio: {video_path}")
    source_audio = config.workdir / "source_16k.wav"
    extract_audio(video_path, source_audio)

    _report_progress(
        config,
        "Запускаю сырой Whisper",
        35,
        100,
        f"{config.whisper_model}, вход={config.source_lang or 'auto'}",
    )
    print(
        "[2/3] Transcribing only "
        f"backend={config.asr_backend} "
        f"model={config.whisper_model} "
        f"source={config.source_lang or 'auto'} "
        f"task={config.whisper_task} "
        f"force_source={config.force_source_language} "
        f"suppress_ascii={config.suppress_plain_ascii_tokens}"
    )
    asr_audio = _whisper_chaos_audio(source_audio, config, purpose="transcript")
    segments = transcribe(asr_audio, config)
    source_duration = probe_duration(source_audio)
    segments = _clamp_segments_to_duration(segments, source_duration)
    if config.artifact_chaos_mode and config.force_source_language and config.source_lang:
        chunk_segments = _harvest_chunked_forced_artifacts(asr_audio, config, config, source_duration)
        if chunk_segments:
            segments = [*segments, *chunk_segments]
    print(f"      ASR segments: {len(segments)}")

    _report_progress(config, "Записываю транскрипт", 90, 100, f"сегментов: {len(segments)}")
    print("[3/3] Writing transcript")
    srt_path = config.workdir / "whisper_only.srt"
    txt_path = config.workdir / "whisper_only.txt"
    write_srt(srt_path, segments, translated=False)
    write_txt(txt_path, segments, translated=False)
    _report_progress(config, "Сырой Whisper готов", 100, 100, None)
    return srt_path, txt_path

