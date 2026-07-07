from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .asr import clear_openai_whisper_cache
from .bot_config import BotSettings
from .models import DubConfig
from .pipeline import run_dub, run_transcript
from .tts import clear_tts_model_caches
from .watermark import add_watermark


ProgressCallback = Callable[[str, int | None, int | None, str | None], None]


ASR_METHOD_CONFIGS = {
    "ow-large-v3-hunt": ("openai-whisper", "turbo", False),
    "ow-large-v3-chaos-backbone": ("openai-whisper", "turbo", False),
    "ow-large-v3-raw-dub": ("openai-whisper", "turbo", True),
    "ow-large-v3-chaos": ("openai-whisper", "turbo", True),
    "ow-large-v3-soft": ("openai-whisper", "turbo", False),
    "ow-large-v3-forced": ("openai-whisper", "turbo", True),
    "fw-large-v3-soft": ("faster-whisper", "large-v3", False),
    "fw-large-v3-forced": ("faster-whisper", "large-v3", True),
    "ow-large-v3-turbo-soft": ("openai-whisper", "turbo", False),
    "ow-large-v2-soft": ("openai-whisper", "large-v2", False),
    "ow-large-v2-forced": ("openai-whisper", "large-v2", True),
    "ow-large-v1-soft": ("openai-whisper", "large-v1", False),
    "ow-large-v1-forced": ("openai-whisper", "large-v1", True),
    "ow-medium-soft": ("openai-whisper", "medium", False),
    "ow-medium-forced": ("openai-whisper", "medium", True),
    "ow-small-soft": ("openai-whisper", "small", False),
    "ow-small-forced": ("openai-whisper", "small", True),
    "fw-small-soft": ("faster-whisper", "small", False),
    "fw-small-forced": ("faster-whisper", "small", True),
}


@dataclass(slots=True)
class JobDocument:
    path: Path
    filename: str
    caption: str = ""


@dataclass(slots=True)
class JobExecutionResult:
    mode: str
    video_path: Path | None = None
    output_filename: str | None = None
    transcript_path: Path | None = None
    transcript_filename: str | None = None
    transcript_text: str = ""
    documents: list[JobDocument] = field(default_factory=list)


def execute_job(
    job: dict[str, Any],
    settings: BotSettings,
    *,
    progress_callback: ProgressCallback | None = None,
) -> JobExecutionResult:
    try:
        return _execute_job(job, settings, progress_callback=progress_callback)
    finally:
        clear_openai_whisper_cache()
        clear_tts_model_caches()


def _execute_job(
    job: dict[str, Any],
    settings: BotSettings,
    *,
    progress_callback: ProgressCallback | None = None,
) -> JobExecutionResult:
    job_dir = Path(str(job["job_dir"]))
    input_path = Path(str(job["input_path"]))
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "dubbed.mp4"

    if job.get("mode") == "raw_text":
        return _execute_raw_text_job(job, settings, output_path, progress_callback)

    config = _build_dub_config(job, settings, output_path)
    config.progress_callback = progress_callback
    result = run_dub(input_path, config)
    send_path = result

    watermarked_path = job_dir / "dubbed_watermarked.mp4"
    if progress_callback:
        progress_callback("Adding watermark", 98, 100, None)
    add_watermark(
        result,
        watermarked_path,
        text=settings.watermark_text,
        image_path=settings.watermark_image,
    )
    send_path = watermarked_path

    transcript_text = _read_transcript_text(job_dir / "work" / "translated.srt")
    transcript_path: Path | None = None
    transcript_filename: str | None = None
    if transcript_text:
        transcript_path = _write_transcript_text(
            job_dir,
            str(job.get("source_title") or input_path.stem),
            transcript_text,
        )
        transcript_filename = transcript_path.name

    return JobExecutionResult(
        mode="dub",
        video_path=send_path,
        output_filename=_lalaschool_filename(str(job.get("source_title") or input_path.stem), send_path.suffix),
        transcript_path=transcript_path,
        transcript_filename=transcript_filename,
        transcript_text=transcript_text,
    )


def result_manifest(result: JobExecutionResult) -> dict[str, Any]:
    data: dict[str, Any] = {"mode": result.mode}
    if result.video_path is not None:
        data["video"] = {"filename": result.video_path.name}
        data["output_filename"] = result.output_filename or result.video_path.name
    if result.transcript_path is not None:
        data["transcript"] = {"filename": result.transcript_path.name}
        data["transcript_filename"] = result.transcript_filename or result.transcript_path.name
    if result.documents:
        data["documents"] = [
            {
                "filename": document.path.name,
                "output_filename": document.filename,
                "caption": document.caption,
            }
            for document in result.documents
        ]
    return data


def _execute_raw_text_job(
    job: dict[str, Any],
    settings: BotSettings,
    output_path: Path,
    progress_callback: ProgressCallback | None,
) -> JobExecutionResult:
    job_dir = Path(str(job["job_dir"]))
    input_path = Path(str(job["input_path"]))
    transcript_config = DubConfig(
        output=output_path,
        workdir=job_dir / "work",
        source_lang=job.get("source_lang") or None,
        target_lang=target_lang_value(job.get("target_lang")),
        asr_backend="openai-whisper",
        whisper_model=settings.whisper_only_model,
        whisper_device=settings.whisper_only_device,
        whisper_compute_type="auto",
        translator="identity",
        tts="none",
        separation="none",
        audio_bed="original",
        glitch_profile="faithful",
        condition_on_previous_text=True,
        hallucination_silence_threshold=None,
        force_source_language=True,
        suppress_plain_ascii_tokens=settings.suppress_plain_ascii_tokens,
        asr_retry_on_repetition=False,
        artifact_chaos_mode=True,
        collapse_repetitions=False,
        progress_callback=progress_callback,
    )
    srt_path, txt_path = run_transcript(input_path, transcript_config)
    meta_path = job_dir / "work" / "whisper_only_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "mode": "raw_text",
                "asr_backend": transcript_config.asr_backend,
                "model": transcript_config.whisper_model,
                "device": transcript_config.whisper_device,
                "source_lang": transcript_config.source_lang,
                "target_lang": transcript_config.target_lang,
                "task": transcript_config.whisper_task,
                "condition_on_previous_text": transcript_config.condition_on_previous_text,
                "hallucination_silence_threshold": transcript_config.hallucination_silence_threshold,
                "force_source_language": transcript_config.force_source_language,
                "suppress_plain_ascii_tokens": transcript_config.suppress_plain_ascii_tokens,
                "artifact_chaos_mode": transcript_config.artifact_chaos_mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    caption = f"Raw Whisper output ({transcript_config.source_lang or 'auto'}, {transcript_config.whisper_task})"
    return JobExecutionResult(
        mode="raw_text",
        documents=[
            JobDocument(srt_path, srt_path.name, caption),
            JobDocument(txt_path, txt_path.name, caption),
            JobDocument(meta_path, meta_path.name, caption),
        ],
    )


def _build_dub_config(job: dict[str, Any], settings: BotSettings, output_path: Path) -> DubConfig:
    job_dir = Path(str(job["job_dir"]))
    target_lang = target_lang_value(job.get("target_lang"))
    tts_provider = str(job.get("tts_provider") or settings.tts)
    if target_lang == "uk" and tts_provider.lower() in {
        "qwen3",
        "qwen3-tts",
        "qwen3tts",
        "chatterbox",
        "chatterbox-tts",
        "chatterboxtts",
    }:
        tts_provider = "f5"
    config = DubConfig(
        output=output_path,
        workdir=job_dir / "work",
        source_lang=None,
        target_lang=target_lang,
        asr_backend=settings.asr_backend,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
        whisper_compute_type=settings.whisper_compute_type,
        translator=settings.translator,
        tts=tts_provider,
        voice=settings.voice,
        speaker_wav=settings.speaker_wav,
        xtts_model=settings.xtts_model,
        xtts_device=settings.xtts_device,
        xtts_speed=settings.xtts_speed,
        f5_python=settings.f5_python,
        f5_model=settings.f5_model,
        f5_hf_repo=settings.f5_hf_repo,
        f5_hf_ckpt_path=settings.f5_hf_ckpt_path,
        f5_hf_vocab_path=settings.f5_hf_vocab_path,
        f5_ckpt_file=settings.f5_ckpt_file,
        f5_vocab_file=settings.f5_vocab_file,
        f5_cache_dir=settings.f5_cache_dir,
        media_cache_dir=settings.media_cache_dir,
        f5_device=settings.f5_device,
        f5_speed=settings.f5_speed,
        f5_nfe_step=settings.f5_nfe_step,
        f5_cfg_strength=settings.f5_cfg_strength,
        f5_target_rms=settings.f5_target_rms,
        f5_cross_fade_duration=settings.f5_cross_fade_duration,
        f5_remove_silence=settings.f5_remove_silence,
        f5_timeout_seconds=settings.f5_timeout_seconds,
        qwen3_python=settings.qwen3_python,
        qwen3_model=settings.qwen3_model,
        qwen3_cache_dir=settings.qwen3_cache_dir,
        qwen3_timeout_seconds=settings.qwen3_timeout_seconds,
        chatterbox_python=settings.chatterbox_python,
        chatterbox_model=settings.chatterbox_model,
        chatterbox_device=settings.chatterbox_device,
        chatterbox_cache_dir=settings.chatterbox_cache_dir,
        chatterbox_exaggeration=settings.chatterbox_exaggeration,
        chatterbox_cfg_weight=settings.chatterbox_cfg_weight,
        chatterbox_timeout_seconds=settings.chatterbox_timeout_seconds,
        multi_speaker=settings.multi_speaker,
        speaker_reference_seconds=settings.speaker_reference_seconds,
        speaker_clustering=settings.speaker_clustering,
        max_speaker_clusters=settings.max_speaker_clusters,
        speaker_cluster_threshold=settings.speaker_cluster_threshold,
        separation=settings.separation,
        separation_device=settings.separation_device,
        demucs_model=settings.demucs_model,
        audio_bed=settings.audio_bed,
        glitch_profile=job.get("glitch_profile", "clean"),
        original_volume=settings.original_volume,
        dub_volume=settings.dub_volume,
        force_source_language=False,
        suppress_plain_ascii_tokens=settings.suppress_plain_ascii_tokens,
        asr_retry_on_repetition=True,
        artifact_source_lang=job.get("source_lang") or None,
        artifact_whisper_model=settings.whisper_only_model,
        artifact_whisper_device=settings.artifact_whisper_device,
        inject_artifacts=settings.inject_artifacts,
        artifact_max_segments=settings.artifact_max_segments,
        artifact_min_gap_seconds=settings.artifact_min_gap_seconds,
        distort_translation=settings.distort_translation,
        translation_pivots=settings.translation_pivots,
        translation_second_pass_ratio=settings.translation_second_pass_ratio,
        collapse_repetitions=settings.collapse_repetitions,
        max_phrase_repeats=settings.max_phrase_repeats,
        max_word_repeats=settings.max_word_repeats,
    )
    apply_text_extraction_method(config, job, settings)
    apply_speaker_count(config, job)
    return config


def speaker_count_value(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text or text in {"auto", "none", "null"}:
        return None
    try:
        count = int(text)
    except ValueError:
        return None
    if 1 <= count <= 9:
        return count
    return None


def target_lang_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "ua":
        text = "uk"
    return text if text in {"ru", "uk", "en"} else "ru"


def apply_speaker_count(config: DubConfig, job: dict[str, Any]) -> None:
    count = speaker_count_value(job.get("speaker_count"))
    if count is None:
        return
    config.max_speaker_clusters = count
    if count <= 1:
        config.multi_speaker = False
        config.speaker_clustering = False
    else:
        config.multi_speaker = True
        config.speaker_clustering = True


def apply_text_extraction_method(config: DubConfig, job: dict[str, Any], settings: BotSettings) -> None:
    selected_source = job.get("source_lang") or None
    method = str(job.get("asr_method") or settings.default_asr_method).strip().lower()
    if method not in ASR_METHOD_CONFIGS:
        method = "ow-large-v3-chaos-backbone"
    backend, model, forced = ASR_METHOD_CONFIGS[method]
    hunt_artifacts = method.endswith("-hunt")
    chaos_backbone = method.endswith("-chaos-backbone")
    raw_dub = method.endswith("-raw-dub")
    chaos_asr = method.endswith("-chaos")

    config.asr_backend = backend
    config.whisper_model = model
    if backend == "openai-whisper":
        config.whisper_device = settings.whisper_only_device
        config.whisper_compute_type = "auto"
    else:
        config.whisper_device = settings.whisper_device
        config.whisper_compute_type = settings.whisper_compute_type

    if forced and not selected_source:
        print(f"Forced ASR method {method} needs non-auto input language; falling back to auto soft ASR", flush=True)
        forced = False

    config.source_lang = selected_source if forced else None
    config.force_source_language = forced
    config.suppress_plain_ascii_tokens = False
    config.asr_retry_on_repetition = not forced
    config.asr_fallback_on_sparse = False
    config.artifact_source_lang = None
    config.input_pivot_lang = None
    config.inject_artifacts = False
    config.artifact_chaos_mode = False
    config.distort_main_translation = False
    config.glitch_profile = "faithful" if forced else "clean"

    if raw_dub and forced:
        config.source_lang = selected_source
        config.force_source_language = True
        config.asr_retry_on_repetition = False
        config.asr_fallback_on_sparse = False
        config.input_pivot_lang = None
        config.artifact_source_lang = None
        config.inject_artifacts = False
        config.artifact_chaos_mode = True
        config.glitch_profile = "faithful"
        config.collapse_repetitions = False
        config.distort_main_translation = True
    elif chaos_asr and forced:
        config.source_lang = selected_source
        config.force_source_language = True
        config.asr_retry_on_repetition = False
        config.asr_fallback_on_sparse = True
        config.input_pivot_lang = None
        config.artifact_source_lang = None
        config.inject_artifacts = False
        config.glitch_profile = "faithful"
        config.collapse_repetitions = False
        config.distort_main_translation = True
    elif chaos_backbone:
        config.source_lang = None
        config.force_source_language = False
        config.asr_retry_on_repetition = True
        config.asr_fallback_on_sparse = False
        config.input_pivot_lang = selected_source
        config.artifact_source_lang = selected_source
        config.inject_artifacts = bool(settings.inject_artifacts and selected_source)
        config.artifact_chaos_mode = True
        config.artifact_max_segments = max(settings.artifact_max_segments, 48)
        config.glitch_profile = "clean"
        config.distort_main_translation = True
    elif hunt_artifacts:
        config.source_lang = None
        config.force_source_language = False
        config.asr_retry_on_repetition = True
        config.input_pivot_lang = selected_source
        config.artifact_source_lang = selected_source
        config.inject_artifacts = bool(settings.inject_artifacts and selected_source)
        config.glitch_profile = "clean"
    elif not forced:
        config.input_pivot_lang = selected_source


def _lalaschool_filename(source_title: str, suffix: str) -> str:
    suffix = suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".mp4"
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", source_title)
    base = re.sub(r"\s+", " ", base).strip(" ._")
    if not base:
        base = "video"
    if "lalaschool" not in base.casefold():
        base = f"{base}_lalaschool"
    if len(base) > 120:
        suffix_text = "_lalaschool"
        base = base[: 120 - len(suffix_text)].rstrip(" ._") + suffix_text
    return f"{base}{suffix}"


def _read_transcript_text(srt_path: Path) -> str:
    if not srt_path.exists():
        return ""
    lines: list[str] = []
    for line in srt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        lines.append(stripped)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _write_transcript_text(job_dir: Path, source_title: str, transcript_text: str) -> Path:
    filename = _lalaschool_filename(f"{source_title}_transcript", ".txt")
    path = job_dir / filename
    path.write_text(transcript_text.strip() + "\n", encoding="utf-8")
    return path


def save_job_snapshot(job_dir: Path, job: dict[str, Any], *, status: str, error: str | None = None) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    data = dict(job)
    data["status"] = status
    data["updated_at"] = time.time()
    if error is not None:
        data["error"] = error
    (job_dir / "job.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _coerce_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
