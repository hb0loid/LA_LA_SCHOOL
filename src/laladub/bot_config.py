from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BotSettings:
    token: str
    workdir: Path
    executor_mode: str
    max_local_jobs: int
    worker_api_host: str
    worker_api_port: int
    worker_api_token: str
    worker_package_path: Path
    worker_package_manifest_path: Path
    worker_version: str
    job_retention_seconds: float
    cleanup_interval_seconds: float
    paid_users: frozenset[int]
    paid_users_file: Path
    admin_users: frozenset[int]
    admin_users_file: Path
    max_active_jobs: int
    max_active_jobs_per_user: int
    watermark_text: str
    watermark_image: Path | None
    max_file_mb: int
    free_max_duration_seconds: float
    paid_max_duration_seconds: float
    translator: str
    tts: str
    voice: str | None
    speaker_wav: Path | None
    xtts_model: str
    xtts_device: str
    xtts_speed: float
    f5_python: Path | None
    f5_model: str
    f5_hf_repo: str
    f5_hf_ckpt_path: str
    f5_hf_vocab_path: str
    f5_ckpt_file: Path | None
    f5_vocab_file: Path | None
    f5_cache_dir: Path
    media_cache_dir: Path
    audio_visual_source_dir: Path | None
    audio_visual_resolution: int
    audio_visual_max_slice_seconds: float
    audio_visual_safety_enabled: bool
    audio_visual_safety_model: str
    audio_visual_safety_threshold: float
    audio_visual_safety_frames: int
    audio_visual_safety_device: str
    f5_device: str
    f5_speed: float
    f5_nfe_step: int
    f5_cfg_strength: float
    f5_target_rms: float
    f5_cross_fade_duration: float
    f5_remove_silence: bool
    f5_timeout_seconds: int
    qwen3_python: Path | None
    qwen3_model: str
    qwen3_cache_dir: Path
    qwen3_timeout_seconds: int
    chatterbox_python: Path | None
    chatterbox_model: str
    chatterbox_device: str
    chatterbox_cache_dir: Path
    chatterbox_exaggeration: float
    chatterbox_cfg_weight: float
    chatterbox_timeout_seconds: int
    cosyvoice_python: Path | None
    cosyvoice_repo_dir: Path
    cosyvoice_model_dir: Path
    cosyvoice_model_id: str
    cosyvoice_mode: str
    cosyvoice_instruction: str
    cosyvoice_device: str
    cosyvoice_speed: float
    cosyvoice_timeout_seconds: int
    multi_speaker: bool
    speaker_reference_seconds: float
    speaker_clustering: bool
    max_speaker_clusters: int
    speaker_cluster_threshold: float
    separation: str
    separation_device: str
    demucs_model: str
    audio_bed: str
    asr_backend: str
    default_asr_method: str
    whisper_model: str
    whisper_only_model: str
    whisper_only_device: str
    artifact_whisper_device: str
    whisper_device: str
    whisper_compute_type: str
    suppress_plain_ascii_tokens: bool
    original_volume: float
    dub_volume: float
    collapse_repetitions: bool
    max_phrase_repeats: int
    max_word_repeats: int
    inject_artifacts: bool
    artifact_max_segments: int
    artifact_ratio: float
    artifact_min_source_segments: int
    artifact_min_gap_seconds: float
    distort_translation: bool
    translation_pivots: str
    translation_second_pass_ratio: float

    def is_paid(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.paid_users

    def is_admin(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.admin_users


def load_bot_settings(*, require_token: bool = True) -> BotSettings:
    token = os.environ.get("LALADUB_BOT_TOKEN", "").strip()
    if require_token and not token:
        raise RuntimeError("Set LALADUB_BOT_TOKEN to your Telegram bot token.")

    paid_users_file = Path(os.environ.get("LALADUB_PAID_USERS_FILE", "paid_users.txt"))
    if paid_users_file.is_file():
        paid_users = _parse_user_ids_file(paid_users_file)
    else:
        paid_users = _parse_user_ids(os.environ.get("LALADUB_PAID_USERS", ""))

    admin_users_file = Path(os.environ.get("LALADUB_ADMIN_USERS_FILE", "admins.txt"))
    admin_users = _parse_user_ids(os.environ.get("LALADUB_ADMIN_USERS", ""))
    if admin_users_file.is_file():
        admin_users.update(_parse_user_ids_file(admin_users_file))

    return BotSettings(
        token=token,
        workdir=Path(os.environ.get("LALADUB_BOT_WORKDIR", "runs/bot")),
        executor_mode=os.environ.get("LALADUB_EXECUTOR_MODE", "local").strip().lower(),
        max_local_jobs=max(0, int(os.environ.get("LALADUB_MAX_LOCAL_JOBS", "1"))),
        worker_api_host=os.environ.get("LALADUB_WORKER_API_HOST", "127.0.0.1").strip() or "127.0.0.1",
        worker_api_port=int(os.environ.get("LALADUB_WORKER_API_PORT", "8765")),
        worker_api_token=os.environ.get("LALADUB_WORKER_API_TOKEN", "").strip(),
        worker_package_path=Path(os.environ.get("LALADUB_WORKER_PACKAGE_PATH", "dist/LaLaDubWorker-update.zip")),
        worker_package_manifest_path=Path(
            os.environ.get("LALADUB_WORKER_PACKAGE_MANIFEST", "dist/LaLaDubWorker-update.manifest.json")
        ),
        worker_version=os.environ.get("LALADUB_WORKER_VERSION", "0.1.0").strip() or "0.1.0",
        job_retention_seconds=max(0.0, float(os.environ.get("LALADUB_JOB_RETENTION_SECONDS", "2592000"))),
        cleanup_interval_seconds=max(60.0, float(os.environ.get("LALADUB_CLEANUP_INTERVAL_SECONDS", "3600"))),
        paid_users=frozenset(paid_users),
        paid_users_file=paid_users_file,
        admin_users=frozenset(admin_users),
        admin_users_file=admin_users_file,
        max_active_jobs=max(1, int(os.environ.get("LALADUB_MAX_ACTIVE_JOBS", "2"))),
        max_active_jobs_per_user=max(1, int(os.environ.get("LALADUB_MAX_ACTIVE_JOBS_PER_USER", "1"))),
        watermark_text=os.environ.get("LALADUB_WATERMARK_TEXT", "La La Local Dub"),
        watermark_image=_optional_path(os.environ.get("LALADUB_WATERMARK_IMAGE")),
        max_file_mb=int(os.environ.get("LALADUB_MAX_FILE_MB", "200")),
        free_max_duration_seconds=float(os.environ.get("LALADUB_FREE_MAX_DURATION_SECONDS", "60")),
        paid_max_duration_seconds=float(os.environ.get("LALADUB_PAID_MAX_DURATION_SECONDS", "0")),
        translator=os.environ.get("LALADUB_TRANSLATOR", "hybrid"),
        tts=os.environ.get("LALADUB_TTS", "xtts"),
        voice=_empty_to_none(os.environ.get("LALADUB_VOICE", "Microsoft Irina Desktop")),
        speaker_wav=_optional_path(os.environ.get("LALADUB_SPEAKER_WAV")),
        xtts_model=os.environ.get("LALADUB_XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
        xtts_device=os.environ.get("LALADUB_XTTS_DEVICE", "cpu"),
        xtts_speed=float(os.environ.get("LALADUB_XTTS_SPEED", "1.0")),
        f5_python=_optional_path(os.environ.get("LALADUB_F5_PYTHON", ".venv-f5tts\\Scripts\\python.exe")),
        f5_model=os.environ.get("LALADUB_F5_MODEL", "F5TTS_v1_Base"),
        f5_hf_repo=os.environ.get("LALADUB_F5_HF_REPO", "Misha24-10/F5-TTS_RUSSIAN"),
        f5_hf_ckpt_path=os.environ.get(
            "LALADUB_F5_HF_CKPT_PATH",
            "F5TTS_v1_Base_v2/model_last_inference.safetensors",
        ),
        f5_hf_vocab_path=os.environ.get("LALADUB_F5_HF_VOCAB_PATH", "F5TTS_v1_Base/vocab.txt"),
        f5_ckpt_file=_optional_path(os.environ.get("LALADUB_F5_CKPT_FILE")),
        f5_vocab_file=_optional_path(os.environ.get("LALADUB_F5_VOCAB_FILE")),
        f5_cache_dir=Path(os.environ.get("LALADUB_F5_CACHE_DIR", "models/f5tts")),
        media_cache_dir=Path(os.environ.get("LALADUB_MEDIA_CACHE_DIR", "runs/cache/media")),
        audio_visual_source_dir=_optional_path(os.environ.get("LALADUB_AUDIO_VISUAL_SOURCE_DIR")),
        audio_visual_resolution=max(64, int(os.environ.get("LALADUB_AUDIO_VISUAL_RESOLUTION", "512"))),
        audio_visual_max_slice_seconds=max(0.2, float(os.environ.get("LALADUB_AUDIO_VISUAL_MAX_SLICE_SECONDS", "3.0"))),
        audio_visual_safety_enabled=_parse_bool(os.environ.get("LALADUB_AUDIO_VISUAL_SAFETY_ENABLED", "1")),
        audio_visual_safety_model=os.environ.get(
            "LALADUB_AUDIO_VISUAL_SAFETY_MODEL",
            "Falconsai/nsfw_image_detection",
        ).strip()
        or "Falconsai/nsfw_image_detection",
        audio_visual_safety_threshold=max(
            0.01,
            min(0.99, float(os.environ.get("LALADUB_AUDIO_VISUAL_SAFETY_THRESHOLD", "0.72"))),
        ),
        audio_visual_safety_frames=max(1, int(os.environ.get("LALADUB_AUDIO_VISUAL_SAFETY_FRAMES", "5"))),
        audio_visual_safety_device=os.environ.get("LALADUB_AUDIO_VISUAL_SAFETY_DEVICE", "cpu").strip() or "cpu",
        f5_device=os.environ.get("LALADUB_F5_DEVICE", "auto"),
        f5_speed=float(os.environ.get("LALADUB_F5_SPEED", "1.0")),
        f5_nfe_step=int(os.environ.get("LALADUB_F5_NFE_STEP", "32")),
        f5_cfg_strength=float(os.environ.get("LALADUB_F5_CFG_STRENGTH", "2.0")),
        f5_target_rms=float(os.environ.get("LALADUB_F5_TARGET_RMS", "0.1")),
        f5_cross_fade_duration=float(os.environ.get("LALADUB_F5_CROSS_FADE_DURATION", "0.15")),
        f5_remove_silence=_parse_bool(os.environ.get("LALADUB_F5_REMOVE_SILENCE", "0")),
        f5_timeout_seconds=int(os.environ.get("LALADUB_F5_TIMEOUT_SECONDS", "1800")),
        qwen3_python=_optional_path(os.environ.get("LALADUB_QWEN3_PYTHON", ".venv-qwen3tts\\Scripts\\python.exe")),
        qwen3_model=os.environ.get("LALADUB_QWEN3_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        qwen3_cache_dir=Path(os.environ.get("LALADUB_QWEN3_CACHE_DIR", "models/qwen3tts")),
        qwen3_timeout_seconds=int(os.environ.get("LALADUB_QWEN3_TIMEOUT_SECONDS", "1800")),
        chatterbox_python=_optional_path(
            os.environ.get("LALADUB_CHATTERBOX_PYTHON", ".venv-chatterbox\\Scripts\\python.exe")
        ),
        chatterbox_model=os.environ.get("LALADUB_CHATTERBOX_MODEL", "v3"),
        chatterbox_device=os.environ.get("LALADUB_CHATTERBOX_DEVICE", "auto"),
        chatterbox_cache_dir=Path(os.environ.get("LALADUB_CHATTERBOX_CACHE_DIR", "models/chatterbox")),
        chatterbox_exaggeration=float(os.environ.get("LALADUB_CHATTERBOX_EXAGGERATION", "0.5")),
        chatterbox_cfg_weight=float(os.environ.get("LALADUB_CHATTERBOX_CFG_WEIGHT", "0.5")),
        chatterbox_timeout_seconds=int(os.environ.get("LALADUB_CHATTERBOX_TIMEOUT_SECONDS", "1800")),
        cosyvoice_python=_optional_path(os.environ.get("LALADUB_COSYVOICE_PYTHON", ".venv-cosyvoice\\Scripts\\python.exe")),
        cosyvoice_repo_dir=Path(os.environ.get("LALADUB_COSYVOICE_REPO_DIR", "models/cosyvoice/CosyVoice")),
        cosyvoice_model_dir=Path(
            os.environ.get("LALADUB_COSYVOICE_MODEL_DIR", "models/cosyvoice/pretrained_models/Fun-CosyVoice3-0.5B")
        ),
        cosyvoice_model_id=os.environ.get("LALADUB_COSYVOICE_MODEL_ID", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"),
        cosyvoice_mode=os.environ.get("LALADUB_COSYVOICE_MODE", "cross_lingual"),
        cosyvoice_instruction=os.environ.get(
            "LALADUB_COSYVOICE_INSTRUCTION",
            "You are a helpful assistant.<|endofprompt|>",
        ),
        cosyvoice_device=os.environ.get("LALADUB_COSYVOICE_DEVICE", "auto"),
        cosyvoice_speed=float(os.environ.get("LALADUB_COSYVOICE_SPEED", "1.0")),
        cosyvoice_timeout_seconds=int(os.environ.get("LALADUB_COSYVOICE_TIMEOUT_SECONDS", "1800")),
        multi_speaker=_parse_bool(os.environ.get("LALADUB_MULTI_SPEAKER", "1")),
        speaker_reference_seconds=float(os.environ.get("LALADUB_SPEAKER_REFERENCE_SECONDS", "5.0")),
        speaker_clustering=_parse_bool(os.environ.get("LALADUB_SPEAKER_CLUSTERING", "1")),
        max_speaker_clusters=max(1, int(os.environ.get("LALADUB_MAX_SPEAKER_CLUSTERS", "6"))),
        speaker_cluster_threshold=float(os.environ.get("LALADUB_SPEAKER_CLUSTER_THRESHOLD", "0.08")),
        separation=os.environ.get("LALADUB_SEPARATION", "demucs"),
        separation_device=os.environ.get("LALADUB_SEPARATION_DEVICE", "cpu"),
        demucs_model=os.environ.get("LALADUB_DEMUCS_MODEL", "htdemucs"),
        audio_bed=os.environ.get("LALADUB_AUDIO_BED", "instrumental"),
        asr_backend=os.environ.get("LALADUB_ASR_BACKEND", "faster-whisper"),
        default_asr_method=os.environ.get("LALADUB_DEFAULT_ASR_METHOD", "ow-large-v3-chaos-backbone").strip().lower(),
        whisper_model=os.environ.get("LALADUB_WHISPER_MODEL", "small"),
        whisper_only_model=os.environ.get("LALADUB_WHISPER_ONLY_MODEL", "turbo"),
        whisper_only_device=os.environ.get("LALADUB_WHISPER_ONLY_DEVICE", "cpu"),
        artifact_whisper_device=os.environ.get("LALADUB_ARTIFACT_WHISPER_DEVICE", "cpu"),
        whisper_device=os.environ.get("LALADUB_WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.environ.get("LALADUB_WHISPER_COMPUTE_TYPE", "int8"),
        suppress_plain_ascii_tokens=_parse_bool(os.environ.get("LALADUB_SUPPRESS_PLAIN_ASCII_TOKENS", "0")),
        original_volume=float(os.environ.get("LALADUB_ORIGINAL_VOLUME", "0.35")),
        dub_volume=float(os.environ.get("LALADUB_DUB_VOLUME", "1.0")),
        collapse_repetitions=_parse_bool(os.environ.get("LALADUB_COLLAPSE_REPETITIONS", "1")),
        max_phrase_repeats=int(os.environ.get("LALADUB_MAX_PHRASE_REPEATS", "2")),
        max_word_repeats=int(os.environ.get("LALADUB_MAX_WORD_REPEATS", "3")),
        inject_artifacts=_parse_bool(os.environ.get("LALADUB_INJECT_ARTIFACTS", "1")),
        artifact_max_segments=int(os.environ.get("LALADUB_ARTIFACT_MAX_SEGMENTS", "12")),
        artifact_ratio=max(0.0, min(1.0, float(os.environ.get("LALADUB_ARTIFACT_RATIO", "0.20")))),
        artifact_min_source_segments=max(0, int(os.environ.get("LALADUB_ARTIFACT_MIN_SOURCE_SEGMENTS", "5"))),
        artifact_min_gap_seconds=float(os.environ.get("LALADUB_ARTIFACT_MIN_GAP_SECONDS", "0.5")),
        distort_translation=_parse_bool(os.environ.get("LALADUB_DISTORT_TRANSLATION", "1")),
        translation_pivots=os.environ.get(
            "LALADUB_TRANSLATION_PIVOTS",
            "input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar|input,en,de|input,ja,ko,en|input,tr,ar,en|en,ms,he,en",
        ),
        translation_second_pass_ratio=max(
            0.0,
            min(1.0, float(os.environ.get("LALADUB_TRANSLATION_SECOND_PASS_RATIO", "0.0"))),
        ),
    )


def _parse_user_ids(raw: str) -> set[int]:
    result: set[int] = set()
    for chunk in raw.replace(";", ",").replace(" ", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        result.add(int(chunk))
    return result


def _parse_user_ids_file(path: Path) -> set[int]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            values.append(value)
    return _parse_user_ids(",".join(values))


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _optional_path(value: str | None) -> Path | None:
    value = _empty_to_none(value)
    return Path(value) if value else None


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "да"}
