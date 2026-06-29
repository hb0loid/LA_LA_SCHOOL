from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    translated_text: str | None = None
    speaker_wav: Path | None = None
    speaker_id: str | None = None
    speaker_ref_text: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def spoken_text(self) -> str:
        return (self.translated_text or self.text).strip()


@dataclass(slots=True)
class DubConfig:
    output: Path
    workdir: Path
    source_lang: str | None = None
    target_lang: str = "ru"
    asr_backend: str = "faster-whisper"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_task: str = "transcribe"
    vad_filter: bool = False
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    hallucination_silence_threshold: float | None = None
    force_source_language: bool = False
    suppress_plain_ascii_tokens: bool = False
    asr_retry_on_repetition: bool = True
    asr_fallback_on_sparse: bool = False
    artifact_source_lang: str | None = None
    input_pivot_lang: str | None = None
    artifact_whisper_model: str = "turbo"
    artifact_whisper_device: str = "cpu"
    artifact_chaos_mode: bool = False
    inject_artifacts: bool = True
    artifact_max_segments: int = 12
    artifact_min_gap_seconds: float = 0.5
    distort_translation: bool = True
    distort_main_translation: bool = False
    translation_pivots: str = "en,de"
    translation_second_pass_ratio: float = 0.0
    collapse_repetitions: bool = True
    max_phrase_repeats: int = 2
    max_word_repeats: int = 3
    multi_speaker: bool = True
    speaker_reference_seconds: float = 3.5
    speaker_clustering: bool = True
    max_speaker_clusters: int = 6
    speaker_cluster_threshold: float = 0.08
    translator: str = "identity"
    libretranslate_url: str = "http://127.0.0.1:5000/translate"
    libretranslate_api_key: str | None = None
    tts: str = "sapi"
    voice: str | None = None
    sapi_rate: int = 0
    sapi_volume: int = 100
    piper_cmd: str = "piper"
    piper_model: Path | None = None
    speaker_wav: Path | None = None
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_device: str = "cpu"
    xtts_speed: float = 1.0
    f5_python: Path | None = None
    f5_model: str = "F5TTS_v1_Base"
    f5_hf_repo: str = "Misha24-10/F5-TTS_RUSSIAN"
    f5_hf_ckpt_path: str = "F5TTS_v1_Base_v2/model_last_inference.safetensors"
    f5_hf_vocab_path: str = "F5TTS_v1_Base/vocab.txt"
    f5_ckpt_file: Path | None = None
    f5_vocab_file: Path | None = None
    f5_cache_dir: Path = Path("models/f5tts")
    media_cache_dir: Path = Path("runs/cache/media")
    f5_device: str = "auto"
    f5_speed: float = 1.0
    f5_nfe_step: int = 32
    f5_cfg_strength: float = 2.0
    f5_target_rms: float = 0.1
    f5_cross_fade_duration: float = 0.15
    f5_remove_silence: bool = False
    f5_timeout_seconds: int = 1800
    separation: str = "none"
    separation_device: str = "cpu"
    demucs_model: str = "htdemucs"
    audio_bed: str = "original"
    glitch_profile: str = "faithful"
    ghost_gap_seconds: float = 2.7
    original_volume: float = 0.18
    dub_volume: float = 1.0
    fit_to_segments: bool = True
    keep_workdir: bool = True
    resume: bool = True
    progress_callback: Callable[[str, int | None, int | None, str | None], None] | None = None
