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
    reference_timing_asr: bool = False
    artifact_source_lang: str | None = None
    input_pivot_lang: str | None = None
    artifact_whisper_model: str = "turbo"
    artifact_whisper_device: str = "cpu"
    artifact_chaos_mode: bool = False
    content_chaos_backbone: bool = False
    inject_artifacts: bool = True
    artifact_max_segments: int = 12
    artifact_ratio: float = 0.2
    artifact_min_source_segments: int = 5
    artifact_min_gap_seconds: float = 0.5
    distort_translation: bool = True
    distort_main_translation: bool = False
    translation_chaos: str = "crooked"
    translation_seed: str | None = None
    translation_pivots: str = "en,de"
    # Every hop is its own API call, so a 7-language chain costs seven requests
    # for one line. Capping the length is what keeps the free translators from
    # rate-limiting a job halfway through.
    max_translation_hops: int = 3
    # Share of foreign-channel mentions in artifacts rebranded to ours (0..1).
    channel_rebrand_share: float = 0.5
    # How many times one short line may repeat across a whole video before the
    # extras are dropped. 0 disables the limit.
    max_line_repeats: int = 5
    translation_second_pass_ratio: float = 0.0
    collapse_repetitions: bool = True
    max_phrase_repeats: int = 2
    max_word_repeats: int = 3
    censor_percent: int = 0
    multi_speaker: bool = True
    speaker_reference_seconds: float = 3.5
    speaker_clustering: bool = True
    max_speaker_clusters: int = 6
    speaker_cluster_threshold: float = 0.08
    speaker_count_exact: int | None = None
    diarization_python: Path | None = None
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_device: str = "auto"
    diarization_cache_dir: Path = Path("models/diarization")
    diarization_token_file: Path | None = None
    diarization_timeout_seconds: int = 1800
    translator: str = "identity"
    libretranslate_url: str = "http://127.0.0.1:5000/translate"
    libretranslate_api_key: str | None = None
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str | None = None
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.9
    llm_timeout_seconds: int = 180
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
    qwen3_python: Path | None = None
    qwen3_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    qwen3_cache_dir: Path = Path("models/qwen3tts")
    qwen3_timeout_seconds: int = 1800
    cosyvoice_python: Path | None = None
    cosyvoice_repo_dir: Path = Path("models/cosyvoice/CosyVoice")
    cosyvoice_model_dir: Path = Path("models/cosyvoice/pretrained_models/Fun-CosyVoice3-0.5B")
    cosyvoice_model_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    cosyvoice_mode: str = "cross_lingual"
    cosyvoice_instruction: str = "You are a helpful assistant.<|endofprompt|>"
    cosyvoice_device: str = "auto"
    cosyvoice_speed: float = 1.0
    cosyvoice_timeout_seconds: int = 1800
    chatterbox_python: Path | None = None
    chatterbox_model: str = "v3"
    chatterbox_device: str = "auto"
    chatterbox_cache_dir: Path = Path("models/chatterbox")
    chatterbox_exaggeration: float = 0.5
    chatterbox_cfg_weight: float = 0.5
    chatterbox_timeout_seconds: int = 1800
    fish_python: Path | None = None
    fish_repo_dir: Path = Path("models/fish-speech")
    fish_model_dir: Path = Path("models/fish-s2-pro")
    fish_device: str = "auto"
    fish_timeout_seconds: int = 1800
    moss_python: Path | None = None
    moss_model_dir: Path = Path("models/moss/MOSS-TTS-Local-Transformer-v1.5")
    moss_codec_dir: Path = Path("models/moss/MOSS-Audio-Tokenizer-v2")
    moss_device: str = "auto"
    moss_timeout_seconds: int = 1800
    separation: str = "none"
    separation_device: str = "cpu"
    bsroformer_python: Path | None = None
    bsroformer_model_dir: Path = Path("models/audio-separator")
    bsroformer_model_file: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    bsroformer_timeout_seconds: int = 600
    demucs_model: str = "htdemucs"
    audio_bed: str = "original"
    glitch_profile: str = "faithful"
    ghost_gap_seconds: float = 2.7
    original_volume: float = 0.18
    dub_volume: float = 1.0
    fit_to_segments: bool = True
    trim_tts_silence: bool = True
    tts_max_pause_seconds: float = 0.3
    keep_workdir: bool = True
    resume: bool = True
    preprocess_only: bool = False
    progress_callback: Callable[[str, int | None, int | None, str | None], None] | None = None
