from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .ffmpeg import which
from .models import DubConfig
from .pipeline import run_dub
from .tts import list_sapi_voices


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dub":
        config = DubConfig(
            output=args.output,
            workdir=args.workdir,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            asr_backend=args.asr_backend,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            whisper_compute_type=args.whisper_compute_type,
            whisper_task=args.whisper_task,
            vad_filter=args.vad_filter,
            condition_on_previous_text=not args.no_condition_on_previous_text,
            initial_prompt=args.initial_prompt,
            hallucination_silence_threshold=args.hallucination_silence_threshold,
            force_source_language=args.force_source_language,
            suppress_plain_ascii_tokens=args.suppress_plain_ascii_tokens,
            translator=args.translator,
            libretranslate_url=args.libretranslate_url,
            libretranslate_api_key=args.libretranslate_api_key,
            tts=args.tts,
            voice=args.voice,
            sapi_rate=args.sapi_rate,
            sapi_volume=args.sapi_volume,
            piper_cmd=args.piper_cmd,
            piper_model=args.piper_model,
            speaker_wav=args.speaker_wav,
            xtts_model=args.xtts_model,
            xtts_device=args.xtts_device,
            xtts_speed=args.xtts_speed,
            f5_python=args.f5_python,
            f5_model=args.f5_model,
            f5_hf_repo=args.f5_hf_repo,
            f5_hf_ckpt_path=args.f5_hf_ckpt_path,
            f5_hf_vocab_path=args.f5_hf_vocab_path,
            f5_ckpt_file=args.f5_ckpt_file,
            f5_vocab_file=args.f5_vocab_file,
            f5_cache_dir=args.f5_cache_dir,
            media_cache_dir=args.media_cache_dir,
            f5_device=args.f5_device,
            f5_speed=args.f5_speed,
            f5_nfe_step=args.f5_nfe_step,
            f5_cfg_strength=args.f5_cfg_strength,
            f5_target_rms=args.f5_target_rms,
            f5_cross_fade_duration=args.f5_cross_fade_duration,
            f5_remove_silence=args.f5_remove_silence,
            f5_timeout_seconds=args.f5_timeout_seconds,
            chatterbox_python=args.chatterbox_python,
            chatterbox_model=args.chatterbox_model,
            chatterbox_device=args.chatterbox_device,
            chatterbox_cache_dir=args.chatterbox_cache_dir,
            chatterbox_exaggeration=args.chatterbox_exaggeration,
            chatterbox_cfg_weight=args.chatterbox_cfg_weight,
            chatterbox_timeout_seconds=args.chatterbox_timeout_seconds,
            cosyvoice_python=args.cosyvoice_python,
            cosyvoice_repo_dir=args.cosyvoice_repo_dir,
            cosyvoice_model_dir=args.cosyvoice_model_dir,
            cosyvoice_model_id=args.cosyvoice_model_id,
            cosyvoice_mode=args.cosyvoice_mode,
            cosyvoice_instruction=args.cosyvoice_instruction,
            cosyvoice_device=args.cosyvoice_device,
            cosyvoice_speed=args.cosyvoice_speed,
            cosyvoice_timeout_seconds=args.cosyvoice_timeout_seconds,
            multi_speaker=not args.no_multi_speaker,
            speaker_reference_seconds=args.speaker_reference_seconds,
            separation=args.separation,
            separation_device=args.separation_device,
            demucs_model=args.demucs_model,
            audio_bed=args.audio_bed,
            glitch_profile=args.glitch_profile,
            ghost_gap_seconds=args.ghost_gap_seconds,
            original_volume=args.original_volume,
            dub_volume=args.dub_volume,
            collapse_repetitions=not args.no_collapse_repetitions,
            max_phrase_repeats=args.max_phrase_repeats,
            max_word_repeats=args.max_word_repeats,
            fit_to_segments=not args.no_fit_to_segments,
            translation_pivots=args.translation_pivots,
            translation_second_pass_ratio=args.translation_second_pass_ratio,
            keep_workdir=True,
        )
        run_dub(args.video, config)
        return

    if args.command == "voices":
        print(list_sapi_voices() or "No SAPI voices found.")
        return

    if args.command == "doctor":
        run_doctor()
        return

    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laladub", description="Local video translator and dubbing pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    dub = subparsers.add_parser("dub", help="Translate and dub a video.")
    dub.add_argument("video", type=Path, help="Input video path.")
    dub.add_argument("--output", "-o", type=Path, required=True, help="Output video path.")
    dub.add_argument("--workdir", type=Path, default=Path("runs/current"), help="Working directory.")
    dub.add_argument("--source-lang", default=None, help="Source language code, e.g. vi, en, ru.")
    dub.add_argument("--target-lang", default="ru", help="Target language code.")
    dub.add_argument("--asr-backend", default="faster-whisper", choices=["faster-whisper", "openai-whisper"])
    dub.add_argument("--whisper-model", default="small", help="faster-whisper model size or local model path.")
    dub.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "cuda"], help="Whisper device.")
    dub.add_argument("--whisper-compute-type", default="auto", help="Whisper compute type, e.g. int8, float16.")
    dub.add_argument("--whisper-task", default="transcribe", choices=["transcribe", "translate"], help="Whisper task.")
    dub.add_argument("--vad-filter", action="store_true", help="Enable VAD. Disabled automatically in glitchy modes.")
    dub.add_argument("--no-condition-on-previous-text", action="store_true")
    dub.add_argument("--initial-prompt", default=None)
    dub.add_argument("--hallucination-silence-threshold", type=float, default=None)
    dub.add_argument(
        "--force-source-language",
        action="store_true",
        help="Tell Whisper to treat --source-lang as the chosen input language.",
    )
    dub.add_argument(
        "--suppress-plain-ascii-tokens",
        action="store_true",
        help="Old hard wrong-language mode: suppress plain ASCII text tokens for non-English source languages.",
    )
    dub.add_argument(
        "--translator",
        default="identity",
        choices=["identity", "hybrid", "googleweb", "mymemory", "argos", "libretranslate"],
        help="Translation provider.",
    )
    dub.add_argument("--libretranslate-url", default="http://127.0.0.1:5000/translate")
    dub.add_argument("--libretranslate-api-key", default=None)
    dub.add_argument(
        "--tts",
        default="sapi",
        choices=["sapi", "piper", "xtts", "f5", "chatterbox", "cosyvoice", "none"],
        help="TTS provider.",
    )
    dub.add_argument("--voice", default=None, help="SAPI voice name.")
    dub.add_argument("--sapi-rate", type=int, default=0, help="SAPI rate from -10 to 10.")
    dub.add_argument("--sapi-volume", type=int, default=100, help="SAPI volume from 0 to 100.")
    dub.add_argument("--piper-cmd", default="piper", help="Piper executable.")
    dub.add_argument("--piper-model", type=Path, default=None, help="Piper .onnx voice model.")
    dub.add_argument("--speaker-wav", type=Path, default=None, help="Speaker reference WAV for clone TTS.")
    dub.add_argument("--xtts-model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    dub.add_argument("--xtts-device", default="cpu", choices=["cpu", "cuda"], help="XTTS device.")
    dub.add_argument("--xtts-speed", type=float, default=1.0, help="XTTS speech speed.")
    dub.add_argument("--f5-python", type=Path, default=Path(".venv-f5tts") / "Scripts" / "python.exe")
    dub.add_argument("--f5-model", default="F5TTS_v1_Base")
    dub.add_argument("--f5-hf-repo", default="Misha24-10/F5-TTS_RUSSIAN")
    dub.add_argument("--f5-hf-ckpt-path", default="F5TTS_v1_Base_v2/model_last_inference.safetensors")
    dub.add_argument("--f5-hf-vocab-path", default="F5TTS_v1_Base/vocab.txt")
    dub.add_argument("--f5-ckpt-file", type=Path, default=None)
    dub.add_argument("--f5-vocab-file", type=Path, default=None)
    dub.add_argument("--f5-cache-dir", type=Path, default=Path("models/f5tts"))
    dub.add_argument("--media-cache-dir", type=Path, default=Path("runs/cache/media"))
    dub.add_argument("--f5-device", default="auto", choices=["auto", "cpu", "cuda"])
    dub.add_argument("--f5-speed", type=float, default=1.0)
    dub.add_argument("--f5-nfe-step", type=int, default=32)
    dub.add_argument("--f5-cfg-strength", type=float, default=2.0)
    dub.add_argument("--f5-target-rms", type=float, default=0.1)
    dub.add_argument("--f5-cross-fade-duration", type=float, default=0.15)
    dub.add_argument("--f5-remove-silence", action="store_true")
    dub.add_argument("--f5-timeout-seconds", type=int, default=1800)
    dub.add_argument("--chatterbox-python", type=Path, default=Path(".venv-chatterbox") / "Scripts" / "python.exe")
    dub.add_argument("--chatterbox-model", default="v3")
    dub.add_argument("--chatterbox-device", default="auto", choices=["auto", "cpu", "cuda"])
    dub.add_argument("--chatterbox-cache-dir", type=Path, default=Path("models/chatterbox"))
    dub.add_argument("--chatterbox-exaggeration", type=float, default=0.5)
    dub.add_argument("--chatterbox-cfg-weight", type=float, default=0.5)
    dub.add_argument("--chatterbox-timeout-seconds", type=int, default=1800)
    dub.add_argument("--cosyvoice-python", type=Path, default=Path(".venv-cosyvoice") / "Scripts" / "python.exe")
    dub.add_argument("--cosyvoice-repo-dir", type=Path, default=Path("models/cosyvoice/CosyVoice"))
    dub.add_argument(
        "--cosyvoice-model-dir",
        type=Path,
        default=Path("models/cosyvoice/pretrained_models/Fun-CosyVoice3-0.5B"),
    )
    dub.add_argument("--cosyvoice-model-id", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    dub.add_argument("--cosyvoice-mode", default="cross_lingual", choices=["cross_lingual", "zero_shot"])
    dub.add_argument("--cosyvoice-instruction", default="You are a helpful assistant.<|endofprompt|>")
    dub.add_argument("--cosyvoice-device", default="auto", choices=["auto", "cpu", "cuda"])
    dub.add_argument("--cosyvoice-speed", type=float, default=1.0)
    dub.add_argument("--cosyvoice-timeout-seconds", type=int, default=1800)
    dub.add_argument("--no-multi-speaker", action="store_true", help="Use one speaker reference for all segments.")
    dub.add_argument(
        "--speaker-reference-seconds",
        type=float,
        default=3.5,
        help="Seconds of source vocals to use as per-segment XTTS speaker reference.",
    )
    dub.add_argument("--separation", default="none", choices=["none", "demucs"], help="Vocal separation provider.")
    dub.add_argument("--separation-device", default="cpu", choices=["cpu", "cuda"], help="Separation device.")
    dub.add_argument("--demucs-model", default="htdemucs", help="Demucs model name.")
    dub.add_argument(
        "--audio-bed",
        default="original",
        choices=["original", "instrumental", "dub-only"],
        help="Final audio bed behind the dub.",
    )
    dub.add_argument(
        "--glitch-profile",
        default="faithful",
        choices=["clean", "faithful", "ghost"],
        help="ASR artifact handling mode.",
    )
    dub.add_argument("--ghost-gap-seconds", type=float, default=2.7, help="Minimum pause length for ghost insertions.")
    dub.add_argument("--original-volume", type=float, default=0.18, help="Original audio volume in final mix.")
    dub.add_argument("--dub-volume", type=float, default=1.0, help="Dub audio volume in final mix.")
    dub.add_argument("--no-collapse-repetitions", action="store_true", help="Keep repeated ASR/translation loops.")
    dub.add_argument("--max-phrase-repeats", type=int, default=2, help="Maximum consecutive repeated phrase copies.")
    dub.add_argument("--max-word-repeats", type=int, default=3, help="Maximum consecutive repeated single words.")
    dub.add_argument(
        "--translation-pivots",
        default="input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar|input,en,de|input,ja,ko,en|input,tr,ar,en|en,ms,he,en",
        help="Round-trip pivot chains separated by |.",
    )
    dub.add_argument(
        "--translation-second-pass-ratio",
        type=float,
        default=0.0,
        help="Share of regular translated segments that go through a second pivot pass.",
    )
    dub.add_argument("--no-fit-to-segments", action="store_true", help="Do not time-stretch TTS clips.")

    subparsers.add_parser("voices", help="List Windows SAPI voices.")
    subparsers.add_parser("doctor", help="Check local tools and optional Python packages.")
    return parser


def run_doctor() -> None:
    checks = [
        ("ffmpeg", bool(which("ffmpeg"))),
        ("ffprobe", bool(which("ffprobe"))),
        ("powershell", bool(which("powershell"))),
        ("piper", bool(which("piper"))),
        ("faster_whisper", importlib.util.find_spec("faster_whisper") is not None),
        ("openai_whisper", importlib.util.find_spec("whisper") is not None),
        ("argostranslate", importlib.util.find_spec("argostranslate") is not None),
        ("TTS", importlib.util.find_spec("TTS") is not None),
        ("f5_tts", importlib.util.find_spec("f5_tts") is not None),
        ("demucs", importlib.util.find_spec("demucs") is not None),
        ("yt_dlp", importlib.util.find_spec("yt_dlp") is not None),
    ]
    width = max(len(name) for name, _ok in checks)
    for name, ok in checks:
        marker = "OK" if ok else "missing"
        print(f"{name.ljust(width)}  {marker}")
