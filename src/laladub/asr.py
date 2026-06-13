from __future__ import annotations

from copy import copy
from functools import lru_cache
from pathlib import Path

from .models import DubConfig, Segment


def transcribe(audio_path: Path, config: DubConfig) -> list[Segment]:
    backend = config.asr_backend.lower()
    if backend == "faster-whisper":
        return _transcribe_faster_whisper(audio_path, config)
    if backend == "openai-whisper":
        return _transcribe_openai_whisper(audio_path, config)
    raise RuntimeError(f"Unknown ASR backend: {config.asr_backend}")


def _transcribe_faster_whisper(audio_path: Path, config: DubConfig) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: python -m pip install -e .[asr]"
        ) from exc

    glitchy = config.glitch_profile in {"faithful", "ghost"}
    vad_filter = config.vad_filter if not glitchy else False

    model = WhisperModel(
        config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
    )

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=config.source_lang,
        task=config.whisper_task,
        vad_filter=vad_filter,
        beam_size=5,
        condition_on_previous_text=config.condition_on_previous_text,
        initial_prompt=config.initial_prompt,
        compression_ratio_threshold=10.0 if glitchy else 2.4,
        log_prob_threshold=-2.0 if glitchy else -1.0,
        no_speech_threshold=1.0 if glitchy else 0.6,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if glitchy else 0.0,
    )
    detected_language = getattr(info, "language", None)
    if config.source_lang is None and detected_language:
        config.source_lang = detected_language
        print(f"      Detected source language: {detected_language}")

    result: list[Segment] = []
    for item in segments_iter:
        text = item.text.strip()
        if not text and config.glitch_profile == "clean":
            continue
        result.append(Segment(start=float(item.start), end=float(item.end), text=text))
    return result


def _transcribe_openai_whisper(audio_path: Path, config: DubConfig) -> list[Segment]:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run: python -m pip install openai-whisper"
        ) from exc

    glitchy = config.glitch_profile in {"faithful", "ghost"}
    device = None if config.whisper_device == "auto" else config.whisper_device
    model = _load_openai_whisper_model(config.whisper_model, device)
    decode_options = {}
    if config.force_source_language and config.source_lang:
        print(f"      Force source language bias: {config.source_lang}")
    if config.suppress_plain_ascii_tokens and config.source_lang and config.source_lang != "en":
        decode_options["suppress_tokens"] = _ascii_text_suppress_tokens(model.is_multilingual, config.source_lang)
        print("      Hard source language bias: suppressing plain ASCII text tokens")

    try:
        result = model.transcribe(
            str(audio_path),
            language=config.source_lang,
            task=config.whisper_task,
            fp16=config.whisper_device == "cuda",
            verbose=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if glitchy else 0.0,
            compression_ratio_threshold=None if glitchy else 2.4,
            logprob_threshold=None if glitchy else -1.0,
            no_speech_threshold=None if glitchy else 0.6,
            condition_on_previous_text=config.condition_on_previous_text,
            initial_prompt=config.initial_prompt,
            word_timestamps=False,
            hallucination_silence_threshold=config.hallucination_silence_threshold,
            **decode_options,
        )
    except RuntimeError as exc:
        if not _is_openai_empty_decoder_error(exc):
            raise
        return _handle_openai_empty_decoder_error(audio_path, config, exc)

    detected_language = result.get("language")
    if config.source_lang is None and detected_language:
        config.source_lang = detected_language
        print(f"      Detected source language: {detected_language}")

    segments: list[Segment] = []
    for item in result.get("segments", []):
        text = str(item.get("text", "")).strip()
        if not text and config.glitch_profile == "clean":
            continue
        segments.append(
            Segment(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=text,
            )
        )
    return segments


def _is_openai_empty_decoder_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "cannot reshape tensor of 0 elements" in message and "shape [1, 0" in message


def _handle_openai_empty_decoder_error(audio_path: Path, config: DubConfig, exc: RuntimeError) -> list[Segment]:
    print(f"      OpenAI Whisper empty decoder result; continuing safely: {exc}")
    if config.glitch_profile == "clean" and not config.force_source_language:
        fallback = copy(config)
        fallback.asr_backend = "faster-whisper"
        fallback.whisper_device = "cpu" if fallback.whisper_device == "auto" else fallback.whisper_device
        fallback.whisper_compute_type = "int8" if fallback.whisper_compute_type == "auto" else fallback.whisper_compute_type
        fallback.vad_filter = False
        try:
            print("      Retrying ASR with faster-whisper fallback")
            return _transcribe_faster_whisper(audio_path, fallback)
        except Exception as fallback_exc:
            print(f"      faster-whisper fallback skipped: {type(fallback_exc).__name__}: {fallback_exc}")
    return []


@lru_cache(maxsize=8)
def _load_openai_whisper_model(model_name: str, device: str | None) -> object:
    import whisper

    return whisper.load_model(model_name, device=device)


def clear_openai_whisper_cache() -> None:
    _load_openai_whisper_model.cache_clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@lru_cache(maxsize=32)
def _ascii_text_suppress_tokens(is_multilingual: bool, language: str) -> list[int]:
    from whisper.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(is_multilingual, language=language, task="transcribe")
    suppress_tokens = [-1]
    for token_id in range(tokenizer.eot):
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            continue
        if _is_plain_ascii_text_token(text):
            suppress_tokens.append(token_id)
    return suppress_tokens


def _is_plain_ascii_text_token(text: str) -> bool:
    has_ascii_alpha = any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)
    if not has_ascii_alpha:
        return False
    return all(ord(char) < 128 for char in text)
