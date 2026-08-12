from __future__ import annotations

import base64
from collections import deque
import contextlib
from copy import copy
import gc
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Iterator

from .ffmpeg import concat_wavs, make_silence, probe_duration
from .models import DubConfig, Segment


class TTSError(RuntimeError):
    pass


def _format_video_position(items: list[tuple[int, Segment, Path]], completed: int) -> str:
    if not items:
        return ""
    index = max(0, min(completed, len(items) - 1))
    position = max(0.0, float(items[index][1].end))
    duration = max(position, max(float(item[1].end) for item in items))

    def stamp(seconds: float) -> str:
        total = max(0, round(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

    ratio = max(0.0, min(1.0, position / max(0.001, duration)))
    filled = round(12 * ratio)
    bar = "[" + "#" * filled + "-" * (12 - filled) + "]"
    return f"видео {bar} {stamp(position)} / {stamp(duration)}"


_XTTS_CACHE: dict[tuple[str, str], object] = {}
_F5_CACHE: dict[tuple[str, str, str, str], object] = {}
_F5_PRONUNCIATION_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Ghiền Mì Gõ", "гиен ми го"),
    ("Ghien Mi Go", "гиен ми го"),
    ("Amara.org", "амара орг"),
    ("YouTube", "ютуб"),
    ("TikTok", "тикток"),
    ("ElevenLabs", "элевенлабс"),
    ("DimaTorzok", "дима торзок"),
    ("Ming Jing", "минь джинг"),
    ("Mingjing", "минь джинг"),
    ("Diandian", "дян дян"),
    ("Dian Dian", "дян дян"),
    ("Altyazı M.K.", "субтитры эм ка"),
    ("Altyazi M.K.", "субтитры эм ка"),
    ("Субтитры М.К.", "субтитры эм ка"),
)


def clear_tts_model_caches() -> None:
    _XTTS_CACHE.clear()
    _F5_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def synthesize_segment(segment: Segment, output_path: Path, config: DubConfig) -> None:
    provider = config.tts.lower()
    text = segment.spoken_text
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "none" or not text:
        make_silence(output_path, max(0.15, segment.duration))
        return

    if provider == "sapi":
        _synthesize_sapi(text, output_path, config)
        return

    if provider == "piper":
        _synthesize_piper(text, output_path, config)
        return

    if provider == "xtts":
        try:
            _synthesize_xtts(segment, text, output_path, config)
        except Exception as exc:
            print(f"      XTTS segment fallback to SAPI: {type(exc).__name__}: {exc}")
            try:
                _synthesize_sapi(text, output_path, config)
            except Exception as sapi_exc:
                print(f"      SAPI segment fallback to silence: {type(sapi_exc).__name__}: {sapi_exc}")
                make_silence(output_path, max(0.15, segment.duration))
        return

    if provider in {"f5", "f5tts"}:
        try:
            _synthesize_f5tts(segment, text, output_path, config)
        except Exception as exc:
            print(f"      F5-TTS segment fallback to XTTS: {type(exc).__name__}: {exc}")
            output_path.unlink(missing_ok=True)
            try:
                _synthesize_xtts(segment, text, output_path, config)
            except Exception as xtts_exc:
                print(f"      XTTS segment fallback to SAPI: {type(xtts_exc).__name__}: {xtts_exc}")
                output_path.unlink(missing_ok=True)
                try:
                    _synthesize_sapi(text, output_path, config)
                except Exception as sapi_exc:
                    print(f"      SAPI segment fallback to silence: {type(sapi_exc).__name__}: {sapi_exc}")
                    make_silence(output_path, max(0.15, segment.duration))
        return

    if provider in {"qwen3", "qwen3-tts", "qwen3tts"}:
        synthesize_qwen3_batch([(1, segment, output_path)], config)
        return

    if provider in {"cosyvoice", "cosyvoice-tts", "cosyvoicetts", "cosy"}:
        try:
            _synthesize_cosyvoice(segment, text, output_path, config)
        except Exception as exc:
            _fallback_segment_to_f5(segment, text, output_path, config, "CosyVoice", exc)
        return

    if provider in {"moss", "moss-tts", "mosstts", "moss-v1.5"}:
        try:
            synthesize_moss_batch([(1, segment, output_path)], config)
        except Exception as exc:
            _fallback_segment_to_f5(segment, text, output_path, config, "MOSS", exc)
        return

    raise TTSError(f"Unknown TTS provider: {config.tts}")


def _fallback_segment_to_f5(
    segment: Segment,
    text: str,
    output_path: Path,
    config: DubConfig,
    provider_label: str,
    error: Exception,
) -> None:
    if _tts_output_ready(output_path):
        print(f"      {provider_label} segment failed after WAV was created; keeping output: {type(error).__name__}: {error}")
        return
    print(f"      {provider_label} segment fallback to F5: {type(error).__name__}: {error}")
    output_path.unlink(missing_ok=True)
    fallback_config = copy(config)
    fallback_config.tts = "f5"
    try:
        _synthesize_f5tts(segment, text, output_path, fallback_config)
    except Exception as f5_exc:
        print(f"      F5 fallback failed; using silence: {type(f5_exc).__name__}: {f5_exc}")
        output_path.unlink(missing_ok=True)
        make_silence(output_path, max(0.15, segment.duration))


def synthesize_qwen3_batch(
    items: list[tuple[int, Segment, Path]],
    config: DubConfig,
) -> None:
    if not items:
        return

    python_path = _resolve_qwen3_python(config)
    runner_path = _repo_root() / "tools" / "qwen3_tts_batch_runner.py"
    if not runner_path.is_file():
        raise TTSError(f"Qwen3-TTS runner does not exist: {runner_path}")

    manifest_items: list[dict[str, object]] = []
    for _index, segment, output_path in items:
        text = _sanitize_text_for_xtts(segment.spoken_text)
        if config.target_lang == "ru":
            text = _apply_f5_pronunciation_dictionary(text)
        speaker_wav = segment.speaker_wav or config.speaker_wav
        if not speaker_wav or not speaker_wav.is_file():
            raise TTSError(f"Qwen3-TTS speaker reference does not exist: {speaker_wav}")

        # The chaos/forced ASR text is intentionally allowed to differ from
        # the real speech in the reference audio. Qwen ICL requires an exact
        # transcript; feeding the distorted text can make generation run to
        # the token ceiling. Clone by speaker vector only for bot jobs.
        reference_text = ""
        manifest_items.append(
            {
                "output": str(output_path.resolve()),
                "text": text,
                "reference": str(speaker_wav.resolve()),
                "reference_text": reference_text,
                "x_vector_only": True,
                "target_seconds": max(0.1, float(segment.duration)),
            }
        )

    manifest_path = config.workdir / "qwen3_batch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": config.qwen3_model,
        "cache_dir": str(_resolve_repo_relative_path(config.qwen3_cache_dir)),
        "language": _qwen3_language(config.target_lang),
        "seed": 42,
        "items": manifest_items,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [str(python_path), str(runner_path), "--manifest", str(manifest_path.resolve())]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip())
        output_queue.put(None)

    reader = threading.Thread(target=read_output, name="qwen3-tts-output", daemon=True)
    reader.start()
    recent_output: deque[str] = deque(maxlen=30)
    deadline = time.monotonic() + max(30, config.qwen3_timeout_seconds)
    segment_deadline: float | None = None
    reader_done = False
    try:
        while process.poll() is None or not reader_done:
            if time.monotonic() >= deadline:
                raise TTSError(f"Qwen3-TTS batch timed out after {config.qwen3_timeout_seconds} seconds.")
            if segment_deadline is not None and time.monotonic() >= segment_deadline:
                raise TTSError("Qwen3-TTS segment exceeded the 120-second safety limit.")
            try:
                line = output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                reader_done = True
                continue
            if line:
                recent_output.append(line)
            if line.startswith("QWEN3_START\t"):
                segment_deadline = time.monotonic() + 120.0
                parts = line.split("\t")
                if len(parts) >= 3 and config.progress_callback is not None:
                    current = int(parts[1])
                    total = max(1, int(parts[2]))
                    percent = 70 + round(20 * max(0, current - 1) / total)
                    position = _format_video_position(items, current - 1)
                    config.progress_callback(
                        "Озвучиваю реплики",
                        percent,
                        100,
                        f"Qwen3 {current}/{total} · {position}",
                    )
            if line.startswith("QWEN3_PROGRESS\t"):
                segment_deadline = None
                parts = line.split("\t")
                if len(parts) >= 3:
                    done = int(parts[1])
                    total = max(1, int(parts[2]))
                    if config.progress_callback is not None:
                        percent = 70 + round(20 * done / total)
                        position = _format_video_position(items, max(0, done - 1))
                        config.progress_callback(
                            "Озвучиваю реплики",
                            percent,
                            100,
                            f"Qwen3 {done}/{total} · {position}",
                        )
    except BaseException:
        _terminate_process_tree(process)
        raise

    return_code = process.wait()
    if return_code != 0:
        details = " ".join(recent_output)
        details = re.sub(r"\s+", " ", details).strip()
        raise TTSError(f"Qwen3-TTS batch failed ({return_code}): {details[-1500:]}")

    missing = [str(path) for _index, _segment, path in items if not path.is_file() or path.stat().st_size < 1024]
    if missing:
        raise TTSError(f"Qwen3-TTS did not create valid WAV files: {', '.join(missing[:3])}")


def synthesize_cosyvoice_batch(
    items: list[tuple[int, Segment, Path]],
    config: DubConfig,
) -> None:
    if not items:
        return

    python_path = _resolve_cosyvoice_python(config)
    runner_path = _repo_root() / "tools" / "cosyvoice_tts_batch_runner.py"
    if not runner_path.is_file():
        raise TTSError(f"CosyVoice batch runner does not exist: {runner_path}")

    manifest_items: list[dict[str, object]] = []
    for _index, segment, output_path in items:
        text = _sanitize_text_for_xtts(segment.spoken_text)
        prompt_text = _sanitize_text_for_xtts(segment.speaker_ref_text or "")
        if config.target_lang == "ru":
            text = _apply_f5_pronunciation_dictionary(text)
            prompt_text = _apply_f5_pronunciation_dictionary(prompt_text)
        if not text:
            make_silence(output_path, max(0.15, segment.duration))
            continue
        speaker_wav = segment.speaker_wav or config.speaker_wav
        if not speaker_wav or not speaker_wav.is_file():
            raise TTSError(f"CosyVoice speaker reference does not exist: {speaker_wav}")
        chunks = _split_text_for_xtts(text, max_chars=260, max_words=40)
        manifest_items.append(
            {
                "output": str(output_path.resolve()),
                "text": text,
                "chunks": chunks,
                "reference": str(speaker_wav.resolve()),
                "prompt_text": prompt_text,
            }
        )

    if not manifest_items:
        return

    manifest_path = config.workdir / "cosyvoice_batch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_dir": str(_resolve_repo_relative_path(config.cosyvoice_repo_dir)),
        "model_dir": str(_resolve_repo_relative_path(config.cosyvoice_model_dir)),
        "model_id": config.cosyvoice_model_id,
        "mode": config.cosyvoice_mode,
        "instruction": config.cosyvoice_instruction,
        "device": config.cosyvoice_device,
        "speed": config.cosyvoice_speed,
        "items": manifest_items,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [str(python_path), str(runner_path), "--manifest", str(manifest_path.resolve())]
    _run_progress_tts_batch(
        command,
        items,
        config,
        label="CosyVoice",
        start_prefix="COSYVOICE_START",
        progress_prefix="COSYVOICE_PROGRESS",
        timeout_seconds=config.cosyvoice_timeout_seconds,
    )

    missing = [str(path) for _index, _segment, path in items if not path.is_file() or path.stat().st_size < 1024]
    if missing:
        raise TTSError(f"CosyVoice did not create valid WAV files: {', '.join(missing[:3])}")


def synthesize_moss_batch(
    items: list[tuple[int, Segment, Path]],
    config: DubConfig,
) -> None:
    if not items:
        return

    python_path = _resolve_moss_python(config)
    runner_path = _repo_root() / "tools" / "moss_tts_batch_runner.py"
    if not runner_path.is_file():
        raise TTSError(f"MOSS batch runner does not exist: {runner_path}")

    language = _moss_language(config.target_lang)
    manifest_items: list[dict[str, object]] = []
    for _index, segment, output_path in items:
        text = _sanitize_text_for_xtts(segment.spoken_text)
        if config.target_lang == "ru":
            text = _apply_f5_pronunciation_dictionary(text)
        if not text:
            make_silence(output_path, max(0.15, segment.duration))
            continue
        speaker_wav = segment.speaker_wav or config.speaker_wav
        if not speaker_wav or not speaker_wav.is_file():
            raise TTSError(f"MOSS speaker reference does not exist: {speaker_wav}")
        source_seconds = max(0.4, float(segment.duration))
        manifest_items.append(
            {
                "output": str(output_path.resolve()),
                "text": text,
                "reference": str(speaker_wav.resolve()),
                "source_seconds": source_seconds,
                "target_seconds": source_seconds,
                "language": language,
            }
        )

    if not manifest_items:
        return

    manifest_path = config.workdir / "moss_batch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_dir": str(_resolve_repo_relative_path(config.moss_model_dir)),
        "codec_dir": str(_resolve_repo_relative_path(config.moss_codec_dir)),
        "device": config.moss_device,
        "seed": 42,
        # Duration control makes MOSS drop words from dense subtitles and fill
        # long windows with hallucinated speech. Let the model stop naturally
        # on its EOS token, as in the official v1.5 inference example.
        "duration_control": False,
        "lead_pause_seconds": 0.3,
        "trail_pause_seconds": 0.25,
        "natural_max_new_tokens": 512,
        "edge_padding_seconds": 0.04,
        "items": manifest_items,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [str(python_path), str(runner_path), "--manifest", str(manifest_path.resolve())]
    _run_progress_tts_batch(
        command,
        items,
        config,
        label="MOSS",
        start_prefix="MOSS_START",
        progress_prefix="MOSS_PROGRESS",
        timeout_seconds=config.moss_timeout_seconds,
        extra_env={"HF_HUB_DISABLE_XET": "1"},
    )

    missing = [str(path) for _index, _segment, path in items if not path.is_file() or path.stat().st_size < 1024]
    if missing:
        raise TTSError(f"MOSS did not create valid WAV files: {', '.join(missing[:3])}")


def _run_progress_tts_batch(
    command: list[str],
    items: list[tuple[int, Segment, Path]],
    config: DubConfig,
    *,
    label: str,
    start_prefix: str,
    progress_prefix: str,
    timeout_seconds: int,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = _subprocess_env(extra_env)
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip())
        output_queue.put(None)

    reader = threading.Thread(target=read_output, name=f"{label.lower()}-tts-output", daemon=True)
    reader.start()
    recent_output: deque[str] = deque(maxlen=30)
    deadline = time.monotonic() + max(30, timeout_seconds)
    segment_deadline: float | None = None
    reader_done = False
    try:
        while process.poll() is None or not reader_done:
            if time.monotonic() >= deadline:
                raise TTSError(f"{label} batch timed out after {timeout_seconds} seconds.")
            if segment_deadline is not None and time.monotonic() >= segment_deadline:
                raise TTSError(f"{label} segment exceeded the 180-second safety limit.")
            try:
                line = output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                reader_done = True
                continue
            if line:
                recent_output.append(line)
            if line.startswith(start_prefix + "\t"):
                segment_deadline = time.monotonic() + 180.0
                parts = line.split("\t")
                if len(parts) >= 3 and config.progress_callback is not None:
                    current = int(parts[1])
                    total = max(1, int(parts[2]))
                    percent = 70 + round(20 * max(0, current - 1) / total)
                    position = _format_video_position(items, current - 1)
                    config.progress_callback(
                        "Озвучиваю реплики",
                        percent,
                        100,
                        f"{label} {current}/{total} · {position}",
                    )
            if line.startswith(progress_prefix + "\t"):
                segment_deadline = None
                parts = line.split("\t")
                if len(parts) >= 3 and config.progress_callback is not None:
                    done = int(parts[1])
                    total = max(1, int(parts[2]))
                    percent = 70 + round(20 * done / total)
                    position = _format_video_position(items, max(0, done - 1))
                    config.progress_callback(
                        "Озвучиваю реплики",
                        percent,
                        100,
                        f"{label} {done}/{total} · {position}",
                    )
    except BaseException:
        _terminate_process_tree(process)
        raise

    return_code = process.wait()
    if return_code != 0:
        details = " ".join(recent_output)
        details = re.sub(r"\s+", " ", details).strip()
        raise TTSError(f"{label} batch failed ({return_code}): {details[-1500:]}")


def list_sapi_voices() -> str:
    script = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() | ForEach-Object {
  $info = $_.VoiceInfo
  "$($info.Name)`t$($info.Culture)`t$($info.Gender)`t$($info.Age)"
}
$synth.Dispose()
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _synthesize_sapi(text: str, output_path: Path, config: DubConfig) -> None:
    script_path = output_path.with_suffix(".ps1")
    text_base64 = base64.b64encode(text.encode("utf-16-le")).decode("ascii")
    script_path.write_text(
        """
param(
  [Parameter(Mandatory=$true)][string]$TextBase64,
  [Parameter(Mandatory=$true)][string]$OutputPath,
  [string]$VoiceName = "",
  [int]$Rate = 0,
  [int]$Volume = 100
)
$ErrorActionPreference = "Stop"
$voice = New-Object -ComObject SAPI.SpVoice
if ($VoiceName -ne "") {
  $selected = $null
  foreach ($token in $voice.GetVoices()) {
    if (($token.GetDescription() -eq $VoiceName) -or ($token.GetAttribute("Name") -eq $VoiceName)) {
      $selected = $token
      break
    }
  }
  if ($null -eq $selected) { throw "SAPI voice not found: $VoiceName" }
  $voice.Voice = $selected
}
$voice.Rate = $Rate
$voice.Volume = $Volume
$textBytes = [Convert]::FromBase64String($TextBase64)
$text = [System.Text.Encoding]::Unicode.GetString($textBytes)
$stream = New-Object -ComObject SAPI.SpFileStream
$format = New-Object -ComObject SAPI.SpAudioFormat
$format.Type = 34
$stream.Format = $format
try {
  $stream.Open($OutputPath, 3, $false)
  $voice.AudioOutputStream = $stream
  [void]$voice.Speak($text, 0)
}
finally {
  if ($stream -ne $null) { $stream.Close() }
}
""".strip(),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-TextBase64",
                text_base64,
                "-OutputPath",
                str(output_path.resolve()),
                "-VoiceName",
                config.voice or "",
                "-Rate",
                str(config.sapi_rate),
                "-Volume",
                str(config.sapi_volume),
            ],
            check=True,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise TTSError(f"SAPI produced an empty WAV file: {output_path}")


def _synthesize_piper(text: str, output_path: Path, config: DubConfig) -> None:
    if not config.piper_model:
        raise TTSError("Piper needs --piper-model path/to/voice.onnx")
    subprocess.run(
        [
            config.piper_cmd,
            "--model",
            str(config.piper_model),
            "--output_file",
            str(output_path),
        ],
        input=text,
        text=True,
        check=True,
    )


def _synthesize_f5tts(segment: Segment, text: str, output_path: Path, config: DubConfig) -> None:
    text = _sanitize_text_for_f5(text, config)
    if not text:
        make_silence(output_path, max(0.15, segment.duration))
        return
    # The reference transcript must describe speaker_wav, not the sentence we
    # are about to generate.  Passing spoken_text here made F5 treat a short
    # target phrase as the transcript of a much longer reference recording;
    # the resulting WAVs were often about ten seconds long regardless of the
    # requested text.  An empty ref_text is intentional: F5 transcribes and
    # caches the actual reference audio itself.
    ref_text = _sanitize_text_for_f5(segment.speaker_ref_text or "", config)

    speaker_wav = segment.speaker_wav or config.speaker_wav
    if not speaker_wav:
        raise TTSError("F5-TTS needs a speaker reference WAV.")
    if not speaker_wav.exists():
        raise TTSError(f"F5-TTS speaker reference does not exist: {speaker_wav}")

    chunks = _split_text_for_xtts(text, max_chars=300, max_words=45)
    if len(chunks) > 1:
        print(f"      F5-TTS split long segment into {len(chunks)} chunks")
        chunk_paths: list[Path] = []
        chunk_dir = output_path.parent / f"{output_path.stem}_f5_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = chunk_dir / f"{index:04d}.wav"
            _f5_to_file(chunk, chunk_path, speaker_wav, config, ref_text)
            chunk_paths.append(chunk_path)
        concat_wavs(chunk_paths, output_path)
    else:
        _f5_to_file(text, output_path, speaker_wav, config, ref_text)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise TTSError(f"F5-TTS produced an empty WAV file: {output_path}")


def _f5_to_file(
    text: str,
    output_path: Path,
    speaker_wav: Path,
    config: DubConfig,
    ref_text: str,
) -> None:
    try:
        model = _load_f5_model(config)
        prepared_ref, prepared_ref_text, corrected_speed = _prepare_f5_reference(
            speaker_wav,
            ref_text,
            config.f5_speed,
        )
        model.infer(
            ref_file=str(prepared_ref),
            ref_text=prepared_ref_text,
            gen_text=text,
            file_wave=str(output_path),
            target_rms=config.f5_target_rms,
            cross_fade_duration=config.f5_cross_fade_duration,
            cfg_strength=config.f5_cfg_strength,
            nfe_step=config.f5_nfe_step,
            speed=corrected_speed,
            remove_silence=config.f5_remove_silence,
        )
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        print(f"      F5-TTS in-process failed, trying isolated runner: {type(exc).__name__}: {exc}")
        _f5_to_file_subprocess(text, output_path, speaker_wav, config, ref_text)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise TTSError(f"F5-TTS produced an empty WAV file: {output_path}")


def _prepare_f5_reference(
    speaker_wav: Path,
    ref_text: str,
    configured_speed: float,
) -> tuple[Path, str, float]:
    """Correct F5's duration estimate when a long ref has very little speech."""
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    prepared_path, prepared_text = preprocess_ref_audio_text(str(speaker_wav), ref_text)
    prepared = Path(prepared_path)
    reference_seconds = max(0.1, probe_duration(prepared))
    words = re.findall(r"[^\W_]+", prepared_text, flags=re.UNICODE)
    letters = re.sub(r"\W+", "", prepared_text, flags=re.UNICODE)
    if not words or not letters:
        return prepared, prepared_text, configured_speed

    expected_speech_seconds = max(0.55, len(words) / 2.4, len(letters) / 15.0)
    duration_correction = max(0.75, min(7.0, reference_seconds / expected_speech_seconds))
    corrected_speed = max(0.25, configured_speed * duration_correction)
    if abs(duration_correction - 1.0) >= 0.15:
        print(
            "      F5 reference duration correction: "
            f"ref={reference_seconds:.2f}s expected_speech={expected_speech_seconds:.2f}s "
            f"speed={corrected_speed:.2f}"
        )
    return prepared, prepared_text, corrected_speed


def _load_f5_model(config: DubConfig) -> object:
    _add_f5_site_packages(config)
    try:
        from f5_tts.api import F5TTS
    except ImportError as exc:
        raise TTSError("F5-TTS is not installed in .venv-f5tts.") from exc

    ckpt_file = _resolve_f5_model_file(config.f5_ckpt_file, config.f5_hf_repo, config.f5_hf_ckpt_path, config.f5_cache_dir)
    vocab_file = _resolve_f5_model_file(config.f5_vocab_file, config.f5_hf_repo, config.f5_hf_vocab_path, config.f5_cache_dir)
    device = _resolve_f5_device(config.f5_device)
    key = (config.f5_model, str(ckpt_file), str(vocab_file), device)
    model = _F5_CACHE.get(key)
    if model is None:
        model = F5TTS(
            model=config.f5_model,
            ckpt_file=str(ckpt_file),
            vocab_file=str(vocab_file),
            device=device,
        )
        _F5_CACHE[key] = model
    return model


def _f5_to_file_subprocess(
    text: str,
    output_path: Path,
    speaker_wav: Path,
    config: DubConfig,
    ref_text: str,
) -> None:
    python_path = _resolve_f5_python(config)
    runner_path = _repo_root() / "tools" / "f5_tts_runner.py"
    if not runner_path.exists():
        raise TTSError(f"F5-TTS runner does not exist: {runner_path}")

    command = [
        str(python_path),
        str(runner_path),
        "--ref-audio",
        str(speaker_wav),
        "--text-base64",
        base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "--ref-text-base64",
        base64.b64encode(ref_text.encode("utf-8")).decode("ascii"),
        "--output",
        str(output_path),
        "--model",
        config.f5_model,
        "--hf-repo",
        config.f5_hf_repo,
        "--hf-ckpt-path",
        config.f5_hf_ckpt_path,
        "--hf-vocab-path",
        config.f5_hf_vocab_path,
        "--cache-dir",
        str(_resolve_repo_relative_path(config.f5_cache_dir)),
        "--device",
        config.f5_device,
        "--speed",
        str(config.f5_speed),
        "--nfe-step",
        str(config.f5_nfe_step),
        "--cfg-strength",
        str(config.f5_cfg_strength),
        "--target-rms",
        str(config.f5_target_rms),
        "--cross-fade-duration",
        str(config.f5_cross_fade_duration),
    ]
    if config.f5_ckpt_file is not None:
        command.extend(["--ckpt-file", str(_resolve_repo_relative_path(config.f5_ckpt_file))])
    if config.f5_vocab_file is not None:
        command.extend(["--vocab-file", str(_resolve_repo_relative_path(config.f5_vocab_file))])
    if config.f5_remove_silence:
        command.append("--remove-silence")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=config.f5_timeout_seconds,
    )
    message = _short_subprocess_output(result.stdout, result.stderr)
    if message:
        print(f"      F5-TTS runner: {message}")


def _add_f5_site_packages(config: DubConfig) -> None:
    python_path = _resolve_f5_python(config)
    site_packages = python_path.parent.parent / "Lib" / "site-packages"
    if not site_packages.exists():
        raise TTSError(f"F5-TTS site-packages does not exist: {site_packages}")
    site_packages_text = str(site_packages)
    if site_packages_text not in sys.path:
        sys.path.insert(0, site_packages_text)


def _resolve_f5_python(config: DubConfig) -> Path:
    python_path = config.f5_python or Path(".venv-f5tts") / "Scripts" / "python.exe"
    python_path = _resolve_repo_relative_path(python_path)
    if not python_path.exists():
        raise TTSError(
            "F5-TTS Python was not found. Expected .venv-f5tts\\Scripts\\python.exe. "
            "Create it with: python -m venv --system-site-packages .venv-f5tts"
        )
    return python_path


def _resolve_qwen3_python(config: DubConfig) -> Path:
    python_path = config.qwen3_python or Path(".venv-qwen3tts") / "Scripts" / "python.exe"
    python_path = _resolve_repo_relative_path(python_path)
    if not python_path.is_file():
        raise TTSError(
            "Qwen3-TTS Python was not found. Expected .venv-qwen3tts\\Scripts\\python.exe."
        )
    return python_path


def _resolve_moss_python(config: DubConfig) -> Path:
    python_path = config.moss_python or Path(".venv-moss") / "Scripts" / "python.exe"
    python_path = _resolve_repo_relative_path(python_path)
    if not python_path.is_file():
        raise TTSError(
            "MOSS Python was not found. Set LALADUB_MOSS_PYTHON to the isolated MOSS environment."
        )
    return python_path


def _synthesize_cosyvoice(segment: Segment, text: str, output_path: Path, config: DubConfig) -> None:
    text = _sanitize_text_for_xtts(text)
    if config.target_lang == "ru":
        text = _apply_f5_pronunciation_dictionary(text)
    if not text:
        make_silence(output_path, max(0.15, segment.duration))
        return

    speaker_wav = segment.speaker_wav or config.speaker_wav
    if not speaker_wav:
        raise TTSError("CosyVoice needs a speaker reference WAV.")
    if not speaker_wav.exists():
        raise TTSError(f"CosyVoice speaker reference does not exist: {speaker_wav}")

    chunks = _split_text_for_xtts(text, max_chars=260, max_words=40)
    if len(chunks) > 1:
        print(f"      CosyVoice split long segment into {len(chunks)} chunks")
        chunk_paths: list[Path] = []
        chunk_dir = output_path.parent / f"{output_path.stem}_cosyvoice_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = chunk_dir / f"{index:04d}.wav"
            _cosyvoice_to_file(chunk, chunk_path, speaker_wav, segment, config)
            chunk_paths.append(chunk_path)
        concat_wavs(chunk_paths, output_path)
    else:
        _cosyvoice_to_file(text, output_path, speaker_wav, segment, config)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise TTSError(f"CosyVoice produced an empty WAV file: {output_path}")


def _cosyvoice_to_file(
    text: str,
    output_path: Path,
    speaker_wav: Path,
    segment: Segment,
    config: DubConfig,
) -> None:
    python_path = _resolve_cosyvoice_python(config)
    runner_path = _repo_root() / "tools" / "cosyvoice_tts_runner.py"
    if not runner_path.exists():
        raise TTSError(f"CosyVoice runner does not exist: {runner_path}")

    prompt_text = _sanitize_text_for_xtts(segment.speaker_ref_text or "")
    if config.target_lang == "ru":
        prompt_text = _apply_f5_pronunciation_dictionary(prompt_text)

    command = [
        str(python_path),
        str(runner_path),
        "--repo-dir",
        str(_resolve_repo_relative_path(config.cosyvoice_repo_dir)),
        "--model-dir",
        str(_resolve_repo_relative_path(config.cosyvoice_model_dir)),
        "--model-id",
        config.cosyvoice_model_id,
        "--ref-audio",
        str(speaker_wav),
        "--text-base64",
        base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "--prompt-text-base64",
        base64.b64encode(prompt_text.encode("utf-8")).decode("ascii"),
        "--output",
        str(output_path),
        "--mode",
        config.cosyvoice_mode,
        "--instruction",
        config.cosyvoice_instruction,
        "--device",
        config.cosyvoice_device,
        "--speed",
        str(config.cosyvoice_speed),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=config.cosyvoice_timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        if _tts_output_ready(output_path):
            details = _short_subprocess_output(exc.stdout, exc.stderr)
            suffix = f": {details}" if details else ""
            print(f"      CosyVoice runner returned {exc.returncode}, but WAV is ready; keeping output{suffix}")
            return
        details = _short_subprocess_output(exc.stdout, exc.stderr)
        if details:
            raise TTSError(f"CosyVoice runner failed: {details}") from exc
        raise TTSError(f"CosyVoice runner failed with exit code {exc.returncode}.") from exc
    message = _short_subprocess_output(result.stdout, result.stderr)
    if message:
        print(f"      CosyVoice runner: {message}")

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise TTSError(f"CosyVoice produced an empty WAV file: {output_path}")


def _resolve_cosyvoice_python(config: DubConfig) -> Path:
    python_path = config.cosyvoice_python or Path(".venv-cosyvoice") / "Scripts" / "python.exe"
    python_path = _resolve_repo_relative_path(python_path)
    if not python_path.is_file():
        raise TTSError(
            "CosyVoice Python was not found. Expected .venv-cosyvoice\\Scripts\\python.exe. "
            "Create it with: python -m venv .venv-cosyvoice"
        )
    return python_path


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    if process.poll() is None:
        process.kill()
        process.wait()


def _qwen3_language(target_lang: str) -> str:
    return {
        "ru": "Russian",
        "en": "English",
        "uk": "Russian",
    }.get((target_lang or "ru").lower(), "Russian")


def _moss_language(target_lang: str) -> str:
    language = {
        "ru": "Russian",
        "en": "English",
    }.get((target_lang or "ru").lower())
    if language is None:
        raise TTSError(f"MOSS-TTS v1.5 does not support target language: {target_lang}")
    return language


def _resolve_f5_model_file(local_path: Path | None, repo: str, repo_path: str, cache_dir: Path) -> Path:
    if local_path is not None:
        path = _resolve_repo_relative_path(local_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TTSError("huggingface_hub is required for F5-TTS model downloads.") from exc

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=repo_path,
            cache_dir=str(_resolve_repo_relative_path(cache_dir)),
        )
    )


def _resolve_f5_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_repo_relative_path(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tts_output_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= 1024


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.upper() == "PYTHONHASHSEED":
            env.pop(key, None)
    env["PYTHONHASHSEED"] = "random"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra:
        env.update(extra)
    return env


def _short_subprocess_output(stdout: str | None, stderr: str | None) -> str:
    combined = " ".join(part.strip() for part in (stdout, stderr) if part and part.strip())
    combined = re.sub(r"\s+", " ", combined)
    if len(combined) > 500:
        combined = "..." + combined[-497:]
    return _console_safe_text(combined)


def _console_safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return text.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="replace")


def _synthesize_xtts(segment: Segment, text: str, output_path: Path, config: DubConfig) -> None:
    text = _sanitize_text_for_xtts(text)
    if not text:
        make_silence(output_path, max(0.15, segment.duration))
        return

    speaker_wav = segment.speaker_wav or config.speaker_wav
    if not speaker_wav:
        raise TTSError("XTTS needs a speaker reference WAV. Set --speaker-wav or enable Demucs/source reference.")
    if not speaker_wav.exists():
        raise TTSError(f"XTTS speaker reference does not exist: {speaker_wav}")

    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise TTSError("Coqui TTS is not installed. Run: python -m pip install -e .[clone]") from exc

    key = (config.xtts_model, config.xtts_device)
    model = _XTTS_CACHE.get(key)
    if model is None:
        use_gpu = config.xtts_device.lower() == "cuda"
        try:
            with _torch_load_legacy_checkpoint_mode():
                model = TTS(config.xtts_model, gpu=use_gpu, progress_bar=False)
        except Exception as exc:
            message = str(exc)
            if "terms of service" in message.lower() or "tos" in message.lower():
                raise TTSError(
                    "XTTS requires accepting Coqui CPML terms. "
                    "If you accept them, set COQUI_TOS_AGREED=1 and restart the bot."
                ) from exc
            raise
        _XTTS_CACHE[key] = model

    language = _xtts_language(config.target_lang)
    chunks = _split_text_for_xtts(text, max_chars=320, max_words=45)
    if len(chunks) > 1:
        print(f"      XTTS split long segment into {len(chunks)} chunks")
        chunk_paths: list[Path] = []
        chunk_dir = output_path.parent / f"{output_path.stem}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = chunk_dir / f"{index:04d}.wav"
            _xtts_to_file(model, chunk, chunk_path, speaker_wav, language, config)
            chunk_paths.append(chunk_path)
        concat_wavs(chunk_paths, output_path)
    else:
        _xtts_to_file(model, text, output_path, speaker_wav, language, config)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise TTSError(f"XTTS produced an empty WAV file: {output_path}")


def _xtts_to_file(
    model: object,
    text: str,
    output_path: Path,
    speaker_wav: Path,
    language: str,
    config: DubConfig,
) -> None:
    try:
        model.tts_to_file(
            text=text,
            speaker_wav=str(speaker_wav),
            language=language,
            speed=config.xtts_speed,
            file_path=str(output_path),
            split_sentences=True,
        )
    except (AssertionError, IndexError, RuntimeError) as exc:
        if not _is_xtts_chunk_error(exc):
            raise

        subchunks = _split_text_for_xtts(text, max_chars=140, max_words=18)
        if len(subchunks) <= 1 and len(text) > 1:
            subchunks = _hard_split_text(text, max_chars=48)
        if len(subchunks) <= 1 and len(text) > 1:
            subchunks = _hard_split_text(text, max_chars=24)
        if len(subchunks) <= 1:
            raise TTSError(
                "XTTS refused a text chunk even after conservative splitting."
            ) from exc

        print(f"      XTTS fallback split into {len(subchunks)} smaller chunks")
        chunk_dir = output_path.parent / f"{output_path.stem}_retry_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths: list[Path] = []
        for index, chunk in enumerate(subchunks, start=1):
            chunk_path = chunk_dir / f"{index:04d}.wav"
            _xtts_to_file(model, chunk, chunk_path, speaker_wav, language, config)
            chunk_paths.append(chunk_path)
        concat_wavs(chunk_paths, output_path)


def _is_xtts_chunk_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "400 tokens" in message or "index out of range in self" in message


def _sanitize_text_for_xtts(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    cleaned: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("C"):
            continue
        if ord(char) > 0xFFFF and not char.isalnum():
            continue
        cleaned.append(char)
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def _sanitize_text_for_f5(text: str, config: DubConfig) -> str:
    text = _sanitize_text_for_xtts(text)
    if not text:
        return ""
    return _apply_f5_pronunciation_dictionary(text)


def _apply_f5_pronunciation_dictionary(text: str) -> str:
    for source, replacement in _F5_PRONUNCIATION_REPLACEMENTS:
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def _split_text_for_xtts(text: str, max_chars: int = 520, max_words: int = 85) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?…。！？])\s+", text)
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    current_words = 0

    for part in sentence_parts:
        for piece in _split_oversized_piece(part, max_chars=max_chars, max_words=max_words):
            piece_words = _word_count(piece)
            candidate = f"{current} {piece}".strip() if current else piece
            if current and (len(candidate) > max_chars or current_words + piece_words > max_words):
                chunks.append(current)
                current = piece
                current_words = piece_words
            else:
                current = candidate
                current_words += piece_words

    if current:
        chunks.append(current)
    return chunks


def _split_oversized_piece(text: str, max_chars: int, max_words: int) -> list[str]:
    if len(text) <= max_chars and _word_count(text) <= max_words:
        return [text]

    soft_parts = [
        part.strip()
        for part in re.split(r"(?<=[,;:،؛，、])\s+", text)
        if part.strip()
    ]
    if len(soft_parts) > 1:
        result: list[str] = []
        for part in soft_parts:
            result.extend(_split_oversized_piece(part, max_chars=max_chars, max_words=max_words))
        return result

    words = text.split()
    if len(words) > 1:
        result = []
        current_words: list[str] = []
        current_len = 0
        for word in words:
            next_len = current_len + len(word) + (1 if current_words else 0)
            if current_words and (len(current_words) >= max_words or next_len > max_chars):
                result.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len = next_len
        if current_words:
            result.append(" ".join(current_words))
        return result

    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def _hard_split_text(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = max(text.rfind(" ", start, end), text.rfind(",", start, end), text.rfind(".", start, end))
            if split_at > start + max_chars // 3:
                end = split_at
        chunk = text[start:end].strip(" ,.;:")
        if chunk:
            chunks.append(chunk)
        start = max(end, start + 1)
    return chunks


def _word_count(text: str) -> int:
    return len(text.split()) if " " in text else max(1, len(text) // 6)


def _xtts_language(target_lang: str) -> str:
    code = target_lang.lower()
    aliases = {
        "zh-cn": "zh-cn",
        "zh": "zh-cn",
        "pt-br": "pt",
    }
    return aliases.get(code, code)


@contextlib.contextmanager
def _torch_load_legacy_checkpoint_mode() -> Iterator[None]:
    """Let older Coqui XTTS checkpoints load under PyTorch 2.6+.

    PyTorch 2.6 changed torch.load's default to weights_only=True. Coqui TTS
    0.22 expects the old default for XTTS checkpoints, so we scope the legacy
    behavior to model initialization instead of changing global process state
    permanently.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    original_load = torch.load

    def patched_load(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_load  # type: ignore[assignment]
