from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path


class ToolError(RuntimeError):
    pass


def which(name: str) -> str | None:
    return shutil.which(name)


def require_tool(name: str) -> str:
    path = which(name)
    if not path:
        raise ToolError(f"Required tool not found on PATH: {name}")
    return path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float:
    ffprobe = require_tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    duration = data.get("format", {}).get("duration")
    if duration is not None:
        return float(duration)

    for stream in data.get("streams", []):
        duration = stream.get("duration")
        if duration is not None:
            return float(duration)

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate:
                return frames / float(rate)

    raise ToolError(f"Could not determine media duration: {path}")


def extract_audio(video_path: Path, wav_path: Path) -> None:
    ffmpeg = require_tool("ffmpeg")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )


def extract_audio_track(video_path: Path, wav_path: Path, sample_rate: int = 44100, channels: int = 2) -> None:
    ffmpeg = require_tool("ffmpeg")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )


def make_silence(wav_path: Path, duration: float, sample_rate: int = 44100) -> None:
    ffmpeg = require_tool("ffmpeg")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",
            "-t",
            f"{max(0.05, duration):.3f}",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )


def normalize_wav(input_path: Path, output_path: Path, sample_rate: int = 44100) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def prepare_voice_reference(input_path: Path, output_path: Path, sample_rate: int = 24000) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            "highpass=f=70,lowpass=f=7600,afftdn=nf=-25,dynaudnorm=f=75:g=15,volume=1.15",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def make_whisper_chaos_audio(input_path: Path, output_path: Path, bass_gain_db: float = 50.0) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            f"bass=g={bass_gain_db:.1f}:f=120:w=1,volume=1.8,alimiter=limit=0.98",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def extract_wav_slice(
    input_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    sample_rate: int = 22050,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.15, duration):.3f}",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def concat_wavs(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise ToolError("concat_wavs needs at least one input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        shutil.copy2(input_paths[0], output_path)
        return

    ffmpeg = require_tool("ffmpeg")
    cmd = [ffmpeg, "-y"]
    for input_path in input_paths:
        cmd.extend(["-i", str(input_path)])

    labels = "".join(f"[{index}:a]" for index in range(len(input_paths)))
    filter_value = f"{labels}concat=n={len(input_paths)}:v=0:a=1[out]"
    cmd.extend(["-filter_complex", filter_value, "-map", "[out]", "-c:a", "pcm_s16le", str(output_path)])
    run(cmd)


def _atempo_chain(factor: float) -> str:
    factor = max(0.25, min(8.0, factor))
    parts: list[float] = []
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    parts.append(factor)
    return ",".join(f"atempo={part:.4f}" for part in parts)


def fit_wav_to_duration(
    input_path: Path,
    output_path: Path,
    target_duration: float,
    *,
    min_tempo: float = 0.92,
    max_tempo: float = 2.15,
) -> None:
    source_duration = probe_duration(input_path)
    if target_duration <= 0.05 or source_duration <= 0.05:
        normalize_wav(input_path, output_path)
        return

    tempo = source_duration / target_duration
    if tempo < min_tempo:
        normalize_wav(input_path, output_path)
        return
    if min_tempo <= tempo <= 1.08:
        normalize_wav(input_path, output_path)
        return

    ffmpeg = require_tool("ffmpeg")
    filter_value = _atempo_chain(min(tempo, max_tempo))
    if tempo > max_tempo:
        filter_value = f"{filter_value},atrim=0:{target_duration:.3f},asetpts=N/SR/TB"

    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-filter:a",
            filter_value,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def delayed_mix(
    items: list[tuple[Path, int]],
    duration: float,
    output_path: Path,
    temp_dir: Path,
    batch_size: int = 60,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not items:
        make_silence(output_path, duration)
        return

    if len(items) > batch_size:
        chunk_paths: list[Path] = []
        for idx in range(0, len(items), batch_size):
            chunk = items[idx : idx + batch_size]
            chunk_path = temp_dir / f"mix_chunk_{idx // batch_size:04d}.wav"
            delayed_mix(chunk, duration, chunk_path, temp_dir, batch_size=batch_size)
            chunk_paths.append(chunk_path)
        delayed_mix([(path, 0) for path in chunk_paths], duration, output_path, temp_dir, batch_size=batch_size)
        return

    cmd = [ffmpeg, "-y"]
    for audio_path, _delay_ms in items:
        cmd.extend(["-i", str(audio_path)])

    filter_parts: list[str] = []
    labels: list[str] = []
    for idx, (_audio_path, delay_ms) in enumerate(items):
        label = f"a{idx}"
        filter_parts.append(f"[{idx}:a]adelay={max(0, delay_ms)}:all=1[{label}]")
        labels.append(f"[{label}]")

    joined_labels = "".join(labels)
    filter_parts.append(
        f"{joined_labels}amix=inputs={len(items)}:duration=longest:dropout_transition=0:normalize=0,"
        f"atrim=0:{duration:.3f},asetpts=N/SR/TB[mix]"
    )
    cmd.extend(["-filter_complex", ";".join(filter_parts), "-map", "[mix]", "-c:a", "pcm_s16le", str(output_path)])
    run(cmd)


def combine_video_audio(
    video_path: Path,
    dub_path: Path,
    output_path: Path,
    original_volume: float,
    dub_volume: float,
    bed_path: Path | None = None,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if bed_path is not None and original_volume > 0:
        filter_complex = (
            f"[1:a]volume={original_volume:.3f}[bed];"
            f"[2:a]volume={dub_volume:.3f}[dub];"
            "[bed][dub]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(bed_path),
            "-i",
            str(dub_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    elif original_volume > 0:
        filter_complex = (
            f"[0:a]volume={original_volume:.3f}[orig];"
            f"[1:a]volume={dub_volume:.3f}[dub];"
            "[orig][dub]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dub_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dub_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    run(cmd)
