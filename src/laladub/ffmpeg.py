from __future__ import annotations

import json
import random
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


def probe_video_dimensions(path: Path) -> tuple[int, int] | None:
    """Frame size of the first video stream, or None when it cannot be read."""
    ffprobe = require_tool("ffprobe")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams") or []
    except Exception:
        return None
    if not streams:
        return None
    width = streams[0].get("width")
    height = streams[0].get("height")
    if not width or not height:
        return None
    return int(width), int(height)


def has_video_stream(path: Path) -> bool:
    ffprobe = require_tool("ffprobe")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return "video" in result.stdout.lower()


def make_audio_visual_video(
    audio_path: Path,
    output_path: Path,
    *,
    source_roots: list[Path],
    temp_dir: Path,
    duration: float | None = None,
    resolution: int = 512,
    min_slice_seconds: float = 0.2,
    max_slice_seconds: float = 3.0,
    min_speed: float = 1.0,
    max_speed: float = 4.0,
    exclude_dirs: list[Path] | None = None,
    safety_enabled: bool = False,
    safety_cache_dir: Path | None = None,
    safety_model: str = "Falconsai/nsfw_image_detection",
    safety_threshold: float = 0.72,
    safety_frames: int = 5,
    safety_device: str = "cpu",
) -> None:
    target_duration = duration if duration is not None else probe_duration(audio_path)
    target_duration = max(0.1, target_duration)
    resolution = max(64, int(resolution))
    temp_dir.mkdir(parents=True, exist_ok=True)

    sources = _find_visual_source_videos(source_roots, exclude_dirs=exclude_dirs or [])
    if not sources:
        _make_fallback_audio_video(audio_path, output_path, target_duration, resolution)
        return

    safety_cache: dict[Path, bool] = {}
    segment_paths: list[Path] = []
    visual_duration = 0.0
    attempts = 0
    max_attempts = max(30, min(800, int(target_duration / max(min_slice_seconds, 0.1) * 3)))
    while visual_duration < target_duration + 1.0 and attempts < max_attempts:
        attempts += 1
        if not sources:
            break
        source = random.choice(sources)
        if safety_enabled and not _visual_source_is_safe(
            source,
            safety_cache=safety_cache,
            safety_cache_dir=safety_cache_dir,
            temp_dir=temp_dir / "safety_frames",
            model_name=safety_model,
            threshold=safety_threshold,
            frame_count=safety_frames,
            device=safety_device,
        ):
            sources.remove(source)
            continue
        try:
            source_duration = probe_duration(source)
        except Exception:
            continue
        if source_duration < 0.3:
            continue

        out_duration = min(
            random.uniform(min_slice_seconds, max_slice_seconds),
            max(0.2, target_duration + 1.0 - visual_duration),
        )
        speed = random.uniform(min_speed, max_speed)
        input_duration = min(max(0.2, out_duration * speed), max(0.2, source_duration))
        max_start = max(0.0, source_duration - input_duration)
        start = random.uniform(0.0, max_start) if max_start > 0.05 else 0.0
        segment_path = temp_dir / f"visual_segment_{len(segment_paths):04d}.mp4"
        try:
            _make_visual_segment(source, segment_path, start, input_duration, speed, resolution)
        except subprocess.CalledProcessError:
            segment_path.unlink(missing_ok=True)
            continue
        if segment_path.exists() and segment_path.stat().st_size > 1024:
            try:
                actual_duration = probe_duration(segment_path)
            except Exception:
                segment_path.unlink(missing_ok=True)
                continue
            if actual_duration < 0.1:
                segment_path.unlink(missing_ok=True)
                continue
            segment_paths.append(segment_path)
            # A short source clip can yield less video than the requested
            # out_duration after speed-up.  Counting the requested value made
            # the loop stop early and players held the final frame until the
            # longer audio track ended.
            visual_duration += actual_duration

    if not segment_paths:
        _make_fallback_audio_video(audio_path, output_path, target_duration, resolution)
        return

    _concat_visual_segments_with_audio(segment_paths, audio_path, output_path, target_duration)


def _visual_source_is_safe(
    path: Path,
    *,
    safety_cache: dict[Path, bool],
    safety_cache_dir: Path | None,
    temp_dir: Path,
    model_name: str,
    threshold: float,
    frame_count: int,
    device: str,
) -> bool:
    resolved = path.resolve()
    cached = safety_cache.get(resolved)
    if cached is not None:
        return cached
    if safety_cache_dir is None:
        safety_cache[resolved] = False
        return False
    from .visual_safety import is_video_safe_for_visual_source

    safe = is_video_safe_for_visual_source(
        path,
        cache_dir=safety_cache_dir,
        temp_dir=temp_dir,
        model_name=model_name,
        threshold=threshold,
        frame_count=frame_count,
        device=device,
    )
    safety_cache[resolved] = safe
    return safe


def _find_visual_source_videos(
    source_roots: list[Path],
    *,
    exclude_dirs: list[Path],
    max_candidates: int = 120,
) -> list[Path]:
    suffixes = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    excluded = [path.resolve() for path in exclude_dirs if path.exists()]
    raw_candidates: list[Path] = []
    seen: set[Path] = set()
    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            resolved = path.resolve()
            if resolved in seen or any(_is_relative_to(resolved, directory) for directory in excluded):
                continue
            if path.name.lower().startswith("input_audio_visual"):
                continue
            seen.add(resolved)
            raw_candidates.append(path)

    random.shuffle(raw_candidates)
    raw_candidates.sort(key=_visual_source_priority)
    # The actual segment extraction below already rejects unreadable inputs.
    # Avoid launching ffprobe once per candidate here; trusted libraries can
    # contain thousands of clips and the old pre-scan delayed every audio job.
    return raw_candidates[:max_candidates]


def _visual_source_priority(path: Path) -> int:
    name = path.name.lower()
    if name.startswith("input") and "audio_visual" not in name:
        return 0
    if name == "dubbed.mp4":
        return 1
    if "watermarked" in name:
        return 3
    return 2


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _make_visual_segment(
    source_path: Path,
    output_path: Path,
    start: float,
    input_duration: float,
    speed: float,
    resolution: int,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = (
        f"setpts=PTS/{max(0.1, speed):.4f},"
        f"scale={resolution}:{resolution}:force_original_aspect_ratio=increase,"
        f"crop={resolution}:{resolution},fps=30,setsar=1,format=yuv420p,"
        "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.1, input_duration):.3f}",
            "-i",
            str(source_path),
            "-an",
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def _concat_visual_segments_with_audio(
    segment_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    duration: float,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_name(f"{output_path.stem}_segments.txt")
    list_path.write_text(
        "".join(f"file '{_concat_file_path(path)}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-i",
            str(audio_path),
            "-t",
            f"{max(0.1, duration):.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def _concat_file_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _make_fallback_audio_video(audio_path: Path, output_path: Path, duration: float, resolution: int) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={resolution}x{resolution}:rate=30",
            "-i",
            str(audio_path),
            "-t",
            f"{max(0.1, duration):.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


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


def trim_video(input_path: Path, output_path: Path, duration: float) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-t",
            f"{max(0.1, duration):.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    )


def compress_video_for_telegram(
    input_path: Path,
    output_path: Path,
    *,
    target_size_mb: float = 45.0,
    max_width: int = 1280,
    audio_bitrate_k: int = 96,
) -> None:
    duration = max(1.0, probe_duration(input_path))
    total_kbit_budget = target_size_mb * 8192 * 0.90
    video_bitrate_k = max(320, int(total_kbit_budget / duration - audio_bitrate_k))
    maxrate_k = max(video_bitrate_k, int(video_bitrate_k * 1.35))
    bufsize_k = maxrate_k * 2
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"scale=w='if(gt(iw,{max_width}),{max_width},iw)':h=-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            f"{video_bitrate_k}k",
            "-maxrate",
            f"{maxrate_k}k",
            "-bufsize",
            f"{bufsize_k}k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            f"{audio_bitrate_k}k",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output_path),
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
            # Preserve speaker timbre for cloning. Aggressive FFT denoising
            # and dynamic normalization made Demucs/reference artifacts more
            # prominent and could turn one voice into a metallic average.
            "highpass=f=65,lowpass=f=10000,volume=1.05,alimiter=limit=0.95",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def make_whisper_chaos_audio(input_path: Path, output_path: Path, gain_db: float = 50.0) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # This path is intentionally not "clean" audio. It is a hard ASR helper for
    # quiet/uneven meme clips where Whisper misses speech under loud effects.
    # The old version was just volume+limiter; this aggressively flattens voice
    # dynamics before limiting.
    voice_crush_filter = (
        "highpass=f=70,"
        "lowpass=f=7800,"
        "compand=attacks=0.002:decays=0.04:"
        "points=-90/-35|-60/-14|-35/-4|-12/0|0/0,"
        "acompressor=threshold=-35dB:ratio=20:attack=1:release=50:makeup=16,"
        "volume=6dB,"
        "alimiter=limit=1.0"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            voice_crush_filter,
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
    effective_duration = max(0.15, duration)
    # A hard -ss/-t cut lands mid-waveform on both edges (confirmed: real
    # slices start/end tens of percent off zero amplitude), which is an
    # audible click on every reference clip and artifact chunk. A short
    # fade removes it without touching alignment or duration.
    fade_seconds = min(0.015, effective_duration / 4)
    fade_out_start = max(0.0, effective_duration - fade_seconds)
    audio_filter = f"afade=t=in:st=0:d={fade_seconds:.4f},afade=t=out:st={fade_out_start:.4f}:d={fade_seconds:.4f}"
    run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{effective_duration:.3f}",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            audio_filter,
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


def _run_mux_with_video_fallback(cmd: list[str]) -> None:
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        fallback = _video_reencode_fallback_cmd(cmd)
        if fallback is None:
            raise
        run(fallback)


def _video_reencode_fallback_cmd(cmd: list[str]) -> list[str] | None:
    try:
        codec_index = cmd.index("-c:v")
    except ValueError:
        return None
    if codec_index + 1 >= len(cmd) or cmd[codec_index + 1] != "copy":
        return None

    fallback = list(cmd)
    fallback[codec_index + 1 : codec_index + 2] = [
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
    ]
    if "-movflags" not in fallback:
        output_path = fallback.pop()
        fallback.extend(["-movflags", "+faststart", output_path])
    return fallback


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
    items: list[tuple[Path, int] | tuple[Path, int, float, float | None]],
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
    normalized_items: list[tuple[Path, int, float, float | None]] = []
    for item in items:
        if len(item) == 2:
            audio_path, delay_ms = item
            normalized_items.append((audio_path, delay_ms, 1.0, None))
        else:
            audio_path, delay_ms, gain, duck_after_seconds = item
            normalized_items.append((audio_path, delay_ms, gain, duck_after_seconds))

    for audio_path, _delay_ms, _gain, _duck_after_seconds in normalized_items:
        cmd.extend(["-i", str(audio_path)])

    filter_parts: list[str] = []
    labels: list[str] = []
    for idx, (_audio_path, delay_ms, gain, duck_after_seconds) in enumerate(normalized_items):
        label = f"a{idx}"
        filters = [f"volume={max(0.0, gain):.3f}"]
        if duck_after_seconds is not None:
            # Keep the full phrase but move its tail behind the newly started
            # foreground line. The expression runs before adelay, so t is
            # relative to this individual clip.
            duck_at = max(0.0, duck_after_seconds)
            duck_done = duck_at + 0.18
            filters.append(
                "volume='if(lt(t,"
                f"{duck_at:.3f}),1,if(lt(t,{duck_done:.3f}),"
                f"1-(t-{duck_at:.3f})*3.8889,0.30))':eval=frame"
            )
        filters.append(f"adelay={max(0, delay_ms)}:all=1")
        filter_parts.append(f"[{idx}:a]{','.join(filters)}[{label}]")
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
            "[bed][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "alimiter=limit=0.95[aout]"
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
            "[orig][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "alimiter=limit=0.95[aout]"
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
    _run_mux_with_video_fallback(cmd)


# ISO 639-2/B tags for the language codes this project's UI exposes. Falls
# back to "und" (undefined) so an unmapped code never breaks the mux.
_MP4_LANGUAGE_TAGS = {
    "ru": "rus",
    "en": "eng",
    "uk": "ukr",
    "vi": "vie",
    "ko": "kor",
    "ja": "jpn",
    "zh": "zho",
    "th": "tha",
    "de": "deu",
    "es": "spa",
    "fr": "fra",
}


def mp4_language_tag(code: str | None) -> str:
    if not code:
        return "und"
    return _MP4_LANGUAGE_TAGS.get(code.strip().lower(), "und")


def combine_video_audio_multitrack(
    video_path: Path,
    original_audio_path: Path,
    dub_path: Path,
    output_path: Path,
    dub_volume: float,
    bed_path: Path | None = None,
    original_lang: str | None = None,
    dub_lang: str | None = None,
) -> None:
    """Mux video with two separate audio streams instead of pre-mixing them:
    the untouched original audio and the dub (optionally laid over the
    instrumental bed). A player can then switch tracks instead of only ever
    hearing one fixed mix.
    """
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg, "-y", "-i", str(video_path), "-i", str(original_audio_path), "-i", str(dub_path)]
    if bed_path is not None:
        cmd.extend(["-i", str(bed_path)])
        filter_complex = (
            f"[2:a]volume={dub_volume:.3f}[dubv];"
            "[3:a][dubv]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "alimiter=limit=0.95[dubout]"
        )
    else:
        filter_complex = f"[2:a]volume={dub_volume:.3f}[dubout]"

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "[dubout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-metadata:s:a:0",
            f"language={mp4_language_tag(original_lang)}",
            "-metadata:s:a:0",
            "title=Original",
            "-metadata:s:a:1",
            f"language={mp4_language_tag(dub_lang)}",
            "-metadata:s:a:1",
            "title=Dub",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            str(output_path),
        ]
    )
    _run_mux_with_video_fallback(cmd)
