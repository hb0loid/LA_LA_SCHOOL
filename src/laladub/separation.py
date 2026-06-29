from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg import require_tool, run
from .models import DubConfig


class SeparationError(RuntimeError):
    pass


@dataclass(slots=True)
class SeparationResult:
    vocals_path: Path
    instrumental_path: Path


def separate_audio(audio_path: Path, output_dir: Path, config: DubConfig) -> SeparationResult | None:
    provider = config.separation.lower()
    if provider == "none":
        return None
    if provider == "demucs":
        return _separate_demucs(audio_path, output_dir, config)
    raise SeparationError(f"Unknown separation provider: {config.separation}")


def _separate_demucs(audio_path: Path, output_dir: Path, config: DubConfig) -> SeparationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        config.demucs_model,
        "-d",
        config.separation_device,
        "-o",
        str(output_dir),
        str(audio_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            env=_python_subprocess_env(),
        )
    except FileNotFoundError as exc:
        print(f"      Demucs is not installed; using fallback separation: {exc}")
        return _fallback_separation(audio_path, output_dir, config, "demucs missing")
    except subprocess.CalledProcessError as exc:
        detail = _short_subprocess_output(exc.stdout, exc.stderr)
        suffix = f": {detail}" if detail else ""
        print(f"      Demucs failed for {audio_path}{suffix}")
        return _fallback_separation(audio_path, output_dir, config, "demucs failed")

    stem_dir = output_dir / config.demucs_model / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"
    if not vocals_path.exists() or not instrumental_path.exists():
        print(f"      Demucs output not found in {stem_dir}; using fallback separation")
        return _fallback_separation(audio_path, output_dir, config, "demucs output missing")
    return SeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)


def _fallback_separation(
    audio_path: Path,
    output_dir: Path,
    config: DubConfig,
    reason: str,
) -> SeparationResult:
    stem_dir = output_dir / config.demucs_model / audio_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"

    try:
        shutil.copy2(audio_path, vocals_path)
        _make_attenuated_bed(audio_path, instrumental_path, volume=0.20)
    except Exception as exc:
        raise SeparationError(f"Fallback separation failed for {audio_path}: {reason}: {exc}") from exc

    print(f"      Fallback separation ready ({reason}): {stem_dir}")
    return SeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)


def _make_attenuated_bed(input_path: Path, output_path: Path, volume: float) -> None:
    ffmpeg = require_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-af",
            f"volume={volume:.3f}",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def _python_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if not _valid_python_hash_seed(env.get("PYTHONHASHSEED")):
        env["PYTHONHASHSEED"] = "random"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _valid_python_hash_seed(value: str | None) -> bool:
    if value is None or value == "random":
        return True
    try:
        number = int(value)
    except ValueError:
        return False
    return 0 <= number <= 4294967295


def _short_subprocess_output(stdout: str | None, stderr: str | None) -> str:
    combined = "\n".join(part.strip() for part in (stdout, stderr) if part and part.strip())
    if not combined:
        return ""
    combined = "\n".join(line.strip() for line in combined.splitlines() if line.strip())
    if len(combined) > 1400:
        return "..." + combined[-1397:]
    return combined
