from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SeparationError("Demucs is not installed. Run: python -m pip install -e .[separation]") from exc
    except subprocess.CalledProcessError as exc:
        raise SeparationError(f"Demucs failed for {audio_path}") from exc

    stem_dir = output_dir / config.demucs_model / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"
    if not vocals_path.exists() or not instrumental_path.exists():
        raise SeparationError(f"Demucs output not found in {stem_dir}")
    return SeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)
