from __future__ import annotations

import os
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
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            env=_python_subprocess_env(),
        )
    except FileNotFoundError as exc:
        raise SeparationError("Demucs is not installed. Run: python -m pip install -e .[separation]") from exc
    except subprocess.CalledProcessError as exc:
        detail = _short_subprocess_output(exc.stdout, exc.stderr)
        suffix = f": {detail}" if detail else ""
        raise SeparationError(f"Demucs failed for {audio_path}{suffix}") from exc

    stem_dir = output_dir / config.demucs_model / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"
    if not vocals_path.exists() or not instrumental_path.exists():
        raise SeparationError(f"Demucs output not found in {stem_dir}")
    return SeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)


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
