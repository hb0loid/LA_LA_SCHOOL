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
    if provider in {"bsroformer", "bs-roformer", "bs_roformer"}:
        return _separate_bsroformer(audio_path, output_dir, config)
    raise SeparationError(f"Unknown separation provider: {config.separation}")


def _resolve_separation_device(requested: str) -> str:
    if requested.strip().lower() != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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
        _resolve_separation_device(config.separation_device),
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_bsroformer_python(config: DubConfig) -> Path:
    python_path = config.bsroformer_python or Path(".venv-bsroformer/Scripts/python.exe")
    if not python_path.is_absolute():
        python_path = _repo_root() / python_path
    if not python_path.is_file():
        raise SeparationError(
            f"BS-Roformer python does not exist: {python_path}. "
            "Run: python -m venv .venv-bsroformer && "
            ".venv-bsroformer\\Scripts\\python.exe -m pip install audio-separator[gpu]"
        )
    return python_path


def _separate_bsroformer(audio_path: Path, output_dir: Path, config: DubConfig) -> SeparationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    python_path = _resolve_bsroformer_python(config)
    runner_path = _repo_root() / "tools" / "bsroformer_separator_runner.py"
    if not runner_path.is_file():
        raise SeparationError(f"BS-Roformer runner does not exist: {runner_path}")

    model_dir = config.bsroformer_model_dir
    if not model_dir.is_absolute():
        model_dir = _repo_root() / model_dir

    cmd = [
        str(python_path),
        str(runner_path),
        "--audio",
        str(audio_path),
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(model_dir),
        "--model-file",
        config.bsroformer_model_file,
        "--device",
        config.separation_device,
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            env=_python_subprocess_env(),
            timeout=max(60, int(config.bsroformer_timeout_seconds)),
        )
    except FileNotFoundError as exc:
        raise SeparationError("BS-Roformer python/runner not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise SeparationError(f"BS-Roformer timed out for {audio_path}") from exc
    except subprocess.CalledProcessError as exc:
        detail = _short_subprocess_output(exc.stdout, exc.stderr)
        suffix = f": {detail}" if detail else ""
        raise SeparationError(f"BS-Roformer failed for {audio_path}{suffix}") from exc

    vocals_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "no_vocals.wav"
    if not vocals_path.exists() or not instrumental_path.exists():
        detail = _short_subprocess_output(completed.stdout, completed.stderr)
        raise SeparationError(f"BS-Roformer output not found in {output_dir}: {detail}")
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
