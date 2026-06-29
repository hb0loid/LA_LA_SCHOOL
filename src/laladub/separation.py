from __future__ import annotations

import contextlib
import json
import os
import re
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
    if provider in {"roformer", "audio-separator", "audio_separator"}:
        return _separate_roformer(audio_path, output_dir, config)
    raise SeparationError(f"Unknown separation provider: {config.separation}")


def separation_model_label(config: DubConfig) -> str:
    provider = config.separation.lower()
    model = (
        config.audio_separator_model
        if provider in {"roformer", "audio-separator", "audio_separator"}
        else config.demucs_model
    )
    label = re.sub(r"[^0-9A-Za-z._-]+", "_", Path(model).stem).strip("._-")
    return label or ("roformer" if provider != "demucs" else "demucs")


def _separate_roformer(audio_path: Path, output_dir: Path, config: DubConfig) -> SeparationResult:
    python_path = config.audio_separator_python or Path(".venv-separator") / "Scripts" / "python.exe"
    if not python_path.is_absolute():
        python_path = _repo_root() / python_path
    runner_path = _repo_root() / "tools" / "audio_separator_runner.py"
    model_dir = config.audio_separator_model_dir
    if not model_dir.is_absolute():
        model_dir = _repo_root() / model_dir

    stem_dir = output_dir / separation_model_label(config) / audio_path.stem
    raw_output_dir = stem_dir / "raw"
    vocals_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"
    manifest_path = stem_dir / "audio_separator_manifest.json"
    stem_dir.mkdir(parents=True, exist_ok=True)

    if not python_path.is_file() or not runner_path.is_file():
        reason = f"runner missing: python={python_path.is_file()} script={runner_path.is_file()}"
        print(f"      BS-RoFormer unavailable ({reason}); falling back to Demucs")
        return _roformer_fallback_to_demucs(audio_path, output_dir, stem_dir, config)

    manifest = {
        "input": str(audio_path.resolve()),
        "output_dir": str(raw_output_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "model": config.audio_separator_model,
        "vocals_output": str(vocals_path.resolve()),
        "instrumental_output": str(instrumental_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [str(python_path), str(runner_path), "--manifest", str(manifest_path)]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_python_subprocess_env(),
        )
    except OSError as exc:
        print(f"      BS-RoFormer could not start ({exc}); falling back to Demucs")
        return _roformer_fallback_to_demucs(audio_path, output_dir, stem_dir, config)
    try:
        output, _ = process.communicate(timeout=max(30, config.audio_separator_timeout_seconds))
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        print(f"      BS-RoFormer timed out after {config.audio_separator_timeout_seconds}s; falling back to Demucs")
        return _roformer_fallback_to_demucs(audio_path, output_dir, stem_dir, config)

    if process.returncode != 0 or not _valid_stem(vocals_path) or not _valid_stem(instrumental_path):
        detail = _short_subprocess_output(output, None)
        suffix = f": {detail}" if detail else ""
        print(f"      BS-RoFormer failed for {audio_path}{suffix}; falling back to Demucs")
        return _roformer_fallback_to_demucs(audio_path, output_dir, stem_dir, config)

    shutil.rmtree(raw_output_dir, ignore_errors=True)
    print(f"      BS-RoFormer separation ready: {stem_dir}")
    return SeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)


def _roformer_fallback_to_demucs(
    audio_path: Path,
    output_dir: Path,
    stem_dir: Path,
    config: DubConfig,
) -> SeparationResult:
    shutil.rmtree(stem_dir / "raw", ignore_errors=True)
    demucs_result = _separate_demucs(audio_path, output_dir, config)
    stem_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"
    shutil.copy2(demucs_result.vocals_path, vocals_path)
    shutil.copy2(demucs_result.instrumental_path, instrumental_path)
    return SeparationResult(vocals_path=vocals_path, instrumental_path=instrumental_path)


def _valid_stem(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= 1024


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
