from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from laladub.ffmpeg import combine_video_audio, delayed_mix, fit_wav_to_duration, probe_duration
from laladub.models import DubConfig, Segment
from laladub.srt import read_srt
from laladub.tts import (
    synthesize_cosyvoice_batch,
    synthesize_qwen3_batch,
    synthesize_segment,
)


PROVIDERS = ("cosyvoice", "f5", "qwen3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LaLaDub voice-cloning TTS engines on one finished job."
    )
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    return parser.parse_args()


def build_segments(job_dir: Path) -> tuple[list[Segment], Path]:
    translated_path = job_dir / "work" / "translated.srt"
    cosy_manifest_path = job_dir / "work" / "cosyvoice_batch_manifest.json"
    if not translated_path.is_file():
        raise FileNotFoundError(translated_path)
    if not cosy_manifest_path.is_file():
        raise FileNotFoundError(cosy_manifest_path)

    timing_segments = read_srt(translated_path, translated=True)
    manifest = json.loads(cosy_manifest_path.read_text(encoding="utf-8"))
    items = list(manifest.get("items") or [])
    if len(timing_segments) != len(items):
        raise RuntimeError(
            f"Segment mismatch: translated={len(timing_segments)}, TTS manifest={len(items)}"
        )

    result: list[Segment] = []
    for timing, item in zip(timing_segments, items, strict=True):
        reference = Path(str(item["reference"])).resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
        text = str(item.get("text") or "").strip()
        if not text:
            raise RuntimeError("The control TTS text contains an empty segment.")
        result.append(
            Segment(
                start=timing.start,
                end=timing.end,
                text=text,
                translated_text=text,
                speaker_wav=reference,
                speaker_ref_text=str(item.get("prompt_text") or ""),
            )
        )
    return result, translated_path


def build_config(provider: str, workdir: Path, output: Path) -> DubConfig:
    return DubConfig(
        output=output,
        workdir=workdir,
        target_lang="ru",
        tts=provider,
        f5_python=REPO_ROOT / ".venv-f5tts" / "Scripts" / "python.exe",
        f5_cache_dir=REPO_ROOT / "models" / "f5tts",
        qwen3_python=REPO_ROOT / ".venv-qwen3tts" / "Scripts" / "python.exe",
        qwen3_cache_dir=REPO_ROOT / "models" / "qwen3tts",
        cosyvoice_python=REPO_ROOT / ".venv-cosyvoice" / "Scripts" / "python.exe",
        cosyvoice_repo_dir=REPO_ROOT / "models" / "cosyvoice" / "CosyVoice",
        cosyvoice_model_dir=(
            REPO_ROOT / "models" / "cosyvoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B"
        ),
        cosyvoice_model_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        cosyvoice_mode="cross_lingual",
        cosyvoice_device="auto",
        fit_to_segments=True,
        resume=False,
    )


def synthesize(provider: str, segments: list[Segment], config: DubConfig) -> float:
    raw_dir = config.workdir / "tts_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    items = [(index, segment, raw_dir / f"{index:05d}.wav") for index, segment in enumerate(segments, 1)]
    started = time.perf_counter()
    if provider == "cosyvoice":
        synthesize_cosyvoice_batch(items, config)
    elif provider == "qwen3":
        synthesize_qwen3_batch(items, config)
    elif provider == "f5":
        for _index, segment, output_path in items:
            synthesize_segment(segment, output_path, config)
    else:
        raise ValueError(provider)
    return time.perf_counter() - started


def assemble(
    video_path: Path,
    bed_path: Path | None,
    segments: list[Segment],
    config: DubConfig,
) -> float:
    started = time.perf_counter()
    fit_dir = config.workdir / "tts_fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    mix_items: list[tuple[Path, int]] = []
    for index, segment in enumerate(segments, 1):
        raw_path = config.workdir / "tts_raw" / f"{index:05d}.wav"
        fit_path = fit_dir / f"{index:05d}.wav"
        fit_wav_to_duration(raw_path, fit_path, max(0.1, segment.duration))
        mix_items.append((fit_path, int(segment.start * 1000)))

    dub_track = config.workdir / "dub_track.wav"
    delayed_mix(mix_items, probe_duration(video_path), dub_track, config.workdir)
    combine_video_audio(
        video_path=video_path,
        dub_path=dub_track,
        output_path=config.output,
        original_volume=0.35,
        dub_volume=1.0,
        bed_path=bed_path,
    )
    return time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if not os.environ.get("PYTHONHASHSEED", "").strip():
        os.environ["PYTHONHASHSEED"] = "42"
    job_dir = args.job_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    segments, translated_path = build_segments(job_dir)
    video_path = (job_dir / "input.mp4").resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    bed_candidate = job_dir / "work" / "separated" / "htdemucs" / "source_mix" / "no_vocals.wav"
    bed_path = bed_candidate.resolve() if bed_candidate.is_file() else None

    shutil.copy2(translated_path, output_root / "control_text.srt")
    (output_root / "control_text.txt").write_text(
        "\n".join(segment.spoken_text for segment in segments), encoding="utf-8"
    )

    results_path = output_root / "benchmark_results.json"
    results: list[dict[str, object]] = []
    if results_path.is_file():
        try:
            previous = json.loads(results_path.read_text(encoding="utf-8"))
            selected = set(args.providers)
            results = [item for item in previous if item.get("provider") not in selected]
        except (OSError, ValueError, TypeError):
            results = []
    for provider in args.providers:
        provider_dir = output_root / provider
        if provider_dir.exists():
            shutil.rmtree(provider_dir)
        provider_dir.mkdir(parents=True)
        output = output_root / f"mrbeast_{provider}.mp4"
        output.unlink(missing_ok=True)
        config = build_config(provider, provider_dir, output)
        print(f"BENCHMARK_START\t{provider}", flush=True)
        try:
            tts_seconds = synthesize(provider, segments, config)
            assembly_seconds = assemble(video_path, bed_path, segments, config)
            result = {
                "provider": provider,
                "status": "ok",
                "segments": len(segments),
                "tts_seconds": round(tts_seconds, 3),
                "assembly_seconds": round(assembly_seconds, 3),
                "total_seconds": round(tts_seconds + assembly_seconds, 3),
                "output": str(output),
            }
        except Exception as exc:
            result = {
                "provider": provider,
                "status": "error",
                "segments": len(segments),
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"BENCHMARK_RESULT\t{json.dumps(result, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
