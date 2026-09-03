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
    synthesize_chatterbox_batch,
    synthesize_cosyvoice_batch,
    synthesize_fish_batch,
    synthesize_moss_batch,
    synthesize_qwen3_batch,
    synthesize_segment,
)


PROVIDERS = ("cosyvoice", "f5", "qwen3", "moss", "chatterbox", "fish")

# Any engine's batch manifest carries the same per-segment text and speaker
# reference, so a job dubbed with one engine can drive a run of another.
MANIFEST_NAMES = (
    "cosyvoice_batch_manifest.json",
    "moss_batch_manifest.json",
    "qwen3_batch_manifest.json",
    "chatterbox_batch_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LaLaDub voice-cloning TTS engines on one finished job."
    )
    parser.add_argument("--job-dir", type=Path, help="Job folder holding work/ and input.mp4.")
    parser.add_argument("--work-dir", type=Path, help="Pipeline workdir, when it is not <job>/work.")
    parser.add_argument("--video", type=Path, help="Source video, when it is not <job>/input.mp4.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    args = parser.parse_args()
    if args.job_dir is None and args.work_dir is None:
        parser.error("pass --job-dir or --work-dir")
    return args


def find_manifest(work_dir: Path) -> Path:
    for name in MANIFEST_NAMES:
        candidate = work_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No TTS batch manifest in {work_dir}")


def build_segments(work_dir: Path) -> tuple[list[Segment], Path]:
    translated_path = work_dir / "translated.srt"
    if not translated_path.is_file():
        raise FileNotFoundError(translated_path)
    manifest_path = find_manifest(work_dir)

    timing_segments = read_srt(translated_path, translated=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        moss_python=Path(r"F:\LaLaSchoolData\tts-lab\MOSS-TTS\.venv\Scripts\python.exe"),
        moss_model_dir=Path(r"F:\LaLaSchoolData\tts-lab\models\MOSS-TTS-Local-Transformer-v1.5"),
        moss_codec_dir=Path(r"F:\LaLaSchoolData\tts-lab\models\MOSS-Audio-Tokenizer-v2"),
        moss_device="auto",
        chatterbox_python=REPO_ROOT / ".venv-chatterbox" / "Scripts" / "python.exe",
        chatterbox_cache_dir=REPO_ROOT / "models" / "chatterbox",
        chatterbox_device="auto",
        fish_python=Path(r"F:\LaLaSchoolData\tts-lab\fish-speech\.venv\Scripts\python.exe"),
        fish_repo_dir=Path(r"F:\LaLaSchoolData\tts-lab\fish-speech"),
        fish_model_dir=Path(r"F:\LaLaSchoolData\tts-lab\models\fish-s2-pro"),
        fish_device="auto",
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
    elif provider == "moss":
        synthesize_moss_batch(items, config)
    elif provider == "chatterbox":
        synthesize_chatterbox_batch(items, config)
    elif provider == "fish":
        synthesize_fish_batch(items, config)
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
    job_dir = args.job_dir.resolve() if args.job_dir else None
    work_dir = (args.work_dir or (job_dir / "work")).resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    segments, translated_path = build_segments(work_dir)
    video_path = (args.video or (job_dir / "input.mp4")).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    bed_path = next(
        (path.resolve() for path in sorted(work_dir.rglob("no_vocals.wav")) if path.is_file()),
        None,
    )

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
