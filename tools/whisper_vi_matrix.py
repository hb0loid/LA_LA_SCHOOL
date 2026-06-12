from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from laladub.asr import transcribe
from laladub.ffmpeg import extract_audio
from laladub.models import DubConfig, Segment
from laladub.pipeline import (
    _is_text_repetition_loop,
    _looks_like_meta_hallucination,
    _translate_artifact_segments,
)
from laladub.quality import collapse_repetitions_in_segments, is_repetitive_loop
from laladub.srt import write_srt, write_txt


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    backend: str
    model: str
    mode: str
    suppress_ascii: bool = False
    compute_type: str = "int8"


@dataclass
class CaseMetrics:
    case_id: str
    backend: str
    model: str
    mode: str
    status: str
    seconds: float
    raw_segments: int = 0
    clean_segments: int = 0
    translated_segments: int = 0
    meta_source_segments: int = 0
    meta_translated_segments: int = 0
    loop_text_segments: int = 0
    repetitive_loop_raw: bool = False
    repetitive_loop_clean: bool = False
    unique_clean_texts: int = 0
    sample_source: list[str] | None = None
    sample_translated: list[str] | None = None
    error: str | None = None

    @property
    def useful_artifact_score(self) -> int:
        return self.meta_source_segments + self.meta_translated_segments


CASES = [
    MatrixCase("openai_tiny_soft", "openai-whisper", "tiny", "soft"),
    MatrixCase("openai_tiny_hard", "openai-whisper", "tiny", "hard", suppress_ascii=True),
    MatrixCase("openai_base_soft", "openai-whisper", "base", "soft"),
    MatrixCase("openai_base_hard", "openai-whisper", "base", "hard", suppress_ascii=True),
    MatrixCase("openai_small_soft", "openai-whisper", "small", "soft"),
    MatrixCase("openai_small_hard", "openai-whisper", "small", "hard", suppress_ascii=True),
    MatrixCase("openai_medium_soft", "openai-whisper", "medium", "soft"),
    MatrixCase("openai_medium_hard", "openai-whisper", "medium", "hard", suppress_ascii=True),
    MatrixCase("openai_large-v1_soft", "openai-whisper", "large-v1", "soft"),
    MatrixCase("openai_large-v1_hard", "openai-whisper", "large-v1", "hard", suppress_ascii=True),
    MatrixCase("openai_large-v2_soft", "openai-whisper", "large-v2", "soft"),
    MatrixCase("openai_large-v2_hard", "openai-whisper", "large-v2", "hard", suppress_ascii=True),
    MatrixCase("openai_large-v3_soft", "openai-whisper", "large-v3", "soft"),
    MatrixCase("openai_large-v3_hard", "openai-whisper", "large-v3", "hard", suppress_ascii=True),
    MatrixCase("openai_large-v3-turbo_soft", "openai-whisper", "large-v3-turbo", "soft"),
    MatrixCase("openai_large-v3-turbo_hard", "openai-whisper", "large-v3-turbo", "hard", suppress_ascii=True),
    MatrixCase("faster_small", "faster-whisper", "small", "faster"),
    MatrixCase("faster_large-v3", "faster-whisper", "large-v3", "faster"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "runs" / "whisper_vi_matrix" / "krokodil")
    parser.add_argument("--only", default="", help="Comma-separated case ids to run.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    audio_path = outdir / "source_16k.wav"
    if not audio_path.exists():
        print(f"Extracting audio -> {audio_path}", flush=True)
        extract_audio(args.video, audio_path)

    selected = set(item.strip() for item in args.only.split(",") if item.strip())
    cases = [case for case in CASES if not selected or case.case_id in selected]

    metrics: list[CaseMetrics] = []
    for case in cases:
        case_dir = outdir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = case_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            metrics.append(CaseMetrics(**json.loads(metrics_path.read_text(encoding="utf-8"))))
            print(f"Skipping existing {case.case_id}", flush=True)
            continue

        print(f"Running {case.case_id}: backend={case.backend} model={case.model} mode={case.mode}", flush=True)
        started = time.perf_counter()
        try:
            case_metrics = run_case(audio_path, case, case_dir)
        except Exception as exc:
            case_metrics = CaseMetrics(
                case_id=case.case_id,
                backend=case.backend,
                model=case.model,
                mode=case.mode,
                status="error",
                seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"ERROR {case.case_id}: {case_metrics.error}", flush=True)

        metrics_path.write_text(
            json.dumps(asdict(case_metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics.append(case_metrics)
        print(
            "Done "
            f"{case.case_id}: status={case_metrics.status} "
            f"raw={case_metrics.raw_segments} clean={case_metrics.clean_segments} "
            f"meta={case_metrics.meta_source_segments}/{case_metrics.meta_translated_segments} "
            f"loops={case_metrics.loop_text_segments} "
            f"sec={case_metrics.seconds:.1f}",
            flush=True,
        )

    write_summary(outdir, metrics)
    print(f"Summary: {outdir / 'summary.md'}", flush=True)
    return 0


def run_case(audio_path: Path, case: MatrixCase, case_dir: Path) -> CaseMetrics:
    started = time.perf_counter()
    config = DubConfig(
        output=case_dir / "unused.mp4",
        workdir=case_dir,
        source_lang="vi",
        target_lang="ru",
        asr_backend=case.backend,
        whisper_model=case.model,
        whisper_device="cpu",
        whisper_compute_type=case.compute_type,
        translator="argos",
        tts="none",
        separation="none",
        audio_bed="original",
        glitch_profile="faithful",
        condition_on_previous_text=True,
        hallucination_silence_threshold=None,
        force_source_language=True,
        suppress_plain_ascii_tokens=case.suppress_ascii,
        collapse_repetitions=True,
    )

    raw_segments = transcribe(audio_path, config)
    write_srt(case_dir / "source_raw.srt", raw_segments, translated=False)
    write_txt(case_dir / "source_raw.txt", raw_segments, translated=False)

    loop_text_segments = sum(_is_text_repetition_loop(segment.text) for segment in raw_segments)
    clean_segments = [
        Segment(start=segment.start, end=segment.end, text=segment.text)
        for segment in raw_segments
        if segment.text.strip() and not _is_text_repetition_loop(segment.text)
    ]
    clean_segments = collapse_repetitions_in_segments(
        clean_segments,
        max_phrase_repeats=2,
        max_word_repeats=2,
        max_ngram_words=8,
    )
    write_srt(case_dir / "source_clean.srt", clean_segments, translated=False)
    write_txt(case_dir / "source_clean.txt", clean_segments, translated=False)

    translated_segments = _translate_artifact_segments(clean_segments, config)
    translated_segments = collapse_repetitions_in_segments(
        translated_segments,
        max_phrase_repeats=2,
        max_word_repeats=2,
        max_ngram_words=8,
    )
    write_srt(case_dir / "translated_ru.srt", translated_segments, translated=True)
    write_txt(case_dir / "translated_ru.txt", translated_segments, translated=True)

    meta_source_segments = sum(_looks_like_meta_hallucination(segment.text) for segment in clean_segments)
    meta_translated_segments = sum(_looks_like_meta_hallucination(segment.spoken_text) for segment in translated_segments)
    unique_clean_texts = len({" ".join(segment.text.casefold().split()) for segment in clean_segments if segment.text.strip()})

    return CaseMetrics(
        case_id=case.case_id,
        backend=case.backend,
        model=case.model,
        mode=case.mode,
        status="ok",
        seconds=time.perf_counter() - started,
        raw_segments=len(raw_segments),
        clean_segments=len(clean_segments),
        translated_segments=len(translated_segments),
        meta_source_segments=meta_source_segments,
        meta_translated_segments=meta_translated_segments,
        loop_text_segments=loop_text_segments,
        repetitive_loop_raw=is_repetitive_loop(raw_segments, min_segments=4, repeated_share=0.65),
        repetitive_loop_clean=is_repetitive_loop(clean_segments, min_segments=4, repeated_share=0.65),
        unique_clean_texts=unique_clean_texts,
        sample_source=sample_artifacts(clean_segments, translated=False),
        sample_translated=sample_artifacts(translated_segments, translated=True),
    )


def sample_artifacts(segments: list[Segment], *, translated: bool) -> list[str]:
    samples: list[str] = []
    for segment in segments:
        text = segment.spoken_text if translated else segment.text
        if _looks_like_meta_hallucination(text):
            samples.append(f"{segment.start:.2f}-{segment.end:.2f}: {text}")
        if len(samples) >= 6:
            break
    if samples:
        return samples
    for segment in segments[:6]:
        text = segment.spoken_text if translated else segment.text
        if text.strip():
            samples.append(f"{segment.start:.2f}-{segment.end:.2f}: {text}")
    return samples


def write_summary(outdir: Path, metrics: list[CaseMetrics]) -> None:
    summary_json = [asdict(item) | {"useful_artifact_score": item.useful_artifact_score} for item in metrics]
    (outdir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = sorted(
        metrics,
        key=lambda item: (
            item.status != "ok",
            -item.useful_artifact_score,
            item.loop_text_segments,
            -item.unique_clean_texts,
        ),
    )
    lines = [
        "# Whisper Vietnamese Forced Matrix",
        "",
        "| case | status | raw | clean | meta src/ru | loop segs | raw loop | unique | sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            f"| {item.case_id} | {item.status} | {item.raw_segments} | {item.clean_segments} | "
            f"{item.meta_source_segments}/{item.meta_translated_segments} | {item.loop_text_segments} | "
            f"{'yes' if item.repetitive_loop_raw else 'no'} | {item.unique_clean_texts} | {item.seconds:.1f} |"
        )
    lines.append("")
    for item in rows:
        lines.append(f"## {item.case_id}")
        if item.error:
            lines.append(f"Error: `{item.error}`")
            lines.append("")
            continue
        lines.append("Source samples:")
        lines.extend(f"- {sample}" for sample in (item.sample_source or []))
        lines.append("")
        lines.append("RU samples:")
        lines.extend(f"- {sample}" for sample in (item.sample_translated or []))
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
