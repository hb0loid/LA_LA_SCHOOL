"""Runs one video through every input language the bot offers, text only.

Whisper is told the audio is language X and transcribes it as if it were,
then that text is translated to the target language. The point is to see
which claimed language produces the funniest wreckage - and to catch any
language that quietly errors instead.

Audio is extracted once and reused, and nothing is synthesised, so this is
transcription plus translation per language and nothing else.

    python tools/language_perl_matrix.py --video path/to/video.mp4

Results land in runs/language-matrix/<name>/: one .txt per language, plus
report.md ranking them and results.json with the raw numbers.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from laladub.asr import transcribe  # noqa: E402
from laladub.bot import ASR_METHOD_CONFIGS, SOURCE_LANGS  # noqa: E402
from laladub.ffmpeg import extract_audio, probe_duration  # noqa: E402
from laladub.models import DubConfig  # noqa: E402
from laladub.pipeline import run_dub  # noqa: E402
from laladub.srt import read_srt  # noqa: E402
from laladub.translation import translate_segments  # noqa: E402

# The live chain set, so full mode distorts the way the bot really does.
DEFAULT_PIVOTS = (
    "input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar"
    "|input,en,de|input,ja,ko,en|input,tr,ar,en|en,th,he,en|en,ms,he,en"
)


@dataclass
class LanguageResult:
    code: str
    label: str
    status: str = "ok"
    seconds: float = 0.0
    segments: int = 0
    source_chars: int = 0
    translated_chars: int = 0
    unique_lines: int = 0
    repeat_share: float = 0.0
    similarity_to_truth: float = 0.0
    artifacts: int = 0
    error: str | None = None
    sample: list[str] = field(default_factory=list)

    lexical_diversity: float = 0.0
    volume: float = 0.0

    @property
    def wreckage_score(self) -> float:
        """How much *readable but wrong* text a language produced.

        Three things have to hold at once. It must differ from the honest
        transcript, or it is just a correct transcription. It must use varied
        words, or it is Whisper stuck in a loop - "Анон Анон Анон" scores far
        from the truth while saying nothing. And there must be enough of it,
        or two junk words beat a page of good material.

        The first version of this counted only whole repeated *lines*, which
        missed loops inside a single line and put those loops at the top.
        """
        if self.status != "ok" or not self.segments:
            return 0.0
        return round(
            (1.0 - self.similarity_to_truth) * self.lexical_diversity * self.volume * 100, 1
        )


def _config(
    workdir: Path,
    source_lang: str,
    target_lang: str,
    method: str,
    *,
    full: bool = False,
) -> DubConfig:
    """Bare mode is the language on its own. Full mode adds what the bot layers
    on top - the distortion chains and the artifact hunt - which is what
    actually reaches people, and is the reason a language that looks empty here
    can be the best one in practice."""
    backend, model, suppress = ASR_METHOD_CONFIGS.get(method, ("openai-whisper", "turbo", False))
    config = DubConfig(
        output=workdir / "unused.mp4",
        workdir=workdir,
        target_lang=target_lang,
        source_lang=source_lang,
        asr_backend=backend,
        whisper_model=model,
        whisper_device="auto",
        whisper_compute_type="auto",
        suppress_plain_ascii_tokens=suppress,
        force_source_language=True,
        translator="hybrid",
        distort_translation=full,
        inject_artifacts=full,
        tts="none",
    )
    if full:
        config.artifact_source_lang = source_lang
        config.artifact_chaos_mode = True
        config.distort_main_translation = True
        config.translation_pivots = DEFAULT_PIVOTS
        config.max_translation_hops = 3
    return config


def _results_from_disk(outdir: Path, truth_text: str, already: set[str]) -> list[LanguageResult]:
    """Rebuilds the numbers for languages a previous run had already written,
    so resuming still produces a report covering all of them."""
    labels = dict(SOURCE_LANGS)
    out: list[LanguageResult] = []
    for path in sorted(outdir.glob("*.txt")):
        if path.name.startswith("_эталон"):
            continue
        code, _, label = path.stem.partition("-")
        if code in already or code not in labels:
            continue
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue
        result = LanguageResult(code=code, label=label)
        result.segments = len(lines)
        counts = Counter(lines)
        result.unique_lines = len(counts)
        result.repeat_share = round(sum(n for n in counts.values() if n > 1) / len(lines), 3)
        joined = " ".join(lines)
        result.translated_chars = len(joined)
        words = _words(joined)
        result.lexical_diversity = round(len(set(words)) / len(words), 3) if words else 0.0
        result.volume = round(min(1.0, len(words) / max(1, len(_words(truth_text)))), 3)
        result.similarity_to_truth = round(
            difflib.SequenceMatcher(None, truth_text, joined).ratio(), 3
        )
        result.sample = lines[:8]
        out.append(result)
    return out


def _run_bot_pipeline(
    video: Path, outdir: Path, source_lang: str, target_lang: str, method: str
) -> list:
    """Runs the bot's own pipeline and stops once the text is ready.

    Reproducing the chaos stages by hand was wrong: the bot makes two ASR
    passes - one forced onto the claimed language as a corruption source, one
    automatic for content and timings - and the artifact hunt depends on how
    those two relate. So the real thing is configured here, the same way the
    bot configures it, and stopped at preprocess_only, which is exactly the
    point where the text exists and nothing has been synthesised yet.
    """
    from laladub.bot import _apply_speaker_count, _apply_text_extraction_method
    from laladub.bot_config import load_bot_settings

    settings = load_bot_settings()
    job_dir = outdir / "jobs" / source_lang
    (job_dir / "work").mkdir(parents=True, exist_ok=True)
    job = {
        "job_dir": str(job_dir),
        "asr_method": method,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "translation_chaos": "crooked",
        "speaker_count": "auto",
        "glitch_profile": "clean",
        "translation_seed": f"matrix-{source_lang}",
    }
    config = DubConfig(
        output=job_dir / "dubbed.mp4",
        workdir=job_dir / "work",
        target_lang=target_lang,
        source_lang=source_lang,
        translator=settings.translator,
        tts="none",
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
        whisper_compute_type=settings.whisper_compute_type,
        artifact_whisper_model=settings.whisper_only_model,
        artifact_whisper_device=settings.artifact_whisper_device,
        inject_artifacts=settings.inject_artifacts,
        artifact_max_segments=settings.artifact_max_segments,
        artifact_ratio=settings.artifact_ratio,
        artifact_min_source_segments=settings.artifact_min_source_segments,
        artifact_min_gap_seconds=settings.artifact_min_gap_seconds,
        distort_translation=settings.distort_translation,
        translation_pivots=settings.translation_pivots,
        max_translation_hops=settings.max_translation_hops,
        translation_second_pass_ratio=settings.translation_second_pass_ratio,
        translation_seed=job["translation_seed"],
        translation_chaos="crooked",
        max_line_repeats=settings.max_line_repeats,
        artifact_source=settings.artifact_source,
        artifact_cross_language_share=settings.artifact_cross_language_share,
        channel_rebrand_share=settings.channel_rebrand_share,
        separation=settings.separation,
        separation_device=settings.separation_device,
        audio_bed=settings.audio_bed,
        collapse_repetitions=True,
    )
    _apply_text_extraction_method(config, job, settings)
    _apply_speaker_count(config, job)
    config.preprocess_only = True
    srt_path = run_dub(video, config)
    if not srt_path or not Path(srt_path).is_file():
        return []
    return read_srt(Path(srt_path), translated=True)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.casefold(), flags=re.UNICODE) if w]


def _texts(segments) -> list[str]:
    return [s.spoken_text.strip() for s in segments if s.spoken_text.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="One video through every input language, text only.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--target", default="ru", help="Language to translate into (default: ru)")
    parser.add_argument("--method", default="ow-large-v3-chaos-backbone", help="ASR method id")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--only", default="", help="Comma-separated language codes to run")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip languages whose text file is already there, to resume an interrupted run",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the whole chaos pipeline (distortion chains + artifacts), as the bot does",
    )
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"Нет файла: {video}", flush=True)
        return 1

    outdir = args.outdir or (
        ROOT / "runs" / "language-matrix" / f"{video.stem}{'-full' if args.full else ''}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "work"
    work.mkdir(parents=True, exist_ok=True)

    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    languages = [
        (code, label)
        for code, label in SOURCE_LANGS
        if code != "auto" and (not wanted or code in wanted)
    ]

    audio = work / "source_16k.wav"
    if not audio.is_file():
        print("Извлекаю аудио...", flush=True)
        extract_audio(video, audio)
    duration = probe_duration(audio)

    # The reference: the language actually spoken, so every other run can be
    # measured against an honest transcript of the same audio.
    truth_code = args.target
    print(f"Эталон ({truth_code})...", flush=True)
    truth_config = _config(work, truth_code, args.target, args.method)
    truth_segments = transcribe(audio, truth_config)
    truth_text = " ".join(_texts(truth_segments))
    (outdir / f"_эталон-{truth_code}.txt").write_text(truth_text, encoding="utf-8")

    results: list[LanguageResult] = []
    for index, (code, label) in enumerate(languages, start=1):
        done = outdir / f"{code}-{label}.txt"
        if args.skip_existing and done.is_file() and done.stat().st_size > 0:
            print(f"[{index}/{len(languages)}] {code} ({label}) - уже есть, пропускаю", flush=True)
            continue
        print(f"[{index}/{len(languages)}] {code} ({label})", flush=True)
        result = LanguageResult(code=code, label=label)
        started = time.perf_counter()
        try:
            if args.full:
                segments = _run_bot_pipeline(video, outdir, code, args.target, args.method)
            else:
                config = _config(work, code, args.target, args.method)
                segments = transcribe(audio, config)
                result.source_chars = sum(len(t) for t in _texts(segments))
                if segments:
                    translate_segments(segments, config)
            lines = _texts(segments)
            result.segments = len(lines)
            result.translated_chars = sum(len(t) for t in lines)
            counts = Counter(lines)
            result.unique_lines = len(counts)
            if lines:
                repeated = sum(n for n in counts.values() if n > 1)
                result.repeat_share = round(repeated / len(lines), 3)
            joined = " ".join(lines)
            result.similarity_to_truth = round(
                difflib.SequenceMatcher(None, truth_text, joined).ratio(), 3
            )
            words = _words(joined)
            result.lexical_diversity = (
                round(len(set(words)) / len(words), 3) if words else 0.0
            )
            truth_words = max(1, len(_words(truth_text)))
            result.volume = round(min(1.0, len(words) / truth_words), 3)
            result.sample = lines[:8]
            (outdir / f"{code}-{label}.txt").write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"    ошибка: {result.error}", flush=True)
        result.seconds = round(time.perf_counter() - started, 1)
        results.append(result)

    if args.skip_existing:
        results.extend(_results_from_disk(outdir, truth_text, {r.code for r in results}))
    results.sort(key=lambda r: r.wreckage_score, reverse=True)
    (outdir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        f"# Перлы по языкам: {video.name}",
        "",
        f"Эталон ({truth_code}): {truth_text[:300]}",
        "",
        "Каждый язык скормлен Whisper как если бы речь была на нём, затем текст",
        f"переведён на «{args.target}». Оценка высокая там, где текста много (объём),",
        "слова разные (разнообразие) и при этом он далёк от эталона. Залипания",
        "вроде «Анон Анон Анон» получают низкую оценку: это поломка, а не перл.",
        "",
        "| Язык | Оценка | Реплик | Разнообразие слов | Объём | Схожесть с эталоном | Сек |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.status != "ok":
            report.append(f"| {r.label} ({r.code}) | ошибка | - | - | - | - | {r.seconds} |")
            continue
        report.append(
            f"| {r.label} ({r.code}) | {r.wreckage_score} | {r.segments} | "
            f"{int(r.lexical_diversity * 100)}% | {int(r.volume * 100)}% | "
            f"{int(r.similarity_to_truth * 100)}% | {r.seconds} |"
        )
    failures = [r for r in results if r.status != "ok"]
    if failures:
        report += ["", "## Ошибки", ""]
        for r in failures:
            report.append(f"- **{r.label} ({r.code})**: {r.error}")
    report += ["", "## Примеры", ""]
    for r in results[:12]:
        if r.status != "ok" or not r.sample:
            continue
        report.append(f"### {r.label} ({r.code}) — {r.wreckage_score}")
        report += [f"- {line}" for line in r.sample[:5]]
        report.append("")
    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"\nГотово: {outdir}", flush=True)
    print(f"Отчёт: {outdir / 'report.md'}", flush=True)
    if failures:
        print(f"Языков с ошибками: {len(failures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
