from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import whisper

from laladub.translation import translate_text
from laladub.models import DubConfig
from laladub.pipeline import _looks_like_meta_hallucination
from laladub.quality import collapse_repetitions


@dataclass(frozen=True)
class Variant:
    name: str
    language: str
    chunk_seconds: float | None
    audio_kind: str


VARIANTS = (
    Variant("01_vi_whole_clean", "vi", None, "clean"),
    Variant("02_vi_18s_compressed", "vi", 18.0, "compressed"),
    Variant("03_ko_24s_warped", "ko", 24.0, "warped"),
    Variant("04_en_18s_warped", "en", 18.0, "warped"),
    Variant("05_tr_18s_compressed", "tr", 18.0, "compressed"),
    Variant("06_de_24s_warped", "de", 24.0, "warped"),
    Variant("07_ja_18s_compressed", "ja", 18.0, "compressed"),
    Variant("08_es_24s_warped", "es", 24.0, "warped"),
)

ARTIFACT_PASSES = (
    ("vi_compressed_5s", "vi", "compressed", 5.0, 2.5),
    ("vi_warped_4s", "vi", "warped", 4.0, 2.0),
    ("en_warped_5s", "en", "warped", 5.0, 2.5),
    ("tr_warped_4s", "tr", "warped", 4.0, 2.0),
)


def _prepare_audio(source: Path, output_dir: Path) -> dict[str, Path]:
    clean = output_dir / "audio_clean.wav"
    compressed = output_dir / "audio_compressed.wav"
    warped = output_dir / "audio_warped.wav"
    commands = (
        (
            clean,
            "aresample=16000",
        ),
        (
            compressed,
            "aresample=16000,acompressor=threshold=-32dB:ratio=10:attack=4:release=120:makeup=8dB,alimiter=limit=0.98",
        ),
        (
            warped,
            "aresample=16000,asetrate=15520,aresample=16000,highpass=f=110,lowpass=f=5400,"
            "acompressor=threshold=-34dB:ratio=12:attack=3:release=100:makeup=9dB,alimiter=limit=0.98",
        ),
    )
    for output, audio_filter in commands:
        if output.exists() and output.stat().st_size > 1024:
            continue
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-af",
                audio_filter,
                str(output),
            ],
            check=True,
        )
    return {"clean": clean, "compressed": compressed, "warped": warped}


def _decode(model: object, audio: object, language: str) -> list[dict]:
    result = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        fp16=True,
        verbose=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=None,
        logprob_threshold=None,
        no_speech_threshold=None,
        condition_on_previous_text=True,
        initial_prompt=None,
        word_timestamps=False,
        hallucination_silence_threshold=None,
    )
    return [dict(item) for item in result.get("segments", [])]


def _decode_variant(model: object, audio_path: Path, variant: Variant) -> list[dict]:
    audio = whisper.load_audio(str(audio_path))
    if variant.chunk_seconds is None:
        return _decode(model, audio, variant.language)

    chunk_samples = max(1, round(variant.chunk_seconds * whisper.audio.SAMPLE_RATE))
    segments: list[dict] = []
    for sample_start in range(0, len(audio), chunk_samples):
        sample_end = min(len(audio), sample_start + chunk_samples)
        duration = (sample_end - sample_start) / whisper.audio.SAMPLE_RATE
        if duration < 0.75:
            continue
        offset = sample_start / whisper.audio.SAMPLE_RATE
        for item in _decode(model, audio[sample_start:sample_end], variant.language):
            item["start"] = float(item.get("start", 0.0)) + offset
            item["end"] = float(item.get("end", 0.0)) + offset
            segments.append(item)
    return segments


def _collapse(text: str, maximum: int = 4) -> str:
    words = text.split()
    result: list[str] = []
    last = ""
    count = 0
    for word in words:
        key = re.sub(r"[^\w]+", "", word.casefold())
        if key and key == last:
            count += 1
            if count > maximum:
                continue
        else:
            last = key
            count = 1
        result.append(word)
    return " ".join(result).strip()


def _translate_windows(segments: list[dict], language: str, output_dir: Path) -> tuple[str, list[dict]]:
    config = DubConfig(output=output_dir / "unused.mp4", workdir=output_dir, translator="googleweb")
    grouped: dict[int, list[dict]] = {}
    for item in segments:
        key = int(float(item.get("start", 0.0)) // 18)
        grouped.setdefault(key, []).append(item)

    translated_groups: list[str] = []
    records: list[dict] = []
    for key in sorted(grouped):
        source = _collapse(" ".join(str(item.get("text", "")).strip() for item in grouped[key]))
        if not source:
            continue
        try:
            translated = translate_text(source, language, "ru", config)
        except Exception as exc:
            translated = f"[translation failed: {type(exc).__name__}] {source}"
        translated = _collapse(translated)
        translated_groups.append(translated)
        records.append({"window": key, "source": source, "translated": translated})
    return "\n".join(translated_groups).strip(), records


def _harvest_neural_meta_artifacts(
    model: object,
    audio_paths: dict[str, Path],
    output_dir: Path,
) -> list[dict]:
    config = DubConfig(output=output_dir / "unused.mp4", workdir=output_dir, translator="googleweb")
    records: list[dict] = []
    seen_source: set[str] = set()
    for pass_name, language, audio_kind, window_seconds, stride_seconds in ARTIFACT_PASSES:
        audio = whisper.load_audio(str(audio_paths[audio_kind]))
        window_samples = max(1, round(window_seconds * whisper.audio.SAMPLE_RATE))
        stride_samples = max(1, round(stride_seconds * whisper.audio.SAMPLE_RATE))
        for sample_start in range(0, len(audio), stride_samples):
            sample_end = min(len(audio), sample_start + window_samples)
            duration = (sample_end - sample_start) / whisper.audio.SAMPLE_RATE
            if duration < 0.75:
                continue
            start = sample_start / whisper.audio.SAMPLE_RATE
            local = _decode(model, audio[sample_start:sample_end], language)
            source = _collapse(" ".join(str(item.get("text", "")).strip() for item in local))
            key = re.sub(r"[^\w]+", " ", source.casefold()).strip()
            if not source or not key or key in seen_source or not _looks_like_meta_hallucination(source):
                continue
            seen_source.add(key)
            try:
                translated = _collapse(translate_text(source, language, "ru", config))
            except Exception as exc:
                print(f"neural artifact translation skipped: {type(exc).__name__}: {exc}")
                continue
            if not translated:
                continue
            records.append(
                {
                    "pass": pass_name,
                    "language": language,
                    "start": round(start, 3),
                    "end": round(start + duration, 3),
                    "source": source,
                    "translated": translated,
                }
            )
    (output_dir / "neural_meta_artifacts.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "neural_meta_artifacts.txt").write_text(
        "\n\n".join(
            f"[{item['pass']} {item['start']:.1f}-{item['end']:.1f}]\n"
            f"RAW: {item['source']}\nRU: {item['translated']}"
            for item in records
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    print(f"neural meta artifacts: {len(records)}", flush=True)
    return records


def _sentence_repeat_score(sentence: str) -> float:
    words = re.findall(r"[^\W_]+", sentence.casefold(), flags=re.UNICODE)
    if len(words) < 2:
        return 0.0
    duplicate_words = len(words) - len(set(words))
    repeated_ngrams = 0
    for size in range(2, min(5, len(words) // 2 + 1)):
        counts: dict[tuple[str, ...], int] = {}
        for index in range(len(words) - size + 1):
            ngram = tuple(words[index : index + size])
            counts[ngram] = counts.get(ngram, 0) + 1
        repeated_ngrams += sum(count - 1 for count in counts.values() if count > 1) * size
    return round((duplicate_words + repeated_ngrams * 1.8) / len(words), 4)


def _replace_with_neural_artifacts(
    text: str,
    artifacts: list[dict],
    seed: str,
) -> tuple[str, dict]:
    protected = text.strip()
    abbreviations = {"Mr.": "Mr<dot>", "Mrs.": "Mrs<dot>", "г-н.": "г-н<dot>"}
    for original, placeholder in abbreviations.items():
        protected = protected.replace(original, placeholder)
    sentences = [
        item.strip().replace("<dot>", ".")
        for item in re.split(r"(?<=[.!?])\s+", protected)
        if item.strip()
    ]
    if not sentences or not artifacts:
        return text.strip(), {"ratio": 0.0, "replacements": []}
    seed_value = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed_value)
    ratio = 0.30 + rng.random() * 0.20
    replacement_count = min(len(artifacts), max(1, round(len(sentences) * ratio)))

    scored = [(index, _sentence_repeat_score(sentence)) for index, sentence in enumerate(sentences)]
    repeated = sorted(
        (item for item in scored if item[1] > 0.12),
        key=lambda item: (item[1], len(sentences[item[0]])),
        reverse=True,
    )
    selected_indices = [index for index, _score in repeated[:replacement_count]]
    remaining_indices = [index for index in range(len(sentences)) if index not in selected_indices]
    rng.shuffle(remaining_indices)
    selected_indices.extend(remaining_indices[: max(0, replacement_count - len(selected_indices))])
    strong_terms = ("subscribe", "kênh", "channel", "subtitle", "amara", "phụ đề", "altyaz", "субтит", "подпис")
    strong = [
        item
        for item in artifacts
        if any(term in str(item["source"]).casefold() for term in strong_terms)
    ]
    weak = [item for item in artifacts if item not in strong]
    rng.shuffle(strong)
    rng.shuffle(weak)
    ordered = strong + weak
    result = list(sentences)
    replacement_log: list[dict] = []
    for sentence_index, artifact in zip(selected_indices, ordered):
        original = result[sentence_index]
        score = _sentence_repeat_score(original)
        replacement = str(artifact["translated"]).strip()
        result[sentence_index] = replacement
        replacement_log.append(
            {
                "sentence_index": sentence_index,
                "reason": "repetition" if score > 0.12 else "random",
                "repeat_score": score,
                "original": original,
                "artifact_raw": artifact["source"],
                "artifact_translated": replacement,
                "artifact_pass": artifact["pass"],
            }
        )
    return " ".join(result), {
        "requested_ratio": round(ratio, 4),
        "actual_ratio": round(len(replacement_log) / len(sentences), 4),
        "sentences": len(sentences),
        "replacements": replacement_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--decorate-only", action="store_true")
    parser.add_argument("--reuse-artifacts", action="store_true")
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    audio_paths: dict[str, Path] | None = None
    model: object | None = None
    if not args.reuse_artifacts or not args.decorate_only:
        audio_paths = _prepare_audio(args.video.resolve(), output_dir)
        model = whisper.load_model(args.model, device="cuda")
    if not args.decorate_only:
        assert audio_paths is not None and model is not None
        for variant in VARIANTS:
            started = time.perf_counter()
            segments = _decode_variant(model, audio_paths[variant.audio_kind], variant)
            translated, records = _translate_windows(segments, variant.language, output_dir)
            raw_text = "\n".join(_collapse(str(item.get("text", ""))) for item in segments).strip()
            (output_dir / f"{variant.name}_raw.txt").write_text(raw_text + "\n", encoding="utf-8")
            (output_dir / f"{variant.name}_ru.txt").write_text(translated + "\n", encoding="utf-8")
            (output_dir / f"{variant.name}.json").write_text(
                json.dumps({"segments": segments, "windows": records}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            elapsed = time.perf_counter() - started
            summary.append(
                {
                    "variant": variant.name,
                    "language": variant.language,
                    "chunk_seconds": variant.chunk_seconds,
                    "audio": variant.audio_kind,
                    "segments": len(segments),
                    "source_chars": len(raw_text),
                    "russian_chars": len(translated),
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    artifact_json = output_dir / "neural_meta_artifacts.json"
    if args.reuse_artifacts:
        neural_artifacts = json.loads(artifact_json.read_text(encoding="utf-8"))
    else:
        assert audio_paths is not None and model is not None
        neural_artifacts = _harvest_neural_meta_artifacts(model, audio_paths, output_dir)

    decorated_sources = (
        ("09_vi_broken_with_meta", "02_vi_18s_compressed_ru.txt"),
        ("10_de_broken_with_meta", "06_de_24s_warped_ru.txt"),
        ("11_es_broken_with_meta", "08_es_24s_warped_ru.txt"),
    )
    mix_manifests: dict[str, dict] = {}
    for output_name, source_name in decorated_sources:
        source_path = output_dir / source_name
        if not source_path.exists():
            raise RuntimeError(f"Missing semantic source: {source_path}")
        semantic = collapse_repetitions(
            source_path.read_text(encoding="utf-8").strip(),
            max_phrase_repeats=3,
            max_word_repeats=3,
            max_ngram_words=8,
        )
        decorated, mix_manifest = _replace_with_neural_artifacts(semantic, neural_artifacts, output_name)
        target = output_dir / f"{output_name}.txt"
        target.write_text(decorated + "\n", encoding="utf-8")
        mix_manifests[output_name] = mix_manifest
        print(f"wrote decorated: {target}", flush=True)
    (output_dir / "neural_artifact_mix_manifest.json").write_text(
        json.dumps(mix_manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
