from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import whisper


SAMPLE_RATE = whisper.audio.SAMPLE_RATE


def _stamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _write_outputs(directory: Path, name: str, segments: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    texts = [str(segment.get("text", "")).strip() for segment in segments]
    (directory / f"{name}.txt").write_text("\n".join(texts), encoding="utf-8")
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{_stamp(float(segment['start']))} --> {_stamp(float(segment['end']))}\n"
            f"{str(segment.get('text', '')).strip()}"
        )
    (directory / f"{name}.srt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    (directory / f"{name}.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _decode(model: object, audio: str | object, language: str) -> list[dict]:
    result = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        fp16=False,
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
    return [dict(segment) for segment in result.get("segments", [])]


def _normalise(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _metrics(segments: list[dict], elapsed: float) -> dict:
    texts = [str(segment.get("text", "")).strip() for segment in segments]
    normalised = [_normalise(text) for text in texts if _normalise(text)]
    counts = Counter(normalised)
    return {
        "elapsed_seconds": round(elapsed, 2),
        "segments": len(segments),
        "characters": sum(len(text) for text in texts),
        "words": sum(len(text.split()) for text in texts),
        "unique_segment_texts": len(counts),
        "duplicate_segment_instances": sum(count - 1 for count in counts.values() if count > 1),
        "most_repeated_segments": [
            {"count": count, "text": text}
            for text, count in counts.most_common(10)
            if count > 1
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(args.model, device="cpu")
    audio = whisper.load_audio(str(args.audio))

    whole_started = time.perf_counter()
    whole_segments = _decode(model, audio, args.language)
    whole_elapsed = time.perf_counter() - whole_started
    _write_outputs(args.output, "whole", whole_segments)

    chunk_samples = max(1, round(args.chunk_seconds * SAMPLE_RATE))
    chunked_segments: list[dict] = []
    chunk_log: list[dict] = []
    chunked_started = time.perf_counter()
    for index, sample_start in enumerate(range(0, len(audio), chunk_samples), start=1):
        sample_end = min(len(audio), sample_start + chunk_samples)
        offset = sample_start / SAMPLE_RATE
        if (sample_end - sample_start) / SAMPLE_RATE < 0.5:
            print(f"chunk {index}: skipped tail shorter than 0.5s", flush=True)
            continue
        started = time.perf_counter()
        local_segments = _decode(model, audio[sample_start:sample_end], args.language)
        elapsed = time.perf_counter() - started
        for segment in local_segments:
            segment["start"] = float(segment.get("start", 0.0)) + offset
            segment["end"] = min(len(audio) / SAMPLE_RATE, float(segment.get("end", 0.0)) + offset)
            segment["chunk_index"] = index
            chunked_segments.append(segment)
        chunk_log.append(
            {
                "chunk": index,
                "start": round(offset, 3),
                "end": round(sample_end / SAMPLE_RATE, 3),
                "segments": len(local_segments),
                "elapsed_seconds": round(elapsed, 2),
            }
        )
        print(f"chunk {index}: {offset:.1f}-{sample_end / SAMPLE_RATE:.1f}s, segments={len(local_segments)}", flush=True)

    chunked_elapsed = time.perf_counter() - chunked_started
    _write_outputs(args.output, f"chunked_{args.chunk_seconds:g}s", chunked_segments)
    summary = {
        "audio": str(args.audio.resolve()),
        "duration_seconds": round(len(audio) / SAMPLE_RATE, 3),
        "model": args.model,
        "language": args.language,
        "chunk_seconds": args.chunk_seconds,
        "whole": _metrics(whole_segments, whole_elapsed),
        "chunked": _metrics(chunked_segments, chunked_elapsed),
        "chunks": chunk_log,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
