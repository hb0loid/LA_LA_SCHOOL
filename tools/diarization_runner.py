from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import wave


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated local pyannote speaker diarization runner.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--token-file", default="")
    parser.add_argument("--num-speakers", type=int, default=0)
    parser.add_argument("--max-speakers", type=int, default=0)
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")

    token = None
    if args.token_file:
        token_path = Path(args.token_file).resolve()
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8-sig").strip() or None

    import torch
    from pyannote.audio import Pipeline

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Diarization requested CUDA, but torch.cuda.is_available() is false.")

    pipeline = Pipeline.from_pretrained(args.model, token=token)
    if pipeline is None:
        raise RuntimeError(
            "Could not load pyannote pipeline. Accept the Hugging Face model conditions "
            "and put a read token in the configured token file."
        )
    pipeline.to(torch.device(device))

    kwargs: dict[str, int] = {}
    if args.num_speakers > 0:
        kwargs["num_speakers"] = args.num_speakers
    elif args.max_speakers > 0:
        kwargs["max_speakers"] = args.max_speakers
    waveform, sample_rate = _read_wav_for_pyannote(audio_path, torch)
    audio = {"waveform": waveform, "sample_rate": sample_rate}
    result = pipeline(audio, **kwargs)
    turns = _result_turns(result)
    initial_speakers = sorted({str(item["speaker"]) for item in turns})
    initial_metrics = _fragmentation_metrics(turns)
    selected_speaker_count = len(initial_speakers)

    # Auto diarization occasionally invents a cluster that alternates with a
    # real speaker every few frames. Such output produces mixed voice banks and
    # makes a TTS sentence change voice mid-phrase. Retry with fewer speakers
    # and keep the materially more stable segmentation. An explicitly selected
    # speaker count is always respected.
    if args.num_speakers <= 0 and selected_speaker_count > 2 and _is_fragmented(initial_metrics):
        best_turns = turns
        best_metrics = initial_metrics
        best_count = selected_speaker_count
        for count in range(selected_speaker_count - 1, 1, -1):
            candidate = _result_turns(pipeline(audio, num_speakers=count))
            metrics = _fragmentation_metrics(candidate)
            if metrics["score"] < best_metrics["score"]:
                best_turns = candidate
                best_metrics = metrics
                best_count = count
            if not _is_fragmented(metrics):
                break
        if best_count != selected_speaker_count and best_metrics["score"] <= initial_metrics["score"] * 0.8:
            print(
                "diarization auto-stabilized "
                f"speakers={selected_speaker_count}->{best_count} "
                f"score={initial_metrics['score']:.3f}->{best_metrics['score']:.3f}",
                flush=True,
            )
            turns = best_turns
            selected_speaker_count = best_count

    speakers = sorted({str(item["speaker"]) for item in turns})
    output_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "device": device,
                "speakers": speakers,
                "turns": turns,
                "initial_speakers": len(initial_speakers),
                "selected_speakers": selected_speaker_count,
                "initial_fragmentation": initial_metrics,
                "selected_fragmentation": _fragmentation_metrics(turns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"diarization speakers={len(speakers)} turns={len(turns)} device={device}")


def _result_turns(result) -> list[dict[str, object]]:
    annotation = getattr(result, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(result, "speaker_diarization", None)
    if annotation is None:
        annotation = result

    turns: list[dict[str, object]] = []
    if hasattr(annotation, "itertracks"):
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
    else:
        for segment, speaker in annotation:
            turns.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
    turns.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return turns


def _fragmentation_metrics(turns: list[dict[str, object]]) -> dict[str, float | int]:
    if not turns:
        return {"turns": 0, "short_turns": 0, "rapid_switches": 0, "short_share": 0.0, "rapid_share": 0.0, "score": 1.0}
    short_turns = sum(
        float(item["end"]) - float(item["start"]) < 0.35
        for item in turns
    )
    rapid_switches = 0
    for previous, current in zip(turns, turns[1:]):
        gap = float(current["start"]) - float(previous["end"])
        if str(current["speaker"]) != str(previous["speaker"]) and gap <= 0.25:
            rapid_switches += 1
    count = len(turns)
    short_share = short_turns / count
    rapid_share = rapid_switches / count
    return {
        "turns": count,
        "short_turns": short_turns,
        "rapid_switches": rapid_switches,
        "short_share": round(short_share, 4),
        "rapid_share": round(rapid_share, 4),
        "score": round(short_share + rapid_share, 4),
    }


def _is_fragmented(metrics: dict[str, float | int]) -> bool:
    return (
        int(metrics["turns"]) >= 8
        and float(metrics["short_share"]) >= 0.12
        and float(metrics["rapid_share"]) >= 0.20
    )


def _read_wav_for_pyannote(path: Path, torch_module):
    import numpy as np

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    waveform = torch_module.from_numpy(samples.copy()).unsqueeze(0)
    return waveform, sample_rate


if __name__ == "__main__":
    main()
