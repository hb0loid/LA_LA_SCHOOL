from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import torch
import torchaudio
import soundfile as sf
from transformers import AutoModel, AutoProcessor


SRT_BLOCK = re.compile(
    r"(?ms)^\s*\d+\s*\n"
    r"(?P<start>\d\d:\d\d:\d\d,\d{3})\s+-->\s+"
    r"(?P<end>\d\d:\d\d:\d\d,\d{3})\s*\n"
    r"(?P<text>.*?)(?=\n\s*\n|\Z)"
)


def _seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone MOSS-TTS benchmark; not used by the bot.")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-segments", type=int, default=0)
    return parser.parse_args()


def _items(job_dir: Path) -> list[dict[str, object]]:
    srt = (job_dir / "work" / "translated.srt").read_text(encoding="utf-8-sig")
    timings = [
        (_seconds(match.group("start")), _seconds(match.group("end")))
        for match in SRT_BLOCK.finditer(srt)
    ]
    manifest = json.loads(
        (job_dir / "work" / "cosyvoice_batch_manifest.json").read_text(encoding="utf-8")
    )
    source = list(manifest.get("items") or [])
    if len(timings) != len(source):
        raise RuntimeError(f"segment mismatch: timings={len(timings)} manifest={len(source)}")
    result = []
    for index, ((start, end), item) in enumerate(zip(timings, source, strict=True), start=1):
        result.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": str(item["text"]).strip(),
                "reference": str(Path(str(item["reference"])).resolve()),
            }
        )
    return result


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = _items(args.job_dir.resolve())
    if args.max_segments > 0:
        items = items[: args.max_segments]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # TorchCodec's Windows wheel cannot resolve the locally installed FFmpeg
    # DLLs with this PyTorch build. MOSS only needs ordinary WAV I/O here.
    def soundfile_load(path: str):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T.copy()), sample_rate

    torchaudio.load = soundfile_load
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir.resolve()),
        codec_path=str(args.codec_dir.resolve()),
        codec_weight_dtype="bf16",
        trust_remote_code=True,
    )
    # The 4B speech model fits the RTX 4070 in bf16; the large 48 kHz codec
    # stays on CPU so both do not compete for the same 12 GB of VRAM.
    processor.audio_tokenizer = processor.audio_tokenizer.to("cpu")
    model = AutoModel.from_pretrained(
        str(args.model_dir.resolve()),
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    loaded_seconds = time.perf_counter() - started

    segment_results = []
    synthesis_started = time.perf_counter()
    with torch.inference_mode():
        for item in items:
            duration = max(0.4, float(item["end"]) - float(item["start"]))
            conversation = [[
                processor.build_user_message(
                    text=str(item["text"]),
                    reference=[str(item["reference"])],
                    tokens=max(5, round(duration * 12.5)),
                    language="Russian",
                )
            ]]
            item_started = time.perf_counter()
            batch = processor(conversation, mode="generation")
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max(128, round(duration * 12.5) + 64),
                do_sample=True,
                audio_temperature=1.2,
                audio_top_p=0.85,
                audio_top_k=25,
                audio_repetition_penalty=1.05,
            )
            messages = [message for message in processor.decode(outputs) if message is not None]
            if not messages or not messages[0].audio_codes_list:
                raise RuntimeError(f"MOSS returned no audio for segment {item['index']}")
            audio = messages[0].audio_codes_list[0]
            output = args.output_dir / f"{int(item['index']):05d}.wav"
            sf.write(
                str(output),
                audio.detach().cpu().to(torch.float32).transpose(0, 1).numpy(),
                int(processor.model_config.sampling_rate),
            )
            segment_results.append(
                {
                    "index": item["index"],
                    "seconds": round(time.perf_counter() - item_started, 3),
                    "output": str(output),
                }
            )
            print(f"MOSS_SEGMENT {item['index']}/{len(items)}", flush=True)

    result = {
        "provider": "moss-tts-local-v1.5",
        "status": "ok",
        "segments": len(items),
        "load_seconds": round(loaded_seconds, 3),
        "tts_seconds": round(time.perf_counter() - synthesis_started, 3),
        "segment_results": segment_results,
    }
    (args.output_dir.parent / "moss_benchmark_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
