from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import re
import threading
import time

import soundfile as sf
import torch

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    WrappedGenerateResponse,
    generate_long,
    init_model,
)
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


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
    parser = argparse.ArgumentParser(description="Standalone Fish Audio S2 benchmark; not used by the bot.")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
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
    return [
        {
            "index": index,
            "start": start,
            "end": end,
            "text": str(item["text"]).strip(),
            "prompt_text": str(item.get("prompt_text") or "").strip(),
            "reference": str(Path(str(item["reference"])).resolve()),
        }
        for index, ((start, end), item) in enumerate(zip(timings, source, strict=True), start=1)
    ]


class SplitDeviceEngine(TTSInferenceEngine):
    """Keep the 4B language model on CUDA and the codec on CPU for 12 GB GPUs."""

    def send_Llama_request(self, req, prompt_tokens, prompt_texts):  # noqa: N802
        response_queue: queue.Queue = queue.Queue()
        self.llama_queue.put(
            GenerateRequest(
                request={
                    "device": "cuda",
                    "max_new_tokens": req.max_new_tokens,
                    "text": req.text,
                    "top_p": req.top_p,
                    "repetition_penalty": req.repetition_penalty,
                    "temperature": req.temperature,
                    "compile": self.compile,
                    "iterative_prompt": req.chunk_length > 0,
                    "chunk_length": req.chunk_length,
                    "prompt_tokens": prompt_tokens,
                    "prompt_text": prompt_texts,
                },
                response_queue=response_queue,
            )
        )
        return response_queue


def _launch_llama_queue(checkpoint_path: str, device: str, precision: torch.dtype):
    """Fish's launcher waits forever when model initialization fails in its thread."""
    input_queue: queue.Queue = queue.Queue()
    init_event = threading.Event()
    init_error: list[BaseException] = []

    def worker() -> None:
        try:
            model, decode_one_token = init_model(
                checkpoint_path, device, precision, compile=False
            )
            with torch.device(device):
                # The upstream launcher allocates for the checkpoint maximum
                # (32,768 tokens), which spills far beyond a 12 GB GPU.  The
                # short benchmark utterances need only a small fraction of it.
                model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=min(model.config.max_seq_len, 2048),
                    dtype=next(model.parameters()).dtype,
                )
        except BaseException as exc:
            init_error.append(exc)
            init_event.set()
            return

        init_event.set()
        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                return
            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **item.request
                ):
                    item.response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as exc:
                item.response_queue.put(
                    WrappedGenerateResponse(status="error", response=exc)
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()
    if init_error:
        raise RuntimeError("Fish S2 model initialization failed") from init_error[0]
    return input_queue


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir.resolve()
    items = _items(args.job_dir.resolve())
    if args.max_segments > 0:
        items = items[: args.max_segments]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    started = time.perf_counter()
    llama_queue = _launch_llama_queue(
        checkpoint_path=str(checkpoint),
        device=args.device,
        precision=torch.float16,
    )
    decoder = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(checkpoint / "codec.pth"),
        device="cpu",
    )
    engine = SplitDeviceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder,
        precision=torch.float32,
        compile=False,
    )
    loaded_seconds = time.perf_counter() - started

    segment_results = []
    synthesis_started = time.perf_counter()
    for item in items:
        item_started = time.perf_counter()
        reference = ServeReferenceAudio(
            audio=Path(str(item["reference"])).read_bytes(),
            text=str(item["prompt_text"]),
        )
        request = ServeTTSRequest(
            text=str(item["text"]),
            references=[reference],
            seed=42 + int(item["index"]),
            use_memory_cache="on",
            normalize=False,
            streaming=False,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            format="wav",
        )
        final_audio = None
        sample_rate = None
        for result in engine.inference(request):
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "final" and result.audio is not None:
                sample_rate, final_audio = result.audio
        if final_audio is None or sample_rate is None:
            raise RuntimeError(f"Fish S2 returned no audio for segment {item['index']}")
        output = args.output_dir / f"{int(item['index']):05d}.wav"
        sf.write(str(output), final_audio, int(sample_rate))
        segment_results.append(
            {
                "index": item["index"],
                "seconds": round(time.perf_counter() - item_started, 3),
                "output": str(output),
            }
        )
        print(f"FISH_SEGMENT {item['index']}/{len(items)}", flush=True)

    result = {
        "provider": "fish-audio-s2-pro",
        "status": "ok",
        "segments": len(items),
        "load_seconds": round(loaded_seconds, 3),
        "tts_seconds": round(time.perf_counter() - synthesis_started, 3),
        "segment_results": segment_results,
    }
    (args.output_dir.parent / "fish_s2_benchmark_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
