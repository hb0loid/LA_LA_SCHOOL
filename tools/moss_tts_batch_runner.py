from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as torch_functional
import torchaudio
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of MOSS-TTS v1.5 voice-cloned WAV files.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(manifest.get("items") or [])
    if not items:
        raise RuntimeError("MOSS manifest contains no items.")

    model_dir = Path(manifest["model_dir"]).resolve()
    codec_dir = Path(manifest["codec_dir"]).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"MOSS model directory does not exist: {model_dir}")
    if not codec_dir.is_dir():
        raise FileNotFoundError(f"MOSS codec directory does not exist: {codec_dir}")

    requested_device = str(manifest.get("device") or "auto").strip().lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MOSS requested CUDA, but torch.cuda.is_available() is false.")

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # TorchCodec's Windows wheel cannot resolve the local FFmpeg DLLs with the
    # PyTorch build used by MOSS. The processor only needs ordinary WAV input.
    def soundfile_load(path: str, *args, **kwargs):
        del args, kwargs
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T.copy()), sample_rate

    torchaudio.load = soundfile_load
    print("MOSS_LOADING", flush=True)
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        codec_path=str(codec_dir),
        codec_weight_dtype="bf16",
        trust_remote_code=True,
    )
    # The 4B speech model fits the RTX 4070 in bf16 only when the large audio
    # tokenizer remains on CPU.
    processor.audio_tokenizer = processor.audio_tokenizer.to("cpu")
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print("MOSS_READY", flush=True)

    seed = int(manifest.get("seed", 42))
    duration_control = bool(manifest.get("duration_control", True))
    lead_pause_seconds = max(0.0, float(manifest.get("lead_pause_seconds", 0.0)))
    trail_pause_seconds = max(0.0, float(manifest.get("trail_pause_seconds", 0.0)))
    completion_margin_seconds = max(0.0, float(manifest.get("completion_margin_seconds", 0.32)))
    edge_padding_seconds = max(0.0, float(manifest.get("edge_padding_seconds", 0.04)))
    natural_max_new_tokens = max(128, int(manifest.get("natural_max_new_tokens", 512)))
    total = len(items)
    with torch.inference_mode():
        for index, item in enumerate(items, start=1):
            output_path = Path(item["output"]).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            reference = Path(item["reference"]).resolve()
            if not reference.is_file():
                raise FileNotFoundError(reference)

            text = str(item.get("text") or "").strip()
            if not text:
                raise RuntimeError(f"MOSS item {index} text is empty.")
            generation_text = text
            if lead_pause_seconds > 0.0:
                generation_text = f"[pause {lead_pause_seconds:.2f}s]{generation_text}"
            if trail_pause_seconds > 0.0:
                # Asking the model itself for a short trailing pause keeps its
                # EOS away from the final consonant. Appending silence after a
                # clipped decode cannot restore the missing sound.
                generation_text = f"{generation_text}[pause {trail_pause_seconds:.2f}s]"
            target_seconds = max(0.4, float(item.get("target_seconds") or 0.4))
            generation_seconds = target_seconds + completion_margin_seconds
            duration_tokens = max(5, round(generation_seconds * 12.5))
            language = str(item.get("language") or "Russian")
            torch.manual_seed(seed + index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + index)

            print(f"MOSS_START\t{index}\t{total}", flush=True)
            user_message = {
                "text": generation_text,
                "reference": [str(reference)],
                "language": language,
            }
            if duration_control:
                user_message["tokens"] = duration_tokens
            conversation = [[processor.build_user_message(**user_message)]]
            batch = processor(conversation, mode="generation")
            frame_budget = max(16, duration_tokens + 8) if duration_control else natural_max_new_tokens
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                # The duration prompt normally makes MOSS stop by itself, but
                # a missed stop token used to run a 0.5-second word all the way
                # to the old 128-frame ceiling (10.24 seconds). Keep a modest
                # completion allowance without permitting that runaway tail.
                max_new_tokens=frame_budget,
                do_sample=True,
                audio_temperature=1.2,
                audio_top_p=0.85,
                audio_top_k=25,
                audio_repetition_penalty=1.05,
            )
            messages = [message for message in processor.decode(outputs) if message is not None]
            if not messages or not messages[0].audio_codes_list:
                raise RuntimeError(f"MOSS returned no audio for item {index}.")

            audio = messages[0].audio_codes_list[0].detach().cpu().to(torch.float32)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            sample_rate = int(processor.model_config.sampling_rate)
            edge_padding_samples = round(edge_padding_seconds * sample_rate)
            if edge_padding_samples > 0:
                audio = torch_functional.pad(audio, (edge_padding_samples, edge_padding_samples))
            sf.write(
                str(output_path),
                audio.transpose(0, 1).numpy(),
                sample_rate,
            )
            if not output_path.is_file() or output_path.stat().st_size < 1024:
                raise RuntimeError(f"MOSS produced an empty WAV file: {output_path}")
            print(f"MOSS_PROGRESS\t{index}\t{total}", flush=True)


if __name__ == "__main__":
    main()
