from __future__ import annotations

import argparse
import base64
import inspect
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Chatterbox multilingual TTS inference runner.")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text-base64", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language-id", default="ru")
    parser.add_argument("--model", default="v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default="models/chatterbox")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    args = parser.parse_args()

    text = base64.b64decode(args.text_base64.encode("ascii")).decode("utf-8")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_audio = Path(args.ref_audio)
    if not ref_audio.is_file():
        raise FileNotFoundError(ref_audio)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    device = _resolve_device(args.device)

    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    pretrained_kwargs = {"device": device}
    if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters:
        pretrained_kwargs["t3_model"] = args.model
    model = ChatterboxMultilingualTTS.from_pretrained(**pretrained_kwargs)
    wav = model.generate(
        text,
        language_id=args.language_id,
        audio_prompt_path=str(ref_audio),
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )
    ta.save(str(output_path), wav, model.sr)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"Chatterbox produced an empty WAV file: {output_path}")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "device": device,
                "language_id": args.language_id,
                "model": args.model,
            },
            ensure_ascii=False,
        )
    )


def _resolve_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


if __name__ == "__main__":
    main()
