from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of Chatterbox multilingual TTS WAV files.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(manifest.get("items") or [])
    if not items:
        raise RuntimeError("Chatterbox manifest contains no items.")

    cache_dir = Path(manifest.get("cache_dir") or "models/chatterbox").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    device = _resolve_device(str(manifest.get("device") or "auto"))
    model_name = str(manifest.get("model") or "v3")
    exaggeration = float(manifest.get("exaggeration", 0.5))
    cfg_weight = float(manifest.get("cfg_weight", 0.5))
    language_id = str(manifest.get("language_id") or "ru")

    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    pretrained_kwargs = {"device": device}
    if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters:
        pretrained_kwargs["t3_model"] = model_name
    model = ChatterboxMultilingualTTS.from_pretrained(**pretrained_kwargs)

    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = Path(item["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ref_audio = Path(item["reference"])
        if not ref_audio.is_file():
            raise FileNotFoundError(ref_audio)

        chunks = [str(chunk).strip() for chunk in (item.get("chunks") or [item.get("text")]) if str(chunk).strip()]
        if not chunks:
            raise RuntimeError(f"Chatterbox item {index} text is empty.")

        print(f"CHATTERBOX_START\t{index}\t{total}", flush=True)
        wav_parts = []
        for chunk in chunks:
            wav = model.generate(
                chunk,
                language_id=str(item.get("language_id") or language_id),
                audio_prompt_path=str(ref_audio),
                exaggeration=float(item.get("exaggeration", exaggeration)),
                cfg_weight=float(item.get("cfg_weight", cfg_weight)),
            )
            wav_parts.append(wav.detach().cpu())
        wav = torch.cat(wav_parts, dim=1) if len(wav_parts) > 1 else wav_parts[0]
        ta.save(str(output_path), wav, model.sr)

        if not output_path.exists() or output_path.stat().st_size < 1024:
            raise RuntimeError(f"Chatterbox produced an empty WAV file: {output_path}")
        print(f"CHATTERBOX_PROGRESS\t{index}\t{total}", flush=True)


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
