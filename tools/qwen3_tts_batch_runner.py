from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import soundfile as sf
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of Qwen3-TTS voice-clone WAV files.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_name = str(manifest["model"])
    cache_dir = Path(manifest["cache_dir"]).resolve()
    language = str(manifest.get("language") or "Russian")
    seed = int(manifest.get("seed", 42))
    items = list(manifest.get("items") or [])
    if not items:
        raise RuntimeError("Qwen3-TTS manifest contains no items.")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS requires CUDA in the configured Python environment.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)

    from qwen_tts import Qwen3TTSModel

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        cache_dir=str(cache_dir),
    )

    prompt_cache: dict[tuple[str, str, bool], object] = {}
    total = len(items)
    for index, item in enumerate(items, start=1):
        started = time.perf_counter()
        output_path = Path(item["output"])
        reference_path = Path(item["reference"])
        text = str(item["text"]).strip()
        reference_text = str(item.get("reference_text") or "").strip()
        x_vector_only = bool(item.get("x_vector_only", False))
        target_seconds = max(0.1, float(item.get("target_seconds") or 0.1))
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_key = (str(reference_path.resolve()), reference_text, x_vector_only)
        prompt = prompt_cache.get(prompt_key)
        if prompt is None:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(reference_path),
                ref_text=None if x_vector_only else reference_text,
                x_vector_only_mode=x_vector_only,
            )
            prompt_cache[prompt_key] = prompt

        estimated_seconds = max(target_seconds * 2.5, len(text) / 10.0, 4.0)
        max_new_tokens = max(96, min(384, math.ceil(estimated_seconds * 12.0)))
        print(f"QWEN3_START\t{index}\t{total}\t{max_new_tokens}", flush=True)
        torch.manual_seed(seed + index)
        torch.cuda.manual_seed_all(seed + index)
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        sf.write(output_path, wavs[0], sample_rate)
        elapsed = time.perf_counter() - started
        print(f"QWEN3_PROGRESS\t{index}\t{total}\t{elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
