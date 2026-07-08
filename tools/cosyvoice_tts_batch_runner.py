from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of CosyVoice zero-shot/cross-lingual WAV files.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(manifest.get("items") or [])
    if not items:
        raise RuntimeError("CosyVoice manifest contains no items.")

    repo_dir = Path(manifest["repo_dir"]).resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice repo does not exist: {repo_dir}")

    device = str(manifest.get("device") or "auto")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "third_party" / "Matcha-TTS"))

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CosyVoice requested CUDA, but torch.cuda.is_available() is false.")

    model_dir = _ensure_model_dir(Path(manifest["model_dir"]), str(manifest.get("model_id") or ""))
    cosyvoice = AutoModel(model_dir=str(model_dir))
    mode = str(manifest.get("mode") or "cross_lingual")
    instruction = str(manifest.get("instruction") or "You are a helpful assistant.<|endofprompt|>")
    speed = float(manifest.get("speed", 1.0))

    total = len(items)
    for index, item in enumerate(items, start=1):
        output_path = Path(item["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ref_audio = Path(item["reference"]).resolve()
        if not ref_audio.is_file():
            raise FileNotFoundError(ref_audio)

        chunks = [str(chunk).strip() for chunk in (item.get("chunks") or [item.get("text")]) if str(chunk).strip()]
        if not chunks:
            raise RuntimeError(f"CosyVoice item {index} text is empty.")
        prompt_text = str(item.get("prompt_text") or "").strip()

        print(f"COSYVOICE_START\t{index}\t{total}", flush=True)
        wav_parts = []
        for chunk in chunks:
            tts_text = _with_instruction(chunk, instruction)
            if mode == "zero_shot":
                effective_prompt = _with_instruction(prompt_text or chunk, instruction)
                generator = cosyvoice.inference_zero_shot(
                    tts_text,
                    effective_prompt,
                    str(ref_audio),
                    stream=False,
                    speed=speed,
                )
            else:
                generator = cosyvoice.inference_cross_lingual(
                    tts_text,
                    str(ref_audio),
                    stream=False,
                    speed=speed,
                )

            generated = []
            for generated_item in generator:
                speech = generated_item.get("tts_speech")
                if speech is not None:
                    generated.append(speech.detach().cpu())
            if not generated:
                raise RuntimeError(f"CosyVoice produced no speech chunks for item {index}.")
            wav_parts.append(torch.cat(generated, dim=1) if len(generated) > 1 else generated[0])

        wav = torch.cat(wav_parts, dim=1) if len(wav_parts) > 1 else wav_parts[0]
        torchaudio.save(str(output_path), wav, cosyvoice.sample_rate)

        if not output_path.exists() or output_path.stat().st_size < 1024:
            raise RuntimeError(f"CosyVoice produced an empty WAV file: {output_path}")
        print(f"COSYVOICE_PROGRESS\t{index}\t{total}", flush=True)


def _ensure_model_dir(model_dir: Path, model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    model_dir = model_dir.resolve()
    if (model_dir / "cosyvoice3.yaml").is_file() or (model_dir / "cosyvoice2.yaml").is_file() or (
        model_dir / "cosyvoice.yaml"
    ).is_file():
        return model_dir

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    )
    if downloaded.resolve() != model_dir and not model_dir.exists():
        shutil.copytree(downloaded, model_dir)
    return model_dir


def _with_instruction(text: str, instruction: str) -> str:
    text = text.strip()
    instruction = instruction.strip()
    if not instruction:
        return text
    if text.startswith(instruction):
        return text
    return f"{instruction}{text}"


if __name__ == "__main__":
    main()
