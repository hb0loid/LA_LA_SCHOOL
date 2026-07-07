from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated CosyVoice zero-shot/cross-lingual TTS runner.")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text-base64", required=True)
    parser.add_argument("--prompt-text-base64", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="cross_lingual", choices=["cross_lingual", "zero_shot"])
    parser.add_argument("--instruction", default="You are a helpful assistant.<|endofprompt|>")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    text = base64.b64decode(args.text_base64.encode("ascii")).decode("utf-8").strip()
    prompt_text = (
        base64.b64decode(args.prompt_text_base64.encode("ascii")).decode("utf-8").strip()
        if args.prompt_text_base64
        else ""
    )
    if not text:
        raise RuntimeError("CosyVoice text is empty.")

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice repo does not exist: {repo_dir}")

    ref_audio = Path(args.ref_audio).resolve()
    if not ref_audio.is_file():
        raise FileNotFoundError(ref_audio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "third_party" / "Matcha-TTS"))

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CosyVoice requested CUDA, but torch.cuda.is_available() is false.")

    model_dir = _ensure_model_dir(Path(args.model_dir), args.model_id)
    cosyvoice = AutoModel(model_dir=str(model_dir))

    tts_text = _with_instruction(text, args.instruction)
    chunks = []
    if args.mode == "zero_shot":
        effective_prompt = _with_instruction(prompt_text or text, args.instruction)
        generator = cosyvoice.inference_zero_shot(
            tts_text,
            effective_prompt,
            str(ref_audio),
            stream=False,
            speed=args.speed,
        )
    else:
        generator = cosyvoice.inference_cross_lingual(
            tts_text,
            str(ref_audio),
            stream=False,
            speed=args.speed,
        )

    for item in generator:
        speech = item.get("tts_speech")
        if speech is None:
            continue
        chunks.append(speech.detach().cpu())

    if not chunks:
        raise RuntimeError("CosyVoice produced no speech chunks.")

    wav = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    torchaudio.save(str(output_path), wav, cosyvoice.sample_rate)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"CosyVoice produced an empty WAV file: {output_path}")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "mode": args.mode,
                "model_dir": str(model_dir),
                "sample_rate": cosyvoice.sample_rate,
                "chunks": len(chunks),
            },
            ensure_ascii=False,
        )
    )


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
    if not instruction or "<|endofprompt|>" in text:
        return text
    return instruction + text


if __name__ == "__main__":
    main()
