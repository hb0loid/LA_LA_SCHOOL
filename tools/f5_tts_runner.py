from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated F5-TTS inference runner.")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text-base64", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ref-text-base64", default="")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--ckpt-file", default="")
    parser.add_argument("--vocab-file", default="")
    parser.add_argument("--hf-repo", default="Misha24-10/F5-TTS_RUSSIAN")
    parser.add_argument("--hf-ckpt-path", default="F5TTS_v1_Base_v2/model_last_inference.safetensors")
    parser.add_argument("--hf-vocab-path", default="F5TTS_v1_Base/vocab.txt")
    parser.add_argument("--cache-dir", default="models/f5tts")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--cross-fade-duration", type=float, default=0.15)
    parser.add_argument("--remove-silence", action="store_true")
    args = parser.parse_args()

    text = base64.b64decode(args.text_base64.encode("ascii")).decode("utf-8")
    ref_text = (
        base64.b64decode(args.ref_text_base64.encode("ascii")).decode("utf-8")
        if args.ref_text_base64
        else ""
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_file = _resolve_model_file(args.ckpt_file, args.hf_repo, args.hf_ckpt_path, args.cache_dir)
    vocab_file = _resolve_model_file(args.vocab_file, args.hf_repo, args.hf_vocab_path, args.cache_dir)
    device = _resolve_device(args.device)

    from f5_tts.api import F5TTS

    tts = F5TTS(
        model=args.model,
        ckpt_file=str(ckpt_file),
        vocab_file=str(vocab_file),
        device=device,
    )
    tts.infer(
        ref_file=str(Path(args.ref_audio)),
        ref_text=ref_text,
        gen_text=text,
        file_wave=str(output_path),
        target_rms=args.target_rms,
        cross_fade_duration=args.cross_fade_duration,
        cfg_strength=args.cfg_strength,
        nfe_step=args.nfe_step,
        speed=args.speed,
        remove_silence=args.remove_silence,
    )

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"F5-TTS produced an empty WAV file: {output_path}")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "device": device,
                "ckpt_file": str(ckpt_file),
                "vocab_file": str(vocab_file),
            },
            ensure_ascii=False,
        )
    )


def _resolve_model_file(local_path: str, repo: str, repo_path: str, cache_dir: str) -> Path:
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=repo_path,
            cache_dir=cache_dir,
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
