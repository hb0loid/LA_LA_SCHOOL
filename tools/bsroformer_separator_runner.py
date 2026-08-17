from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate vocals/instrumental with BS-Roformer.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    import torch
    from audio_separator.separator import Separator

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu = device == "cuda"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model_file_dir=str(args.model_dir),
        output_dir=str(args.output_dir),
        output_format="WAV",
        use_autocast=use_gpu,
    )
    # audio-separator picks its own onnxruntime/torch execution provider from
    # what is installed (onnxruntime-gpu here); there is no direct "device"
    # argument on Separator itself.
    print("BSROFORMER_LOADING", flush=True)
    separator.load_model(model_filename=args.model_file)
    print("BSROFORMER_READY", flush=True)

    output_files = separator.separate(str(args.audio))
    if len(output_files) != 2:
        raise RuntimeError(f"Expected 2 output stems (vocals, instrumental), got {len(output_files)}: {output_files}")

    vocals_path = None
    instrumental_path = None
    for name in output_files:
        path = args.output_dir / name
        if "(Vocals)" in name:
            vocals_path = path
        elif "(Instrumental)" in name:
            instrumental_path = path
    if vocals_path is None or instrumental_path is None:
        raise RuntimeError(f"Could not identify vocals/instrumental stems in: {output_files}")

    # Match the plain vocals.wav / no_vocals.wav naming the rest of the
    # pipeline expects from the Demucs path, instead of audio-separator's
    # own "<source>_(Vocals)_<model>.wav" naming.
    final_vocals = args.output_dir / "vocals.wav"
    final_instrumental = args.output_dir / "no_vocals.wav"
    shutil.move(str(vocals_path), str(final_vocals))
    shutil.move(str(instrumental_path), str(final_instrumental))

    print(
        "BSROFORMER_DONE\t"
        + json.dumps({"vocals": str(final_vocals), "instrumental": str(final_instrumental)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
