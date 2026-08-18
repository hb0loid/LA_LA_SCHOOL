"""Generate a batch of Fish Speech S2 WAV files from a manifest.

Runs inside the Fish Speech checkout's own virtualenv, like the MOSS runner does,
so the heavy model stack never has to coexist with the bot's environment. The
model is loaded once and reused for every item - loading it per phrase would
dominate the runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Fish Speech S2 synthesis.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = list(manifest.get("items") or [])
    if not items:
        raise RuntimeError("Fish manifest contains no items.")

    repo_dir = Path(str(manifest["repo_dir"])).resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import torch
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    model_dir = Path(str(manifest["model_dir"])).resolve()
    requested = str(manifest.get("device") or "auto").strip().lower()
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    precision = torch.bfloat16 if device != "cpu" else torch.float32

    print("FISH_LOADING", flush=True)
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=model_dir,
        device=device,
        precision=precision,
        compile=False,
    )
    decoder_model = load_decoder_model(
        config_name=str(manifest.get("decoder_config") or "modded_dac_vq"),
        checkpoint_path=model_dir / "codec.pth",
        device=device,
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=False,
        precision=precision,
    )
    print("FISH_READY", flush=True)

    seed = manifest.get("seed")
    written: list[str] = []
    for index, item in enumerate(items, start=1):
        text = str(item.get("text") or "").strip()
        output_path = Path(str(item["output"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        references = []
        reference = str(item.get("reference") or "").strip()
        if reference:
            reference_path = Path(reference)
            if reference_path.is_file():
                references.append(
                    ServeReferenceAudio(
                        audio=reference_path.read_bytes(),
                        text=str(item.get("prompt_text") or ""),
                    )
                )

        request = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            format="wav",
            seed=int(seed) if seed is not None else None,
            normalize=True,
            streaming=False,
        )

        started = time.perf_counter()
        audio = None
        for result in engine.inference(request):
            # The engine yields progress events; the final one carries the audio.
            if getattr(result, "code", "") == "error":
                raise RuntimeError(f"Fish failed on item {index}: {result.error}")
            if getattr(result, "audio", None) is not None:
                audio = result.audio
        if audio is None:
            raise RuntimeError(f"Fish produced no audio for item {index}")

        sample_rate, samples = audio
        import soundfile as sf

        sf.write(str(output_path), samples, int(sample_rate))
        written.append(str(output_path))
        print(
            f"FISH_ITEM\t{index}/{len(items)}\t{time.perf_counter() - started:.2f}s\t{output_path.name}",
            flush=True,
        )

    print("FISH_DONE\t" + json.dumps({"written": written}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
