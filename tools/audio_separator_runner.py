from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BS-RoFormer separation from a JSON manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    input_path = Path(manifest["input"]).resolve()
    output_dir = Path(manifest["output_dir"]).resolve()
    model_dir = Path(manifest["model_dir"]).resolve()
    vocals_output = Path(manifest["vocals_output"]).resolve()
    instrumental_output = Path(manifest["instrumental_output"]).resolve()
    model_name = str(manifest["model"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocals_output.parent.mkdir(parents=True, exist_ok=True)
    instrumental_output.parent.mkdir(parents=True, exist_ok=True)

    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=str(output_dir),
        output_format="WAV",
        model_file_dir=str(model_dir),
        use_autocast=True,
        log_level=logging.WARNING,
    )
    separator.load_model(model_filename=model_name)
    output_files = separator.separate(str(input_path))

    stems: dict[str, Path] = {}
    for output_file in output_files:
        path = Path(output_file)
        if not path.is_absolute():
            path = output_dir / path
        lowered = path.name.lower()
        if "(vocals)" in lowered:
            stems["vocals"] = path
        elif "(instrumental)" in lowered:
            stems["instrumental"] = path

    if "vocals" not in stems or "instrumental" not in stems:
        raise RuntimeError(f"Audio Separator returned unexpected stems: {output_files}")

    shutil.copy2(stems["vocals"], vocals_output)
    shutil.copy2(stems["instrumental"], instrumental_output)
    print(
        json.dumps(
            {
                "vocals": str(vocals_output),
                "instrumental": str(instrumental_output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
