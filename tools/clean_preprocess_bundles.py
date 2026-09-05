"""Deletes preprocess bundles left over from before they were cleaned on unpack.

A bundle is the worker's result package. The coordinator unpacks it into the
job's work/ directory and never reads it again, but until now it stayed on disk
for the job's whole retention - a thousand of them, tens of gigabytes, every one
a duplicate of a directory sitting right next to it.

Only removes a bundle whose job has clearly moved past it: the work directory
exists and holds the files the bundle was carrying. Run with --apply to delete;
without it, only reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = ("translated.srt", "source_16k.wav")


def _unpacked(job_dir: Path) -> bool:
    """Whether this job already has what the bundle was carrying."""
    workdir = job_dir / "work"
    if not workdir.is_dir():
        return False
    return all((workdir / name).is_file() for name in REQUIRED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("runs/bot-release"),
        help="Where job folders live.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete.")
    args = parser.parse_args()

    freed = 0
    removed = 0
    kept = 0
    for bundle in args.workdir.glob("*/*/remote_result/documents/*.zip"):
        job_dir = bundle.parents[2]
        try:
            size = bundle.stat().st_size
        except OSError:
            continue
        if not _unpacked(job_dir):
            kept += 1
            continue
        freed += size
        removed += 1
        if args.apply:
            bundle.unlink(missing_ok=True)

    verb = "Удалено" if args.apply else "Можно удалить"
    print(f"{verb}: {removed} пакет(ов), {freed / (1024 ** 3):.1f} ГБ")
    if kept:
        print(f"Оставлено (работа ещё не распакована): {kept}")


if __name__ == "__main__":
    main()
