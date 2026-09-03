from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from laladub.bot import _find_proposal_video_path, _job_number, _target_lang_value
from laladub.bot_config import load_bot_settings
from laladub.library import LibraryStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill the permanent video library from jobs that already finished before it existed."
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be archived, copy nothing.")
    args = parser.parse_args()

    settings = load_bot_settings()
    library_store = LibraryStore(settings.library_db)
    if not args.dry_run:
        settings.library_dir.mkdir(parents=True, exist_ok=True)

    total_done = 0
    archived = 0
    already_in_library = 0
    no_video_found = 0
    bytes_copied = 0

    for job_json in sorted(settings.workdir.glob("*/*/job.json")):
        try:
            job = json.loads(job_json.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"skip (bad json) {job_json}: {exc}")
            continue
        if not isinstance(job, dict) or str(job.get("status") or "") != "done":
            continue
        if str(job.get("mode") or "") == "raw_text":
            continue
        total_done += 1

        job_dir = job_json.parent
        job["job_dir"] = str(job_dir)
        job_number = _job_number(job)

        existing = library_store.get(job_number)
        if existing is not None and Path(existing.video_path).is_file():
            already_in_library += 1
            continue

        source_path = _find_proposal_video_path(job_dir, job)
        if source_path is None:
            no_video_found += 1
            continue

        dest_path = settings.library_dir / f"{job_number}{source_path.suffix}"
        size = source_path.stat().st_size
        print(f"{'[dry-run] ' if args.dry_run else ''}archive job {job_number}: {source_path} ({size / 1e6:.1f} MB)")
        if not args.dry_run:
            shutil.copy2(source_path, dest_path)
            library_store.add(
                job_number=job_number,
                user_id=int(job.get("user_id") or 0),
                source_title=str(job.get("source_title") or ""),
                target_lang=_target_lang_value(job.get("target_lang")),
                video_path=str(dest_path),
                output_filename=str(job.get("proposal_output_filename") or dest_path.name),
            )
        archived += 1
        bytes_copied += size

    print(
        f"\nDone. done_jobs_seen={total_done} archived={archived} "
        f"already_in_library={already_in_library} no_video_found={no_video_found} "
        f"total_size={bytes_copied / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
