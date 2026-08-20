from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from laladub.bot import _format_status_counts, _job_status_counts


class JobStatusCountsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workdir = Path(self._tempdir.name)

    def _write_job(self, subpath: str, status: str) -> None:
        job_dir = self.workdir / subpath
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    def test_missing_workdir_is_empty(self) -> None:
        self.assertEqual(_job_status_counts(self.workdir / "nope"), {})

    def test_terminal_statuses_are_excluded(self) -> None:
        # /queue's file-based count exists to catch jobs stuck mid-flight
        # after a restart - done/failed/rejected pile up by the thousands
        # under the retention window and aren't relevant to that check.
        self._write_job("1/1", "done")
        self._write_job("1/2", "failed")
        self._write_job("1/3", "rejected")
        self._write_job("1/4", "running")
        counts = _job_status_counts(self.workdir)
        self.assertEqual(counts, {"running": 1})

    def test_counts_active_statuses(self) -> None:
        self._write_job("1/1", "running")
        self._write_job("1/2", "running")
        self._write_job("1/3", "queued")
        self._write_job("1/4", "select_source")
        counts = _job_status_counts(self.workdir)
        self.assertEqual(counts, {"running": 2, "queued": 1, "select_source": 1})

    def test_unreadable_json_counts_as_bad_json(self) -> None:
        job_dir = self.workdir / "1" / "1"
        job_dir.mkdir(parents=True)
        (job_dir / "job.json").write_text("not json", encoding="utf-8")
        self.assertEqual(_job_status_counts(self.workdir), {"bad_json": 1})


class FormatStatusCountsTests(unittest.TestCase):
    def test_empty_counts(self) -> None:
        self.assertEqual(_format_status_counts({}), "нет задач")

    def test_preferred_statuses_come_first(self) -> None:
        counts = {"select_source": 2, "running": 5, "bad_json": 1}
        self.assertEqual(_format_status_counts(counts), "running=5, select_source=2, bad_json=1")

    def test_zero_counts_are_omitted(self) -> None:
        counts = {"running": 0, "queued": 3}
        self.assertEqual(_format_status_counts(counts), "queued=3")


if __name__ == "__main__":
    unittest.main()
