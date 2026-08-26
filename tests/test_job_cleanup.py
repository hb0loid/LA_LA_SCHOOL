from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from laladub.bot import _cleanup_finished_jobs_once

RETENTION = 30 * 24 * 3600


class JobCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workdir = Path(self._tempdir.name)
        self.settings = SimpleNamespace(workdir=self.workdir, job_retention_seconds=float(RETENTION))

    def _job(self, number: str, status: str, age_days: float, *, broken: bool = False) -> Path:
        job_dir = self.workdir / "111" / number
        job_dir.mkdir(parents=True)
        stamp = time.time() - age_days * 24 * 3600
        path = job_dir / "job.json"
        if broken:
            path.write_text("{not json", encoding="utf-8")
        else:
            path.write_text(
                json.dumps({"status": status, "updated_at": stamp}), encoding="utf-8"
            )
        (job_dir / "input.mp4").write_bytes(b"x" * 16)
        os.utime(path, (stamp, stamp))
        return job_dir

    def test_finished_job_past_retention_is_deleted(self) -> None:
        job_dir = self._job("1", "done", age_days=40)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 1)
        self.assertFalse(job_dir.exists())

    def test_recent_job_is_kept(self) -> None:
        job_dir = self._job("2", "done", age_days=1)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 0)
        self.assertTrue(job_dir.exists())

    def test_job_stuck_in_running_past_retention_is_deleted(self) -> None:
        # Regression: only done/failed/rejected were swept, so a job a restart
        # left in running sat on disk - and in /queue's "stuck" count - forever.
        # Nothing genuinely running can be a month stale.
        job_dir = self._job("3", "running", age_days=60)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 1)
        self.assertFalse(job_dir.exists())

    def test_job_stuck_in_queued_past_retention_is_deleted(self) -> None:
        job_dir = self._job("4", "queued", age_days=60)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 1)
        self.assertFalse(job_dir.exists())

    def test_running_job_within_retention_is_never_touched(self) -> None:
        job_dir = self._job("5", "running", age_days=0)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 0)
        self.assertTrue(job_dir.exists())

    def test_abandoned_dialog_past_retention_is_deleted(self) -> None:
        job_dir = self._job("6", "select_target_lang", age_days=45)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 1)
        self.assertFalse(job_dir.exists())

    def test_abandoned_dialog_within_retention_is_kept(self) -> None:
        job_dir = self._job("7", "select_target_lang", age_days=3)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 0)
        self.assertTrue(job_dir.exists())

    def test_unreadable_job_past_retention_is_deleted(self) -> None:
        job_dir = self._job("8", "", age_days=50, broken=True)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 1)
        self.assertFalse(job_dir.exists())

    def test_unreadable_job_within_retention_is_kept(self) -> None:
        # A half-written job.json from a job saving right now must survive.
        job_dir = self._job("9", "", age_days=0, broken=True)
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 0)
        self.assertTrue(job_dir.exists())

    def test_retention_of_zero_disables_cleanup(self) -> None:
        job_dir = self._job("10", "done", age_days=400)
        self.settings.job_retention_seconds = 0.0
        deleted, _freed = _cleanup_finished_jobs_once(self.settings)
        self.assertEqual(deleted, 0)
        self.assertTrue(job_dir.exists())


if __name__ == "__main__":
    unittest.main()
