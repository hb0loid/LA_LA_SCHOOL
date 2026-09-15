from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from laladub.worker import _cleanup_worker_jobs


class WorkerCleanupTests(unittest.TestCase):
    def test_removes_terminal_jobs_but_keeps_running_job(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workdir = Path(tempdir)
            done = workdir / "jobs" / "done"
            running = workdir / "jobs" / "running"
            for path, status in ((done, "done"), (running, "running")):
                path.mkdir(parents=True)
                (path / "job.json").write_text(json.dumps({"status": status}), encoding="utf-8")
                (path / "payload.bin").write_bytes(b"x" * 128)
            deleted, freed = _cleanup_worker_jobs(workdir)
            self.assertEqual(deleted, 1)
            self.assertGreaterEqual(freed, 128)
            self.assertFalse(done.exists())
            self.assertTrue(running.exists())

    def test_removes_only_old_orphan_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workdir = Path(tempdir)
            old = workdir / "jobs" / "old"
            fresh = workdir / "jobs" / "fresh"
            old.mkdir(parents=True)
            fresh.mkdir(parents=True)
            timestamp = time.time() - 8 * 60 * 60
            os.utime(old, (timestamp, timestamp))
            deleted, _freed = _cleanup_worker_jobs(workdir)
            self.assertEqual(deleted, 1)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())


if __name__ == "__main__":
    unittest.main()
