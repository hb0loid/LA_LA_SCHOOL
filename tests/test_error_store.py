from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.error_store import ErrorStore, error_fingerprint


class ErrorStoreTests(unittest.TestCase):
    def test_similar_job_errors_are_grouped(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = ErrorStore(Path(tempdir) / "errors.sqlite3")
            for job in ("60001", "60002"):
                store.record(
                    source="local_pipeline",
                    details="RuntimeError: failed",
                    traceback_text="Traceback\nRuntimeError: model failed for job 60001",
                    job_number=job,
                    user_id=int(job),
                    created_at=100.0,
                )
            groups = store.pending_groups(now=120.0)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].count, 2)
            self.assertEqual(groups[0].job_numbers, ("60001", "60002"))

    def test_notification_cooldown_prevents_spam(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = ErrorStore(Path(tempdir) / "errors.sqlite3")
            fingerprint = store.record(source="download", details="blocked", created_at=100.0)
            store.mark_notified(fingerprint, notified_at=120.0)
            store.record(source="download", details="blocked", created_at=130.0)
            self.assertEqual(store.pending_groups(now=200.0), [])
            self.assertEqual(len(store.pending_groups(now=1100.0)), 1)

    def test_fingerprint_ignores_large_job_numbers(self) -> None:
        left = error_fingerprint("worker", "x", "RuntimeError: job 60001 failed")
        right = error_fingerprint("worker", "x", "RuntimeError: job 60002 failed")
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
