from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from laladub.performance import PerformanceHistory, performance_record, record_terminal_job


class PerformanceTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workdir = Path(self._tempdir.name)

    def _job(self, number: int, *, duration: float, total: float, tts: str = "moss") -> tuple[Path, dict]:
        job_dir = self.workdir / "123" / str(number)
        job_dir.mkdir(parents=True, exist_ok=True)
        finished = 10_000.0 + number
        job = {
            "status": "done",
            "queued_at": finished - total,
            "started_at": finished - total + 20,
            "finished_at": finished,
            "quota_duration_ms": round(duration * 1000),
            "mode": "dub",
            "tts_provider": tts,
            "stage_seconds": {"Whisper": 12.5, "TTS": 44.0},
        }
        (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        return job_dir, job

    def test_record_contains_timings_but_not_text(self) -> None:
        job_dir, job = self._job(1, duration=60, total=600)
        job["translated_text"] = "private transcript"
        record = performance_record(job_dir, job)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["duration_seconds"], 60)
        self.assertEqual(record["total_seconds"], 600)
        self.assertEqual(record["stage_seconds"]["TTS"], 44.0)
        self.assertNotIn("translated_text", record)
        self.assertNotIn("private transcript", json.dumps(record))

    def test_terminal_job_is_written_only_once(self) -> None:
        job_dir, job = self._job(2, duration=30, total=300)
        record_terminal_job(job_dir, job)
        record_terminal_job(job_dir, job)
        path = self.workdir / "_telemetry" / "performance.jsonl"
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_history_bootstraps_and_estimates_similar_job(self) -> None:
        for index, total in enumerate((270, 290, 300, 310, 330, 350), start=10):
            self._job(index, duration=30, total=total)
        history = PerformanceHistory(self.workdir, refresh_seconds=0)
        estimate = history.estimate({"mode": "dub", "tts_provider": "moss", "quota_duration_ms": 30_000})
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertGreaterEqual(estimate.sample_count, 5)
        self.assertLess(estimate.low_seconds, estimate.seconds)
        self.assertGreater(estimate.high_seconds, estimate.seconds)
        self.assertGreater(estimate.seconds, 250)
        self.assertLess(estimate.seconds, 360)

    def test_failed_jobs_are_diagnostic_but_not_eta_samples(self) -> None:
        job_dir, job = self._job(30, duration=30, total=300)
        job["status"] = "failed"
        job["error"] = "TimeoutError: too slow"
        (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        record = performance_record(job_dir, job)
        assert record is not None
        self.assertEqual(record["error_type"], "TimeoutError")
        record_terminal_job(job_dir, job)
        history = PerformanceHistory(self.workdir, refresh_seconds=0)
        self.assertEqual(history.sample_count, 0)

    def test_recovery_timestamp_after_start_does_not_make_negative_queue(self) -> None:
        job_dir, job = self._job(31, duration=30, total=300)
        job["queued_at"] = job["started_at"] + 100
        record = performance_record(job_dir, job)
        assert record is not None
        self.assertEqual(record["queue_seconds"], 0)
        self.assertEqual(record["total_seconds"], record["processing_seconds"])


if __name__ == "__main__":
    unittest.main()
