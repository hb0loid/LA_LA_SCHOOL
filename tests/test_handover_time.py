from __future__ import annotations

import unittest
from pathlib import Path

from laladub.performance import performance_record


def _job(**overrides: object) -> dict:
    base = {
        "status": "done",
        "queued_at": 1000.0,
        "started_at": 1100.0,
        "finished_at": 2603.0,
        "quota_duration_ms": 13_000,
        "mode": "dub",
        "tts_provider": "moss",
    }
    base.update(overrides)
    return base


class HandoverTests(unittest.TestCase):
    """A split job waits again between the laptop finishing preparation and the
    main PC reaching the voice. That is queueing, not work - counting it as work
    made a 13-second video look like 1503 seconds of computation when 1288 of
    them were spent waiting, and the runtime estimate is built from that number.
    """

    def _record(self, job: dict) -> dict:
        record = performance_record(Path("runs") / "123" / "57489", job)
        assert record is not None
        return record

    def test_the_wait_leaves_the_work_and_joins_the_queue(self) -> None:
        record = self._record(
            _job(
                remote_preprocess_completed_at=1207.0,
                local_continuation_started_at=2495.0,
            )
        )
        self.assertEqual(round(record["processing_seconds"]), 1503 - 1288)
        self.assertEqual(round(record["queue_seconds"]), 100 + 1288)
        self.assertEqual(round(record["handover_seconds"]), 1288)

    def test_the_total_still_counts_every_second_waited(self) -> None:
        """The person waited all of it, however it is attributed."""
        record = self._record(
            _job(
                remote_preprocess_completed_at=1207.0,
                local_continuation_started_at=2495.0,
            )
        )
        self.assertEqual(round(record["total_seconds"]), 1603)

    def test_a_job_done_on_one_machine_is_unchanged(self) -> None:
        record = self._record(_job())
        self.assertEqual(round(record["processing_seconds"]), 1503)
        self.assertEqual(round(record["queue_seconds"]), 100)
        self.assertIsNone(record["handover_seconds"])

    def test_a_negative_gap_is_ignored(self) -> None:
        """Clock skew or a resumed job must not inflate the work instead."""
        record = self._record(
            _job(
                remote_preprocess_completed_at=2495.0,
                local_continuation_started_at=1207.0,
            )
        )
        self.assertEqual(round(record["processing_seconds"]), 1503)


if __name__ == "__main__":
    unittest.main()
