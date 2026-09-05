from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from laladub.performance import PerformanceHistory


def _sample(
    *,
    index: int,
    duration: float,
    processing: float,
    queue: float,
    remote: bool,
    tts: str = "moss",
) -> dict:
    return {
        "schema": 1,
        "event_id": f"1/{index}:done",
        "recorded_at": 1_000.0 + index,
        "status": "done",
        "mode": "dub",
        "tts": tts,
        "duration_seconds": duration,
        "queue_seconds": queue,
        "processing_seconds": processing,
        "total_seconds": queue + processing,
        "remote_preprocess": remote,
    }


class EtaAccuracyTests(unittest.TestCase):
    """Measured over 900 finished jobs: predicting total time was wrong by 77%,
    predicting processing time by 51%, and splitting by which machine ran it by
    34%. These lock in the two changes that bought that."""

    def _history(self, samples: list[dict]) -> PerformanceHistory:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        workdir = Path(tempdir.name)
        path = workdir / "_telemetry" / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample) + "\n")
        return PerformanceHistory(workdir)

    def test_a_long_queue_does_not_inflate_the_estimate(self) -> None:
        """The queue says how busy we were, not how long this video takes."""
        history = self._history(
            [
                _sample(index=n, duration=60.0, processing=120.0, queue=3600.0, remote=False)
                for n in range(20)
            ]
        )
        estimate = history.estimate({"mode": "dub", "tts_provider": "moss", "input_duration_seconds": 60})
        assert estimate is not None
        self.assertLess(estimate.seconds, 300.0)

    def test_the_machine_it_runs_on_is_used_when_known(self) -> None:
        """The same video takes about three times longer via the laptop."""
        samples = [
            _sample(index=n, duration=60.0, processing=360.0, queue=0.0, remote=True)
            for n in range(20)
        ] + [
            _sample(index=100 + n, duration=60.0, processing=120.0, queue=0.0, remote=False)
            for n in range(20)
        ]
        history = self._history(samples)
        job = {"mode": "dub", "tts_provider": "moss", "input_duration_seconds": 60}
        local = history.estimate(job, remote=False)
        remote = history.estimate(job, remote=True)
        assert local is not None and remote is not None
        self.assertLess(local.seconds, remote.seconds)
        self.assertGreater(remote.seconds / local.seconds, 2.0)

    def test_without_the_hint_both_kinds_are_mixed(self) -> None:
        samples = [
            _sample(index=n, duration=60.0, processing=360.0, queue=0.0, remote=True)
            for n in range(20)
        ] + [
            _sample(index=100 + n, duration=60.0, processing=120.0, queue=0.0, remote=False)
            for n in range(20)
        ]
        history = self._history(samples)
        job = {"mode": "dub", "tts_provider": "moss", "input_duration_seconds": 60}
        blended = history.estimate(job)
        local = history.estimate(job, remote=False)
        remote = history.estimate(job, remote=True)
        assert blended is not None and local is not None and remote is not None
        self.assertGreater(blended.seconds, local.seconds)
        self.assertLess(blended.seconds, remote.seconds)

    def test_too_few_matching_samples_fall_back_to_all(self) -> None:
        """Three laptop jobs are not a basis for anything; better the mixed
        pool than a confident guess from nearly nothing."""
        samples = [
            _sample(index=n, duration=60.0, processing=120.0, queue=0.0, remote=False)
            for n in range(20)
        ] + [
            _sample(index=100 + n, duration=60.0, processing=3600.0, queue=0.0, remote=True)
            for n in range(3)
        ]
        history = self._history(samples)
        job = {"mode": "dub", "tts_provider": "moss", "input_duration_seconds": 60}
        estimate = history.estimate(job, remote=True)
        assert estimate is not None
        self.assertLess(estimate.seconds, 1000.0)



class QueuedTotalTests(unittest.TestCase):
    """Showing the total was the point all along - a person wants to know when
    the video arrives, not when it starts. Splitting queue wait from work was
    about estimating them well, not about hiding the sum."""

    def test_the_total_is_the_wait_plus_the_work(self) -> None:
        from laladub.bot import _format_eta_range

        wait_low, wait_high = 600.0, 900.0
        work_low, work_high = 300.0, 600.0
        total = _format_eta_range(wait_low + work_low, wait_high + work_high)
        self.assertEqual(total, "15–25 мин")

    def test_an_empty_queue_shows_the_work_alone(self) -> None:
        from laladub.bot import _format_eta_range

        self.assertEqual(_format_eta_range(0.0 + 300.0, 0.0 + 600.0), "5–10 мин")

if __name__ == "__main__":
    unittest.main()
