from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from laladub.bot import _JobScheduler
from laladub.bot_config import load_bot_settings


def _scheduler(root: Path) -> _JobScheduler:
    settings = replace(load_bot_settings(require_token=False), workdir=root)
    return _JobScheduler(settings)


class WorkerLivenessTests(unittest.TestCase):
    def test_progress_for_a_reclaimed_job_still_counts_the_worker_online(self) -> None:
        """A worker talking to us is alive, whatever job it thinks it is on.

        The coordinator drops a lease after a heartbeat timeout; the worker only
        finds out when it finishes. Every post it made in between was thrown
        away, so a busy worker looked missing for the rest of the job.
        """

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tempdir:
                scheduler = _scheduler(Path(tempdir))
                await scheduler.remote_progress(
                    "no-such-job", {"heartbeat_only": True, "worker_id": "worker-pc"}
                )
                async with scheduler._lock:
                    counts = scheduler._remote_worker_counts_locked()
                self.assertEqual(counts["online"], 1)
                self.assertEqual(counts["busy"], 0)

        asyncio.run(run())

    def test_progress_without_a_worker_id_is_still_ignored(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tempdir:
                scheduler = _scheduler(Path(tempdir))
                await scheduler.remote_progress("no-such-job", {"heartbeat_only": True})
                async with scheduler._lock:
                    counts = scheduler._remote_worker_counts_locked()
                self.assertEqual(counts["online"], 0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()


class WorkerTtlTests(unittest.TestCase):
    def test_a_worker_that_spoke_a_minute_ago_is_still_online(self) -> None:
        """The old thirty-second window was the whole bug.

        A worker inside one long step says nothing but its heartbeat, and the
        coordinator only had to be busy for half a minute to declare it gone.
        """

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tempdir:
                scheduler = _scheduler(Path(tempdir))
                scheduler.note_worker_seen("worker-pc")
                scheduler._remote_workers["worker-pc"]["last_seen"] -= 60.0
                async with scheduler._lock:
                    self.assertEqual(scheduler._remote_worker_counts_locked()["online"], 1)
                scheduler._remote_workers["worker-pc"]["last_seen"] -= 120.0
                async with scheduler._lock:
                    self.assertEqual(scheduler._remote_worker_counts_locked()["online"], 0)

        asyncio.run(run())

    def test_note_worker_seen_needs_no_lock(self) -> None:
        """It is called from the HTTP thread, so it must not wait on anything."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tempdir:
                scheduler = _scheduler(Path(tempdir))
                async with scheduler._lock:
                    scheduler.note_worker_seen("worker-pc")
                    self.assertIn("worker-pc", scheduler._remote_workers)

        asyncio.run(run())
