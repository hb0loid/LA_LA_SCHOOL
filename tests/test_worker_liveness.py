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
