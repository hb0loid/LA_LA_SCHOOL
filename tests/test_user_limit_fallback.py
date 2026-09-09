from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from laladub.bot import _JobScheduler
from laladub.bot_config import load_bot_settings


def _scheduler(root: Path) -> _JobScheduler:
    settings = replace(
        load_bot_settings(require_token=False),
        workdir=root,
        executor_mode="hybrid",
        max_active_jobs=2,
        max_local_jobs=1,
        max_active_jobs_per_user=1,
    )
    return _JobScheduler(settings)


def _item(user_id: int, sequence: int, *, preprocessed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        user_id=user_id,
        sequence=sequence,
        premium=False,
        job={
            "mode": "dub",
            "tts_provider": "moss",
            "target_lang": "ru",
            "remote_preprocess_completed_at": 1.0 if preprocessed else None,
        },
    )


class UserLimitFallbackTests(unittest.TestCase):
    """The one-job-per-person rule exists so one author cannot hold both
    machines while others wait. It was never meant to leave a machine idle when
    there is work sitting right there - which is what it did whenever the queue
    was one person deep."""

    def _run(self, build) -> object:
        async def go():
            with tempfile.TemporaryDirectory() as tempdir:
                scheduler = _scheduler(Path(tempdir))
                build(scheduler)
                async with scheduler._lock:
                    return scheduler._next_startable_index(execution_kind="local")

        return asyncio.run(go())

    def test_a_second_job_of_the_same_person_is_taken_when_nothing_else_can_be(self) -> None:
        def build(scheduler):
            scheduler._pending = [(100, 1, _item(7, 1))]
            scheduler._active_by_user[7] = 1
            scheduler._active_total = 1

        self.assertEqual(self._run(build), 0)

    def test_someone_else_is_always_preferred(self) -> None:
        def build(scheduler):
            scheduler._pending = [(100, 1, _item(7, 1)), (100, 2, _item(8, 2))]
            scheduler._active_by_user[7] = 1
            scheduler._active_total = 1

        # Index 1 is the other person's job, even though 7's came first.
        self.assertEqual(self._run(build), 1)

    def test_the_overall_slot_limit_still_holds(self) -> None:
        """Relaxing one rule must not quietly relax the other."""

        def build(scheduler):
            scheduler._pending = [(100, 1, _item(7, 1))]
            scheduler._active_by_user[7] = 1
            scheduler._active_total = 2

        self.assertIsNone(self._run(build))

    def test_a_break_still_stops_this_machine(self) -> None:
        from laladub.bot import set_break

        def build(scheduler):
            scheduler._pending = [(100, 1, _item(7, 1))]
            set_break(scheduler._settings, float("inf"))

        self.assertIsNone(self._run(build))


if __name__ == "__main__":
    unittest.main()
