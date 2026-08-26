from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from telegram.ext import ApplicationHandlerStop

from laladub.update_dedupe import UpdateDeduplicator, build_replay_guard


class UpdateDeduplicatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.path = Path(self._tempdir.name) / "last_update.json"

    def test_a_fresh_update_is_new(self) -> None:
        self.assertTrue(UpdateDeduplicator(self.path).is_new(10))

    def test_the_same_update_is_not_new_twice(self) -> None:
        dedupe = UpdateDeduplicator(self.path)
        self.assertTrue(dedupe.is_new(10))
        self.assertFalse(dedupe.is_new(10))

    def test_an_older_update_is_refused(self) -> None:
        dedupe = UpdateDeduplicator(self.path)
        dedupe.is_new(10)
        self.assertFalse(dedupe.is_new(9))

    def test_a_newer_update_passes(self) -> None:
        dedupe = UpdateDeduplicator(self.path)
        dedupe.is_new(10)
        self.assertTrue(dedupe.is_new(11))

    def test_the_id_survives_a_restart(self) -> None:
        # The whole point: the replay arrives in a *new process*, so an
        # in-memory set would let it straight through.
        UpdateDeduplicator(self.path).is_new(10)
        self.assertFalse(UpdateDeduplicator(self.path).is_new(10))

    def test_a_missing_file_starts_from_scratch(self) -> None:
        self.assertTrue(UpdateDeduplicator(self.path).is_new(1))

    def test_a_corrupt_file_starts_from_scratch(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertTrue(UpdateDeduplicator(self.path).is_new(1))

    def test_an_update_without_an_id_is_always_let_through(self) -> None:
        self.assertTrue(UpdateDeduplicator(self.path).is_new(None))


class ReplayGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dedupe = UpdateDeduplicator(Path(self._tempdir.name) / "last_update.json")
        self.guard = build_replay_guard(self.dedupe)

    async def test_a_new_update_is_passed_on(self) -> None:
        await self.guard(SimpleNamespace(update_id=5), None)

    async def test_a_replayed_update_stops_the_chain(self) -> None:
        # Regression: a hard restart replayed an unacknowledged /show and the
        # video was sent again, once per restart - seventeen times in a week.
        await self.guard(SimpleNamespace(update_id=5), None)
        with self.assertRaises(ApplicationHandlerStop):
            await self.guard(SimpleNamespace(update_id=5), None)

    async def test_a_replay_after_a_restart_stops_the_chain(self) -> None:
        await self.guard(SimpleNamespace(update_id=5), None)
        restarted = build_replay_guard(UpdateDeduplicator(self.dedupe.path))
        with self.assertRaises(ApplicationHandlerStop):
            await restarted(SimpleNamespace(update_id=5), None)

    async def test_later_updates_still_flow_after_a_replay(self) -> None:
        await self.guard(SimpleNamespace(update_id=5), None)
        with self.assertRaises(ApplicationHandlerStop):
            await self.guard(SimpleNamespace(update_id=5), None)
        await self.guard(SimpleNamespace(update_id=6), None)


if __name__ == "__main__":
    unittest.main()
