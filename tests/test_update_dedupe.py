from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from telegram.ext import ApplicationHandlerStop

from laladub.update_dedupe import (
    RecentActions,
    UpdateDeduplicator,
    build_completion_marker,
    build_replay_guard,
)


class UpdateDeduplicatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.path = Path(self._tempdir.name) / "last_update.json"

    def test_a_fresh_update_is_new(self) -> None:
        self.assertTrue(UpdateDeduplicator(self.path).is_new(10))

    def test_the_same_update_is_not_new_once_it_is_done(self) -> None:
        dedupe = UpdateDeduplicator(self.path)
        self.assertTrue(dedupe.is_new(10))
        dedupe.mark_done(10)
        self.assertFalse(dedupe.is_new(10))

    def test_an_update_still_being_handled_is_not_yet_blocked(self) -> None:
        # The whole point of marking on completion: an update killed
        # mid-handling must come back rather than vanish.
        dedupe = UpdateDeduplicator(self.path)
        self.assertTrue(dedupe.is_new(10))
        self.assertTrue(dedupe.is_new(10))

    def test_an_older_update_is_refused(self) -> None:
        dedupe = UpdateDeduplicator(self.path)
        dedupe.mark_done(10)
        self.assertFalse(dedupe.is_new(9))

    def test_a_newer_update_passes(self) -> None:
        dedupe = UpdateDeduplicator(self.path)
        dedupe.mark_done(10)
        self.assertTrue(dedupe.is_new(11))

    def test_the_id_survives_a_restart(self) -> None:
        # The replay arrives in a *new process*, so an in-memory set would
        # let it straight through.
        UpdateDeduplicator(self.path).mark_done(10)
        self.assertFalse(UpdateDeduplicator(self.path).is_new(10))

    def test_a_missing_file_starts_from_scratch(self) -> None:
        self.assertTrue(UpdateDeduplicator(self.path).is_new(1))

    def test_a_corrupt_file_starts_from_scratch(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertTrue(UpdateDeduplicator(self.path).is_new(1))

    def test_an_update_without_an_id_is_always_let_through(self) -> None:
        dedupe = UpdateDeduplicator(self.path)
        self.assertTrue(dedupe.is_new(None))
        dedupe.mark_done(None)
        self.assertTrue(dedupe.is_new(None))


class ReplayGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dedupe = UpdateDeduplicator(Path(self._tempdir.name) / "last_update.json")
        self.guard = build_replay_guard(self.dedupe)
        self.mark = build_completion_marker(self.dedupe)

    async def test_a_new_update_is_passed_on(self) -> None:
        await self.guard(SimpleNamespace(update_id=5), None)

    async def test_a_finished_update_stops_the_chain_on_replay(self) -> None:
        # Regression: a hard restart replayed an unacknowledged /show and the
        # video was sent again, once per restart - seventeen times in a week.
        await self.guard(SimpleNamespace(update_id=5), None)
        await self.mark(SimpleNamespace(update_id=5), None)
        with self.assertRaises(ApplicationHandlerStop):
            await self.guard(SimpleNamespace(update_id=5), None)

    async def test_an_unfinished_update_is_allowed_back(self) -> None:
        # Killed before the marker ran: better to redo the work than to drop a
        # video someone sent for dubbing.
        await self.guard(SimpleNamespace(update_id=5), None)
        await self.guard(SimpleNamespace(update_id=5), None)

    async def test_a_replay_after_a_restart_stops_the_chain(self) -> None:
        await self.guard(SimpleNamespace(update_id=5), None)
        await self.mark(SimpleNamespace(update_id=5), None)
        restarted = build_replay_guard(UpdateDeduplicator(self.dedupe.path))
        with self.assertRaises(ApplicationHandlerStop):
            await restarted(SimpleNamespace(update_id=5), None)

    async def test_later_updates_still_flow_after_a_replay(self) -> None:
        await self.guard(SimpleNamespace(update_id=5), None)
        await self.mark(SimpleNamespace(update_id=5), None)
        with self.assertRaises(ApplicationHandlerStop):
            await self.guard(SimpleNamespace(update_id=5), None)
        await self.guard(SimpleNamespace(update_id=6), None)


class EditedMessageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dedupe = UpdateDeduplicator(Path(self._tempdir.name) / "last_update.json")
        self.guard = build_replay_guard(self.dedupe)

    async def test_an_edit_is_refused(self) -> None:
        # Regression: someone kept editing one nine-day-old "/show 42456"
        # message. Telegram re-delivers an edit as a fresh update, so the
        # command ran again every time - 25 copies of the video in a group
        # from a command that was only ever typed once.
        edit = SimpleNamespace(
            update_id=7,
            edited_message=SimpleNamespace(message_id=32507, chat=SimpleNamespace(id=-100)),
        )
        with self.assertRaises(ApplicationHandlerStop):
            await self.guard(edit, None)

    async def test_an_edited_channel_post_is_refused_too(self) -> None:
        edit = SimpleNamespace(
            update_id=7,
            edited_message=None,
            edited_channel_post=SimpleNamespace(message_id=1, chat=SimpleNamespace(id=-100)),
        )
        with self.assertRaises(ApplicationHandlerStop):
            await self.guard(edit, None)

    async def test_an_ordinary_message_still_passes(self) -> None:
        fresh = SimpleNamespace(update_id=7, edited_message=None, edited_channel_post=None)
        await self.guard(fresh, None)

    async def test_a_refused_edit_does_not_burn_the_update_id(self) -> None:
        # The edit never counted as handled, so a genuine later update with a
        # higher id must still be accepted normally.
        edit = SimpleNamespace(
            update_id=7,
            edited_message=SimpleNamespace(message_id=1, chat=SimpleNamespace(id=-100)),
        )
        with self.assertRaises(ApplicationHandlerStop):
            await self.guard(edit, None)
        await self.guard(SimpleNamespace(update_id=8, edited_message=None), None)


class RecentActionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.path = Path(self._tempdir.name) / "recent.json"

    def test_an_unseen_key_has_no_age(self) -> None:
        self.assertIsNone(RecentActions(self.path).seconds_since("a"))

    def test_a_recorded_key_reports_its_age(self) -> None:
        recent = RecentActions(self.path, window_seconds=600.0)
        recent.record("a", now=1000.0)
        self.assertAlmostEqual(recent.seconds_since("a", now=1060.0), 60.0)

    def test_a_key_past_the_window_is_forgotten(self) -> None:
        recent = RecentActions(self.path, window_seconds=600.0)
        recent.record("a", now=1000.0)
        self.assertIsNone(recent.seconds_since("a", now=2000.0))

    def test_records_survive_a_restart(self) -> None:
        # This is what makes it useful: the replay arrives in a new process.
        RecentActions(self.path, window_seconds=600.0).record("a", now=1000.0)
        reopened = RecentActions(self.path, window_seconds=600.0)
        self.assertIsNotNone(reopened.seconds_since("a", now=1060.0))

    def test_a_corrupt_file_starts_empty(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(RecentActions(self.path).seconds_since("a"))

    def test_only_the_newest_entries_are_kept(self) -> None:
        recent = RecentActions(self.path, window_seconds=600.0, keep=5)
        for n in range(20):
            recent.record(f"k{n}", now=1000.0 + n)
        reopened = RecentActions(self.path, window_seconds=600.0)
        self.assertIsNone(reopened.seconds_since("k0", now=1000.0))
        self.assertIsNotNone(reopened.seconds_since("k19", now=1020.0))


if __name__ == "__main__":
    unittest.main()
