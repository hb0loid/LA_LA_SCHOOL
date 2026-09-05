from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from laladub.proposal_bot import _parse_interval, _parse_quiet, _schedule_settings_line
from laladub.proposal_store import ProposalStore, quiet_shift


class QuietShiftTests(unittest.TestCase):
    """Nobody wants posts going out at four in the morning."""

    @staticmethod
    def _at(day: int, hour: int) -> float:
        return datetime(2026, 9, day, hour, 30).timestamp()

    def test_a_slot_inside_the_window_moves_to_its_end(self) -> None:
        moved = quiet_shift(self._at(5, 3), (0, 8))
        self.assertEqual(datetime.fromtimestamp(moved).hour, 8)
        self.assertEqual(datetime.fromtimestamp(moved).day, 5)

    def test_a_slot_outside_the_window_is_untouched(self) -> None:
        original = self._at(5, 12)
        self.assertEqual(quiet_shift(original, (0, 8)), original)

    def test_a_window_across_midnight_works(self) -> None:
        """23:00-07:00 is the shape people actually ask for."""
        late = quiet_shift(self._at(5, 23), (23, 7))
        self.assertEqual(datetime.fromtimestamp(late).hour, 7)
        self.assertEqual(datetime.fromtimestamp(late).day, 6)
        early = quiet_shift(self._at(5, 2), (23, 7))
        self.assertEqual(datetime.fromtimestamp(early).day, 5)
        self.assertEqual(datetime.fromtimestamp(early).hour, 7)

    def test_no_window_changes_nothing(self) -> None:
        original = self._at(5, 3)
        self.assertEqual(quiet_shift(original, None), original)

    def test_a_window_covering_the_whole_day_is_ignored(self) -> None:
        original = self._at(5, 3)
        self.assertEqual(quiet_shift(original, (8, 8)), original)


class ParsingTests(unittest.TestCase):
    def test_minutes_unless_hours_are_named(self) -> None:
        self.assertEqual(_parse_interval("15"), 900.0)
        self.assertEqual(_parse_interval("15м"), 900.0)
        self.assertEqual(_parse_interval("1ч"), 3600.0)
        self.assertEqual(_parse_interval("2h"), 7200.0)

    def test_nonsense_is_refused(self) -> None:
        self.assertIsNone(_parse_interval(""))
        self.assertIsNone(_parse_interval("скоро"))
        self.assertIsNone(_parse_interval("0"))

    def test_quiet_hours_accept_the_obvious_forms(self) -> None:
        self.assertEqual(_parse_quiet("00:00-08:00"), (0, 8))
        self.assertEqual(_parse_quiet("0-8"), (0, 8))
        self.assertEqual(_parse_quiet("23-7"), (23, 7))

    def test_quiet_hours_refuse_the_useless_ones(self) -> None:
        self.assertIsNone(_parse_quiet("8-8"))
        self.assertIsNone(_parse_quiet("ночью"))
        self.assertIsNone(_parse_quiet("25-30"))

    def test_the_summary_line_reads_plainly(self) -> None:
        self.assertIn("30 мин", _schedule_settings_line(1800, None))
        self.assertIn("1 ч", _schedule_settings_line(3600, None))
        self.assertIn("00:00–08:00", _schedule_settings_line(1800, (0, 8)))


class StoreScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposals.sqlite3")

    def test_the_interval_defaults_and_persists(self) -> None:
        self.assertEqual(self.store.post_interval_seconds(), 1800)
        self.store.set_post_interval_seconds(900)
        self.assertEqual(self.store.post_interval_seconds(), 900)

    def test_an_absurd_interval_is_clamped(self) -> None:
        self.assertGreaterEqual(self.store.set_post_interval_seconds(1), 60)
        self.assertLessEqual(self.store.set_post_interval_seconds(10**9), 86400)

    def test_quiet_hours_round_trip(self) -> None:
        self.assertIsNone(self.store.quiet_hours())
        self.store.set_quiet_hours((0, 8))
        self.assertEqual(self.store.quiet_hours(), (0, 8))
        self.store.set_quiet_hours(None)
        self.assertIsNone(self.store.quiet_hours())

    def test_rescheduling_respaces_the_whole_queue(self) -> None:
        """Changing the interval only for new posts would leave a queue built
        at the old spacing untouched - the opposite of what was asked for."""
        now = time.time()
        for index in range(4):
            self.store.create_submission(
                job_number=str(5000 + index),
                user_id=1,
                chat_id=1,
                author_name="кто-то",
                author_username=None,
                video_path=Path("video.mp4"),
                output_filename="video.mp4",
            )
            # create_submission returns nothing useful here; ids start at 1.
            self.store.schedule_post(
                submission_id=index + 1,
                destination="main",
                target_chat="@channel",
                moderator_id=1,
                scheduled_for=now + index * 1800,
            )
        moved = self.store.reschedule_pending(interval_seconds=300, quiet=None, now=now)
        self.assertEqual(moved, 3)
        times = [item.scheduled_for for item in self.store.pending_scheduled_posts()]
        gaps = [round(b - a) for a, b in zip(times, times[1:])]
        self.assertEqual(gaps, [300, 300, 300])


if __name__ == "__main__":
    unittest.main()
