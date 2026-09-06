from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
import time
import unittest
from datetime import datetime
from pathlib import Path

from laladub.proposal_bot import (
    _parse_interval,
    _parse_quiet,
    _schedule_settings_line,
    _spacing_label,
    _until_label,
)
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

    def test_the_dynamic_mode_says_so(self) -> None:
        self.assertEqual(_spacing_label(1800, "dynamic", 2), "по длине видео ×2")
        self.assertIn("по длине видео ×1.5", _schedule_settings_line(1800, None, "dynamic", 1.5))

    def test_how_far_off_reads_roughly(self) -> None:
        self.assertEqual(_until_label(300), "через 5 мин")
        self.assertEqual(_until_label(7200), "через 2 ч")
        self.assertEqual(_until_label(2 * 86400), "через 2 дн")

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



class DynamicSpacingTests(unittest.TestCase):
    """The pause after a post is that post's own video length times a
    multiplier: a one-minute clip should not hold the channel as long as a
    ten-minute one."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposals.sqlite3")

    def _queue(self, durations_minutes: list[float], start: float) -> None:
        for index, minutes in enumerate(durations_minutes):
            self.store.create_submission(
                job_number=str(6000 + index),
                user_id=1,
                chat_id=1,
                author_name="кто-то",
                author_username=None,
                video_path=Path("video.mp4"),
                output_filename="video.mp4",
                duration_ms=int(minutes * 60_000),
            )
            self.store.schedule_post(
                submission_id=index + 1,
                destination="main",
                target_chat="@channel",
                moderator_id=1,
                scheduled_for=start + index,
            )

    def _slots_minutes(self) -> list[float]:
        times = [item.scheduled_for for item in self.store.pending_scheduled_posts()]
        return [round((t - times[0]) / 60) for t in times]

    def test_the_worked_example_with_multiplier_one(self) -> None:
        """Videos of 1 and 10 minutes: 16:00, 16:01, 16:11."""
        start = datetime(2026, 9, 5, 16, 0).timestamp()
        self._queue([1, 10, 1], start)
        self.store.set_post_mode("dynamic")
        self.store.set_post_multiplier(1)
        self.store.reschedule_pending(interval_seconds=1800, quiet=None, now=start)
        self.assertEqual(self._slots_minutes(), [0, 1, 11])

    def test_the_worked_example_with_multiplier_two(self) -> None:
        """The same videos doubled: 16:00, 16:02, 16:22."""
        start = datetime(2026, 9, 5, 16, 0).timestamp()
        self._queue([1, 10, 1], start)
        self.store.set_post_mode("dynamic")
        self.store.set_post_multiplier(2)
        self.store.reschedule_pending(interval_seconds=1800, quiet=None, now=start)
        self.assertEqual(self._slots_minutes(), [0, 2, 22])

    def test_an_unknown_duration_falls_back_to_the_fixed_interval(self) -> None:
        """Better the old spacing than posting the next one immediately."""
        start = datetime(2026, 9, 5, 16, 0).timestamp()
        self._queue([0, 5], start)
        self.store.set_post_mode("dynamic")
        self.store.set_post_multiplier(1)
        self.store.reschedule_pending(interval_seconds=900, quiet=None, now=start)
        self.assertEqual(self._slots_minutes(), [0, 15])

    def test_quiet_hours_still_win(self) -> None:
        start = datetime(2026, 9, 5, 23, 30).timestamp()
        self._queue([40, 5], start)
        self.store.set_post_mode("dynamic")
        self.store.set_post_multiplier(1)
        self.store.reschedule_pending(interval_seconds=1800, quiet=(0, 8), now=start)
        times = [item.scheduled_for for item in self.store.pending_scheduled_posts()]
        self.assertEqual(datetime.fromtimestamp(times[0]).hour, 23)
        # 23:30 plus forty minutes lands at 00:10, inside the window.
        self.assertEqual(datetime.fromtimestamp(times[1]).hour, 8)

    def test_fixed_mode_is_unaffected_by_duration(self) -> None:
        start = datetime(2026, 9, 5, 16, 0).timestamp()
        self._queue([1, 10, 1], start)
        self.store.reschedule_pending(interval_seconds=600, quiet=None, now=start)
        self.assertEqual(self._slots_minutes(), [0, 10, 20])


class DeliveryPacingTests(unittest.TestCase):
    """The delivery loop paces itself by this. It read the fixed interval even
    in dynamic mode, so posts left every thirty minutes whatever the schedule
    said, and the queue fell 46 posts behind while looking healthy."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposals.sqlite3")

    def _publish(self, duration_minutes: float) -> None:
        self.store.create_submission(
            job_number="7000",
            user_id=1,
            chat_id=1,
            author_name="кто-то",
            author_username=None,
            video_path=Path("video.mp4"),
            output_filename="video.mp4",
            duration_ms=int(duration_minutes * 60_000),
        )
        # Marked published directly: the moderation handshake is not what is
        # under test here, the pacing that follows a published post is.
        # closing() as well as the transaction: on Windows an open handle
        # blocks the temporary directory from being removed.
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute(
                "UPDATE submissions SET status='published', destination='main', "
                "publication_message_id=5, updated_at=? WHERE id=1",
                (time.time(),),
            )

    def test_fixed_mode_uses_the_interval(self) -> None:
        self.store.set_post_interval_seconds(900)
        self._publish(4)
        self.assertEqual(self.store.spacing_after_last_post("main"), 900)

    def test_dynamic_mode_uses_the_published_video(self) -> None:
        self.store.set_post_mode("dynamic")
        self.store.set_post_multiplier(7)
        self._publish(4)
        self.assertEqual(self.store.spacing_after_last_post("main"), 4 * 60 * 7)

    def test_nothing_published_yet_falls_back(self) -> None:
        self.store.set_post_mode("dynamic")
        self.store.set_post_interval_seconds(600)
        self.assertEqual(self.store.spacing_after_last_post("main"), 600)


class OverdueLabelTests(unittest.TestCase):
    def test_a_slot_in_the_past_says_so(self) -> None:
        self.assertEqual(_until_label(-7200), "просрочен на 2 ч")

    def test_a_slot_just_reached_is_not_called_overdue(self) -> None:
        self.assertEqual(_until_label(-5), "через 1 мин")

if __name__ == "__main__":
    unittest.main()
