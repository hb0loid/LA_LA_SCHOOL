from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.bot import _find_proposal_video_path
from laladub.karma import (
    format_karma_milli,
    karma_milli_for_duration,
    level_for_karma,
    visible_karma,
)
from laladub.proposal_bot import (
    _author_caption,
    _find_submission_subtitles,
    _karma_tag,
    _level_up_message,
    _moderation_caption,
    _moderation_keyboard,
)
from laladub.proposal_store import ProposalStore, Submission


class ProposalStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ProposalStore(Path(self.tempdir.name) / "proposal.sqlite3")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_submission_is_idempotent_and_karma_is_adjusted(self) -> None:
        submission, created = self.store.create_submission(
            job_number="38598",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username="test_user",
            video_path=Path("video.mp4"),
            output_filename="video_lalaschool.mp4",
        )
        self.assertTrue(created)
        duplicate, created_again = self.store.create_submission(
            job_number="38598",
            user_id=123,
            chat_id=123,
            author_name="Новое имя",
            author_username="new_user",
            video_path=Path("video.mp4"),
            output_filename="video_lalaschool.mp4",
        )
        self.assertFalse(created_again)
        self.assertEqual(duplicate.id, submission.id)
        self.assertEqual(duplicate.author_name, "Новое имя")

        self.assertIsNotNone(self.store.try_claim(submission.id, 631551040))
        _updated, delta = self.store.finish_decision(
            submission_id=submission.id,
            moderator_id=631551040,
            destination="shame",
            award_milli=800,
            publication_chat_id=-1001,
            publication_message_id=10,
        )
        self.assertEqual(delta, 800)

        self.assertIsNotNone(self.store.try_claim(submission.id, 631551040))
        updated, delta = self.store.finish_decision(
            submission_id=submission.id,
            moderator_id=631551040,
            destination="main",
            award_milli=4_000,
            publication_chat_id=-1002,
            publication_message_id=11,
        )
        self.assertEqual(delta, 3_200)
        self.assertEqual(updated.karma_milli, 4_000)
        self.assertEqual(self.store.karma_total(123), 4_000)
        self.assertEqual(self.store.karma_summary(123), (4_000, 2))
        self.assertEqual(self.store.karma_users(), [123])

    def test_author_message_outbox(self) -> None:
        submission, _created = self.store.create_submission(
            job_number="1",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=Path("video.mp4"),
            output_filename="video.mp4",
        )
        self.store.enqueue_author_message(submission.id, 631551040, "Поправь начало")
        pending = self.store.pending_author_messages()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["job_number"], "1")
        self.store.finish_author_message(pending[0]["id"])
        self.assertEqual(self.store.pending_author_messages(), [])

    def test_daily_usage_is_atomic_and_idempotent(self) -> None:
        accepted, used = self.store.reserve_daily_usage(
            user_id=123, job_number="1", day_key="2026-08-15", duration_ms=200_000, limit_ms=300_000
        )
        self.assertTrue(accepted)
        self.assertEqual(used, 200_000)
        accepted_again, used_again = self.store.reserve_daily_usage(
            user_id=123, job_number="1", day_key="2026-08-15", duration_ms=200_000, limit_ms=300_000
        )
        self.assertTrue(accepted_again)
        self.assertEqual(used_again, 200_000)
        rejected, used_after = self.store.reserve_daily_usage(
            user_id=123, job_number="2", day_key="2026-08-15", duration_ms=101_000, limit_ms=300_000
        )
        self.assertFalse(rejected)
        self.assertEqual(used_after, 200_000)


class ProposalUiTests(unittest.TestCase):
    def test_old_job_uses_watermarked_video_for_forced_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            (job_dir / "dubbed.mp4").write_bytes(b"plain")
            watermarked = job_dir / "dubbed_watermarked.mp4"
            watermarked.write_bytes(b"watermarked")
            self.assertEqual(_find_proposal_video_path(job_dir, {"status": "done"}), watermarked)

    def test_stale_saved_proposal_path_falls_back_to_old_job_output(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            fallback = job_dir / "dubbed.mp4"
            fallback.write_bytes(b"video")
            selected = _find_proposal_video_path(
                job_dir,
                {"proposal_video_path": str(job_dir / "missing.mp4")},
            )
            self.assertEqual(selected, fallback)

    def test_author_caption_is_clickable_and_escaped(self) -> None:
        submission = Submission(
            id=1,
            job_number="2",
            user_id=123,
            chat_id=123,
            author_name="макщ <3",
            author_username="hboloid",
            video_path="video.mp4",
            output_filename="video.mp4",
            status="pending",
            destination=None,
            karma_milli=0,
            duration_ms=0,
            publication_chat_id=None,
            publication_message_id=None,
            created_at=0,
            updated_at=0,
        )
        caption = _author_caption(submission)
        self.assertEqual(caption, 'Прислал <a href="https://t.me/hboloid">макщ &lt;3</a>')

    def test_moderation_buttons_have_expected_order(self) -> None:
        keyboard = _moderation_keyboard(7).inline_keyboard
        labels = [[button.text for button in row] for row in keyboard]
        self.assertEqual(
            labels,
            [
                ["В La La School", "В Ghien Mi Go"],
                ["Передать сообщение"],
                ["Посмотреть субтитры"],
            ],
        )

    def test_subtitle_button_prefers_translated_srt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            work_dir = job_dir / "work"
            work_dir.mkdir()
            translated = work_dir / "translated.srt"
            translated.write_text("1\n00:00:00,000 --> 00:00:01,000\nТест\n", encoding="utf-8")
            submission = Submission(
                id=1,
                job_number="2",
                user_id=123,
                chat_id=123,
                author_name="Тест",
                author_username=None,
                video_path=str(job_dir / "dubbed.mp4"),
                output_filename="dubbed.mp4",
                status="pending",
                destination=None,
                karma_milli=0,
                duration_ms=0,
                publication_chat_id=None,
                publication_message_id=None,
                created_at=0,
                updated_at=0,
            )
            self.assertEqual(_find_submission_subtitles(submission), translated)

    def test_published_caption_is_marked_with_check(self) -> None:
        submission = Submission(
            id=1,
            job_number="2",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path="dubbed.mp4",
            output_filename="dubbed.mp4",
            status="published",
            destination="main",
            karma_milli=1_500,
            duration_ms=15_000,
            publication_chat_id=-1001,
            publication_message_id=5,
            created_at=0,
            updated_at=0,
        )
        self.assertIn("Статус: ✅ Опубликовано", _moderation_caption(submission))
        self.assertIn("Решение: La La School", _moderation_caption(submission))

    def test_level_up_message_lists_new_privileges(self) -> None:
        message = _level_up_message(5_900, 6_100)
        self.assertIsNotNone(message)
        self.assertIn("Автор", message)
        self.assertIn("5 минут", message)
        self.assertIn("Приоритет в очереди: +1", message)
        self.assertIn("Лимит задач в очереди: 1", message)
        self.assertIsNone(_level_up_message(6_100, 6_900))
        self.assertIsNone(_level_up_message(60_100, 6_100))

    def test_karma_tag_fits_telegram_limit(self) -> None:
        self.assertEqual(_karma_tag(6_999), "Карма: 6")
        self.assertLessEqual(len(_karma_tag(12345678901234567890)), 16)


class KarmaRulesTests(unittest.TestCase):
    def test_duration_awards_accumulate_fractionally(self) -> None:
        awards = [karma_milli_for_duration(seconds * 1000, "main") for seconds in (15, 25, 40)]
        self.assertEqual(awards, [1_500, 2_500, 4_000])
        self.assertEqual(visible_karma(sum(awards[:1])), 1)
        self.assertEqual(visible_karma(sum(awards[:2])), 4)
        self.assertEqual(visible_karma(sum(awards)), 8)

    def test_shame_channel_is_five_times_lower(self) -> None:
        self.assertEqual(karma_milli_for_duration(15_000, "shame"), 300)
        self.assertEqual(karma_milli_for_duration(25_000, "shame"), 500)
        self.assertEqual(karma_milli_for_duration(40_000, "shame"), 800)

    def test_levels_use_visible_whole_karma(self) -> None:
        self.assertEqual(level_for_karma(5_999).name, "Участник")
        self.assertEqual(level_for_karma(6_000).name, "Автор")
        thresholds = (0, 6_000, 60_000, 360_000, 720_000, 1_440_000, 2_880_000)
        self.assertEqual([level_for_karma(value).daily_minutes for value in thresholds], [1, 5, 10, 15, 20, 25, 30])
        self.assertEqual([level_for_karma(value).queue_limit for value in thresholds], [1, 1, 2, 2, 3, 3, 3])
        self.assertEqual(format_karma_milli(1_234), "1,234")


if __name__ == "__main__":
    unittest.main()
