from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from laladub.bot import _find_proposal_video_path
from laladub.karma import (
    format_karma_milli,
    karma_milli_for_duration,
    level_for_karma,
    visible_karma,
)
from laladub.proposal_bot import (
    _author_caption,
    _find_submission_original_video,
    _find_submission_subtitles,
    _karma_tag,
    _level_up_message,
    _moderation_caption,
    _moderation_keyboard,
    _transcript_text_for_submission,
    comment_on_channel_forward,
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

    def test_karma_before_milli_is_captured_once_at_first_submission(self) -> None:
        submission, created = self.store.create_submission(
            job_number="1",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=Path("video.mp4"),
            output_filename="video.mp4",
            karma_before_milli=6_000,
        )
        self.assertTrue(created)
        self.assertEqual(submission.karma_before_milli, 6_000)

        # Resubmitting the same job_number/user_id updates other fields but
        # must not overwrite the karma snapshot from the original submission.
        duplicate, created_again = self.store.create_submission(
            job_number="1",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=Path("video.mp4"),
            output_filename="video.mp4",
            karma_before_milli=9_000,
        )
        self.assertFalse(created_again)
        self.assertEqual(duplicate.karma_before_milli, 6_000)

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
            user_id=123, job_number="1", duration_ms=200_000, limit_ms=300_000
        )
        self.assertTrue(accepted)
        self.assertEqual(used, 200_000)
        accepted_again, used_again = self.store.reserve_daily_usage(
            user_id=123, job_number="1", duration_ms=200_000, limit_ms=300_000
        )
        self.assertTrue(accepted_again)
        self.assertEqual(used_again, 200_000)
        rejected, used_after = self.store.reserve_daily_usage(
            user_id=123, job_number="2", duration_ms=101_000, limit_ms=300_000
        )
        self.assertFalse(rejected)
        self.assertEqual(used_after, 200_000)

    def test_daily_usage_is_a_rolling_24h_window_not_a_calendar_day(self) -> None:
        day = 86400.0
        now = 1_800_000_000.0
        # Usage from just over 24h ago must not count anymore...
        self.store.reserve_daily_usage(
            user_id=1, job_number="old", duration_ms=250_000, limit_ms=300_000, now=now - day - 1
        )
        self.assertEqual(self.store.daily_usage_ms(1, now=now), 0)
        self.assertIsNone(self.store.next_usage_free_at(1, now=now))
        # ...but usage from just under 24h ago still does.
        self.store.reserve_daily_usage(
            user_id=2, job_number="recent", duration_ms=250_000, limit_ms=300_000, now=now - day + 1
        )
        self.assertEqual(self.store.daily_usage_ms(2, now=now), 250_000)
        self.assertAlmostEqual(self.store.next_usage_free_at(2, now=now), now + 1, delta=1)
        rejected, _used = self.store.reserve_daily_usage(
            user_id=2, job_number="recent-2", duration_ms=60_000, limit_ms=300_000, now=now
        )
        self.assertFalse(rejected)
        # Once that old usage ages out (at now + 1, i.e. 24h after it happened), the
        # same request fits again.
        accepted, used = self.store.reserve_daily_usage(
            user_id=2, job_number="recent-2", duration_ms=60_000, limit_ms=300_000, now=now + 2
        )
        self.assertTrue(accepted)
        self.assertEqual(used, 60_000)

    def test_users_with_recent_usage(self) -> None:
        day = 86400.0
        now = 1_800_000_000.0
        self.store.reserve_daily_usage(user_id=1, job_number="a", duration_ms=1_000, limit_ms=999_999, now=now)
        self.store.reserve_daily_usage(
            user_id=2, job_number="b", duration_ms=1_000, limit_ms=999_999, now=now - day - 10
        )
        self.assertEqual(self.store.users_with_recent_usage(now - day), [1])

    def test_find_by_publication_matches_exact_chat_and_message(self) -> None:
        submission, _created = self.store.create_submission(
            job_number="1",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=Path("video.mp4"),
            output_filename="video.mp4",
        )
        self.store.try_claim(submission.id, 631551040)
        self.store.finish_decision(
            submission_id=submission.id,
            moderator_id=631551040,
            destination="main",
            award_milli=100,
            publication_chat_id=-1001,
            publication_message_id=55,
        )
        found = self.store.find_by_publication(-1001, 55)
        self.assertIsNotNone(found)
        self.assertEqual(found.id, submission.id)
        self.assertIsNone(self.store.find_by_publication(-1001, 999))

    def test_mark_comment_posted_is_idempotent(self) -> None:
        submission, _created = self.store.create_submission(
            job_number="1",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=Path("video.mp4"),
            output_filename="video.mp4",
        )
        self.assertTrue(self.store.mark_comment_posted(submission.id))
        self.assertFalse(self.store.mark_comment_posted(submission.id))


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
            karma_before_milli=0,
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

    def _make_submission_for_subtitles(self, job_dir: Path) -> Submission:
        return Submission(
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
            karma_before_milli=0,
            duration_ms=0,
            publication_chat_id=None,
            publication_message_id=None,
            created_at=0,
            updated_at=0,
        )

    def test_subtitle_button_prefers_plain_text_transcript_over_srt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            work_dir = job_dir / "work"
            work_dir.mkdir()
            (work_dir / "translated.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nТест\n", encoding="utf-8"
            )
            transcript = job_dir / "video_transcript_lalaschool.txt"
            transcript.write_text("Тест\n", encoding="utf-8")
            submission = self._make_submission_for_subtitles(job_dir)
            self.assertEqual(_find_submission_subtitles(submission), transcript)

    def test_subtitle_button_falls_back_to_srt_when_no_transcript_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            work_dir = job_dir / "work"
            work_dir.mkdir()
            translated = work_dir / "translated.srt"
            translated.write_text("1\n00:00:00,000 --> 00:00:01,000\nТест\n", encoding="utf-8")
            submission = self._make_submission_for_subtitles(job_dir)
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
            karma_before_milli=3_000,
            duration_ms=15_000,
            publication_chat_id=-1001,
            publication_message_id=5,
            created_at=0,
            updated_at=0,
        )
        self.assertIn("Статус: ✅ Опубликовано", _moderation_caption(submission))
        self.assertIn("Решение: La La School", _moderation_caption(submission))
        self.assertIn("Карма на момент отправки: 3", _moderation_caption(submission))

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


class TranscriptTextTests(unittest.TestCase):
    def _submission_for(self, job_dir: Path) -> Submission:
        return Submission(
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
            karma_before_milli=0,
            duration_ms=0,
            publication_chat_id=None,
            publication_message_id=None,
            created_at=0,
            updated_at=0,
        )

    def test_reads_plain_text_transcript_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            (job_dir / "video_transcript_lalaschool.txt").write_text("  Привет мир  \n", encoding="utf-8")
            text = _transcript_text_for_submission(self._submission_for(job_dir))
            self.assertEqual(text, "Привет мир")

    def test_strips_timestamps_when_only_srt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            work_dir = job_dir / "work"
            work_dir.mkdir()
            (work_dir / "translated.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nПривет\n\n2\n00:00:01,000 --> 00:00:02,000\nмир\n",
                encoding="utf-8",
            )
            text = _transcript_text_for_submission(self._submission_for(job_dir))
            self.assertEqual(text, "Привет мир")

    def test_returns_none_when_nothing_found(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            text = _transcript_text_for_submission(self._submission_for(Path(tempdir)))
            self.assertIsNone(text)


class _SendMessageSpy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> None:
        self.calls.append(kwargs)


class CommentOnChannelForwardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")

    def _publish(self, *, job_dir: Path) -> Submission:
        submission, _created = self.store.create_submission(
            job_number="1",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=job_dir / "dubbed.mp4",
            output_filename="dubbed.mp4",
        )
        self.store.try_claim(submission.id, 631551040)
        updated, _delta = self.store.finish_decision(
            submission_id=submission.id,
            moderator_id=631551040,
            destination="main",
            award_milli=100,
            publication_chat_id=-1001,
            publication_message_id=55,
        )
        return updated

    def _context(self, send_spy: _SendMessageSpy, video_spy: _SendMessageSpy | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"store": self.store}),
            bot=SimpleNamespace(send_message=send_spy, send_video=video_spy or _SendMessageSpy()),
        )

    async def test_ignores_non_automatic_forward_messages(self) -> None:
        send_spy = _SendMessageSpy()
        update = SimpleNamespace(
            effective_message=SimpleNamespace(is_automatic_forward=False, forward_origin=None)
        )
        await comment_on_channel_forward(update, self._context(send_spy))
        self.assertEqual(send_spy.calls, [])

    async def test_ignores_forward_with_no_matching_submission(self) -> None:
        send_spy = _SendMessageSpy()
        message = SimpleNamespace(
            is_automatic_forward=True,
            forward_origin=SimpleNamespace(chat=SimpleNamespace(id=-1001), message_id=999),
            chat_id=-2002,
            message_id=7,
        )
        update = SimpleNamespace(effective_message=message)
        await comment_on_channel_forward(update, self._context(send_spy))
        self.assertEqual(send_spy.calls, [])

    async def test_posts_transcript_as_reply_on_matching_forward(self) -> None:
        with tempfile.TemporaryDirectory() as job_tempdir:
            job_dir = Path(job_tempdir)
            (job_dir / "video_transcript_lalaschool.txt").write_text("Привет мир", encoding="utf-8")
            self._publish(job_dir=job_dir)

            send_spy = _SendMessageSpy()
            message = SimpleNamespace(
                is_automatic_forward=True,
                forward_origin=SimpleNamespace(chat=SimpleNamespace(id=-1001), message_id=55),
                chat_id=-2002,
                message_id=7,
            )
            update = SimpleNamespace(effective_message=message)
            await comment_on_channel_forward(update, self._context(send_spy))

            self.assertEqual(len(send_spy.calls), 1)
            call = send_spy.calls[0]
            self.assertEqual(call["chat_id"], -2002)
            self.assertEqual(call["reply_to_message_id"], 7)
            self.assertEqual(call["text"], "Привет мир")

    async def test_does_not_post_twice_for_the_same_submission(self) -> None:
        with tempfile.TemporaryDirectory() as job_tempdir:
            job_dir = Path(job_tempdir)
            (job_dir / "video_transcript_lalaschool.txt").write_text("Привет мир", encoding="utf-8")
            self._publish(job_dir=job_dir)

            message = SimpleNamespace(
                is_automatic_forward=True,
                forward_origin=SimpleNamespace(chat=SimpleNamespace(id=-1001), message_id=55),
                chat_id=-2002,
                message_id=7,
            )
            update = SimpleNamespace(effective_message=message)

            first_spy = _SendMessageSpy()
            await comment_on_channel_forward(update, self._context(first_spy))
            self.assertEqual(len(first_spy.calls), 1)

            second_spy = _SendMessageSpy()
            await comment_on_channel_forward(update, self._context(second_spy))
            self.assertEqual(second_spy.calls, [])

    def _forward_update(self) -> SimpleNamespace:
        return SimpleNamespace(
            effective_message=SimpleNamespace(
                is_automatic_forward=True,
                forward_origin=SimpleNamespace(chat=SimpleNamespace(id=-1001), message_id=55),
                chat_id=-2002,
                message_id=7,
            )
        )

    async def test_posts_the_original_video_alongside_the_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as job_tempdir:
            job_dir = Path(job_tempdir)
            (job_dir / "video_transcript_lalaschool.txt").write_text("Привет мир", encoding="utf-8")
            (job_dir / "input.mp4").write_bytes(b"x" * 4096)
            self._publish(job_dir=job_dir)

            send_spy, video_spy = _SendMessageSpy(), _SendMessageSpy()
            await comment_on_channel_forward(self._forward_update(), self._context(send_spy, video_spy))

            # The dub is the channel post itself, so only the source goes below it.
            self.assertEqual(len(video_spy.calls), 1)
            video_call = video_spy.calls[0]
            self.assertEqual(video_call["chat_id"], -2002)
            self.assertEqual(video_call["reply_to_message_id"], 7)
            self.assertEqual(video_call["caption"], "Оригинал")
            self.assertEqual(len(send_spy.calls), 1)

    async def test_posts_the_transcript_even_when_no_original_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as job_tempdir:
            job_dir = Path(job_tempdir)
            (job_dir / "video_transcript_lalaschool.txt").write_text("Привет мир", encoding="utf-8")
            self._publish(job_dir=job_dir)

            send_spy, video_spy = _SendMessageSpy(), _SendMessageSpy()
            await comment_on_channel_forward(self._forward_update(), self._context(send_spy, video_spy))

            self.assertEqual(video_spy.calls, [])
            self.assertEqual(len(send_spy.calls), 1)

    async def test_posts_the_original_even_when_no_transcript_exists(self) -> None:
        with tempfile.TemporaryDirectory() as job_tempdir:
            job_dir = Path(job_tempdir)
            (job_dir / "input.mp4").write_bytes(b"x" * 4096)
            self._publish(job_dir=job_dir)

            send_spy, video_spy = _SendMessageSpy(), _SendMessageSpy()
            await comment_on_channel_forward(self._forward_update(), self._context(send_spy, video_spy))

            self.assertEqual(len(video_spy.calls), 1)
            self.assertEqual(send_spy.calls, [])


class _FloodThenSucceedSpy:
    """Fails with Telegram's rate-limit error the first time, then succeeds."""

    def __init__(self, failures: int = 1) -> None:
        self.calls: list[dict] = []
        self._remaining = failures

    async def __call__(self, **kwargs) -> None:
        if self._remaining > 0:
            self._remaining -= 1
            from telegram.error import RetryAfter

            raise RetryAfter(0)
        self.calls.append(kwargs)


class _AlwaysFloodSpy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> None:
        from telegram.error import RetryAfter

        raise RetryAfter(0)


class CommentFloodControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.job_dir = Path(self._tempdir.name) / "job"
        self.job_dir.mkdir()
        (self.job_dir / "video_transcript_lalaschool.txt").write_text("Привет мир", encoding="utf-8")

    def _publish(self) -> Submission:
        submission, _created = self.store.create_submission(
            job_number="1", user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=self.job_dir / "dubbed.mp4", output_filename="dubbed.mp4",
        )
        self.store.try_claim(submission.id, 631551040)
        updated, _delta = self.store.finish_decision(
            submission_id=submission.id, moderator_id=631551040, destination="main",
            award_milli=100, publication_chat_id=-1001, publication_message_id=55,
        )
        return updated

    def _update(self) -> SimpleNamespace:
        return SimpleNamespace(
            effective_message=SimpleNamespace(
                is_automatic_forward=True,
                forward_origin=SimpleNamespace(chat=SimpleNamespace(id=-1001), message_id=55),
                chat_id=-2002,
                message_id=7,
            )
        )

    def _context(self, send_spy: object) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"store": self.store}),
            bot=SimpleNamespace(send_message=send_spy, send_video=_SendMessageSpy()),
        )

    async def test_retries_after_a_rate_limit_instead_of_dropping_the_comment(self) -> None:
        submission = self._publish()
        spy = _FloodThenSucceedSpy(failures=1)
        await comment_on_channel_forward(self._update(), self._context(spy))
        # It went out on the retry rather than being lost to the backlog.
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0]["text"], "Привет мир")
        self.assertFalse(self.store.mark_comment_posted(submission.id))

    async def test_releases_the_claim_when_nothing_could_be_sent(self) -> None:
        submission = self._publish()
        await comment_on_channel_forward(self._update(), self._context(_AlwaysFloodSpy()))
        # Still unclaimed, so a later attempt is not blocked by a comment that
        # never actually appeared.
        self.assertTrue(self.store.mark_comment_posted(submission.id))


class FindOriginalVideoTests(unittest.TestCase):
    def _submission(self, job_dir: Path) -> Submission:
        return Submission(
            id=1,
            job_number="2",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=str(job_dir / "dubbed.mp4"),
            output_filename="dubbed.mp4",
            status="published",
            destination="main",
            karma_milli=0,
            karma_before_milli=0,
            duration_ms=0,
            publication_chat_id=-1001,
            publication_message_id=55,
            created_at=0,
            updated_at=0,
        )

    def test_finds_the_downloaded_source(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            original = job_dir / "input.mp4"
            original.write_bytes(b"x" * 4096)
            self.assertEqual(_find_submission_original_video(self._submission(job_dir)), original)

    def test_ignores_the_trimmed_and_audio_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            job_dir = Path(tempdir)
            (job_dir / "input_daily_trimmed.mp4").write_bytes(b"x" * 4096)
            (job_dir / "input_audio.mp3").write_bytes(b"x" * 4096)
            self.assertIsNone(_find_submission_original_video(self._submission(job_dir)))

    def test_returns_none_when_the_source_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            self.assertIsNone(_find_submission_original_video(self._submission(Path(tempdir))))


if __name__ == "__main__":
    unittest.main()
