from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from laladub.bot import _find_proposal_video_path
from laladub.karma import (
    format_karma_milli,
    karma_milli_for_duration,
    level_for_karma,
    visible_karma,
)
from laladub.proposal_bot import (
    DELAYED_POST_INTERVAL_SECONDS,
    ProposalBotSettings,
    _author_caption,
    _find_submission_original_video,
    _find_submission_subtitles,
    _delete_moderator_cards,
    _karma_tag,
    _level_up_message,
    _moderation_caption,
    _moderation_keyboard,
    _process_scheduled_post,
    _transcript_text_for_submission,
    _split_telegram_text,
    clean_command,
    comment_on_channel_forward,
    moderation_callback,
    post_command,
    scheduled_command,
    send_due_scheduled_posts,
    timer_command,
    unpost_command,
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

    def _minimal_submission(self) -> Submission:
        return Submission(
            id=1, job_number="2", user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path="dubbed.mp4", output_filename="dubbed.mp4", status="pending", destination=None,
            karma_milli=0, karma_before_milli=0, duration_ms=0, publication_chat_id=None,
            publication_message_id=None, created_at=0, updated_at=0,
        )

    def test_transcript_is_embedded_as_a_collapsed_quote(self) -> None:
        caption = _moderation_caption(self._minimal_submission(), transcript="Привет мир")
        self.assertIn("<blockquote expandable>Привет мир</blockquote>", caption)

    def test_no_transcript_means_no_blockquote(self) -> None:
        caption = _moderation_caption(self._minimal_submission(), transcript=None)
        self.assertNotIn("<blockquote", caption)

    def test_transcript_is_html_escaped(self) -> None:
        caption = _moderation_caption(self._minimal_submission(), transcript="<script> & друзья")
        self.assertIn("&lt;script&gt; &amp; друзья", caption)
        self.assertNotIn("<script>", caption)

    def test_long_transcript_is_truncated_to_fit_the_caption_limit(self) -> None:
        caption = _moderation_caption(self._minimal_submission(), transcript="ы" * 2000)
        self.assertLessEqual(len(caption), 1024)
        self.assertIn("…</blockquote>", caption)

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

    async def test_long_transcript_is_posted_in_multiple_replies_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as job_tempdir:
            job_dir = Path(job_tempdir)
            transcript = ("длинная строка " * 700).strip()
            (job_dir / "video_transcript_lalaschool.txt").write_text(transcript, encoding="utf-8")
            self._publish(job_dir=job_dir)

            send_spy = _SendMessageSpy()
            await comment_on_channel_forward(self._forward_update(), self._context(send_spy))

            self.assertGreater(len(send_spy.calls), 1)
            self.assertTrue(all(len(call["text"]) <= 4000 for call in send_spy.calls))
            self.assertEqual("".join(call["text"] for call in send_spy.calls), transcript)
            self.assertTrue(all(call["reply_to_message_id"] == 7 for call in send_spy.calls))

    def test_split_telegram_text_hard_splits_a_single_oversized_word(self) -> None:
        text = "я" * 9001
        parts = _split_telegram_text(text)
        self.assertEqual([len(part) for part in parts], [4000, 4000, 1001])
        self.assertEqual("".join(parts), text)

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


class _RaisingAnswer:
    """Simulates Telegram's "Query is too old" BadRequest on query.answer()."""

    async def __call__(self, *args: object, **kwargs: object) -> None:
        from telegram.error import BadRequest

        raise BadRequest("Query is too old and response timeout expired or query id is invalid")


class ModerationCallbackClaimReleaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.settings = ProposalBotSettings(
            token="x",
            database=Path(self._tempdir.name) / "proposal.sqlite3",
            moderator_ids=frozenset({631551040}),
            main_channel="@elevenlabss",
            shame_channel="@ghienmigo",
            karma_chat="@lalaschoo",
        )

    def _submission(self) -> Submission:
        submission, _created = self.store.create_submission(
            job_number="1", user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / "dubbed.mp4", output_filename="dubbed.mp4",
        )
        return submission

    def _context(self) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": self.settings, "store": self.store}),
            bot=SimpleNamespace(),
        )

    async def test_a_failed_answer_does_not_leave_the_claim_stuck(self) -> None:
        # Regression test: query.answer() used to run outside the try/except
        # that releases the claim, so a stale-callback BadRequest from
        # Telegram left the submission locked as "being processed" for the
        # full 10-minute staleness window with no decision ever applied.
        submission = self._submission()
        query = SimpleNamespace(
            data=f"mod:reject:{submission.id}",
            answer=_RaisingAnswer(),
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=631551040))
        await moderation_callback(update, self._context())

        updated = self.store.get_submission(submission.id)
        self.assertEqual(updated.destination, "rejected")
        # The claim must be released immediately, not left to expire after the
        # 10-minute staleness window - a second moderator can act right away.
        self.assertIsNotNone(self.store.try_claim(submission.id, 999))


class DelayedPostingStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")

    def _submission(self, job_number: str) -> Submission:
        submission, _created = self.store.create_submission(
            job_number=job_number, user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / f"{job_number}.mp4", output_filename="dubbed.mp4",
        )
        return submission

    def test_disabled_by_default(self) -> None:
        self.assertFalse(self.store.delayed_posting_enabled())

    def test_toggle_persists(self) -> None:
        self.store.set_delayed_posting_enabled(True)
        self.assertTrue(self.store.delayed_posting_enabled())
        self.store.set_delayed_posting_enabled(False)
        self.assertFalse(self.store.delayed_posting_enabled())

    def test_next_slot_is_now_when_nothing_scheduled(self) -> None:
        now = 1_000_000.0
        self.assertEqual(self.store.next_available_slot("main", interval_seconds=1800, now=now), now)

    def test_next_slot_chains_after_a_pending_schedule(self) -> None:
        now = 1_000_000.0
        submission = self._submission("1")
        self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=now + 900,
        )
        # A request arriving before that slot fires chains onto +interval, not
        # onto "now" - this is what keeps two posts from ever sharing a slot.
        slot = self.store.next_available_slot("main", interval_seconds=1800, now=now)
        self.assertEqual(slot, now + 900 + 1800)

    def test_next_slot_is_now_when_the_last_post_is_old_enough(self) -> None:
        now = 1_000_000.0
        submission = self._submission("1")
        self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=now - 4000,
        )
        self.assertEqual(self.store.next_available_slot("main", interval_seconds=1800, now=now), now)

    def test_channels_are_spaced_independently(self) -> None:
        now = 1_000_000.0
        submission = self._submission("1")
        self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=now + 900,
        )
        self.assertEqual(self.store.next_available_slot("shame", interval_seconds=1800, now=now), now)

    def test_cancel_is_idempotent_and_frees_the_slot(self) -> None:
        submission = self._submission("1")
        item = self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=1_000_000.0,
        )
        self.assertEqual(len(self.store.pending_scheduled_posts()), 1)
        self.assertIsNotNone(self.store.cancel_scheduled_post(item.id))
        self.assertEqual(self.store.pending_scheduled_posts(), [])
        self.assertIsNone(self.store.cancel_scheduled_post(item.id))

    def test_due_scheduled_posts_only_returns_items_at_or_past_their_slot(self) -> None:
        first = self._submission("1")
        second = self._submission("2")
        self.store.schedule_post(
            submission_id=first.id, destination="main", target_chat="@x", moderator_id=1, scheduled_for=1_000_000.0
        )
        self.store.schedule_post(
            submission_id=second.id, destination="main", target_chat="@x", moderator_id=1, scheduled_for=2_000_000.0
        )
        due = self.store.due_scheduled_posts(now=1_500_000.0)
        self.assertEqual([item.submission_id for item in due], [first.id])

    def test_pending_schedule_for_submission(self) -> None:
        submission = self._submission("1")
        self.assertIsNone(self.store.pending_schedule_for_submission(submission.id))
        self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=1_000_000.0,
        )
        self.assertIsNotNone(self.store.pending_schedule_for_submission(submission.id))


class BotNotesAndCleanupStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")

    def _submission(self, job_number: str) -> Submission:
        submission, _created = self.store.create_submission(
            job_number=job_number, user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / f"{job_number}.mp4", output_filename="dubbed.mp4",
        )
        return submission

    def test_forgetting_a_scheduled_submissions_message_does_not_make_it_look_new(self) -> None:
        # Regression: /clean forgets a scheduled post's tracking row so it no
        # longer needs live decision buttons - but the submission is still
        # 'pending' under the hood (the decision applies only when it
        # actually posts). Without this guard, the 5s delivery loop would
        # see "pending, no tracked message" and resend it as if brand new,
        # over and over, every time /clean ran.
        submission = self._submission("1")
        self.store.record_moderator_message(submission.id, 631551040, -100, 10)
        self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=631551040,
            scheduled_for=1_000_000.0,
        )
        self.store.forget_moderator_message(submission.id, 631551040)
        self.assertEqual(self.store.pending_for_moderator(631551040), [])

    def test_take_bot_notes_returns_and_clears_them(self) -> None:
        self.store.record_bot_note(631551040, -100, 1)
        self.store.record_bot_note(631551040, -100, 2)
        self.store.record_bot_note(999, -100, 3)  # a different moderator - untouched
        notes = self.store.take_bot_notes(631551040)
        self.assertEqual(sorted(notes), [(-100, 1), (-100, 2)])
        self.assertEqual(self.store.take_bot_notes(631551040), [])
        self.assertEqual(self.store.take_bot_notes(999), [(-100, 3)])

    def test_resolved_moderator_video_messages_excludes_pending(self) -> None:
        pending_submission = self._submission("1")
        self.store.record_moderator_message(pending_submission.id, 631551040, -100, 10)

        rejected_submission = self._submission("2")
        self.store.try_claim(rejected_submission.id, 631551040)
        self.store.finish_decision(
            submission_id=rejected_submission.id, moderator_id=631551040, destination="rejected", award_milli=0,
            publication_chat_id=None, publication_message_id=None,
        )
        self.store.record_moderator_message(rejected_submission.id, 631551040, -100, 20)

        resolved = self.store.resolved_moderator_video_messages(631551040)
        self.assertEqual(resolved, [(rejected_submission.id, -100, 20)])

    def test_resolved_moderator_video_messages_includes_scheduled(self) -> None:
        # A submission sitting in the delayed queue is still 'pending' (the
        # decision hasn't actually been applied yet), but it no longer shows
        # live buttons, so /clean should sweep it up too.
        scheduled_submission = self._submission("1")
        self.store.record_moderator_message(scheduled_submission.id, 631551040, -100, 10)
        self.store.schedule_post(
            submission_id=scheduled_submission.id, destination="main", target_chat="@x", moderator_id=631551040,
            scheduled_for=1_000_000.0,
        )

        still_pending_submission = self._submission("2")
        self.store.record_moderator_message(still_pending_submission.id, 631551040, -100, 20)

        resolved = self.store.resolved_moderator_video_messages(631551040)
        self.assertEqual(resolved, [(scheduled_submission.id, -100, 10)])

    def test_forget_moderator_message_removes_the_tracking_row(self) -> None:
        submission = self._submission("1")
        self.store.record_moderator_message(submission.id, 631551040, -100, 10)
        self.store.forget_moderator_message(submission.id, 631551040)
        self.assertEqual(self.store.moderator_messages(submission.id), [])


class _FakePublishedMessage:
    def __init__(self, chat_id: object, message_id: int) -> None:
        self.chat_id = chat_id
        self.message_id = message_id


class ModerationCallbackDelayedPostingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.store.set_delayed_posting_enabled(True)
        self.settings = ProposalBotSettings(
            token="x",
            database=Path(self._tempdir.name) / "proposal.sqlite3",
            moderator_ids=frozenset({631551040}),
            main_channel="@elevenlabss",
            shame_channel="@ghienmigo",
            karma_chat="@lalaschoo",
        )

    def _submission(self, job_number: str) -> Submission:
        submission, _created = self.store.create_submission(
            job_number=job_number, user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / f"{job_number}.mp4", output_filename="dubbed.mp4",
        )
        return submission

    def _context(self) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": self.settings, "store": self.store}),
            bot=SimpleNamespace(edit_message_caption=AsyncMock()),
        )

    def _click(self, destination_action: str, submission: Submission) -> tuple[SimpleNamespace, SimpleNamespace]:
        query = SimpleNamespace(
            data=f"mod:{destination_action}:{submission.id}",
            answer=AsyncMock(),
            message=SimpleNamespace(reply_text=AsyncMock(), text="video"),
        )
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=631551040))
        return update, query

    async def test_publish_click_schedules_instead_of_publishing_immediately(self) -> None:
        submission = self._submission("1")
        update, query = self._click("main", submission)
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock()) as publish:
            await moderation_callback(update, self._context())
        publish.assert_not_called()
        self.assertIsNone(self.store.get_submission(submission.id).destination)
        pending = self.store.pending_scheduled_posts()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].destination, "main")
        query.message.reply_text.assert_awaited_once()

    async def test_scheduling_hides_the_decision_buttons_on_the_moderator_message(self) -> None:
        submission = self._submission("1")
        self.store.record_moderator_message(submission.id, 631551040, -100, 55)
        update, _query = self._click("main", submission)
        context = self._context()
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock()):
            await moderation_callback(update, context)
        context.bot.edit_message_caption.assert_awaited_once()
        kwargs = context.bot.edit_message_caption.call_args.kwargs
        self.assertIsNone(kwargs["reply_markup"])
        self.assertIn("Запланировано", kwargs["caption"])

    async def test_second_click_on_the_same_submission_is_rejected_not_double_scheduled(self) -> None:
        submission = self._submission("1")
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock()):
            update, _query = self._click("main", submission)
            await moderation_callback(update, self._context())
            update2, query2 = self._click("main", submission)
            await moderation_callback(update2, self._context())
        self.assertEqual(len(self.store.pending_scheduled_posts()), 1)
        query2.answer.assert_awaited()
        self.assertIn("запланировано", query2.answer.call_args.args[0].lower())

    async def test_second_and_third_click_chain_into_consecutive_slots(self) -> None:
        # Three quick clicks with nothing sent yet still land on three
        # different slots, each `DELAYED_POST_INTERVAL_SECONDS` apart - the
        # not-yet-sent rows already reserve their place in the chain.
        first = self._submission("1")
        second = self._submission("2")
        third = self._submission("3")
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock()):
            update1, _query1 = self._click("main", first)
            await moderation_callback(update1, self._context())
            update2, _query2 = self._click("main", second)
            await moderation_callback(update2, self._context())
            update3, _query3 = self._click("main", third)
            await moderation_callback(update3, self._context())

        first_slot = self.store.pending_schedule_for_submission(first.id).scheduled_for
        second_slot = self.store.pending_schedule_for_submission(second.id).scheduled_for
        third_slot = self.store.pending_schedule_for_submission(third.id).scheduled_for
        self.assertAlmostEqual(second_slot - first_slot, DELAYED_POST_INTERVAL_SECONDS, delta=2)
        self.assertAlmostEqual(third_slot - second_slot, DELAYED_POST_INTERVAL_SECONDS, delta=2)

    async def test_reject_bypasses_the_delay_queue(self) -> None:
        submission = self._submission("1")
        update, _query = self._click("reject", submission)
        await moderation_callback(update, self._context())
        self.assertEqual(self.store.get_submission(submission.id).destination, "rejected")
        self.assertEqual(self.store.pending_scheduled_posts(), [])


class ScheduledQueueCommandsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.settings = ProposalBotSettings(
            token="x",
            database=Path(self._tempdir.name) / "proposal.sqlite3",
            moderator_ids=frozenset({631551040}),
            main_channel="@elevenlabss",
            shame_channel="@ghienmigo",
            karma_chat="@lalaschoo",
        )

    def _submission_and_schedule(self, job_number: str = "1"):
        submission, _created = self.store.create_submission(
            job_number=job_number, user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / f"{job_number}.mp4", output_filename="dubbed.mp4",
        )
        item = self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@elevenlabss", moderator_id=631551040,
            scheduled_for=1_000_000.0,
        )
        return submission, item

    def _context(self, args: list[str] | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": self.settings, "store": self.store}),
            bot=SimpleNamespace(send_message=AsyncMock(), edit_message_caption=AsyncMock()),
            args=args or [],
        )

    def _update(self, user_id: int = 631551040) -> tuple[SimpleNamespace, SimpleNamespace]:
        message = SimpleNamespace(reply_text=AsyncMock())
        return SimpleNamespace(effective_user=SimpleNamespace(id=user_id), effective_message=message), message

    async def test_scheduled_command_sends_a_single_message_for_the_whole_queue(self) -> None:
        self._submission_and_schedule("1")
        self._submission_and_schedule("2")
        update, message = self._update()
        await scheduled_command(update, self._context())
        message.reply_text.assert_awaited_once()
        text = message.reply_text.call_args.args[0]
        self.assertIn("№1", text)
        self.assertIn("№2", text)
        self.assertIn("/post", text)
        self.assertIn("/unpost", text)

    async def test_scheduled_command_reports_an_empty_queue(self) -> None:
        update, message = self._update()
        await scheduled_command(update, self._context())
        self.assertIn("пуста", message.reply_text.call_args.args[0])

    async def test_post_sends_by_job_number_and_marks_sent(self) -> None:
        submission, _item = self._submission_and_schedule("42")
        update, message = self._update()
        fake_message = _FakePublishedMessage(chat_id=-1001, message_id=42)
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock(return_value=fake_message)):
            await post_command(update, self._context(["42"]))
        updated = self.store.get_submission(submission.id)
        self.assertEqual(updated.destination, "main")
        self.assertEqual(self.store.pending_scheduled_posts(), [])
        self.assertIn("отправлена", message.reply_text.call_args.args[0])

    async def test_post_with_unknown_job_number_reports_not_found(self) -> None:
        update, message = self._update()
        await post_command(update, self._context(["999"]))
        self.assertIn("не найдена", message.reply_text.call_args.args[0])

    async def test_unpost_cancels_and_restores_decision_buttons(self) -> None:
        submission, _item = self._submission_and_schedule("42")
        self.store.record_moderator_message(submission.id, 631551040, -100, 55)
        update, message = self._update()
        context = self._context(["42"])
        await unpost_command(update, context)
        self.assertEqual(self.store.pending_scheduled_posts(), [])
        self.assertIn("снята", message.reply_text.call_args.args[0])
        context.bot.edit_message_caption.assert_awaited_once()
        self.assertIsNotNone(context.bot.edit_message_caption.call_args.kwargs["reply_markup"])

    async def test_non_moderator_cannot_post_or_unpost(self) -> None:
        self._submission_and_schedule("42")
        update, message = self._update(user_id=1)
        await post_command(update, self._context(["42"]))
        await unpost_command(update, self._context(["42"]))
        self.assertEqual(len(self.store.pending_scheduled_posts()), 1)
        message.reply_text.assert_not_awaited()


class CleanCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.settings = ProposalBotSettings(
            token="x",
            database=Path(self._tempdir.name) / "proposal.sqlite3",
            moderator_ids=frozenset({631551040}),
            main_channel="@elevenlabss",
            shame_channel="@ghienmigo",
            karma_chat="@lalaschoo",
        )

    def _submission(self, job_number: str) -> Submission:
        submission, _created = self.store.create_submission(
            job_number=job_number, user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / f"{job_number}.mp4", output_filename="dubbed.mp4",
        )
        return submission

    def _context(self, delete_message: AsyncMock | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": self.settings, "store": self.store}),
            bot=SimpleNamespace(delete_message=delete_message or AsyncMock()),
        )

    def _update(self, user_id: int = 631551040) -> tuple[SimpleNamespace, SimpleNamespace]:
        message = SimpleNamespace(reply_text=AsyncMock())
        return SimpleNamespace(effective_user=SimpleNamespace(id=user_id), effective_message=message), message

    async def test_deletes_notes_and_resolved_videos_but_keeps_pending(self) -> None:
        pending_submission = self._submission("1")
        self.store.record_moderator_message(pending_submission.id, 631551040, -100, 10)

        rejected_submission = self._submission("2")
        self.store.try_claim(rejected_submission.id, 631551040)
        self.store.finish_decision(
            submission_id=rejected_submission.id, moderator_id=631551040, destination="rejected", award_milli=0,
            publication_chat_id=None, publication_message_id=None,
        )
        self.store.record_moderator_message(rejected_submission.id, 631551040, -100, 20)

        scheduled_submission = self._submission("3")
        self.store.record_moderator_message(scheduled_submission.id, 631551040, -100, 40)
        self.store.schedule_post(
            submission_id=scheduled_submission.id, destination="main", target_chat="@elevenlabss",
            moderator_id=631551040, scheduled_for=1_000_000.0,
        )

        self.store.record_bot_note(631551040, -100, 30)
        self.store.record_bot_note(631551040, -100, 31)

        update, message = self._update()
        context = self._context()
        await clean_command(update, context)

        deleted_ids = {call.args[1] for call in context.bot.delete_message.call_args_list}
        self.assertEqual(deleted_ids, {20, 30, 31, 40})
        self.assertIn("Очищено сообщений: 4", message.reply_text.call_args.args[0])
        # The pending video's tracking row survives - it's still awaiting a decision.
        self.assertEqual(self.store.moderator_messages(pending_submission.id), [(-100, 10)])
        # The resolved and scheduled videos' tracking rows are gone with the messages.
        self.assertEqual(self.store.moderator_messages(rejected_submission.id), [])
        self.assertEqual(self.store.moderator_messages(scheduled_submission.id), [])
        # Cancelling it is still possible by job number even though the chat message is gone.
        self.assertIsNotNone(self.store.find_pending_schedule_by_job_number("3"))

    async def test_a_failed_delete_is_counted_but_does_not_abort(self) -> None:
        self.store.record_bot_note(631551040, -100, 30)
        self.store.record_bot_note(631551040, -100, 31)
        delete_message = AsyncMock(side_effect=[Exception("gone"), None])
        update, message = self._update()
        await clean_command(update, self._context(delete_message))
        self.assertIn("Очищено сообщений: 1", message.reply_text.call_args.args[0])
        self.assertIn("Не удалось удалить: 1", message.reply_text.call_args.args[0])
        # The failed note remains tracked so a later /clean can retry it.
        self.assertIn((-100, 30), self.store.bot_notes(631551040))

    async def test_failed_video_delete_remains_tracked_for_retry(self) -> None:
        submission = self._submission("4")
        self.store.try_claim(submission.id, 631551040)
        self.store.finish_decision(
            submission_id=submission.id, moderator_id=631551040, destination="rejected", award_milli=0,
            publication_chat_id=None, publication_message_id=None,
        )
        self.store.record_moderator_message(submission.id, 631551040, -100, 50)
        delete_message = AsyncMock(side_effect=Exception("message can't be deleted"))
        update, _message = self._update()
        await clean_command(update, self._context(delete_message))
        self.assertEqual(self.store.moderator_messages(submission.id), [(-100, 50)])

    async def test_decision_cleanup_deletes_all_moderator_cards(self) -> None:
        submission = self._submission("5")
        self.store.record_moderator_message(submission.id, 631551040, 631551040, 60)
        self.store.record_moderator_message(submission.id, 7123813884, 7123813884, 61)
        delete_message = AsyncMock()
        deleted, failed = await _delete_moderator_cards(
            SimpleNamespace(delete_message=delete_message), self.store, submission.id
        )
        self.assertEqual((deleted, failed), (2, 0))
        self.assertEqual(self.store.moderator_messages(submission.id), [])

    async def test_non_moderator_does_nothing(self) -> None:
        self.store.record_bot_note(1, -100, 30)
        update, message = self._update(user_id=1)
        context = self._context()
        await clean_command(update, context)
        context.bot.delete_message.assert_not_called()
        message.reply_text.assert_not_awaited()
        # The note wasn't consumed either - a real moderator running /clean
        # later could still see it (not that a non-moderator should have one).
        self.assertEqual(self.store.take_bot_notes(1), [(-100, 30)])


class BacklogSpacingStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")

    def _submission(self, job_number: str) -> Submission:
        submission, _created = self.store.create_submission(
            job_number=job_number, user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / f"{job_number}.mp4", output_filename="dubbed.mp4",
        )
        return submission

    def test_no_publications_yet_means_nothing_blocks_the_first_post(self) -> None:
        self.assertIsNone(self.store.last_published_at("main"))

    def test_only_counts_what_actually_went_out(self) -> None:
        # A merely *planned* post must not look like a recent publication,
        # otherwise the very post waiting to go out would block itself.
        submission = self._submission("1")
        self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=1_000_000.0,
        )
        self.assertIsNone(self.store.last_published_at("main"))

    def test_a_sent_post_counts(self) -> None:
        submission = self._submission("1")
        item = self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=1_000_000.0,
        )
        self.store.mark_scheduled_sent(item.id)
        self.assertIsNotNone(self.store.last_published_at("main"))

    def test_channels_are_tracked_separately(self) -> None:
        submission = self._submission("1")
        item = self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@x", moderator_id=1,
            scheduled_for=1_000_000.0,
        )
        self.store.mark_scheduled_sent(item.id)
        self.assertIsNotNone(self.store.last_published_at("main"))
        self.assertIsNone(self.store.last_published_at("shame"))


class BacklogSendingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.settings = ProposalBotSettings(
            token="x",
            database=Path(self._tempdir.name) / "proposal.sqlite3",
            moderator_ids=frozenset({631551040}),
            main_channel="@elevenlabss",
            shame_channel="@ghienmigo",
            karma_chat="@lalaschoo",
        )

    def _context(self) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": self.settings, "store": self.store}),
            bot=SimpleNamespace(send_message=AsyncMock(), edit_message_caption=AsyncMock()),
        )

    def _overdue(self, job_number: str, destination: str = "main") -> None:
        submission, _created = self.store.create_submission(
            job_number=job_number, user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / f"{job_number}.mp4", output_filename="dubbed.mp4",
        )
        self.store.schedule_post(
            submission_id=submission.id, destination=destination,
            target_chat="@elevenlabss" if destination == "main" else "@ghienmigo",
            moderator_id=631551040, scheduled_for=1_000_000.0,  # long past
        )

    async def test_a_backlog_goes_out_one_at_a_time_not_in_a_burst(self) -> None:
        # Regression: after the bot was down, every overdue post fired at once,
        # dumping the whole queue into the channel back to back.
        for n in ("1", "2", "3", "4", "5"):
            self._overdue(n)
        self.assertEqual(len(self.store.due_scheduled_posts()), 5)

        published = _FakePublishedMessage(chat_id=-1001, message_id=7)
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock(return_value=published)):
            sent = await send_due_scheduled_posts(self._context())
        self.assertEqual(sent, 1)
        self.assertEqual(len(self.store.pending_scheduled_posts()), 4)

    async def test_each_channel_gets_its_own_first_post(self) -> None:
        self._overdue("1", destination="main")
        self._overdue("2", destination="shame")

        published = _FakePublishedMessage(chat_id=-1001, message_id=7)
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock(return_value=published)):
            sent = await send_due_scheduled_posts(self._context())
        self.assertEqual(sent, 2)
        self.assertEqual(self.store.pending_scheduled_posts(), [])

    async def test_nothing_due_sends_nothing(self) -> None:
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock()) as publish:
            sent = await send_due_scheduled_posts(self._context())
        self.assertEqual(sent, 0)
        publish.assert_not_called()


class ProcessScheduledPostTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.settings = ProposalBotSettings(
            token="x",
            database=Path(self._tempdir.name) / "proposal.sqlite3",
            moderator_ids=frozenset({631551040}),
            main_channel="@elevenlabss",
            shame_channel="@ghienmigo",
            karma_chat="@lalaschoo",
        )

    def _context(self) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": self.settings, "store": self.store}),
            bot=SimpleNamespace(send_message=AsyncMock()),
        )

    async def test_due_post_is_published_without_pestering_the_moderator(self) -> None:
        submission, _created = self.store.create_submission(
            job_number="1", user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / "dubbed.mp4", output_filename="dubbed.mp4",
        )
        item = self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@elevenlabss", moderator_id=631551040,
            scheduled_for=1_000_000.0,
        )
        fake_message = _FakePublishedMessage(chat_id=-1001, message_id=7)
        context = self._context()
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock(return_value=fake_message)):
            ok = await _process_scheduled_post(context, item)
        self.assertTrue(ok)
        updated = self.store.get_submission(submission.id)
        self.assertEqual(updated.destination, "main")
        self.assertEqual(updated.publication_message_id, 7)
        self.assertEqual(self.store.due_scheduled_posts(now=2_000_000.0), [])
        # The post going out is visible in the channel; a "published, karma +N"
        # message to the moderator was only something to scroll past and clean
        # up later. It now goes to the log instead.
        context.bot.send_message.assert_not_awaited()

    async def test_publish_failure_keeps_the_post_pending_for_a_retry(self) -> None:
        submission, _created = self.store.create_submission(
            job_number="1", user_id=123, chat_id=123, author_name="Тест", author_username=None,
            video_path=Path(self._tempdir.name) / "dubbed.mp4", output_filename="dubbed.mp4",
        )
        item = self.store.schedule_post(
            submission_id=submission.id, destination="main", target_chat="@elevenlabss", moderator_id=631551040,
            scheduled_for=1_000_000.0,
        )
        context = self._context()
        with patch("laladub.proposal_bot._publish_to_channel", new=AsyncMock(side_effect=RuntimeError("boom"))):
            ok = await _process_scheduled_post(context, item)
        self.assertFalse(ok)
        self.assertEqual(self.store.get_submission(submission.id).destination, None)
        self.assertEqual(len(self.store.due_scheduled_posts(now=2_000_000.0)), 1)


class TimerCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposal.sqlite3")
        self.settings = ProposalBotSettings(
            token="x",
            database=Path(self._tempdir.name) / "proposal.sqlite3",
            moderator_ids=frozenset({631551040}),
            main_channel="@elevenlabss",
            shame_channel="@ghienmigo",
            karma_chat="@lalaschoo",
        )

    def _context(self, args: list[str] | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": self.settings, "store": self.store}),
            args=args or [],
        )

    async def test_bare_command_toggles_state(self) -> None:
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=631551040), effective_message=message)
        await timer_command(update, self._context())
        self.assertTrue(self.store.delayed_posting_enabled())
        await timer_command(update, self._context())
        self.assertFalse(self.store.delayed_posting_enabled())

    async def test_explicit_on_and_off(self) -> None:
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=631551040), effective_message=message)
        await timer_command(update, self._context(["on"]))
        self.assertTrue(self.store.delayed_posting_enabled())
        await timer_command(update, self._context(["off"]))
        self.assertFalse(self.store.delayed_posting_enabled())

    async def test_non_moderator_cannot_toggle(self) -> None:
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await timer_command(update, self._context(["on"]))
        self.assertFalse(self.store.delayed_posting_enabled())
        message.reply_text.assert_not_awaited()


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
