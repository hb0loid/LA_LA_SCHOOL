from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from laladub.bot import (
    MAX_TEXT_REVIEW_ATTEMPTS,
    REVIEW_MODE_OPTIONS,
    _reset_translation_outputs,
    _text_review_keyboard,
    select_review_mode,
    text_review_callback,
)
from laladub.text_review import TextReviewStore


class ReviewStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = TextReviewStore(Path(self._tempdir.name) / "reviews.sqlite3")

    def test_records_each_decision(self) -> None:
        for attempt, decision in enumerate(("rejected", "rejected", "approved"), start=1):
            self.store.record(
                job_number="42", user_id=1, attempt=attempt, decision=decision, text=f"вариант {attempt}"
            )
        self.assertEqual(self.store.summary(), {"rejected": 2, "approved": 1})

    def test_recent_can_filter_by_decision(self) -> None:
        self.store.record(job_number="1", user_id=1, attempt=1, decision="approved", text="да")
        self.store.record(job_number="2", user_id=1, attempt=1, decision="cancelled", text="нет")
        approved = self.store.recent(decision="approved")
        self.assertEqual([r.text for r in approved], ["да"])

    def test_attempts_for_job_tracks_the_retry_counter(self) -> None:
        self.assertEqual(self.store.attempts_for_job("42"), 0)
        self.store.record(job_number="42", user_id=1, attempt=1, decision="rejected", text="a")
        self.store.record(job_number="42", user_id=1, attempt=2, decision="approved", text="b")
        self.assertEqual(self.store.attempts_for_job("42"), 2)

    def test_language_pair_is_kept_for_later_study(self) -> None:
        self.store.record(
            job_number="7", user_id=1, attempt=1, decision="approved", text="текст",
            source_lang="vi", target_lang="ru",
        )
        record = self.store.recent()[0]
        self.assertEqual((record.source_lang, record.target_lang), ("vi", "ru"))


class ReviewKeyboardTests(unittest.TestCase):
    def _codes(self, attempt: int) -> set[str]:
        markup = _text_review_keyboard("42", attempt)
        return {button.callback_data for row in markup.inline_keyboard for button in row}

    def test_all_three_choices_offered_early_on(self) -> None:
        self.assertEqual(self._codes(1), {"rv:ok:42", "rv:again:42", "rv:drop:42"})

    def test_retry_disappears_on_the_last_attempt(self) -> None:
        # Each retry costs a full ASR + translation pass, so the option goes
        # away rather than letting someone loop forever.
        codes = self._codes(MAX_TEXT_REVIEW_ATTEMPTS)
        self.assertNotIn("rv:again:42", codes)
        self.assertIn("rv:ok:42", codes)
        self.assertIn("rv:drop:42", codes)


class ResetTranslationOutputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.job_dir = Path(self._tempdir.name)
        self.work = self.job_dir / "work"
        self.work.mkdir()

    def test_drops_the_text_but_keeps_the_expensive_audio_stages(self) -> None:
        (self.work / "translated.srt").write_text("текст", encoding="utf-8")
        (self.work / "source.srt").write_text("текст", encoding="utf-8")
        (self.work / "source_16k.wav").write_bytes(b"audio")
        (self.work / "resume_state.json").write_text(
            json.dumps({"audio": True, "separation": True, "translated": True, "segment_count": 5}),
            encoding="utf-8",
        )

        _reset_translation_outputs(self.job_dir)

        self.assertFalse((self.work / "translated.srt").exists())
        self.assertFalse((self.work / "source.srt").exists())
        # Re-extracting audio and re-running separation would be the slow part.
        self.assertTrue((self.work / "source_16k.wav").exists())
        state = json.loads((self.work / "resume_state.json").read_text(encoding="utf-8"))
        self.assertTrue(state["audio"])
        self.assertTrue(state["separation"])
        self.assertNotIn("translated", state)
        self.assertNotIn("segment_count", state)

    def test_missing_resume_state_is_not_an_error(self) -> None:
        (self.work / "translated.srt").write_text("текст", encoding="utf-8")
        _reset_translation_outputs(self.job_dir)
        self.assertFalse((self.work / "translated.srt").exists())


class SelectReviewModeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.job_dir = Path(self._tempdir.name)

    def _query(self, data: str):
        return SimpleNamespace(
            data=data,
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
            message=SimpleNamespace(edit_text=AsyncMock(), reply_text=AsyncMock(), chat_id=-1),
        )

    async def test_choice_is_stored_and_the_job_is_enqueued(self) -> None:
        job = {
            "job_dir": str(self.job_dir),
            "input_path": str(self.job_dir / "input.mp4"),
            "visual_mode": "original",
            "source_lang": None,
            "speaker_count": "auto",
            "target_lang": "ru",
            "tts_provider": "moss",
        }
        query = self._query("rev:review")
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace(user_data={"job": job}, application=SimpleNamespace(bot_data={}))
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_review_mode(update, context)
        self.assertEqual(job["review_mode"], "review")
        enqueue.assert_awaited_once()

    async def test_unknown_mode_is_rejected(self) -> None:
        job = {"job_dir": str(self.job_dir), "input_path": "x"}
        query = self._query("rev:whatever")
        update = SimpleNamespace(callback_query=query)
        context = SimpleNamespace(user_data={"job": job}, application=SimpleNamespace(bot_data={}))
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_review_mode(update, context)
        enqueue.assert_not_called()
        self.assertNotIn("review_mode", job)

    def test_both_modes_are_offered(self) -> None:
        self.assertEqual([code for code, _label in REVIEW_MODE_OPTIONS], ["direct", "review"])


class TextReviewCallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.store = TextReviewStore(self.root / "reviews.sqlite3")
        self.job_dir = self.root / "77" / "4242"
        (self.job_dir / "work").mkdir(parents=True)
        # _load_job_snapshot treats a job whose input file is gone as dead.
        (self.job_dir / "input.mp4").write_bytes(b"video")
        (self.job_dir / "work" / "translated.srt").write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nПривет мир\n", encoding="utf-8"
        )
        self.job = {
            "job_dir": str(self.job_dir),
            "input_path": str(self.job_dir / "input.mp4"),
            "review_mode": "review",
            "review_attempt": 1,
            "source_lang": "vi",
            "target_lang": "ru",
        }
        self._write_snapshot("awaiting_review")

    def _write_snapshot(self, status: str) -> None:
        data = dict(self.job)
        data["status"] = status
        (self.job_dir / "job.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _call(self, action: str):
        query = SimpleNamespace(
            data=f"rv:{action}:4242",
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
            message=SimpleNamespace(chat_id=77, reply_text=AsyncMock()),
        )
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=77))
        context = SimpleNamespace(
            application=SimpleNamespace(
                bot_data={
                    "settings": SimpleNamespace(workdir=self.root),
                    "review_store": self.store,
                }
            ),
            bot=SimpleNamespace(send_message=AsyncMock()),
        )
        return update, context, query

    async def test_approving_records_the_choice_and_resumes(self) -> None:
        update, context, query = self._call("ok")
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await text_review_callback(update, context)
        enqueue.assert_awaited_once()
        self.assertEqual(self.store.summary(), {"approved": 1})
        self.assertEqual(self.store.recent()[0].text, "Привет мир")

    async def test_rejecting_asks_for_another_variant(self) -> None:
        update, context, query = self._call("again")
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await text_review_callback(update, context)
        enqueue.assert_awaited_once()
        self.assertEqual(self.store.summary(), {"rejected": 1})
        # The prepared text is dropped so the next run builds a fresh one.
        self.assertFalse((self.job_dir / "work" / "translated.srt").exists())

    async def test_cancelling_ends_the_job(self) -> None:
        update, context, query = self._call("drop")
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue, patch(
            "laladub.bot._release_daily_allowance", new=AsyncMock()
        ):
            await text_review_callback(update, context)
        enqueue.assert_not_called()
        self.assertEqual(self.store.summary(), {"cancelled": 1})
        saved = json.loads((self.job_dir / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "rejected")

    async def test_a_job_no_longer_awaiting_review_is_ignored(self) -> None:
        # Guards against a stale button being pressed twice.
        self._write_snapshot("done")
        update, context, query = self._call("ok")
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await text_review_callback(update, context)
        enqueue.assert_not_called()
        self.assertEqual(self.store.summary(), {})


if __name__ == "__main__":
    unittest.main()
