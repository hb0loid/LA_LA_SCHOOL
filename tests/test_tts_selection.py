from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from laladub.bot import TTS_METHOD_CHOICES, select_target_lang, select_tts_method


class _Query:
    def __init__(self, data: str) -> None:
        self.data = data
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()
        self.message = SimpleNamespace(chat_id=-1, edit_text=AsyncMock(), reply_text=AsyncMock())


def _context(job: dict) -> SimpleNamespace:
    return SimpleNamespace(
        user_data={"job": job},
        application=SimpleNamespace(bot_data={}),
    )


class SelectTargetLangTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.job_dir = Path(self._tempdir.name)
        # Real jobs always reach the target-language screen with the earlier
        # steps (visual/source/speakers) already answered - _advance_selection
        # re-walks from the start otherwise, so these are pre-filled here too.
        self.job = {
            "job_dir": str(self.job_dir),
            "input_path": str(self.job_dir / "input.mp4"),
            "visual_mode": "original",
            "source_lang": None,
            "speaker_count": "auto",
            "review_mode": "direct",
        }

    async def test_ru_skips_the_engine_screen_while_only_one_engine_is_offered(self) -> None:
        # A screen with a single button is just an extra tap; the choice is
        # made for the user and the job goes straight into the queue.
        query = _Query("tgt:ru")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_target_lang(update, context)
        enqueue.assert_awaited_once()
        self.assertEqual(self.job["target_lang"], "ru")
        self.assertEqual(self.job["tts_provider"], "moss")

    async def test_uk_auto_selects_f5_without_a_choice_screen(self) -> None:
        query = _Query("tgt:uk")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_target_lang(update, context)
        enqueue.assert_awaited_once()
        # user_data["job"] is popped by the auto-select branch before enqueueing;
        # the dict object itself still reflects the final state.
        self.assertEqual(self.job["target_lang"], "uk")
        self.assertEqual(self.job["tts_provider"], "f5")

    async def test_en_skips_it_too(self) -> None:
        query = _Query("tgt:en")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_target_lang(update, context)
        enqueue.assert_awaited_once()
        self.assertEqual(self.job["tts_provider"], "moss")

    async def test_the_screen_returns_once_a_second_engine_is_offered(self) -> None:
        # The skip is driven by the offer list, not hardcoded - restoring an
        # engine must bring the choice back without touching this branch.
        query = _Query("tgt:ru")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        two = [*TTS_METHOD_CHOICES, ("cosyvoice", "CosyVoice")]
        with patch("laladub.bot.TTS_METHOD_CHOICES", two),              patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_target_lang(update, context)
        enqueue.assert_not_called()
        self.assertNotIn("tts_provider", self.job)
        self.assertIn("Выбери движок озвучки", query.message.edit_text.call_args.args[0])


class SelectTtsMethodTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.job_dir = Path(self._tempdir.name)
        self.job = {
            "job_dir": str(self.job_dir),
            "input_path": str(self.job_dir / "input.mp4"),
            "visual_mode": "original",
            "source_lang": None,
            "speaker_count": "auto",
            "target_lang": "ru",
            "review_mode": "direct",
        }

    async def test_moss_choice_enqueues_with_moss(self) -> None:
        query = _Query("tts:moss")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_tts_method(update, context)
        enqueue.assert_awaited_once()
        self.assertEqual(self.job["tts_provider"], "moss")

    async def test_hidden_cosyvoice_is_rejected(self) -> None:
        # Still a working engine, just not offered - a stale button from an
        # older message must not slip it back in.
        query = _Query("tts:cosyvoice")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_tts_method(update, context)
        enqueue.assert_not_called()
        self.assertNotIn("tts_provider", self.job)

    async def test_unoffered_engine_is_rejected_even_if_the_code_is_valid(self) -> None:
        # qwen3 is a real provider the pipeline still supports, but it hung
        # mid-batch in testing and is deliberately not offered in this menu -
        # a stray/replayed "tts:qwen3" callback must not slip through.
        query = _Query("tts:qwen3")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_tts_method(update, context)
        enqueue.assert_not_called()
        self.assertNotIn("tts_provider", self.job)
        query.edit_message_text.assert_awaited_once()
        self.assertIn("Неизвестный", query.edit_message_text.call_args.args[0])


class TtsMethodChoicesTests(unittest.TestCase):
    def test_only_moss_is_offered(self) -> None:
        self.assertEqual([code for code, _label in TTS_METHOD_CHOICES], ["moss"])


if __name__ == "__main__":
    unittest.main()
