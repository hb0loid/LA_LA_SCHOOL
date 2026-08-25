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

    async def test_ru_shows_the_engine_choice_instead_of_enqueueing(self) -> None:
        query = _Query("tgt:ru")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_target_lang(update, context)
        enqueue.assert_not_called()
        self.assertEqual(context.user_data["job"]["target_lang"], "ru")
        self.assertNotIn("tts_provider", context.user_data["job"])
        markup = query.message.edit_text.call_args.kwargs["reply_markup"]
        labels_and_data = [
            (button.text, button.callback_data) for row in markup.inline_keyboard for button in row
        ]
        # Only moss and cosyvoice are offered - qwen3 hung in testing, f5 is
        # Ukrainian-only and picked automatically, not via this menu.
        codes = [data.split(":", 1)[1] for _text, data in labels_and_data if data != "back:target"]
        self.assertEqual(set(codes), {"moss", "cosyvoice"})

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

    async def test_en_also_shows_the_engine_choice(self) -> None:
        query = _Query("tgt:en")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_target_lang(update, context)
        enqueue.assert_not_called()
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

    async def test_cosyvoice_choice_enqueues_with_cosyvoice(self) -> None:
        query = _Query("tts:cosyvoice")
        context = _context(self.job)
        update = SimpleNamespace(callback_query=query)
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await select_tts_method(update, context)
        enqueue.assert_awaited_once()
        self.assertEqual(self.job["tts_provider"], "cosyvoice")

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
    def test_only_moss_and_cosyvoice_are_offered(self) -> None:
        self.assertEqual([code for code, _label in TTS_METHOD_CHOICES], ["moss", "cosyvoice"])


if __name__ == "__main__":
    unittest.main()
