from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from laladub.bot import (
    _advance_selection,
    preset_command,
    preset_wizard_callback,
)
from laladub.preset_store import PresetStore


class _Message:
    def __init__(self) -> None:
        self.replies: list[tuple[str, object]] = []
        self.edits: list[tuple[str, object]] = []

    async def reply_text(self, text: str, reply_markup: object = None) -> None:
        self.replies.append((text, reply_markup))

    async def edit_text(self, text: str, reply_markup: object = None) -> None:
        self.edits.append((text, reply_markup))


class _Query:
    def __init__(self, data: str) -> None:
        self.data = data
        self.answer = AsyncMock()
        self.edit_message_text = AsyncMock()
        self.message = _Message()


def _context(preset_store: PresetStore) -> SimpleNamespace:
    return SimpleNamespace(
        application=SimpleNamespace(bot_data={"preset_store": preset_store}),
        user_data={},
        args=[],
    )


class PresetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PresetStore(Path(self._tempdir.name) / "presets.sqlite3")

    def test_unconfigured_user_has_an_all_ask_preset(self) -> None:
        preset = self.store.get_preset(1)
        self.assertEqual(preset.as_dict(), {field: None for field in preset.as_dict()})

    def test_set_and_read_back_a_single_field(self) -> None:
        self.store.set_preset_field(1, "target_lang", "ru")
        preset = self.store.get_preset(1)
        self.assertEqual(preset.target_lang, "ru")
        self.assertIsNone(preset.source_lang)

    def test_setting_a_second_field_preserves_the_first(self) -> None:
        self.store.set_preset_field(1, "target_lang", "ru")
        self.store.set_preset_field(1, "source_lang", "vi")
        preset = self.store.get_preset(1)
        self.assertEqual(preset.target_lang, "ru")
        self.assertEqual(preset.source_lang, "vi")

    def test_setting_a_field_to_none_means_ask_every_time(self) -> None:
        self.store.set_preset_field(1, "target_lang", "ru")
        self.store.set_preset_field(1, "target_lang", None)
        preset = self.store.get_preset(1)
        self.assertIsNone(preset.target_lang)

    def test_clear_preset_resets_every_field(self) -> None:
        self.store.set_preset_field(1, "target_lang", "ru")
        self.store.clear_preset(1)
        preset = self.store.get_preset(1)
        self.assertEqual(preset.target_lang, None)

    def test_presets_are_isolated_per_user(self) -> None:
        self.store.set_preset_field(1, "target_lang", "ru")
        self.store.set_preset_field(2, "target_lang", "en")
        self.assertEqual(self.store.get_preset(1).target_lang, "ru")
        self.assertEqual(self.store.get_preset(2).target_lang, "en")


class PresetCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PresetStore(Path(self._tempdir.name) / "presets.sqlite3")

    async def test_starts_the_wizard_on_the_first_step(self) -> None:
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await preset_command(update, _context(self.store))
        self.assertEqual(len(message.replies), 1)
        text, markup = message.replies[0]
        self.assertIn("шаг 1/6", text)
        self.assertIn("Видеоряд", text)
        codes = {button.callback_data for row in markup.inline_keyboard for button in row}
        self.assertIn("pset:visual_mode:ask", codes)
        self.assertIn("pset:visual_mode:original", codes)

    async def test_reset_clears_a_previously_saved_preset(self) -> None:
        self.store.set_preset_field(1, "target_lang", "ru")
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        context = _context(self.store)
        context.args = ["reset"]
        await preset_command(update, context)
        self.assertIsNone(self.store.get_preset(1).target_lang)
        self.assertTrue(any("сброшен" in text for text, _markup in message.replies))


class PresetWizardCallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PresetStore(Path(self._tempdir.name) / "presets.sqlite3")

    async def _answer(self, user_id: int, data: str) -> _Query:
        query = _Query(data)
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=user_id))
        await preset_wizard_callback(update, _context(self.store))
        return query

    async def test_answering_a_step_saves_it_and_advances_to_the_next(self) -> None:
        query = await self._answer(1, "pset:visual_mode:original")
        self.assertEqual(self.store.get_preset(1).visual_mode, "original")
        args, kwargs = query.edit_message_text.call_args
        text = args[0]
        markup = kwargs["reply_markup"]
        self.assertIn("шаг 2/6", text)
        self.assertIn("Входной", text)
        codes = {button.callback_data for row in markup.inline_keyboard for button in row}
        self.assertIn("pset:source_lang:vi", codes)

    async def test_choosing_ask_stores_none(self) -> None:
        await self._answer(1, "pset:target_lang:ask")
        self.assertIsNone(self.store.get_preset(1).target_lang)

    async def test_completing_every_step_shows_a_summary(self) -> None:
        await self._answer(1, "pset:visual_mode:original")
        await self._answer(1, "pset:source_lang:vi")
        await self._answer(1, "pset:speaker_count:auto")
        await self._answer(1, "pset:target_lang:ru")
        await self._answer(1, "pset:tts_provider:moss")
        query = await self._answer(1, "pset:review_mode:direct")
        text = query.edit_message_text.call_args.args[0]
        self.assertIn("Пресет сохранён", text)
        self.assertIn("Видеоряд: Оставить исходный видеоряд", text)
        self.assertIn("Входной язык: Вьетнамский", text)
        self.assertIn("Количество голосов: Авто", text)
        self.assertIn("Язык озвучки: Русский", text)
        self.assertIn("Движок озвучки: MOSS", text)


class AdvanceSelectionPresetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.job_dir = Path(self._tempdir.name)

    def _job(self, preset: dict) -> dict:
        return {
            "job_dir": str(self.job_dir),
            "input_path": str(self.job_dir / "input.mp4"),
            "input_source": "telegram_upload",
            "_preset": preset,
        }

    async def test_fully_configured_preset_skips_every_screen_and_enqueues(self) -> None:
        job = self._job(
            {
                "visual_mode": "original",
                "source_lang": "vi",
                "speaker_count": "auto",
                "target_lang": "ru",
                "tts_provider": "moss",
                "review_mode": "direct",
            }
        )
        target = _Message()
        context = SimpleNamespace(user_data={"job": job})
        update = SimpleNamespace(effective_chat=SimpleNamespace(id=-1), effective_user=SimpleNamespace(id=1))
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await _advance_selection(update, context, job, target)
        enqueue.assert_awaited_once()
        self.assertEqual(job["source_lang"], "vi")
        self.assertEqual(job["speaker_count"], "auto")
        self.assertEqual(job["target_lang"], "ru")
        self.assertEqual(job["tts_provider"], "moss")
        self.assertIn("Ставлю задачу в очередь", target.edits[0][0])
        self.assertNotIn("job", context.user_data)

    async def test_ask_for_one_field_shows_only_that_screen(self) -> None:
        job = self._job(
            {
                "visual_mode": "original",
                "source_lang": "vi",
                "speaker_count": "auto",
                "target_lang": "ask",
                "tts_provider": "moss",
                "review_mode": "direct",
            }
        )
        target = _Message()
        context = SimpleNamespace(user_data={"job": job})
        update = SimpleNamespace()
        with patch("laladub.bot._enqueue_job", new=AsyncMock()) as enqueue:
            await _advance_selection(update, context, job, target)
        enqueue.assert_not_called()
        self.assertEqual(job["source_lang"], "vi")
        self.assertNotIn("target_lang", job)
        text, _markup = target.edits[0]
        self.assertIn("Выбери язык озвучки", text)


if __name__ == "__main__":
    unittest.main()
