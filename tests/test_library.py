from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from laladub.bot import _archive_finished_dub
from laladub.library import LibraryStore, _last_show_call, show_command


class LibraryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = LibraryStore(Path(self._tempdir.name) / "library.sqlite3")

    def test_unknown_job_number_returns_none(self) -> None:
        self.assertIsNone(self.store.get("999"))

    def test_add_then_get(self) -> None:
        self.store.add(
            job_number="42367",
            user_id=123,
            source_title="Mr Beast",
            target_lang="ru",
            video_path="/videos/42367.mp4",
            output_filename="42367_lalaschool.mp4",
        )
        entry = self.store.get("42367")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, 123)
        self.assertEqual(entry.source_title, "Mr Beast")
        self.assertEqual(entry.video_path, "/videos/42367.mp4")

    def test_re_adding_the_same_job_number_overwrites(self) -> None:
        self.store.add(
            job_number="42367", user_id=123, source_title="Old", target_lang="ru",
            video_path="/videos/old.mp4", output_filename="old.mp4",
        )
        self.store.add(
            job_number="42367", user_id=123, source_title="New", target_lang="en",
            video_path="/videos/new.mp4", output_filename="new.mp4",
        )
        entry = self.store.get("42367")
        self.assertEqual(entry.source_title, "New")
        self.assertEqual(entry.video_path, "/videos/new.mp4")


class ShowCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = LibraryStore(Path(self._tempdir.name) / "library.sqlite3")
        _last_show_call.clear()
        self.addCleanup(_last_show_call.clear)

    def _context(self, args: list[str] | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"library_store": self.store}),
            bot=SimpleNamespace(send_video=AsyncMock()),
            args=args or [],
        )

    def _update(self, user_id: int = 777) -> tuple[SimpleNamespace, SimpleNamespace]:
        message = SimpleNamespace(reply_text=AsyncMock())
        return (
            SimpleNamespace(
                effective_message=message,
                effective_chat=SimpleNamespace(id=555),
                effective_user=SimpleNamespace(id=user_id),
            ),
            message,
        )

    async def test_no_argument_asks_for_usage(self) -> None:
        update, message = self._update()
        await show_command(update, self._context())
        self.assertIn("Использование", message.reply_text.call_args.args[0])

    async def test_unknown_job_number_is_reported(self) -> None:
        update, message = self._update()
        await show_command(update, self._context(["999"]))
        self.assertIn("не найдена", message.reply_text.call_args.args[0])

    async def test_missing_file_is_reported(self) -> None:
        self.store.add(
            job_number="42367", user_id=123, source_title="Mr Beast", target_lang="ru",
            video_path=str(Path(self._tempdir.name) / "gone.mp4"), output_filename="gone.mp4",
        )
        update, message = self._update()
        await show_command(update, self._context(["42367"]))
        self.assertIn("утерян", message.reply_text.call_args.args[0])

    async def test_sends_the_video_with_a_caption(self) -> None:
        video_path = Path(self._tempdir.name) / "42367.mp4"
        video_path.write_bytes(b"fake video bytes")
        self.store.add(
            job_number="42367", user_id=123, source_title="Mr Beast", target_lang="ru",
            video_path=str(video_path), output_filename="42367_lalaschool.mp4",
        )
        update, _message = self._update()
        context = self._context(["42367"])
        with patch("laladub.bot._telegram_sendable_video_path", new=AsyncMock(return_value=video_path)), \
             patch("laladub.bot.video_upload_metadata", new=AsyncMock(return_value={})):
            await show_command(update, context)
        context.bot.send_video.assert_awaited_once()
        kwargs = context.bot.send_video.call_args.kwargs
        self.assertEqual(kwargs["chat_id"], 555)
        self.assertIn("№42367", kwargs["caption"])
        self.assertIn("Mr Beast", kwargs["caption"])
        self.assertEqual(kwargs["filename"], "42367_lalaschool.mp4")

    async def test_no_library_store_is_a_silent_no_op(self) -> None:
        update, message = self._update()
        context = self._context(["42367"])
        context.application.bot_data.pop("library_store")
        await show_command(update, context)
        message.reply_text.assert_not_awaited()

    async def test_second_call_within_the_cooldown_is_rejected(self) -> None:
        update, _message = self._update(user_id=42)
        await show_command(update, self._context(["999"]))  # consumes the cooldown slot

        update2, message2 = self._update(user_id=42)
        await show_command(update2, self._context(["999"]))
        self.assertIn("Слишком часто", message2.reply_text.call_args.args[0])

    async def test_different_users_have_independent_cooldowns(self) -> None:
        update, _message = self._update(user_id=1)
        await show_command(update, self._context(["999"]))

        update2, message2 = self._update(user_id=2)
        await show_command(update2, self._context(["999"]))
        self.assertIn("не найдена", message2.reply_text.call_args.args[0])

    async def test_call_after_the_cooldown_expires_goes_through(self) -> None:
        update, _message = self._update(user_id=42)
        await show_command(update, self._context(["999"]))
        _last_show_call[42] -= 31  # pretend the cooldown already elapsed

        update2, message2 = self._update(user_id=42)
        await show_command(update2, self._context(["999"]))
        self.assertIn("не найдена", message2.reply_text.call_args.args[0])


class ArchiveFinishedDubTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = LibraryStore(Path(self._tempdir.name) / "library.sqlite3")
        self.library_dir = Path(self._tempdir.name) / "archive"

    def _context(self) -> SimpleNamespace:
        settings = SimpleNamespace(library_dir=self.library_dir)
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"settings": settings, "library_store": self.store})
        )

    def _job(self, job_dir: Path) -> dict:
        video_path = job_dir / "dubbed.mp4"
        video_path.write_bytes(b"fake video bytes")
        return {
            "job_dir": str(job_dir),
            "user_id": 123,
            "source_title": "Mr Beast",
            "target_lang": "ru",
            "proposal_output_filename": "42367_lalaschool.mp4",
        }

    async def test_copies_the_video_and_records_it(self) -> None:
        job_dir = Path(self._tempdir.name) / "jobs" / "42367"
        job_dir.mkdir(parents=True)
        job = self._job(job_dir)
        await _archive_finished_dub(self._context(), job)

        entry = self.store.get("42367")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.user_id, 123)
        self.assertEqual(entry.source_title, "Mr Beast")
        archived_path = Path(entry.video_path)
        self.assertTrue(archived_path.is_file())
        self.assertEqual(archived_path.read_bytes(), b"fake video bytes")
        # Lives outside the job's own dir, so job-retention cleanup can't take it.
        self.assertTrue(str(archived_path).startswith(str(self.library_dir)))

    async def test_no_video_in_the_job_dir_is_a_silent_no_op(self) -> None:
        job_dir = Path(self._tempdir.name) / "jobs" / "1"
        job_dir.mkdir(parents=True)
        job = {"job_dir": str(job_dir), "user_id": 123, "source_title": "", "target_lang": "ru"}
        await _archive_finished_dub(self._context(), job)
        self.assertIsNone(self.store.get("1"))

    async def test_missing_library_store_is_a_silent_no_op(self) -> None:
        job_dir = Path(self._tempdir.name) / "jobs" / "42367"
        job_dir.mkdir(parents=True)
        job = self._job(job_dir)
        context = self._context()
        context.application.bot_data.pop("library_store")
        await _archive_finished_dub(context, job)
        self.assertEqual(list(self.library_dir.glob("*")), [])


if __name__ == "__main__":
    unittest.main()
