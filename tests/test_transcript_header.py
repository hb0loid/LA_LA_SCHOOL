from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.bot import _write_transcript_text
from laladub.languages import SOURCE_LANGS, TARGET_LANGS, transcript_header


class TranscriptHeaderTests(unittest.TestCase):
    def test_both_languages_are_named(self) -> None:
        # The bot shows this while a job runs and then it is gone; afterwards
        # nothing said what a given video had been dubbed from.
        said = transcript_header({"source_lang": "vi", "target_lang": "ru"})
        self.assertEqual(said, "[Вьетнамский → Русский]")

    def test_auto_is_spelled_out(self) -> None:
        said = transcript_header({"source_lang": "auto", "target_lang": "uk"})
        self.assertEqual(said, "[Любой язык → Украинский]")

    def test_a_missing_source_reads_as_auto(self) -> None:
        self.assertIn("Любой язык", transcript_header({"target_lang": "ru"}))

    def test_no_job_gives_no_header(self) -> None:
        self.assertEqual(transcript_header(None), "")
        self.assertEqual(transcript_header({}), "")

    def test_an_unknown_code_is_shown_rather_than_hidden(self) -> None:
        self.assertIn("zz", transcript_header({"source_lang": "zz", "target_lang": "ru"}))


class TranscriptFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.job_dir = Path(self._tempdir.name)

    def test_the_header_is_the_first_line(self) -> None:
        path = _write_transcript_text(
            self.job_dir, "видео", "Первая реплика.", {"source_lang": "vi", "target_lang": "ru"}
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "[Вьетнамский → Русский]")
        self.assertIn("Первая реплика.", "\n".join(lines))

    def test_without_a_job_the_file_is_unchanged(self) -> None:
        path = _write_transcript_text(self.job_dir, "видео", "Первая реплика.")
        self.assertEqual(path.read_text(encoding="utf-8").strip(), "Первая реплика.")


class SharedLanguageListTests(unittest.TestCase):
    def test_the_lists_live_in_one_place(self) -> None:
        # They were in bot.py, which job_runner deliberately does not import.
        # Duplicating them is how settings drift apart - see the artifact ratio,
        # hardcoded in three places and silently ignoring its own setting.
        from laladub import bot

        self.assertIs(bot.SOURCE_LANGS, SOURCE_LANGS)
        self.assertIs(bot.TARGET_LANGS, TARGET_LANGS)


if __name__ == "__main__":
    unittest.main()
