from __future__ import annotations

import unittest

from laladub.bot import _queued_detail


class QueuedDetailTests(unittest.TestCase):
    def test_languages_are_named_not_coded(self) -> None:
        # The line used to read "вход=vi, цель=ru, метод=..., ASR=openai-whisper
        # turbo, pivot=vi" - accurate and unreadable.
        said = _queued_detail({"source_lang": "vi", "speaker_count": "auto"}, "ru")
        self.assertIn("Вьетнамский", said)
        self.assertIn("Русский", said)
        self.assertNotIn("vi", said)

    def test_auto_source_is_spelled_out(self) -> None:
        self.assertIn("Любой язык", _queued_detail({"source_lang": "auto"}, "ru"))

    def test_a_missing_source_reads_as_auto(self) -> None:
        self.assertIn("Любой язык", _queued_detail({}, "ru"))

    def test_the_voice_count_is_included(self) -> None:
        self.assertIn("голоса: 2", _queued_detail({"source_lang": "ko", "speaker_count": 2}, "en"))

    def test_no_machinery_leaks_into_it(self) -> None:
        # Engine, ASR backend and pivot are not choices the user made - there
        # is one engine - and reading them told nobody anything.
        said = _queued_detail({"source_lang": "vi", "speaker_count": "auto"}, "ru")
        for noise in ("ASR", "pivot", "TTS", "whisper", "метод"):
            self.assertNotIn(noise, said)

    def test_an_unknown_code_is_shown_as_is_rather_than_crashing(self) -> None:
        self.assertIn("zz", _queued_detail({"source_lang": "zz"}, "ru"))


if __name__ == "__main__":
    unittest.main()
