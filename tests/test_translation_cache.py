from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laladub.translation as translation
from laladub.models import DubConfig
from laladub.translation import (
    _looks_bad_machine_translation,
    _looks_like_translator_error,
    _translate_hybrid_text,
    _translation_cache_get,
)

RATE_LIMIT_REPLY = "QUERY LENGTH LIMIT EXCEEDED. MAX ALLOWED QUERY: 500 CHARS"
RU_LIMIT_REPLY = "Превышен лимит длины запроса. Максимально допустимое количество запросов: 500 символов"


class TranslatorErrorDetectionTests(unittest.TestCase):
    def test_recognises_mymemory_limit_replies(self) -> None:
        for text in (RATE_LIMIT_REPLY, RU_LIMIT_REPLY):
            with self.subTest(text=text[:30]):
                self.assertTrue(_looks_like_translator_error(text))
                self.assertTrue(_looks_bad_machine_translation(text))

    def test_ordinary_translations_are_not_errors(self) -> None:
        for text in ("Установка Windows", "Whose burger is this?", "Пошли!"):
            with self.subTest(text=text):
                self.assertFalse(_looks_like_translator_error(text))
                self.assertFalse(_looks_bad_machine_translation(text))


class TranslationCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        root = Path(self._tempdir.name)
        self.config = DubConfig(
            output=root / "out.mp4",
            workdir=root,
            media_cache_dir=root / "cache" / "media",
        )

    def test_repeated_phrase_is_translated_once(self) -> None:
        calls: list[str] = []

        def fake(text: str, source: str, target: str) -> str:
            calls.append(text)
            return f"translated:{text}"

        with patch.object(translation, "_translate_googleweb_text", side_effect=fake):
            first = _translate_hybrid_text("Привет", "ru", "en", self.config)
            second = _translate_hybrid_text("Привет", "ru", "en", self.config)

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_language_pair_is_part_of_the_key(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake(text: str, source: str, target: str) -> str:
            calls.append((source, target))
            return f"{target}:{text}"

        with patch.object(translation, "_translate_googleweb_text", side_effect=fake):
            _translate_hybrid_text("Привет", "ru", "en", self.config)
            _translate_hybrid_text("Привет", "ru", "de", self.config)

        self.assertEqual(calls, [("ru", "en"), ("ru", "de")])

    def test_rate_limit_reply_is_never_cached(self) -> None:
        # Caching a 429/limit answer would freeze one bad moment into every
        # later run of the same phrase.
        def limited(text: str, source: str, target: str) -> str:
            return RATE_LIMIT_REPLY

        with patch.object(translation, "_translate_googleweb_text", side_effect=limited), patch.object(
            translation, "_translate_mymemory_text", side_effect=limited
        ), patch.object(translation, "_translate_argos_provider_text", side_effect=lambda t, s, g: "запасной перевод"):
            result = _translate_hybrid_text("Другая фраза", "ru", "en", self.config)

        self.assertEqual(result, "запасной перевод")
        self.assertEqual(_translation_cache_get("Другая фраза", "ru", "en", self.config), "запасной перевод")

    def test_cache_survives_a_new_config_object(self) -> None:
        with patch.object(translation, "_translate_googleweb_text", side_effect=lambda t, s, g: "готово"):
            _translate_hybrid_text("Фраза", "ru", "en", self.config)

        reopened = DubConfig(
            output=self.config.output,
            workdir=self.config.workdir,
            media_cache_dir=self.config.media_cache_dir,
        )
        self.assertEqual(_translation_cache_get("Фраза", "ru", "en", reopened), "готово")

    def test_missing_cache_dir_disables_caching_quietly(self) -> None:
        config = DubConfig(output=Path("out.mp4"), workdir=Path("."), media_cache_dir=None)
        calls: list[str] = []

        def fake(text: str, source: str, target: str) -> str:
            calls.append(text)
            return "ok"

        with patch.object(translation, "_translate_googleweb_text", side_effect=fake):
            _translate_hybrid_text("Привет", "ru", "en", config)
            _translate_hybrid_text("Привет", "ru", "en", config)

        # No cache available, so both calls go through - but nothing blows up.
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
