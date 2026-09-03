from __future__ import annotations

import unittest

from laladub.bot import SOURCE_LANGS, TARGET_LANGS
from laladub.translation import ARGOS_LANG_ALIASES, _argos_lang


class SourceLanguageTests(unittest.TestCase):
    def test_auto_is_offered_first(self) -> None:
        self.assertEqual(SOURCE_LANGS[0][0], "auto")

    def test_familiar_artifact_languages_are_pinned_after_auto(self) -> None:
        self.assertEqual([code for code, _label in SOURCE_LANGS[:3]], ["auto", "vi", "ru"])

    def test_codes_are_unique(self) -> None:
        codes = [code for code, _label in SOURCE_LANGS]
        self.assertEqual(len(codes), len(set(codes)))

    def test_labels_are_unique(self) -> None:
        # Two identical entries in a 46-button list are impossible to tell apart.
        labels = [label for _code, label in SOURCE_LANGS]
        self.assertEqual(len(labels), len(set(labels)))

    def test_whisper_can_transcribe_every_offered_language(self) -> None:
        try:
            from whisper.tokenizer import LANGUAGES as whisper_languages
        except Exception:  # pragma: no cover - depends on the optional install
            self.skipTest("whisper не установлен")
        for code, label in SOURCE_LANGS:
            if code == "auto":
                continue
            with self.subTest(language=f"{code} ({label})"):
                self.assertIn(code, whisper_languages)

    def test_every_offered_language_has_a_local_translator(self) -> None:
        # The lesson from Malay: a language in this menu with no local route is
        # a job that dies the moment the online translator is rate-limited.
        try:
            import laladub.translation  # noqa: F401 - sets ARGOS_CHUNK_TYPE
            import argostranslate.package as pkg
        except Exception:  # pragma: no cover - depends on the optional install
            self.skipTest("argostranslate не установлен")
        installed = {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}
        if not installed:
            self.skipTest("пакеты argos не установлены в этом окружении")
        for code, label in SOURCE_LANGS:
            if code == "auto":
                continue
            argos_code = _argos_lang(code)
            with self.subTest(language=f"{code} ({label})"):
                # Argos routes everything through English, so a round trip out
                # of the language is what a source language needs.
                self.assertTrue(
                    (argos_code, "en") in installed or argos_code == "en",
                    f"нет пакета {argos_code}->en",
                )

    def test_every_target_language_has_a_local_translator(self) -> None:
        try:
            import laladub.translation  # noqa: F401
            import argostranslate.package as pkg
        except Exception:  # pragma: no cover
            self.skipTest("argostranslate не установлен")
        installed = {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}
        if not installed:
            self.skipTest("пакеты argos не установлены в этом окружении")
        for code, label in TARGET_LANGS:
            argos_code = _argos_lang(code)
            with self.subTest(language=f"{code} ({label})"):
                self.assertTrue(
                    ("en", argos_code) in installed or argos_code == "en",
                    f"нет пакета en->{argos_code}",
                )


class ArgosLanguageAliasTests(unittest.TestCase):
    def test_norwegian_is_translated_to_the_argos_spelling(self) -> None:
        # Whisper says "no", the package is filed under "nb". Without this the
        # language was detected fine and then had no translator at all.
        self.assertEqual(_argos_lang("no"), "nb")

    def test_nynorsk_is_folded_into_norwegian(self) -> None:
        # "nn" is a frequent false positive on music, and used to kill the job.
        self.assertEqual(_argos_lang("nn"), "nb")

    def test_an_unaliased_code_is_left_alone(self) -> None:
        self.assertEqual(_argos_lang("vi"), "vi")

    def test_case_and_spacing_do_not_defeat_the_alias(self) -> None:
        self.assertEqual(_argos_lang(" NO "), "nb")

    def test_an_empty_code_does_not_raise(self) -> None:
        self.assertEqual(_argos_lang(""), "")

    def test_no_alias_points_at_another_alias(self) -> None:
        # One hop only - a chain would depend on dict ordering.
        for target in ARGOS_LANG_ALIASES.values():
            with self.subTest(target=target):
                self.assertNotIn(target, ARGOS_LANG_ALIASES)


if __name__ == "__main__":
    unittest.main()
