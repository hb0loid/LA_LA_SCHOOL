from __future__ import annotations

import unittest

from laladub.translation import _postprocess_translated_text, _repair_mojibake

# Verbatim from job 45691, where MyMemory answered a Whisper hallucination with
# a translation-memory hit that had been uploaded as CP1251 bytes.
REAL_MOJIBAKE = (
    "Óñòàíîâêà Windows: 0. Âûêëþ÷èòå Àíòèâèðóñ 1. Çàïóñòèòå óñòàíîâùèê "
    "ArchiCAD-21-RUS-3010-1. 1. exe è óñòàíîâèòå ïðîãðàììó."
)


class RepairMojibakeTests(unittest.TestCase):
    def test_repairs_real_cp1251_payload(self) -> None:
        repaired = _repair_mojibake(REAL_MOJIBAKE)
        self.assertTrue(repaired.startswith("Установка Windows: 0. Выключите Антивирус"))
        self.assertIn("Запустите установщик", repaired)
        # Latin runs inside the text survive untouched.
        self.assertIn("ArchiCAD-21-RUS-3010-1", repaired)

    def test_leaves_healthy_russian_alone(self) -> None:
        text = "Установка Windows: выключите антивирус"
        self.assertEqual(_repair_mojibake(text), text)

    def test_leaves_plain_ascii_alone(self) -> None:
        text = "Hello world, this is a normal sentence"
        self.assertEqual(_repair_mojibake(text), text)

    def test_leaves_accented_latin_alone(self) -> None:
        # These re-decode into the odd stray Cyrillic letter, which is exactly
        # why the repair demands several whole Cyrillic words before firing.
        for text in (
            "Grüße aus München, schöne Straße",
            "El niño está en la montaña",
            "Cafe au lait, s'il vous plait — naïve coeur",
        ):
            with self.subTest(text=text):
                self.assertEqual(_repair_mojibake(text), text)

    def test_empty_text(self) -> None:
        self.assertEqual(_repair_mojibake(""), "")

    def test_text_outside_latin1_is_left_alone(self) -> None:
        text = "日本語のテキスト"
        self.assertEqual(_repair_mojibake(text), text)


class PostprocessIntegrationTests(unittest.TestCase):
    def test_translation_postprocessing_repairs_mojibake(self) -> None:
        repaired = _postprocess_translated_text(REAL_MOJIBAKE, "ru")
        self.assertTrue(repaired.startswith("Установка Windows"))

    def test_repair_also_applies_to_non_russian_targets(self) -> None:
        repaired = _postprocess_translated_text(REAL_MOJIBAKE, "en")
        self.assertTrue(repaired.startswith("Установка Windows"))

    def test_empty_translation_is_passed_through(self) -> None:
        self.assertEqual(_postprocess_translated_text("", "ru"), "")


if __name__ == "__main__":
    unittest.main()
