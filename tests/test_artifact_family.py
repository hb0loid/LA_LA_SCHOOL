from __future__ import annotations

import unittest

from laladub.pipeline import _artifact_family


class SubscribeFamilyTests(unittest.TestCase):
    """Whisper's channel-outro hallucinations arrive in many phrasings.

    They all have to land in one family, because the per-family cap is what
    stops a single video from carrying four near-identical "subscribe" lines -
    which is exactly what job 45953 ended up with.
    """

    def test_all_subscribe_phrasings_share_one_family(self) -> None:
        # Verbatim from job 45953, where these four ran within 25 seconds.
        phrasings = [
            "Спасибо, что подписались и увидимся снова.",
            "Не забудьте поставить лайк, поделиться и подписаться, чтобы поддержать мой канал",
            "Пожалуйста, подпишитесь на канал, чтобы поддержать мой канал.",
            "Подпишитесь на канал Ghiền Mì Gõ, чтобы не пропустить новые видео.",
            "Подпишись на канал!",
            "Please subscribe to my channel",
            "Bitte abonnieren nicht vergessen",
            "Hãy đăng ký kênh Ghiền Mì Gõ",
        ]
        families = {_artifact_family(text) for text in phrasings}
        self.assertEqual(families, {"subscribe"})

    def test_other_families_are_untouched(self) -> None:
        for text, expected in [
            ("Продолжение следует...", "continued"),
            ("Субтитры сделал DimaTorzok", "subtitle_credit"),
            ("Девушки отдыхают", "girls_resting"),
            ("Редактор субтитров Егорова", "editor_credit"),
            ("Спасибо за субтитры", "thanks_subtitles"),
            ("Подогнал субтитры под видео", "subtitle_sync"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(_artifact_family(text), expected)

    def test_ordinary_dialogue_is_not_a_subscribe_artifact(self) -> None:
        for text in ("Это Бургер Кинг", "Чей это бургер?", "That's Red Robin"):
            with self.subTest(text=text):
                self.assertNotEqual(_artifact_family(text), "subscribe")


if __name__ == "__main__":
    unittest.main()
