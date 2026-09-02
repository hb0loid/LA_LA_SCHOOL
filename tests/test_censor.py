from __future__ import annotations

import re
import unittest

from laladub.censor import censor_text


class CensorBankTests(unittest.TestCase):
    def test_new_russian_insult_inflections_are_censored(self) -> None:
        text = "шлюхам педиком педерастами проститутками"
        censored, count = censor_text(text, percent=100, seed="new-terms")
        self.assertEqual(count, 4)
        for source in text.split():
            self.assertNotIn(source, censored.casefold())

    def test_new_transliterations_are_censored(self) -> None:
        censored, count = censor_text(
            "shlyukha pedik pederast shalava",
            percent=100,
            seed="translit",
        )
        self.assertEqual(count, 4)
        self.assertNotIn("shlyukha", censored.casefold())
        self.assertNotIn("pedik", censored.casefold())

    def test_short_matches_usually_receive_short_replacements(self) -> None:
        short = 0
        total = 200
        for index in range(total):
            censored, count = censor_text("шлюха", percent=100, seed=f"short-{index}")
            self.assertEqual(count, 1)
            if len(re.findall(r"[\wёЁ]+", censored, flags=re.UNICODE)) <= 2:
                short += 1
        self.assertGreaterEqual(short, 170)


if __name__ == "__main__":
    unittest.main()
