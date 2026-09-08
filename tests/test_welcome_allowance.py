from __future__ import annotations

import unittest

from laladub.bot import daily_limit_ms
from laladub.karma import KARMA_LEVELS, PREMIUM_LEVEL, WELCOME_ALLOWANCE_MINUTES

NEWCOMER = KARMA_LEVELS[0]
WELCOME_MS = WELCOME_ALLOWANCE_MINUTES * 60_000


class WelcomeAllowanceTests(unittest.TestCase):
    """A newcomer gets one minute a day, which is one video - and a first video
    is often the one that goes wrong. Measured: they hit the wall on 41% of
    their days, against 2-7% for everyone above them. This is a one-off
    allowance, not a raised daily limit: a few minutes per person ever rather
    than every day forever."""

    def test_someone_brand_new_gets_the_whole_allowance(self) -> None:
        limit = daily_limit_ms(NEWCOMER, lifetime_used_ms=0)
        self.assertEqual(limit, NEWCOMER.daily_minutes * 60_000 + WELCOME_MS)

    def test_it_is_spent_as_it_is_used(self) -> None:
        half = WELCOME_MS // 2
        limit = daily_limit_ms(NEWCOMER, lifetime_used_ms=half)
        self.assertEqual(limit, NEWCOMER.daily_minutes * 60_000 + WELCOME_MS - half)

    def test_once_gone_it_never_comes_back(self) -> None:
        for used in (WELCOME_MS, WELCOME_MS * 10):
            limit = daily_limit_ms(NEWCOMER, lifetime_used_ms=used)
            self.assertEqual(limit, NEWCOMER.daily_minutes * 60_000)

    def test_it_applies_to_every_level_but_matters_only_at_the_bottom(self) -> None:
        """Anyone above the first rung has used far more than it long ago."""
        for level in KARMA_LEVELS[1:] + (PREMIUM_LEVEL,):
            self.assertEqual(
                daily_limit_ms(level, lifetime_used_ms=WELCOME_MS),
                level.daily_minutes * 60_000,
            )

    def test_a_negative_history_cannot_inflate_it(self) -> None:
        self.assertEqual(
            daily_limit_ms(NEWCOMER, lifetime_used_ms=-1000),
            NEWCOMER.daily_minutes * 60_000 + WELCOME_MS,
        )

    def test_the_ladder_itself_is_untouched(self) -> None:
        """The allowance buys a look at the product, not a shortcut up."""
        self.assertEqual(NEWCOMER.daily_minutes, 1)
        self.assertEqual([level.daily_minutes for level in KARMA_LEVELS[:3]], [1, 5, 10])


if __name__ == "__main__":
    unittest.main()
