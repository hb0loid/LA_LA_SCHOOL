from __future__ import annotations

import unittest

from laladub.karma import PREMIUM_KARMA_SHARE, karma_milli_for_duration


class PremiumKarmaTests(unittest.TestCase):
    """Premium already grants more than the top of the karma ladder, so karma
    earned while it is active does nothing until the subscription lapses - and
    then locks in a rank bought rather than climbed. The videos are real, so the
    karma is not withheld, only counted at the shame channel's reduced rate."""

    def test_a_premium_author_earns_the_reduced_share(self) -> None:
        full = karma_milli_for_duration(600_000, "main")
        reduced = karma_milli_for_duration(600_000, "main", premium=True)
        self.assertEqual(reduced, round(full * PREMIUM_KARMA_SHARE))
        self.assertLess(reduced, full)

    def test_an_ordinary_author_is_unchanged(self) -> None:
        self.assertEqual(
            karma_milli_for_duration(600_000, "main"),
            karma_milli_for_duration(600_000, "main", premium=False),
        )

    def test_the_reduction_applies_to_the_shame_channel_too(self) -> None:
        """Otherwise the cheaper channel would become the better one to farm."""
        full = karma_milli_for_duration(600_000, "shame")
        reduced = karma_milli_for_duration(600_000, "shame", premium=True)
        self.assertLess(reduced, full)

    def test_a_rejected_video_still_earns_nothing(self) -> None:
        self.assertEqual(karma_milli_for_duration(600_000, "rejected", premium=True), 0)

    def test_karma_is_reduced_not_removed(self) -> None:
        """The channel did receive the video; withholding it entirely would
        punish the one person who pays and also works."""
        self.assertGreater(karma_milli_for_duration(600_000, "main", premium=True), 0)


if __name__ == "__main__":
    unittest.main()
