from __future__ import annotations

import unittest
from pathlib import Path

from laladub.models import DubConfig
from laladub.pipeline import OUR_CHANNEL_NAME, _rebrand_foreign_channel


def _config(share: float, seed: str = "seed") -> DubConfig:
    return DubConfig(
        output=Path("out.mp4"),
        workdir=Path("."),
        target_lang="ru",
        translation_seed=seed,
        channel_rebrand_share=share,
    )


ORIGINAL = "Подпишитесь на канал Ghiền Mì Gõ, чтобы не пропустить новые видео."


class ChannelRebrandTests(unittest.TestCase):
    def test_always_rebrands_at_full_share(self) -> None:
        result = _rebrand_foreign_channel(ORIGINAL, _config(1.0))
        self.assertIn(OUR_CHANNEL_NAME, result)
        self.assertNotIn("Ghiền", result)

    def test_never_rebrands_at_zero_share(self) -> None:
        self.assertEqual(_rebrand_foreign_channel(ORIGINAL, _config(0.0)), ORIGINAL)

    def test_the_rest_of_the_sentence_survives(self) -> None:
        result = _rebrand_foreign_channel(ORIGINAL, _config(1.0))
        self.assertEqual(result, f"Подпишитесь на канал {OUR_CHANNEL_NAME}, чтобы не пропустить новые видео.")

    def test_a_stuttered_name_collapses_into_one_mention(self) -> None:
        # Whisper latches onto the name and repeats it; the swap must leave one
        # clean mention, not four copies of ours.
        text = "Подпишитесь на канал Ghiền Mì Ghiền Mì Ghiền Mì Ghiền, чтобы не пропустить новые видео."
        result = _rebrand_foreign_channel(text, _config(1.0))
        self.assertEqual(result.count(OUR_CHANNEL_NAME), 1)
        self.assertNotIn("Ghiền", result)

    def test_latin_spelling_is_matched_too(self) -> None:
        result = _rebrand_foreign_channel("для канала Ghien Mi Go сегодня", _config(1.0))
        self.assertIn(OUR_CHANNEL_NAME, result)

    def test_text_without_the_name_is_untouched(self) -> None:
        text = "Подпишитесь на канал, чтобы получать больше информации."
        self.assertEqual(_rebrand_foreign_channel(text, _config(1.0)), text)

    def test_empty_text_is_untouched(self) -> None:
        self.assertEqual(_rebrand_foreign_channel("", _config(1.0)), "")

    def test_the_same_line_and_seed_decide_the_same_way(self) -> None:
        # A rerun of a job must not reshuffle which mentions were swapped.
        first = _rebrand_foreign_channel(ORIGINAL, _config(0.5))
        second = _rebrand_foreign_channel(ORIGINAL, _config(0.5))
        self.assertEqual(first, second)

    def test_half_share_lands_near_half(self) -> None:
        swapped = sum(
            OUR_CHANNEL_NAME in _rebrand_foreign_channel(f"{ORIGINAL} {n}", _config(0.5))
            for n in range(400)
        )
        self.assertGreater(swapped, 150)
        self.assertLess(swapped, 250)

    def test_a_higher_share_swaps_more_than_a_lower_one(self) -> None:
        def swapped(share: float) -> int:
            return sum(
                OUR_CHANNEL_NAME in _rebrand_foreign_channel(f"{ORIGINAL} {n}", _config(share))
                for n in range(400)
            )

        self.assertGreater(swapped(0.9), swapped(0.2))


if __name__ == "__main__":
    unittest.main()
