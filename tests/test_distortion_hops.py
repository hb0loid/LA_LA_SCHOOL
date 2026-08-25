from __future__ import annotations

import unittest
from pathlib import Path

from laladub.models import DubConfig
from laladub.pipeline import _translation_distortion_chains

# The live configuration, including the variants that used to cost 5-7 calls.
LIVE_PIVOTS = (
    "input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar"
    "|input,en,de|input,ja,ko,en|input,tr,ar,en|en,ms,he,en"
)


def _config(hops: int, pivots: str = LIVE_PIVOTS) -> DubConfig:
    return DubConfig(
        output=Path("out.mp4"),
        workdir=Path("."),
        target_lang="ru",
        source_lang="vi",
        translation_pivots=pivots,
        max_translation_hops=hops,
    )


class DistortionHopCapTests(unittest.TestCase):
    def test_no_chain_exceeds_the_cap(self) -> None:
        chains = _translation_distortion_chains(_config(3))
        self.assertTrue(chains)
        for chain in chains:
            with self.subTest(chain=" -> ".join(chain)):
                self.assertLessEqual(len(chain) - 1, 3)

    def test_long_variants_are_trimmed_not_discarded(self) -> None:
        # Variety is the point of distortion, so a 5-hop variant should come
        # back shortened rather than vanish and leave fewer options.
        uncapped = _translation_distortion_chains(_config(99))
        capped = _translation_distortion_chains(_config(3))
        self.assertGreater(max(len(c) - 1 for c in uncapped), 3)
        self.assertEqual(len(capped), len(uncapped))

    def test_chains_still_start_and_end_at_the_target_language(self) -> None:
        for chain in _translation_distortion_chains(_config(3)):
            with self.subTest(chain=" -> ".join(chain)):
                self.assertEqual(chain[0], "ru")
                self.assertEqual(chain[-1], "ru")

    def test_cap_lowers_the_average_cost_per_line(self) -> None:
        uncapped = _translation_distortion_chains(_config(99))
        capped = _translation_distortion_chains(_config(3))
        before = sum(len(c) - 1 for c in uncapped) / len(uncapped)
        after = sum(len(c) - 1 for c in capped) / len(capped)
        self.assertLess(after, before)

    def test_cap_never_collapses_a_chain_below_one_pivot(self) -> None:
        # A chain has to leave the target language and come back, otherwise
        # there is no distortion left to speak of.
        for chain in _translation_distortion_chains(_config(2)):
            with self.subTest(chain=" -> ".join(chain)):
                self.assertGreaterEqual(len(chain), 3)

    def test_already_short_config_is_untouched(self) -> None:
        chains = _translation_distortion_chains(_config(3, "en,de|en,fr"))
        self.assertEqual([" -> ".join(c) for c in chains], ["ru -> en -> de -> ru", "ru -> en -> fr -> ru"])


if __name__ == "__main__":
    unittest.main()
