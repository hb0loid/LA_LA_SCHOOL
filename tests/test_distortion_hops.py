from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from laladub import pipeline
from laladub.models import DubConfig
from laladub.pipeline import _translation_distortion_chains

# The live configuration, including the variants that used to cost 5-7 calls.
LIVE_PIVOTS = (
    "input,en|input,ja,en|input,tr,de,en|en,de|en,fr|en,es|en,ja,ko|en,tr,ar"
    "|input,en,de|input,ja,ko,en|input,tr,ar,en|en,th,he,en"
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


class UnparseablePivotTests(unittest.TestCase):
    def test_a_chain_through_malay_is_dropped(self) -> None:
        # Argos has no sentence splitter for Malay, so this chain could only
        # ever run online. Once the online side hit its rate limit the whole
        # chain collapsed and the line was replaced with mangled text - 253 of
        # the 256 collapses in the logs came from exactly this variant.
        chains = _translation_distortion_chains(_config(9, "en,ms,he,en"))
        self.assertEqual(chains, [])

    def test_a_chain_through_azerbaijani_is_dropped(self) -> None:
        chains = _translation_distortion_chains(_config(9, "en,az,en"))
        self.assertEqual(chains, [])

    def test_the_thai_replacement_survives(self) -> None:
        chains = _translation_distortion_chains(_config(9, "en,th,he,en"))
        self.assertEqual(chains, [["ru", "en", "th", "he", "en", "ru"]])

    def test_dubbing_into_an_unparseable_language_distorts_nothing(self) -> None:
        # A chain starts at the target and comes back to it, so dubbing into
        # Malay means reading Malay on the first hop. Leaving the translation
        # undistorted is the honest outcome; mangled text is not.
        config = _config(9, "en,de")
        config.target_lang = "ms"
        self.assertEqual(_translation_distortion_chains(config), [])

    def test_the_live_configuration_loses_no_variant_to_the_filter(self) -> None:
        with patch.object(pipeline, "UNPARSEABLE_PIVOT_LANGS", frozenset()):
            unfiltered = _translation_distortion_chains(_config(3))
        self.assertEqual(_translation_distortion_chains(_config(3)), unfiltered)


class ShippedPivotDefaultsTests(unittest.TestCase):
    def test_no_shipped_default_pivots_through_an_unparseable_language(self) -> None:
        # The Malay chain was written into six places - two launch scripts, the
        # settings default, the CLI default, and both job builders. Fixing one
        # left the worker, which is what actually runs the translation, still
        # on the broken chain. Any copy that drifts back trips this.
        root = Path(__file__).resolve().parent.parent
        offenders: list[str] = []
        for path in [*root.glob("src/laladub/*.py"), *root.glob("tools/**/*.ps1"), *root.glob("*.ps1")]:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if "|" not in line or "en," not in line:
                    continue
                for variant in line.split("|"):
                    hops = [hop.strip().strip("\"'") for hop in variant.split(",")]
                    # Only the hops a chain reads out of matter; the last one is
                    # the target it lands on.
                    if any(hop in pipeline.UNPARSEABLE_PIVOT_LANGS for hop in hops[:-1]):
                        offenders.append(f"{path.relative_to(root)}:{number}")
                        break
        self.assertEqual(offenders, [], f"pivot chains through unparseable languages: {offenders}")

