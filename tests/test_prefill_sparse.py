from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from laladub import pipeline
from laladub.models import DubConfig, Segment
from laladub.pipeline import _prefill_sparse_source_segments


def _config() -> DubConfig:
    return DubConfig(
        output=Path("out.mp4"),
        workdir=Path("."),
        target_lang="ru",
        source_lang="vi",
        input_pivot_lang="en",
    )


def _thin() -> list[Segment]:
    # One short line over two minutes: what the sparse check is for.
    return [Segment(start=0.0, end=2.0, text="привет")]


def _fallback() -> list[Segment]:
    # Covers the full 120 seconds with no gap long enough to look sparse.
    return [Segment(start=float(n) * 5, end=float(n) * 5 + 5, text=f"реплика {n}") for n in range(0, 24)]


class PrefillSparseTests(unittest.TestCase):
    def test_a_thin_transcript_is_topped_up_before_translation(self) -> None:
        # The whole point: the fill lands while the text is still untranslated,
        # so one translation pass covers everything.
        with patch.object(
            pipeline, "_stable_fallback_source_asr", return_value=(_fallback(), "en")
        ):
            result = _prefill_sparse_source_segments(_thin(), _config(), Path("a.wav"), 120.0)
        self.assertGreater(len(result), len(_thin()))

    def test_nothing_is_translated_by_the_prefill_itself(self) -> None:
        with patch.object(
            pipeline, "_stable_fallback_source_asr", return_value=(_fallback(), "en")
        ), patch.object(pipeline, "_translate_dub_segments") as translate:
            _prefill_sparse_source_segments(_thin(), _config(), Path("a.wav"), 120.0)
        translate.assert_not_called()

    def test_a_healthy_transcript_is_left_alone(self) -> None:
        healthy = _fallback()
        with patch.object(pipeline, "_stable_fallback_source_asr") as asr:
            result = _prefill_sparse_source_segments(healthy, _config(), Path("a.wav"), 120.0)
        asr.assert_not_called()
        self.assertIs(result, healthy)

    def test_a_fallback_that_is_also_thin_changes_nothing(self) -> None:
        thin = _thin()
        with patch.object(
            pipeline, "_stable_fallback_source_asr", return_value=([], None)
        ):
            result = _prefill_sparse_source_segments(thin, _config(), Path("a.wav"), 120.0)
        self.assertIs(result, thin)

    def test_a_failing_fallback_asr_does_not_break_the_job(self) -> None:
        # It is an extra; the post-translation fill still gets its turn.
        thin = _thin()
        with patch.object(
            pipeline, "_stable_fallback_source_asr", side_effect=RuntimeError("boom")
        ):
            result = _prefill_sparse_source_segments(thin, _config(), Path("a.wav"), 120.0)
        self.assertIs(result, thin)

    def test_a_detected_language_fills_in_an_unknown_source(self) -> None:
        config = _config()
        config.source_lang = None
        with patch.object(
            pipeline, "_stable_fallback_source_asr", return_value=(_fallback(), "en")
        ):
            _prefill_sparse_source_segments(_thin(), config, Path("a.wav"), 120.0)
        self.assertEqual(config.source_lang, "en")

    def test_a_known_source_language_is_not_overwritten(self) -> None:
        config = _config()
        with patch.object(
            pipeline, "_stable_fallback_source_asr", return_value=(_fallback(), "en")
        ):
            _prefill_sparse_source_segments(_thin(), config, Path("a.wav"), 120.0)
        self.assertEqual(config.source_lang, "vi")

    def test_it_is_skipped_when_sparse_fill_is_off(self) -> None:
        config = _config()
        config.input_pivot_lang = None
        config.artifact_chaos_mode = False
        thin = _thin()
        with patch.object(pipeline, "_stable_fallback_source_asr") as asr:
            result = _prefill_sparse_source_segments(thin, config, Path("a.wav"), 120.0)
        asr.assert_not_called()
        self.assertIs(result, thin)


if __name__ == "__main__":
    unittest.main()
