from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from laladub import pipeline
from laladub.models import DubConfig, Segment
from laladub.pipeline import _fill_sparse_dub_segments_safely
from laladub.translation import TranslationError


def _config() -> DubConfig:
    return DubConfig(output=Path("out.mp4"), workdir=Path("."), target_lang="ru")


def _segments() -> list[Segment]:
    return [Segment(start=0.0, end=1.0, text="привет", translated_text="привет")]


class SparseFillGuardTests(unittest.TestCase):
    def test_a_missing_argos_route_no_longer_kills_the_job(self) -> None:
        # Regression: Whisper detected the fallback language as nn, no local
        # route to the target existed, and the TranslationError propagated all
        # the way out of run_dub - the whole dub lost over an optional top-up.
        original = _segments()
        error = TranslationError(
            "Argos package is missing for nn->id. Auto-install could not create "
            "a direct or nn->en->id route."
        )
        with patch.object(pipeline, "_fill_sparse_dub_segments", side_effect=error):
            result = _fill_sparse_dub_segments_safely(original, _config(), Path("a.wav"), 10.0)
        self.assertEqual(result, original)

    def test_any_other_failure_is_survived_too(self) -> None:
        original = _segments()
        with patch.object(pipeline, "_fill_sparse_dub_segments", side_effect=RuntimeError("boom")):
            result = _fill_sparse_dub_segments_safely(original, _config(), Path("a.wav"), 10.0)
        self.assertEqual(result, original)

    def test_a_successful_fill_is_passed_through(self) -> None:
        filled = [*_segments(), Segment(start=1.0, end=2.0, text="ещё", translated_text="ещё")]
        with patch.object(pipeline, "_fill_sparse_dub_segments", return_value=filled):
            result = _fill_sparse_dub_segments_safely(_segments(), _config(), Path("a.wav"), 10.0)
        self.assertEqual(result, filled)

    def test_the_failure_is_reported_rather_than_swallowed_silently(self) -> None:
        with patch.object(pipeline, "_fill_sparse_dub_segments", side_effect=RuntimeError("boom")), \
             patch("builtins.print") as printed:
            _fill_sparse_dub_segments_safely(_segments(), _config(), Path("a.wav"), 10.0)
        said = " ".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn("Sparse fill skipped", said)
        self.assertIn("boom", said)


if __name__ == "__main__":
    unittest.main()
