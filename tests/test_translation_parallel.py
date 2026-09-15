from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laladub.translation as translation
from laladub.models import DubConfig, Segment


class HybridTranslationPoolTests(unittest.TestCase):
    def test_parallel_translation_preserves_segment_order(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            config = DubConfig(
                output=root / "out.mp4",
                workdir=root / "work",
                source_lang="vi",
                target_lang="ru",
                translator="hybrid",
            )
            segments = [
                Segment(start=0, end=1, text="one"),
                Segment(start=1, end=2, text="two"),
                Segment(start=2, end=3, text="three"),
            ]

            def fake(text: str, source: str, target: str, _config: DubConfig) -> str:
                return f"{text}:{source}->{target}"

            with patch.dict(os.environ, {"LALADUB_TRANSLATION_WORKERS": "3"}), patch.object(
                translation, "_translate_hybrid_text", side_effect=fake
            ):
                translated = translation.translate_segments(segments, config)

            self.assertEqual(
                [segment.translated_text for segment in translated],
                ["one:vi->ru", "two:vi->ru", "three:vi->ru"],
            )


if __name__ == "__main__":
    unittest.main()
