"""Borrowed artifacts are translated from the language they are really in."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from laladub import pipeline
from laladub.hallucination_catalog import HallucinationCatalog
from laladub.models import DubConfig, Segment


def _config(tmp: Path) -> DubConfig:
    return DubConfig(output=tmp / "out.mp4", workdir=tmp / "work", target_lang="ru", source_lang="vi")


class CatalogueLanguageTests(unittest.TestCase):
    def test_pick_keeps_the_language_of_each_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phrases.csv"
            rows = ["lang,phrase,count"]
            rows += [f'vi,"вьетнамская фраза номер {n}",1' for n in range(30)]
            rows += [f'hi,"अश स ब स ग क च ब स {n}",1' for n in range(30)]
            path.write_text("\n".join(rows), encoding="utf-8")
            picked = HallucinationCatalog(path).pick("vi", 10, seed="s", cross_language_share=1.0)
        self.assertTrue(picked)
        for item in picked:
            expected = "hi" if "अश" in item.phrase else "vi"
            self.assertEqual(item.lang, expected)


class ArtifactTranslationTests(unittest.TestCase):
    def test_each_artifact_is_translated_from_its_own_language(self) -> None:
        seen: dict[str, list[str]] = {}

        def fake_translate(segments, config):
            seen.setdefault(config.source_lang, []).extend(s.text for s in segments)
            for segment in segments:
                segment.translated_text = f"ru({segment.text})"
            return segments

        artifacts = [
            Segment(0, 1, "अश स ब स", source_lang="hi"),
            Segment(2, 3, "cảm ơn các bạn", source_lang="vi"),
            Segment(4, 5, "thanks for watching"),
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "translate_segments", fake_translate):
            pipeline._translate_artifact_segments(artifacts, _config(Path(tmp)))
        self.assertEqual(seen["hi"], ["अश स ब स"])
        self.assertEqual(seen["vi"], ["cảm ơn các bạn"])
        self.assertEqual(seen["en"], ["thanks for watching"])

    def test_an_untranslatable_language_is_dropped_not_fatal(self) -> None:
        def fake_translate(segments, config):
            if config.source_lang == "hi":
                raise RuntimeError("no route")
            for segment in segments:
                segment.translated_text = "ок"
            return segments

        artifacts = [Segment(0, 1, "अश स ब स", source_lang="hi"), Segment(2, 3, "xin chào", source_lang="vi")]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pipeline, "translate_segments", fake_translate):
            kept = pipeline._translate_artifact_segments(artifacts, _config(Path(tmp)))
        self.assertEqual([s.text for s in kept], ["xin chào"])

    def test_a_phrase_still_in_a_foreign_script_is_dropped(self) -> None:
        self.assertTrue(pipeline._spoken_in_foreign_script("अश स ब स ग क", "ru"))
        self.assertTrue(pipeline._spoken_in_foreign_script("ПОЙТЕ МОЛИТВУ अश", "ru"))
        self.assertFalse(pipeline._spoken_in_foreign_script("Я люблю их всех", "ru"))
        self.assertFalse(pipeline._spoken_in_foreign_script("la la school để không bỏ lỡ", "ru"))
        self.assertFalse(pipeline._spoken_in_foreign_script("Ghiền Mì Gõ — школа", "ru"))
        # A target written in another script keeps its own letters.
        self.assertFalse(pipeline._spoken_in_foreign_script("こんにちは", "ja"))


if __name__ == "__main__":
    unittest.main()
