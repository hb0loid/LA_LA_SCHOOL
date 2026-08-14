from __future__ import annotations

import unittest

from laladub.models import Segment
from laladub.quality import limit_repeated_segment_phrases, suppress_pathological_segment_loops


def _segments(texts: list[str], *, step: float = 1.0) -> list[Segment]:
    return [
        Segment(start=index * step, end=(index + 1) * step, text=text, translated_text=text)
        for index, text in enumerate(texts)
    ]


class PathologicalSegmentLoopTests(unittest.TestCase):
    def test_collapses_one_word_whisper_storm(self) -> None:
        cleaned = suppress_pathological_segment_loops(_segments(["Птица"] * 40))
        self.assertEqual([segment.spoken_text for segment in cleaned], ["Птица"] * 6)

    def test_collapses_low_diversity_alternating_loop(self) -> None:
        cleaned = suppress_pathological_segment_loops(
            _segments((["Конец", "Джо", "Последний"] * 12))
        )
        counts = {word: sum(segment.spoken_text == word for segment in cleaned) for word in ("Конец", "Джо", "Последний")}
        self.assertEqual(counts, {"Конец": 5, "Джо": 5, "Последний": 5})

    def test_keeps_sparse_repeated_joke(self) -> None:
        texts = ["Азкабан", "Новая фраза один", "Новая фраза два", "Новая фраза три"] * 5
        cleaned = suppress_pathological_segment_loops(_segments(texts, step=4.0))
        self.assertEqual([segment.spoken_text for segment in cleaned], texts)

    def test_keeps_short_natural_exchange(self) -> None:
        texts = ["Да", "Нет", "Да", "Нет", "Да", "Нет"]
        cleaned = suppress_pathological_segment_loops(_segments(texts))
        self.assertEqual([segment.spoken_text for segment in cleaned], texts)

    def test_caps_only_large_global_artifact_flood(self) -> None:
        flooded = limit_repeated_segment_phrases(_segments(["Большое спасибо"] * 12, step=10.0))
        self.assertEqual(len(flooded), 3)

        ordinary = limit_repeated_segment_phrases(_segments(["Большое спасибо"] * 4, step=10.0))
        self.assertEqual(len(ordinary), 4)


if __name__ == "__main__":
    unittest.main()
