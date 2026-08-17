from __future__ import annotations

import unittest

from laladub.models import Segment
from laladub.pipeline import (
    _dub_is_too_sparse,
    _largest_unspoken_stretch,
    _tighten_overlong_segments,
)


def _segment(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text, translated_text=text)


def _dense(count: int, *, step: float = 2.0) -> list[Segment]:
    return [
        _segment(index * step, index * step + step, "Обычная реплика примерно такой длины")
        for index in range(count)
    ]


class TightenOverlongSegmentsTests(unittest.TestCase):
    def test_shrinks_a_segment_that_claims_a_whole_silent_window(self) -> None:
        segments = [_segment(0.0, 29.98, "Девушки отдыхают"), _segment(30.0, 31.7, "Дальше речь")]
        tightened = _tighten_overlong_segments(segments)
        self.assertLess(tightened[0].end, 5.0)
        self.assertEqual(tightened[0].start, 0.0)
        # The neighbour is ordinary and must be left exactly as it was.
        self.assertEqual(tightened[1].end, 31.7)

    def test_leaves_ordinary_segments_alone(self) -> None:
        segments = _dense(6)
        self.assertIs(_tighten_overlong_segments(segments), segments)

    def test_keeps_text_and_speaker_metadata(self) -> None:
        original = Segment(
            start=0.0,
            end=30.0,
            text="исходник",
            translated_text="Девушки отдыхают",
            speaker_id="SPEAKER_01",
        )
        tightened = _tighten_overlong_segments([original])[0]
        self.assertEqual(tightened.text, "исходник")
        self.assertEqual(tightened.translated_text, "Девушки отдыхают")
        self.assertEqual(tightened.speaker_id, "SPEAKER_01")

    def test_is_idempotent_so_resume_does_not_redo_work(self) -> None:
        segments = [_segment(0.0, 29.98, "Девушки отдыхают"), _segment(30.0, 31.7, "Дальше речь")]
        once = _tighten_overlong_segments(segments)
        self.assertIs(_tighten_overlong_segments(once), once)

    def test_ignores_segments_with_no_text(self) -> None:
        segments = [Segment(start=0.0, end=30.0, text="", translated_text="")]
        self.assertIs(_tighten_overlong_segments(segments), segments)


class LargestUnspokenStretchTests(unittest.TestCase):
    def test_measures_a_hole_hidden_inside_one_long_segment(self) -> None:
        # The segment claims 30 seconds but says two words, so most of it is silent.
        segments = [_segment(0.0, 29.98, "Девушки отдыхают"), _segment(30.0, 32.0, "Дальше речь")]
        self.assertGreater(_largest_unspoken_stretch(segments, 32.0), 20.0)

    def test_measures_a_plain_gap_between_segments(self) -> None:
        segments = [_segment(0.0, 2.0, "Первая реплика"), _segment(40.0, 42.0, "Вторая реплика")]
        self.assertGreater(_largest_unspoken_stretch(segments, 42.0), 30.0)

    def test_counts_the_tail_after_the_last_segment(self) -> None:
        segments = [_segment(0.0, 2.0, "Единственная реплика")]
        self.assertGreater(_largest_unspoken_stretch(segments, 60.0), 50.0)

    def test_dense_speech_leaves_no_large_stretch(self) -> None:
        segments = _dense(30)
        self.assertLess(_largest_unspoken_stretch(segments, 60.0), 5.0)

    def test_empty_input_is_entirely_unspoken(self) -> None:
        self.assertEqual(_largest_unspoken_stretch([], 42.0), 42.0)


class DubSparsenessTests(unittest.TestCase):
    def test_long_silent_opening_is_reported_sparse(self) -> None:
        # Healthy by every aggregate count, yet the first 27 seconds say nothing:
        # this is the job 40814 shape that used to slip through.
        segments = [_segment(0.0, 29.98, "Девушки отдыхают"), *_dense(60, step=2.0)]
        shifted = [segments[0]] + [
            _segment(30.0 + index * 2.0, 32.0 + index * 2.0, segment.spoken_text)
            for index, segment in enumerate(segments[1:])
        ]
        self.assertTrue(_dub_is_too_sparse(shifted, 150.0))

    def test_dense_dub_is_not_reported_sparse(self) -> None:
        self.assertFalse(_dub_is_too_sparse(_dense(60, step=2.0), 120.0))


if __name__ == "__main__":
    unittest.main()
