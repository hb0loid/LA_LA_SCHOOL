from __future__ import annotations

import unittest

from laladub.models import Segment
from laladub.quality import limit_phrase_repeats_across_segments


def _segments(texts: list[str]) -> list[Segment]:
    return [
        Segment(start=float(i), end=float(i) + 1.0, text=text, translated_text=text)
        for i, text in enumerate(texts)
    ]


def _filler(count: int) -> list[str]:
    return [f"обычная реплика номер {n}" for n in range(count)]


class LineRepeatLimitTests(unittest.TestCase):
    def test_a_phrase_past_the_limit_is_dropped(self) -> None:
        segments = _segments([*_filler(40), *["Интервью"] * 20])
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=5)
        said = [s.translated_text for s in kept]
        self.assertEqual(said.count("Интервью"), 5)

    def test_the_first_occurrences_are_the_ones_kept(self) -> None:
        segments = _segments([*_filler(40), *["Интервью"] * 10])
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=3)
        keys = [s.translated_text for s in kept]
        self.assertEqual(keys[40:], ["Интервью"] * 3)

    def test_a_phrase_within_the_limit_is_untouched(self) -> None:
        segments = _segments([*_filler(40), *["Интервью"] * 4])
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=5)
        self.assertEqual(len(kept), len(segments))

    def test_punctuation_and_case_do_not_split_a_phrase(self) -> None:
        # "Интервью", "интервью." and "Интервью!" are the same collapsed line.
        segments = _segments([*_filler(40), *["Интервью", "интервью.", "Интервью!"] * 4])
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=5)
        self.assertEqual(len(kept) - 40, 5)

    def test_long_lines_are_never_dropped(self) -> None:
        # A long line repeating verbatim is far more likely to be genuine.
        long_line = "это довольно длинная реплика которая повторяется целиком"
        segments = _segments([*_filler(40), *[long_line] * 20])
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=5)
        self.assertEqual(len(kept), len(segments))

    def test_short_videos_are_left_alone(self) -> None:
        # Too little material to tell a collapse from a running joke.
        segments = _segments(["Что?"] * 20)
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=5)
        self.assertEqual(len(kept), len(segments))

    def test_zero_disables_the_limit(self) -> None:
        segments = _segments([*_filler(40), *["Интервью"] * 20])
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=0)
        self.assertEqual(len(kept), len(segments))

    def test_different_phrases_are_counted_separately(self) -> None:
        segments = _segments([*_filler(40), *["Интервью"] * 8, *["Свяжитесь с нами"] * 8])
        kept = limit_phrase_repeats_across_segments(segments, max_repeats=5)
        said = [s.translated_text for s in kept]
        self.assertEqual(said.count("Интервью"), 5)
        self.assertEqual(said.count("Свяжитесь с нами"), 5)

    def test_the_summary_line_stays_ascii_safe(self) -> None:
        # This log goes to a cp1251 console; a stray "×" used to raise
        # UnicodeEncodeError and take the pipeline down with it.
        segments = _segments([*_filler(40), *["Интервью"] * 20])
        printed: list[str] = []
        import builtins

        real_print = builtins.print
        try:
            builtins.print = lambda *a, **k: printed.append(str(a[0]) if a else "")
            limit_phrase_repeats_across_segments(segments, max_repeats=5)
        finally:
            builtins.print = real_print
        joined = " ".join(printed)
        self.assertIn("Dropped", joined)
        joined.encode("cp1251", errors="strict")


if __name__ == "__main__":
    unittest.main()
