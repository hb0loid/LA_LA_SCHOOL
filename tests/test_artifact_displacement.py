from __future__ import annotations

import unittest

from laladub.models import Segment
from laladub.pipeline import (
    CHAOS_ARTIFACT_DISPLACEMENT_SHARE,
    _base_segment_replaced_by_chaos_artifact,
)


def _line(start: float, end: float) -> Segment:
    return Segment(start=start, end=end, text="реплика", translated_text="реплика")


class ArtifactDisplacementTests(unittest.TestCase):
    def test_a_line_fully_covered_is_displaced(self) -> None:
        self.assertTrue(_base_segment_replaced_by_chaos_artifact(_line(1.0, 2.0), [(0.5, 3.0)]))

    def test_a_line_merely_clipped_survives(self) -> None:
        # Regression: 0.6 seconds of overlap used to delete a line outright, so
        # two artifacts displaced four real lines. Overlapping speech is mixed
        # with ducking; deleting dialogue is the bigger loss.
        self.assertFalse(_base_segment_replaced_by_chaos_artifact(_line(0.0, 4.0), [(3.4, 6.0)]))

    def test_a_long_line_touched_at_the_edge_survives(self) -> None:
        self.assertFalse(_base_segment_replaced_by_chaos_artifact(_line(0.0, 10.0), [(9.0, 12.0)]))

    def test_a_short_line_inside_an_artifact_is_displaced(self) -> None:
        self.assertTrue(_base_segment_replaced_by_chaos_artifact(_line(2.0, 2.5), [(1.0, 5.0)]))

    def test_no_overlap_leaves_the_line_alone(self) -> None:
        self.assertFalse(_base_segment_replaced_by_chaos_artifact(_line(0.0, 1.0), [(5.0, 6.0)]))

    def test_a_zero_length_line_is_never_displaced(self) -> None:
        self.assertFalse(_base_segment_replaced_by_chaos_artifact(_line(1.0, 1.0), [(0.0, 5.0)]))

    def test_the_threshold_is_the_one_that_decides(self) -> None:
        # Just under and just over the share, to pin the boundary itself.
        line = _line(0.0, 10.0)
        under = CHAOS_ARTIFACT_DISPLACEMENT_SHARE * 10.0 - 0.5
        over = CHAOS_ARTIFACT_DISPLACEMENT_SHARE * 10.0 + 0.5
        self.assertFalse(_base_segment_replaced_by_chaos_artifact(line, [(0.0, under)]))
        self.assertTrue(_base_segment_replaced_by_chaos_artifact(line, [(0.0, over)]))

    def test_any_one_artifact_can_displace(self) -> None:
        line = _line(0.0, 2.0)
        self.assertTrue(_base_segment_replaced_by_chaos_artifact(line, [(50.0, 51.0), (0.0, 2.0)]))


if __name__ == "__main__":
    unittest.main()
