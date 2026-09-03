from __future__ import annotations

import unittest

from laladub.pipeline import _artifact_harvest_windows


class ArtifactHarvestWindowTests(unittest.TestCase):
    def test_short_video_keeps_all_available_windows(self) -> None:
        windows = _artifact_harvest_windows(10.0, 0.20)

        self.assertEqual(windows, [(0.0, 5.0), (5.0, 5.0)])

    def test_one_minute_video_samples_five_windows(self) -> None:
        windows = _artifact_harvest_windows(60.0, 0.20)

        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[0], (0.0, 5.0))
        self.assertEqual(windows[-1], (55.0, 5.0))

    def test_long_video_is_capped_at_twelve_evenly_spaced_windows(self) -> None:
        windows = _artifact_harvest_windows(209.0, 0.20)

        self.assertEqual(len(windows), 12)
        self.assertEqual(windows[0], (0.0, 5.0))
        self.assertEqual(windows[-1], (204.0, 5.0))
        self.assertEqual(windows, sorted(windows))


if __name__ == "__main__":
    unittest.main()
