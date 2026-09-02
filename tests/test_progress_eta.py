from __future__ import annotations

import unittest

from laladub.bot import _ProgressState


class ProgressEtaTests(unittest.TestCase):
    def test_progress_based_eta_survives_without_historical_estimate(self) -> None:
        progress = _ProgressState("Full dubbing", "52254")
        progress._started_at -= 8 * 60  # Simulate an already-running legacy job.
        progress.update("Перевожу текст", 52, 100, "цепочка vi->ru")
        remaining = progress.remaining_seconds()
        self.assertIsNotNone(remaining)
        assert remaining is not None
        self.assertGreater(remaining, 7 * 60)
        self.assertIn("Осталось примерно:", progress.render())

    def test_completed_progress_hides_remaining_time(self) -> None:
        progress = _ProgressState("Full dubbing", "1", estimated_total_seconds=600)
        progress.finish("Готово")
        self.assertIsNone(progress.remaining_seconds())
        self.assertNotIn("Осталось примерно:", progress.render())


if __name__ == "__main__":
    unittest.main()
