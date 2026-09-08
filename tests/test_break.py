from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from laladub.bot import _break_label, break_until, set_break


class BreakTests(unittest.TestCase):
    """Frees the main PC while its owner is using it for something else. Only
    this machine stops - the laptop goes on preparing, so the queue keeps
    moving and work waiting for a voice piles up until the break ends."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.settings = SimpleNamespace(workdir=Path(self._tempdir.name))

    def test_no_break_by_default(self) -> None:
        self.assertIsNone(break_until(self.settings))

    def test_an_open_ended_break_stays_on(self) -> None:
        set_break(self.settings, float("inf"))
        self.assertEqual(break_until(self.settings), float("inf"))

    def test_a_timed_break_ends_by_itself(self) -> None:
        """Forgetting to end one should cost an hour, not a night."""
        set_break(self.settings, time.time() - 1)
        self.assertIsNone(break_until(self.settings))
        self.assertFalse((Path(self._tempdir.name) / "break.flag").exists())

    def test_a_timed_break_holds_until_it_runs_out(self) -> None:
        until = time.time() + 1800
        set_break(self.settings, until)
        self.assertAlmostEqual(break_until(self.settings), until, delta=1)

    def test_it_can_be_ended_early(self) -> None:
        set_break(self.settings, float("inf"))
        set_break(self.settings, None)
        self.assertIsNone(break_until(self.settings))

    def test_a_damaged_flag_is_treated_as_open_ended(self) -> None:
        """Better a break someone notices than one silently ignored."""
        (Path(self._tempdir.name) / "break.flag").write_text("мусор", encoding="utf-8")
        self.assertEqual(break_until(self.settings), float("inf"))


class BreakLabelTests(unittest.TestCase):
    def test_it_says_how_much_is_left(self) -> None:
        self.assertEqual(_break_label(float("inf")), "без срока")
        self.assertIn("мин", _break_label(time.time() + 600))
        self.assertIn("ч", _break_label(time.time() + 7200))


if __name__ == "__main__":
    unittest.main()
