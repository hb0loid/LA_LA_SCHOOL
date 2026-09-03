from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.worker import REPORT_TAIL_BYTES, _log_tail, _worker_log_dir


class WorkerReportTests(unittest.TestCase):
    """The laptop answers nothing from outside, so when it dies the reason is
    written on a machine the other side cannot read. The worker carries it over
    on the way back up."""

    def test_a_short_log_is_sent_whole(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker.log"
            path.write_bytes(b"first line\nsecond line\n")
            self.assertEqual(_log_tail(path).splitlines(), ["first line", "second line"])

    def test_a_long_log_is_cut_to_the_tail_at_a_line_break(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "worker.log"
            path.write_text("x" * REPORT_TAIL_BYTES + "\nthe part that matters\n", encoding="utf-8")
            tail = _log_tail(path)
            self.assertEqual(tail, "the part that matters")

    def test_a_missing_log_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            self.assertEqual(_log_tail(Path(tempdir) / "nope.log"), "")

    def test_the_log_dir_is_found_two_levels_above_the_workdir(self) -> None:
        """The launcher runs the worker from the package root and passes
        runs/worker as its workdir."""
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "logs").mkdir()
            (root / "runs" / "worker").mkdir(parents=True)
            self.assertEqual(_worker_log_dir(root / "runs" / "worker"), root / "logs")


if __name__ == "__main__":
    unittest.main()
