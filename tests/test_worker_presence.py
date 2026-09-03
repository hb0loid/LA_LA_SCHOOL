from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.worker_watch import WorkerPresence


class WorkerPresenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.path = Path(self._tempdir.name) / "presence.json"

    def _watch(self, grace: float = 180.0) -> WorkerPresence:
        return WorkerPresence(self.path, grace_seconds=grace)

    def test_a_healthy_worker_says_nothing(self) -> None:
        watch = self._watch()
        self.assertIsNone(watch.observe(1, now=1000.0))
        self.assertIsNone(watch.observe(1, now=2000.0))

    def test_a_brief_absence_is_not_announced(self) -> None:
        # A worker restarting must not raise an alarm.
        watch = self._watch()
        watch.observe(1, now=1000.0)
        self.assertIsNone(watch.observe(0, now=1010.0))
        self.assertIsNone(watch.observe(0, now=1100.0))

    def test_a_lasting_absence_is_announced_once(self) -> None:
        watch = self._watch()
        watch.observe(1, now=1000.0)
        watch.observe(0, now=1010.0)
        said = watch.observe(0, now=1400.0)
        self.assertIsNotNone(said)
        self.assertIn("не выходит на связь", said)
        self.assertIsNone(watch.observe(0, now=2000.0))

    def test_the_message_says_how_long(self) -> None:
        watch = self._watch()
        watch.observe(1, now=0.0)
        watch.observe(0, now=0.0)
        said = watch.observe(0, now=600.0)
        self.assertIn("10 мин", said)

    def test_a_return_is_announced_only_after_an_announced_absence(self) -> None:
        watch = self._watch()
        watch.observe(1, now=0.0)
        watch.observe(0, now=0.0)
        watch.observe(0, now=400.0)  # absence announced
        said = watch.observe(1, now=500.0)
        self.assertIn("снова на связи", said)

    def test_a_return_after_a_brief_blip_is_not_announced(self) -> None:
        watch = self._watch()
        watch.observe(1, now=0.0)
        watch.observe(0, now=10.0)
        self.assertIsNone(watch.observe(1, now=20.0))

    def test_state_survives_a_restart_of_the_bot(self) -> None:
        # The point: restarting the bot must not re-announce an absence it has
        # already reported.
        watch = self._watch()
        watch.observe(1, now=0.0)
        watch.observe(0, now=0.0)
        self.assertIsNotNone(watch.observe(0, now=400.0))
        reopened = self._watch()
        self.assertIsNone(reopened.observe(0, now=500.0))

    def test_a_return_is_still_announced_after_a_restart(self) -> None:
        watch = self._watch()
        watch.observe(1, now=0.0)
        watch.observe(0, now=0.0)
        watch.observe(0, now=400.0)
        reopened = self._watch()
        self.assertIn("снова на связи", reopened.observe(1, now=500.0))

    def test_a_second_absence_is_announced_again(self) -> None:
        watch = self._watch()
        watch.observe(1, now=0.0)
        watch.observe(0, now=0.0)
        watch.observe(0, now=400.0)
        watch.observe(1, now=500.0)
        watch.observe(0, now=600.0)
        self.assertIsNotNone(watch.observe(0, now=1000.0))

    def test_a_corrupt_state_file_does_not_break_it(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self._watch().observe(1, now=0.0))


if __name__ == "__main__":
    unittest.main()
