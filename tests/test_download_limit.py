from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from laladub.bot import _max_file_mb_for
from laladub.download import _restore_cached_download


def _settings(*, admins: set[int], paid: set[int]) -> SimpleNamespace:
    return SimpleNamespace(
        max_file_mb=200,
        max_file_mb_premium=0,
        is_admin=lambda uid: uid in admins,
        is_paid=lambda uid: uid in paid,
    )


class MaxFileMbForTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _settings(admins={1}, paid={2})

    def test_an_ordinary_user_keeps_the_normal_cap(self) -> None:
        self.assertEqual(_max_file_mb_for(self.settings, None, 99), 200)

    def test_an_admin_has_no_cap(self) -> None:
        self.assertEqual(_max_file_mb_for(self.settings, None, 1), 0)

    def test_a_paid_user_has_no_cap(self) -> None:
        self.assertEqual(_max_file_mb_for(self.settings, None, 2), 0)

    def test_a_subscriber_has_no_cap(self) -> None:
        store = SimpleNamespace(active_subscription=lambda uid: object())
        self.assertEqual(_max_file_mb_for(self.settings, store, 99), 0)

    def test_an_expired_subscriber_keeps_the_normal_cap(self) -> None:
        store = SimpleNamespace(active_subscription=lambda uid: None)
        self.assertEqual(_max_file_mb_for(self.settings, store, 99), 200)

    def test_an_unknown_user_keeps_the_normal_cap(self) -> None:
        self.assertEqual(_max_file_mb_for(self.settings, None, None), 200)

    def test_a_premium_cap_can_be_a_number_rather_than_unlimited(self) -> None:
        self.settings.max_file_mb_premium = 1000
        self.assertEqual(_max_file_mb_for(self.settings, None, 1), 1000)


class DownloadSizeGateTests(unittest.TestCase):
    """The cap is enforced in three places; zero must disable all of them."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dir = Path(self._tempdir.name)

    def _cached(self, size_bytes: int) -> Path:
        cache = self.dir / "cache"
        cache.mkdir()
        video = cache / "input.mp4"
        video.write_bytes(b"x" * size_bytes)
        return video

    def _restore(self, max_bytes: int) -> Path | None:
        video = self._cached(2048)
        out = self.dir / "out"
        out.mkdir(exist_ok=True)
        # The fixture is not real media, so the ffprobe check has to be stubbed
        # out - otherwise the call returns None for that reason and the test
        # would pass without the size gate ever being consulted.
        with patch("laladub.download._download_cache_entry", return_value=video.parent),              patch("laladub.download.has_video_and_audio", return_value=True):
            return _restore_cached_download("u", out, max_bytes)

    def test_an_oversized_cached_file_is_refused_under_a_cap(self) -> None:
        self.assertIsNone(self._restore(1024))

    def test_a_cached_file_within_the_cap_is_accepted(self) -> None:
        self.assertIsNotNone(self._restore(4096))

    def test_the_oversized_file_is_accepted_with_no_cap(self) -> None:
        self.assertIsNotNone(self._restore(0))


if __name__ == "__main__":
    unittest.main()
