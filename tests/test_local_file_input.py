from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from laladub.bot import _local_file_access_allowed, _local_file_path_from_text, _stage_local_file


class LocalFileInputTests(unittest.TestCase):
    def test_windows_and_file_url_paths_are_recognized(self) -> None:
        self.assertEqual(_local_file_path_from_text(r'"F:\Видео\тест.mp4"'), Path(r"F:\Видео\тест.mp4"))
        self.assertEqual(_local_file_path_from_text("file:///F:/Видео/тест%201.mp4"), Path("F:/Видео/тест 1.mp4"))
        self.assertIsNone(_local_file_path_from_text("https://example.com/video.mp4"))

    def test_access_is_limited_to_configured_user(self) -> None:
        with patch.dict(os.environ, {"LALADUB_LOCAL_FILE_USERS": "631551040"}):
            self.assertTrue(_local_file_access_allowed(631551040))
            self.assertFalse(_local_file_access_allowed(7123813884))

    def test_local_file_is_staged_without_changing_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            source = root / "source.mp4"
            destination = root / "job" / "input.mp4"
            source.write_bytes(b"video")
            _stage_local_file(source, destination)
            self.assertEqual(destination.read_bytes(), b"video")


if __name__ == "__main__":
    unittest.main()
