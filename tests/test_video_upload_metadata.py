from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from laladub.ffmpeg import probe_video_dimensions

FFMPEG = shutil.which("ffmpeg")


def _make_video(path: Path, *, seconds: float, width: int, height: int) -> None:
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate=10:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=44100:cl=mono:d={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@unittest.skipIf(FFMPEG is None, "ffmpeg is required to build the sample video")
class ProbeVideoDimensionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.video = Path(self._tempdir.name) / "clip.mp4"

    def test_reads_the_frame_size(self) -> None:
        _make_video(self.video, seconds=1.0, width=320, height=240)
        self.assertEqual(probe_video_dimensions(self.video), (320, 240))

    def test_reads_a_portrait_frame_size(self) -> None:
        _make_video(self.video, seconds=1.0, width=240, height=320)
        self.assertEqual(probe_video_dimensions(self.video), (240, 320))

    def test_missing_file_returns_none_instead_of_raising(self) -> None:
        self.assertIsNone(probe_video_dimensions(Path(self._tempdir.name) / "nope.mp4"))

    def test_non_video_returns_none(self) -> None:
        text = Path(self._tempdir.name) / "notes.txt"
        text.write_text("no video here", encoding="utf-8")
        self.assertIsNone(probe_video_dimensions(text))


@unittest.skipIf(FFMPEG is None, "ffmpeg is required to build the sample video")
class VideoUploadMetadataTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.video = Path(self._tempdir.name) / "clip.mp4"

    async def test_reports_duration_and_size_for_telegram(self) -> None:
        from laladub.bot import video_upload_metadata

        _make_video(self.video, seconds=3.0, width=320, height=240)
        metadata = await video_upload_metadata(self.video)
        # Without these Telegram renders the clip as 00:00 however valid it is.
        self.assertEqual(metadata["width"], 320)
        self.assertEqual(metadata["height"], 240)
        self.assertAlmostEqual(metadata["duration"], 3, delta=1)
        self.assertIsInstance(metadata["duration"], int)

    async def test_unreadable_file_yields_no_metadata_instead_of_failing(self) -> None:
        from laladub.bot import video_upload_metadata

        broken = Path(self._tempdir.name) / "broken.mp4"
        broken.write_bytes(b"not really a video")
        self.assertEqual(await video_upload_metadata(broken), {})


if __name__ == "__main__":
    unittest.main()
