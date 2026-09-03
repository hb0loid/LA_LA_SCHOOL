from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from laladub.ffmpeg import (
    _telegram_bitrate_plan,
    combine_video_audio_multitrack,
    compress_video_for_telegram,
    mp4_language_tag,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _video(path: Path, seconds: float = 2.0) -> None:
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=10:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )


def _tone(path: Path, frequency: int, seconds: float = 2.0) -> None:
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}",
         "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True,
    )


def _audio_streams(path: Path) -> list[dict]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index:stream_disposition=default:stream_tags=language",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout).get("streams", [])


@unittest.skipIf(FFMPEG is None or FFPROBE is None, "ffmpeg/ffprobe are required")
class MultitrackMuxTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        root = Path(self._tempdir.name)
        self.video = root / "clip.mp4"
        self.original = root / "original.wav"
        self.dub = root / "dub.wav"
        self.bed = root / "bed.wav"
        self.output = root / "out.mp4"
        _video(self.video)
        _tone(self.original, 220)
        _tone(self.dub, 440)
        _tone(self.bed, 110)

    def _mux(self, **kwargs) -> None:
        combine_video_audio_multitrack(
            video_path=self.video,
            original_audio_path=self.original,
            dub_path=self.dub,
            output_path=self.output,
            dub_volume=1.0,
            **kwargs,
        )

    def test_produces_two_audio_tracks(self) -> None:
        self._mux(bed_path=self.bed, bed_volume=0.35)
        self.assertEqual(len(_audio_streams(self.output)), 2)

    def test_dub_comes_first_and_is_the_default_track(self) -> None:
        # A player that ignores dispositions and grabs the first audio stream
        # must still get the dub rather than the untranslated original.
        self._mux(bed_path=self.bed, bed_volume=0.35, original_lang="vi", dub_lang="ru")
        streams = _audio_streams(self.output)
        self.assertEqual(streams[0]["disposition"]["default"], 1)
        self.assertEqual(streams[0]["tags"]["language"], "rus")
        self.assertEqual(streams[1]["disposition"]["default"], 0)
        self.assertEqual(streams[1]["tags"]["language"], "vie")

    def test_works_without_an_instrumental_bed(self) -> None:
        self._mux(original_lang="en", dub_lang="ru")
        streams = _audio_streams(self.output)
        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0]["tags"]["language"], "rus")

    def test_telegram_compression_keeps_both_tracks(self) -> None:
        # Compression used to map only the first audio stream, quietly dropping
        # the original from any video large enough to need it.
        self._mux(bed_path=self.bed, bed_volume=0.35, original_lang="vi", dub_lang="ru")
        compressed = Path(self._tempdir.name) / "small.mp4"
        compress_video_for_telegram(self.output, compressed, target_size_mb=1.0)
        streams = _audio_streams(compressed)
        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0]["disposition"]["default"], 1)


class LanguageTagTests(unittest.TestCase):
    def test_maps_known_codes(self) -> None:
        self.assertEqual(mp4_language_tag("ru"), "rus")
        self.assertEqual(mp4_language_tag("vi"), "vie")

    def test_unknown_and_missing_codes_fall_back_to_undefined(self) -> None:
        self.assertEqual(mp4_language_tag("xx"), "und")
        self.assertEqual(mp4_language_tag(None), "und")
        self.assertEqual(mp4_language_tag(""), "und")


class TelegramBitratePlanTests(unittest.TestCase):
    def test_long_video_budget_is_not_overridden_by_old_320k_floor(self) -> None:
        video_k, audio_k = _telegram_bitrate_plan(
            250,
            audio_tracks=1,
            requested_audio_bitrate_k=96,
        )
        self.assertLess(video_k, 320)
        self.assertLessEqual(video_k + audio_k, 250)

    def test_multiple_audio_tracks_share_the_same_total_budget(self) -> None:
        video_k, audio_k = _telegram_bitrate_plan(
            250,
            audio_tracks=2,
            requested_audio_bitrate_k=96,
        )
        self.assertLessEqual(video_k + audio_k * 2, 250)


if __name__ == "__main__":
    unittest.main()
