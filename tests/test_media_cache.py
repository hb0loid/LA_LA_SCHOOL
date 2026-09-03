from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from laladub.pipeline import (
    _covers_source_duration,
    _store_cached_file,
    _store_legacy_media_cache,
)

FFMPEG = shutil.which("ffmpeg")


def _make_video(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            FFMPEG, "-y",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, capture_output=True,
    )


def _make_wav(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            FFMPEG, "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path),
        ],
        check=True, capture_output=True,
    )


@unittest.skipIf(FFMPEG is None, "ffmpeg is required to build the sample media")
class CoversSourceDurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dir = Path(self._tempdir.name)

    def test_full_length_audio_is_accepted(self) -> None:
        video = self.dir / "v.mp4"
        audio = self.dir / "a.wav"
        _make_video(video, 12.0)
        _make_wav(audio, 12.0)
        self.assertTrue(_covers_source_duration(audio, video))

    def test_truncated_audio_is_rejected(self) -> None:
        # The real failure: a 4-minute video whose cached audio only covers the
        # first minute, leaving the rest of the dub in the original voice.
        video = self.dir / "v.mp4"
        audio = self.dir / "a.wav"
        _make_video(video, 20.0)
        _make_wav(audio, 5.0)
        self.assertFalse(_covers_source_duration(audio, video))

    def test_small_shortfall_is_tolerated(self) -> None:
        video = self.dir / "v.mp4"
        audio = self.dir / "a.wav"
        _make_video(video, 12.0)
        _make_wav(audio, 10.0)
        self.assertTrue(_covers_source_duration(audio, video))

    def test_unprobeable_input_does_not_block_the_normal_path(self) -> None:
        self.assertTrue(_covers_source_duration(self.dir / "nope.wav", self.dir / "nope.mp4"))


@unittest.skipIf(FFMPEG is None, "ffmpeg is required to build the sample media")
class StoreLegacyMediaCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dir = Path(self._tempdir.name)
        self.cache_entry = self.dir / "cache"
        self.cache_entry.mkdir()
        self.legacy_work = self.dir / "legacy" / "work"
        self.legacy_work.mkdir(parents=True)

    def _config(self):
        from laladub.models import DubConfig

        return DubConfig(
            output=self.dir / "out.mp4",
            workdir=self.dir / "work",
            demucs_model="htdemucs",
        )

    def test_seeding_is_refused_when_the_legacy_audio_is_truncated(self) -> None:
        # Regression: a job that ran on a daily-quota-trimmed copy keeps the
        # untrimmed original beside it, so the hash matches the full file while
        # its work audio covers only part of it. Seeding from that poisoned the
        # cache for every later dub of the same video.
        video = self.dir / "input.mp4"
        _make_video(video, 20.0)
        _make_wav(self.legacy_work / "source_16k.wav", 5.0)

        stored = _store_legacy_media_cache(self.cache_entry, self.legacy_work, self._config(), video)
        self.assertFalse(stored)
        self.assertFalse((self.cache_entry / "source_16k.wav").exists())

    def test_seeding_proceeds_for_full_length_legacy_audio(self) -> None:
        video = self.dir / "input.mp4"
        _make_video(video, 12.0)
        _make_wav(self.legacy_work / "source_16k.wav", 12.0)

        stored = _store_legacy_media_cache(self.cache_entry, self.legacy_work, self._config(), video)
        self.assertTrue(stored)
        self.assertTrue((self.cache_entry / "source_16k.wav").exists())


class StoreCachedFileOverwriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dir = Path(self._tempdir.name)
        self.cache_entry = self.dir / "cache"
        self.cache_entry.mkdir()

    def _write(self, path: Path, payload: bytes) -> Path:
        path.write_bytes(payload)
        return path

    def test_existing_entry_is_kept_by_default(self) -> None:
        self._write(self.cache_entry / "source_16k.wav", b"old" * 1024)
        fresh = self._write(self.dir / "fresh.wav", b"new" * 1024)
        _store_cached_file(self.cache_entry, fresh, "source_16k.wav")
        self.assertEqual((self.cache_entry / "source_16k.wav").read_bytes(), b"old" * 1024)

    def test_overwrite_replaces_a_poisoned_entry(self) -> None:
        self._write(self.cache_entry / "source_16k.wav", b"old" * 1024)
        fresh = self._write(self.dir / "fresh.wav", b"new" * 1024)
        _store_cached_file(self.cache_entry, fresh, "source_16k.wav", overwrite=True)
        self.assertEqual((self.cache_entry / "source_16k.wav").read_bytes(), b"new" * 1024)


if __name__ == "__main__":
    unittest.main()
