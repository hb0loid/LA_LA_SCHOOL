from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from laladub.ffmpeg import extract_audio, extract_audio_track, probe_duration, trim_video


class DurationProbeTests(unittest.TestCase):
    def test_video_stream_wins_over_bogus_container_audio_duration(self) -> None:
        payload = {
            "streams": [
                {"codec_type": "video", "duration": "22.920000"},
                {"codec_type": "audio", "duration": "28082.176000"},
            ],
            "format": {"duration": "28082.176000"},
        }
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        with (
            patch("laladub.ffmpeg.require_tool", return_value="ffprobe"),
            patch("laladub.ffmpeg.subprocess.run", return_value=completed),
        ):
            self.assertEqual(probe_duration(Path("broken.mp4")), 22.92)

    def test_audio_only_uses_container_duration(self) -> None:
        payload = {
            "streams": [{"codec_type": "audio", "duration": "15.0"}],
            "format": {"duration": "15.0"},
        }
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        with (
            patch("laladub.ffmpeg.require_tool", return_value="ffprobe"),
            patch("laladub.ffmpeg.subprocess.run", return_value=completed),
        ):
            self.assertEqual(probe_duration(Path("audio.m4a")), 15.0)


class TrimVideoTests(unittest.TestCase):
    def test_trim_reencodes_audio_and_resets_its_timestamps(self) -> None:
        with (
            patch("laladub.ffmpeg.require_tool", return_value="ffmpeg"),
            patch("laladub.ffmpeg.run") as run_mock,
        ):
            trim_video(Path("input.mp4"), Path("output.mp4"), 12.5)

        command = run_mock.call_args.args[0]
        self.assertIn("0:v:0", command)
        self.assertIn("0:a:0?", command)
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-af") + 1], "asetpts=N/SR/TB,atrim=0:12.500")


class AudioExtractionTests(unittest.TestCase):
    def test_whisper_audio_resets_broken_input_timestamps(self) -> None:
        with (
            patch("laladub.ffmpeg.require_tool", return_value="ffmpeg"),
            patch("laladub.ffmpeg.run") as run_mock,
        ):
            extract_audio(Path("input.mp4"), Path("source.wav"))
        command = run_mock.call_args.args[0]
        self.assertEqual(command[command.index("-af") + 1], "asetpts=N/SR/TB")

    def test_mix_audio_resets_broken_input_timestamps(self) -> None:
        with (
            patch("laladub.ffmpeg.require_tool", return_value="ffmpeg"),
            patch("laladub.ffmpeg.run") as run_mock,
        ):
            extract_audio_track(Path("input.mp4"), Path("source_mix.wav"))
        command = run_mock.call_args.args[0]
        self.assertEqual(command[command.index("-af") + 1], "asetpts=N/SR/TB")


if __name__ == "__main__":
    unittest.main()
