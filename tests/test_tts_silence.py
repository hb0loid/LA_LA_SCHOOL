from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from laladub.pipeline import _trim_tts_silence

SAMPLE_RATE = 44100


def _tone(seconds: float, amplitude: float = 0.5, frequency: float = 220.0) -> list[float]:
    count = int(SAMPLE_RATE * seconds)
    return [amplitude * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE) for i in range(count)]


def _silence(seconds: float) -> list[float]:
    return [0.0] * int(SAMPLE_RATE * seconds)


def _write(path: Path, samples: list[float]) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"".join(int(s * 32767.0).to_bytes(2, "little", signed=True) for s in samples))


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


class TrimTtsSilenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.path = Path(self._tempdir.name) / "phrase.wav"

    def test_drops_padding_around_the_phrase(self) -> None:
        _write(self.path, _silence(2.0) + _tone(1.0) + _silence(2.0))
        self.assertTrue(_trim_tts_silence(self.path, 0.3))
        # 1s of speech plus a small pad on each side, nowhere near the original 5s.
        self.assertLess(_duration(self.path), 1.5)
        self.assertGreater(_duration(self.path), 0.9)

    def test_clamps_a_long_pause_inside_the_phrase(self) -> None:
        _write(self.path, _tone(0.5) + _silence(3.0) + _tone(0.5))
        self.assertTrue(_trim_tts_silence(self.path, 0.3))
        # Both halves of speech survive; the 3s gap collapses to roughly the cap.
        self.assertLess(_duration(self.path), 1.7)
        self.assertGreater(_duration(self.path), 1.0)

    def test_keeps_a_pause_that_is_already_short(self) -> None:
        _write(self.path, _tone(0.5) + _silence(0.2) + _tone(0.5))
        before = _duration(self.path)
        _trim_tts_silence(self.path, 0.3)
        self.assertAlmostEqual(_duration(self.path), before, delta=0.15)

    def test_leaves_continuous_speech_untouched(self) -> None:
        _write(self.path, _tone(2.0))
        before = _duration(self.path)
        _trim_tts_silence(self.path, 0.3)
        self.assertAlmostEqual(_duration(self.path), before, delta=0.05)

    def test_silent_clip_is_left_alone(self) -> None:
        _write(self.path, _silence(1.0))
        before = _duration(self.path)
        self.assertFalse(_trim_tts_silence(self.path, 0.3))
        self.assertEqual(_duration(self.path), before)

    def test_output_stays_readable_mono_16bit(self) -> None:
        _write(self.path, _silence(1.0) + _tone(1.0) + _silence(1.0))
        _trim_tts_silence(self.path, 0.3)
        with wave.open(str(self.path), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
