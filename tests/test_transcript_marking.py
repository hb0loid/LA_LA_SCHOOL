from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.bot import _mark_transcript_line, _read_transcript_text


class MarkTranscriptLineTests(unittest.TestCase):
    def test_an_artifact_line_is_upper_cased(self) -> None:
        said = _mark_transcript_line("Спасибо за просмотр", {"спасибо за просмотр"})
        self.assertEqual(said, "СПАСИБО ЗА ПРОСМОТР")

    def test_an_ordinary_line_is_left_alone(self) -> None:
        said = _mark_transcript_line("Обычная реплика", {"спасибо за просмотр"})
        self.assertEqual(said, "Обычная реплика")

    def test_a_censored_phrase_is_upper_cased_inside_a_line(self) -> None:
        said = _mark_transcript_line("Он сказал [censored] и ушёл", set())
        self.assertIn("[CENSORED]", said)
        self.assertIn("Он сказал", said)

    def test_matching_ignores_spacing_and_case(self) -> None:
        said = _mark_transcript_line("  Спасибо   ЗА просмотр  ", {"спасибо за просмотр"})
        self.assertEqual(said.strip(), "СПАСИБО   ЗА ПРОСМОТР")

    def test_an_empty_artifact_set_still_marks_censorship(self) -> None:
        self.assertIn("[CENSORED]", _mark_transcript_line("а [censored] б", set()))


class ReadTranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.job = Path(self._tempdir.name)
        (self.job / "work" / "debug").mkdir(parents=True)

    def _write(self, name: str, lines: list[str], where: Path) -> Path:
        blocks = []
        for index, text in enumerate(lines, start=1):
            blocks.append(f"{index}\n00:00:0{index},000 --> 00:00:0{index},900\n{text}\n")
        path = where / name
        path.write_text("\n".join(blocks), encoding="utf-8")
        return path

    def test_injected_artifacts_are_marked_in_the_transcript(self) -> None:
        srt = self._write("translated.srt", ["Живая реплика", "Спасибо за просмотр"], self.job / "work")
        self._write("artifact_injected.srt", ["Спасибо за просмотр"], self.job / "work" / "debug")
        text = _read_transcript_text(srt)
        self.assertIn("СПАСИБО ЗА ПРОСМОТР", text)
        self.assertIn("Живая реплика", text)

    def test_without_the_artifact_file_nothing_is_marked(self) -> None:
        srt = self._write("translated.srt", ["Живая реплика", "Спасибо за просмотр"], self.job / "work")
        text = _read_transcript_text(srt)
        self.assertIn("Спасибо за просмотр", text)
        self.assertNotIn("СПАСИБО", text)

    def test_marking_can_be_turned_off(self) -> None:
        srt = self._write("translated.srt", ["Спасибо за просмотр"], self.job / "work")
        self._write("artifact_injected.srt", ["Спасибо за просмотр"], self.job / "work" / "debug")
        self.assertNotIn("СПАСИБО", _read_transcript_text(srt, mark=False))

    def test_a_missing_file_gives_empty_text(self) -> None:
        self.assertEqual(_read_transcript_text(self.job / "нет.srt"), "")


if __name__ == "__main__":
    unittest.main()
