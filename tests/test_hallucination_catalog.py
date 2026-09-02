from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.hallucination_catalog import HallucinationCatalog, shared_catalog


def _write(path: Path, rows: list[tuple[str, str, int]]) -> None:
    lines = ["lang,phrase,count"]
    lines += [f'{lang},"{phrase}",{count}' for lang, phrase, count in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.path = Path(self._tempdir.name) / "phrases.csv"

    def _catalog(self, rows: list[tuple[str, str, int]]) -> HallucinationCatalog:
        _write(self.path, rows)
        return HallucinationCatalog(self.path)

    def test_phrases_come_from_the_asked_language(self) -> None:
        rows = [("vi", f"вьетнамская фраза номер {n}", 1) for n in range(30)]
        rows += [("de", f"немецкая фраза номер {n}", 1) for n in range(30)]
        picked = self._catalog(rows).phrases("vi", 5, seed="s", cross_language_share=0.0)
        self.assertTrue(all("вьетнамская" in p for p in picked))

    def test_a_missing_file_yields_nothing_rather_than_raising(self) -> None:
        catalog = HallucinationCatalog(Path(self._tempdir.name) / "нет.csv")
        self.assertEqual(catalog.phrases("vi", 5), [])

    def test_an_unknown_language_falls_back_to_the_whole_catalogue(self) -> None:
        # Nothing to be faithful to, so borrowing everything beats returning
        # nothing and leaving the dub without artifacts.
        rows = [("de", f"немецкая фраза номер {n}", 1) for n in range(30)]
        picked = self._catalog(rows).phrases("xx", 4, seed="s", cross_language_share=0.0)
        self.assertEqual(len(picked), 4)

    def test_short_fragments_are_dropped(self) -> None:
        rows = [("vi", "aah", 99), ("vi", "нормальная длинная фраза", 1)]
        picked = self._catalog(rows).phrases("vi", 5, seed="s", cross_language_share=0.0)
        self.assertEqual(picked, ["нормальная длинная фраза"])

    def test_the_same_seed_gives_the_same_phrases(self) -> None:
        rows = [("vi", f"фраза номер {n} про разное", 1) for n in range(40)]
        catalog = self._catalog(rows)
        first = catalog.phrases("vi", 6, seed="job-1", cross_language_share=0.0)
        second = catalog.phrases("vi", 6, seed="job-1", cross_language_share=0.0)
        self.assertEqual(first, second)

    def test_a_different_seed_gives_different_phrases(self) -> None:
        rows = [("vi", f"фраза номер {n} про разное", 1) for n in range(40)]
        catalog = self._catalog(rows)
        first = catalog.phrases("vi", 6, seed="job-1", cross_language_share=0.0)
        second = catalog.phrases("vi", 6, seed="job-2", cross_language_share=0.0)
        self.assertNotEqual(first, second)

    def test_phrases_are_never_repeated_within_one_draw(self) -> None:
        rows = [("vi", f"фраза номер {n} про разное", 1) for n in range(40)]
        picked = self._catalog(rows).phrases("vi", 10, seed="s", cross_language_share=0.0)
        self.assertEqual(len(picked), len(set(picked)))

    def test_near_duplicates_are_avoided(self) -> None:
        # A section of the real catalogue is 65% variations of "subscribe to
        # the channel"; six of those in one dub is six identical lines.
        rows = [("vi", f"подпишитесь на канал чтобы {n}", 5) for n in range(20)]
        rows += [("vi", f"совершенно другая мысль про кота {n}", 1) for n in range(20)]
        picked = self._catalog(rows).phrases("vi", 4, seed="s", cross_language_share=0.0)
        subscribe = sum(1 for p in picked if "подпишитесь" in p)
        self.assertLess(subscribe, len(picked))

    def test_a_thin_language_borrows_more_from_elsewhere(self) -> None:
        # Three phrases cannot carry eight slots on their own.
        rows = [("vi", f"вьетнамская фраза номер {n}", 1) for n in range(3)]
        rows += [("de", f"немецкая фраза номер {n}", 1) for n in range(60)]
        picked = self._catalog(rows).phrases("vi", 8, seed="s", cross_language_share=0.05)
        self.assertTrue(any("немецкая" in p for p in picked))

    def test_a_rich_language_mostly_keeps_to_itself(self) -> None:
        rows = [("vi", f"вьетнамская фраза номер {n}", 1) for n in range(200)]
        rows += [("de", f"немецкая фраза номер {n}", 1) for n in range(200)]
        picked = self._catalog(rows).phrases("vi", 6, seed="s", cross_language_share=0.15)
        own = sum(1 for p in picked if "вьетнамская" in p)
        self.assertGreaterEqual(own, 4)

    def test_zero_cross_share_stays_in_the_language(self) -> None:
        rows = [("vi", f"вьетнамская фраза номер {n}", 1) for n in range(60)]
        rows += [("de", f"немецкая фраза номер {n}", 1) for n in range(60)]
        picked = self._catalog(rows).phrases("vi", 5, seed="s", cross_language_share=0.0)
        self.assertFalse(any("немецкая" in p for p in picked))

    def test_asking_for_nothing_returns_nothing(self) -> None:
        rows = [("vi", "нормальная длинная фраза", 1)]
        self.assertEqual(self._catalog(rows).phrases("vi", 0, seed="s"), [])

    def test_weights_favour_common_phrases(self) -> None:
        rows = [("vi", "очень частая галлюцинация здесь", 1000)]
        rows += [("vi", f"редкая фраза номер {n} тут", 1) for n in range(50)]
        catalog = self._catalog(rows)
        hits = sum(
            1
            for n in range(40)
            if "очень частая" in " ".join(catalog.phrases("vi", 1, seed=f"s{n}", cross_language_share=0.0))
        )
        self.assertGreater(hits, 20)

    def test_the_shipped_catalogue_is_present_and_parses(self) -> None:
        catalog = shared_catalog()
        self.assertGreater(catalog.size(), 5000)
        self.assertGreater(len(catalog.languages), 50)

    def test_the_shipped_catalogue_covers_the_common_input_languages(self) -> None:
        from laladub.bot import SOURCE_LANGS

        catalog = shared_catalog()
        # The eight most-picked languages cover 90% of jobs; those must not
        # fall back to borrowing everything.
        for code, label in SOURCE_LANGS[1:9]:
            with self.subTest(language=f"{code} ({label})"):
                self.assertGreater(catalog.size(code), 0)


if __name__ == "__main__":
    unittest.main()
