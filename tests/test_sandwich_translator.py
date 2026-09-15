"""Local -> one online hop -> local, and the profile that travels to a worker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from laladub import translation
from laladub.bot_config import TEXT_PROFILE_FIELDS, load_bot_settings
from laladub.models import DubConfig, Segment


def _config(tmp: Path, **overrides) -> DubConfig:
    values = dict(
        output=tmp / "out.mp4",
        workdir=tmp / "work",
        target_lang="ru",
        source_lang="vi",
        translator="sandwich",
        translation_seed="job-1",
    )
    values.update(overrides)
    return DubConfig(**values)


class MiddleLanguageTests(unittest.TestCase):
    def test_is_stable_for_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            first = translation.sandwich_middle_lang(config)
            second = translation.sandwich_middle_lang(config)
        self.assertEqual(first, second)
        self.assertIn(first, translation.SANDWICH_DEFAULT_LANGS.split(","))

    def test_never_the_source_target_or_english(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), sandwich_langs="en,ru,vi,de")
            self.assertEqual(translation.sandwich_middle_lang(config), "de")

    def test_different_jobs_walk_different_roads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            picks = {
                translation.sandwich_middle_lang(_config(Path(tmp), translation_seed=f"job-{n}"))
                for n in range(40)
            }
        self.assertGreater(len(picks), 3)


class SandwichHopTests(unittest.TestCase):
    def test_one_online_hop_between_two_local_ones(self) -> None:
        calls: list[tuple[str, str, str]] = []

        def local(text, source, target):
            calls.append(("local", source, target))
            return f"{target}<{text}>"

        def online(text, source, target, config):
            calls.append(("online", source, target))
            return f"{target}<{text}>"

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            translation, "_translate_argos_provider_text", local
        ), mock.patch.object(translation, "_translate_hybrid_text", online), mock.patch.object(
            translation, "_translation_cache_get", lambda *a, **k: None
        ), mock.patch.object(translation, "_translation_cache_put", lambda *a, **k: None):
            config = _config(Path(tmp), sandwich_langs="de")
            segments = [Segment(0, 1, "xin chào"), Segment(1, 2, "cảm ơn nhé")]
            translation.translate_segments(segments, config)

        self.assertEqual(segments[0].translated_text, "ru<de<en<xin chào>>>")
        kinds = [kind for kind, _s, _t in calls]
        self.assertEqual(kinds.count("online"), 2)  # one per line
        self.assertEqual(kinds.count("local"), 4)  # two per line
        self.assertIn(("online", "en", "de"), calls)
        self.assertIn(("local", "vi", "en"), calls)
        self.assertIn(("local", "de", "ru"), calls)

    def test_a_missing_local_road_goes_online_instead_of_failing(self) -> None:
        def local(text, source, target):
            if source == "zh":
                raise RuntimeError("no zh package")
            return f"{target}<{text}>"

        online_hops: list[tuple[str, str]] = []

        def online(text, source, target, config):
            online_hops.append((source, target))
            return f"{target}<{text}>"

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            translation, "_translate_argos_provider_text", local
        ), mock.patch.object(translation, "_translate_hybrid_text", online), mock.patch.object(
            translation, "_translation_cache_get", lambda *a, **k: None
        ), mock.patch.object(translation, "_translation_cache_put", lambda *a, **k: None):
            config = _config(Path(tmp), source_lang="zh", sandwich_langs="de")
            segments = [Segment(0, 1, "你好")]
            translation.translate_segments(segments, config)

        self.assertEqual(segments[0].translated_text, "ru<de<en<你好>>>")
        self.assertEqual(online_hops, [("zh", "en"), ("en", "de")])


class TextProfileTests(unittest.TestCase):
    def test_worker_takes_the_coordinator_profile_over_its_own(self) -> None:
        env = {
            "LALADUB_TRANSLATOR": "hybrid",
            "LALADUB_DISTORT_TRANSLATION": "1",
            "LALADUB_TRANSLATION_SECOND_PASS_RATIO": "0.45",
            "LALADUB_ARTIFACT_RATIO": "0.20",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            worker_side = load_bot_settings(require_token=False)
        profile = {
            "translator": "sandwich",
            "sandwich_langs": "de,fr",
            "distort_translation": False,
            "translation_second_pass_ratio": 0.0,
            "artifact_ratio": 0.30,
        }
        applied = worker_side.with_text_profile(profile)
        self.assertEqual(applied.translator, "sandwich")
        self.assertEqual(applied.sandwich_langs, "de,fr")
        self.assertFalse(applied.distort_translation)
        self.assertEqual(applied.translation_second_pass_ratio, 0.0)
        self.assertEqual(applied.artifact_ratio, 0.30)
        # Untouched fields stay the worker's own.
        self.assertEqual(applied.translation_pivots, worker_side.translation_pivots)

    def test_profile_round_trips_through_json_types(self) -> None:
        settings = load_bot_settings(require_token=False)
        profile = settings.text_profile()
        self.assertEqual(set(profile), set(TEXT_PROFILE_FIELDS))
        # JSON turns nothing here into something the receiver cannot coerce.
        self.assertEqual(settings.with_text_profile(dict(profile)), settings)

    def test_an_empty_profile_changes_nothing(self) -> None:
        settings = load_bot_settings(require_token=False)
        self.assertIs(settings.with_text_profile(None), settings)
        self.assertIs(settings.with_text_profile({}), settings)


if __name__ == "__main__":
    unittest.main()
