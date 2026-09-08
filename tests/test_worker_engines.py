from __future__ import annotations

import unittest

from laladub.bot import PREPROCESS_ENGINES, _remote_stage_for_job, effective_tts_provider


class EffectiveEngineTests(unittest.TestCase):
    """MOSS speaks Russian and English but not Ukrainian, so a Ukrainian job
    asking for a heavy engine is voiced by F5. Deciding that in one place keeps
    the scheduler and the pipeline from disagreeing about it."""

    def test_russian_and_english_keep_moss(self) -> None:
        for target in ("ru", "en"):
            job = {"tts_provider": "moss", "target_lang": target}
            self.assertEqual(effective_tts_provider(job), "moss")

    def test_ukrainian_falls_to_f5(self) -> None:
        job = {"tts_provider": "moss", "target_lang": "uk"}
        self.assertEqual(effective_tts_provider(job), "f5")

    def test_a_light_engine_is_left_alone(self) -> None:
        job = {"tts_provider": "f5", "target_lang": "uk"}
        self.assertEqual(effective_tts_provider(job), "f5")


class RemoteStageTests(unittest.TestCase):
    """A worker that has the engine can finish the job; one that has not can
    still do everything up to the voice."""

    def test_a_worker_without_the_engine_only_prepares(self) -> None:
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(_remote_stage_for_job(job, frozenset({"f5"})), "preprocess")

    def test_a_worker_with_the_engine_finishes_it(self) -> None:
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(_remote_stage_for_job(job, frozenset({"moss"})), "complete")

    def test_a_ukrainian_job_needs_f5_not_moss(self) -> None:
        """The engine that matters is the one that will really voice it."""
        job = {"tts_provider": "moss", "target_lang": "uk"}
        self.assertEqual(_remote_stage_for_job(job, frozenset({"moss"})), "preprocess")
        self.assertEqual(_remote_stage_for_job(job, frozenset({"f5"})), "complete")

    def test_saying_nothing_keeps_the_old_assumption(self) -> None:
        """A worker on an older build reports no engines at all."""
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(_remote_stage_for_job(job), "preprocess")

    def test_raw_text_never_needs_a_voice(self) -> None:
        job = {"mode": "raw_text", "tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(_remote_stage_for_job(job, frozenset()), "complete")

    def test_the_heavy_engines_are_the_ones_split(self) -> None:
        self.assertEqual(PREPROCESS_ENGINES, frozenset({"moss", "cosyvoice", "qwen3"}))


if __name__ == "__main__":
    unittest.main()
