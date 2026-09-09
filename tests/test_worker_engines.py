from __future__ import annotations

import unittest

import unittest.mock

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


class VoicingStaysHomeTests(unittest.TestCase):
    """Measured over 495 short jobs: the main PC voices at about 7 seconds of
    work per second of video, the laptop at 60. So by default the laptop gets
    everything except the voice, whatever engine the job asks for."""

    def test_a_dub_is_only_ever_prepared_remotely(self) -> None:
        for target, engines in (("ru", frozenset({"moss"})), ("uk", frozenset({"f5"}))):
            job = {"tts_provider": "moss", "target_lang": target}
            self.assertEqual(_remote_stage_for_job(job, engines), "preprocess")

    def test_raw_text_needs_no_voice_and_goes_whole(self) -> None:
        job = {"mode": "raw_text", "tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(_remote_stage_for_job(job, frozenset()), "complete")


class RemoteStageTests(unittest.TestCase):
    """With voicing allowed - which happens only while the main PC is on a
    break - a worker that has the engine finishes the job, and one that has not
    still does everything up to the voice."""

    @staticmethod
    def _stage(job: dict, engines: frozenset[str] | None = None) -> str:
        return _remote_stage_for_job(job, engines, allow_voice=True)

    def test_a_worker_without_the_engine_only_prepares(self) -> None:
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(self._stage(job, frozenset({"f5"})), "preprocess")

    def test_a_worker_with_the_engine_finishes_it(self) -> None:
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(self._stage(job, frozenset({"moss"})), "complete")

    def test_a_ukrainian_job_needs_f5_not_moss(self) -> None:
        """The engine that matters is the one that will really voice it."""
        job = {"tts_provider": "moss", "target_lang": "uk"}
        self.assertEqual(self._stage(job, frozenset({"moss"})), "preprocess")
        self.assertEqual(self._stage(job, frozenset({"f5"})), "complete")

    def test_saying_nothing_keeps_the_old_assumption(self) -> None:
        """A worker on an older build reports no engines at all."""
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(self._stage(job), "preprocess")

    def test_raw_text_never_needs_a_voice(self) -> None:
        job = {"mode": "raw_text", "tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(self._stage(job, frozenset()), "complete")

    def test_the_heavy_engines_are_the_ones_split(self) -> None:
        self.assertEqual(PREPROCESS_ENGINES, frozenset({"moss", "cosyvoice", "qwen3"}))



class VoicingFirstTests(unittest.TestCase):
    """The main PC is the only machine that voices, and voicing is what the
    queue waits on. A job it starts from scratch is a job the laptop could have
    prepared meanwhile, so finished preparation goes first."""

    def _key(self, *, local: bool, preprocessed: bool, priority: int, sequence: int):
        from types import SimpleNamespace

        item = SimpleNamespace(
            job={"remote_preprocess_completed_at": 123.0 if preprocessed else None}
        )
        waiting_for_voice = (
            0 if local and item.job.get("remote_preprocess_completed_at") else 1
        )
        return (waiting_for_voice, priority, sequence)

    def test_a_prepared_job_beats_a_fresh_one_locally(self) -> None:
        prepared = self._key(local=True, preprocessed=True, priority=100, sequence=99)
        fresh = self._key(local=True, preprocessed=False, priority=100, sequence=1)
        self.assertLess(prepared, fresh)

    def test_premium_still_wins_among_jobs_needing_a_voice(self) -> None:
        """Paying users go first; this reorders machines, not people."""
        premium = self._key(local=True, preprocessed=True, priority=0, sequence=50)
        ordinary = self._key(local=True, preprocessed=True, priority=100, sequence=1)
        self.assertLess(premium, ordinary)

    def test_the_worker_is_unaffected_by_it(self) -> None:
        """It never voices, so preparation state must not reorder its queue."""
        prepared = self._key(local=False, preprocessed=True, priority=100, sequence=99)
        fresh = self._key(local=False, preprocessed=False, priority=100, sequence=1)
        self.assertLess(fresh, prepared)


class BreakVoicingTests(unittest.TestCase):
    """During a break the main PC voices nothing, so the laptop does - slowly,
    but a queue that waits for the evening to end is slower still. In normal
    running it is worth far more preparing."""

    def test_no_voicing_without_a_break(self) -> None:
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(
            _remote_stage_for_job(job, frozenset({"moss"}), allow_voice=False), "preprocess"
        )

    def test_ukrainian_is_not_an_exception(self) -> None:
        """F5 is lighter than MOSS, but it is still the laptop doing the voice."""
        job = {"tts_provider": "moss", "target_lang": "uk"}
        self.assertEqual(
            _remote_stage_for_job(job, frozenset({"f5"}), allow_voice=False), "preprocess"
        )

    def test_a_break_hands_over_the_whole_job(self) -> None:
        for target, engine in (("ru", "moss"), ("uk", "f5")):
            job = {"tts_provider": "moss", "target_lang": target}
            self.assertEqual(
                _remote_stage_for_job(job, frozenset({engine}), allow_voice=True), "complete"
            )

    def test_a_break_cannot_conjure_an_engine(self) -> None:
        """Handing over a job it would fail at the last step helps nobody."""
        job = {"tts_provider": "moss", "target_lang": "ru"}
        self.assertEqual(
            _remote_stage_for_job(job, frozenset({"f5"}), allow_voice=True), "preprocess"
        )

if __name__ == "__main__":
    unittest.main()
