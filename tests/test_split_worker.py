from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from laladub.bot import _ApplicationContext, _JobScheduler, _install_preprocess_bundle, _remote_stage_for_job
from laladub.job_runner import JobDocument, JobExecutionResult, result_manifest


class SplitWorkerTests(unittest.TestCase):
    def test_moss_job_uses_remote_preprocessing(self) -> None:
        self.assertEqual(_remote_stage_for_job({"mode": "dub", "tts_provider": "moss"}), "preprocess")
        self.assertEqual(_remote_stage_for_job({"mode": "dub"}), "preprocess")
        self.assertEqual(_remote_stage_for_job({"mode": "dub", "tts_provider": "f5"}), "complete")
        self.assertEqual(_remote_stage_for_job({"mode": "raw_text", "tts_provider": "moss"}), "complete")

    def test_preprocess_manifest_names_bundle(self) -> None:
        result = JobExecutionResult(
            mode="preprocess",
            documents=[JobDocument(Path("preprocess_bundle.zip"), "preprocess_bundle.zip")],
            preprocess_seconds=12.5,
        )
        manifest = result_manifest(result)
        self.assertEqual(manifest["preprocess"]["filename"], "preprocess_bundle.zip")
        self.assertEqual(manifest["preprocess_seconds"], 12.5)

    def test_preprocess_bundle_replaces_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            job_dir = root / "job"
            workdir = job_dir / "work"
            workdir.mkdir(parents=True)
            (workdir / "old.txt").write_text("old", encoding="utf-8")
            archive_path = root / "bundle.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("translated.srt", "translated")
                archive.writestr("source_16k.wav", b"wave")
                archive.writestr("resume_state.json", "{}")
            _install_preprocess_bundle(archive_path, job_dir)
            self.assertFalse((workdir / "old.txt").exists())
            self.assertEqual((workdir / "translated.srt").read_text(encoding="utf-8"), "translated")

    def test_preprocess_bundle_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            archive_path = root / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "bad")
            with self.assertRaises(RuntimeError):
                _install_preprocess_bundle(archive_path, root / "job")
            self.assertFalse((root / "outside.txt").exists())


class _DummyStatus:
    async def edit_text(self, _text: str) -> None:
        return None


class _DummySettings(SimpleNamespace):
    def is_paid(self, _user_id: int | None) -> bool:
        return True

    def is_admin(self, _user_id: int | None) -> bool:
        return False


class SplitSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_preprocessed_job_returns_to_local_queue(self) -> None:
        import asyncio

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            job_dir = root / "job"
            job_dir.mkdir()
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            settings = _DummySettings(
                executor_mode="hybrid",
                max_active_jobs=2,
                max_active_jobs_per_user=1,
                max_local_jobs=0,
                workdir=root,
                tts="moss",
            )
            scheduler = _JobScheduler(settings)
            application = SimpleNamespace(
                bot=SimpleNamespace(),
                bot_data={"job_scheduler": scheduler},
                create_task=asyncio.create_task,
            )
            context = _ApplicationContext(application)
            self.assertIsNone(await scheduler.lease_remote(context, "worker-test"))
            job = {
                "job_dir": str(job_dir),
                "input_path": str(input_path),
                "target_lang": "ru",
                "tts_provider": "moss",
                "mode": "dub",
            }
            await scheduler.enqueue(
                context,
                chat_id=1,
                user_id=None,
                job=job,
                status_message=_DummyStatus(),
            )
            lease = await scheduler.lease_remote(context, "worker-test")
            self.assertIsNotNone(lease)
            self.assertEqual(lease["job"]["remote_stage"], "preprocess")

            result_dir = job_dir / "remote_result" / "documents"
            result_dir.mkdir(parents=True)
            bundle = result_dir / "preprocess_bundle.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("translated.srt", "translated")
                archive.writestr("source_16k.wav", b"wave")
                archive.writestr("resume_state.json", "{}")
            await scheduler.complete_remote(
                context,
                lease["job_id"],
                {
                    "mode": "preprocess",
                    "preprocess": {"filename": bundle.name},
                    "preprocess_seconds": 4.5,
                },
            )
            for _ in range(20):
                if job.get("remote_preprocess_completed_at"):
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(job.get("remote_preprocess_completed_at"))
            self.assertEqual(job.get("remote_preprocess_seconds"), 4.5)
            self.assertTrue((job_dir / "work" / "translated.srt").is_file())
            snapshot = await scheduler.snapshot()
            self.assertEqual(snapshot["leased_remote"], 0)
            self.assertEqual(snapshot["pending_total"], 1)


if __name__ == "__main__":
    unittest.main()
