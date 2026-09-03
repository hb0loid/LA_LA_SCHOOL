from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from laladub.bot import _ApplicationContext, _JobScheduler, _set_maintenance_enabled


class _Status:
    def __init__(self) -> None:
        self.text = ""

    async def edit_text(self, text: str, reply_markup: object = None) -> None:
        self.text = text


class _Settings(SimpleNamespace):
    def is_paid(self, _user_id: int | None) -> bool:
        return True

    def is_admin(self, user_id: int | None) -> bool:
        return user_id in getattr(self, "admins", set())


class _ZeroKarmaStore:
    def karma_total(self, _user_id: int) -> int:
        return 0


class MaintenanceInterruptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)
        self.settings = _Settings(
            executor_mode="local",
            max_active_jobs=2,
            max_active_jobs_per_user=2,
            max_local_jobs=2,
            workdir=self.root,
            tts="moss",
            admins={999},
        )
        self.scheduler = _JobScheduler(self.settings)
        self.application = SimpleNamespace(
            bot=SimpleNamespace(),
            bot_data={"job_scheduler": self.scheduler, "proposal_store": _ZeroKarmaStore()},
            create_task=asyncio.create_task,
        )
        self.context = _ApplicationContext(self.application)

    async def _enqueue(self, name: str, user_id: int) -> tuple[Path, _Status]:
        job_dir = self.root / name
        job_dir.mkdir()
        status = _Status()
        accepted = await self.scheduler.enqueue(
            self.context,
            chat_id=1,
            user_id=user_id,
            job={"job_dir": str(job_dir), "mode": "dub", "tts_provider": "moss"},
            status_message=status,
        )
        self.assertTrue(accepted)
        return job_dir, status

    async def test_turning_maintenance_on_cancels_a_running_job(self) -> None:
        started = asyncio.Event()

        async def never_finishes(*_args, **_kwargs):
            started.set()
            await asyncio.sleep(3600)

        with unittest.mock.patch("laladub.bot._process_job", new=never_finishes):
            job_dir, status = await self._enqueue("job1", user_id=123)
            await asyncio.wait_for(started.wait(), timeout=5)

            _set_maintenance_enabled(self.settings, True)
            await self.scheduler.maintenance_changed(self.context)
            # Let the cancellation unwind and the cleanup task run.
            for _ in range(50):
                await asyncio.sleep(0.01)

        snapshot = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["status"], "queued")
        self.assertEqual(snapshot["interrupted_by"], "maintenance")
        self.assertIn("технические работы", status.text)

    async def test_admin_jobs_keep_running_through_maintenance(self) -> None:
        started = asyncio.Event()

        async def never_finishes(*_args, **_kwargs):
            started.set()
            await asyncio.sleep(3600)

        with unittest.mock.patch("laladub.bot._process_job", new=never_finishes):
            job_dir, _status = await self._enqueue("adminjob", user_id=999)
            await asyncio.wait_for(started.wait(), timeout=5)

            _set_maintenance_enabled(self.settings, True)
            await self.scheduler.maintenance_changed(self.context)
            for _ in range(20):
                await asyncio.sleep(0.01)

            snapshot = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["status"], "starting")
            self.assertNotIn("interrupted_by", snapshot)

    async def test_turning_maintenance_off_does_not_cancel_anything(self) -> None:
        started = asyncio.Event()

        async def never_finishes(*_args, **_kwargs):
            started.set()
            await asyncio.sleep(3600)

        with unittest.mock.patch("laladub.bot._process_job", new=never_finishes):
            job_dir, _status = await self._enqueue("job2", user_id=123)
            await asyncio.wait_for(started.wait(), timeout=5)

            _set_maintenance_enabled(self.settings, False)
            await self.scheduler.maintenance_changed(self.context)
            for _ in range(20):
                await asyncio.sleep(0.01)

            snapshot = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["status"], "starting")


if __name__ == "__main__":
    unittest.main()
