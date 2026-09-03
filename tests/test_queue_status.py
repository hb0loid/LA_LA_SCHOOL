from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from laladub.bot import _job_status_counts, queue_status


class JobStatusCountsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workdir = Path(self._tempdir.name)

    def _write_job(self, subpath: str, status: str) -> None:
        job_dir = self.workdir / subpath
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    def test_missing_workdir_is_empty(self) -> None:
        self.assertEqual(_job_status_counts(self.workdir / "nope"), {})

    def test_terminal_statuses_are_excluded(self) -> None:
        # /queue's file scan exists to catch jobs stranded mid-flight; finished
        # ones pile up by the thousand under the retention window.
        self._write_job("1/1", "done")
        self._write_job("1/2", "failed")
        self._write_job("1/3", "rejected")
        self._write_job("1/4", "running")
        self.assertEqual(_job_status_counts(self.workdir), {"running": 1})

    def test_counts_active_statuses(self) -> None:
        self._write_job("1/1", "running")
        self._write_job("1/2", "running")
        self._write_job("1/3", "queued")
        self._write_job("1/4", "select_source")
        self.assertEqual(
            _job_status_counts(self.workdir), {"running": 2, "queued": 1, "select_source": 1}
        )

    def test_unreadable_json_counts_as_bad_json(self) -> None:
        job_dir = self.workdir / "1" / "1"
        job_dir.mkdir(parents=True)
        (job_dir / "job.json").write_text("not json", encoding="utf-8")
        self.assertEqual(_job_status_counts(self.workdir), {"bad_json": 1})


def _live(**overrides: object) -> dict[str, object]:
    base = {
        "busy_machines": 0,
        "online_machines": 2,
        "local_machine_busy": 0,
        "local_machine_total": 1,
        "active_local": 0,
        "max_local_jobs": 1,
        "remote_workers_busy": 0,
        "remote_workers_online": 1,
        "remote_workers_idle": 1,
        "remote_workers_stale": 0,
        "active_total": 0,
        "max_active_jobs": 2,
        "pending_total": 0,
        "pending_premium": 0,
        "pending_normal": 0,
        "active_users": 0,
    }
    base.update(overrides)
    return base


class QueueStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workdir = Path(self._tempdir.name)

    def _write_job(self, subpath: str, status: str) -> None:
        job_dir = self.workdir / subpath
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    async def _render(self, live: dict[str, object]) -> str:
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_message=message)
        context = SimpleNamespace(
            application=SimpleNamespace(
                bot_data={
                    "settings": SimpleNamespace(workdir=self.workdir),
                    "job_scheduler": SimpleNamespace(snapshot=AsyncMock(return_value=live)),
                }
            )
        )
        await queue_status(update, context)
        return message.reply_text.call_args.args[0]

    async def test_idle_machines_read_as_free(self) -> None:
        text = await self._render(_live())
        self.assertIn("Сейчас в работе: 0 из 2", text)
        self.assertIn("Основной ПК — свободен", text)
        self.assertIn("Воркер — свободен", text)
        self.assertIn("Очередь пуста", text)

    async def test_busy_machines_read_as_busy(self) -> None:
        text = await self._render(
            _live(active_total=2, local_machine_busy=1, active_local=1, remote_workers_busy=1, active_users=2)
        )
        self.assertIn("Сейчас в работе: 2 из 2", text)
        self.assertIn("Основной ПК — занят", text)
        self.assertIn("Воркер — занят", text)

    async def test_offline_worker_is_called_out(self) -> None:
        text = await self._render(_live(remote_workers_online=0, remote_workers_idle=0))
        self.assertIn("Воркер — не на связи", text)

    async def test_queue_split_shown_only_when_both_kinds_wait(self) -> None:
        mixed = await self._render(_live(pending_total=3, pending_premium=1, pending_normal=2))
        self.assertIn("Ждут очереди: 3", mixed)
        self.assertIn("премиум 1", mixed)

        premium_only = await self._render(_live(pending_total=2, pending_premium=2))
        self.assertIn("Ждут очереди: 2", premium_only)
        self.assertNotIn("премиум", premium_only)

    async def test_stranded_jobs_are_reported_with_resume_hint(self) -> None:
        for index in range(3):
            self._write_job(f"1/{index}", "running")
        text = await self._render(_live())
        self.assertIn("Зависли после перезапуска: 3", text)
        self.assertIn("/resume", text)

    async def test_jobs_the_scheduler_knows_about_are_not_called_stranded(self) -> None:
        self._write_job("1/1", "running")
        self._write_job("1/2", "running")
        text = await self._render(_live(active_total=2))
        self.assertNotIn("Зависли", text)

    async def test_abandoned_dialogs_are_summed_into_one_line(self) -> None:
        self._write_job("1/1", "select_visual")
        self._write_job("1/2", "select_source")
        self._write_job("1/3", "select_tts")
        text = await self._render(_live())
        self.assertIn("Брошенные диалоги: 3", text)
        # The raw per-status dump is what made this section unreadable.
        self.assertNotIn("select_visual", text)
        self.assertNotIn("select_source", text)

    async def test_clean_state_hides_the_problem_block(self) -> None:
        text = await self._render(_live())
        for marker in ("Зависли", "Брошенные диалоги", "Повреждённых"):
            self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
