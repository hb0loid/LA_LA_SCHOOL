from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from laladub.bot import _status_job_line, status_command


class _Message:
    def __init__(self) -> None:
        self.text: str | None = None

    async def reply_text(self, text: str, **kwargs: object) -> None:
        self.text = text


def _run(report: dict[str, object], *, admin: bool = True) -> str | None:
    message = _Message()
    settings = SimpleNamespace(is_admin=lambda user_id: admin)
    scheduler = SimpleNamespace(status_report=lambda: _as_coro(report))
    context = SimpleNamespace(
        application=SimpleNamespace(
            bot_data={"settings": settings, "job_scheduler": scheduler}
        )
    )
    update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
    asyncio.run(status_command(update, context))
    return message.text


async def _as_coro(value: dict[str, object]) -> dict[str, object]:
    return value


def _report(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "now": 0.0,
        "running": [],
        "local": [],
        "pending": [],
        "workers": [],
        "traffic_quiet": None,
        "active_total": 0,
        "active_local": 0,
        "max_active_jobs": 2,
        "max_local_jobs": 1,
        "executor_mode": "hybrid",
        "maintenance": False,
    }
    base.update(overrides)
    return base


class StatusCommandTests(unittest.TestCase):
    def test_an_idle_system_reads_as_idle(self) -> None:
        text = _run(_report())
        assert text is not None
        self.assertIn("свободен", text)
        self.assertIn("Воркеров нет", text)
        self.assertIn("Очередь пуста", text)

    def test_it_is_admin_only(self) -> None:
        """It exposes other people's jobs and the scheduler's internals."""
        self.assertIsNone(_run(_report(), admin=False))

    def test_an_abandoned_lease_is_called_out(self) -> None:
        """The failure that cost a slot all morning: the coordinator holding a
        job for a worker that had already gone back to asking for new work."""
        text = _run(
            _report(
                active_total=2,
                running=[
                    {
                        "number": "55022",
                        "user_id": 1,
                        "where": "worker-pc",
                        "kind": "remote_preprocess",
                        "source": "vi",
                        "target": "ru",
                        "premium": False,
                        "stage": "Распознавание",
                        "percent": 40,
                        "elapsed": 12600.0,
                        "silence": 12600.0,
                        "moved_on": True,
                    }
                ],
            )
        )
        assert text is not None
        self.assertIn("#55022", text)
        self.assertIn("занят другим", text)
        self.assertIn("3:30:00", text)

    def test_a_waiting_queue_shows_why_each_job_waits(self) -> None:
        text = _run(
            _report(
                pending=[
                    {
                        "number": "55100",
                        "user_id": 2,
                        "source": "vi",
                        "target": "ru",
                        "premium": True,
                        "priority": 0,
                        "waiting": 90.0,
                        "forced_local": False,
                        "preprocessed": True,
                    }
                ]
            )
        )
        assert text is not None
        self.assertIn("1. ⭐ #55100", text)
        self.assertIn("озвучка на основном ПК", text)
        self.assertIn("ждёт 1:30", text)

    def test_a_long_queue_is_folded_away(self) -> None:
        """Eleven waiting jobs buried the machines, which is the part you open
        /status to look at. They go in a quote that starts closed."""
        pending = [
            {
                "number": str(n),
                "user_id": 1,
                "source": "vi",
                "target": "ru",
                "premium": False,
                "priority": 100,
                "waiting": 1.0,
                "forced_local": False,
                "preprocessed": False,
            }
            for n in range(20)
        ]
        text = _run(_report(pending=pending))
        assert text is not None
        self.assertIn("В очереди: 20", text)
        self.assertIn("<blockquote expandable>", text)
        # Folded, not dropped: all twenty are in there.
        self.assertIn("20. ", text)


class StatusLineTests(unittest.TestCase):
    def test_premium_is_marked(self) -> None:
        entry = {"number": "1", "source": "vi", "target": "ru", "premium": True}
        self.assertTrue(_status_job_line(entry).startswith("⭐"))

    def test_a_missing_number_does_not_break_the_line(self) -> None:
        entry = {"number": "", "source": None, "target": None, "premium": False}
        self.assertIn("#?", _status_job_line(entry))



class StatusLengthTests(unittest.TestCase):
    """Telegram refuses a message over 4096 characters outright, and the queue
    is the part that grows without limit. Better a trimmed list than no report."""

    def test_a_huge_queue_still_fits(self) -> None:
        pending = [
            {
                "number": f"5{n:04d}",
                "user_id": 1,
                "source": "vi",
                "target": "ru",
                "premium": False,
                "priority": 100,
                "waiting": 3600.0,
                "forced_local": False,
                "preprocessed": False,
            }
            for n in range(200)
        ]
        text = _run(_report(pending=pending))
        assert text is not None
        self.assertLessEqual(len(text), 4096)
        self.assertIn("В очереди: 200", text)
        self.assertIn("и ещё", text)
        self.assertTrue(text.count("<blockquote expandable>") == 1)
        self.assertTrue(text.rstrip().endswith("</i>"))

if __name__ == "__main__":
    unittest.main()
