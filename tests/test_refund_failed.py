from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from laladub.bot import _refund_failed_job
from laladub.proposal_store import ProposalStore

LIMIT_MS = 10 * 60_000


class RefundFailedJobTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposals.sqlite3")

    def _context(self) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"proposal_store": self.store})
        )

    def _item(self, user_id: int, job_number: str) -> SimpleNamespace:
        return SimpleNamespace(user_id=user_id, job={"job_dir": f"runs/{job_number}"})

    def _reserve(self, user_id: int, job_number: str, minutes: float) -> None:
        ok, _used = self.store.reserve_daily_usage(
            user_id=user_id,
            job_number=job_number,
            duration_ms=int(minutes * 60_000),
            limit_ms=LIMIT_MS,
        )
        self.assertTrue(ok)

    async def test_the_minutes_come_back(self) -> None:
        self._reserve(1, "42", 3.0)
        await _refund_failed_job(self._context(), self._item(1, "42"))
        self.assertEqual(self.store.daily_usage_ms(1), 0)

    async def test_refunding_twice_is_harmless(self) -> None:
        # It is called from finally, and the failure paths may reach it more
        # than once; release is a plain DELETE.
        self._reserve(1, "42", 3.0)
        await _refund_failed_job(self._context(), self._item(1, "42"))
        await _refund_failed_job(self._context(), self._item(1, "42"))
        self.assertEqual(self.store.daily_usage_ms(1), 0)

    async def test_other_jobs_keep_their_minutes(self) -> None:
        self._reserve(1, "42", 3.0)
        self._reserve(1, "43", 2.0)
        await _refund_failed_job(self._context(), self._item(1, "42"))
        self.assertEqual(self.store.daily_usage_ms(1), 2 * 60_000)

    async def test_another_user_is_untouched(self) -> None:
        self._reserve(1, "42", 3.0)
        self._reserve(2, "42", 4.0)
        await _refund_failed_job(self._context(), self._item(1, "42"))
        self.assertEqual(self.store.daily_usage_ms(2), 4 * 60_000)

    async def test_the_freed_minutes_can_be_spent_again(self) -> None:
        # The point of the refund: a failure must not cost the user their day.
        self._reserve(1, "42", 9.0)
        await _refund_failed_job(self._context(), self._item(1, "42"))
        ok, _used = self.store.reserve_daily_usage(
            user_id=1, job_number="44", duration_ms=9 * 60_000, limit_ms=LIMIT_MS
        )
        self.assertTrue(ok)

    async def test_a_failing_store_does_not_break_the_failure_path(self) -> None:
        # This runs in a finally block while a job is already failing; raising
        # here would bury the real error.
        with patch(
            "laladub.bot._release_daily_allowance",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            await _refund_failed_job(self._context(), self._item(1, "42"))

    async def test_a_missing_user_is_ignored(self) -> None:
        await _refund_failed_job(self._context(), SimpleNamespace(user_id=None, job={}))


if __name__ == "__main__":
    unittest.main()
