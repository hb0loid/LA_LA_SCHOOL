from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from laladub.bot import _reserve_daily_allowance, _today_key
from laladub.premium_store import PremiumStore
from laladub.proposal_store import ProposalStore


class _Status:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def edit_text(self, text: str) -> None:
        self.messages.append(text)


class _Settings(SimpleNamespace):
    def is_paid(self, _user_id: int | None) -> bool:
        return False

    def is_admin(self, _user_id: int | None) -> bool:
        return False


class DailyQuotaTrimTests(unittest.IsolatedAsyncioTestCase):
    async def test_video_is_trimmed_to_remaining_daily_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            store = ProposalStore(root / "proposal.sqlite3")
            store.reserve_daily_usage(
                user_id=123,
                job_number="previous",
                day_key=_today_key(),
                duration_ms=30_000,
                limit_ms=60_000,
            )
            input_path = root / "job" / "input.mp4"
            input_path.parent.mkdir()
            input_path.write_bytes(b"source")
            job = {
                "job_dir": str(input_path.parent),
                "input_path": str(input_path),
            }
            context = SimpleNamespace(
                application=SimpleNamespace(
                    bot_data={
                        "settings": _Settings(),
                        "proposal_store": store,
                    }
                )
            )
            status = _Status()

            def fake_trim(_source: Path, destination: Path, _duration: float) -> None:
                destination.write_bytes(b"x" * 2048)

            with (
                patch("laladub.bot.probe_duration", return_value=120.0),
                patch("laladub.bot.trim_video", side_effect=fake_trim) as trim_mock,
                patch("laladub.bot.has_video_and_audio", return_value=True),
            ):
                accepted = await _reserve_daily_allowance(
                    context,
                    status_message=status,
                    user_id=123,
                    job=job,
                )

            self.assertTrue(accepted)
            self.assertTrue(job["daily_trimmed"])
            self.assertEqual(job["daily_original_duration_ms"], 120_000)
            self.assertEqual(job["daily_trimmed_duration_ms"], 30_000)
            self.assertEqual(job["quota_duration_ms"], 30_000)
            self.assertEqual(store.daily_usage_ms(123, _today_key()), 60_000)
            self.assertTrue(Path(job["input_path"]).is_file())
            trim_mock.assert_called_once()
            self.assertTrue(any("В работу пойдёт" in message for message in status.messages))


class PremiumQuotaTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_subscription_uses_capped_premium_limit_not_karma_level(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            store = ProposalStore(root / "proposal.sqlite3")
            premium_store = PremiumStore(root / "premium.sqlite3")
            premium_store.record_payment(
                user_id=123, telegram_payment_charge_id="c1", stars_amount=250, days=30
            )
            input_path = root / "job" / "input.mp4"
            input_path.parent.mkdir()
            input_path.write_bytes(b"source")
            job = {"job_dir": str(input_path.parent), "input_path": str(input_path)}
            context = SimpleNamespace(
                application=SimpleNamespace(
                    bot_data={
                        "settings": _Settings(),
                        "proposal_store": store,
                        "premium_store": premium_store,
                    }
                )
            )
            status = _Status()

            # A karma level-0 user is normally capped at 1 minute/day; a 40-minute
            # video would be rejected/trimmed on that tier but fits the premium
            # tier's 60 minutes/day, so it should be accepted untrimmed.
            with patch("laladub.bot.probe_duration", return_value=40 * 60.0):
                accepted = await _reserve_daily_allowance(
                    context, status_message=status, user_id=123, job=job
                )

            self.assertTrue(accepted)
            self.assertNotIn("daily_trimmed", job)
            self.assertEqual(store.daily_usage_ms(123, _today_key()), 40 * 60 * 1000)


if __name__ == "__main__":
    unittest.main()
