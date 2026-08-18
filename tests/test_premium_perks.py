from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from laladub.bot import (
    _describe_telegram_user,
    _get_censor_percent,
    _is_premium_user,
    _remember_job_and_ask_source,
    _set_censor_percent,
    admin_prem_owners,
    censored,
    mycensor_command,
    watermark_command,
)
from laladub.premium_store import PremiumStore


class _Status:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def edit_text(self, text: str, reply_markup: object = None) -> None:
        self.messages.append(text)

    async def reply_text(self, text: str, reply_markup: object = None) -> None:
        self.messages.append(text)


class _Message:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply_text(self, text: str, reply_markup: object = None) -> None:
        self.replies.append(text)


class _Settings(SimpleNamespace):
    def __init__(
        self, *, paid: set[int] = frozenset(), admins: set[int] = frozenset(), workdir: Path | None = None
    ) -> None:
        super().__init__()
        self._paid = paid
        self._admins = admins
        self.workdir = workdir or Path(tempfile.mkdtemp())

    def is_paid(self, user_id: int | None) -> bool:
        return user_id in self._paid

    def is_admin(self, user_id: int | None) -> bool:
        return user_id in self._admins


def _context(
    settings: _Settings, premium_store: PremiumStore, *, args: list[str] | None = None, bot: object | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        application=SimpleNamespace(
            bot_data={"settings": settings, "premium_store": premium_store},
        ),
        user_data={},
        args=args or [],
        # Defaults to "lookup unavailable" so callers that never touch context.bot
        # (most commands) are unaffected, and admin_prem_owners falls back to the
        # bare user id exactly like it did before username lookup existed.
        bot=bot or SimpleNamespace(get_chat=AsyncMock(side_effect=Exception("no chat history"))),
    )


class IsPremiumUserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PremiumStore(Path(self._tempdir.name) / "premium.sqlite3")

    def test_paid_allowlist_is_premium(self) -> None:
        settings = _Settings(paid={1})
        self.assertTrue(_is_premium_user(settings, self.store, 1))

    def test_admin_is_premium(self) -> None:
        settings = _Settings(admins={2})
        self.assertTrue(_is_premium_user(settings, self.store, 2))

    def test_active_subscription_is_premium(self) -> None:
        settings = _Settings()
        self.store.record_payment(user_id=3, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        self.assertTrue(_is_premium_user(settings, self.store, 3))

    def test_plain_user_is_not_premium(self) -> None:
        settings = _Settings()
        self.assertFalse(_is_premium_user(settings, self.store, 4))

    def test_none_user_id_is_not_premium(self) -> None:
        settings = _Settings()
        self.assertFalse(_is_premium_user(settings, self.store, None))


class RememberJobAppliesPerksTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PremiumStore(Path(self._tempdir.name) / "premium.sqlite3")

    async def test_non_premium_user_gets_global_censor_and_watermark_on(self) -> None:
        settings = _Settings()
        context = _context(settings, self.store)
        job_dir = Path(self._tempdir.name) / "job"
        job_dir.mkdir()
        await _remember_job_and_ask_source(
            context, _Status(), job_dir, job_dir / "input.mp4", "title",
            input_source="telegram_audio", user_id=999,
        )
        job = context.user_data["job"]
        self.assertTrue(job["watermark_enabled"])
        self.assertEqual(job["censor_percent"], 100)

    async def test_premium_user_gets_personal_overrides(self) -> None:
        settings = _Settings(paid={1})
        self.store.set_watermark_enabled(1, False)
        self.store.set_censor_percent(1, 55)
        context = _context(settings, self.store)
        job_dir = Path(self._tempdir.name) / "job2"
        job_dir.mkdir()
        await _remember_job_and_ask_source(
            context, _Status(), job_dir, job_dir / "input.mp4", "title",
            input_source="telegram_audio", user_id=1,
        )
        job = context.user_data["job"]
        self.assertFalse(job["watermark_enabled"])
        self.assertEqual(job["censor_percent"], 55)

    async def test_premium_user_without_explicit_censor_override_uses_global(self) -> None:
        settings = _Settings(paid={1})
        self.store.set_watermark_enabled(1, False)
        context = _context(settings, self.store)
        job_dir = Path(self._tempdir.name) / "job3"
        job_dir.mkdir()
        await _remember_job_and_ask_source(
            context, _Status(), job_dir, job_dir / "input.mp4", "title",
            input_source="telegram_audio", user_id=1,
        )
        job = context.user_data["job"]
        self.assertFalse(job["watermark_enabled"])
        self.assertEqual(job["censor_percent"], 100)


class WatermarkCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PremiumStore(Path(self._tempdir.name) / "premium.sqlite3")

    async def test_non_premium_user_is_told_its_a_perk(self) -> None:
        settings = _Settings()
        context = _context(settings, self.store, args=["off"])
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await watermark_command(update, context)
        self.assertTrue(any("премиум" in text.lower() for text in message.replies))
        self.assertTrue(self.store.get_user_settings(1).watermark_enabled)

    async def test_premium_user_can_toggle_off_and_on(self) -> None:
        settings = _Settings(paid={1})
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)

        await watermark_command(update, _context(settings, self.store, args=["off"]))
        self.assertFalse(self.store.get_user_settings(1).watermark_enabled)

        await watermark_command(update, _context(settings, self.store, args=["on"]))
        self.assertTrue(self.store.get_user_settings(1).watermark_enabled)


class MyCensorCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PremiumStore(Path(self._tempdir.name) / "premium.sqlite3")

    async def test_non_premium_user_is_told_its_a_perk(self) -> None:
        settings = _Settings()
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await mycensor_command(update, _context(settings, self.store, args=["50"]))
        self.assertTrue(any("премиум" in text.lower() for text in message.replies))
        self.assertIsNone(self.store.get_user_settings(1).censor_percent)

    async def test_premium_user_can_set_and_clear_override(self) -> None:
        settings = _Settings(paid={1})
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=_Message())

        await mycensor_command(update, _context(settings, self.store, args=["30"]))
        self.assertEqual(self.store.get_user_settings(1).censor_percent, 30)

        await mycensor_command(update, _context(settings, self.store, args=["default"]))
        self.assertIsNone(self.store.get_user_settings(1).censor_percent)

    async def test_premium_user_invalid_value_is_rejected(self) -> None:
        settings = _Settings(paid={1})
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await mycensor_command(update, _context(settings, self.store, args=["150"]))
        self.assertIsNone(self.store.get_user_settings(1).censor_percent)
        self.assertTrue(any("Использование" in text for text in message.replies))


class GlobalCensorDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.settings = _Settings(workdir=Path(self._tempdir.name))

    def test_defaults_to_fully_on_when_never_configured(self) -> None:
        self.assertEqual(_get_censor_percent(self.settings), 100)

    def test_explicit_zero_persists_instead_of_falling_back_to_default(self) -> None:
        _set_censor_percent(self.settings, 0)
        self.assertEqual(_get_censor_percent(self.settings), 0)

    def test_explicit_value_persists(self) -> None:
        _set_censor_percent(self.settings, 42)
        self.assertEqual(_get_censor_percent(self.settings), 42)


class CensoredCommandGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PremiumStore(Path(self._tempdir.name) / "premium.sqlite3")
        self.workdir = Path(self._tempdir.name) / "bot"

    async def test_non_admin_non_premium_is_pointed_to_premium(self) -> None:
        settings = _Settings(workdir=self.workdir)
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await censored(update, _context(settings, self.store, args=["0"]))
        self.assertTrue(any("администратору" in text for text in message.replies))
        self.assertTrue(any("/premium" in text for text in message.replies))
        self.assertEqual(_get_censor_percent(settings), 100)

    async def test_non_admin_premium_user_is_pointed_to_mycensor(self) -> None:
        settings = _Settings(paid={1}, workdir=self.workdir)
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await censored(update, _context(settings, self.store, args=["0"]))
        self.assertTrue(any("администратору" in text for text in message.replies))
        self.assertTrue(any("/mycensor" in text for text in message.replies))
        self.assertEqual(_get_censor_percent(settings), 100)

    async def test_admin_can_change_global_censor(self) -> None:
        settings = _Settings(admins={7}, workdir=self.workdir)
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=7), effective_message=message)
        await censored(update, _context(settings, self.store, args=["30"]))
        self.assertEqual(_get_censor_percent(settings), 30)


class PremOwnersCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PremiumStore(Path(self._tempdir.name) / "premium.sqlite3")

    async def test_non_admin_is_rejected(self) -> None:
        settings = _Settings()
        self.store.record_payment(user_id=42, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)
        await admin_prem_owners(update, _context(settings, self.store))
        self.assertTrue(any("администратору" in text for text in message.replies))
        self.assertFalse(any("42" in text for text in message.replies))

    async def test_admin_sees_active_subscribers(self) -> None:
        settings = _Settings(admins={7})
        self.store.record_payment(user_id=42, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=7), effective_message=message)
        await admin_prem_owners(update, _context(settings, self.store))
        self.assertTrue(any("42" in text for text in message.replies))

    async def test_admin_sees_empty_state_message(self) -> None:
        settings = _Settings(admins={7})
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=7), effective_message=message)
        await admin_prem_owners(update, _context(settings, self.store))
        self.assertTrue(any("нет" in text for text in message.replies))

    async def test_shows_username_when_it_can_be_resolved(self) -> None:
        settings = _Settings(admins={7})
        self.store.record_payment(user_id=42, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=7), effective_message=message)
        bot = SimpleNamespace(
            get_chat=AsyncMock(
                return_value=SimpleNamespace(username="hboloid", first_name="Hyper", last_name=None)
            )
        )
        await admin_prem_owners(update, _context(settings, self.store, bot=bot))
        self.assertTrue(any("@hboloid" in text and "Hyper" in text and "42" in text for text in message.replies))

    async def test_falls_back_to_bare_id_when_lookup_fails(self) -> None:
        settings = _Settings(admins={7})
        self.store.record_payment(user_id=42, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        message = _Message()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=7), effective_message=message)
        # Default _context() bot.get_chat raises - simulates a user who blocked
        # the bot or never started a chat with it, so Telegram has nothing to give.
        await admin_prem_owners(update, _context(settings, self.store))
        self.assertTrue(any("42" in text for text in message.replies))


class DescribeTelegramUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_username_and_name_both_present(self) -> None:
        bot = SimpleNamespace(
            get_chat=AsyncMock(return_value=SimpleNamespace(username="hboloid", first_name="Hyper", last_name="B"))
        )
        self.assertEqual(await _describe_telegram_user(bot, 42), "@hboloid Hyper B (42)")

    async def test_only_name_no_username(self) -> None:
        bot = SimpleNamespace(
            get_chat=AsyncMock(return_value=SimpleNamespace(username=None, first_name="Hyper", last_name=None))
        )
        self.assertEqual(await _describe_telegram_user(bot, 42), "Hyper (42)")

    async def test_nothing_resolvable_falls_back_to_bare_id(self) -> None:
        bot = SimpleNamespace(
            get_chat=AsyncMock(return_value=SimpleNamespace(username=None, first_name=None, last_name=None))
        )
        self.assertEqual(await _describe_telegram_user(bot, 42), "42")

    async def test_lookup_failure_falls_back_to_bare_id(self) -> None:
        bot = SimpleNamespace(get_chat=AsyncMock(side_effect=Exception("blocked")))
        self.assertEqual(await _describe_telegram_user(bot, 42), "42")


if __name__ == "__main__":
    unittest.main()
