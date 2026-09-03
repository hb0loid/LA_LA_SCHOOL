from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from laladub.premium_store import PremiumStore


class PremiumStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = PremiumStore(Path(self._tempdir.name) / "premium.sqlite3")

    def test_record_payment_activates_subscription(self) -> None:
        subscription = self.store.record_payment(
            user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30
        )
        self.assertIsNotNone(subscription)
        active = self.store.active_subscription(1)
        self.assertIsNotNone(active)
        self.assertAlmostEqual(active.expires_at, time.time() + 30 * 86400, delta=5)

    def test_duplicate_charge_id_is_idempotent(self) -> None:
        first = self.store.record_payment(user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        self.assertIsNotNone(first)
        duplicate = self.store.record_payment(
            user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30
        )
        self.assertIsNone(duplicate)
        # Only one row should exist - expiry unchanged by the duplicate.
        active = self.store.active_subscription(1)
        self.assertEqual(active.telegram_payment_charge_id, "c1")
        self.assertAlmostEqual(active.expires_at, first.expires_at, delta=1)

    def test_repeat_purchase_stacks_from_current_expiry(self) -> None:
        first = self.store.record_payment(user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        second = self.store.record_payment(user_id=1, telegram_payment_charge_id="c2", stars_amount=250, days=30)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertAlmostEqual(second.expires_at - first.expires_at, 30 * 86400, delta=5)

    def test_active_subscription_none_for_unknown_user(self) -> None:
        self.assertIsNone(self.store.active_subscription(999))

    def test_revoke_deactivates_subscription(self) -> None:
        self.store.record_payment(user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        revoked = self.store.revoke(1, status="refunded")
        self.assertTrue(revoked)
        self.assertIsNone(self.store.active_subscription(1))
        # Revoking again (nothing active) is a no-op, not an error.
        self.assertFalse(self.store.revoke(1, status="refunded"))

    def test_grant_manual_creates_active_subscription_without_real_charge(self) -> None:
        subscription = self.store.grant_manual(user_id=2, days=7, admin_id=555)
        self.assertEqual(subscription.stars_amount, 0)
        self.assertEqual(subscription.granted_by, "555")
        active = self.store.active_subscription(2)
        self.assertIsNotNone(active)
        self.assertAlmostEqual(active.expires_at, time.time() + 7 * 86400, delta=5)

    def test_latest_charge_id_reflects_active_subscription(self) -> None:
        self.assertIsNone(self.store.latest_charge_id(1))
        self.store.record_payment(user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        self.assertEqual(self.store.latest_charge_id(1), "c1")
        self.store.revoke(1, status="refunded")
        self.assertIsNone(self.store.latest_charge_id(1))

    def test_expired_subscription_is_not_active(self) -> None:
        # Simulate an already-expired row by granting a 0-day subscription.
        self.store.record_payment(user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=0)
        self.assertIsNone(self.store.active_subscription(1))

    def test_list_active_subscriptions_dedupes_stacked_purchases(self) -> None:
        self.store.record_payment(user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        self.store.record_payment(user_id=1, telegram_payment_charge_id="c2", stars_amount=250, days=30)
        owners = self.store.list_active_subscriptions()
        self.assertEqual([s.user_id for s in owners], [1])
        self.assertEqual(owners[0].telegram_payment_charge_id, "c2")

    def test_list_active_subscriptions_excludes_expired_and_revoked(self) -> None:
        self.store.record_payment(user_id=1, telegram_payment_charge_id="c1", stars_amount=250, days=30)
        self.store.record_payment(user_id=2, telegram_payment_charge_id="c2", stars_amount=250, days=0)
        self.store.record_payment(user_id=3, telegram_payment_charge_id="c3", stars_amount=250, days=30)
        self.store.revoke(3, status="refunded")
        owners = self.store.list_active_subscriptions()
        self.assertEqual([s.user_id for s in owners], [1])

    def test_list_active_subscriptions_includes_manual_grants(self) -> None:
        self.store.grant_manual(user_id=9, days=7, admin_id=555)
        owners = self.store.list_active_subscriptions()
        self.assertEqual([s.user_id for s in owners], [9])
        self.assertEqual(owners[0].granted_by, "555")

    def test_user_settings_default_to_watermark_on_no_censor_override(self) -> None:
        settings = self.store.get_user_settings(42)
        self.assertTrue(settings.watermark_enabled)
        self.assertIsNone(settings.censor_percent)

    def test_set_watermark_enabled_persists_and_is_independent_of_censor(self) -> None:
        self.store.set_censor_percent(1, 40)
        self.store.set_watermark_enabled(1, False)
        settings = self.store.get_user_settings(1)
        self.assertFalse(settings.watermark_enabled)
        self.assertEqual(settings.censor_percent, 40)

        self.store.set_watermark_enabled(1, True)
        settings = self.store.get_user_settings(1)
        self.assertTrue(settings.watermark_enabled)
        self.assertEqual(settings.censor_percent, 40)

    def test_set_censor_percent_none_clears_override(self) -> None:
        self.store.set_censor_percent(1, 75)
        self.assertEqual(self.store.get_user_settings(1).censor_percent, 75)
        self.store.set_censor_percent(1, None)
        self.assertIsNone(self.store.get_user_settings(1).censor_percent)


if __name__ == "__main__":
    unittest.main()
