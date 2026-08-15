from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.proposal_bot import _author_caption, _karma_tag, _moderation_keyboard
from laladub.proposal_store import ProposalStore, Submission


class ProposalStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = ProposalStore(Path(self.tempdir.name) / "proposal.sqlite3")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_submission_is_idempotent_and_karma_is_adjusted(self) -> None:
        submission, created = self.store.create_submission(
            job_number="38598",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username="test_user",
            video_path=Path("video.mp4"),
            output_filename="video_lalaschool.mp4",
        )
        self.assertTrue(created)
        duplicate, created_again = self.store.create_submission(
            job_number="38598",
            user_id=123,
            chat_id=123,
            author_name="Новое имя",
            author_username="new_user",
            video_path=Path("video.mp4"),
            output_filename="video_lalaschool.mp4",
        )
        self.assertFalse(created_again)
        self.assertEqual(duplicate.id, submission.id)
        self.assertEqual(duplicate.author_name, "Новое имя")

        self.assertIsNotNone(self.store.try_claim(submission.id, 631551040))
        _updated, delta = self.store.finish_decision(
            submission_id=submission.id,
            moderator_id=631551040,
            destination="shame",
            award=1,
            publication_chat_id=-1001,
            publication_message_id=10,
        )
        self.assertEqual(delta, 1)

        self.assertIsNotNone(self.store.try_claim(submission.id, 631551040))
        updated, delta = self.store.finish_decision(
            submission_id=submission.id,
            moderator_id=631551040,
            destination="main",
            award=5,
            publication_chat_id=-1002,
            publication_message_id=11,
        )
        self.assertEqual(delta, 4)
        self.assertEqual(updated.karma_award, 5)
        self.assertEqual(self.store.karma_total(123), 5)
        self.assertEqual(self.store.karma_summary(123), (5, 2))
        self.assertEqual(self.store.karma_users(), [123])

    def test_author_message_outbox(self) -> None:
        submission, _created = self.store.create_submission(
            job_number="1",
            user_id=123,
            chat_id=123,
            author_name="Тест",
            author_username=None,
            video_path=Path("video.mp4"),
            output_filename="video.mp4",
        )
        self.store.enqueue_author_message(submission.id, 631551040, "Поправь начало")
        pending = self.store.pending_author_messages()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["job_number"], "1")
        self.store.finish_author_message(pending[0]["id"])
        self.assertEqual(self.store.pending_author_messages(), [])


class ProposalUiTests(unittest.TestCase):
    def test_author_caption_is_clickable_and_escaped(self) -> None:
        submission = Submission(
            id=1,
            job_number="2",
            user_id=123,
            chat_id=123,
            author_name="макщ <3",
            author_username="hboloid",
            video_path="video.mp4",
            output_filename="video.mp4",
            status="pending",
            destination=None,
            karma_award=0,
            publication_chat_id=None,
            publication_message_id=None,
            created_at=0,
            updated_at=0,
        )
        caption = _author_caption(submission)
        self.assertEqual(caption, 'Прислал <a href="https://t.me/hboloid">макщ &lt;3</a>')

    def test_moderation_buttons_have_expected_order(self) -> None:
        keyboard = _moderation_keyboard(7).inline_keyboard
        labels = [[button.text for button in row] for row in keyboard]
        self.assertEqual(
            labels,
            [
                ["В La La School", "В Ghien Mi Go"],
                ["Передать сообщение"],
                ["Не публиковать"],
            ],
        )

    def test_karma_tag_fits_telegram_limit(self) -> None:
        self.assertEqual(_karma_tag(6), "Карма: 6")
        self.assertLessEqual(len(_karma_tag(12345678901234567890)), 16)


if __name__ == "__main__":
    unittest.main()
