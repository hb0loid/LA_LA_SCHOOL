from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laladub.bot import _proposal_keyboard
from laladub.proposal_bot import _publication_caption
from laladub.proposal_store import ProposalStore


class ProposalCommentTests(unittest.TestCase):
    def test_comment_survives_submission_store_and_is_published_before_author(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            video = root / "dubbed.mp4"
            video.write_bytes(b"video")
            store = ProposalStore(root / "proposal.sqlite3")
            submission, created = store.create_submission(
                job_number="42",
                user_id=123,
                chat_id=123,
                author_name="Автор",
                author_username="author",
                video_path=video,
                output_filename="dubbed.mp4",
                author_comment="Смешной <момент> & финал",
            )

            self.assertTrue(created)
            self.assertEqual(submission.author_comment, "Смешной <момент> & финал")
            caption = _publication_caption(submission)
            self.assertTrue(caption.startswith("Смешной &lt;момент&gt; &amp; финал\n\n"))
            self.assertIn("Прислал", caption)

    def test_finished_job_keyboard_has_comment_then_submit(self) -> None:
        keyboard = _proposal_keyboard("42").inline_keyboard
        self.assertEqual([row[0].text for row in keyboard], ["Добавить комментарий", "Отправить в предложку"])
        self.assertEqual(keyboard[0][0].callback_data, "proposal:comment:42")
        self.assertEqual(keyboard[1][0].callback_data, "proposal:submit:42")


if __name__ == "__main__":
    unittest.main()
