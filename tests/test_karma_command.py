from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from laladub.bot import karma_command
from laladub.proposal_store import ProposalStore


class _Message:
    def __init__(self) -> None:
        self.reply_text = AsyncMock()


def _update(user_id: int) -> SimpleNamespace:
    message = _Message()
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_message=message,
    )


class KarmaCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.store = ProposalStore(Path(self._tempdir.name) / "proposals.sqlite3")

    def _context(self, args: list[str] | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={"proposal_store": self.store}),
            args=args or [],
        )

    def _award(self, user_id: int, name: str, karma: int) -> None:
        # Karma is only ever written as a side effect of finish_decision, which
        # needs a whole publish to run; the rows themselves are what matters
        # here, so they go in directly.
        import sqlite3

        submission, _created = self.store.create_submission(
            job_number=str(user_id), user_id=user_id, chat_id=user_id, author_name=name,
            author_username=None, video_path=Path("v.mp4"), output_filename="v.mp4",
        )
        connection = sqlite3.connect(self.store.path)
        try:
            connection.execute(
                "INSERT INTO karma_events (submission_id, user_id, delta, old_award,"
                " new_award, reason, moderator_id, created_at)"
                " VALUES (?, ?, ?, 0, ?, 'test', 1, 0)",
                (submission.id, user_id, karma, karma),
            )
            connection.commit()
        finally:
            connection.close()

    async def _say(self, user_id: int, args: list[str] | None = None) -> str:
        update = _update(user_id)
        await karma_command(update, self._context(args))
        return update.effective_message.reply_text.call_args.args[0]

    async def test_own_karma_is_reported(self) -> None:
        self._award(1, "Тест", 5000)
        said = await self._say(1)
        self.assertIn("Твоя карма", said)

    async def test_own_karma_points_at_the_leaderboard(self) -> None:
        self._award(1, "Тест", 5000)
        self.assertIn("/karma all", await self._say(1))

    async def test_all_lists_everyone_best_first(self) -> None:
        self._award(1, "Первый", 9000)
        self._award(2, "Второй", 3000)
        said = await self._say(1, ["all"])
        self.assertLess(said.index("Первый"), said.index("Второй"))

    async def test_the_caller_is_marked_in_the_list(self) -> None:
        self._award(1, "Первый", 9000)
        self._award(2, "Второй", 3000)
        said = await self._say(2, ["all"])
        self.assertIn("Второй", said)
        self.assertIn("← ты", said)

    async def test_someone_outside_the_top_still_sees_their_own(self) -> None:
        for n in range(3, 30):
            self._award(n, f"Юзер{n}", n * 1000)
        self._award(1, "Скромный", 10)
        said = await self._say(1, ["all"])
        self.assertIn("Твоя карма", said)

    async def test_russian_all_is_accepted_too(self) -> None:
        self._award(1, "Тест", 5000)
        self.assertIn("Таблица лидеров", await self._say(1, ["все"]))

    async def test_an_empty_leaderboard_says_so(self) -> None:
        self.assertIn("никто", await self._say(1, ["all"]))

    async def test_a_missing_store_is_reported_not_crashed(self) -> None:
        update = _update(1)
        context = SimpleNamespace(application=SimpleNamespace(bot_data={}), args=[])
        await karma_command(update, context)
        self.assertIn("недоступна", update.effective_message.reply_text.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
