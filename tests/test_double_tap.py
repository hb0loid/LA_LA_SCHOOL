"""A second tap on the same button must not be reported as an error."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from laladub import bot


class DoubleTapTests(unittest.TestCase):
    def _run(self, error: Exception) -> mock.AsyncMock:
        recorded = mock.AsyncMock()
        query = SimpleNamespace(data="src:auto")
        update = SimpleNamespace(
            callback_query=query, effective_user=None, effective_chat=None, effective_message=None
        )
        context = SimpleNamespace(error=error)
        with mock.patch.object(bot, "_record_error", recorded):
            asyncio.run(bot._telegram_error_handler(update, context))
        return recorded

    def test_identical_edit_is_not_an_error(self) -> None:
        error = RuntimeError(
            "Message is not modified: specified new message content and reply markup "
            "are exactly the same as a current content and reply markup of the message"
        )
        self.assertFalse(self._run(error).await_count)

    def test_other_errors_are_still_recorded(self) -> None:
        recorded = self._run(RuntimeError("something else broke"))
        self.assertEqual(recorded.await_count, 1)
        self.assertEqual(recorded.await_args.kwargs["stage"], "Кнопка src:auto")


if __name__ == "__main__":
    unittest.main()
