from __future__ import annotations

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "laladub"


def _handler_names(source: str) -> set[str]:
    return set(re.findall(r'CommandHandler\("([a-z_]+)"', source))


def _menu_names(source: str) -> set[str]:
    return set(re.findall(r'^\s*\("([a-z_]+)", "', source, re.MULTILINE))


class CommandMenuTests(unittest.TestCase):
    """A command nobody can see is a command only its author can use. Every
    handler belongs in a menu - the admin ones in the admins' own menu."""

    def test_every_bot_command_is_listed_somewhere(self) -> None:
        source = (SRC / "bot.py").read_text(encoding="utf-8")
        missing = _handler_names(source) - _menu_names(source)
        self.assertEqual(missing, set(), f"нет в меню: {sorted(missing)}")

    def test_every_proposal_command_is_listed(self) -> None:
        source = (SRC / "proposal_bot.py").read_text(encoding="utf-8")
        missing = _handler_names(source) - _menu_names(source)
        self.assertEqual(missing, set(), f"нет в меню: {sorted(missing)}")

    def test_admin_commands_are_scoped_to_admins(self) -> None:
        """Listing them to everyone would be worse than hiding them."""
        source = (SRC / "bot.py").read_text(encoding="utf-8")
        self.assertIn("BotCommandScopeChat", source)
        public = source.split("admin_only = [")[0]
        for name in ("grant_premium", "maintenance", "starbalance"):
            self.assertNotIn(f'("{name}", "', public, f"{name} попал в общее меню")


if __name__ == "__main__":
    unittest.main()
