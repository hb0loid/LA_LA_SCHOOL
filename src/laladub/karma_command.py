"""The /karma command, shared by both bots.

Kept out of bot.py so the proposal bot can register it too: importing bot.py
pulls in the whole ML pipeline, which that process has no reason to load.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .karma import KARMA_SCALE, level_for_karma, next_level_for_karma, visible_karma
from .proposal_store import ProposalStore

KARMA_LEADERBOARD_SIZE = 20
# The leaderboard is a chunky message and the command works in groups, so one
# person cannot make the bot post it over and over.
KARMA_COOLDOWN_SECONDS = 15.0
_last_karma_call: dict[int, float] = {}

_ALL_WORDS = {"all", "все", "всё", "весь", "топ", "top"}


def _store(context: Any) -> ProposalStore | None:
    data = context.application.bot_data
    # The two bots file the same store under different keys.
    return data.get("proposal_store") or data.get("store")


def _shorten_name(name: str) -> str:
    # Display names carry emoji and decoration; the list stays readable when
    # every row is about the same width.
    cleaned = " ".join(str(name).split())
    return cleaned if len(cleaned) <= 24 else cleaned[:23] + "…"


async def karma_command(update: Any, context: Any) -> None:
    """/karma shows your own karma, /karma all the leaderboard."""
    user = update.effective_user
    message = update.effective_message
    if user is None or message is None:
        return
    store = _store(context)
    if store is None:
        await message.reply_text("Карма сейчас недоступна.")
        return

    user_id = int(user.id)
    now = time.time()
    elapsed = now - _last_karma_call.get(user_id, 0.0)
    if elapsed < KARMA_COOLDOWN_SECONDS:
        await message.reply_text(f"Слишком часто — подожди ещё {round(KARMA_COOLDOWN_SECONDS - elapsed)} сек.")
        return
    _last_karma_call[user_id] = now

    args = getattr(context, "args", None) or []
    wants_all = bool(args) and str(args[0]).strip().casefold() in _ALL_WORDS
    if not wants_all:
        karma_milli = await asyncio.to_thread(store.karma_total, user_id)
        level = level_for_karma(karma_milli)
        lines = [f"⭐ Твоя карма: {visible_karma(karma_milli)}", f"Уровень: {level.name}"]
        next_level = next_level_for_karma(karma_milli)
        if next_level is None:
            lines.append("Достигнут максимальный уровень.")
        else:
            remaining = max(0, next_level.minimum * KARMA_SCALE - karma_milli)
            lines.append(f"До уровня «{next_level.name}»: {(remaining + KARMA_SCALE - 1) // KARMA_SCALE}")
        lines.append("")
        lines.append("Таблица лидеров: /karma all")
        await message.reply_text("\n".join(lines))
        return

    rows = await asyncio.to_thread(store.karma_leaderboard, KARMA_LEADERBOARD_SIZE)
    if not rows:
        await message.reply_text("Пока никто не заработал карму.")
        return

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lines = ["🏆 Таблица лидеров по карме", ""]
    for place, (row_user_id, total_milli, name) in enumerate(rows, start=1):
        marker = medals.get(place, f"{place}.")
        mine = " ← ты" if row_user_id == user_id else ""
        lines.append(f"{marker} {_shorten_name(name)} — {visible_karma(total_milli)}{mine}")

    if all(row_user_id != user_id for row_user_id, _total, _name in rows):
        # Being outside the top is the interesting case, so it gets its own line
        # rather than leaving the reader to wonder where they stand.
        own = await asyncio.to_thread(store.karma_total, user_id)
        lines.extend(["", f"Твоя карма: {visible_karma(own)}"])
    await message.reply_text("\n".join(lines))
