from __future__ import annotations

import asyncio
import contextlib
import html
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ffmpeg import probe_duration
from .karma import format_karma_milli, karma_milli_for_duration, level_for_karma, visible_karma
from .proposal_store import ProposalStore, Submission


@dataclass(frozen=True, slots=True)
class ProposalBotSettings:
    token: str
    database: Path
    moderator_ids: frozenset[int]
    main_channel: str
    shame_channel: str
    karma_chat: str


def load_settings() -> ProposalBotSettings:
    token = os.environ.get("LALADUB_PROPOSAL_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set LALADUB_PROPOSAL_BOT_TOKEN")
    moderator_ids = frozenset(_parse_ids(os.environ.get("LALADUB_PROPOSAL_MODERATORS", "")))
    if not moderator_ids:
        raise RuntimeError("Set LALADUB_PROPOSAL_MODERATORS")
    return ProposalBotSettings(
        token=token,
        database=Path(os.environ.get("LALADUB_PROPOSAL_DB", "runs/proposal/proposals.sqlite3")),
        moderator_ids=moderator_ids,
        main_channel=os.environ.get("LALADUB_PROPOSAL_MAIN_CHANNEL", "@elevenlabss").strip() or "@elevenlabss",
        shame_channel=os.environ.get("LALADUB_PROPOSAL_SHAME_CHANNEL", "@ghienmigo").strip() or "@ghienmigo",
        karma_chat=os.environ.get("LALADUB_PROPOSAL_KARMA_CHAT", "@lalaschoo").strip() or "@lalaschoo",
    )


def main() -> None:
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CallbackQueryHandler,
            ChatMemberHandler,
            CommandHandler,
            MessageHandler,
            filters,
        )
    except ImportError as exc:
        raise RuntimeError("Install bot dependencies first: python -m pip install -e .[bot]") from exc

    settings = load_settings()
    store = ProposalStore(settings.database)
    print(
        "La La School Proposal Bot starting "
        f"database={settings.database} moderators={len(settings.moderator_ids)} "
        f"channels={settings.main_channel},{settings.shame_channel} karma_chat={settings.karma_chat}",
        flush=True,
    )
    application = Application.builder().token(settings.token).post_init(_post_init).build()
    application.bot_data["settings"] = settings
    application.bot_data["store"] = store
    private_chat = filters.ChatType.PRIVATE
    application.add_error_handler(_error_handler)
    application.add_handler(CommandHandler("start", start, filters=private_chat))
    application.add_handler(CommandHandler("pending", pending, filters=private_chat))
    application.add_handler(CommandHandler("cancel", cancel, filters=private_chat))
    application.add_handler(CallbackQueryHandler(moderation_callback, pattern=r"^mod:"))
    application.add_handler(ChatMemberHandler(karma_member_changed, ChatMemberHandler.CHAT_MEMBER))
    application.add_handler(MessageHandler(private_chat & filters.TEXT & ~filters.COMMAND, relay_message))
    application.add_handler(MessageHandler(~private_chat & filters.VIDEO, comment_on_channel_forward))
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=False)


async def _post_init(application: Any) -> None:
    settings: ProposalBotSettings = application.bot_data["settings"]
    karma_chat = await application.bot.get_chat(settings.karma_chat)
    application.bot_data["karma_chat_id"] = int(karma_chat.id)
    commands = [
        ("start", "Открыть предложку"),
        ("pending", "Проверить новые видео"),
        ("cancel", "Отменить ввод сообщения"),
    ]
    await application.bot.set_my_commands(commands)
    asyncio.create_task(_delivery_loop(application))
    asyncio.create_task(_sync_all_karma_tags(application))


async def _error_handler(update: object, context: Any) -> None:
    print("Proposal bot error:\n" + "".join(traceback.format_exception(context.error)), flush=True)


def _is_moderator(settings: ProposalBotSettings, user_id: int | None) -> bool:
    return user_id is not None and user_id in settings.moderator_ids


async def start(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    user_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, user_id):
        await update.effective_message.reply_text("Этот бот предназначен только для модераторов предложки.")
        return
    await update.effective_message.reply_text(
        "Предложка La La School запущена. Новые видео будут появляться здесь автоматически."
    )
    await _deliver_for_moderator(context.application, int(user_id))


async def pending(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    user_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, user_id):
        await update.effective_message.reply_text("Нет доступа.")
        return
    delivered = await _deliver_for_moderator(context.application, int(user_id))
    if delivered == 0:
        await update.effective_message.reply_text("Новых видео пока нет.")


async def cancel(update: Any, context: Any) -> None:
    context.user_data.pop("relay_submission_id", None)
    await update.effective_message.reply_text("Ввод сообщения отменён.")


async def _delivery_loop(application: Any) -> None:
    settings: ProposalBotSettings = application.bot_data["settings"]
    while True:
        for moderator_id in settings.moderator_ids:
            try:
                await _deliver_for_moderator(application, moderator_id)
            except Exception as exc:
                print(
                    f"Proposal delivery to moderator {moderator_id} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        await asyncio.sleep(5.0)


async def _deliver_for_moderator(application: Any, moderator_id: int) -> int:
    store: ProposalStore = application.bot_data["store"]
    submissions = await asyncio.to_thread(store.pending_for_moderator, moderator_id, limit=10)
    delivered = 0
    for submission in submissions:
        try:
            message = await _send_moderation_video(application.bot, moderator_id, submission)
        except Exception as exc:
            print(
                f"Could not deliver submission {submission.id} to moderator {moderator_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        await asyncio.to_thread(
            store.record_moderator_message,
            submission.id,
            moderator_id,
            int(message.chat_id),
            int(message.message_id),
        )
        delivered += 1
    return delivered


async def _send_moderation_video(bot: Any, moderator_id: int, submission: Submission) -> Any:
    from .bot import _telegram_sendable_video_path, video_upload_metadata

    video_path = Path(submission.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    send_path = await _telegram_sendable_video_path(video_path)
    metadata = await video_upload_metadata(send_path)
    with send_path.open("rb") as file_obj:
        return await bot.send_video(
            chat_id=moderator_id,
            video=file_obj,
            filename=submission.output_filename,
            caption=_moderation_caption(submission),
            parse_mode="HTML",
            reply_markup=_moderation_keyboard(submission.id),
            supports_streaming=True,
            read_timeout=300,
            write_timeout=300,
            connect_timeout=60,
            pool_timeout=60,
            **metadata,
        )


async def moderation_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    moderator_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, moderator_id):
        await query.answer("Нет доступа.", show_alert=True)
        return

    parts = str(query.data or "").split(":")
    if len(parts) != 3 or not parts[2].isdigit():
        await query.answer("Некорректная кнопка.", show_alert=True)
        return
    action = parts[1]
    submission_id = int(parts[2])
    submission = await asyncio.to_thread(store.get_submission, submission_id)
    if submission is None:
        await query.answer("Заявка не найдена.", show_alert=True)
        return

    if action == "message":
        context.user_data["relay_submission_id"] = submission_id
        await query.answer()
        from telegram import ForceReply

        await query.message.reply_text(
            f"Напиши сообщение автору работы №{submission.job_number}. Я передам его через основной бот.",
            reply_markup=ForceReply(selective=True),
        )
        return

    if action == "subtitles":
        subtitles = _find_submission_subtitles(submission)
        if subtitles is None:
            await query.answer("Субтитры этой работы не найдены.", show_alert=True)
            return
        await query.answer("Отправляю субтитры…")
        try:
            with subtitles.open("rb") as document:
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=document,
                    filename=f"work_{submission.job_number}_subtitles{subtitles.suffix}",
                    caption=f"Субтитры работы №{submission.job_number}",
                )
        except Exception as exc:
            await query.message.reply_text(f"Не удалось отправить субтитры: {type(exc).__name__}: {exc}")
        return

    decisions = {
        "main": ("main", settings.main_channel),
        "shame": ("shame", settings.shame_channel),
        "reject": ("rejected", None),
    }
    decision = decisions.get(action)
    if decision is None:
        await query.answer("Неизвестное действие.", show_alert=True)
        return
    destination, target_chat = decision
    if submission.destination == destination and (
        destination == "rejected" or submission.publication_message_id is not None
    ):
        await query.answer("Это решение уже применено.")
        return

    claimed = await asyncio.to_thread(store.try_claim, submission_id, int(moderator_id))
    if claimed is None:
        await query.answer("Эту заявку сейчас обрабатывает другой модератор.", show_alert=True)
        return

    await query.answer("Публикую…" if target_chat else "Отмечаю…")
    new_message = None
    karma_before = await asyncio.to_thread(store.karma_total, claimed.user_id)
    try:
        duration_ms = claimed.duration_ms
        if duration_ms <= 0 and Path(claimed.video_path).is_file():
            duration_ms = max(0, round(await asyncio.to_thread(probe_duration, Path(claimed.video_path)) * 1000))
            claimed = await asyncio.to_thread(store.set_submission_duration, claimed.id, duration_ms)
        award_milli = karma_milli_for_duration(duration_ms, destination)
        if target_chat is not None:
            new_message = await _publish_to_channel(context.bot, target_chat, claimed)
        updated, delta = await asyncio.to_thread(
            store.finish_decision,
            submission_id=submission_id,
            moderator_id=int(moderator_id),
            destination=destination,
            award_milli=award_milli,
            publication_chat_id=int(new_message.chat_id) if new_message is not None else None,
            publication_message_id=int(new_message.message_id) if new_message is not None else None,
        )
    except Exception as exc:
        if new_message is not None:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(new_message.chat_id, new_message.message_id)
        await asyncio.to_thread(store.release_claim, submission_id, int(moderator_id))
        await query.message.reply_text(f"Не удалось применить решение: {type(exc).__name__}: {exc}")
        return

    if (
        claimed.publication_chat_id is not None
        and claimed.publication_message_id is not None
        and (
            claimed.publication_chat_id != updated.publication_chat_id
            or claimed.publication_message_id != updated.publication_message_id
        )
    ):
        with contextlib.suppress(Exception):
            await context.bot.delete_message(claimed.publication_chat_id, claimed.publication_message_id)

    karma_after = await asyncio.to_thread(store.karma_total, updated.user_id)
    level_message = _level_up_message(karma_before, karma_after)
    if level_message:
        try:
            await asyncio.to_thread(
                store.enqueue_author_message,
                updated.id,
                int(moderator_id),
                level_message,
            )
        except Exception as exc:
            print(
                f"Could not enqueue level-up notice user={updated.user_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )

    tag_updated = await _sync_karma_tag(context.bot, settings, store, updated.user_id)
    await _refresh_moderator_messages(context.bot, store, updated)
    delta_text = f"Карма: {format_karma_milli(delta, signed=True)}" if delta else "Карма без изменений"
    tag_text = " Тег в группе обновлён." if tag_updated else ""
    await query.message.reply_text(f"Решение применено. {delta_text}.{tag_text}")


async def karma_member_changed(update: Any, context: Any) -> None:
    event = update.chat_member
    if event is None or int(event.chat.id) != context.application.bot_data.get("karma_chat_id"):
        return
    member = event.new_chat_member
    user = member.user
    if user.is_bot or str(member.status) != "member":
        return
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    await _sync_karma_tag(context.bot, settings, store, int(user.id))


async def _sync_all_karma_tags(application: Any) -> None:
    settings: ProposalBotSettings = application.bot_data["settings"]
    store: ProposalStore = application.bot_data["store"]
    user_ids = await asyncio.to_thread(store.karma_users)
    for user_id in user_ids:
        await _sync_karma_tag(application.bot, settings, store, user_id)


async def _sync_karma_tag(
    bot: Any,
    settings: ProposalBotSettings,
    store: ProposalStore,
    user_id: int,
) -> bool:
    total_milli, event_count = await asyncio.to_thread(store.karma_summary, user_id)
    if event_count <= 0:
        return False
    try:
        await bot.set_chat_member_tag(
            chat_id=settings.karma_chat,
            user_id=user_id,
            tag=_karma_tag(total_milli),
        )
        print(f"Karma tag updated user={user_id} total_milli={total_milli}", flush=True)
        return True
    except Exception as exc:
        print(
            f"Karma tag update skipped user={user_id} total_milli={total_milli}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False


def _karma_tag(total_milli: int) -> str:
    total = visible_karma(total_milli)
    primary = f"Карма: {total}"
    if len(primary) <= 16:
        return primary
    return f"К: {int(total)}"[:16]


def _level_up_message(before_milli: int, after_milli: int) -> str | None:
    old_level = level_for_karma(before_milli)
    new_level = level_for_karma(after_milli)
    if new_level.minimum <= old_level.minimum:
        return None
    priority = "обычный" if new_level.priority_bonus == 0 else f"+{new_level.priority_bonus}"
    return (
        "🎉 Новый уровень кармы!\n\n"
        f"Теперь твой уровень — {new_level.name}.\n"
        f"Доступно: {new_level.daily_minutes} минут перевода в сутки.\n"
        f"Приоритет в очереди: {priority}.\n"
        f"Лимит задач в очереди: {new_level.queue_limit}.\n"
        "Лимита на длину одного видео нет — учитывается общий суточный лимит."
    )


async def _publish_to_channel(bot: Any, target_chat: str, submission: Submission) -> Any:
    from .bot import _telegram_sendable_video_path, video_upload_metadata

    video_path = Path(submission.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    send_path = await _telegram_sendable_video_path(video_path)
    metadata = await video_upload_metadata(send_path)
    with send_path.open("rb") as file_obj:
        return await bot.send_video(
            chat_id=target_chat,
            video=file_obj,
            filename=submission.output_filename,
            caption=_author_caption(submission),
            parse_mode="HTML",
            supports_streaming=True,
            read_timeout=300,
            write_timeout=300,
            connect_timeout=60,
            pool_timeout=60,
            **metadata,
        )


async def _refresh_moderator_messages(bot: Any, store: ProposalStore, submission: Submission) -> None:
    messages = await asyncio.to_thread(store.moderator_messages, submission.id)
    for chat_id, message_id in messages:
        with contextlib.suppress(Exception):
            await bot.edit_message_caption(
                chat_id=chat_id,
                message_id=message_id,
                caption=_moderation_caption(submission),
                parse_mode="HTML",
                reply_markup=(
                    _moderation_keyboard(submission.id)
                    if submission.status == "pending"
                    else None
                ),
            )


async def relay_message(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    moderator_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, moderator_id):
        return
    submission_id = context.user_data.pop("relay_submission_id", None)
    if submission_id is None:
        return
    text = str(update.effective_message.text or "").strip()
    if not text:
        await update.effective_message.reply_text("Пустое сообщение не отправлено.")
        return
    submission = await asyncio.to_thread(store.get_submission, int(submission_id))
    if submission is None:
        await update.effective_message.reply_text("Заявка уже не найдена.")
        return
    await asyncio.to_thread(store.enqueue_author_message, submission.id, int(moderator_id), text)
    await update.effective_message.reply_text(
        f"Сообщение для автора работы №{submission.job_number} поставлено на отправку."
    )


def _moderation_keyboard(submission_id: int) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("В La La School", callback_data=f"mod:main:{submission_id}"),
                InlineKeyboardButton("В Ghien Mi Go", callback_data=f"mod:shame:{submission_id}"),
            ],
            [InlineKeyboardButton("Передать сообщение", callback_data=f"mod:message:{submission_id}")],
            [InlineKeyboardButton("Посмотреть субтитры", callback_data=f"mod:subtitles:{submission_id}")],
        ]
    )


def _moderation_caption(submission: Submission) -> str:
    status_labels = {
        "pending": "Ожидает решения",
        "published": "✅ Опубликовано",
        "rejected": "✅ Не опубликовано",
    }
    destination_labels = {
        "main": "La La School",
        "shame": "Ghien Mi Go",
        "rejected": "Не публиковать",
    }
    lines = [
        f"<b>Предложка №{submission.id}</b>",
        f"Работа №{html.escape(submission.job_number)}",
        _author_caption(submission),
        f"Карма на момент отправки: {visible_karma(submission.karma_before_milli)}",
        f"Статус: {status_labels.get(submission.status, html.escape(submission.status))}",
    ]
    if submission.destination:
        lines.append(f"Решение: {destination_labels.get(submission.destination, html.escape(submission.destination))}")
        lines.append(f"Карма за работу: {format_karma_milli(submission.karma_milli, signed=True)}")
    return "\n".join(lines)


def _author_caption(submission: Submission) -> str:
    name = html.escape(submission.author_name.strip() or str(submission.user_id))
    if submission.author_username:
        href = f"https://t.me/{html.escape(submission.author_username.lstrip('@'), quote=True)}"
    else:
        href = f"tg://user?id={submission.user_id}"
    return f'Прислал <a href="{href}">{name}</a>'


def _find_submission_subtitles(submission: Submission) -> Path | None:
    job_dir = Path(submission.video_path).parent
    candidates = [
        *sorted(job_dir.glob("*_transcript_lalaschool.txt")),
        job_dir / "work" / "translated.srt",
    ]
    work_dir = job_dir / "work"
    if work_dir.is_dir():
        candidates.extend(sorted(work_dir.glob("input_*.srt")))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _transcript_text_for_submission(submission: Submission) -> str | None:
    path = _find_submission_subtitles(submission)
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() != ".srt":
        text = raw.strip()
        return text or None
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        lines.append(stripped)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return text or None


async def comment_on_channel_forward(update: Any, context: Any) -> None:
    """Telegram auto-forwards every channel post into its linked Discussion Group as a
    new message; this catches that forward and replies to it with the transcript,
    which is how a bot "comments" on a channel post - there's no dedicated API for it."""
    message = update.effective_message
    if message is None or not message.is_automatic_forward:
        return
    origin = getattr(message, "forward_origin", None)
    origin_chat = getattr(origin, "chat", None)
    if origin_chat is None:
        return
    store: ProposalStore = context.application.bot_data["store"]
    submission = await asyncio.to_thread(
        store.find_by_publication, int(origin_chat.id), int(origin.message_id)
    )
    if submission is None:
        return
    text = _transcript_text_for_submission(submission)
    if not text:
        return
    if not await asyncio.to_thread(store.mark_comment_posted, submission.id):
        return
    if len(text) > 4000:
        text = text[:4000] + "…"
    try:
        await context.bot.send_message(chat_id=message.chat_id, text=text, reply_to_message_id=message.message_id)
    except Exception as exc:
        print(f"Comment post failed for submission {submission.id}: {type(exc).__name__}: {exc}", flush=True)


def _parse_ids(raw: str) -> set[int]:
    values: set[int] = set()
    for token in raw.replace(";", ",").replace("\n", ",").split(","):
        token = token.strip().replace("\u00a0", "").replace(" ", "")
        if token.isdigit():
            values.add(int(token))
    return values


if __name__ == "__main__":
    main()
