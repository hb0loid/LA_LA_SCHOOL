from __future__ import annotations

import asyncio
import contextlib
import html
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datetime import datetime

from .ffmpeg import probe_duration
from .karma import format_karma_milli, karma_milli_for_duration, level_for_karma, visible_karma
from .karma_command import karma_command
from .library import LibraryStore, show_command
from .update_dedupe import (
    UpdateDeduplicator,
    build_completion_marker,
    build_replay_guard,
)
from .proposal_store import ProposalStore, ScheduledPost, Submission

# How far apart consecutive posts to the same channel are spaced when /timer
# delayed posting is on.
DELAYED_POST_INTERVAL_SECONDS = 30 * 60


@dataclass(frozen=True, slots=True)
class ProposalBotSettings:
    token: str
    database: Path
    moderator_ids: frozenset[int]
    main_channel: str
    shame_channel: str
    karma_chat: str
    library_db: Path = Path("runs/library/library.sqlite3")
    library_dir: Path = Path("runs/library/videos")


class _ApplicationContext:
    def __init__(self, application: Any) -> None:
        self.application = application
        self.bot = application.bot


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
        library_db=Path(os.environ.get("LALADUB_LIBRARY_DB", "runs/library/library.sqlite3")),
        library_dir=Path(os.environ.get("LALADUB_LIBRARY_DIR", "runs/library/videos")),
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
            TypeHandler,
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
    application.bot_data["library_store"] = LibraryStore(settings.library_db)
    private_chat = filters.ChatType.PRIVATE
    application.add_error_handler(_error_handler)
    # Before every other handler: an update Telegram redelivered after a hard
    # restart must not publish or post a second time.
    update_dedupe = UpdateDeduplicator(settings.database.parent / "last_update.json")
    application.add_handler(
        TypeHandler(Update, build_replay_guard(update_dedupe)),
        group=-100,
    )
    application.add_handler(CommandHandler("start", start, filters=private_chat))
    application.add_handler(CommandHandler("pending", pending, filters=private_chat))
    application.add_handler(CommandHandler("cancel", cancel, filters=private_chat))
    application.add_handler(CommandHandler("timer", timer_command, filters=private_chat))
    application.add_handler(CommandHandler("scheduled", scheduled_command, filters=private_chat))
    application.add_handler(CommandHandler("post", post_command, filters=private_chat))
    application.add_handler(CommandHandler("unpost", unpost_command, filters=private_chat))
    application.add_handler(CommandHandler("show", show_command))
    application.add_handler(CommandHandler("karma", karma_command))
    application.add_handler(CommandHandler("clean", clean_command, filters=private_chat))
    application.add_handler(CallbackQueryHandler(moderation_callback, pattern=r"^mod:"))
    application.add_handler(ChatMemberHandler(karma_member_changed, ChatMemberHandler.CHAT_MEMBER))
    application.add_handler(MessageHandler(private_chat & filters.TEXT & ~filters.COMMAND, relay_message))
    application.add_handler(MessageHandler(~private_chat & filters.VIDEO, comment_on_channel_forward))
    # Last group of all: reached only once every other handler has finished,
    # which is what lets the update count as handled and its replay be refused.
    application.add_handler(
        TypeHandler(Update, build_completion_marker(update_dedupe)),
        group=1000,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=False)


async def _post_init(application: Any) -> None:
    settings: ProposalBotSettings = application.bot_data["settings"]
    karma_chat = await application.bot.get_chat(settings.karma_chat)
    application.bot_data["karma_chat_id"] = int(karma_chat.id)
    commands = [
        ("start", "Открыть предложку"),
        ("pending", "Проверить новые видео"),
        ("cancel", "Отменить ввод сообщения"),
        ("timer", "Вкл/выкл отложенную публикацию"),
        ("scheduled", "Очередь отложенных постов"),
        ("post", "Отправить отложенный пост сейчас"),
        ("unpost", "Снять пост с отложенной публикации"),
        ("show", "Показать готовую работу из библиотеки"),
        ("karma", "Своя карма, /karma all — таблица лидеров"),
        ("clean", "Очистить чат от лишних сообщений"),
    ]
    await application.bot.set_my_commands(commands)
    asyncio.create_task(_delivery_loop(application))
    asyncio.create_task(_sync_all_karma_tags(application))
    asyncio.create_task(_scheduled_post_loop(application))


async def _error_handler(update: object, context: Any) -> None:
    print("Proposal bot error:\n" + "".join(traceback.format_exception(context.error)), flush=True)


def _is_moderator(settings: ProposalBotSettings, user_id: int | None) -> bool:
    return user_id is not None and user_id in settings.moderator_ids


async def _note(store: ProposalStore, moderator_id: int, message: Any) -> None:
    """Tracks a status/confirmation message the bot just sent a moderator, so
    /clean can find and remove it later without touching still-pending videos."""
    if message is None:
        return
    with contextlib.suppress(Exception):
        await asyncio.to_thread(store.record_bot_note, moderator_id, int(message.chat_id), int(message.message_id))


async def start(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    user_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, user_id):
        await update.effective_message.reply_text("Этот бот предназначен только для модераторов предложки.")
        return
    sent = await update.effective_message.reply_text(
        "Предложка La La School запущена. Новые видео будут появляться здесь автоматически."
    )
    await _note(store, int(user_id), sent)
    await _deliver_for_moderator(context.application, int(user_id))


async def pending(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    user_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, user_id):
        await update.effective_message.reply_text("Нет доступа.")
        return
    delivered = await _deliver_for_moderator(context.application, int(user_id))
    if delivered == 0:
        sent = await update.effective_message.reply_text("Новых видео пока нет.")
        await _note(store, int(user_id), sent)


async def cancel(update: Any, context: Any) -> None:
    context.user_data.pop("relay_submission_id", None)
    store: ProposalStore = context.application.bot_data["store"]
    user_id = getattr(update.effective_user, "id", None)
    sent = await update.effective_message.reply_text("Ввод сообщения отменён.")
    if user_id is not None:
        await _note(store, int(user_id), sent)


async def timer_command(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    moderator_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, moderator_id):
        return

    args = context.args or []
    action = str(args[0]).strip().lower() if args else ""
    current = await asyncio.to_thread(store.delayed_posting_enabled)
    if action in {"on", "off"}:
        enabled = action == "on"
    elif action:
        sent = await update.effective_message.reply_text("Использование: /timer, /timer on или /timer off.")
        await _note(store, int(moderator_id), sent)
        return
    else:
        enabled = not current

    if enabled != current:
        await asyncio.to_thread(store.set_delayed_posting_enabled, enabled)
    state = "включена" if enabled else "выключена"
    suffix = (
        f" Посты в один канал теперь выходят не чаще раза в {DELAYED_POST_INTERVAL_SECONDS // 60} мин."
        if enabled
        else ""
    )
    sent = await update.effective_message.reply_text(f"Отложенная публикация теперь {state}.{suffix}")
    await _note(store, int(moderator_id), sent)


async def scheduled_command(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    moderator_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, moderator_id):
        return

    items = await asyncio.to_thread(store.pending_scheduled_posts)
    if not items:
        sent = await update.effective_message.reply_text("Очередь отложенных постов пуста.")
        await _note(store, int(moderator_id), sent)
        return

    channel_names = {"main": "La La School", "shame": "Ghien Mi Go"}
    lines = ["Очередь отложенных постов:", ""]
    for item in items:
        submission = await asyncio.to_thread(store.get_submission, item.submission_id)
        label = f"№{submission.job_number}" if submission is not None else f"id{item.submission_id}"
        when = datetime.fromtimestamp(item.scheduled_for).strftime("%d.%m %H:%M")
        channel = channel_names.get(item.destination, item.destination)
        lines.append(f"Работа {label} → {channel}, отправка в {when}")
    lines.append("")
    lines.append("Отправить сейчас: /post номер\nОтменить: /unpost номер")
    sent = await update.effective_message.reply_text("\n".join(lines))
    await _note(store, int(moderator_id), sent)


async def post_command(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    moderator_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, moderator_id):
        return

    args = context.args or []
    job_number = str(args[0]).strip() if args else ""
    if not job_number:
        sent = await update.effective_message.reply_text("Использование: /post номер_работы")
        await _note(store, int(moderator_id), sent)
        return

    item = await asyncio.to_thread(store.find_pending_schedule_by_job_number, job_number)
    if item is None:
        sent = await update.effective_message.reply_text(f"Работа №{job_number} не найдена в очереди отложенных постов.")
        await _note(store, int(moderator_id), sent)
        return

    ok = await _process_scheduled_post(context, item)
    if ok:
        sent = await update.effective_message.reply_text(f"Работа №{job_number} отправлена.")
    else:
        sent = await update.effective_message.reply_text(f"Не удалось отправить работу №{job_number}, смотри лог.")
    await _note(store, int(moderator_id), sent)


async def unpost_command(update: Any, context: Any) -> None:
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    moderator_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, moderator_id):
        return

    args = context.args or []
    job_number = str(args[0]).strip() if args else ""
    if not job_number:
        sent = await update.effective_message.reply_text("Использование: /unpost номер_работы")
        await _note(store, int(moderator_id), sent)
        return

    item = await asyncio.to_thread(store.find_pending_schedule_by_job_number, job_number)
    if item is None:
        sent = await update.effective_message.reply_text(f"Работа №{job_number} не найдена в очереди отложенных постов.")
        await _note(store, int(moderator_id), sent)
        return

    cancelled = await asyncio.to_thread(store.cancel_scheduled_post, item.id)
    if cancelled is None:
        sent = await update.effective_message.reply_text(f"Работа №{job_number} уже отправлена или отменена.")
        await _note(store, int(moderator_id), sent)
        return

    sent = await update.effective_message.reply_text(f"Работа №{job_number} снята с отложенной публикации.")
    await _note(store, int(moderator_id), sent)
    # Give the decision buttons back on the moderator's copy of the post -
    # scheduling had hidden them, and the submission is unresolved again.
    submission = await asyncio.to_thread(store.get_submission, cancelled.submission_id)
    if submission is not None:
        await _refresh_moderator_messages(context.bot, store, submission)


async def clean_command(update: Any, context: Any) -> None:
    """Deletes every status/confirmation message the bot has sent this
    moderator, plus any video post that no longer needs a decision - the
    decision is already final, or it's sitting in the delayed queue (still
    reachable by job number via /post and /unpost). Only a video still
    showing live decision buttons is left alone. Only covers messages sent
    since this command shipped: there's no way to look up a chat's older
    history through the Bot API."""
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    moderator_id = getattr(update.effective_user, "id", None)
    if not _is_moderator(settings, moderator_id):
        return
    moderator_id = int(moderator_id)

    deleted = 0
    failed = 0

    notes = await asyncio.to_thread(store.take_bot_notes, moderator_id)
    for chat_id, message_id in notes:
        try:
            await context.bot.delete_message(chat_id, message_id)
            deleted += 1
        except Exception:
            failed += 1

    resolved = await asyncio.to_thread(store.resolved_moderator_video_messages, moderator_id)
    for submission_id, chat_id, message_id in resolved:
        try:
            await context.bot.delete_message(chat_id, message_id)
            deleted += 1
        except Exception:
            failed += 1
        await asyncio.to_thread(store.forget_moderator_message, submission_id, moderator_id)

    summary = f"Очищено сообщений: {deleted}."
    if failed:
        summary += f" Не удалось удалить: {failed}."
    sent = await update.effective_message.reply_text(summary)
    # Tracked for the *next* /clean, not this one - deleting your own receipt
    # immediately after showing it would just be confusing.
    await _note(store, moderator_id, sent)


async def send_due_scheduled_posts(context: Any) -> int:
    """Publishes whatever is due, keeping the channel spacing. Returns how many
    actually went out."""
    store: ProposalStore = context.application.bot_data["store"]
    due = await asyncio.to_thread(store.due_scheduled_posts)
    sent = 0
    for item in due:
        # Several posts come due at once whenever the bot was down for a while.
        # Publishing that backlog back to back is the very thing the delay
        # exists to prevent, so keep honouring the spacing and let the rest
        # wait for their turn.
        last_sent = await asyncio.to_thread(store.last_published_at, item.destination)
        if last_sent is not None and time.time() - last_sent < DELAYED_POST_INTERVAL_SECONDS:
            continue
        if await _process_scheduled_post(context, item):
            sent += 1
    return sent


async def _scheduled_post_loop(application: Any) -> None:
    context = _ApplicationContext(application)
    while True:
        try:
            await send_due_scheduled_posts(context)
        except Exception as exc:
            print(f"Scheduled post loop failed: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(20.0)


async def _process_scheduled_post(context: Any, item: ScheduledPost) -> bool:
    """Actually publishes one due scheduled post. Returns whether it succeeded -
    on failure the row stays 'pending' so the loop retries it on its next pass."""
    store: ProposalStore = context.application.bot_data["store"]
    submission = await asyncio.to_thread(store.get_submission, item.submission_id)
    if submission is None:
        await asyncio.to_thread(store.mark_scheduled_sent, item.id)
        return False

    claimed = await asyncio.to_thread(store.try_claim, item.submission_id, item.moderator_id)
    if claimed is None:
        # Someone is mid-decision on this submission right now - retry next tick.
        return False

    karma_before = await asyncio.to_thread(store.karma_total, claimed.user_id)
    try:
        updated, delta = await _finish_and_publish(
            context, claimed, item.moderator_id, item.destination, item.target_chat
        )
    except Exception as exc:
        await asyncio.to_thread(store.release_claim, item.submission_id, item.moderator_id)
        print(f"Scheduled post {item.id} (submission {item.submission_id}) failed: {type(exc).__name__}: {exc}", flush=True)
        return False

    await asyncio.to_thread(store.mark_scheduled_sent, item.id)
    effects_text = await _after_decision_effects(context, updated, karma_before, item.moderator_id, delta)
    print(f"Scheduled post published: job {updated.job_number}, {effects_text}", flush=True)
    return True


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
    transcript = await asyncio.to_thread(_transcript_text_for_submission, submission)
    with send_path.open("rb") as file_obj:
        return await bot.send_video(
            chat_id=moderator_id,
            video=file_obj,
            filename=submission.output_filename,
            caption=_moderation_caption(submission, transcript=transcript),
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

        sent = await query.message.reply_text(
            f"Напиши сообщение автору работы №{submission.job_number}. Я передам его через основной бот.",
            reply_markup=ForceReply(selective=True),
        )
        await _note(store, int(moderator_id), sent)
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

    existing_schedule = await asyncio.to_thread(store.pending_schedule_for_submission, submission_id)
    if existing_schedule is not None:
        when = datetime.fromtimestamp(existing_schedule.scheduled_for).strftime("%H:%M")
        await query.answer(f"Уже запланировано на {when}. Очередь: /scheduled.", show_alert=True)
        return

    if target_chat is not None and await asyncio.to_thread(store.delayed_posting_enabled):
        claimed = await asyncio.to_thread(store.try_claim, submission_id, int(moderator_id))
        if claimed is None:
            await query.answer("Эту заявку сейчас обрабатывает другой модератор.", show_alert=True)
            return
        # The claim only needs to guard this brief scheduling step, not the
        # up-to-30-minute wait for the slot - release it immediately so it
        # doesn't sit blocking other decisions until the 10-minute staleness
        # window expires.
        await asyncio.to_thread(store.release_claim, submission_id, int(moderator_id))
        slot = await asyncio.to_thread(
            store.next_available_slot, destination, interval_seconds=DELAYED_POST_INTERVAL_SECONDS
        )
        await asyncio.to_thread(
            store.schedule_post,
            submission_id=submission_id,
            destination=destination,
            target_chat=target_chat,
            moderator_id=int(moderator_id),
            scheduled_for=slot,
        )
        when = datetime.fromtimestamp(slot).strftime("%H:%M")
        with contextlib.suppress(Exception):
            await query.answer(f"Запланировано на {when}.", show_alert=True)
        await _refresh_moderator_messages(context.bot, store, submission, scheduled_for=slot)
        sent = await query.message.reply_text(
            f"Работа №{submission.job_number} запланирована на {when}. Очередь: /scheduled."
        )
        await _note(store, int(moderator_id), sent)
        return

    claimed = await asyncio.to_thread(store.try_claim, submission_id, int(moderator_id))
    if claimed is None:
        await query.answer("Эту заявку сейчас обрабатывает другой модератор.", show_alert=True)
        return

    # Answering is just UI feedback (clears the loading spinner) - if Telegram
    # rejects it (e.g. "Query is too old"), that must not abort the actual
    # publish and leave the claim stuck for the full staleness window.
    with contextlib.suppress(Exception):
        await query.answer("Публикую…" if target_chat else "Отмечаю…")
    karma_before = await asyncio.to_thread(store.karma_total, claimed.user_id)
    try:
        updated, delta = await _finish_and_publish(context, claimed, int(moderator_id), destination, target_chat)
    except Exception as exc:
        await asyncio.to_thread(store.release_claim, submission_id, int(moderator_id))
        sent = await query.message.reply_text(f"Не удалось применить решение: {type(exc).__name__}: {exc}")
        await _note(store, int(moderator_id), sent)
        return

    # The decision is already visible on the card itself, so the extra
    # "applied, karma +N" message was just another thing to scroll past and
    # later clean up. Everything it used to say now goes to the log.
    effects_text = await _after_decision_effects(context, updated, karma_before, int(moderator_id), delta)
    print(f"Decision applied: job {updated.job_number}, {effects_text}", flush=True)


async def _finish_and_publish(
    context: Any,
    claimed: Submission,
    moderator_id: int,
    destination: str,
    target_chat: str | None,
) -> tuple[Submission, int]:
    """Runs the actual publish + karma-award pipeline for a claimed submission.
    Shared by the immediate-decision path and the delayed-post sender so both
    apply the exact same effects at the moment a post actually goes out."""
    store: ProposalStore = context.application.bot_data["store"]
    duration_ms = claimed.duration_ms
    if duration_ms <= 0 and Path(claimed.video_path).is_file():
        duration_ms = max(0, round(await asyncio.to_thread(probe_duration, Path(claimed.video_path)) * 1000))
        claimed = await asyncio.to_thread(store.set_submission_duration, claimed.id, duration_ms)
    award_milli = karma_milli_for_duration(duration_ms, destination)
    new_message = None
    try:
        if target_chat is not None:
            new_message = await _publish_to_channel(context.bot, target_chat, claimed)
        updated, delta = await asyncio.to_thread(
            store.finish_decision,
            submission_id=claimed.id,
            moderator_id=moderator_id,
            destination=destination,
            award_milli=award_milli,
            publication_chat_id=int(new_message.chat_id) if new_message is not None else None,
            publication_message_id=int(new_message.message_id) if new_message is not None else None,
        )
    except Exception:
        if new_message is not None:
            with contextlib.suppress(Exception):
                await context.bot.delete_message(new_message.chat_id, new_message.message_id)
        raise

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

    return updated, delta


async def _after_decision_effects(
    context: Any, updated: Submission, karma_before: int, moderator_id: int, delta: int
) -> str:
    """Karma level-up notice, group karma tag, moderator message refresh - and
    a summary line for whoever should be told the decision went through."""
    settings: ProposalBotSettings = context.application.bot_data["settings"]
    store: ProposalStore = context.application.bot_data["store"]
    karma_after = await asyncio.to_thread(store.karma_total, updated.user_id)
    level_message = _level_up_message(karma_before, karma_after)
    if level_message:
        try:
            await asyncio.to_thread(store.enqueue_author_message, updated.id, moderator_id, level_message)
        except Exception as exc:
            print(
                f"Could not enqueue level-up notice user={updated.user_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )

    tag_updated = await _sync_karma_tag(context.bot, settings, store, updated.user_id)
    await _refresh_moderator_messages(context.bot, store, updated)
    delta_text = f"Карма: {format_karma_milli(delta, signed=True)}" if delta else "Карма без изменений"
    tag_text = " Тег в группе обновлён." if tag_updated else ""
    return f"{delta_text}.{tag_text}"


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


async def _refresh_moderator_messages(
    bot: Any, store: ProposalStore, submission: Submission, *, scheduled_for: float | None = None
) -> None:
    transcript = await asyncio.to_thread(_transcript_text_for_submission, submission)
    caption = _moderation_caption(submission, transcript=transcript)
    if scheduled_for is not None:
        when = datetime.fromtimestamp(scheduled_for).strftime("%H:%M")
        caption += f"\n⏳ Запланировано на {when}"
    messages = await asyncio.to_thread(store.moderator_messages, submission.id)
    for chat_id, message_id in messages:
        with contextlib.suppress(Exception):
            await bot.edit_message_caption(
                chat_id=chat_id,
                message_id=message_id,
                caption=caption,
                parse_mode="HTML",
                reply_markup=(
                    _moderation_keyboard(submission.id)
                    if submission.status == "pending" and scheduled_for is None
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
        sent = await update.effective_message.reply_text("Пустое сообщение не отправлено.")
        await _note(store, int(moderator_id), sent)
        return
    submission = await asyncio.to_thread(store.get_submission, int(submission_id))
    if submission is None:
        sent = await update.effective_message.reply_text("Заявка уже не найдена.")
        await _note(store, int(moderator_id), sent)
        return
    await asyncio.to_thread(store.enqueue_author_message, submission.id, int(moderator_id), text)
    sent = await update.effective_message.reply_text(
        f"Сообщение для автора работы №{submission.job_number} поставлено на отправку."
    )
    await _note(store, int(moderator_id), sent)


def _moderation_keyboard(submission_id: int) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("В La La School", callback_data=f"mod:main:{submission_id}"),
                InlineKeyboardButton("В Ghien Mi Go", callback_data=f"mod:shame:{submission_id}"),
            ],
            [InlineKeyboardButton("Передать сообщение", callback_data=f"mod:message:{submission_id}")],
        ]
    )


_MODERATION_CAPTION_LIMIT = 1024


def _moderation_caption(submission: Submission, *, transcript: str | None = None) -> str:
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
    text = "\n".join(lines)

    if transcript:
        # Subtitles are shown inline as a collapsed quote instead of behind a
        # separate button - the mod reads them before every decision anyway.
        # Video captions are hard-capped at 1024 chars, so long transcripts
        # are truncated to whatever's left after the rest of the caption.
        wrapper_overhead = len("\n\n<blockquote expandable></blockquote>")
        budget = _MODERATION_CAPTION_LIMIT - len(text) - wrapper_overhead
        if budget > 20:
            snippet = transcript if len(transcript) <= budget else transcript[: budget - 1].rstrip() + "…"
            text += f"\n\n<blockquote expandable>{html.escape(snippet)}</blockquote>"

    return text


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


def _find_submission_original_video(submission: Submission) -> Path | None:
    """The source video the dub was made from, as downloaded into the job folder."""
    job_dir = Path(submission.video_path).parent
    for candidate in sorted(job_dir.glob("input.*")):
        if (
            candidate.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}
            and candidate.is_file()
            and candidate.stat().st_size > 1024
        ):
            return candidate
    return None


async def _send_respecting_flood_limit(send: Any, *, attempts: int = 4) -> None:
    """Run a send, waiting out Telegram's rate limit instead of losing the message.

    A restart replays whatever forwards piled up, so several posts can want a
    comment at once and trip flood control. That is a "wait and retry" answer,
    not a failure - treating it as one silently dropped the comment for the post
    that happened to be behind the backlog.
    """
    from telegram.error import RetryAfter

    for attempt in range(attempts):
        try:
            await send()
            return
        except RetryAfter as exc:
            if attempt == attempts - 1:
                raise
            delay = float(getattr(exc, "retry_after", 5) or 5) + 1.0
            print(f"Comment send rate-limited, waiting {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)


async def _reply_with_original_video(context: Any, message: Any, submission: Submission) -> None:
    from .bot import _telegram_sendable_video_path, video_upload_metadata

    original = _find_submission_original_video(submission)
    if original is None:
        return
    send_path = await _telegram_sendable_video_path(original)
    metadata = await video_upload_metadata(send_path)

    async def send() -> None:
        # Reopened per attempt: a retry cannot replay an already-consumed handle.
        with send_path.open("rb") as file_obj:
            await context.bot.send_video(
                chat_id=message.chat_id,
                video=file_obj,
                filename=f"work_{submission.job_number}_original{send_path.suffix}",
                caption="Оригинал",
                reply_to_message_id=message.message_id,
                supports_streaming=True,
                read_timeout=300,
                write_timeout=300,
                connect_timeout=60,
                pool_timeout=60,
                **metadata,
            )

    await _send_respecting_flood_limit(send)


async def comment_on_channel_forward(update: Any, context: Any) -> None:
    """Telegram auto-forwards every channel post into its linked Discussion Group as a
    new message; this catches that forward and replies under it, which is how a bot
    "comments" on a channel post - there is no dedicated API for it.

    Posts the untranslated source video first and the transcript below it. The dub
    itself is not repeated here: it is already the channel post these replies hang off.
    """
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
    has_original = _find_submission_original_video(submission) is not None
    if not text and not has_original:
        return
    if not await asyncio.to_thread(store.mark_comment_posted, submission.id):
        return

    # Each reply is sent on its own: a missing original should not cost us the
    # transcript, and a transcript that will not send should not hide the video.
    delivered = False
    if has_original:
        try:
            await _reply_with_original_video(context, message, submission)
            delivered = True
        except Exception as exc:
            print(
                f"Original video comment failed for submission {submission.id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    if text:
        if len(text) > 4000:
            text = text[:4000] + "…"

        async def send_text() -> None:
            await context.bot.send_message(
                chat_id=message.chat_id, text=text, reply_to_message_id=message.message_id
            )

        try:
            await _send_respecting_flood_limit(send_text)
            delivered = True
        except Exception as exc:
            print(f"Comment post failed for submission {submission.id}: {type(exc).__name__}: {exc}", flush=True)

    if not delivered:
        # The claim was taken before sending to keep a redelivered forward from
        # double-posting. Nothing went out, so hand it back rather than leave the
        # post permanently marked as commented.
        await asyncio.to_thread(store.clear_comment_posted, submission.id)
        print(f"Comment claim released for submission {submission.id}: nothing was sent", flush=True)


def _parse_ids(raw: str) -> set[int]:
    values: set[int] = set()
    for token in raw.replace(";", ",").replace("\n", ",").split(","):
        token = token.strip().replace("\u00a0", "").replace(" ", "")
        if token.isdigit():
            values.add(int(token))
    return values


if __name__ == "__main__":
    main()
