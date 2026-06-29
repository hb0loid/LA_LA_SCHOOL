from __future__ import annotations

import asyncio
import contextlib
import hashlib
import heapq
import json
import multiprocessing
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .bot_config import BotSettings, load_bot_settings
from .download import download_video_url, extract_url, has_video_and_audio
from .ffmpeg import compress_video_for_telegram, make_audio_visual_video, probe_duration, trim_video
from .models import DubConfig
from .pipeline import run_dub, run_transcript
from .watermark import add_watermark


def _force_current_python_for_child_processes() -> None:
    multiprocessing.set_executable(sys.executable)
    if hasattr(sys, "_base_executable"):
        sys._base_executable = sys.executable


_force_current_python_for_child_processes()


SOURCE_LANGS = [
    ("auto", "Авто"),
    ("vi", "Вьетнамский"),
    ("ko", "Корейский"),
    ("tr", "Турецкий"),
    ("en", "Английский"),
    ("ru", "Русский"),
    ("ja", "Японский"),
    ("zh", "Китайский"),
    ("th", "Тайский"),
    ("de", "Немецкий"),
    ("es", "Испанский"),
    ("fr", "Французский"),
    ("it", "Итальянский"),
    ("pt", "Португальский"),
    ("ar", "Арабский"),
    ("he", "Иврит"),
    ("hi", "Хинди"),
    ("id", "Индонезийский"),
    ("ms", "Малайзийский"),
    ("az", "Азербайджанский"),
    ("pl", "Польский"),
    ("uk", "Украинский"),
]

ASR_METHODS = [
    ("ow-large-v3-chaos-backbone", "Поломанный дубляж"),
]

ASR_METHOD_CONFIGS = {
    "ow-large-v3-hunt": ("openai-whisper", "turbo", False),
    "ow-large-v3-chaos-backbone": ("openai-whisper", "turbo", False),
    "ow-large-v3-raw-dub": ("openai-whisper", "turbo", True),
    "ow-large-v3-chaos": ("openai-whisper", "turbo", True),
    "ow-large-v3-soft": ("openai-whisper", "turbo", False),
    "ow-large-v3-forced": ("openai-whisper", "turbo", True),
    "fw-large-v3-soft": ("faster-whisper", "large-v3", False),
    "fw-large-v3-forced": ("faster-whisper", "large-v3", True),
    "ow-large-v3-turbo-soft": ("openai-whisper", "turbo", False),
    "ow-large-v2-soft": ("openai-whisper", "large-v2", False),
    "ow-large-v2-forced": ("openai-whisper", "large-v2", True),
    "ow-large-v1-soft": ("openai-whisper", "large-v1", False),
    "ow-large-v1-forced": ("openai-whisper", "large-v1", True),
    "ow-medium-soft": ("openai-whisper", "medium", False),
    "ow-medium-forced": ("openai-whisper", "medium", True),
    "ow-small-soft": ("openai-whisper", "small", False),
    "ow-small-forced": ("openai-whisper", "small", True),
    "fw-small-soft": ("faster-whisper", "small", False),
    "fw-small-forced": ("faster-whisper", "small", True),
}

SPEAKER_COUNT_OPTIONS = [("auto", "Авто"), *[(str(index), str(index)) for index in range(1, 10)]]
VISUAL_MODE_OPTIONS = [
    ("original", "Оставить исходный видеоряд"),
    ("random", "Видеоряд скучный, сделай прикольно"),
]
TARGET_LANGS = [
    ("ru", "Русский"),
    ("uk", "Украинский"),
    ("en", "Английский"),
]
TTS_METHODS = [
    ("f5", "F5 (быстрее)"),
    ("qwen3", "Qwen3 (медленнее)"),
]

TELEGRAM_SAFE_VIDEO_BYTES = 45 * 1024 * 1024
TELEGRAM_DIRECT_DOWNLOAD_SAFE_BYTES = 20 * 1024 * 1024
RECOVERABLE_JOB_STATUSES = {"starting", "running", "queued", "ready"}
CLEANUP_JOB_STATUSES = {"done", "failed", "rejected"}
_LAST_STATUS_TEXT: dict[tuple[int | None, int | None], str] = {}
MAINTENANCE_MESSAGE = (
    "?????? ??????? ??????????? ?????? ??? ?????. "
    "?????? ????? ??????? ?????."
)



class _ApplicationContext:
    def __init__(self, application: Any) -> None:
        self.application = application
        self.bot = application.bot


def main() -> None:
    try:
        from telegram import Update
        from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters
    except ImportError as exc:
        raise RuntimeError("Install bot dependencies first: python -m pip install -e .[bot]") from exc

    settings = load_bot_settings()
    if settings.executor_mode not in {"local", "remote", "hybrid"}:
        raise RuntimeError("LALADUB_EXECUTOR_MODE must be local, remote, or hybrid.")
    if settings.executor_mode in {"remote", "hybrid"} and not settings.worker_api_token:
        raise RuntimeError("Set LALADUB_WORKER_API_TOKEN when remote workers are enabled.")
    settings.workdir.mkdir(parents=True, exist_ok=True)
    print(
        "La La Dub Bot starting "
        f"workdir={settings.workdir} "
        f"executor={settings.executor_mode} "
        f"local_jobs={settings.max_local_jobs} "
        f"worker_api={settings.worker_api_host}:{settings.worker_api_port} "
        f"translator={settings.translator} "
        f"tts={settings.tts} "
        f"f5={settings.f5_model}/{settings.f5_device} "
        f"qwen3={settings.qwen3_model} "
        f"multi_speaker={settings.multi_speaker} "
        f"speaker_clustering={settings.speaker_clustering}/{settings.max_speaker_clusters} "
        f"separation={settings.separation} "
        f"audio_bed={settings.audio_bed} "
        f"watermark={settings.watermark_image} "
        f"audio_visual={settings.audio_visual_source_dir or settings.workdir}/"
        f"{settings.audio_visual_resolution}/max_slice={settings.audio_visual_max_slice_seconds} "
        f"audio_visual_safety={settings.audio_visual_safety_enabled}/"
        f"{settings.audio_visual_safety_model}/"
        f"{settings.audio_visual_safety_threshold}/"
        f"{settings.audio_visual_safety_frames}/"
        f"{settings.audio_visual_safety_device} "
        f"paid_users={len(settings.paid_users)} "
        f"duration_limits=free:{settings.free_max_duration_seconds}/paid:{settings.paid_max_duration_seconds} "
        f"collapse_repetitions={settings.collapse_repetitions}/"
        f"{settings.max_phrase_repeats}/{settings.max_word_repeats} "
        f"inject_artifacts={settings.inject_artifacts}/"
        f"{settings.artifact_max_segments}/{settings.artifact_min_gap_seconds} "
        f"distort={settings.distort_translation}/{settings.translation_pivots}/pass2={settings.translation_second_pass_ratio} "
        f"asr={settings.asr_backend} "
        f"queue={settings.max_active_jobs}/{settings.max_active_jobs_per_user} "
        f"job_retention={settings.job_retention_seconds}s "
        f"default_asr_method={settings.default_asr_method} "
        f"whisper={settings.whisper_model}/{settings.whisper_device}/{settings.whisper_compute_type} "
        f"whisper_only={settings.whisper_only_model}/{settings.whisper_only_device} "
        f"maintenance={_maintenance_enabled(settings)} "
        f"suppress_ascii={settings.suppress_plain_ascii_tokens}",
        flush=True,
    )

    application = Application.builder().token(settings.token).post_init(_setup_bot_commands).build()
    application.bot_data["settings"] = settings
    application.bot_data["job_scheduler"] = _JobScheduler(settings)
    application.add_handler(MessageHandler(filters.ALL, maintenance_message_gate), group=-1)
    application.add_handler(CallbackQueryHandler(maintenance_callback_gate), group=-1)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("me", me))
    application.add_handler(CommandHandler("queue", queue_status))
    application.add_handler(CommandHandler("resume", resume))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(CommandHandler("maintenance", maintenance))
    application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, receive_video))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.AUDIO, receive_audio))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, receive_link))
    application.add_handler(CallbackQueryHandler(select_visual_mode, pattern=r"^vis:"))
    application.add_handler(CallbackQueryHandler(select_source, pattern=r"^src:"))
    application.add_handler(CallbackQueryHandler(select_asr_method, pattern=r"^asr:"))
    application.add_handler(CallbackQueryHandler(select_speaker_count, pattern=r"^spk:"))
    application.add_handler(CallbackQueryHandler(select_target_lang, pattern=r"^tgt:"))
    application.add_handler(CallbackQueryHandler(select_tts_method, pattern=r"^tts:"))
    application.add_handler(CallbackQueryHandler(resume_callback, pattern=r"^resume:"))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


def _maintenance_flag_path(settings: BotSettings) -> Path:
    return settings.workdir / "maintenance.flag"


def _maintenance_enabled(settings: BotSettings) -> bool:
    return _maintenance_flag_path(settings).is_file()


def _set_maintenance_enabled(settings: BotSettings, enabled: bool, *, user_id: int | None = None) -> None:
    flag_path = _maintenance_flag_path(settings)
    if not enabled:
        flag_path.unlink(missing_ok=True)
        return

    flag_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = flag_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "enabled_at": time.time(),
                "enabled_by": user_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(flag_path)


def _maintenance_blocks_user(settings: BotSettings, user_id: int | None) -> bool:
    return _maintenance_enabled(settings) and not settings.is_paid(user_id)


async def _setup_bot_commands(application: Any) -> None:
    settings: BotSettings = application.bot_data["settings"]
    if settings.executor_mode in {"remote", "hybrid"} and "worker_api" not in application.bot_data:
        from .worker_api import start_worker_api

        application.bot_data["worker_api"] = start_worker_api(
            application,
            host=settings.worker_api_host,
            port=settings.worker_api_port,
            token=settings.worker_api_token,
        )
    commands = [
        ("start", "Инструкция"),
        ("queue", "Показать очередь задач"),
        ("resume", "Продолжить последнюю задачу"),
        ("me", "Показать Telegram ID"),
        ("cancel", "Сбросить текущую задачу"),
    ]
    await application.bot.set_my_commands(commands)
    with contextlib.suppress(Exception):
        from telegram import BotCommandScopeAllPrivateChats, BotCommandScopeDefault

        await application.bot.set_my_commands(commands, scope=BotCommandScopeDefault())
        await application.bot.set_my_commands(commands, scope=BotCommandScopeAllPrivateChats())
    with contextlib.suppress(Exception):
        from telegram import MenuButtonCommands

        await application.bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    asyncio.create_task(_recover_interrupted_jobs(application))
    asyncio.create_task(_cleanup_finished_jobs_loop(application))
    asyncio.create_task(_maintenance_watch_loop(application))


async def _maintenance_watch_loop(application: Any) -> None:
    settings: BotSettings = application.bot_data["settings"]
    scheduler: _JobScheduler = application.bot_data["job_scheduler"]
    previous = _maintenance_enabled(settings)
    while True:
        await asyncio.sleep(2.0)
        current = _maintenance_enabled(settings)
        if current == previous:
            continue
        previous = current
        print(f"Maintenance mode changed: enabled={current}", flush=True)
        try:
            await scheduler.maintenance_changed(_ApplicationContext(application))
        except Exception:
            print(traceback.format_exc(), flush=True)


async def maintenance_message_gate(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if not _maintenance_blocks_user(settings, user.id if user else None):
        return

    from telegram.ext import ApplicationHandlerStop

    context.user_data.clear()
    if update.effective_message is not None:
        await update.effective_message.reply_text(MAINTENANCE_MESSAGE, reply_markup=_remove_reply_keyboard())
    raise ApplicationHandlerStop


async def maintenance_callback_gate(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if not _maintenance_blocks_user(settings, user.id if user else None):
        return

    from telegram.ext import ApplicationHandlerStop

    context.user_data.clear()
    query = update.callback_query
    if query is not None:
        await query.answer("Ведутся технические работы. Попробуй позже.", show_alert=True)
        with contextlib.suppress(Exception):
            await query.edit_message_text(MAINTENANCE_MESSAGE)
    raise ApplicationHandlerStop


async def maintenance(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None or not settings.is_paid(user.id):
        await update.effective_message.reply_text("Команда доступна только администратору.")
        return

    action = str(context.args[0] if context.args else "status").strip().lower()
    if action in {"on", "enable", "1"}:
        _set_maintenance_enabled(settings, True, user_id=user.id)
        enabled = True
    elif action in {"off", "disable", "0"}:
        _set_maintenance_enabled(settings, False, user_id=user.id)
        enabled = False
    elif action in {"status", "state"}:
        enabled = _maintenance_enabled(settings)
    else:
        await update.effective_message.reply_text("Использование: /maintenance on, /maintenance off или /maintenance status")
        return

    scheduler: _JobScheduler = context.application.bot_data["job_scheduler"]
    await scheduler.maintenance_changed(context)
    state = "включён" if enabled else "выключен"
    details = (
        "Обычные пользователи временно заблокированы. Текущие задачи продолжат работу."
        if enabled
        else "Приём новых задач восстановлен, сохранённая очередь продолжит работу."
    )
    await update.effective_message.reply_text(f"Режим технических работ {state}.\n{details}")


async def start(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    free_limit = settings.free_max_duration_seconds if settings.free_max_duration_seconds > 0 else None
    limit_text = _format_duration(free_limit) if free_limit is not None else "без лимита"
    await update.effective_message.reply_text(
        "Пришли видео или аудио. Я попрошу выбрать видеоряд, input-язык, количество голосов и язык озвучки.\n\n"
        "Input-язык используется как промежуточный язык перевода и источник Whisper-артефактов.\n"
        f"У бесплатных пользователей лимит: {limit_text}. На итоговом видео будет водяной знак.",
        reply_markup=_remove_reply_keyboard(),
    )


async def me(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    status = "оплачен" if settings.is_paid(user.id if user else None) else "бесплатный"
    await update.effective_message.reply_text(
        f"Твой Telegram ID: {user.id}\nСтатус: {status}",
        reply_markup=_remove_reply_keyboard(),
    )


async def queue_status(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    scheduler: _JobScheduler = context.application.bot_data["job_scheduler"]
    live = await scheduler.snapshot()
    disk_counts = await asyncio.to_thread(_job_status_counts, settings.workdir)

    live_total = live["active_total"] + live["pending_total"]
    lines = [
        "Очередь задач",
        f"Машины заняты: {live['busy_machines']}/{live['online_machines']}",
        f"Основной ПК: {live['local_machine_busy']}/{live['local_machine_total']} (слоты {live['active_local']}/{live['max_local_jobs']})",
        f"Воркеры: занято {live['remote_workers_busy']}/{live['remote_workers_online']}, свободно {live['remote_workers_idle']}",
        f"В работе сейчас: {live['active_total']}/{live['max_active_jobs']}",
        f"Ждёт в живой очереди: {live['pending_total']}",
        f"Всего в живом процессе: {live_total}",
        f"Премиум в очереди: {live['pending_premium']}",
        f"Обычных в очереди: {live['pending_normal']}",
        f"Пользователей с активными задачами: {live['active_users']}",
        "",
        "По файлам:",
        _format_status_counts(disk_counts),
    ]
    if live["remote_workers_stale"]:
        lines.insert(4, f"Воркеры без свежего пинга: {live['remote_workers_stale']}")
    if disk_counts.get("running", 0) + disk_counts.get("queued", 0) > live_total:
        lines.extend(
            [
                "",
                "Примечание: файловые running/queued могут включать зависшие задачи после перезапуска. Их можно подхватить через /resume.",
            ]
        )
    await update.effective_message.reply_text("\n".join(lines), reply_markup=_remove_reply_keyboard())


async def cancel(update: Any, context: Any) -> None:
    context.user_data.clear()
    await update.effective_message.reply_text("Отменил. Можешь прислать другое видео.", reply_markup=_remove_reply_keyboard())


async def _clear_reply_keyboard(message: Any) -> None:
    # Inline keyboards do not need a separate cleanup message. Existing reply
    # keyboards are removed by the next normal bot response where appropriate.
    return


async def _delete_message_later(message: Any, delay_seconds: float) -> None:
    await asyncio.sleep(delay_seconds)
    with contextlib.suppress(Exception):
        await message.delete()


async def resume(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None:
        return
    job = _find_latest_resumable_job(settings, user.id)
    if not job:
        await update.effective_message.reply_text("Не нашёл незавершённую задачу для продолжения.")
        return

    job["resume"] = "1"
    status_message = await update.effective_message.reply_text("Ставлю задачу в очередь.")
    await _enqueue_job(update, context, job, status_message)


async def resume_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None:
        await query.edit_message_text("Не удалось определить пользователя.")
        return

    job_key = query.data.split(":", 1)[1]
    job_dir = settings.workdir / str(user.id) / job_key
    job = _load_job_snapshot(job_dir)
    if not job:
        await query.edit_message_text("Не нашёл данные задачи для продолжения.")
        return
    job["resume"] = "1"
    await query.edit_message_text("Ставлю задачу в очередь.")
    await _enqueue_job(update, context, job, query.message)


async def receive_video(update: Any, context: Any) -> None:
    message = update.effective_message
    settings: BotSettings = context.application.bot_data["settings"]
    media = message.video or message.document
    if not media:
        return

    file_size = getattr(media, "file_size", 0) or 0
    max_configured_bytes = settings.max_file_mb * 1024 * 1024
    direct_limit_bytes = min(max_configured_bytes, TELEGRAM_DIRECT_DOWNLOAD_SAFE_BYTES)
    if file_size > direct_limit_bytes:
        await message.reply_text(_direct_video_too_large_text(direct_limit_bytes))
        return

    await _clear_reply_keyboard(message)

    user_id = update.effective_user.id

    job_dir = settings.workdir / str(user_id) / str(message.message_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = _guess_suffix(media)
    input_path = job_dir / f"input{suffix}"
    source_title = _source_title_from_media(media, message.message_id)

    status = await message.reply_text("Скачиваю видео...")
    try:
        tg_file = await media.get_file()
        await tg_file.download_to_drive(custom_path=str(input_path))
    except Exception as exc:
        if _is_telegram_file_too_big(exc):
            await status.edit_text(_direct_video_too_large_text(direct_limit_bytes))
        else:
            traceback_text = traceback.format_exc()
            print(traceback_text, flush=True)
            (job_dir / "error.log").write_text(traceback_text, encoding="utf-8")
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            await status.edit_text(f"Не смог скачать видео из Telegram:\n{details}")
        return

    input_path = await _prepare_input_video_duration(status, settings, user_id, input_path)
    if input_path is None:
        return

    await _remember_job_and_ask_source(context, status, job_dir, input_path, source_title, input_source="telegram_upload")


async def receive_audio(update: Any, context: Any) -> None:
    message = update.effective_message
    settings: BotSettings = context.application.bot_data["settings"]
    media = message.audio or message.voice or message.document
    if not media:
        return

    file_size = getattr(media, "file_size", 0) or 0
    max_configured_bytes = settings.max_file_mb * 1024 * 1024
    direct_limit_bytes = min(max_configured_bytes, TELEGRAM_DIRECT_DOWNLOAD_SAFE_BYTES)
    if file_size > direct_limit_bytes:
        await message.reply_text(_direct_video_too_large_text(direct_limit_bytes))
        return

    await _clear_reply_keyboard(message)

    user_id = update.effective_user.id
    job_dir = settings.workdir / str(user_id) / str(message.message_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = _guess_audio_suffix(media)
    audio_path = job_dir / f"input_audio{suffix}"
    source_title = _source_title_from_audio(media, message.message_id)

    status = await message.reply_text("Скачиваю аудио...")
    try:
        tg_file = await media.get_file()
        await tg_file.download_to_drive(custom_path=str(audio_path))
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        (job_dir / "error.log").write_text(traceback_text, encoding="utf-8")
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        await status.edit_text(f"Не смог скачать аудио:\n{details}")
        return

    await _remember_job_and_ask_source(context, status, job_dir, audio_path, source_title, input_source="telegram_audio")


async def receive_link(update: Any, context: Any) -> None:
    message = update.effective_message
    settings: BotSettings = context.application.bot_data["settings"]
    text = message.text or ""
    url = extract_url(text)
    if not url:
        await message.reply_text(
            "Пришли видеофайл или ссылку на видео, например YouTube.",
            reply_markup=_remove_reply_keyboard(),
        )
        return

    await _clear_reply_keyboard(message)

    user_id = update.effective_user.id
    job_dir = settings.workdir / str(user_id) / str(message.message_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    status = await message.reply_text("Скачиваю видео по ссылке на основном ПК...")
    try:
        input_path = await asyncio.to_thread(download_video_url, url, job_dir, settings.max_file_mb)
        source_title = _source_title_from_download(job_dir, input_path, url)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        (job_dir / "error.log").write_text(traceback_text, encoding="utf-8")
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        await status.edit_text(f"Не смог скачать видео по ссылке:\n{details}")
        return

    input_path = await _prepare_input_video_duration(status, settings, user_id, input_path)
    if input_path is None:
        return

    await _remember_job_and_ask_source(
        context,
        status,
        job_dir,
        input_path,
        source_title,
        input_source="coordinator_download",
        source_url=url,
    )


async def _prepare_input_audio_as_video(
    status: Any,
    settings: BotSettings,
    user_id: int | None,
    job_dir: Path,
    audio_path: Path,
) -> Path | None:
    try:
        duration = await asyncio.to_thread(probe_duration, audio_path)
    except Exception as exc:
        print(f"Audio duration check failed: {type(exc).__name__}: {exc}", flush=True)
        await status.edit_text(f"Не смог определить длительность аудио:\n{type(exc).__name__}: {exc}")
        return None

    target_duration = duration
    limit = _duration_limit_seconds(settings, user_id)
    if limit is not None and duration > limit + 0.5:
        target_duration = limit
        await status.edit_text(
            f"Аудио длиннее лимита: {_format_duration(duration)}.\n"
            f"Взял первые {_format_duration(limit)} и собираю видеоряд."
        )
    else:
        await status.edit_text("Собираю случайный видеоряд для аудио...")

    visual_source = settings.audio_visual_source_dir or settings.workdir
    source_roots = [visual_source]
    if visual_source.resolve() != settings.workdir.resolve():
        source_roots.append(settings.workdir)
    output_path = job_dir / "input_audio_visual.mp4"
    await asyncio.to_thread(
        make_audio_visual_video,
        audio_path,
        output_path,
        source_roots=source_roots,
        temp_dir=job_dir / "audio_visual_segments",
        duration=target_duration,
        resolution=settings.audio_visual_resolution,
        max_slice_seconds=settings.audio_visual_max_slice_seconds,
        exclude_dirs=[job_dir],
        safety_enabled=settings.audio_visual_safety_enabled,
        safety_cache_dir=settings.media_cache_dir / "visual_safety",
        safety_model=settings.audio_visual_safety_model,
        safety_threshold=settings.audio_visual_safety_threshold,
        safety_frames=settings.audio_visual_safety_frames,
        safety_device=settings.audio_visual_safety_device,
    )

    if not output_path.exists() or output_path.stat().st_size < 1024:
        await status.edit_text("Не удалось собрать видео из аудио: получился пустой файл.")
        return None
    if not await asyncio.to_thread(has_video_and_audio, output_path):
        await status.edit_text("Не удалось собрать видео из аудио: в результате нет видео и аудио.")
        return None

    await status.edit_text("Аудио превращено в видео. Продолжаю.")
    return output_path


async def _prepare_random_visual_video(
    status: Any,
    settings: BotSettings,
    job_dir: Path,
    input_path: Path,
) -> Path | None:
    try:
        duration = await asyncio.to_thread(probe_duration, input_path)
    except Exception as exc:
        print(f"Video duration check failed for random visual: {type(exc).__name__}: {exc}", flush=True)
        await _safe_edit_status(status, f"Не смог определить длительность видео:\n{type(exc).__name__}: {exc}")
        return None

    visual_source = settings.audio_visual_source_dir or settings.workdir
    source_roots = [visual_source]
    if visual_source.resolve() != settings.workdir.resolve():
        source_roots.append(settings.workdir)
    output_path = job_dir / f"{input_path.stem}_fun_visual.mp4"
    await asyncio.to_thread(
        make_audio_visual_video,
        input_path,
        output_path,
        source_roots=source_roots,
        temp_dir=job_dir / "fun_visual_segments",
        duration=duration,
        resolution=settings.audio_visual_resolution,
        max_slice_seconds=settings.audio_visual_max_slice_seconds,
        exclude_dirs=[job_dir],
    )

    if not output_path.exists() or output_path.stat().st_size < 1024:
        await _safe_edit_status(status, "Не удалось собрать прикольный видеоряд: получился пустой файл.")
        return None
    if not await asyncio.to_thread(has_video_and_audio, output_path):
        await _safe_edit_status(status, "Не удалось собрать прикольный видеоряд: в результате нет видео и аудио.")
        return None
    return output_path


async def _prepare_input_video_duration(
    status: Any,
    settings: BotSettings,
    user_id: int | None,
    input_path: Path,
) -> Path | None:
    limit = _duration_limit_seconds(settings, user_id)
    if limit is None:
        return input_path

    try:
        duration = await asyncio.to_thread(probe_duration, input_path)
    except Exception as exc:
        print(f"Video duration check skipped: {type(exc).__name__}: {exc}", flush=True)
        return input_path

    if not _duration_limit_exceeded(settings, user_id, duration):
        return input_path

    trimmed_path = input_path.with_name(f"{input_path.stem}_trimmed{input_path.suffix or '.mp4'}")
    await status.edit_text(_duration_trim_text(settings, user_id, duration))
    try:
        await asyncio.to_thread(trim_video, input_path, trimmed_path, limit)
    except Exception as exc:
        print(f"Video trim failed: {type(exc).__name__}: {exc}", flush=True)
        await status.edit_text(
            "Видео длиннее лимита, и не удалось автоматически обрезать его.\n"
            f"{type(exc).__name__}: {exc}"
        )
        return None

    if not trimmed_path.exists() or trimmed_path.stat().st_size < 1024:
        await status.edit_text("Не удалось обрезать видео: получился пустой файл.")
        return None

    try:
        trimmed_duration = await asyncio.to_thread(probe_duration, trimmed_path)
        print(
            f"Video trimmed for duration limit: {duration:.2f}s -> {trimmed_duration:.2f}s ({trimmed_path})",
            flush=True,
        )
    except Exception as exc:
        print(f"Trimmed video duration check skipped: {type(exc).__name__}: {exc}", flush=True)

    await status.edit_text(
        f"Видео длиннее лимита. Взял первые {_format_duration(limit)} и продолжаю."
    )
    return trimmed_path


async def _ensure_job_input_video(
    status: Any,
    job: dict[str, Any],
    settings: BotSettings,
    input_path: Path,
) -> Path | None:
    if await asyncio.to_thread(has_video_and_audio, input_path):
        return input_path

    job_dir = Path(str(job["job_dir"]))
    source_url = str(job.get("source_url") or _download_meta_url(job_dir) or "").strip()
    if not source_url:
        await _safe_edit_status(
            status,
            "Файл не содержит полноценное видео с аудио. Пришли видеофайл или ссылку заново.",
        )
        return None

    await _safe_edit_status(
        status,
        "Скачанный файл оказался не видео, а отдельной дорожкой. Перекачиваю ссылку заново...",
    )
    try:
        redownloaded_path = await asyncio.to_thread(download_video_url, source_url, job_dir, settings.max_file_mb)
    except Exception as exc:
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        await _safe_edit_status(status, f"Не смог перекачать видео по ссылке:\n{details}")
        return None

    if not await asyncio.to_thread(has_video_and_audio, redownloaded_path):
        await _safe_edit_status(status, "Перекачанный файл всё ещё не содержит полноценное видео с аудио.")
        return None

    job["input_path"] = str(redownloaded_path)
    _save_job_snapshot(job_dir, job, status="queued", redownloaded=True)
    return redownloaded_path


def _download_meta_url(job_dir: Path) -> str | None:
    meta_path = job_dir / "download_meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = data.get("webpage_url")
    return str(value) if value else None


def _duration_limit_seconds(settings: BotSettings, user_id: int | None) -> float | None:
    value = settings.paid_max_duration_seconds if settings.is_paid(user_id) else settings.free_max_duration_seconds
    if value <= 0:
        return None
    return value


def _duration_limit_exceeded(settings: BotSettings, user_id: int | None, duration: float) -> bool:
    limit = _duration_limit_seconds(settings, user_id)
    return limit is not None and duration > limit + 0.5


def _duration_limit_text(settings: BotSettings, user_id: int | None, duration: float) -> str:
    limit = _duration_limit_seconds(settings, user_id)
    tier = "платной" if settings.is_paid(user_id) else "бесплатной"
    if limit is None:
        return "Видео принято."
    return (
        f"В {tier} версии лимит видео: {_format_duration(limit)}.\n"
        f"Это видео: {_format_duration(duration)}."
    )


def _duration_trim_text(settings: BotSettings, user_id: int | None, duration: float) -> str:
    limit = _duration_limit_seconds(settings, user_id)
    tier = "платной" if settings.is_paid(user_id) else "бесплатной"
    if limit is None:
        return "Видео принято."
    return (
        f"В {tier} версии лимит видео: {_format_duration(limit)}.\n"
        f"Это видео: {_format_duration(duration)}.\n"
        "Обрезаю и возьму начало видео."
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.0f} МБ"


def _direct_video_too_large_text(limit_bytes: int) -> str:
    return (
        "Файл слишком большой для прямой загрузки через Telegram.\n"
        f"Лимит для видеофайла: {_format_mb(limit_bytes)}.\n"
        "Пришли ссылку на YouTube/TikTok/другой источник или сожми файл перед отправкой."
    )


def _is_telegram_file_too_big(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return "file is too big" in text or "too big" in text


async def _remember_job_and_ask_source(
    context: Any,
    status: Any,
    job_dir: Path,
    input_path: Path,
    source_title: str,
    *,
    input_source: str,
    source_url: str | None = None,
) -> None:
    context.user_data["job"] = {
        "job_dir": str(job_dir),
        "input_path": str(input_path),
        "source_title": source_title,
        "input_source": input_source,
    }
    if source_url:
        context.user_data["job"]["source_url"] = source_url
    if input_source == "telegram_audio":
        context.user_data["job"]["visual_mode"] = "random"
        _save_job_snapshot(job_dir, context.user_data["job"], status="select_source")
        await _ask_source_language(status, context.user_data["job"])
        return

    context.user_data["job"]["visual_mode"] = "original"
    _save_job_snapshot(job_dir, context.user_data["job"], status="select_visual")
    try:
        await status.edit_text(
            "Выбери видеоряд.",
            reply_markup=_language_keyboard("vis", VISUAL_MODE_OPTIONS, columns=1),
        )
    except Exception as exc:
        if "Message can't be edited" not in f"{type(exc).__name__}: {exc}":
            raise
        await status.reply_text(
            "Выбери видеоряд.",
            reply_markup=_language_keyboard("vis", VISUAL_MODE_OPTIONS, columns=1),
        )


async def _ask_source_language(status: Any, job: dict[str, Any]) -> None:
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_source")
    text = (
        "Выбери input-язык. Он будет использоваться как промежуточный язык перевода "
        "и источник Whisper-артефактов."
    )
    reply_markup = _language_keyboard("src", SOURCE_LANGS)
    try:
        await status.edit_text(text, reply_markup=reply_markup)
    except Exception as exc:
        if "Message can't be edited" not in f"{type(exc).__name__}: {exc}":
            raise
        await status.reply_text(text, reply_markup=reply_markup)


async def select_visual_mode(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    mode = query.data.split(":", 1)[1]
    if mode not in {"original", "random"}:
        await query.edit_message_text("Неизвестный режим видеоряда. Пришли видео ещё раз.")
        return

    job["visual_mode"] = mode
    job_dir = Path(str(job["job_dir"]))
    _save_job_snapshot(job_dir, job, status="select_source")

    await _ask_source_language(query.message, job)


async def select_source(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    source_lang = query.data.split(":", 1)[1]
    job["source_lang"] = None if source_lang == "auto" else source_lang
    job["asr_method"] = "ow-large-v3-chaos-backbone"
    job["mode"] = "dub"
    job["glitch_profile"] = "clean"
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_speakers")
    await query.edit_message_text(
        "Выбери количество голосов.",
        reply_markup=_language_keyboard("spk", SPEAKER_COUNT_OPTIONS, columns=5),
    )


async def select_asr_method(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    method = query.data.split(":", 1)[1]
    if method not in ASR_METHOD_CONFIGS:
        await query.edit_message_text("Неизвестный метод извлечения текста. Пришли видео ещё раз.")
        return
    if method == "ow-large-v3-raw-dub" and not job.get("source_lang"):
        await query.edit_message_text("Для сырого forced-дубляжа нужен конкретный input-язык, не авто. Пришли видео ещё раз и выбери язык.")
        context.user_data.pop("job", None)
        return

    job["asr_method"] = method
    job["mode"] = "dub"
    job["glitch_profile"] = "clean"
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_speakers")
    await query.edit_message_text(
        "Выбери количество голосов.",
        reply_markup=_language_keyboard("spk", SPEAKER_COUNT_OPTIONS, columns=5),
    )


async def select_speaker_count(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    value = query.data.split(":", 1)[1]
    allowed_values = {code for code, _label in SPEAKER_COUNT_OPTIONS}
    if value not in allowed_values:
        await query.edit_message_text("Неизвестное количество голосов. Пришли видео ещё раз.")
        return

    job["speaker_count"] = "auto" if value == "auto" else int(value)
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_target")
    await query.edit_message_text(
        "Выбери язык озвучки.",
        reply_markup=_language_keyboard("tgt", TARGET_LANGS, columns=2),
    )


async def select_target_lang(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    target_lang = _target_lang_value(query.data.split(":", 1)[1])
    if target_lang not in {"ru", "uk", "en"}:
        await query.edit_message_text("Неизвестный язык озвучки. Пришли видео ещё раз.")
        return

    job["target_lang"] = target_lang
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_tts")
    await query.edit_message_text(
        "Выбери метод озвучки.",
        reply_markup=_language_keyboard("tts", TTS_METHODS, columns=2),
    )


async def select_tts_method(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    tts_provider = _tts_provider_value(query.data.split(":", 1)[1])
    if tts_provider is None:
        await query.edit_message_text("Неизвестный метод озвучки. Пришли видео ещё раз.")
        return
    if _target_lang_value(job.get("target_lang")) == "uk" and tts_provider == "qwen3":
        await query.edit_message_text(
            "Qwen3 не поддерживает украинскую озвучку. Выбери F5.",
            reply_markup=_language_keyboard("tts", TTS_METHODS, columns=2),
        )
        return

    job["tts_provider"] = tts_provider
    _save_job_snapshot(Path(job["job_dir"]), job, status="queued")
    await query.edit_message_text(
        f"Ставлю задачу в очередь. Голоса: {_speaker_count_label(job.get('speaker_count'))}. "
        f"Язык озвучки: {_target_lang_label(job.get('target_lang'))}. "
        f"Движок: {_tts_method_label(tts_provider)}."
    )
    context.user_data.pop("job", None)
    await _enqueue_job(update, context, job, query.message)


async def _enqueue_job(update: Any, context: Any, job: dict[str, Any], status_message: Any) -> None:
    scheduler: _JobScheduler = context.application.bot_data["job_scheduler"]
    settings: BotSettings = context.application.bot_data["settings"]
    job["target_lang"] = _target_lang_value(job.get("target_lang"))
    chat = update.effective_chat
    user = update.effective_user
    if chat is None:
        await _safe_edit_status(status_message, "Не удалось определить чат для задачи.")
        return
    job["chat_id"] = chat.id
    if user is not None:
        job["user_id"] = user.id
        job["is_paid"] = settings.is_paid(user.id)

    input_path = Path(str(job.get("input_path") or ""))
    if input_path.exists():
        job_dir = Path(job["job_dir"])
        if job.get("input_source") == "telegram_audio":
            input_path = await _prepare_input_audio_as_video(status_message, settings, user.id if user else None, job_dir, input_path)
            if input_path is None:
                _save_job_snapshot(job_dir, job, status="rejected", error="audio_visual_failed")
                return
            job["input_path"] = str(input_path)
            _save_job_snapshot(job_dir, job, status="queued", audio_visual=True)
        else:
            input_path = await _ensure_job_input_video(status_message, job, settings, input_path)
            if input_path is None:
                _save_job_snapshot(job_dir, job, status="rejected", error="invalid_input_media")
                return
            prepared_input_path = await _prepare_input_video_duration(
                status_message,
                settings,
                user.id if user else None,
                input_path,
            )
            if prepared_input_path is None:
                _save_job_snapshot(job_dir, job, status="rejected", error="duration_limit")
                return
            if prepared_input_path != input_path:
                input_path = prepared_input_path
                job["input_path"] = str(prepared_input_path)
                _save_job_snapshot(job_dir, job, status="queued", trimmed=True)
            if job.get("visual_mode") == "random":
                await _safe_edit_status(status_message, "Собираю случайный видеоряд...")
                visual_path = await _prepare_random_visual_video(status_message, settings, job_dir, input_path)
                if visual_path is None:
                    _save_job_snapshot(job_dir, job, status="rejected", error="random_visual_failed")
                    return
                job["input_path"] = str(visual_path)
                _save_job_snapshot(job_dir, job, status="queued", random_visual=True)

    await scheduler.enqueue(
        context,
        chat_id=chat.id,
        user_id=user.id if user else None,
        job=job,
        status_message=status_message,
    )


async def _recover_interrupted_jobs(application: Any) -> None:
    settings: BotSettings = application.bot_data["settings"]
    scheduler: _JobScheduler = application.bot_data["job_scheduler"]
    context = _ApplicationContext(application)
    jobs = await asyncio.to_thread(_find_recoverable_jobs, settings)
    if not jobs:
        print("Startup recovery: no interrupted jobs found", flush=True)
        return

    recovered = 0
    skipped = 0
    print(f"Startup recovery: found {len(jobs)} interrupted job(s)", flush=True)
    for job in jobs:
        job_dir = Path(str(job["job_dir"]))
        chat_id = _job_chat_id(job)
        user_id = _job_user_id(job)
        if chat_id is None:
            skipped += 1
            _save_job_snapshot(job_dir, job, status="failed", error="startup_recovery_missing_chat_id")
            print(f"Startup recovery skipped, chat_id missing: {job_dir}", flush=True)
            continue

        job["resume"] = "1"
        job["recovered_at"] = time.time()
        job["target_lang"] = _target_lang_value(job.get("target_lang"))
        try:
            status_message = await application.bot.send_message(
                chat_id=chat_id,
                text="Бот был перезапущен. Автоматически продолжаю задачу и ставлю её обратно в очередь.",
                read_timeout=60,
                write_timeout=60,
                connect_timeout=30,
                pool_timeout=30,
            )
        except Exception as exc:
            skipped += 1
            _save_job_snapshot(job_dir, job, status="failed", error=f"startup_recovery_send_failed: {exc}")
            print(f"Startup recovery send failed for {job_dir}: {type(exc).__name__}: {exc}", flush=True)
            continue

        await scheduler.enqueue(
            context,
            chat_id=chat_id,
            user_id=user_id,
            job=job,
            status_message=status_message,
        )
        recovered += 1

    print(f"Startup recovery: recovered={recovered}, skipped={skipped}", flush=True)


def _find_recoverable_jobs(settings: BotSettings) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    if not settings.workdir.exists():
        return []
    for path in settings.workdir.rglob("job.json"):
        job_dir = path.parent
        job = _load_job_snapshot(job_dir)
        if not job:
            continue
        status = str(job.get("status") or "")
        if status not in RECOVERABLE_JOB_STATUSES:
            continue
        if not job.get("asr_method") and job.get("mode") != "raw_text":
            continue
        updated_at = float(job.get("queued_at") or job.get("updated_at") or path.stat().st_mtime)
        candidates.append((updated_at, job))
    candidates.sort(key=lambda item: item[0])
    return [job for _updated_at, job in candidates]


async def _cleanup_finished_jobs_loop(application: Any) -> None:
    settings: BotSettings = application.bot_data["settings"]
    if settings.job_retention_seconds <= 0:
        print("Job cleanup disabled: LALADUB_JOB_RETENTION_SECONDS=0", flush=True)
        return

    await asyncio.sleep(30)
    while True:
        try:
            deleted, bytes_freed = await asyncio.to_thread(_cleanup_finished_jobs_once, settings)
            if deleted:
                print(
                    f"Job cleanup: deleted={deleted}, freed={_format_bytes(bytes_freed)}",
                    flush=True,
                )
        except Exception as exc:
            print(f"Job cleanup failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        await asyncio.sleep(settings.cleanup_interval_seconds)


def _cleanup_finished_jobs_once(settings: BotSettings) -> tuple[int, int]:
    if settings.job_retention_seconds <= 0 or not settings.workdir.exists():
        return (0, 0)

    now = time.time()
    cutoff = now - settings.job_retention_seconds
    deleted = 0
    bytes_freed = 0
    workdir = settings.workdir.resolve()
    for path in settings.workdir.glob("*/*/job.json"):
        try:
            job_dir = path.parent.resolve()
            job_dir.relative_to(workdir)
        except ValueError:
            continue

        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(job, dict):
            continue

        status = str(job.get("status") or "")
        if status not in CLEANUP_JOB_STATUSES:
            continue
        finished_at = _coerce_float(job.get("finished_at"))
        updated_at = _coerce_float(job.get("updated_at"))
        marker = finished_at or updated_at or path.stat().st_mtime
        if marker > cutoff:
            continue

        size = _directory_size(job_dir)
        shutil.rmtree(job_dir)
        deleted += 1
        bytes_freed += size

    return (deleted, bytes_freed)


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        with contextlib.suppress(OSError):
            if child.is_file():
                total += child.stat().st_size
    return total


def _coerce_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _job_user_id(job: dict[str, Any]) -> int | None:
    user_id = _coerce_int(job.get("user_id"))
    if user_id is not None:
        return user_id
    try:
        return int(Path(str(job["job_dir"])).parent.name)
    except Exception:
        return None


def _job_chat_id(job: dict[str, Any]) -> int | str | None:
    chat_id = job.get("chat_id")
    if isinstance(chat_id, str) and chat_id.strip():
        parsed = _coerce_int(chat_id)
        return parsed if parsed is not None else chat_id
    parsed = _coerce_int(chat_id)
    if parsed is not None:
        return parsed
    return _job_user_id(job)


def _coerce_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


class _QueuedJob:
    def __init__(
        self,
        *,
        key: str,
        priority: int,
        sequence: int,
        chat_id: int | str,
        user_id: int | None,
        job: dict[str, Any],
        status_message: Any,
        enqueued_at: float,
        premium: bool,
    ) -> None:
        self.key = key
        self.priority = priority
        self.sequence = sequence
        self.chat_id = chat_id
        self.user_id = user_id
        self.job = job
        self.status_message = status_message
        self.enqueued_at = enqueued_at
        self.premium = premium
        self.job_id = _remote_job_id(job)
        self.worker_id: str | None = None
        self.execution_kind: str | None = None
        self.progress: _ProgressState | None = None
        self.progress_task: asyncio.Task[Any] | None = None


class _JobScheduler:
    def __init__(self, settings: BotSettings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._pending: list[tuple[int, int, _QueuedJob]] = []
        self._active_total = 0
        self._active_local = 0
        self._active_by_user: dict[int, int] = {}
        self._known_jobs: set[str] = set()
        self._leased: dict[str, _QueuedJob] = {}
        self._remote_workers: dict[str, dict[str, Any]] = {}
        self._remote_worker_ttl = 30.0
        self._sequence = 0

    async def enqueue(
        self,
        context: Any,
        *,
        chat_id: int | str,
        user_id: int | None,
        job: dict[str, Any],
        status_message: Any,
    ) -> None:
        key = _job_queue_key(job)
        async with self._lock:
            if key in self._known_jobs:
                await _safe_edit_status(status_message, "Эта задача уже в очереди или выполняется.")
                return

            self._sequence += 1
            premium = self._settings.is_paid(user_id)
            item = _QueuedJob(
                key=key,
                priority=0 if premium else 10,
                sequence=self._sequence,
                chat_id=chat_id,
                user_id=user_id,
                job=job,
                status_message=status_message,
                enqueued_at=time.time(),
                premium=premium,
            )
            job["queued_at"] = item.enqueued_at
            job["queue_priority"] = "premium" if premium else "normal"
            self._known_jobs.add(key)
            heapq.heappush(self._pending, (item.priority, item.sequence, item))
            _save_job_snapshot(Path(job["job_dir"]), job, status="queued")
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()

    async def finish(self, context: Any, item: _QueuedJob) -> None:
        async with self._lock:
            self._active_total = max(0, self._active_total - 1)
            if item.execution_kind == "local":
                self._active_local = max(0, self._active_local - 1)
            if item.user_id is not None:
                current = self._active_by_user.get(item.user_id, 0) - 1
                if current > 0:
                    self._active_by_user[item.user_id] = current
                else:
                    self._active_by_user.pop(item.user_id, None)
            self._known_jobs.discard(item.key)
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()

    async def maintenance_changed(self, context: Any) -> None:
        async with self._lock:
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()

    async def lease_remote(self, context: Any, worker_id: str) -> dict[str, Any] | None:
        async with self._lock:
            if self._settings.executor_mode not in {"remote", "hybrid"}:
                raise RuntimeError("Remote workers are disabled. Set LALADUB_EXECUTOR_MODE=remote or hybrid.")
            self._mark_remote_worker_locked(worker_id, active_job_id=None)
            index = self._next_startable_index(execution_kind="remote")
            if index is None:
                return None
            _, _, item = self._pending.pop(index)
            heapq.heapify(self._pending)
            self._active_total += 1
            if item.user_id is not None:
                self._active_by_user[item.user_id] = self._active_by_user.get(item.user_id, 0) + 1
            item.worker_id = worker_id
            item.execution_kind = "remote"
            item.job["started_at"] = time.time()
            item.job["worker_id"] = worker_id
            item.job["is_paid"] = self._settings.is_paid(item.user_id)
            _save_job_snapshot(Path(item.job["job_dir"]), item.job, status="running")
            item.progress = _ProgressState("Raw Whisper" if item.job.get("mode") == "raw_text" else "Full dubbing")
            item.progress.update("Remote worker leased", 1, 100, worker_id)
            item.progress_task = context.application.create_task(_progress_updater(item.status_message, item.progress))
            self._leased[item.job_id] = item
            self._mark_remote_worker_locked(worker_id, active_job_id=item.job_id)
            await _safe_edit_status(item.status_message, item.progress.render())
            await self._refresh_pending_locked()
            payload = _remote_job_payload(item.job)
            return {
                "job_id": item.job_id,
                "job": payload,
                "input_filename": Path(str(item.job["input_path"])).name,
            }

    async def remote_input_path(self, job_id: str) -> Path | None:
        async with self._lock:
            item = self._leased.get(job_id)
            if item is None:
                return None
            path = Path(str(item.job.get("input_path") or ""))
            return path if path.exists() else None

    async def remote_upload_path(self, job_id: str, kind: str, filename: str) -> Path | None:
        async with self._lock:
            item = self._leased.get(job_id)
            if item is None:
                return None
            safe_kind = _safe_upload_name(kind)
            if safe_kind not in {"video", "transcript", "documents", "error"}:
                raise ValueError(f"Unsupported upload kind: {kind}")
            safe_filename = _safe_upload_name(filename) or "upload.bin"
            path = Path(str(item.job["job_dir"])) / "remote_result" / safe_kind / safe_filename
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    async def remote_progress(self, job_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            item = self._leased.get(job_id)
            if item is None or item.progress is None:
                return
            item.progress.update(
                str(payload.get("stage") or ""),
                _coerce_int(payload.get("current")),
                _coerce_int(payload.get("total")),
                None if payload.get("detail") is None else str(payload.get("detail")),
            )
            item.job["worker_stage"] = str(payload.get("stage") or "")
            item.job["worker_detail"] = str(payload.get("detail") or "")
            _save_job_snapshot(Path(item.job["job_dir"]), item.job, status="running")

    async def complete_remote(self, context: Any, job_id: str, manifest: dict[str, Any]) -> None:
        async with self._lock:
            item = self._leased.get(job_id)
            if item is None:
                raise RuntimeError(f"Unknown leased job: {job_id}")
            if item.worker_id:
                self._mark_remote_worker_locked(item.worker_id, active_job_id=None)
            context.application.create_task(self._finalize_remote_item(context, item, manifest))

    async def fail_remote(self, context: Any, job_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            item = self._leased.get(job_id)
            if item is None:
                raise RuntimeError(f"Unknown leased job: {job_id}")
            if item.worker_id:
                self._mark_remote_worker_locked(item.worker_id, active_job_id=None)
            context.application.create_task(self._fail_remote_item(context, item, payload))

    async def _finalize_remote_item(self, context: Any, item: _QueuedJob, manifest: dict[str, Any]) -> None:
        try:
            await _send_remote_worker_result(context, item, manifest)
        finally:
            async with self._lock:
                self._leased.pop(item.job_id, None)
            await self.finish(context, item)

    async def _fail_remote_item(self, context: Any, item: _QueuedJob, payload: dict[str, Any]) -> None:
        try:
            details = str(payload.get("error") or "Remote worker failed")
            traceback_text = str(payload.get("traceback") or details)
            job_dir = Path(str(item.job["job_dir"]))
            (job_dir / "error.log").write_text(traceback_text, encoding="utf-8")
            _save_job_snapshot(job_dir, item.job, status="failed", error=details)
            await _finish_progress(item.progress, item.progress_task, item.status_message, "Error", failed=True, detail=details)
            await context.bot.send_message(
                chat_id=item.chat_id,
                text=f"Task failed:\n{details}",
                reply_markup=_resume_keyboard(job_dir),
            )
        finally:
            async with self._lock:
                self._leased.pop(item.job_id, None)
            await self.finish(context, item)

    async def _dispatch_locked(self, context: Any) -> None:
        while self._can_start_local_worker():
            index = self._next_startable_index(execution_kind="local")
            if index is None:
                return
            _, _, item = self._pending.pop(index)
            heapq.heapify(self._pending)
            self._active_total += 1
            self._active_local += 1
            if item.user_id is not None:
                self._active_by_user[item.user_id] = self._active_by_user.get(item.user_id, 0) + 1
            item.execution_kind = "local"
            item.job["started_at"] = time.time()
            _save_job_snapshot(Path(item.job["job_dir"]), item.job, status="starting")
            context.application.create_task(self._run_item(context, item))

    def _can_start_local_worker(self) -> bool:
        if self._settings.executor_mode == "remote":
            return False
        local_limit = self._settings.max_active_jobs if self._settings.executor_mode == "local" else self._settings.max_local_jobs
        if local_limit <= 0:
            return False
        if self._active_local >= local_limit:
            return False
        return self._active_total < self._settings.max_active_jobs

    def _next_startable_index(self, *, execution_kind: str) -> int | None:
        best_index: int | None = None
        best_key: tuple[int, int] | None = None
        for index, (priority, sequence, item) in enumerate(self._pending):
            if not self._can_start(item, execution_kind=execution_kind):
                continue
            key = (priority, sequence)
            if best_key is None or key < best_key:
                best_index = index
                best_key = key
        return best_index

    def _can_start(self, item: _QueuedJob, *, execution_kind: str) -> bool:
        if self._active_total >= self._settings.max_active_jobs:
            return False
        if _maintenance_enabled(self._settings) and not item.premium:
            return False
        if execution_kind == "remote" and _target_lang_value(item.job.get("target_lang")) != "ru":
            return False
        if execution_kind == "remote" and _tts_provider_value(item.job.get("tts_provider")) == "qwen3":
            return False
        if item.user_id is None:
            return True
        return self._active_by_user.get(item.user_id, 0) < self._settings.max_active_jobs_per_user

    async def _run_item(self, context: Any, item: _QueuedJob) -> None:
        try:
            await _process_job(context, item.chat_id, item.user_id, item.job, item.status_message)
        finally:
            await self.finish(context, item)

    async def _refresh_pending_locked(self) -> None:
        pending = sorted((priority, sequence, item) for priority, sequence, item in self._pending)
        for position, (_, _, item) in enumerate(pending, start=1):
            await _safe_edit_status(item.status_message, self._queue_text(item, position))

    def _queue_text(self, item: _QueuedJob, position: int) -> str:
        title = "Сырой Whisper" if item.job.get("mode") == "raw_text" else "Полноценный дубляж"
        if _maintenance_enabled(self._settings) and not item.premium:
            return "\n".join(
                [
                    f"{title}: Приостановлен",
                    MAINTENANCE_MESSAGE,
                    "Задача сохранена и продолжится после завершения работ.",
                ]
            )
        active_for_user = self._active_by_user.get(item.user_id, 0) if item.user_id is not None else 0
        tier = "премиум" if item.premium else "обычный"
        target_label = _target_lang_label(item.job.get("target_lang"))
        tts_label = _tts_method_label(item.job.get("tts_provider") or self._settings.tts)
        return "\n".join(
            [
                f"{title}: В очереди",
                f"Позиция: {position}",
                f"Сейчас выполняется: {self._active_total}/{self._settings.max_active_jobs}",
                f"У тебя выполняется: {active_for_user}/{self._settings.max_active_jobs_per_user}",
                f"Язык озвучки: {target_label}",
                f"Движок: {tts_label}",
                f"Приоритет: {tier}",
            ]
        )

    def _mark_remote_worker_locked(self, worker_id: str, *, active_job_id: str | None) -> None:
        worker_id = str(worker_id or "worker").strip() or "worker"
        state = self._remote_workers.get(worker_id, {})
        state["last_seen"] = time.time()
        state["active_job_id"] = active_job_id
        self._remote_workers[worker_id] = state

    def _remote_worker_counts_locked(self) -> dict[str, int]:
        now = time.time()
        online = 0
        busy = 0
        stale = 0
        for state in self._remote_workers.values():
            active_job_id = str(state.get("active_job_id") or "")
            last_seen = float(state.get("last_seen") or 0.0)
            is_busy = bool(active_job_id)
            is_online = is_busy or (now - last_seen) <= self._remote_worker_ttl
            if is_online:
                online += 1
            else:
                stale += 1
            if is_busy:
                busy += 1
        return {
            "online": online,
            "busy": busy,
            "idle": max(0, online - busy),
            "stale": stale,
        }

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            pending_premium = sum(1 for _priority, _sequence, item in self._pending if item.premium)
            pending_total = len(self._pending)
            remote_counts = self._remote_worker_counts_locked()
            max_local_jobs = (
                self._settings.max_active_jobs
                if self._settings.executor_mode == "local"
                else self._settings.max_local_jobs
            )
            local_machine_total = 1 if max_local_jobs > 0 else 0
            local_machine_busy = 1 if self._active_local > 0 and local_machine_total else 0
            return {
                "active_total": self._active_total,
                "active_local": self._active_local,
                "leased_remote": len(self._leased),
                "local_machine_total": local_machine_total,
                "local_machine_busy": local_machine_busy,
                "remote_workers_online": remote_counts["online"],
                "remote_workers_busy": remote_counts["busy"],
                "remote_workers_idle": remote_counts["idle"],
                "remote_workers_stale": remote_counts["stale"],
                "online_machines": local_machine_total + remote_counts["online"],
                "busy_machines": local_machine_busy + remote_counts["busy"],
                "active_users": len(self._active_by_user),
                "pending_total": pending_total,
                "pending_premium": pending_premium,
                "pending_normal": pending_total - pending_premium,
                "max_active_jobs": self._settings.max_active_jobs,
                "max_local_jobs": max_local_jobs,
                "max_active_jobs_per_user": self._settings.max_active_jobs_per_user,
            }


def _job_queue_key(job: dict[str, Any]) -> str:
    return str(Path(str(job["job_dir"])).resolve())


def _remote_job_id(job: dict[str, Any]) -> str:
    user_id = _coerce_int(job.get("user_id"))
    job_name = Path(str(job.get("job_dir") or "job")).name
    if user_id is not None and job_name:
        return f"{user_id}_{_safe_upload_name(job_name)}"
    digest = hashlib.sha256(_job_queue_key(job).encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"job_{digest}"


def _remote_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    blocked = {"job_dir", "input_path", "url", "source_url", "download_url", "original_url"}
    payload = {key: value for key, value in job.items() if key not in blocked}
    payload["target_lang"] = _target_lang_value(job.get("target_lang"))
    return payload


def _safe_upload_name(value: str) -> str:
    value = Path(str(value)).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._ ")
    return value[:180]


def _job_status_counts(workdir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not workdir.exists():
        return counts
    for path in workdir.rglob("job.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            status = "bad_json"
        else:
            status = str(data.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _format_status_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "нет задач"
    preferred = ["running", "starting", "queued", "select_source", "select_method", "ready", "failed", "rejected", "done"]
    parts = [f"{status}={counts[status]}" for status in preferred if counts.get(status)]
    parts.extend(f"{status}={count}" for status, count in sorted(counts.items()) if status not in preferred)
    return ", ".join(parts)


class _ProgressState:
    def __init__(self, title: str) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._title = title
        self._stage = "В очереди"
        self._detail = ""
        self._current = 0
        self._total = 100
        self._done = False
        self._failed = False

    def update(
        self,
        stage: str,
        current: int | None = None,
        total: int | None = None,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            if stage:
                self._stage = stage
            if total is not None and total > 0:
                self._total = total
            if current is not None:
                self._current = max(0, min(current, self._total))
            if detail is not None:
                self._detail = detail

    def finish(self, stage: str, *, failed: bool = False, detail: str | None = None) -> None:
        with self._lock:
            self._stage = stage
            self._current = self._total
            self._done = True
            self._failed = failed
            if detail is not None:
                self._detail = detail

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def render(self) -> str:
        with self._lock:
            title = self._title
            stage = self._stage
            detail = self._detail
            current = self._current
            total = self._total
            done = self._done
            failed = self._failed
            elapsed = time.monotonic() - self._started_at

        percent = round(current * 100 / max(1, total))
        status = "Ошибка" if failed else "Готово" if done else "В работе"
        lines = [
            f"{title}: {status}",
            f"{_progress_bar(percent)} {percent}%",
            f"Этап: {stage}",
            f"Прошло: {_format_elapsed(elapsed)}",
        ]
        if detail:
            lines.append(f"Детали: {_short_status_detail(detail)}")
        return "\n".join(lines)


def _progress_bar(percent: int, width: int = 20) -> str:
    percent = max(0, min(100, percent))
    filled = round(width * percent / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _short_status_detail(detail: str, limit: int = 180) -> str:
    detail = re.sub(r"\s+", " ", detail).strip()
    if len(detail) <= limit:
        return detail
    return detail[: limit - 3].rstrip() + "..."


def _asr_method_label(method: str) -> str:
    return next((label for code, label in ASR_METHODS if code == method), method)


def _target_lang_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "ua":
        text = "uk"
    return text if text in {"ru", "uk", "en"} else "ru"


def _target_lang_label(value: Any) -> str:
    target_lang = _target_lang_value(value)
    return next((label for code, label in TARGET_LANGS if code == target_lang), target_lang)


def _tts_provider_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"qwen3", "qwen3-tts", "qwen3tts"}:
        return "qwen3"
    if text in {"f5", "f5-tts", "f5tts"}:
        return "f5"
    return None


def _tts_method_label(value: Any) -> str:
    provider = _tts_provider_value(value)
    return next((label for code, label in TTS_METHODS if code == provider), str(value or "Авто"))


def _speaker_count_value(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text or text in {"auto", "none", "null"}:
        return None
    try:
        count = int(text)
    except ValueError:
        return None
    if 1 <= count <= 9:
        return count
    return None


def _speaker_count_label(value: Any) -> str:
    count = _speaker_count_value(value)
    return "Авто" if count is None else str(count)


def _apply_speaker_count(config: DubConfig, job: dict[str, Any]) -> None:
    count = _speaker_count_value(job.get("speaker_count"))
    if count is None:
        return
    config.max_speaker_clusters = count
    if count <= 1:
        config.multi_speaker = False
        config.speaker_clustering = False
    else:
        config.multi_speaker = True
        config.speaker_clustering = True


async def _progress_updater(message: Any, progress: _ProgressState, interval_seconds: float = 2.5) -> None:
    last_text = getattr(message, "text", "") or ""
    while True:
        text = progress.render()
        if text != last_text:
            await _safe_edit_status(message, text)
            last_text = text
        if progress.is_done():
            return
        await asyncio.sleep(interval_seconds)


async def _safe_edit_status(message: Any, text: str) -> None:
    key = (getattr(getattr(message, "chat", None), "id", None), getattr(message, "message_id", None))
    if _LAST_STATUS_TEXT.get(key) == text:
        return
    try:
        await message.edit_text(text)
        _LAST_STATUS_TEXT[key] = text
    except Exception as exc:
        if "Message is not modified" in str(exc):
            _LAST_STATUS_TEXT[key] = text
            return
        print(f"Progress edit skipped: {type(exc).__name__}: {exc}", flush=True)


async def _finish_progress(
    progress: _ProgressState | None,
    progress_task: asyncio.Task[Any] | None,
    status_message: Any,
    stage: str,
    *,
    failed: bool = False,
    detail: str | None = None,
) -> None:
    if progress is not None:
        progress.finish(stage, failed=failed, detail=detail)
    if progress is not None and status_message is not None:
        await _safe_edit_status(status_message, progress.render())
    if progress_task is not None:
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task


def _apply_text_extraction_method(config: DubConfig, job: dict[str, str], settings: BotSettings) -> None:
    selected_source = job.get("source_lang") or None
    method = job.get("asr_method") or settings.default_asr_method
    if method not in ASR_METHOD_CONFIGS:
        method = "ow-large-v3-chaos-backbone"
    backend, model, forced = ASR_METHOD_CONFIGS[method]
    hunt_artifacts = method.endswith("-hunt")
    chaos_backbone = method.endswith("-chaos-backbone")
    raw_dub = method.endswith("-raw-dub")
    chaos_asr = method.endswith("-chaos")

    config.asr_backend = backend
    config.whisper_model = model
    if backend == "openai-whisper":
        config.whisper_device = settings.whisper_only_device
        config.whisper_compute_type = "auto"
    else:
        config.whisper_device = settings.whisper_device
        config.whisper_compute_type = settings.whisper_compute_type

    if forced and not selected_source:
        print(f"Forced ASR method {method} needs non-auto input language; falling back to auto soft ASR", flush=True)
        forced = False

    config.source_lang = selected_source if forced else None
    config.force_source_language = forced
    config.suppress_plain_ascii_tokens = False
    config.asr_retry_on_repetition = not forced
    config.asr_fallback_on_sparse = False
    config.artifact_source_lang = None
    config.input_pivot_lang = None
    config.inject_artifacts = False
    config.artifact_chaos_mode = False
    config.distort_main_translation = False
    config.glitch_profile = "faithful" if forced else "clean"

    if raw_dub and forced:
        config.source_lang = selected_source
        config.force_source_language = True
        config.asr_retry_on_repetition = False
        config.asr_fallback_on_sparse = False
        config.input_pivot_lang = None
        config.artifact_source_lang = None
        config.inject_artifacts = False
        config.artifact_chaos_mode = True
        config.glitch_profile = "faithful"
        config.collapse_repetitions = False
        config.distort_main_translation = True
    elif chaos_asr and forced:
        config.source_lang = selected_source
        config.force_source_language = True
        config.asr_retry_on_repetition = False
        config.asr_fallback_on_sparse = True
        config.input_pivot_lang = None
        config.artifact_source_lang = None
        config.inject_artifacts = False
        config.glitch_profile = "faithful"
        config.collapse_repetitions = False
        config.distort_main_translation = True
    elif chaos_backbone:
        config.source_lang = None
        config.force_source_language = False
        config.asr_retry_on_repetition = True
        config.asr_fallback_on_sparse = False
        config.input_pivot_lang = selected_source
        config.artifact_source_lang = selected_source
        config.inject_artifacts = bool(settings.inject_artifacts and selected_source)
        config.artifact_chaos_mode = True
        config.artifact_max_segments = max(settings.artifact_max_segments, 48)
        config.glitch_profile = "clean"
        config.distort_main_translation = True
    elif hunt_artifacts:
        config.source_lang = None
        config.force_source_language = False
        config.asr_retry_on_repetition = True
        config.input_pivot_lang = selected_source
        config.artifact_source_lang = selected_source
        config.inject_artifacts = bool(settings.inject_artifacts and selected_source)
        config.glitch_profile = "clean"
    elif not forced:
        config.input_pivot_lang = selected_source


async def _process_job(
    context: Any,
    chat_id: int | str,
    user_id: int | None,
    job: dict[str, Any],
    status_message: Any | None = None,
) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    job_dir = Path(job["job_dir"])
    output_path = job_dir / "dubbed.mp4"
    target_lang = _target_lang_value(job.get("target_lang"))
    job["target_lang"] = target_lang
    tts_provider = _tts_provider_value(job.get("tts_provider")) or settings.tts
    if target_lang == "uk" and tts_provider.lower() in {"qwen3", "qwen3-tts", "qwen3tts"}:
        tts_provider = "f5"
    _save_job_snapshot(job_dir, job, status="running")
    progress: _ProgressState | None = None
    progress_task: asyncio.Task[Any] | None = None

    config = DubConfig(
        output=output_path,
        workdir=job_dir / "work",
        source_lang=None,
        target_lang=target_lang,
        asr_backend=settings.asr_backend,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
        whisper_compute_type=settings.whisper_compute_type,
        translator=settings.translator,
        tts=tts_provider,
        voice=settings.voice,
        speaker_wav=settings.speaker_wav,
        xtts_model=settings.xtts_model,
        xtts_device=settings.xtts_device,
        xtts_speed=settings.xtts_speed,
        f5_python=settings.f5_python,
        f5_model=settings.f5_model,
        f5_hf_repo=settings.f5_hf_repo,
        f5_hf_ckpt_path=settings.f5_hf_ckpt_path,
        f5_hf_vocab_path=settings.f5_hf_vocab_path,
        f5_ckpt_file=settings.f5_ckpt_file,
        f5_vocab_file=settings.f5_vocab_file,
        f5_cache_dir=settings.f5_cache_dir,
        media_cache_dir=settings.media_cache_dir,
        f5_device=settings.f5_device,
        f5_speed=settings.f5_speed,
        f5_nfe_step=settings.f5_nfe_step,
        f5_cfg_strength=settings.f5_cfg_strength,
        f5_target_rms=settings.f5_target_rms,
        f5_cross_fade_duration=settings.f5_cross_fade_duration,
        f5_remove_silence=settings.f5_remove_silence,
        f5_timeout_seconds=settings.f5_timeout_seconds,
        qwen3_python=settings.qwen3_python,
        qwen3_model=settings.qwen3_model,
        qwen3_cache_dir=settings.qwen3_cache_dir,
        qwen3_timeout_seconds=settings.qwen3_timeout_seconds,
        multi_speaker=settings.multi_speaker,
        speaker_reference_seconds=settings.speaker_reference_seconds,
        speaker_clustering=settings.speaker_clustering,
        max_speaker_clusters=settings.max_speaker_clusters,
        speaker_cluster_threshold=settings.speaker_cluster_threshold,
        separation=settings.separation,
        separation_device=settings.separation_device,
        demucs_model=settings.demucs_model,
        audio_bed=settings.audio_bed,
        glitch_profile=job.get("glitch_profile", "clean"),
        original_volume=settings.original_volume,
        dub_volume=settings.dub_volume,
        force_source_language=False,
        suppress_plain_ascii_tokens=settings.suppress_plain_ascii_tokens,
        asr_retry_on_repetition=True,
        artifact_source_lang=job.get("source_lang") or None,
        artifact_whisper_model=settings.whisper_only_model,
        artifact_whisper_device=settings.whisper_only_device,
        inject_artifacts=settings.inject_artifacts,
        artifact_max_segments=settings.artifact_max_segments,
        artifact_min_gap_seconds=settings.artifact_min_gap_seconds,
        distort_translation=settings.distort_translation,
        translation_pivots=settings.translation_pivots,
        translation_second_pass_ratio=settings.translation_second_pass_ratio,
        collapse_repetitions=settings.collapse_repetitions,
        max_phrase_repeats=settings.max_phrase_repeats,
        max_word_repeats=settings.max_word_repeats,
    )
    _apply_text_extraction_method(config, job, settings)
    _apply_speaker_count(config, job)

    try:
        if job.get("mode") == "raw_text":
            transcript_config = DubConfig(
                output=output_path,
                workdir=job_dir / "work",
                source_lang=job.get("source_lang") or None,
                target_lang=target_lang,
                asr_backend="openai-whisper",
                whisper_model=settings.whisper_only_model,
                whisper_device=settings.whisper_only_device,
                whisper_compute_type="auto",
                translator="identity",
                tts="none",
                separation="none",
                audio_bed="original",
                glitch_profile="faithful",
                condition_on_previous_text=True,
                hallucination_silence_threshold=None,
                force_source_language=True,
                suppress_plain_ascii_tokens=settings.suppress_plain_ascii_tokens,
                asr_retry_on_repetition=False,
                artifact_chaos_mode=True,
                collapse_repetitions=False,
            )
            progress = _ProgressState("Сырой Whisper")
            progress.update(
                "В очереди",
                1,
                100,
                f"вход={transcript_config.source_lang or 'auto'}, model={settings.whisper_only_model}",
            )
            if status_message is not None:
                await _safe_edit_status(status_message, progress.render())
            else:
                status_message = await context.bot.send_message(chat_id=chat_id, text=progress.render())
            progress_task = asyncio.create_task(_progress_updater(status_message, progress))
            transcript_config.progress_callback = progress.update

            srt_path, txt_path = await asyncio.to_thread(run_transcript, Path(job["input_path"]), transcript_config)
            meta_path = job_dir / "work" / "whisper_only_meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "mode": "raw_text",
                        "asr_backend": transcript_config.asr_backend,
                        "model": transcript_config.whisper_model,
                        "device": transcript_config.whisper_device,
                        "source_lang": transcript_config.source_lang,
                        "target_lang": transcript_config.target_lang,
                        "task": transcript_config.whisper_task,
                        "condition_on_previous_text": transcript_config.condition_on_previous_text,
                        "hallucination_silence_threshold": transcript_config.hallucination_silence_threshold,
                        "force_source_language": transcript_config.force_source_language,
                        "suppress_plain_ascii_tokens": transcript_config.suppress_plain_ascii_tokens,
                        "artifact_chaos_mode": transcript_config.artifact_chaos_mode,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            progress.update("Отправляю файлы", 96, 100, "SRT, TXT, meta JSON")
            for path in (srt_path, txt_path, meta_path):
                with path.open("rb") as file_obj:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=file_obj,
                        filename=path.name,
                        caption=(
                            "Сырой вывод Whisper "
                            f"({transcript_config.source_lang or 'auto'}, {transcript_config.whisper_task})"
                        ),
                        read_timeout=300,
                        write_timeout=300,
                        connect_timeout=60,
                        pool_timeout=60,
                    )
            _save_job_snapshot(job_dir, job, status="done")
            await _finish_progress(progress, progress_task, status_message, "Готово", detail="SRT, TXT и meta JSON отправлены")
            return

        progress = _ProgressState("Полноценный дубляж")
        progress.update(
            "В очереди",
            1,
            100,
            (
                f"вход={job.get('source_lang') or 'auto'}, "
                f"цель={target_lang}, "
                f"метод={_asr_method_label(job.get('asr_method') or settings.default_asr_method)}, "
                f"голоса={_speaker_count_label(job.get('speaker_count'))}, "
                f"TTS={_tts_method_label(tts_provider)}, "
                f"ASR={config.asr_backend} {config.whisper_model}, "
                f"pivot={config.input_pivot_lang or '-'}"
            ),
        )
        if status_message is not None:
            await _safe_edit_status(status_message, progress.render())
        else:
            status_message = await context.bot.send_message(chat_id=chat_id, text=progress.render())
        progress_task = asyncio.create_task(_progress_updater(status_message, progress))
        config.progress_callback = progress.update

        result = await asyncio.to_thread(run_dub, Path(job["input_path"]), config)
        send_path = result

        watermarked_path = job_dir / "dubbed_watermarked.mp4"
        progress.update("Добавляю водяной знак", 98, 100, None)
        await asyncio.to_thread(
            add_watermark,
            result,
            watermarked_path,
            text=settings.watermark_text,
            image_path=settings.watermark_image,
        )
        send_path = watermarked_path

        if _video_needs_telegram_compression(send_path):
            progress.update("Сжимаю видео для Telegram", 99, 100, _format_file_size(send_path.stat().st_size))
        else:
            progress.update("Отправляю видео", 99, 100, None)
        output_filename = _lalaschool_filename(job.get("source_title") or Path(job["input_path"]).stem, send_path.suffix)
        transcript_text = _read_transcript_text(job_dir / "work" / "translated.srt")
        _save_job_snapshot(job_dir, job, status="ready")
        await _send_video_file(
            context.bot,
            chat_id,
            send_path,
            output_filename,
        )
        if transcript_text:
            progress.update("Отправляю транскрипт", 99, 100, None)
            transcript_path = _write_transcript_text(
                job_dir,
                job.get("source_title") or Path(job["input_path"]).stem,
                transcript_text,
            )
            with transcript_path.open("rb") as file_obj:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=file_obj,
                    filename=transcript_path.name,
                    caption="Транскрипт",
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=60,
                    pool_timeout=60,
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Транскрипт не найден.",
                read_timeout=300,
                write_timeout=300,
                connect_timeout=60,
                pool_timeout=60,
            )
        final_detail = "Видео и транскрипт отправлены" if transcript_text else "Видео отправлено, транскрипт не найден"
        _save_job_snapshot(job_dir, job, status="done")
        await _finish_progress(progress, progress_task, status_message, "Готово", detail=final_detail)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        error_log = job_dir / "error.log"
        error_log.write_text(traceback_text, encoding="utf-8")
        _save_job_snapshot(job_dir, job, status="failed", error=details)
        await _finish_progress(progress, progress_task, status_message, "Ошибка", failed=True, detail=details)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Задача упала:\n{details}",
            reply_markup=_resume_keyboard(job_dir),
        )


async def _send_remote_worker_result(context: Any, item: _QueuedJob, manifest: dict[str, Any]) -> None:
    job = item.job
    job_dir = Path(str(job["job_dir"]))
    progress = item.progress
    progress_task = item.progress_task
    status_message = item.status_message
    try:
        if progress is not None:
            progress.update("Sending worker result", 96, 100, item.worker_id)
        mode = str(manifest.get("mode") or job.get("mode") or "dub")
        if mode == "raw_text":
            documents = manifest.get("documents") or []
            if not isinstance(documents, list) or not documents:
                raise RuntimeError("Worker completed raw_text job without documents.")
            for document in documents:
                if not isinstance(document, dict):
                    continue
                filename = str(document.get("filename") or "")
                path = _remote_result_file(job_dir, "documents", filename)
                output_filename = str(document.get("output_filename") or path.name)
                caption = str(document.get("caption") or "Raw Whisper output")
                with path.open("rb") as file_obj:
                    await context.bot.send_document(
                        chat_id=item.chat_id,
                        document=file_obj,
                        filename=output_filename,
                        caption=caption,
                        read_timeout=300,
                        write_timeout=300,
                        connect_timeout=60,
                        pool_timeout=60,
                    )
            _save_job_snapshot(job_dir, job, status="done")
            await _finish_progress(progress, progress_task, status_message, "Done", detail="Worker documents sent")
            return

        video_info = manifest.get("video")
        if not isinstance(video_info, dict):
            raise RuntimeError("Worker completed dub job without video.")
        video_path = _remote_result_file(job_dir, "video", str(video_info.get("filename") or ""))
        output_filename = str(
            manifest.get("output_filename")
            or _lalaschool_filename(job.get("source_title") or video_path.stem, video_path.suffix)
        )
        _save_job_snapshot(job_dir, job, status="ready")
        await _send_video_file(context.bot, item.chat_id, video_path, output_filename)

        transcript_sent = False
        transcript_info = manifest.get("transcript")
        if isinstance(transcript_info, dict):
            transcript_path = _remote_result_file(job_dir, "transcript", str(transcript_info.get("filename") or ""))
            if transcript_path.exists():
                transcript_filename = str(manifest.get("transcript_filename") or transcript_path.name)
                with transcript_path.open("rb") as file_obj:
                    await context.bot.send_document(
                        chat_id=item.chat_id,
                        document=file_obj,
                        filename=transcript_filename,
                        caption="Transcript",
                        read_timeout=300,
                        write_timeout=300,
                        connect_timeout=60,
                        pool_timeout=60,
                    )
                transcript_sent = True
        if not transcript_sent:
            await context.bot.send_message(
                chat_id=item.chat_id,
                text="Transcript not found.",
                read_timeout=300,
                write_timeout=300,
                connect_timeout=60,
                pool_timeout=60,
            )
        _save_job_snapshot(job_dir, job, status="done")
        detail = "Worker video and transcript sent" if transcript_sent else "Worker video sent, transcript not found"
        await _finish_progress(progress, progress_task, status_message, "Done", detail=detail)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        (job_dir / "error.log").write_text(traceback_text, encoding="utf-8")
        _save_job_snapshot(job_dir, job, status="failed", error=details)
        await _finish_progress(progress, progress_task, status_message, "Error", failed=True, detail=details)
        await context.bot.send_message(
            chat_id=item.chat_id,
            text=f"Task failed:\n{details}",
            reply_markup=_resume_keyboard(job_dir),
        )


def _remote_result_file(job_dir: Path, kind: str, filename: str) -> Path:
    safe_kind = _safe_upload_name(kind)
    safe_filename = _safe_upload_name(filename)
    if safe_kind not in {"video", "transcript", "documents", "error"} or not safe_filename:
        raise RuntimeError(f"Bad worker result path: {kind}/{filename}")
    path = job_dir / "remote_result" / safe_kind / safe_filename
    if not path.exists():
        raise RuntimeError(f"Worker result file is missing: {path}")
    return path


def _job_snapshot_path(job_dir: Path) -> Path:
    return job_dir / "job.json"


def _save_job_snapshot(job_dir: Path, job: dict[str, Any], *, status: str, error: str | None = None) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    data = dict(job)
    now = time.time()
    data["status"] = status
    data["updated_at"] = now
    if status in CLEANUP_JOB_STATUSES and not data.get("finished_at"):
        data["finished_at"] = now
    if error is not None:
        data["error"] = error
    _job_snapshot_path(job_dir).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_job_snapshot(job_dir: Path) -> dict[str, Any] | None:
    path = _job_snapshot_path(job_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    input_path = Path(str(data.get("input_path") or ""))
    if not input_path.exists():
        return None
    data["job_dir"] = str(job_dir)
    data["input_path"] = str(input_path)
    return data


def _find_latest_resumable_job(settings: BotSettings, user_id: int) -> dict[str, Any] | None:
    user_dir = settings.workdir / str(user_id)
    if not user_dir.exists():
        return None

    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in user_dir.glob("*/job.json"):
        job_dir = path.parent
        job = _load_job_snapshot(job_dir)
        if not job:
            continue
        if str(job.get("status") or "") == "done":
            continue
        updated_at = float(job.get("updated_at") or path.stat().st_mtime)
        candidates.append((updated_at, job))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _resume_keyboard(job_dir: Path) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Продолжить задачу", callback_data=f"resume:{job_dir.name}")]]
    )


def _remove_reply_keyboard() -> Any:
    from telegram import ReplyKeyboardRemove

    return ReplyKeyboardRemove()


def _language_keyboard(prefix: str, items: list[tuple[str, str]], columns: int = 2) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    rows = []
    for index in range(0, len(items), columns):
        row_items = items[index : index + columns]
        rows.append([InlineKeyboardButton(label, callback_data=f"{prefix}:{code}") for code, label in row_items])
    return InlineKeyboardMarkup(rows)


def _source_title_from_media(media: Any, fallback_id: int | str) -> str:
    file_name = getattr(media, "file_name", "") or ""
    if file_name:
        return Path(file_name).stem
    return f"telegram_video_{fallback_id}"


def _source_title_from_audio(media: Any, fallback_id: int | str) -> str:
    file_name = getattr(media, "file_name", "") or ""
    if file_name:
        return Path(file_name).stem
    title = getattr(media, "title", "") or ""
    if title:
        return str(title).strip()
    return f"telegram_audio_{fallback_id}"


def _source_title_from_download(job_dir: Path, input_path: Path, url: str) -> str:
    meta_path = job_dir / "download_meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        title = str(data.get("title") or "").strip()
        if title:
            return title

    stem = input_path.stem.strip()
    if stem and stem != "input":
        return stem
    host = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).split("/", 1)[0].strip()
    return host or "video"


def _lalaschool_filename(source_title: str, suffix: str) -> str:
    suffix = suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".mp4"
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", source_title)
    base = re.sub(r"\s+", " ", base).strip(" ._")
    if not base:
        base = "video"
    if "lalaschool" not in base.casefold():
        base = f"{base}_lalaschool"
    if len(base) > 120:
        suffix_text = "_lalaschool"
        base = base[: 120 - len(suffix_text)].rstrip(" ._") + suffix_text
    return f"{base}{suffix}"


def _read_transcript_text(srt_path: Path) -> str:
    if not srt_path.exists():
        return ""
    lines: list[str] = []
    for line in srt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        lines.append(stripped)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _write_transcript_text(job_dir: Path, source_title: str, transcript_text: str) -> Path:
    filename = _lalaschool_filename(f"{source_title}_transcript", ".txt")
    path = job_dir / filename
    path.write_text(transcript_text.strip() + "\n", encoding="utf-8")
    return path


def _video_needs_telegram_compression(video_path: Path) -> bool:
    try:
        return video_path.stat().st_size > TELEGRAM_SAFE_VIDEO_BYTES
    except OSError:
        return False


def _format_file_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _telegram_video_output_path(video_path: Path, suffix: str) -> Path:
    output_suffix = video_path.suffix if video_path.suffix.lower() == ".mp4" else ".mp4"
    return video_path.with_name(f"{video_path.stem}{suffix}{output_suffix}")


async def _telegram_sendable_video_path(
    video_path: Path,
    *,
    target_size_mb: float = 45.0,
    suffix: str = "_telegram",
    force: bool = False,
) -> Path:
    if not force and not _video_needs_telegram_compression(video_path):
        return video_path

    output_path = _telegram_video_output_path(video_path, suffix)
    source_mtime = video_path.stat().st_mtime
    if output_path.exists() and output_path.stat().st_size > 1024 and output_path.stat().st_mtime >= source_mtime:
        if force or output_path.stat().st_size <= TELEGRAM_SAFE_VIDEO_BYTES:
            return output_path

    await asyncio.to_thread(
        compress_video_for_telegram,
        video_path,
        output_path,
        target_size_mb=target_size_mb,
    )
    if output_path.stat().st_size > TELEGRAM_SAFE_VIDEO_BYTES and target_size_mb > 35.0:
        return await _telegram_sendable_video_path(
            video_path,
            target_size_mb=35.0,
            suffix="_telegram_small",
            force=True,
        )
    return output_path


def _is_request_entity_too_large(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return "request entity too large" in text or "entity too large" in text or "413" in text


async def _send_video_file(
    bot: Any,
    chat_id: int | str,
    video_path: Path,
    filename: str,
    *,
    caption: str | None = None,
    reply_markup: Any = None,
) -> None:
    send_path = await _telegram_sendable_video_path(video_path)
    try:
        await _send_video_file_once(bot, chat_id, send_path, filename, caption=caption, reply_markup=reply_markup)
    except Exception as exc:
        if not _is_request_entity_too_large(exc):
            raise
        retry_path = await _telegram_sendable_video_path(
            video_path,
            target_size_mb=35.0,
            suffix="_telegram_small",
            force=True,
        )
        await _send_video_file_once(bot, chat_id, retry_path, filename, caption=caption, reply_markup=reply_markup)


async def _send_video_file_once(
    bot: Any,
    chat_id: int | str,
    video_path: Path,
    filename: str,
    *,
    caption: str | None = None,
    reply_markup: Any = None,
) -> None:
    with video_path.open("rb") as file_obj:
        await bot.send_video(
            chat_id=chat_id,
            video=file_obj,
            filename=filename,
            caption=caption,
            reply_markup=reply_markup,
            supports_streaming=True,
            read_timeout=300,
            write_timeout=300,
            connect_timeout=60,
            pool_timeout=60,
        )


def _guess_suffix(media: Any) -> str:
    file_name = getattr(media, "file_name", "") or ""
    suffix = Path(file_name).suffix.lower()
    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        return suffix
    mime_type = getattr(media, "mime_type", "") or ""
    if "quicktime" in mime_type:
        return ".mov"
    if "matroska" in mime_type:
        return ".mkv"
    if "webm" in mime_type:
        return ".webm"
    return ".mp4"


def _guess_audio_suffix(media: Any) -> str:
    file_name = getattr(media, "file_name", "") or ""
    suffix = Path(file_name).suffix.lower()
    if suffix in {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".oga", ".opus", ".flac", ".webm", ".mp4"}:
        return suffix
    mime_type = (getattr(media, "mime_type", "") or "").lower()
    if "mpeg" in mime_type or "mp3" in mime_type:
        return ".mp3"
    if "mp4" in mime_type or "m4a" in mime_type:
        return ".m4a"
    if "wav" in mime_type:
        return ".wav"
    if "flac" in mime_type:
        return ".flac"
    if "webm" in mime_type:
        return ".webm"
    if "ogg" in mime_type or "opus" in mime_type:
        return ".ogg"
    return ".ogg"


if __name__ == "__main__":
    main()
