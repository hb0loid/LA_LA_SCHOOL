from __future__ import annotations

import asyncio
import contextlib
import heapq
import json
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .bot_config import BotSettings, load_bot_settings
from .download import download_video_url, extract_url
from .ffmpeg import probe_duration, trim_video
from .models import DubConfig
from .pipeline import run_dub, run_transcript
from .watermark import add_watermark


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
    ("hi", "Хинди"),
    ("id", "Индонезийский"),
    ("pl", "Польский"),
    ("uk", "Украинский"),
]

ASR_METHODS = [
    ("ow-large-v3-chaos-backbone", "Поломанный дубляж"),
    ("ow-large-v3-raw-dub", "Сырой forced дубляж"),
]

ASR_METHOD_CONFIGS = {
    "ow-large-v3-hunt": ("openai-whisper", "large-v3", False),
    "ow-large-v3-chaos-backbone": ("openai-whisper", "large-v3", False),
    "ow-large-v3-raw-dub": ("openai-whisper", "large-v3", True),
    "ow-large-v3-chaos": ("openai-whisper", "large-v3", True),
    "ow-large-v3-soft": ("openai-whisper", "large-v3", False),
    "ow-large-v3-forced": ("openai-whisper", "large-v3", True),
    "fw-large-v3-soft": ("faster-whisper", "large-v3", False),
    "fw-large-v3-forced": ("faster-whisper", "large-v3", True),
    "ow-large-v3-turbo-soft": ("openai-whisper", "large-v3-turbo", False),
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


def main() -> None:
    try:
        from telegram import Update
        from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters
    except ImportError as exc:
        raise RuntimeError("Install bot dependencies first: python -m pip install -e .[bot]") from exc

    settings = load_bot_settings()
    settings.workdir.mkdir(parents=True, exist_ok=True)
    print(
        "La La Dub Bot starting "
        f"workdir={settings.workdir} "
        f"translator={settings.translator} "
        f"tts={settings.tts} "
        f"f5={settings.f5_model}/{settings.f5_device} "
        f"multi_speaker={settings.multi_speaker} "
        f"speaker_clustering={settings.speaker_clustering}/{settings.max_speaker_clusters} "
        f"separation={settings.separation} "
        f"audio_bed={settings.audio_bed} "
        f"watermark={settings.watermark_image} "
        f"paid_users={len(settings.paid_users)} "
        f"duration_limits=free:{settings.free_max_duration_seconds}/paid:{settings.paid_max_duration_seconds} "
        f"collapse_repetitions={settings.collapse_repetitions}/"
        f"{settings.max_phrase_repeats}/{settings.max_word_repeats} "
        f"inject_artifacts={settings.inject_artifacts}/"
        f"{settings.artifact_max_segments}/{settings.artifact_min_gap_seconds} "
        f"distort={settings.distort_translation}/{settings.translation_pivots} "
        f"asr={settings.asr_backend} "
        f"queue={settings.max_active_jobs}/{settings.max_active_jobs_per_user} "
        f"default_asr_method={settings.default_asr_method} "
        f"whisper={settings.whisper_model}/{settings.whisper_device}/{settings.whisper_compute_type} "
        f"whisper_only={settings.whisper_only_model}/{settings.whisper_only_device} "
        f"suppress_ascii={settings.suppress_plain_ascii_tokens}",
        flush=True,
    )

    application = Application.builder().token(settings.token).post_init(_setup_bot_commands).build()
    application.bot_data["settings"] = settings
    application.bot_data["job_scheduler"] = _JobScheduler(settings)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("me", me))
    application.add_handler(CommandHandler("resume", resume))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, receive_video))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, receive_link))
    application.add_handler(CallbackQueryHandler(select_source, pattern=r"^src:"))
    application.add_handler(CallbackQueryHandler(select_asr_method, pattern=r"^asr:"))
    application.add_handler(CallbackQueryHandler(resume_callback, pattern=r"^resume:"))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


async def _setup_bot_commands(application: Any) -> None:
    await application.bot.set_my_commands(
        [
            ("start", "Инструкция"),
            ("resume", "Продолжить последнюю задачу"),
            ("me", "Показать Telegram ID"),
            ("cancel", "Сбросить текущую задачу"),
        ]
    )


async def start(update: Any, context: Any) -> None:
    await update.effective_message.reply_text(
        "Пришли видео. Я попрошу выбрать input-язык и метод извлечения текста. Язык озвучки всегда русский.\n\n"
        "Soft-методы дают Whisper самому определить исходник, а выбранный input-язык используется как промежуточный перевод. "
        "Forced-методы насильно задают выбранный input-язык самому Whisper.\n"
        "У бесплатных пользователей лимит видео 3 минуты и на видео будет водяной знак."
    )


async def me(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    status = "оплачен" if settings.is_paid(user.id if user else None) else "бесплатный"
    await update.effective_message.reply_text(f"Твой Telegram ID: {user.id}\nСтатус: {status}")


async def cancel(update: Any, context: Any) -> None:
    context.user_data.clear()
    await update.effective_message.reply_text("Отменил. Можешь прислать другое видео.")


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
    if file_size > settings.max_file_mb * 1024 * 1024:
        await message.reply_text(f"Видео слишком большое. Лимит: {settings.max_file_mb} МБ.")
        return

    user_id = update.effective_user.id

    job_dir = settings.workdir / str(user_id) / str(message.message_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = _guess_suffix(media)
    input_path = job_dir / f"input{suffix}"
    source_title = _source_title_from_media(media, message.message_id)

    status = await message.reply_text("Скачиваю видео...")
    tg_file = await media.get_file()
    await tg_file.download_to_drive(custom_path=str(input_path))
    input_path = await _prepare_input_video_duration(status, settings, user_id, input_path)
    if input_path is None:
        return

    await _remember_job_and_ask_source(context, status, job_dir, input_path, source_title)


async def receive_link(update: Any, context: Any) -> None:
    message = update.effective_message
    settings: BotSettings = context.application.bot_data["settings"]
    text = message.text or ""
    url = extract_url(text)
    if not url:
        await message.reply_text("Пришли видеофайл или ссылку на видео, например YouTube.")
        return

    user_id = update.effective_user.id
    job_dir = settings.workdir / str(user_id) / str(message.message_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    status = await message.reply_text("Скачиваю видео по ссылке...")
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

    await _remember_job_and_ask_source(context, status, job_dir, input_path, source_title)


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


async def _remember_job_and_ask_source(
    context: Any,
    status: Any,
    job_dir: Path,
    input_path: Path,
    source_title: str,
) -> None:
    context.user_data["job"] = {
        "job_dir": str(job_dir),
        "input_path": str(input_path),
        "source_title": source_title,
    }
    _save_job_snapshot(job_dir, context.user_data["job"], status="select_source")
    await status.edit_text(
        "Выбери input-язык. Для soft-методов это будет промежуточный язык перевода; "
        "для forced-методов это будет язык, насильно заданный Whisper."
    )
    await status.edit_reply_markup(reply_markup=_language_keyboard("src", SOURCE_LANGS))


async def select_source(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    source_lang = query.data.split(":", 1)[1]
    job["source_lang"] = None if source_lang == "auto" else source_lang
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_method")
    await query.edit_message_text(
        "Выбери метод извлечения текста.",
        reply_markup=_language_keyboard("asr", ASR_METHODS, columns=1),
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
    job["target_lang"] = "ru"
    _save_job_snapshot(Path(job["job_dir"]), job, status="queued")
    await query.edit_message_text("Ставлю задачу в очередь. Язык озвучки: русский.")
    context.user_data.pop("job", None)
    await _enqueue_job(update, context, job, query.message)


async def _enqueue_job(update: Any, context: Any, job: dict[str, Any], status_message: Any) -> None:
    scheduler: _JobScheduler = context.application.bot_data["job_scheduler"]
    settings: BotSettings = context.application.bot_data["settings"]
    job["target_lang"] = "ru"
    chat = update.effective_chat
    user = update.effective_user
    if chat is None:
        await _safe_edit_status(status_message, "Не удалось определить чат для задачи.")
        return

    input_path = Path(str(job.get("input_path") or ""))
    if input_path.exists():
        prepared_input_path = await _prepare_input_video_duration(
            status_message,
            settings,
            user.id if user else None,
            input_path,
        )
        if prepared_input_path is None:
            _save_job_snapshot(Path(job["job_dir"]), job, status="rejected", error="duration_limit")
            return
        if prepared_input_path != input_path:
            job["input_path"] = str(prepared_input_path)
            _save_job_snapshot(Path(job["job_dir"]), job, status="queued", trimmed=True)

    await scheduler.enqueue(
        context,
        chat_id=chat.id,
        user_id=user.id if user else None,
        job=job,
        status_message=status_message,
    )


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


class _JobScheduler:
    def __init__(self, settings: BotSettings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._pending: list[tuple[int, int, _QueuedJob]] = []
        self._active_total = 0
        self._active_by_user: dict[int, int] = {}
        self._known_jobs: set[str] = set()
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
            if item.user_id is not None:
                current = self._active_by_user.get(item.user_id, 0) - 1
                if current > 0:
                    self._active_by_user[item.user_id] = current
                else:
                    self._active_by_user.pop(item.user_id, None)
            self._known_jobs.discard(item.key)
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()

    async def _dispatch_locked(self, context: Any) -> None:
        while self._active_total < self._settings.max_active_jobs:
            index = self._next_startable_index()
            if index is None:
                return
            _, _, item = self._pending.pop(index)
            heapq.heapify(self._pending)
            self._active_total += 1
            if item.user_id is not None:
                self._active_by_user[item.user_id] = self._active_by_user.get(item.user_id, 0) + 1
            item.job["started_at"] = time.time()
            _save_job_snapshot(Path(item.job["job_dir"]), item.job, status="starting")
            context.application.create_task(self._run_item(context, item))

    def _next_startable_index(self) -> int | None:
        best_index: int | None = None
        best_key: tuple[int, int] | None = None
        for index, (priority, sequence, item) in enumerate(self._pending):
            if not self._can_start(item):
                continue
            key = (priority, sequence)
            if best_key is None or key < best_key:
                best_index = index
                best_key = key
        return best_index

    def _can_start(self, item: _QueuedJob) -> bool:
        if self._active_total >= self._settings.max_active_jobs:
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
        active_for_user = self._active_by_user.get(item.user_id, 0) if item.user_id is not None else 0
        tier = "премиум" if item.premium else "обычный"
        return "\n".join(
            [
                f"{title}: В очереди",
                f"Позиция: {position}",
                f"Сейчас выполняется: {self._active_total}/{self._settings.max_active_jobs}",
                f"У тебя выполняется: {active_for_user}/{self._settings.max_active_jobs_per_user}",
                f"Приоритет: {tier}",
            ]
        )


def _job_queue_key(job: dict[str, Any]) -> str:
    return str(Path(str(job["job_dir"])).resolve())


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
    try:
        await message.edit_text(text)
    except Exception as exc:
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
    _save_job_snapshot(job_dir, job, status="running")
    progress: _ProgressState | None = None
    progress_task: asyncio.Task[Any] | None = None

    config = DubConfig(
        output=output_path,
        workdir=job_dir / "work",
        source_lang=None,
        target_lang="ru",
        asr_backend=settings.asr_backend,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
        whisper_compute_type=settings.whisper_compute_type,
        translator=settings.translator,
        tts=settings.tts,
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
        f5_device=settings.f5_device,
        f5_speed=settings.f5_speed,
        f5_nfe_step=settings.f5_nfe_step,
        f5_cfg_strength=settings.f5_cfg_strength,
        f5_target_rms=settings.f5_target_rms,
        f5_cross_fade_duration=settings.f5_cross_fade_duration,
        f5_remove_silence=settings.f5_remove_silence,
        f5_timeout_seconds=settings.f5_timeout_seconds,
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
        collapse_repetitions=settings.collapse_repetitions,
        max_phrase_repeats=settings.max_phrase_repeats,
        max_word_repeats=settings.max_word_repeats,
    )
    _apply_text_extraction_method(config, job, settings)

    try:
        if job.get("mode") == "raw_text":
            transcript_config = DubConfig(
                output=output_path,
                workdir=job_dir / "work",
                source_lang=job.get("source_lang") or None,
                target_lang="ru",
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
                "цель=ru, "
                f"метод={_asr_method_label(job.get('asr_method') or settings.default_asr_method)}, "
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

        if not settings.is_paid(user_id):
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


def _job_snapshot_path(job_dir: Path) -> Path:
    return job_dir / "job.json"


def _save_job_snapshot(job_dir: Path, job: dict[str, Any], *, status: str, error: str | None = None) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    data = dict(job)
    data["status"] = status
    data["updated_at"] = time.time()
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


async def _send_video_file(
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


if __name__ == "__main__":
    main()
