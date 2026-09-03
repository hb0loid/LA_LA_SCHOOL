from __future__ import annotations

import asyncio
import contextlib
import hashlib
import heapq
import html
import json
import math
import multiprocessing
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .bot_config import BotSettings, load_bot_settings
from .download import download_video_url, extract_url, has_video_and_audio
from .ffmpeg import (
    compress_video_for_telegram,
    make_audio_visual_video,
    probe_duration,
    probe_video_dimensions,
    trim_video,
)
from .karma import KARMA_SCALE, PREMIUM_LEVEL, KarmaLevel, level_for_karma, next_level_for_karma, visible_karma
from .karma_command import karma_command
from .asr import clear_openai_whisper_cache
from .library import LibraryStore, show_command
from .models import DubConfig
from .pipeline import run_dub, run_transcript
from .performance import PerformanceHistory, job_duration_seconds, merge_stage_seconds, record_terminal_job
from .premium_store import PremiumStore, Subscription, UserSettings
from .preset_store import PRESET_FIELDS, PresetStore, UserPreset
from .text_review import TextReviewStore
from .worker_watch import WorkerPresence
from .update_dedupe import (
    UpdateDeduplicator,
    build_completion_marker,
    build_replay_guard,
)
from .proposal_store import ProposalStore
from .tts import clear_tts_model_caches
from .watermark import add_watermark


def _force_current_python_for_child_processes() -> None:
    multiprocessing.set_executable(sys.executable)
    if hasattr(sys, "_base_executable"):
        sys._base_executable = sys.executable


_force_current_python_for_child_processes()


# Ordered by how often people actually pick each one, measured over 3505 jobs:
# Vietnamese alone is 48%, and the first eight cover 90%. Everything past that
# is there so a rare video is not turned away - Argos has a round trip through
# English for each, and Whisper can transcribe all of them.
from .languages import SOURCE_LANGS, TARGET_LANGS, transcript_header  # noqa: E402

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

TTS_METHODS = [
    ("moss", "MOSS — лучше качество, дольше ждать"),
    ("cosyvoice", "CosyVoice — быстрее, но попроще"),
    ("f5", "F5 (для украинского)"),
    ("qwen3", "Qwen3 (нестабильный, не предлагается)"),
]
# What's actually offered to the user: qwen3 hung mid-batch during testing (a
# reference-voice-dependent stall, not a fluke of the run) and f5 is Ukrainian's
# only compatible engine, auto-picked without asking - see select_target_lang.
# CosyVoice is hidden for now - MOSS got fast enough that the trade-off it
# offered (quicker, rougher) no longer buys anything. Re-add its tuple here to
# bring it back; the engine itself is untouched and still works.
TTS_METHOD_CHOICES = [("moss", "MOSS — лучше качество, дольше ждать")]
_HIDDEN_TTS_CHOICES = [("cosyvoice", "CosyVoice — быстрее, но попроще")]

# The last step of the setup wizard: озвучка занимает больше всего времени, so
# the text can be shown first and voiced only if it is worth voicing.
REVIEW_MODE_OPTIONS = [
    ("direct", "🎬 Сразу дубляж"),
    ("review", "📝 Сначала показать текст"),
]
# Hidden for now: with MOSS this fast, the extra confirmation step costs more
# attention than the voicing it saves. Flip to True to offer it again - the
# whole review flow (buttons, retries, stats) stays in place either way.
REVIEW_STEP_OFFERED = False
# Each rejected variant costs a full ASR+translation pass, so retries are capped.
MAX_TEXT_REVIEW_ATTEMPTS = 3

TELEGRAM_SAFE_VIDEO_BYTES = 45 * 1024 * 1024
TELEGRAM_DIRECT_DOWNLOAD_SAFE_BYTES = 20 * 1024 * 1024
RECOVERABLE_JOB_STATUSES = {"starting", "running", "queued", "ready"}
CLEANUP_JOB_STATUSES = {"done", "failed", "rejected"}
_LAST_STATUS_TEXT: dict[tuple[int | None, int | None], str] = {}
_TELEGRAM_EDIT_BACKOFF_UNTIL = 0.0
_TELEGRAM_EDIT_BACKOFF_LOGGED_UNTIL = 0.0
MAINTENANCE_MESSAGE = (
    "Сейчас ведутся технические работы над ботом. "
    "Попробуй снова немного позже."
)



class _ApplicationContext:
    def __init__(self, application: Any) -> None:
        self.application = application
        self.bot = application.bot


def main() -> None:
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CallbackQueryHandler,
            CommandHandler,
            MessageHandler,
            PreCheckoutQueryHandler,
            TypeHandler,
            filters,
        )
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
        f"cosyvoice={settings.cosyvoice_model_id}/{settings.cosyvoice_device}/{settings.cosyvoice_mode} "
        f"moss={settings.moss_model_dir}/{settings.moss_device} "
        f"multi_speaker={settings.multi_speaker} "
        f"speaker_clustering={settings.speaker_clustering}/{settings.max_speaker_clusters} "
        f"diarization={settings.diarization_model}/{settings.diarization_device} "
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
        f"paid_users={len(settings.paid_users)}/{settings.paid_users_file} "
        f"admin_users={len(settings.admin_users)}/{settings.admin_users_file} "
        f"proposal={settings.proposal_enabled}/{settings.proposal_db} "
        "duration_limits=daily-karma/no-per-video "
        f"collapse_repetitions={settings.collapse_repetitions}/"
        f"{settings.max_phrase_repeats}/{settings.max_word_repeats} "
        f"inject_artifacts={settings.inject_artifacts}/"
        f"{settings.artifact_max_segments}/{settings.artifact_ratio}/"
        f"{settings.artifact_min_source_segments}/{settings.artifact_min_gap_seconds} "
        f"distort={settings.distort_translation}/{settings.translation_pivots}/pass2={settings.translation_second_pass_ratio} "
        f"asr={settings.asr_backend} "
        f"queue={settings.max_active_jobs}/{settings.max_active_jobs_per_user} "
        f"job_retention={settings.job_retention_seconds}s "
        f"default_asr_method={settings.default_asr_method} "
        f"whisper={settings.whisper_model}/{settings.whisper_device}/{settings.whisper_compute_type} "
        f"whisper_only={settings.whisper_only_model}/{settings.whisper_only_device} "
        f"artifact_whisper={settings.whisper_only_model}/{settings.artifact_whisper_device} "
        f"maintenance={_maintenance_enabled(settings)} "
        f"suppress_ascii={settings.suppress_plain_ascii_tokens}",
        flush=True,
    )

    application = Application.builder().token(settings.token).post_init(_setup_bot_commands).build()
    application.bot_data["settings"] = settings
    application.bot_data["job_scheduler"] = _JobScheduler(settings)
    application.bot_data["proposal_store"] = ProposalStore(settings.proposal_db)
    application.bot_data["premium_store"] = PremiumStore(settings.premium_db)
    application.bot_data["preset_store"] = PresetStore(settings.preset_db)
    application.bot_data["library_store"] = LibraryStore(settings.library_db)
    application.bot_data["review_store"] = TextReviewStore(settings.review_db)
    private_chat = filters.ChatType.PRIVATE
    application.add_error_handler(_telegram_error_handler)
    # Before every other handler: an update Telegram redelivered after a hard
    # restart must not run its side effects a second time.
    update_dedupe = UpdateDeduplicator(settings.workdir / "last_update.json")
    application.add_handler(
        TypeHandler(Update, build_replay_guard(update_dedupe)),
        group=-100,
    )
    application.add_handler(MessageHandler(private_chat, maintenance_message_gate), group=-1)
    application.add_handler(CallbackQueryHandler(maintenance_callback_gate), group=-1)
    application.add_handler(CommandHandler("start", start, filters=private_chat))
    application.add_handler(CommandHandler("me", me, filters=private_chat))
    application.add_handler(CommandHandler("karma", karma_command))
    application.add_handler(CommandHandler("queue", queue_status, filters=private_chat))
    application.add_handler(CommandHandler("resume", resume, filters=private_chat))
    application.add_handler(CommandHandler("send", send_to_proposal, filters=private_chat))
    application.add_handler(CommandHandler("censored", censored, filters=private_chat))
    application.add_handler(CommandHandler("cancel", cancel, filters=private_chat))
    application.add_handler(CommandHandler("maintenance", maintenance, filters=private_chat))
    application.add_handler(CommandHandler("premium", premium_command, filters=private_chat))
    application.add_handler(CommandHandler("paysupport", paysupport_command, filters=private_chat))
    application.add_handler(CommandHandler("grant_premium", admin_grant_premium, filters=private_chat))
    application.add_handler(CommandHandler("revoke_premium", admin_revoke_premium, filters=private_chat))
    application.add_handler(CommandHandler("refund_premium", admin_refund_premium, filters=private_chat))
    application.add_handler(CommandHandler("prem_owners", admin_prem_owners, filters=private_chat))
    application.add_handler(CommandHandler("reviews", admin_reviews, filters=private_chat))
    application.add_handler(CommandHandler("starbalance", admin_star_balance, filters=private_chat))
    application.add_handler(CommandHandler("watermark", watermark_command, filters=private_chat))
    application.add_handler(CommandHandler("mycensor", mycensor_command, filters=private_chat))
    application.add_handler(CommandHandler("preset", preset_command, filters=private_chat))
    application.add_handler(CommandHandler("show", show_command))
    application.add_handler(CallbackQueryHandler(premium_buy_callback, pattern=r"^premium_buy$"))
    application.add_handler(PreCheckoutQueryHandler(premium_precheckout))
    application.add_handler(MessageHandler(private_chat & filters.SUCCESSFUL_PAYMENT, premium_successful_payment))
    application.add_handler(MessageHandler(private_chat & (filters.VIDEO | filters.Document.VIDEO), receive_video))
    application.add_handler(
        MessageHandler(private_chat & (filters.AUDIO | filters.VOICE | filters.Document.AUDIO), receive_audio)
    )
    application.add_handler(MessageHandler(private_chat & filters.TEXT & ~filters.COMMAND, receive_link))
    application.add_handler(CallbackQueryHandler(selection_back, pattern=r"^back:"))
    application.add_handler(CallbackQueryHandler(select_visual_mode, pattern=r"^vis:"))
    application.add_handler(CallbackQueryHandler(select_source, pattern=r"^src:"))
    application.add_handler(CallbackQueryHandler(select_asr_method, pattern=r"^asr:"))
    application.add_handler(CallbackQueryHandler(select_speaker_count, pattern=r"^spk:"))
    application.add_handler(CallbackQueryHandler(select_target_lang, pattern=r"^tgt:"))
    application.add_handler(CallbackQueryHandler(select_tts_method, pattern=r"^tts:"))
    application.add_handler(CallbackQueryHandler(select_review_mode, pattern=r"^rev:"))
    application.add_handler(CallbackQueryHandler(text_review_callback, pattern=r"^rv:"))
    application.add_handler(CallbackQueryHandler(preset_wizard_callback, pattern=r"^pset:"))
    application.add_handler(CallbackQueryHandler(resume_callback, pattern=r"^resume:"))
    application.add_handler(CallbackQueryHandler(proposal_callback, pattern=r"^proposal:"))
    # Last group of all: reached only once every other handler has finished,
    # which is what lets the update count as handled and its replay be refused.
    application.add_handler(
        TypeHandler(Update, build_completion_marker(update_dedupe)),
        group=1000,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=False)


def _pipeline_process_entry(
    operation: str,
    input_path: str,
    config: DubConfig,
    result_connection: Any,
    progress_connection: Any,
) -> None:
    def report(stage: str, current: int | None, total: int | None, detail: str | None) -> None:
        progress_connection.send((stage, current, total, detail))

    config.progress_callback = report
    try:
        if operation == "dub":
            result = run_dub(Path(input_path), config)
            result_connection.send({"ok": True, "paths": [str(result)]})
        elif operation == "transcript":
            srt_path, txt_path = run_transcript(Path(input_path), config)
            result_connection.send({"ok": True, "paths": [str(srt_path), str(txt_path)]})
        else:
            raise ValueError(f"Unknown pipeline operation: {operation}")
    except BaseException:
        result_connection.send({"ok": False, "error": traceback.format_exc()})
    finally:
        result_connection.close()
        progress_connection.close()


def _terminate_pipeline_process(process: multiprocessing.Process) -> None:
    if process.pid is None:
        return
    if sys.platform == "win32":
        # A venv Python launcher can create another Python process. Terminating only
        # the launcher leaves the expensive pipeline running as an orphan.
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    elif process.is_alive():
        process.terminate()


async def _run_pipeline_isolated(
    operation: str,
    input_path: Path,
    config: DubConfig,
    progress: _ProgressState,
) -> list[Path]:
    config.progress_callback = None
    process_context = multiprocessing.get_context("spawn")
    result_parent, result_child = process_context.Pipe(duplex=False)
    progress_parent, progress_child = process_context.Pipe(duplex=False)
    process = process_context.Process(
        target=_pipeline_process_entry,
        args=(operation, str(input_path), config, result_child, progress_child),
        name=f"laladub-{operation}-{input_path.stem}",
    )
    process.start()
    result_child.close()
    progress_child.close()
    started_at = time.monotonic()
    completed = False

    def drain_progress() -> None:
        while True:
            try:
                if not progress_parent.poll():
                    break
                stage, current, total, detail = progress_parent.recv()
            except (EOFError, OSError):
                break
            progress.update(stage, current, total, detail)

    try:
        while process.is_alive():
            if time.monotonic() - started_at > 8 * 60 * 60:
                raise TimeoutError(f"Pipeline process exceeded 8 hours: {operation} {input_path}")
            drain_progress()
            await asyncio.sleep(0.2)

        process.join(timeout=2)
        drain_progress()

        if not result_parent.poll(2):
            raise RuntimeError(f"Pipeline process exited with code {process.exitcode} without a result")
        payload = result_parent.recv()
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or "Pipeline process failed"))
        completed = True
        return [Path(value) for value in payload.get("paths") or []]
    finally:
        if not completed and process.is_alive():
            _terminate_pipeline_process(process)
            process.join(timeout=5)
        result_parent.close()
        progress_parent.close()


async def _telegram_error_handler(update: Any, context: Any) -> None:
    try:
        from telegram.error import Forbidden
    except Exception:
        Forbidden = None  # type: ignore[assignment]

    error = getattr(context, "error", None)
    if Forbidden is not None and isinstance(error, Forbidden):
        user_id = getattr(getattr(update, "effective_user", None), "id", None) if update is not None else None
        print(f"Telegram send skipped: bot blocked by user {user_id or '?'}", flush=True)
        return
    print("Unhandled Telegram update error:", flush=True)
    if error is not None:
        print("".join(traceback.format_exception(type(error), error, error.__traceback__)), flush=True)


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
    return _maintenance_enabled(settings) and not settings.is_admin(user_id)


def _censor_settings_path(settings: BotSettings) -> Path:
    return settings.workdir / "censored_settings.json"


def _load_censor_settings(settings: BotSettings) -> dict[str, int]:
    path = _censor_settings_path(settings)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = max(0, min(100, int(value)))
        except (TypeError, ValueError):
            continue
    return result


def _save_censor_settings(settings: BotSettings, data: dict[str, int]) -> None:
    path = _censor_settings_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _get_censor_percent(settings: BotSettings) -> int:
    data = _load_censor_settings(settings)
    if "global" in data:
        return data["global"]
    legacy_values = [value for key, value in data.items() if key.isdigit()]
    if legacy_values:
        return max(legacy_values)
    # Never explicitly configured - default to fully on rather than off.
    return 100


def _set_censor_percent(settings: BotSettings, percent: int) -> None:
    data = {key: value for key, value in _load_censor_settings(settings).items() if not key.isdigit()}
    # Store the value explicitly (including 0) so an admin can actually turn
    # it off - dropping the key on 0 would make it indistinguishable from
    # "never configured", which now defaults to 100, not 0.
    data["global"] = max(0, min(100, int(percent)))
    _save_censor_settings(settings, data)


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
        ("resume", "Продолжить задачу по номеру"),
        ("send", "Отправить старую работу в предложку"),
        ("me", "Показать профиль и карму"),
        ("karma", "Своя карма, /karma all — таблица лидеров"),
        ("cancel", "Сбросить текущую задачу"),
        ("preset", "Настроить пресет выбора"),
        ("show", "Показать готовую работу из библиотеки"),
        ("premium", "Оформить премиум"),
        ("paysupport", "Поддержка по платежам"),
        ("watermark", "Премиум: вкл/выкл водяной знак"),
        ("mycensor", "Премиум: свой уровень censor"),
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
    asyncio.create_task(_worker_presence_loop(application))
    asyncio.create_task(_maintenance_watch_loop(application))
    asyncio.create_task(_daily_quota_reminder_loop(application))
    asyncio.create_task(application.bot_data["job_scheduler"].watch_remote_leases(_ApplicationContext(application)))
    if settings.proposal_enabled:
        asyncio.create_task(_proposal_outbox_loop(application))


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


async def _daily_quota_reminder_loop(application: Any) -> None:
    """The daily quota is a rolling 24h window, so there's no shared reset moment
    to announce - instead, poll everyone with recent usage and DM whoever's usage
    just dropped back below their limit (i.e. they were fully capped and now aren't)."""
    settings: BotSettings = application.bot_data["settings"]
    capped_users: set[int] = set()
    while True:
        await asyncio.sleep(180.0)
        try:
            store: ProposalStore | None = application.bot_data.get("proposal_store")
            premium_store: PremiumStore | None = application.bot_data.get("premium_store")
            if store is None:
                continue
            now = time.time()
            candidates = await asyncio.to_thread(store.users_with_recent_usage, now - 86400.0)
            still_capped: set[int] = set()
            for user_id in candidates:
                if settings.is_paid(user_id) or settings.is_admin(user_id):
                    continue
                karma_milli = await asyncio.to_thread(store.karma_total, user_id)
                level, _subscription = await asyncio.to_thread(_effective_level, premium_store, user_id, karma_milli)
                limit_ms = level.daily_minutes * 60_000
                used_ms = await asyncio.to_thread(store.daily_usage_ms, user_id, now)
                if used_ms >= limit_ms:
                    still_capped.add(user_id)
                elif user_id in capped_users:
                    with contextlib.suppress(Exception):
                        await application.bot.send_message(
                            chat_id=user_id,
                            text="✅ Суточный лимит на видео снова доступен, можно присылать новые работы.",
                        )
            capped_users = still_capped
        except Exception:
            print("Daily quota reminder loop failed:\n" + traceback.format_exc(), flush=True)


async def _proposal_outbox_loop(application: Any) -> None:
    store: ProposalStore = application.bot_data["proposal_store"]
    while True:
        try:
            messages = await asyncio.to_thread(store.pending_author_messages, limit=20)
            for item in messages:
                error = None
                try:
                    await application.bot.send_message(
                        chat_id=int(item["user_id"]),
                        text=(
                            f"Сообщение от модерации по работе №{item['job_number']}:\n\n"
                            f"{item['text']}"
                        ),
                        read_timeout=60,
                        write_timeout=60,
                        connect_timeout=30,
                        pool_timeout=30,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"Proposal author message failed: {error}", flush=True)
                await asyncio.to_thread(store.finish_author_message, int(item["id"]), error=error)
        except Exception:
            print("Proposal outbox loop failed:\n" + traceback.format_exc(), flush=True)
        await asyncio.sleep(3.0)


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

    query = update.callback_query
    if query is not None and str(query.data or "").startswith("proposal:"):
        return

    from telegram.ext import ApplicationHandlerStop

    context.user_data.clear()
    if query is not None:
        await query.answer("Ведутся технические работы. Попробуй позже.", show_alert=True)
        with contextlib.suppress(Exception):
            await query.edit_message_text(MAINTENANCE_MESSAGE)
    raise ApplicationHandlerStop


async def maintenance(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None or not settings.is_admin(user.id):
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
        "Обычные пользователи временно заблокированы. Текущие задачи прерваны, "
        "авторы могут продолжить их через /resume НОМЕР."
        if enabled
        else "Приём новых задач восстановлен, сохранённая очередь продолжит работу."
    )
    await update.effective_message.reply_text(f"Режим технических работ {state}.\n{details}")


async def premium_command(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    if user is None:
        return
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    if settings.is_paid(user.id) or settings.is_admin(user.id):
        await update.effective_message.reply_text(
            "У тебя уже безлимитный доступ, платная подписка не нужна.",
            reply_markup=_remove_reply_keyboard(),
        )
        return

    subscription = (
        await asyncio.to_thread(premium_store.active_subscription, user.id) if premium_store is not None else None
    )
    lines = [
        "⭐ Премиум подписка",
        f"{PREMIUM_LEVEL.daily_minutes} минут в сутки, приоритет в очереди +{PREMIUM_LEVEL.priority_bonus}, "
        f"до {PREMIUM_LEVEL.queue_limit} задач одновременно.",
        f"Цена: {settings.premium_price_stars} ⭐ за {settings.premium_days} дней.",
    ]
    if subscription is not None:
        expiry = datetime.fromtimestamp(subscription.expires_at).strftime("%d.%m.%Y")
        lines.extend(["", f"Уже активна, действует до {expiry}.", "Оплата продлит её ещё на срок подписки."])
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"Оформить премиум за {settings.premium_price_stars} ⭐", callback_data="premium_buy")]]
    )
    await update.effective_message.reply_text("\n".join(lines), reply_markup=keyboard)


async def premium_buy_callback(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    query = update.callback_query
    user = update.effective_user
    if query is None or user is None:
        return
    await query.answer()
    from telegram import LabeledPrice

    await context.bot.send_invoice(
        chat_id=user.id,
        title="Премиум La La School",
        description=(
            f"{PREMIUM_LEVEL.daily_minutes} минут в сутки, приоритет в очереди +{PREMIUM_LEVEL.priority_bonus}, "
            f"до {PREMIUM_LEVEL.queue_limit} задач одновременно, на {settings.premium_days} дней."
        ),
        payload=f"premium:{user.id}",
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(f"Премиум {settings.premium_days} дней", settings.premium_price_stars)],
    )


async def premium_precheckout(update: Any, context: Any) -> None:
    query = update.pre_checkout_query
    if query is None:
        return
    user = update.effective_user
    if user is None or query.invoice_payload != f"premium:{user.id}":
        await query.answer(ok=False, error_message="Не удалось проверить платёж, попробуй оформить заново через /premium.")
        return
    await query.answer(ok=True)


async def premium_successful_payment(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    payment = update.message.successful_payment if update.message is not None else None
    if user is None or payment is None or premium_store is None:
        return
    subscription = await asyncio.to_thread(
        premium_store.record_payment,
        user_id=user.id,
        telegram_payment_charge_id=payment.telegram_payment_charge_id,
        stars_amount=int(payment.total_amount),
        days=settings.premium_days,
    )
    if subscription is None:
        return
    expiry = datetime.fromtimestamp(subscription.expires_at).strftime("%d.%m.%Y")
    await update.effective_message.reply_text(
        f"Спасибо! Премиум активен до {expiry}.\nПосмотреть статус: /me",
        reply_markup=_remove_reply_keyboard(),
    )


async def paysupport_command(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    await update.effective_message.reply_text(
        "По вопросам оплаты премиума (включая возврат звёзд) "
        f"{settings.paysupport_contact}.",
        reply_markup=_remove_reply_keyboard(),
    )


async def admin_grant_premium(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    if user is None or not settings.is_admin(user.id):
        await update.effective_message.reply_text("Команда доступна только администратору.")
        return
    args = context.args or []
    if not args or not args[0].lstrip("-").isdigit():
        await update.effective_message.reply_text("Использование: /grant_premium <user_id> [дней]")
        return
    target_id = int(args[0])
    days = int(args[1]) if len(args) > 1 and args[1].isdigit() else settings.premium_days
    subscription = await asyncio.to_thread(
        premium_store.grant_manual, user_id=target_id, days=days, admin_id=user.id
    )
    expiry = datetime.fromtimestamp(subscription.expires_at).strftime("%d.%m.%Y")
    await update.effective_message.reply_text(f"Премиум для {target_id} выдан до {expiry}.")


async def admin_revoke_premium(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    if user is None or not settings.is_admin(user.id):
        await update.effective_message.reply_text("Команда доступна только администратору.")
        return
    args = context.args or []
    if not args or not args[0].lstrip("-").isdigit():
        await update.effective_message.reply_text("Использование: /revoke_premium <user_id>")
        return
    target_id = int(args[0])
    revoked = await asyncio.to_thread(premium_store.revoke, target_id, status="admin_revoked")
    await update.effective_message.reply_text(
        f"Премиум для {target_id} отключён." if revoked else f"У {target_id} нет активной подписки."
    )


async def admin_refund_premium(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    if user is None or not settings.is_admin(user.id):
        await update.effective_message.reply_text("Команда доступна только администратору.")
        return
    args = context.args or []
    if not args or not args[0].lstrip("-").isdigit():
        await update.effective_message.reply_text("Использование: /refund_premium <user_id>")
        return
    target_id = int(args[0])
    charge_id = await asyncio.to_thread(premium_store.latest_charge_id, target_id)
    if charge_id is None or charge_id.startswith("manual-"):
        await update.effective_message.reply_text(f"У {target_id} нет оплаченной звёздами подписки для возврата.")
        return
    try:
        await context.bot.refund_star_payment(user_id=target_id, telegram_payment_charge_id=charge_id)
    except Exception as exc:
        await update.effective_message.reply_text(f"Возврат не удался: {type(exc).__name__}: {exc}")
        return
    await asyncio.to_thread(premium_store.revoke, target_id, status="refunded")
    await update.effective_message.reply_text(f"Звёзды возвращены, премиум для {target_id} отключён.")


async def admin_reviews(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    review_store: TextReviewStore | None = context.application.bot_data.get("review_store")
    user = update.effective_user
    if user is None or not settings.is_admin(user.id):
        await update.effective_message.reply_text("Команда доступна только администратору.")
        return
    if review_store is None:
        await update.effective_message.reply_text("Сбор статистики не настроен.")
        return

    summary = await asyncio.to_thread(review_store.summary)
    if not summary:
        await update.effective_message.reply_text("Пока никто не пользовался проверкой текста.")
        return

    approved = summary.get("approved", 0)
    rejected = summary.get("rejected", 0)
    cancelled = summary.get("cancelled", 0)
    total = approved + rejected + cancelled
    lines = [
        "📝 Проверка текста перед озвучкой",
        "",
        f"✅ Принято: {approved}",
        f"🔄 Отклонено вариантов: {rejected}",
        f"❌ Задач отменено: {cancelled}",
    ]
    if total:
        lines.append(f"Доля принятых: {approved * 100 // total}%")

    accepted = await asyncio.to_thread(review_store.recent, "approved", 3)
    if accepted:
        lines.extend(["", "Последние принятые:"])
        for record in accepted:
            preview = re.sub(r"\s+", " ", record.text)[:70]
            lines.append(f"№{record.job_number} (вариант {record.attempt}): {preview}")
    await update.effective_message.reply_text("\n".join(lines), reply_markup=_remove_reply_keyboard())


async def admin_prem_owners(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    if user is None or not settings.is_admin(user.id):
        await update.effective_message.reply_text("Команда доступна только администратору.")
        return
    subscriptions = (
        await asyncio.to_thread(premium_store.list_active_subscriptions) if premium_store is not None else []
    )
    if not subscriptions:
        await update.effective_message.reply_text("Активных премиум-подписок нет.")
        return
    lines = [f"⭐ Активные премиум-подписки: {len(subscriptions)}", ""]
    for subscription in subscriptions:
        expiry = datetime.fromtimestamp(subscription.expires_at).strftime("%d.%m.%Y")
        if subscription.granted_by:
            source = f"выдано вручную (админ {subscription.granted_by})"
        else:
            purchased = datetime.fromtimestamp(subscription.purchased_at).strftime("%d.%m.%Y")
            source = f"{subscription.stars_amount} ⭐, куплено {purchased}"
        who = await _describe_telegram_user(context.bot, subscription.user_id)
        lines.append(f"{who} — до {expiry} ({source})")
    text = "\n".join(lines)
    if len(text) > 3900:
        text = text[:3900] + "\n…"
    await update.effective_message.reply_text(text)


def _format_telegram_person(person: Any, fallback_id: int) -> str:
    """"@username Full Name (id)" from an already-fetched User/Chat-like object -
    no API call, unlike _describe_telegram_user below."""
    name = " ".join(
        part for part in (getattr(person, "first_name", None), getattr(person, "last_name", None)) if part
    ).strip()
    username = f"@{person.username}" if getattr(person, "username", None) else None
    label = " ".join(part for part in (username, name) if part).strip()
    return f"{label} ({fallback_id})" if label else str(fallback_id)


async def _describe_telegram_user(bot: Any, user_id: int) -> str:
    """Best-effort "@username Full Name (id)" label for admin listings - nothing
    is stored locally, so this is a live lookup and can fail (blocked bot, no
    shared chat history yet), in which case it just falls back to the bare id."""
    try:
        chat = await bot.get_chat(user_id)
    except Exception:
        return str(user_id)
    return _format_telegram_person(chat, user_id)


def _describe_transaction_partner(partner: Any) -> str:
    kind = type(partner).__name__
    if kind == "TransactionPartnerUser":
        person = getattr(partner, "user", None)
        return _format_telegram_person(person, person.id) if person is not None else "пользователь"
    if kind == "TransactionPartnerFragment":
        return "Fragment (вывод/возврат)"
    if kind == "TransactionPartnerTelegramAds":
        return "реклама Telegram Ads"
    if kind == "TransactionPartnerTelegramApi":
        return "оплата запросов Bot API"
    if kind == "TransactionPartnerAffiliateProgram":
        return "партнёрская программа"
    if kind == "TransactionPartnerChat":
        chat = getattr(partner, "chat", None)
        title = getattr(chat, "title", None) if chat is not None else None
        return f"чат «{title}»" if title else "чат"
    return "источник неизвестен"


async def admin_star_balance(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None or not settings.is_admin(user.id):
        await update.effective_message.reply_text("Команда доступна только администратору.")
        return

    try:
        balance = await context.bot.get_my_star_balance()
    except Exception as exc:
        await update.effective_message.reply_text(f"Не удалось получить баланс: {type(exc).__name__}: {exc}")
        return

    lines = [f"⭐ Баланс бота: {balance.amount}"]
    if balance.nanostar_amount:
        lines[0] += f" + {balance.nanostar_amount / 1_000_000_000:.3f}"

    try:
        transactions = await context.bot.get_star_transactions(limit=10)
    except Exception as exc:
        lines.append(f"\nНе удалось получить историю операций: {type(exc).__name__}: {exc}")
        await update.effective_message.reply_text("\n".join(lines))
        return

    if transactions.transactions:
        lines.append("")
        lines.append("Последние операции:")
        for transaction in transactions.transactions:
            when = transaction.date.strftime("%d.%m %H:%M")
            if transaction.source is not None:
                lines.append(f"{when}  +{transaction.amount} ⭐  {_describe_transaction_partner(transaction.source)}")
            else:
                who = _describe_transaction_partner(transaction.receiver) if transaction.receiver else "неизвестно"
                lines.append(f"{when}  -{transaction.amount} ⭐  {who}")

    text = "\n".join(lines)
    if len(text) > 3900:
        text = text[:3900] + "\n…"
    await update.effective_message.reply_text(text)


def _is_premium_user(settings: BotSettings, premium_store: PremiumStore | None, user_id: int | None) -> bool:
    if user_id is None:
        return False
    if settings.is_paid(user_id) or settings.is_admin(user_id):
        return True
    return premium_store is not None and premium_store.active_subscription(user_id) is not None


def _max_file_mb_for(
    settings: BotSettings, premium_store: PremiumStore | None, user_id: int | None
) -> int:
    """Download size cap for this user, in MB. 0 means no cap."""
    if _is_premium_user(settings, premium_store, user_id):
        return settings.max_file_mb_premium
    return settings.max_file_mb


_CENSOR_MARK_PATTERN: object = None

_PREMIUM_ONLY_TEXT = "Это премиум-функция. Оформить: /premium"


async def watermark_command(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    if user is None:
        return
    if not _is_premium_user(settings, premium_store, user.id):
        await update.effective_message.reply_text(_PREMIUM_ONLY_TEXT, reply_markup=_remove_reply_keyboard())
        return

    args = context.args or []
    action = str(args[0]).strip().lower() if args else ""
    if action not in {"on", "off"}:
        current = await asyncio.to_thread(premium_store.get_user_settings, user.id)
        state = "включён" if current.watermark_enabled else "выключен"
        await update.effective_message.reply_text(
            f"Водяной знак сейчас {state}.\nИспользование: /watermark on или /watermark off",
            reply_markup=_remove_reply_keyboard(),
        )
        return

    enabled = action == "on"
    await asyncio.to_thread(premium_store.set_watermark_enabled, user.id, enabled)
    state = "включён" if enabled else "выключен"
    await update.effective_message.reply_text(
        f"Водяной знак теперь {state} для твоих новых задач.", reply_markup=_remove_reply_keyboard()
    )


async def mycensor_command(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    user = update.effective_user
    if user is None:
        return
    if not _is_premium_user(settings, premium_store, user.id):
        await update.effective_message.reply_text(_PREMIUM_ONLY_TEXT, reply_markup=_remove_reply_keyboard())
        return

    args = context.args or []
    raw_value = str(args[0]).strip().lower() if args else ""
    if not raw_value:
        current = await asyncio.to_thread(premium_store.get_user_settings, user.id)
        global_percent = _get_censor_percent(settings)
        state = f"{current.censor_percent}%" if current.censor_percent is not None else f"общий ({global_percent}%)"
        await update.effective_message.reply_text(
            f"Твой censor сейчас: {state}.\n"
            "Использование: /mycensor 0..100 или /mycensor default (использовать общий).",
            reply_markup=_remove_reply_keyboard(),
        )
        return

    if raw_value in {"default", "auto", "global"}:
        await asyncio.to_thread(premium_store.set_censor_percent, user.id, None)
        await update.effective_message.reply_text(
            "Личный censor сброшен, для тебя снова используется общий уровень.",
            reply_markup=_remove_reply_keyboard(),
        )
        return

    if not raw_value.isdigit() or not 0 <= int(raw_value) <= 100:
        await update.effective_message.reply_text(
            "Использование: /mycensor 0..100 или /mycensor default", reply_markup=_remove_reply_keyboard()
        )
        return

    percent = int(raw_value)
    await asyncio.to_thread(premium_store.set_censor_percent, user.id, percent)
    await update.effective_message.reply_text(
        f"Личный censor для твоих задач теперь: {percent}%.", reply_markup=_remove_reply_keyboard()
    )


# Same field order the normal send-a-video flow asks in (see _advance_selection).
# Each entry: (field name in PresetStore, screen title, option list, keyboard columns).
PRESET_WIZARD_STEPS: list[tuple[str, str, list[tuple[str, str]], int]] = [
    ("visual_mode", "Видеоряд", VISUAL_MODE_OPTIONS, 1),
    ("source_lang", "Входной язык", SOURCE_LANGS, 2),
    ("speaker_count", "Количество голосов", SPEAKER_COUNT_OPTIONS, 5),
    ("target_lang", "Язык озвучки", TARGET_LANGS, 2),
]
# Nothing to decide while a step has a single answer - the normal flow picks it
# without asking, so a preset for it would only ever restate that pick.
if len(TTS_METHOD_CHOICES) > 1:
    PRESET_WIZARD_STEPS.append(("tts_provider", "Движок озвучки", TTS_METHOD_CHOICES, 1))
if REVIEW_STEP_OFFERED:
    PRESET_WIZARD_STEPS.append(("review_mode", "Проверка текста", REVIEW_MODE_OPTIONS, 1))


def _preset_step_keyboard(field: str, options: list[tuple[str, str]], columns: int) -> Any:
    items = [("ask", "❓ Спрашивать каждый раз"), *options]
    return _language_keyboard(f"pset:{field}", items, columns=columns)


def _preset_step_text(step_index: int, title: str) -> str:
    return (
        f"Настройка пресета — шаг {step_index}/{len(PRESET_WIZARD_STEPS)}.\n"
        f"{title}: нажми так же, как обычно выбираешь при отправке видео, "
        "или «Спрашивать каждый раз», если хочешь продолжать выбирать вручную."
    )


def _preset_field_label(field: str, value: str) -> str:
    if field == "visual_mode":
        return next((label for code, label in VISUAL_MODE_OPTIONS if code == value), value)
    if field == "source_lang":
        return next((label for code, label in SOURCE_LANGS if code == value), value)
    if field == "speaker_count":
        return _speaker_count_label(value)
    if field == "target_lang":
        return _target_lang_label(value)
    if field == "tts_provider":
        return _tts_method_label(value)
    if field == "review_mode":
        return next((label for code, label in REVIEW_MODE_OPTIONS if code == value), value)
    return value


_PRESET_FIELD_TITLES = {field: title for field, title, _options, _columns in PRESET_WIZARD_STEPS}


def _preset_summary_text(preset: UserPreset) -> str:
    lines = ["Пресет сохранён:"]
    # Only the steps the wizard actually asks about - a hidden step (see
    # REVIEW_STEP_OFFERED) has no title and nothing to report.
    for field in _PRESET_FIELD_TITLES:
        value = getattr(preset, field)
        shown = "спрашивать каждый раз" if value is None else _preset_field_label(field, value)
        lines.append(f"{_PRESET_FIELD_TITLES[field]}: {shown}")
    lines.append("")
    lines.append("Изменить: /preset. Сбросить всё: /preset reset.")
    return "\n".join(lines)


async def preset_command(update: Any, context: Any) -> None:
    preset_store: PresetStore | None = context.application.bot_data.get("preset_store")
    user = update.effective_user
    if user is None or preset_store is None:
        return

    args = context.args or []
    if args and str(args[0]).strip().lower() in {"reset", "clear"}:
        await asyncio.to_thread(preset_store.clear_preset, user.id)
        await update.effective_message.reply_text(
            "Пресет сброшен, дальше буду спрашивать всё вручную. /preset — настроить заново."
        )
        return

    field, title, options, columns = PRESET_WIZARD_STEPS[0]
    await update.effective_message.reply_text(
        _preset_step_text(1, title), reply_markup=_preset_step_keyboard(field, options, columns)
    )


async def preset_wizard_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    preset_store: PresetStore | None = context.application.bot_data.get("preset_store")
    if user is None or preset_store is None:
        return

    _, field, value = query.data.split(":", 2)
    if field not in PRESET_FIELDS:
        await query.edit_message_text("Неизвестный шаг пресета. Набери /preset ещё раз.")
        return

    await asyncio.to_thread(preset_store.set_preset_field, user.id, field, None if value == "ask" else value)

    step_index = next(index for index, item in enumerate(PRESET_WIZARD_STEPS) if item[0] == field)
    next_index = step_index + 1
    if next_index >= len(PRESET_WIZARD_STEPS):
        preset = await asyncio.to_thread(preset_store.get_preset, user.id)
        await query.edit_message_text(_preset_summary_text(preset))
        return

    next_field, next_title, next_options, next_columns = PRESET_WIZARD_STEPS[next_index]
    await query.edit_message_text(
        _preset_step_text(next_index + 1, next_title),
        reply_markup=_preset_step_keyboard(next_field, next_options, next_columns),
    )


async def start(update: Any, context: Any) -> None:
    await update.effective_message.reply_text(
        "Пришли видео или аудио. Я попрошу выбрать видеоряд, input-язык, количество голосов и язык озвучки.\n\n"
        "Input-язык используется как промежуточный язык перевода и источник Whisper-артефактов.\n"
        "Номер работы будет указан в статусе. Для принудительного восстановления: /resume НОМЕР.\n"
        "У новых пользователей доступно 3 минуты перевода в сутки. Лимит растёт вместе с кармой; "
        "ограничения на длину одного видео нет. Профиль: /me.\n"
        "На итоговом видео будет водяной знак.",
        reply_markup=_remove_reply_keyboard(),
    )


def _effective_level(
    premium_store: PremiumStore | None, user_id: int | None, karma_milli: int
) -> tuple[KarmaLevel, Subscription | None]:
    """Karma-tier level, unless a capped Stars subscription is active - that one wins instead."""
    if premium_store is not None and user_id is not None:
        subscription = premium_store.active_subscription(user_id)
        if subscription is not None:
            return PREMIUM_LEVEL, subscription
    return level_for_karma(karma_milli), None


async def me(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None:
        return
    store: ProposalStore | None = context.application.bot_data.get("proposal_store")
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    karma_milli = await asyncio.to_thread(store.karma_total, user.id) if store is not None else 0
    privileged = settings.is_paid(user.id) or settings.is_admin(user.id)
    level, subscription = await asyncio.to_thread(_effective_level, premium_store, user.id, karma_milli)
    lines = [
        "👤 Твой профиль",
        f"Telegram ID: {user.id}",
        f"Уровень: {level.name}",
        f"Карма: {visible_karma(karma_milli)}{' ∞' if subscription is not None else ''}",
    ]
    next_level = next_level_for_karma(karma_milli)
    if next_level is None:
        lines.append("Достигнут максимальный уровень.")
    else:
        remaining = max(0, next_level.minimum * KARMA_SCALE - karma_milli)
        lines.append(f"До уровня «{next_level.name}»: {(remaining + KARMA_SCALE - 1) // KARMA_SCALE}")

    if privileged:
        lines.extend(["", "Лимит перевода: без ограничений", "Приоритет: премиум"])
    else:
        used_ms = await asyncio.to_thread(store.daily_usage_ms, user.id) if store is not None else 0
        limit_ms = level.daily_minutes * 60_000
        lines.extend(
            [
                "",
                f"Использовано за последние 24ч: {_format_duration_ms(used_ms)} из {_format_duration_ms(limit_ms)}",
                f"Осталось: {_format_duration_ms(max(0, limit_ms - used_ms))}",
                f"Приоритет в очереди: {'обычный' if level.priority_bonus == 0 else f'+{level.priority_bonus}'}",
                f"Лимит задач в очереди: {level.queue_limit}",
            ]
        )
        free_at = await asyncio.to_thread(store.next_usage_free_at, user.id) if store is not None else None
        if free_at is not None:
            lines.append(f"Начнёт освобождаться: {datetime.fromtimestamp(free_at).strftime('%H:%M %d.%m')}")
        if subscription is not None:
            expiry = datetime.fromtimestamp(subscription.expires_at).strftime("%d.%m.%Y")
            lines.append(f"Премиум активен до: {expiry}")

    if store is not None:
        publications = await asyncio.to_thread(store.publication_summary, user.id)
        main_count, _main_ms = publications.get("main", (0, 0))
        shame_count, _shame_ms = publications.get("shame", (0, 0))
        lines.extend(["", "Опубликовано:", f"La La School — {main_count}", f"Ghien Mi Go — {shame_count}"])
    await update.effective_message.reply_text("\n".join(lines), reply_markup=_remove_reply_keyboard())


async def censored(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None:
        return
    if not settings.is_admin(user.id):
        premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
        if _is_premium_user(settings, premium_store, user.id):
            hint = "Свой личный уровень можно настроить через /mycensor."
        else:
            hint = _PREMIUM_ONLY_TEXT
        await update.effective_message.reply_text(f"Команда доступна только администратору.\n{hint}")
        return

    raw_value = str(context.args[0]).strip() if context.args else ""
    if len(context.args) != 1 or not raw_value.isdigit():
        current = _get_censor_percent(settings)
        await update.effective_message.reply_text(
            f"Использование: /censored 0..100\nГлобально сейчас: {current}%\n0 выключает режим.",
            reply_markup=_remove_reply_keyboard(),
        )
        return

    percent = int(raw_value)
    if not 0 <= percent <= 100:
        await update.effective_message.reply_text(
            "Использование: /censored 0..100",
            reply_markup=_remove_reply_keyboard(),
        )
        return

    _set_censor_percent(settings, percent)
    if percent <= 0:
        text = "Experimental censor глобально выключен для следующих задач."
    else:
        text = (
            f"Experimental censor глобально включён: {percent}%.\n"
            "Если в переводе найдутся запрещённые слова/маты/slurs, они с этой вероятностью будут заменяться на предупреждения/censored-фразы."
        )
    await update.effective_message.reply_text(text, reply_markup=_remove_reply_keyboard())


async def queue_status(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    scheduler: _JobScheduler = context.application.bot_data["job_scheduler"]
    live = await scheduler.snapshot()
    # This is only a diagnostic cross-check for recently stranded work.  The
    # live scheduler already owns the real queue, so walking the whole 30-day
    # archive here just makes /queue appear dead on a large F: drive.
    disk_counts = await asyncio.to_thread(
        _job_status_counts,
        settings.workdir,
        newer_than=time.time() - 2 * 86400,
    )

    live_total = live["active_total"] + live["pending_total"]

    lines = [f"🎬 Сейчас в работе: {live['active_total']} из {live['max_active_jobs']}", ""]

    local_state = "занят" if live["local_machine_busy"] else "свободен"
    lines.append(f"💻 Основной ПК — {local_state} ({live['active_local']}/{live['max_local_jobs']})")

    if live["remote_workers_online"]:
        worker_state = "занят" if live["remote_workers_busy"] else "свободен"
        lines.append(
            f"📡 Воркер — {worker_state} "
            f"({live['remote_workers_busy']}/{live['remote_workers_online']})"
        )
    else:
        # "Not on the air" is true but says nothing, and most of the time today
        # it meant the laptop was installing an update we had just published.
        served_at = float(context.application.bot_data.get("worker_update_served_at") or 0.0)
        if time.time() - served_at < WORKER_UPDATE_QUIET_SECONDS:
            lines.append("📡 Воркер — ставит обновление")
        else:
            lines.append("📡 Воркер — не на связи")
    if live["remote_workers_stale"]:
        lines.append(f"   ⚠️ без свежего пинга: {live['remote_workers_stale']}")

    lines.append("")
    if live["pending_total"]:
        waiting = f"⏳ Ждут очереди: {live['pending_total']}"
        # Only worth splitting out when the mix actually matters.
        if live["pending_premium"] and live["pending_normal"]:
            waiting += f" (премиум {live['pending_premium']}, обычных {live['pending_normal']})"
        lines.append(waiting)
    else:
        lines.append("⏳ Очередь пуста")
    lines.append(f"👤 Пользователей с задачами: {live['active_users']}")

    # Anything on disk that the live scheduler does not know about is a job left
    # behind by a restart - that is the only part of the file scan worth showing.
    stuck = max(0, sum(disk_counts.get(name, 0) for name in ("running", "starting", "queued", "ready")) - live_total)
    unfinished = sum(count for name, count in disk_counts.items() if name.startswith("select_"))
    broken = disk_counts.get("bad_json", 0)

    if stuck or unfinished or broken:
        lines.append("")
        if stuck:
            lines.append(f"🔁 Зависли после перезапуска: {stuck} — подхватить: /resume НОМЕР")
        if unfinished:
            lines.append(f"💤 Брошенные диалоги: {unfinished} (прислали видео, но не выбрали настройки)")
        if broken:
            lines.append(f"⚠️ Повреждённых записей: {broken}")

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
    requested_number = str(context.args[0]).strip() if context.args else ""
    if len(context.args) > 1 or (requested_number and not requested_number.isdigit()):
        await update.effective_message.reply_text("Использование: /resume НОМЕР_РАБОТЫ")
        return

    job = (
        _find_resumable_job_by_number(settings, user.id, requested_number)
        if requested_number
        else _find_latest_resumable_job(settings, user.id)
    )
    if not job:
        if requested_number:
            await update.effective_message.reply_text(
                f"Не нашёл твою работу №{requested_number} или её исходный файл уже удалён."
            )
        else:
            await update.effective_message.reply_text(
                "Не нашёл незавершённую задачу. Укажи номер явно: /resume НОМЕР_РАБОТЫ"
            )
        return

    _prepare_job_for_resume(job)
    job_number = _job_number(job)
    status_message = await update.effective_message.reply_text(
        f"Принудительно восстанавливаю работу №{job_number} и ставлю её в очередь."
    )
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
    _prepare_job_for_resume(job)
    await query.edit_message_text(f"Восстанавливаю работу №{_job_number(job)} и ставлю её в очередь.")
    await _enqueue_job(update, context, job, query.message)


async def send_to_proposal(update: Any, context: Any) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    chat = update.effective_chat
    message = update.effective_message
    if user is None or chat is None or message is None:
        return
    requested_number = str(context.args[0]).strip() if context.args else ""
    if len(context.args) != 1 or not requested_number.isdigit():
        await message.reply_text("Использование: /send НОМЕР_РАБОТЫ")
        return
    if not settings.proposal_enabled or "proposal_store" not in context.application.bot_data:
        await message.reply_text("Предложка сейчас отключена.")
        return

    job_number = str(int(requested_number))
    job_dir = settings.workdir / str(user.id) / job_number
    job = _load_job_snapshot(job_dir)
    if not isinstance(job, dict):
        await message.reply_text(f"Не нашёл твою работу №{job_number}. Возможно, её файлы уже удалены.")
        return
    owner_id = _coerce_int(job.get("user_id"))
    if owner_id is not None and owner_id != user.id:
        await message.reply_text("Эта работа принадлежит другому пользователю.")
        return
    if str(job.get("status") or "") not in {"ready", "done"}:
        await message.reply_text(f"Работа №{job_number} ещё не завершена.")
        return

    video_path = _find_proposal_video_path(job_dir, job)
    if video_path is None:
        await message.reply_text(f"Итоговый видеофайл работы №{job_number} уже не найден.")
        return
    try:
        submission, created = await _create_proposal_submission(
            context,
            job=job,
            job_number=job_number,
            user=user,
            chat_id=int(chat.id),
            video_path=video_path,
        )
    except Exception as exc:
        print("Forced proposal submission failed:\n" + traceback.format_exc(), flush=True)
        await message.reply_text(f"Не удалось отправить работу №{job_number}: {type(exc).__name__}")
        return
    if created:
        await message.reply_text(f"Работа №{job_number} отправлена в предложку.")
    else:
        await message.reply_text(f"Работа №{job_number} уже находится в предложке.")


async def proposal_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    settings: BotSettings = context.application.bot_data["settings"]
    user = update.effective_user
    if user is None:
        await query.answer("Не удалось определить пользователя.", show_alert=True)
        return
    if not settings.proposal_enabled or "proposal_store" not in context.application.bot_data:
        await query.answer("Предложка сейчас отключена.", show_alert=True)
        return

    parts = str(query.data or "").split(":")
    if len(parts) != 3:
        await query.answer("Некорректная кнопка.", show_alert=True)
        return
    action, value = parts[1], parts[2]
    if action == "submitted":
        await query.answer("Видео уже отправлено в предложку.")
        return
    if action != "submit" or not value.isdigit():
        await query.answer("Некорректная кнопка.", show_alert=True)
        return

    job_dir = settings.workdir / str(user.id) / value
    snapshot_path = _job_snapshot_path(job_dir)
    try:
        job = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        await query.answer("Данные этой работы уже не найдены.", show_alert=True)
        return
    if not isinstance(job, dict) or _coerce_int(job.get("user_id")) != user.id:
        await query.answer("Эта работа принадлежит другому пользователю.", show_alert=True)
        return
    if str(job.get("status") or "") not in {"ready", "done"}:
        await query.answer("Работа ещё не завершена.", show_alert=True)
        return

    video_path = _find_proposal_video_path(job_dir, job)
    if video_path is None:
        await query.answer("Итоговый видеофайл уже не найден.", show_alert=True)
        return
    try:
        submission, created = await _create_proposal_submission(
            context,
            job=job,
            job_number=value,
            user=user,
            chat_id=int(update.effective_chat.id),
            video_path=video_path,
        )
    except Exception as exc:
        print("Proposal submission failed:\n" + traceback.format_exc(), flush=True)
        await query.answer(f"Не удалось отправить: {type(exc).__name__}", show_alert=True)
        return

    with contextlib.suppress(Exception):
        await query.edit_message_reply_markup(reply_markup=_proposal_submitted_keyboard(submission.id))
    await query.answer("Отправлено в предложку." if created else "Видео уже находится в предложке.")


def _find_proposal_video_path(job_dir: Path, job: dict[str, Any]) -> Path | None:
    candidates = [
        Path(str(job.get("proposal_video_path") or "")),
        job_dir / "dubbed_watermarked.mp4",
        job_dir / "dubbed.mp4",
    ]
    remote_video_dir = job_dir / "remote_result" / "video"
    if remote_video_dir.is_dir():
        candidates.extend(
            sorted(
                (path for path in remote_video_dir.iterdir() if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


async def _create_proposal_submission(
    context: Any,
    *,
    job: dict[str, Any],
    job_number: str,
    user: Any,
    chat_id: int,
    video_path: Path,
) -> tuple[Any, bool]:
    output_filename = str(job.get("proposal_output_filename") or "").strip()
    if not output_filename:
        output_filename = _lalaschool_filename(
            job.get("source_title") or video_path.stem,
            video_path.suffix,
        )
    author_name = str(getattr(user, "full_name", "") or getattr(user, "username", "") or user.id).strip()
    author_username = str(getattr(user, "username", "") or "").strip() or None
    duration_ms = max(0, round(await asyncio.to_thread(probe_duration, video_path) * 1000))
    store: ProposalStore = context.application.bot_data["proposal_store"]
    karma_before_milli = await asyncio.to_thread(store.karma_total, user.id)
    return await asyncio.to_thread(
        store.create_submission,
        job_number=job_number,
        user_id=user.id,
        chat_id=chat_id,
        author_name=author_name,
        author_username=author_username,
        video_path=video_path,
        output_filename=output_filename,
        duration_ms=duration_ms,
        karma_before_milli=karma_before_milli,
    )


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

    await _remember_job_and_ask_source(
        context, status, job_dir, input_path, source_title, input_source="telegram_upload", update=update, user_id=user_id
    )


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

    await _remember_job_and_ask_source(
        context, status, job_dir, audio_path, source_title, input_source="telegram_audio", update=update, user_id=user_id
    )


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
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    max_file_mb = _max_file_mb_for(settings, premium_store, user_id)
    try:
        input_path = await asyncio.to_thread(download_video_url, url, job_dir, max_file_mb)
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
        update=update,
        user_id=user_id,
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
        await _safe_edit_status(status, f"Не смог определить длительность аудио:\n{type(exc).__name__}: {exc}")
        return None

    target_duration = duration
    await _safe_edit_status(status, "Собираю случайный видеоряд для аудио...")

    source_roots = _trusted_visual_source_roots(settings)
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
        await _safe_edit_status(status, "Не удалось собрать видео из аудио: получился пустой файл.")
        return None
    if not await asyncio.to_thread(has_video_and_audio, output_path):
        await _safe_edit_status(status, "Не удалось собрать видео из аудио: в результате нет видео и аудио.")
        return None

    await _safe_edit_status(status, "Аудио превращено в видео. Продолжаю.")
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

    source_roots = _trusted_visual_source_roots(settings)
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
    # Per-video limits were replaced by the karma-based daily allowance.
    return input_path


def _trusted_visual_source_roots(settings: BotSettings) -> list[Path]:
    source = settings.audio_visual_source_dir
    if source is None:
        raise RuntimeError("LALADUB_AUDIO_VISUAL_SOURCE_DIR is required for generated video mode.")
    source = source.resolve()
    if not source.is_dir():
        raise RuntimeError(f"Trusted visual source directory does not exist: {source}")
    return [source]


async def _ensure_job_input_video(
    status: Any,
    job: dict[str, Any],
    settings: BotSettings,
    input_path: Path,
    max_file_mb: int | None = None,
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
        redownloaded_path = await asyncio.to_thread(
            download_video_url,
            source_url,
            job_dir,
            settings.max_file_mb if max_file_mb is None else max_file_mb,
        )
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


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_duration_ms(duration_ms: int) -> str:
    return _format_duration(max(0, int(duration_ms)) / 1000)


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
    update: Any = None,
    user_id: int | None = None,
    source_url: str | None = None,
) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    preset_store: PresetStore | None = context.application.bot_data.get("preset_store")
    censor_percent = _get_censor_percent(settings)
    watermark_enabled = True
    if _is_premium_user(settings, premium_store, user_id):
        user_settings = await asyncio.to_thread(premium_store.get_user_settings, user_id)
        watermark_enabled = user_settings.watermark_enabled
        if user_settings.censor_percent is not None:
            censor_percent = user_settings.censor_percent
    preset = (
        await asyncio.to_thread(preset_store.get_preset, user_id)
        if preset_store is not None and user_id is not None
        else None
    )
    context.user_data["job"] = {
        "job_dir": str(job_dir),
        "input_path": str(input_path),
        "source_title": source_title,
        "input_source": input_source,
        "translation_seed": job_dir.name,
        "censor_percent": censor_percent,
        "watermark_enabled": watermark_enabled,
        "_preset": preset.as_dict() if preset is not None else {},
    }
    if source_url:
        context.user_data["job"]["source_url"] = source_url
    await _advance_selection(update, context, context.user_data["job"], status)


def _preset_choice(job: dict[str, Any], field: str) -> str | None:
    """The user's saved answer for `field`, or None when it's set to "ask
    every time" (or the user never configured a preset at all)."""
    choice = (job.get("_preset") or {}).get(field)
    return choice if choice and choice != "ask" else None


async def _show_selection_screen(target: Any, text: str, markup: Any) -> None:
    try:
        await target.edit_text(text, reply_markup=markup)
    except Exception as exc:
        if "Message can't be edited" not in f"{type(exc).__name__}: {exc}":
            raise
        await target.reply_text(text, reply_markup=markup)


async def _show_visual_screen(target: Any, job: dict[str, Any]) -> None:
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_visual")
    await _show_selection_screen(
        target,
        "Выбери видеоряд.",
        _language_keyboard("vis", VISUAL_MODE_OPTIONS, columns=1, back_callback="back:cancel"),
    )


async def _show_speakers_screen(target: Any, job: dict[str, Any]) -> None:
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_speakers")
    await _show_selection_screen(
        target,
        "Выбери количество голосов.",
        _language_keyboard("spk", SPEAKER_COUNT_OPTIONS, columns=5, back_callback="back:source"),
    )


async def _show_target_screen(target: Any, job: dict[str, Any]) -> None:
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_target")
    await _show_selection_screen(
        target,
        "Выбери язык озвучки.",
        _language_keyboard("tgt", TARGET_LANGS, columns=2, back_callback="back:speakers"),
    )


async def _show_tts_screen(target: Any, job: dict[str, Any]) -> None:
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_tts")
    await _show_selection_screen(
        target,
        "Выбери движок озвучки.",
        _language_keyboard("tts", TTS_METHOD_CHOICES, columns=1, back_callback="back:target"),
    )


async def _show_review_mode_screen(target: Any, job: dict[str, Any]) -> None:
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_review")
    await _show_selection_screen(
        target,
        "Последний шаг.\n"
        "«Сразу дубляж» — как обычно.\n"
        "«Сначала показать текст» — пришлю распознанный текст до озвучки, "
        "чтобы не тратить время на заведомо неудачный вариант.",
        _language_keyboard("rev", REVIEW_MODE_OPTIONS, columns=1, back_callback="back:target"),
    )


async def _advance_selection(update: Any, context: Any, job: dict[str, Any], target: Any) -> None:
    """Drives the video-setup wizard one step at a time: any field the user's
    saved /preset already answers is filled in and skipped silently, and the
    first field still left on "ask every time" gets its picker screen shown."""
    job_dir = Path(job["job_dir"])

    if "visual_mode" not in job:
        if job.get("input_source") == "telegram_audio":
            job["visual_mode"] = "random"
        else:
            choice = _preset_choice(job, "visual_mode")
            if choice:
                job["visual_mode"] = choice
            else:
                await _show_visual_screen(target, job)
                return

    if "source_lang" not in job:
        choice = _preset_choice(job, "source_lang")
        if choice:
            job["source_lang"] = None if choice == "auto" else choice
            job["asr_method"] = "ow-large-v3-chaos-backbone"
            job["mode"] = "dub"
            job["glitch_profile"] = "clean"
        else:
            await _ask_source_language(target, job)
            return

    if "speaker_count" not in job:
        choice = _preset_choice(job, "speaker_count")
        if choice:
            job["speaker_count"] = "auto" if choice == "auto" else int(choice)
        else:
            await _show_speakers_screen(target, job)
            return

    if "target_lang" not in job:
        choice = _preset_choice(job, "target_lang")
        if choice:
            job["target_lang"] = choice
            job["translation_chaos"] = "crooked"
            _ensure_translation_seed(job)
        else:
            await _show_target_screen(target, job)
            return

    if job["target_lang"] == "uk":
        # Neither MOSS nor CosyVoice speaks Ukrainian; F5 is the only compatible
        # engine, so it's picked without an extra user-facing screen.
        job["tts_provider"] = "f5"
    elif "tts_provider" not in job:
        choice = _preset_choice(job, "tts_provider")
        if choice and choice in {code for code, _label in TTS_METHOD_CHOICES}:
            job["tts_provider"] = choice
        elif len(TTS_METHOD_CHOICES) == 1:
            # A screen offering a single button is just an extra tap. It comes
            # back on its own the moment a second engine is offered again.
            job["tts_provider"] = TTS_METHOD_CHOICES[0][0]
        else:
            await _show_tts_screen(target, job)
            return

    if "review_mode" not in job:
        choice = _preset_choice(job, "review_mode")
        if not REVIEW_STEP_OFFERED:
            job["review_mode"] = "direct"
        elif choice and choice in {code for code, _label in REVIEW_MODE_OPTIONS}:
            job["review_mode"] = choice
        else:
            await _show_review_mode_screen(target, job)
            return

    job.pop("_preset", None)
    _save_job_snapshot(job_dir, job, status="queued")
    await _show_selection_screen(
        target,
        f"Ставлю задачу в очередь. Голоса: {_speaker_count_label(job.get('speaker_count'))}. "
        f"Язык озвучки: {_target_lang_label(job.get('target_lang'))}. "
        f"Движок: {_tts_method_label(job.get('tts_provider'))}.",
        None,
    )
    context.user_data.pop("job", None)
    await _enqueue_job(update, context, job, target)


async def _ask_source_language(status: Any, job: dict[str, Any]) -> None:
    _save_job_snapshot(Path(job["job_dir"]), job, status="select_source")
    text = (
        "Выбери input-язык. Если выбрать конкретный язык, Whisper будет принудительно "
        "слышать на нём всё видео — даже когда настоящий язык другой. Авто оставляет "
        "обычное распознавание."
    )
    back_callback = "back:cancel" if job.get("input_source") == "telegram_audio" else "back:visual"
    reply_markup = _language_keyboard("src", SOURCE_LANGS, back_callback=back_callback)
    try:
        await status.edit_text(text, reply_markup=reply_markup)
    except Exception as exc:
        if "Message can't be edited" not in f"{type(exc).__name__}: {exc}":
            raise
        await status.reply_text(text, reply_markup=reply_markup)


async def selection_back(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Пришли видео или аудио ещё раз.")
        return

    destination = query.data.split(":", 1)[1]
    job_dir = Path(str(job["job_dir"]))
    if destination == "cancel":
        context.user_data.pop("job", None)
        _save_job_snapshot(job_dir, job, status="rejected", error="selection_cancelled")
        await query.edit_message_text("Выбор отменён. Пришли видео или аудио, когда захочешь начать заново.")
        return
    if destination == "visual" and job.get("input_source") != "telegram_audio":
        await _show_visual_screen(query.message, job)
        return
    if destination == "source":
        await _ask_source_language(query.message, job)
        return
    if destination == "speakers":
        await _show_speakers_screen(query.message, job)
        return
    if destination == "target":
        await _show_target_screen(query.message, job)
        return
    if destination == "chaos":
        # Compatibility for an old TTS keyboard left open before this step was
        # removed: send the user back to the target-language screen.
        await _show_target_screen(query.message, job)
        return

    await query.edit_message_text("Не удалось вернуться назад. Пришли файл ещё раз.")


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
    await _advance_selection(update, context, job, query.message)


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
    await _advance_selection(update, context, job, query.message)


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
        reply_markup=_language_keyboard("spk", SPEAKER_COUNT_OPTIONS, columns=5, back_callback="back:source"),
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
    await _advance_selection(update, context, job, query.message)


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
    job["translation_chaos"] = "crooked"
    _ensure_translation_seed(job)
    await _advance_selection(update, context, job, query.message)


async def select_tts_method(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    tts_provider = _tts_provider_value(query.data.split(":", 1)[1])
    if tts_provider is None or tts_provider not in {code for code, _label in TTS_METHOD_CHOICES}:
        await query.edit_message_text("Неизвестный метод озвучки. Пришли видео ещё раз.")
        return

    job["tts_provider"] = tts_provider
    job["translation_chaos"] = _translation_chaos_value(job.get("translation_chaos")) or "crooked"
    _ensure_translation_seed(job)
    await _advance_selection(update, context, job, query.message)


async def select_review_mode(update: Any, context: Any) -> None:
    query = update.callback_query
    await query.answer()
    job = context.user_data.get("job")
    if not job:
        await query.edit_message_text("Нет активной задачи. Сначала пришли видео.")
        return

    mode = query.data.split(":", 1)[1]
    if mode not in {code for code, _label in REVIEW_MODE_OPTIONS}:
        await query.edit_message_text("Неизвестный режим. Пришли видео ещё раз.")
        return

    job["review_mode"] = mode
    await _advance_selection(update, context, job, query.message)


def _text_review_keyboard(job_number: str, attempt: int) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    rows = [[InlineKeyboardButton("✅ Озвучить этот текст", callback_data=f"rv:ok:{job_number}")]]
    if attempt < MAX_TEXT_REVIEW_ATTEMPTS:
        rows.append([InlineKeyboardButton("🔄 Другой вариант", callback_data=f"rv:again:{job_number}")])
    rows.append([InlineKeyboardButton("❌ Отменить задачу", callback_data=f"rv:drop:{job_number}")])
    return InlineKeyboardMarkup(rows)


async def _send_text_for_review(context: Any, chat_id: int | str, job: dict[str, Any]) -> None:
    """Shows the prepared dub text and waits for the author's verdict."""
    job_dir = Path(str(job["job_dir"]))
    job_number = _job_number(job)
    attempt = int(job.get("review_attempt") or 1)
    text = _read_transcript_text(job_dir / "work" / "translated.srt") or "(текст пустой)"

    header = f"📝 Вариант {attempt} из {MAX_TEXT_REVIEW_ATTEMPTS} — работа №{job_number}"
    body = text if len(text) <= 3200 else text[:3200] + "…"
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"{header}\n\n<blockquote expandable>{html.escape(body)}</blockquote>",
        parse_mode="HTML",
        reply_markup=_text_review_keyboard(job_number, attempt),
    )


async def text_review_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    user = update.effective_user
    settings: BotSettings = context.application.bot_data["settings"]
    review_store: TextReviewStore | None = context.application.bot_data.get("review_store")

    parts = str(query.data or "").split(":", 2)
    if len(parts) != 3:
        await query.answer("Некорректная кнопка.", show_alert=True)
        return
    action, job_number = parts[1], parts[2]

    job_dir = settings.workdir / str(user.id) / job_number
    job = _load_job_snapshot(job_dir)
    if job is None:
        await query.answer("Задача не найдена.", show_alert=True)
        return
    if str(job.get("status") or "") != "awaiting_review":
        await query.answer("Этот вариант уже не ждёт решения.")
        return

    attempt = int(job.get("review_attempt") or 1)
    text = _read_transcript_text(job_dir / "work" / "translated.srt") or ""

    def remember(decision: str) -> None:
        if review_store is None:
            return
        with contextlib.suppress(Exception):
            review_store.record(
                job_number=job_number,
                user_id=int(user.id),
                attempt=attempt,
                decision=decision,
                text=text,
                source_lang=str(job.get("source_lang") or ""),
                target_lang=str(job.get("target_lang") or ""),
            )

    if action == "drop":
        await query.answer("Отменяю.")
        remember("cancelled")
        _save_job_snapshot(job_dir, job, status="rejected", error="text_review_cancelled")
        await _release_daily_allowance(context, user.id, job)
        await query.edit_message_text(f"Задача №{job_number} отменена.")
        return

    if action == "again":
        if attempt >= MAX_TEXT_REVIEW_ATTEMPTS:
            await query.answer("Больше вариантов не осталось.", show_alert=True)
            return
        await query.answer("Готовлю другой вариант…")
        remember("rejected")
        job["review_attempt"] = attempt + 1
        # A fresh seed is what makes the distortion chains land differently;
        # without it the same text would simply be rebuilt.
        job["translation_seed"] = f"{job_number}-v{attempt + 1}"
        _reset_translation_outputs(job_dir)
        await query.edit_message_text(f"Вариант {attempt} отклонён. Готовлю следующий…")
        await _enqueue_job(update, context, job, query.message)
        return

    if action == "ok":
        await query.answer("Запускаю озвучку…")
        remember("approved")
        job["review_approved"] = True
        await query.edit_message_text(f"Вариант {attempt} принят. Озвучиваю работу №{job_number}…")
        await _enqueue_job(update, context, job, query.message)
        return

    await query.answer("Неизвестное действие.", show_alert=True)


def _reset_translation_outputs(job_dir: Path) -> None:
    """Drops the prepared text so the next run rebuilds it from the audio.

    The extracted audio and separated stems are deliberately kept - they do not
    change between variants, and redoing them would waste the expensive part.
    """
    work = job_dir / "work"
    for name in ("translated.srt", "source.srt"):
        with contextlib.suppress(Exception):
            (work / name).unlink(missing_ok=True)
    state_path = work / "resume_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    for key in ("translated", "source_asr", "artifacts", "preprocess_complete", "segment_count", "sparse_fill"):
        state.pop(key, None)
    with contextlib.suppress(Exception):
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


async def _enqueue_job(update: Any, context: Any, job: dict[str, Any], status_message: Any) -> None:
    scheduler: _JobScheduler = context.application.bot_data["job_scheduler"]
    settings: BotSettings = context.application.bot_data["settings"]
    job["target_lang"] = _target_lang_value(job.get("target_lang"))
    job["translation_chaos"] = _translation_chaos_value(job.get("translation_chaos")) or "crooked"
    _ensure_translation_seed(job)
    chat = update.effective_chat
    user = update.effective_user
    if chat is None:
        await _safe_edit_status(status_message, "Не удалось определить чат для задачи.")
        return
    job["chat_id"] = chat.id
    if user is not None:
        job["user_id"] = user.id
        job["is_paid"] = settings.is_paid(user.id)

    if await scheduler.reattach_if_known(context, job, status_message):
        return

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
            input_path = await _ensure_job_input_video(
                status_message,
                job,
                settings,
                input_path,
                # Re-downloading must not hit a cap the first download cleared.
                _max_file_mb_for(
                    settings,
                    context.application.bot_data.get("premium_store"),
                    user.id if user else None,
                ),
            )
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

    if not await _reserve_daily_allowance(
        context,
        status_message=status_message,
        user_id=user.id if user else None,
        job=job,
    ):
        _save_job_snapshot(Path(job["job_dir"]), job, status="rejected", error="daily_limit")
        return

    enqueued = await scheduler.enqueue(
        context,
        chat_id=chat.id,
        user_id=user.id if user else None,
        job=job,
        status_message=status_message,
    )
    if not enqueued:
        await _release_daily_allowance(context, user.id if user else None, job)
        _save_job_snapshot(Path(job["job_dir"]), job, status="rejected", error="queue_limit")


async def _reserve_daily_allowance(
    context: Any,
    *,
    status_message: Any,
    user_id: int | None,
    job: dict[str, Any],
) -> bool:
    settings: BotSettings = context.application.bot_data["settings"]
    if user_id is None or settings.is_paid(user_id) or settings.is_admin(user_id):
        return True
    store: ProposalStore | None = context.application.bot_data.get("proposal_store")
    if store is None:
        await _safe_edit_status(status_message, "Не удалось проверить суточный лимит. Попробуй ещё раз позже.")
        return False
    input_path = Path(str(job.get("input_path") or ""))
    try:
        duration_ms = max(0, round(await asyncio.to_thread(probe_duration, input_path) * 1000))
    except Exception as exc:
        print(f"Daily quota duration failed: {type(exc).__name__}: {exc}", flush=True)
        await _safe_edit_status(status_message, "Не смог определить длительность видео для суточного лимита.")
        return False
    karma_milli = await asyncio.to_thread(store.karma_total, user_id)
    premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
    level, _subscription = await asyncio.to_thread(_effective_level, premium_store, user_id, karma_milli)
    limit_ms = level.daily_minutes * 60_000
    accepted, used_ms = await asyncio.to_thread(
        store.reserve_daily_usage,
        user_id=user_id,
        job_number=_job_number(job),
        duration_ms=duration_ms,
        limit_ms=limit_ms,
    )
    job["quota_duration_ms"] = duration_ms
    if accepted:
        job["daily_used_ms_after_enqueue"] = used_ms
        return True
    remaining_ms = max(0, limit_ms - used_ms)
    if remaining_ms >= 1000:
        original_duration_ms = duration_ms
        trimmed_path = input_path.with_name(f"{input_path.stem}_daily_trimmed{input_path.suffix}")
        await _safe_edit_status(
            status_message,
            "Видео длиннее оставшегося суточного лимита.\n"
            f"Обрезаю начало до {_format_duration_ms(remaining_ms)}…",
        )
        try:
            await asyncio.to_thread(trim_video, input_path, trimmed_path, remaining_ms / 1000.0)
            if not trimmed_path.is_file() or trimmed_path.stat().st_size < 1024:
                raise RuntimeError("получился пустой файл")
            if not await asyncio.to_thread(has_video_and_audio, trimmed_path):
                raise RuntimeError("в обрезанном файле нет видео или аудио")
        except Exception as exc:
            print(f"Daily quota trim failed: {type(exc).__name__}: {exc}", flush=True)
            trimmed_path.unlink(missing_ok=True)
            await _safe_edit_status(
                status_message,
                "Не смог обрезать видео до оставшегося суточного лимита.\n"
                f"{type(exc).__name__}: {exc}",
            )
            return False

        accepted, used_after_trim_ms = await asyncio.to_thread(
            store.reserve_daily_usage,
            user_id=user_id,
            job_number=_job_number(job),
            duration_ms=remaining_ms,
            limit_ms=limit_ms,
        )
        if accepted:
            job["input_path"] = str(trimmed_path)
            job["quota_duration_ms"] = remaining_ms
            job["daily_used_ms_after_enqueue"] = used_after_trim_ms
            job["daily_trimmed"] = True
            job["daily_original_duration_ms"] = original_duration_ms
            job["daily_trimmed_duration_ms"] = remaining_ms
            await _safe_edit_status(
                status_message,
                "Видео обрезано по остатку суточного лимита.\n"
                f"Было: {_format_duration_ms(original_duration_ms)}. "
                f"В работу пойдёт: {_format_duration_ms(remaining_ms)}.",
            )
            return True
        trimmed_path.unlink(missing_ok=True)
        used_ms = used_after_trim_ms
        remaining_ms = max(0, limit_ms - used_ms)

    free_at = await asyncio.to_thread(store.next_usage_free_at, user_id)
    when_text = (
        f"Часть лимита освободится {datetime.fromtimestamp(free_at).strftime('%H:%M %d.%m')}."
        if free_at is not None
        else "Лимит обновится в течение суток."
    )
    await _safe_edit_status(
        status_message,
        "Суточный лимит уже израсходован (считается скользящим окном за последние 24 часа).\n"
        f"Уровень: {level.name}. Лимит: {_format_duration_ms(limit_ms)}.\n"
        f"Использовано: {_format_duration_ms(used_ms)}. Осталось: {_format_duration_ms(remaining_ms)}.\n"
        f"{when_text}",
    )
    return False


async def _refund_failed_job(context: Any, item: "_QueuedJob") -> None:
    """Give the minutes back when a job did not deliver anything.

    The daily allowance pays for a finished dub, so a failure must not spend
    it. Refunding was already attempted on the failure paths, but only after
    the "Task failed" message had been sent - if that send raised (the user
    blocked the bot, a network blip) the refund never ran, and 16 failed jobs
    were still holding minutes. Called from finally now, so nothing earlier
    can skip it, and release is a plain DELETE so calling it twice is safe.

    Interrupted jobs are deliberately left alone: they stay resumable, and
    reserve_daily_usage recognises the existing row rather than charging again.
    """
    with contextlib.suppress(Exception):
        await _release_daily_allowance(context, item.user_id, item.job)


async def _release_daily_allowance(context: Any, user_id: int | None, job: dict[str, Any]) -> None:
    if user_id is None:
        return
    store: ProposalStore | None = context.application.bot_data.get("proposal_store")
    if store is None:
        return
    await asyncio.to_thread(store.release_daily_usage, user_id, _job_number(job))


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
        # Do not send one Telegram message per recovered job.  A restart can
        # restore dozens of items, and that burst previously triggered hours
        # of Telegram flood control.  The job will create a status message
        # only when it actually starts running.
        status_message = None

        input_path = Path(str(job.get("input_path") or ""))
        if job.get("input_source") == "telegram_audio" and not job.get("audio_visual"):
            prepared_path = await _prepare_input_audio_as_video(
                status_message,
                settings,
                user_id,
                job_dir,
                input_path,
            )
            if prepared_path is None:
                skipped += 1
                _save_job_snapshot(job_dir, job, status="rejected", error="audio_visual_recovery_failed")
                continue
            job["input_path"] = str(prepared_path)
            _save_job_snapshot(job_dir, job, status="queued", audio_visual=True)
        elif job.get("visual_mode") == "random" and not job.get("random_visual"):
            prepared_path = await _prepare_random_visual_video(status_message, settings, job_dir, input_path)
            if prepared_path is None:
                skipped += 1
                _save_job_snapshot(job_dir, job, status="rejected", error="random_visual_recovery_failed")
                continue
            job["input_path"] = str(prepared_path)
            _save_job_snapshot(job_dir, job, status="queued", random_visual=True)

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


# Downloading the package, unpacking it and starting Python again takes well
# under this on the laptop; anything longer is a real absence worth hearing.
WORKER_UPDATE_QUIET_SECONDS = 300.0


async def _worker_presence_loop(application: object) -> None:
    """Says out loud when the worker goes missing, and when it returns.

    The bot always knew - workers check in and the TTL decides - but the fact
    only showed up in /queue, so an absence was found by accident hours later.
    The worker takes 43% of the jobs; those hours are paid for in queue time.
    """
    settings: BotSettings = application.bot_data["settings"]
    scheduler: _JobScheduler = application.bot_data["job_scheduler"]
    presence = WorkerPresence(settings.workdir / "worker_presence.json")
    admins = sorted(settings.admin_users)
    if not admins:
        return
    await asyncio.sleep(60)
    while True:
        try:
            live = await scheduler.snapshot()
            # A worker that just took an update is restarting into it, which is
            # a minute or two of silence we caused ourselves. Not news.
            served_at = float(application.bot_data.get("worker_update_served_at") or 0.0)
            if time.time() - served_at < WORKER_UPDATE_QUIET_SECONDS:
                await asyncio.sleep(60)
                continue
            message = presence.observe(int(live.get("remote_workers_online") or 0))
            if message:
                print(f"Worker presence: {message.splitlines()[0]}", flush=True)
                for admin_id in admins:
                    with contextlib.suppress(Exception):
                        await application.bot.send_message(chat_id=admin_id, text=message)
        except Exception as exc:
            print(f"Worker presence check failed: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(60)


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

        # A job.json that no longer parses is dead weight: nothing can resume
        # it and every scan re-reads it. Age it by the file itself.
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            job = None
        if not isinstance(job, dict):
            if path.stat().st_mtime > cutoff:
                continue
            size = _directory_size(job_dir)
            shutil.rmtree(job_dir, ignore_errors=True)
            deleted += 1
            bytes_freed += size
            continue

        # Past the retention window every job goes, whatever its status. Only
        # done/failed/rejected used to be swept, so a job left in running or
        # queued by a restart - or a dialog the user never answered - stayed on
        # disk and in /queue forever. A live job updates itself constantly, so
        # nothing running can be this old.
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


# Priority is "lower goes first". Premium sits at 0, ordinary at 100 minus a
# karma bonus, so the tiers are separated by a wide gap.
PREMIUM_PRIORITY = 0
# How far ahead a job returning from the worker jumps within its own tier.
CONTINUATION_PRIORITY_BOOST = 50


def _continuation_priority(priority: int, premium: bool) -> int:
    """Priority for a job coming back from the worker: ahead of its own tier,
    but never out of it. Paying users always go first, so an ordinary job -
    however far along - is floored just below premium rather than allowed to
    overtake it, whatever the boost and karma bonuses add up to."""
    boosted = priority - CONTINUATION_PRIORITY_BOOST
    if premium:
        return boosted
    return max(PREMIUM_PRIORITY + 1, boosted)


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
        queue_limit: int | None,
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
        self.queue_limit = queue_limit
        self.job_id = _remote_job_id(job)
        self.worker_id: str | None = None
        self.execution_kind: str | None = None
        self.progress: _ProgressState | None = None
        self.progress_task: asyncio.Task[Any] | None = None
        # The task running this job locally, kept so maintenance can cancel it
        # mid-flight instead of waiting for the pipeline to finish on its own.
        self.runner_task: asyncio.Task[Any] | None = None
        self.remote_last_seen_at: float | None = None
        self.remote_heartbeat_seen = False


class _JobScheduler:
    def __init__(self, settings: BotSettings) -> None:
        self._settings = settings
        self._performance = PerformanceHistory(settings.workdir)
        self._lock = asyncio.Lock()
        self._pending: list[tuple[int, int, _QueuedJob]] = []
        self._active_total = 0
        self._active_local = 0
        self._active_by_user: dict[int, int] = {}
        self._known_jobs: set[str] = set()
        self._items_by_key: dict[str, _QueuedJob] = {}
        self._leased: dict[str, _QueuedJob] = {}
        self._remote_workers: dict[str, dict[str, Any]] = {}
        # Thirty seconds was the whole reason the worker kept "disappearing".
        # A worker deep in one long step says nothing but its heartbeat, and the
        # coordinator only has to be busy for half a minute to miss it. Two
        # minutes of real silence is still plenty to notice a worker that died.
        self._remote_worker_ttl = 120.0
        self._sequence = 0

    async def reattach_if_known(self, context: Any, job: dict[str, Any], status_message: Any) -> bool:
        key = _job_queue_key(job)
        async with self._lock:
            if key not in self._known_jobs:
                return False
            await self._reattach_locked(context, key, job, status_message)
            return True

    async def _reattach_locked(
        self,
        context: Any,
        key: str,
        job: dict[str, Any],
        status_message: Any,
    ) -> None:
        existing = self._items_by_key.get(key)
        if existing is not None:
            existing.status_message = status_message
            if existing.progress is not None:
                existing.progress_task = context.application.create_task(
                    _progress_updater(status_message, existing.progress)
                )
                await _safe_edit_status(status_message, existing.progress.render())
            else:
                position = self._pending_position(existing)
                await _safe_edit_status(
                    status_message,
                    self._queue_text(existing, position) if position else (
                        f"Работа №{_job_number(job)} восстанавливает текущий этап."
                    ),
                )
        else:
            await _safe_edit_status(
                status_message,
                f"Работа №{_job_number(job)} уже выполняется. Статус скоро обновится.",
            )

    async def enqueue(
        self,
        context: Any,
        *,
        chat_id: int | str,
        user_id: int | None,
        job: dict[str, Any],
        status_message: Any,
    ) -> bool:
        if job_duration_seconds(job) is None:
            input_path = Path(str(job.get("input_path") or ""))
            if input_path.is_file():
                try:
                    duration = await asyncio.to_thread(probe_duration, input_path)
                except Exception as exc:
                    print(
                        f"ETA duration probe failed for {input_path.name}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                else:
                    if duration > 0:
                        job["input_duration_seconds"] = round(duration, 3)
        key = _job_queue_key(job)
        async with self._lock:
            if key in self._known_jobs:
                await self._reattach_locked(context, key, job, status_message)
                return True

            self._sequence += 1
            premium = self._settings.is_paid(user_id) or self._settings.is_admin(user_id)
            priority_bonus = 0
            level_name = "Участник"
            queue_limit: int | None = None
            if not premium and user_id is not None:
                store: ProposalStore | None = context.application.bot_data.get("proposal_store")
                premium_store: PremiumStore | None = context.application.bot_data.get("premium_store")
                karma_milli = await asyncio.to_thread(store.karma_total, user_id) if store is not None else 0
                level, _subscription = await asyncio.to_thread(_effective_level, premium_store, user_id, karma_milli)
                priority_bonus = level.priority_bonus
                level_name = level.name
                queue_limit = level.queue_limit
                queued_for_user = sum(
                    1 for existing in self._items_by_key.values() if existing.user_id == user_id
                )
                if not job.get("recovered_at") and queued_for_user >= queue_limit:
                    await _safe_edit_status(
                        status_message,
                        f"Лимит задач в очереди для уровня «{level_name}»: {queue_limit}.\n"
                        "Дождись завершения одной из своих задач и попробуй снова.",
                    )
                    return False
            enqueue_attempt_at = time.time()
            first_queued_at = _coerce_float(job.get("first_queued_at")) or _coerce_float(job.get("queued_at"))
            if first_queued_at is None:
                first_queued_at = enqueue_attempt_at
            item = _QueuedJob(
                key=key,
                priority=0 if premium else 100 - priority_bonus,
                sequence=self._sequence,
                chat_id=chat_id,
                user_id=user_id,
                job=job,
                status_message=status_message,
                enqueued_at=enqueue_attempt_at,
                premium=premium,
                queue_limit=queue_limit,
            )
            job["first_queued_at"] = first_queued_at
            job["queue_attempt_at"] = item.enqueued_at
            job["queued_at"] = first_queued_at
            job["queue_priority"] = "premium" if premium else f"karma_{priority_bonus}"
            job["queue_priority_label"] = (
                "премиум" if premium else "обычный" if priority_bonus == 0 else f"+{priority_bonus} ({level_name})"
            )
            estimate = self._performance.estimate(job)
            if estimate is not None:
                job["initial_eta_seconds"] = round(estimate.seconds, 3)
                job["initial_eta_low_seconds"] = round(estimate.low_seconds, 3)
                job["initial_eta_high_seconds"] = round(estimate.high_seconds, 3)
                job["initial_eta_samples"] = estimate.sample_count
            self._known_jobs.add(key)
            self._items_by_key[key] = item
            heapq.heappush(self._pending, (item.priority, item.sequence, item))
            _save_job_snapshot(Path(job["job_dir"]), job, status="queued")
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()
            return True

    async def finish(self, context: Any, item: _QueuedJob) -> None:
        async with self._lock:
            snapshot = _load_job_snapshot(Path(str(item.job.get("job_dir") or "")))
            if snapshot is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        record_terminal_job,
                        Path(str(snapshot["job_dir"])),
                        snapshot,
                    )
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
            self._items_by_key.pop(item.key, None)
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()

    async def maintenance_changed(self, context: Any) -> None:
        interrupted = 0
        async with self._lock:
            if _maintenance_enabled(self._settings):
                # Turning maintenance on stops the machine now rather than
                # letting whatever is mid-render keep the GPU busy for another
                # hour. Cancelling the runner unwinds _run_pipeline_isolated,
                # whose finally kills the pipeline process tree.
                for item in list(self._items_by_key.values()):
                    if self._settings.is_admin(item.user_id):
                        continue
                    task = item.runner_task
                    if task is not None and not task.done():
                        task.cancel()
                        interrupted += 1
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()
        if interrupted:
            print(f"Maintenance: interrupted {interrupted} running job(s)", flush=True)

    async def watch_remote_leases(self, context: Any) -> None:
        while True:
            await asyncio.sleep(15.0)
            now = time.time()
            async with self._lock:
                stale = [
                    item
                    for item in self._leased.values()
                    if item.execution_kind == "remote_preprocess"
                    and now - float(item.remote_last_seen_at or 0.0)
                    # Heartbeats come every 10 seconds, so 90 was nine misses -
                    # except one slow post could block the worker's heartbeat
                    # thread for longer than that on its own, and the job was
                    # taken away from a worker still running it. Nine jobs were
                    # lost that way and redone on the main PC.
                    > (240.0 if item.remote_heartbeat_seen else 1200.0)
                ]
                await self._dispatch_locked(context)
            for item in stale:
                silence = now - float(item.remote_last_seen_at or 0.0)
                print(
                    f"Remote preprocessing heartbeat timed out job={item.job_id} "
                    f"worker={item.worker_id} silence={silence:.0f}s",
                    flush=True,
                )
                await self._fallback_remote_preprocess(context, item, "worker heartbeat timed out")

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
            remote_stage = _remote_stage_for_job(item.job)
            item.worker_id = worker_id
            item.execution_kind = "remote_preprocess" if remote_stage == "preprocess" else "remote"
            item.remote_last_seen_at = time.time()
            item.remote_heartbeat_seen = False
            item.job.setdefault("started_at", time.time())
            item.job["worker_id"] = worker_id
            item.job["is_paid"] = self._settings.is_paid(item.user_id)
            if remote_stage == "preprocess":
                item.job["remote_preprocess_started_at"] = time.time()
            _save_job_snapshot(Path(item.job["job_dir"]), item.job, status="running")
            item.progress = _ProgressState(
                "Raw Whisper" if item.job.get("mode") == "raw_text" else "Full dubbing",
                _job_number(item.job),
                estimated_total_seconds=_coerce_float(item.job.get("initial_eta_seconds")),
            )
            item.progress.update("Remote worker leased", 1, 100, worker_id)
            item.progress_task = context.application.create_task(_progress_updater(item.status_message, item.progress))
            self._leased[item.job_id] = item
            self._mark_remote_worker_locked(worker_id, active_job_id=item.job_id)
            await _safe_edit_status(item.status_message, item.progress.render())
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()
            payload = _remote_job_payload(item.job)
            if remote_stage == "preprocess":
                payload["remote_stage"] = "preprocess"
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
            if item is None:
                # The job is gone - reclaimed after a heartbeat timeout, or
                # already finished elsewhere - but the worker plainly is not.
                # Dropping the whole post used to make a busy worker look
                # offline until it finished and asked for its next lease.
                reported = str(payload.get("worker_id") or "").strip()
                if reported:
                    self._mark_remote_worker_locked(reported, active_job_id=None)
                return
            item.remote_last_seen_at = time.time()
            if item.worker_id:
                self._mark_remote_worker_locked(item.worker_id, active_job_id=item.job_id)
            if payload.get("heartbeat_only"):
                item.remote_heartbeat_seen = True
                return
            if item.progress is None:
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
            item.remote_last_seen_at = time.time()
            if item.worker_id:
                self._mark_remote_worker_locked(item.worker_id, active_job_id=None)
            if item.execution_kind == "remote_preprocess":
                context.application.create_task(self._finalize_remote_preprocess(context, item, manifest))
            else:
                context.application.create_task(self._finalize_remote_item(context, item, manifest))

    async def fail_remote(self, context: Any, job_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            item = self._leased.get(job_id)
            if item is None:
                raise RuntimeError(f"Unknown leased job: {job_id}")
            if item.worker_id:
                self._mark_remote_worker_locked(item.worker_id, active_job_id=None)
            if item.execution_kind == "remote_preprocess":
                details = str(payload.get("error") or "Remote preprocessing failed")
                context.application.create_task(
                    self._fallback_remote_preprocess(context, item, f"worker error: {details}")
                )
            else:
                context.application.create_task(self._fail_remote_item(context, item, payload))

    async def _finalize_remote_preprocess(
        self,
        context: Any,
        item: _QueuedJob,
        manifest: dict[str, Any],
    ) -> None:
        try:
            preprocess_info = manifest.get("preprocess")
            if not isinstance(preprocess_info, dict):
                raise RuntimeError("Worker completed preprocessing without an artifact package.")
            archive = _remote_result_file(
                Path(str(item.job["job_dir"])),
                "documents",
                str(preprocess_info.get("filename") or ""),
            )
            await asyncio.to_thread(_install_preprocess_bundle, archive, Path(str(item.job["job_dir"])))
            completed_at = time.time()
            item.job["remote_preprocess_completed_at"] = completed_at
            item.job["remote_preprocess_seconds"] = float(
                manifest.get("preprocess_seconds")
                or max(0.0, completed_at - float(item.job.get("remote_preprocess_started_at") or completed_at))
            )
            item.job["remote_preprocess_worker"] = item.worker_id
            await self._requeue_after_remote_preprocess(context, item, fallback=False)
        except Exception as exc:
            print("Remote preprocessing import failed:\n" + traceback.format_exc(), flush=True)
            await self._fallback_remote_preprocess(
                context,
                item,
                f"artifact import failed: {type(exc).__name__}: {exc}",
            )

    async def _fallback_remote_preprocess(self, context: Any, item: _QueuedJob, reason: str) -> None:
        item.job["remote_preprocess_fallback"] = reason
        item.job["force_local"] = True
        await self._requeue_after_remote_preprocess(context, item, fallback=True)

    async def _requeue_after_remote_preprocess(
        self,
        context: Any,
        item: _QueuedJob,
        *,
        fallback: bool,
    ) -> None:
        async with self._lock:
            if self._leased.pop(item.job_id, None) is None:
                return
            if item.worker_id:
                self._mark_remote_worker_locked(item.worker_id, active_job_id=None)
            self._active_total = max(0, self._active_total - 1)
            if item.user_id is not None:
                current = self._active_by_user.get(item.user_id, 0) - 1
                if current > 0:
                    self._active_by_user[item.user_id] = current
                else:
                    self._active_by_user.pop(item.user_id, None)
            if item.progress_task is not None:
                item.progress_task.cancel()
            if item.progress is not None:
                item.job["stage_seconds"] = merge_stage_seconds(
                    item.job.get("stage_seconds"), item.progress.stage_seconds()
                )
            item.progress_task = None
            item.progress = None
            item.execution_kind = None
            item.remote_last_seen_at = None
            # A job coming back from the worker is half finished, so it goes
            # ahead of jobs in its tier that have not started. Waiting behind
            # them was costing 48% of all processing time across 694 jobs -
            # 162 hours of a job sitting ready while the single local slot
            # worked through arrivals that were nowhere near done.
            if not item.job.get("continuation_boosted"):
                item.priority = _continuation_priority(item.priority, item.premium)
                item.job["continuation_boosted"] = True
            heapq.heappush(self._pending, (item.priority, item.sequence, item))
            detail = "ноут недоступен, продолжаю локально" if fallback else "подготовка с ноутбука получена"
            _save_job_snapshot(Path(item.job["job_dir"]), item.job, status="queued", split_stage=detail)
            await self._dispatch_locked(context)
            await self._refresh_pending_locked()

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
            await _refund_failed_job(context, item)
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
            item.job.setdefault("started_at", time.time())
            if item.job.get("remote_preprocess_completed_at"):
                item.job["local_continuation_started_at"] = time.time()
            _save_job_snapshot(Path(item.job["job_dir"]), item.job, status="starting")
            title = "Сырой Whisper" if item.job.get("mode") == "raw_text" else "Полноценный дубляж"
            item.progress = _ProgressState(
                title,
                _job_number(item.job),
                estimated_total_seconds=_coerce_float(item.job.get("initial_eta_seconds")),
            )
            item.runner_task = context.application.create_task(self._run_item(context, item))

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
        if _maintenance_enabled(self._settings) and not self._settings.is_admin(item.user_id):
            return False
        if execution_kind == "remote" and _target_lang_value(item.job.get("target_lang")) != "ru":
            return False
        if execution_kind == "remote" and (
            item.job.get("force_local") or item.job.get("remote_preprocess_completed_at")
        ):
            return False
        if (
            execution_kind == "local"
            and self._settings.executor_mode == "hybrid"
            and not item.job.get("force_local")
            and not item.job.get("remote_preprocess_completed_at")
            and _remote_stage_for_job(item.job) == "preprocess"
            and self._remote_worker_counts_locked()["idle"] > 0
        ):
            return False
        if item.user_id is None:
            return True
        return self._active_by_user.get(item.user_id, 0) < self._settings.max_active_jobs_per_user

    async def _run_item(self, context: Any, item: _QueuedJob) -> None:
        interrupted = False
        job_failed = False
        try:
            if item.progress is None:
                title = "Сырой Whisper" if item.job.get("mode") == "raw_text" else "Полноценный дубляж"
                item.progress = _ProgressState(title, _job_number(item.job))
            await _process_job(
                context,
                item.chat_id,
                item.user_id,
                item.job,
                item.status_message,
                progress_state=item.progress,
            )
            snapshot = _load_job_snapshot(Path(str(item.job["job_dir"])))
            if (
                snapshot is not None
                and str(snapshot.get("status") or "") == "done"
                and item.job.get("remote_preprocess_completed_at")
            ):
                finished_at = time.time()
                snapshot["split_completed_at"] = finished_at
                snapshot["local_continuation_seconds"] = max(
                    0.0,
                    finished_at - float(item.job.get("local_continuation_started_at") or finished_at),
                )
                snapshot["split_total_seconds"] = max(
                    0.0,
                    finished_at - float(item.job.get("remote_preprocess_started_at") or finished_at),
                )
                item.job.update(snapshot)
                _save_job_snapshot(Path(str(item.job["job_dir"])), snapshot, status="done")
            job_failed = snapshot is not None and str(snapshot.get("status") or "") == "failed"
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            if job_failed:
                await _refund_failed_job(context, item)
            if interrupted:
                # This task is already unwinding a cancellation, so awaiting the
                # cleanup here would be cut short - hand it to a fresh task.
                context.application.create_task(self._finish_interrupted(context, item))
            else:
                await self.finish(context, item)

    async def _finish_interrupted(self, context: Any, item: _QueuedJob) -> None:
        """Puts a maintenance-cancelled job back into a resumable state."""
        job_dir = Path(str(item.job["job_dir"]))
        with contextlib.suppress(Exception):
            # "queued" is recoverable, so the pipeline picks up from its resume
            # state instead of redoing the stages it already paid for.
            _save_job_snapshot(job_dir, item.job, status="queued", interrupted_by="maintenance")
        await self.finish(context, item)
        with contextlib.suppress(Exception):
            await _safe_edit_status(
                item.status_message,
                f"Задача №{_job_number(item.job)} остановлена: включены технические работы.\n"
                f"Продолжить, когда работы закончатся: /resume {_job_number(item.job)}",
            )

    async def _refresh_pending_locked(self) -> None:
        pending = sorted((priority, sequence, item) for priority, sequence, item in self._pending)
        for position, (_, _, item) in enumerate(pending, start=1):
            await _safe_edit_status(item.status_message, self._queue_text(item, position))

    def _pending_position(self, target: _QueuedJob) -> int | None:
        pending = sorted((priority, sequence, item) for priority, sequence, item in self._pending)
        for position, (_, _, item) in enumerate(pending, start=1):
            if item is target:
                return position
        return None

    def _queue_text(self, item: _QueuedJob, position: int) -> str:
        title = "Сырой Whisper" if item.job.get("mode") == "raw_text" else "Полноценный дубляж"
        job_number = _job_number(item.job)
        if _maintenance_enabled(self._settings) and not self._settings.is_admin(item.user_id):
            return "\n".join(
                [
                    f"{title}: Приостановлен",
                    f"Работа №{job_number}",
                    MAINTENANCE_MESSAGE,
                    "Задача сохранена и продолжится после завершения работ.",
                ]
            )
        active_for_user = self._active_by_user.get(item.user_id, 0) if item.user_id is not None else 0
        tier = str(item.job.get("queue_priority_label") or ("премиум" if item.premium else "обычный"))
        target_label = _target_lang_label(item.job.get("target_lang"))
        tts_label = _tts_method_label(item.job.get("tts_provider") or self._settings.tts)
        lines = [
            f"{title}: В очереди",
            f"Работа №{job_number}",
            f"Позиция: {position}",
            f"Сейчас выполняется: {self._active_total}/{self._settings.max_active_jobs}",
            f"У тебя выполняется: {active_for_user}/{self._settings.max_active_jobs_per_user}",
        ]
        wait_range = self._queue_wait_range(item)
        if wait_range is not None:
            lines.append(f"До начала примерно: {_format_eta_range(*wait_range)}")
        if item.queue_limit is not None and item.user_id is not None:
            jobs_for_user = sum(
                1 for existing in self._items_by_key.values() if existing.user_id == item.user_id
            )
            lines.append(f"Твоих задач в системе: {jobs_for_user}/{item.queue_limit}")
        if item.job.get("daily_trimmed"):
            lines.append(
                "Обрезано по дневному лимиту: "
                f"{_format_duration_ms(item.job.get('daily_original_duration_ms') or 0)} → "
                f"{_format_duration_ms(item.job.get('daily_trimmed_duration_ms') or 0)}"
            )
        lines.extend(
            [
                f"Язык озвучки: {target_label}",
                f"Движок: {tts_label}",
                f"Приоритет: {tier}",
            ]
        )
        return "\n".join(lines)

    def _queue_wait_range(self, target: _QueuedJob) -> tuple[float, float] | None:
        pending = sorted((priority, sequence, item) for priority, sequence, item in self._pending)
        ahead: list[_QueuedJob] = []
        for _priority, _sequence, item in pending:
            if item is target:
                break
            ahead.append(item)

        work_seconds = 0.0
        usable_samples = 0
        pending_ids = {id(item) for _priority, _sequence, item in pending}
        for item in self._items_by_key.values():
            if id(item) in pending_ids or item is target or item.execution_kind is None:
                continue
            if item.progress is not None:
                remaining = item.progress.remaining_seconds()
            else:
                estimate = self._performance.estimate(item.job)
                remaining = estimate.seconds * 0.5 if estimate is not None else None
            if remaining is not None:
                work_seconds += remaining
                usable_samples += 1
        for item in ahead:
            estimate = self._performance.estimate(item.job)
            if estimate is not None:
                work_seconds += estimate.seconds
                usable_samples += 1
        if usable_samples == 0 or work_seconds < 20:
            return None
        slots = max(1, self._settings.max_active_jobs)
        wait = work_seconds / slots
        return (max(15.0, wait * 0.65), max(30.0, wait * 1.45))

    def note_worker_seen(self, worker_id: str | None, job_id: str | None = None) -> None:
        """Records that a worker just spoke to us. Called from the HTTP thread.

        Deliberately takes no lock and touches no event loop: the point is that
        a busy coordinator must never be able to make a live worker look dead.
        Dict and attribute writes are atomic enough for a timestamp.
        """
        now = time.time()
        if job_id:
            item = self._leased.get(job_id)
            if item is not None:
                item.remote_last_seen_at = now
                if not worker_id:
                    worker_id = item.worker_id
        name = str(worker_id or "").strip()
        if not name:
            return
        state = dict(self._remote_workers.get(name) or {})
        state["last_seen"] = now
        state.setdefault("active_job_id", None)
        self._remote_workers[name] = state

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


def _remote_stage_for_job(job: dict[str, Any]) -> str:
    if str(job.get("mode") or "dub") == "raw_text":
        return "complete"
    provider = _tts_provider_value(job.get("tts_provider")) or "moss"
    return "preprocess" if provider in {"qwen3", "cosyvoice", "moss"} else "complete"


def _safe_upload_name(value: str) -> str:
    value = Path(str(value)).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._ ")
    return value[:180]


def _job_status_counts(workdir: Path, *, newer_than: float | None = None) -> dict[str, int]:
    """Counts job.json files by status, skipping terminal ones (done/failed/
    rejected) - this feeds the "stuck job" cross-check against the live
    scheduler, and the retention window means those already number in the
    thousands, drowning out anything actually relevant to /queue."""
    counts: dict[str, int] = {}
    if not workdir.exists():
        return counts
    # Jobs always live at <workdir>/<user id>/<job number>/job.json.  A
    # recursive walk also descends into every job's large ``work`` tree on F:,
    # which made /queue take minutes once the archive grew to thousands of
    # jobs.  Inspect the fixed layout and, for the interactive status command,
    # do not even open inactive user trees from the old archive.
    paths: list[Path] = []
    for user_dir in workdir.iterdir():
        try:
            if not user_dir.is_dir():
                continue
            if newer_than is not None and user_dir.stat().st_mtime < newer_than:
                continue
            for job_dir in user_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                if newer_than is not None and job_dir.stat().st_mtime < newer_than:
                    continue
                path = job_dir / "job.json"
                if path.is_file():
                    paths.append(path)
        except OSError:
            continue

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            status = "bad_json"
        else:
            status = str(data.get("status") or "unknown")
        if status in CLEANUP_JOB_STATUSES:
            continue
        counts[status] = counts.get(status, 0) + 1
    return counts


class _ProgressState:
    def __init__(
        self,
        title: str,
        job_number: str = "",
        *,
        estimated_total_seconds: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._stage_started_at = self._started_at
        self._title = title
        self._job_number = job_number
        self._stage = "В очереди"
        self._stage_seconds: dict[str, float] = {}
        self._estimated_total_seconds = (
            max(0.0, float(estimated_total_seconds))
            if estimated_total_seconds is not None
            else None
        )
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
            if stage and stage != self._stage:
                self._close_current_stage_locked()
                self._stage = stage
                self._stage_started_at = time.monotonic()
            if total is not None and total > 0:
                self._total = total
            if current is not None:
                self._current = max(0, min(current, self._total))
            if detail is not None:
                self._detail = detail

    def finish(self, stage: str, *, failed: bool = False, detail: str | None = None) -> None:
        with self._lock:
            self._close_current_stage_locked()
            self._stage = stage
            self._stage_started_at = time.monotonic()
            self._current = self._total
            self._done = True
            self._failed = failed
            if detail is not None:
                self._detail = detail

    def is_done(self) -> bool:
        with self._lock:
            return self._done

    def elapsed_seconds(self) -> float:
        with self._lock:
            return max(0.0, time.monotonic() - self._started_at)

    def remaining_seconds(self) -> float | None:
        with self._lock:
            if self._done:
                return None
            elapsed = max(0.0, time.monotonic() - self._started_at)
            percent = round(self._current * 100 / max(1, self._total))
            if self._estimated_total_seconds is None:
                if 5 <= percent < 98 and elapsed >= 30:
                    return min(8 * 60 * 60.0, elapsed * (100 - percent) / max(1, percent))
                return None
            remaining = max(0.0, self._estimated_total_seconds - elapsed)
            if 10 <= percent < 98 and elapsed >= 30:
                pace_remaining = elapsed * (100 - percent) / max(1, percent)
                remaining = max(
                    remaining,
                    min(pace_remaining, self._estimated_total_seconds * 3.0),
                )
            return remaining if remaining >= 15 else None

    def stage_seconds(self) -> dict[str, float]:
        with self._lock:
            result = dict(self._stage_seconds)
            if not self._done and self._stage:
                elapsed = max(0.0, time.monotonic() - self._stage_started_at)
                result[self._stage] = result.get(self._stage, 0.0) + elapsed
            return {key: round(value, 3) for key, value in result.items()}

    def _close_current_stage_locked(self) -> None:
        if not self._stage:
            return
        elapsed = max(0.0, time.monotonic() - self._stage_started_at)
        self._stage_seconds[self._stage] = self._stage_seconds.get(self._stage, 0.0) + elapsed

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
            *([f"Работа №{self._job_number}"] if self._job_number else []),
            f"{_progress_bar(percent)} {percent}%",
            f"Этап: {stage}",
            f"Прошло: {_format_elapsed(elapsed)}",
        ]
        if detail:
            lines.append(f"Детали: {_short_status_detail(detail)}")
        remaining = self.remaining_seconds()
        if remaining is not None and not done:
            lines.append(f"Осталось примерно: {_format_eta_range(remaining * 0.7, remaining * 1.4)}")
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


def _format_eta_range(low_seconds: float, high_seconds: float) -> str:
    low_seconds = max(0.0, float(low_seconds))
    high_seconds = max(low_seconds, float(high_seconds))
    if high_seconds < 60:
        return "меньше минуты"
    low_minutes = max(1, int(low_seconds // 60))
    high_minutes = max(low_minutes + 1, int(math.ceil(high_seconds / 60)))
    if high_minutes < 60:
        return f"{low_minutes}–{high_minutes} мин"
    low_hours = low_minutes / 60
    high_hours = high_minutes / 60
    if low_hours >= 1:
        return f"{low_hours:.1f}–{high_hours:.1f} ч".replace(".0", "")
    return f"{low_minutes} мин – {high_hours:.1f} ч".replace(".0", "")


def _capture_progress_metrics(job: dict[str, Any], progress: _ProgressState | None) -> None:
    if progress is None:
        return
    job["stage_seconds"] = merge_stage_seconds(job.get("stage_seconds"), progress.stage_seconds())


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


def _source_lang_label(value: Any) -> str:
    code = str(value or "auto").strip()
    return next((label for item, label in SOURCE_LANGS if item == code), code)


def _queued_detail(job: dict[str, Any], target_lang: Any) -> str:
    # An arrow rather than "с вьетнамского на русский": the language names are
    # stored in the nominative, and declining them correctly is a bigger
    # problem than it is worth solving for a status line.
    source = str(job.get("source_lang") or "auto")
    where_from = "Любой язык" if source == "auto" else _source_lang_label(source)
    voices = _speaker_count_label(job.get("speaker_count")).lower()
    return f"{where_from} → {_target_lang_label(target_lang)}, голоса: {voices}"


def _target_lang_label(value: Any) -> str:
    target_lang = _target_lang_value(value)
    return next((label for code, label in TARGET_LANGS if code == target_lang), target_lang)


def _translation_chaos_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "normal": "normal",
        "clean": "normal",
        "аккуратно": "normal",
        "нормально": "normal",
        "crooked": "crooked",
        "current": "crooked",
        "криво": "crooked",
        "nightmare": "nightmare",
        "кошмар": "nightmare",
        "кошмарно": "nightmare",
        "destroy": "destroy",
        "full": "destroy",
        "garbage": "destroy",
        "уничтожить": "destroy",
        "распад": "destroy",
    }
    return aliases.get(text)


def _ensure_translation_seed(job: dict[str, Any]) -> str:
    seed = str(job.get("translation_seed") or "").strip()
    if not seed:
        job_dir = str(job.get("job_dir") or "").strip()
        seed = Path(job_dir).name if job_dir else "unknown"
        job["translation_seed"] = seed
    return seed


def _tts_provider_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"qwen3", "qwen3-tts", "qwen3tts"}:
        return "qwen3"
    if text in {"f5", "f5-tts", "f5tts"}:
        return "f5"
    if text in {"cosyvoice", "cosyvoice-tts", "cosyvoicetts", "cosy"}:
        return "cosyvoice"
    if text in {"moss", "moss-tts", "mosstts", "moss-v1.5"}:
        return "moss"
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
    config.speaker_count_exact = count
    if count is None:
        return
    config.max_speaker_clusters = count
    if count <= 1:
        config.multi_speaker = False
        config.speaker_clustering = False
    else:
        config.multi_speaker = True
        config.speaker_clustering = True


def _apply_translation_chaos(config: DubConfig, job: dict[str, Any], settings: BotSettings) -> None:
    chaos = _translation_chaos_value(job.get("translation_chaos")) or "crooked"
    job["translation_chaos"] = chaos
    config.translation_chaos = chaos
    config.translation_seed = _ensure_translation_seed(job)

    if config.content_chaos_backbone:
        config.translation_chaos = "destroy"
        config.translation_pivots = (
            f"{settings.translation_pivots}|"
            "input,ja,ko,tr,ar,en|input,zh,ja,ko,en|"
            "en,ja,ko,tr,en|en,th,he,en"
        )
        config.translation_second_pass_ratio = max(settings.translation_second_pass_ratio, 0.72)
        # Was pinned at 0.20, which silently ignored LALADUB_ARTIFACT_RATIO in
        # the mode nearly every job uses (3416 of 3552). The floor keeps the
        # old density as the minimum; raising the setting now actually works.
        config.artifact_ratio = max(settings.artifact_ratio, 0.20)
        config.artifact_max_segments = max(config.artifact_max_segments, 64)
        return

    if chaos == "normal":
        config.translation_pivots = "input,en|en,de|en,fr|en,es"
        config.translation_second_pass_ratio = 0.12
        config.artifact_ratio = min(config.artifact_ratio, 0.10)
        config.artifact_max_segments = min(config.artifact_max_segments, 16)
        return

    if chaos == "crooked":
        config.translation_pivots = settings.translation_pivots
        config.translation_second_pass_ratio = settings.translation_second_pass_ratio
        return

    if chaos == "nightmare":
        config.translation_pivots = (
            f"{settings.translation_pivots}|"
            "input,ja,ko,en|input,tr,ar,he,en|input,zh,ja,en|"
            "en,de,fr,es,en|en,ja,ko,tr,en|en,th,he,ar,en|en,ms,he,ar,en|"
            "input,en,ja,ko,tr,ar,en"
        )
        config.translation_second_pass_ratio = max(settings.translation_second_pass_ratio, 0.70)
        config.artifact_ratio = max(config.artifact_ratio, 0.30)
        config.artifact_max_segments = max(config.artifact_max_segments, 64)
        return

    if chaos == "destroy":
        config.translation_pivots = (
            f"{settings.translation_pivots}|"
            "input,ja,ko,tr,ar,en|"
            "input,zh,ja,ko,en|"
            "en,ja,ko,tr,en"
        )
        config.translation_second_pass_ratio = max(settings.translation_second_pass_ratio, 0.55)
        config.artifact_ratio = max(config.artifact_ratio, 0.45)
        config.artifact_max_segments = max(config.artifact_max_segments, 96)


async def _progress_updater(message: Any, progress: _ProgressState, interval_seconds: float = 30.0) -> None:
    last_text = getattr(message, "text", "") or ""
    while True:
        text = progress.render()
        if text != last_text:
            await _safe_edit_status(message, text)
            last_text = text
        if progress.is_done():
            return
        await asyncio.sleep(interval_seconds)


async def _safe_edit_status(message: Any, text: str) -> bool:
    global _TELEGRAM_EDIT_BACKOFF_UNTIL, _TELEGRAM_EDIT_BACKOFF_LOGGED_UNTIL

    if message is None:
        return False
    key = (getattr(getattr(message, "chat", None), "id", None), getattr(message, "message_id", None))
    if _LAST_STATUS_TEXT.get(key) == text:
        return True
    now = time.monotonic()
    if now < _TELEGRAM_EDIT_BACKOFF_UNTIL:
        return False
    try:
        await message.edit_text(text)
        _LAST_STATUS_TEXT[key] = text
        return True
    except Exception as exc:
        if "Message is not modified" in str(exc):
            _LAST_STATUS_TEXT[key] = text
            return True
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                seconds = float(retry_after.total_seconds())
            except AttributeError:
                seconds = float(retry_after)
            except (TypeError, ValueError):
                seconds = 60.0
            seconds = max(1.0, seconds)
            _TELEGRAM_EDIT_BACKOFF_UNTIL = max(_TELEGRAM_EDIT_BACKOFF_UNTIL, now + seconds)
            if _TELEGRAM_EDIT_BACKOFF_UNTIL > _TELEGRAM_EDIT_BACKOFF_LOGGED_UNTIL:
                _TELEGRAM_EDIT_BACKOFF_LOGGED_UNTIL = _TELEGRAM_EDIT_BACKOFF_UNTIL
                print(f"Telegram progress edits paused for {seconds:.0f}s after flood control", flush=True)
            return False
        print(f"Progress edit skipped: {type(exc).__name__}: {exc}", flush=True)
        return False


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
    config.reference_timing_asr = False
    config.artifact_source_lang = None
    config.input_pivot_lang = None
    config.inject_artifacts = False
    config.artifact_chaos_mode = False
    config.content_chaos_backbone = False
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
    elif chaos_backbone and selected_source:
        # Keep the deliberately wrong ASR pass as a controlled corruption
        # source, while the automatic pass supplies content and timings.
        config.source_lang = selected_source
        config.force_source_language = True
        config.asr_retry_on_repetition = False
        config.asr_fallback_on_sparse = True
        config.reference_timing_asr = True
        config.input_pivot_lang = selected_source
        config.artifact_source_lang = selected_source
        config.inject_artifacts = True
        config.artifact_chaos_mode = True
        config.content_chaos_backbone = True
        config.glitch_profile = "faithful"
        config.collapse_repetitions = True
        config.distort_main_translation = True
    elif chaos_backbone:
        config.source_lang = None
        config.force_source_language = False
        config.asr_retry_on_repetition = True
        config.asr_fallback_on_sparse = False
        config.reference_timing_asr = False
        config.input_pivot_lang = None
        config.artifact_source_lang = None
        config.inject_artifacts = False
        config.artifact_chaos_mode = False
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
    progress_state: _ProgressState | None = None,
) -> None:
    settings: BotSettings = context.application.bot_data["settings"]
    job_dir = Path(job["job_dir"])
    output_path = job_dir / "dubbed.mp4"
    target_lang = _target_lang_value(job.get("target_lang"))
    job["target_lang"] = target_lang
    tts_provider = _tts_provider_value(job.get("tts_provider")) or settings.tts
    if target_lang == "uk" and tts_provider.lower() in {
        "qwen3",
        "qwen3-tts",
        "qwen3tts",
        "cosyvoice",
        "cosyvoice-tts",
        "cosyvoicetts",
        "cosy",
        "moss",
        "moss-tts",
        "mosstts",
        "moss-v1.5",
    }:
        tts_provider = "f5"
    job["translation_chaos"] = _translation_chaos_value(job.get("translation_chaos")) or "crooked"
    _ensure_translation_seed(job)
    _save_job_snapshot(job_dir, job, status="running")
    progress: _ProgressState | None = progress_state
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
        cosyvoice_python=settings.cosyvoice_python,
        cosyvoice_repo_dir=settings.cosyvoice_repo_dir,
        cosyvoice_model_dir=settings.cosyvoice_model_dir,
        cosyvoice_model_id=settings.cosyvoice_model_id,
        cosyvoice_mode=settings.cosyvoice_mode,
        cosyvoice_instruction=settings.cosyvoice_instruction,
        cosyvoice_device=settings.cosyvoice_device,
        cosyvoice_speed=settings.cosyvoice_speed,
        cosyvoice_timeout_seconds=settings.cosyvoice_timeout_seconds,
        moss_python=settings.moss_python,
        moss_model_dir=settings.moss_model_dir,
        moss_codec_dir=settings.moss_codec_dir,
        moss_device=settings.moss_device,
        moss_timeout_seconds=settings.moss_timeout_seconds,
        multi_speaker=settings.multi_speaker,
        speaker_reference_seconds=settings.speaker_reference_seconds,
        speaker_clustering=settings.speaker_clustering,
        max_speaker_clusters=settings.max_speaker_clusters,
        speaker_cluster_threshold=settings.speaker_cluster_threshold,
        diarization_python=settings.diarization_python,
        diarization_model=settings.diarization_model,
        diarization_device=settings.diarization_device,
        diarization_cache_dir=settings.diarization_cache_dir,
        diarization_token_file=settings.diarization_token_file,
        diarization_timeout_seconds=settings.diarization_timeout_seconds,
        separation=settings.separation,
        separation_device=settings.separation_device,
        demucs_model=settings.demucs_model,
        audio_bed=settings.audio_bed,
        glitch_profile=job.get("glitch_profile", "clean"),
        original_volume=settings.original_volume,
        dub_volume=settings.dub_volume,
        trim_tts_silence=settings.trim_tts_silence,
        tts_max_pause_seconds=settings.tts_max_pause_seconds,
        force_source_language=False,
        suppress_plain_ascii_tokens=settings.suppress_plain_ascii_tokens,
        asr_retry_on_repetition=True,
        artifact_source_lang=job.get("source_lang") or None,
        artifact_whisper_model=settings.whisper_only_model,
        artifact_whisper_device=settings.artifact_whisper_device,
        inject_artifacts=settings.inject_artifacts,
        artifact_max_segments=settings.artifact_max_segments,
        artifact_ratio=settings.artifact_ratio,
        artifact_min_source_segments=settings.artifact_min_source_segments,
        artifact_min_gap_seconds=settings.artifact_min_gap_seconds,
        distort_translation=settings.distort_translation,
        translation_chaos=_translation_chaos_value(job.get("translation_chaos")) or "crooked",
        translation_seed=_ensure_translation_seed(job),
        translation_pivots=settings.translation_pivots,
        max_translation_hops=settings.max_translation_hops,
        channel_rebrand_share=settings.channel_rebrand_share,
        max_line_repeats=settings.max_line_repeats,
        artifact_source=settings.artifact_source,
        artifact_cross_language_share=settings.artifact_cross_language_share,
        translation_second_pass_ratio=settings.translation_second_pass_ratio,
        collapse_repetitions=settings.collapse_repetitions,
        max_phrase_repeats=settings.max_phrase_repeats,
        max_word_repeats=settings.max_word_repeats,
        censor_percent=int(job.get("censor_percent") or 0),
    )
    _apply_text_extraction_method(config, job, settings)
    _apply_translation_chaos(config, job, settings)
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
            progress = progress_state or _ProgressState("Сырой Whisper", _job_number(job))
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
            srt_path, txt_path = await _run_pipeline_isolated(
                "transcript",
                Path(job["input_path"]),
                transcript_config,
                progress,
            )
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
            _capture_progress_metrics(job, progress)
            _save_job_snapshot(job_dir, job, status="done")
            await _finish_progress(progress, progress_task, status_message, "Готово", detail="SRT, TXT и meta JSON отправлены")
            return

        progress = progress_state or _ProgressState("Полноценный дубляж", _job_number(job))
        progress.update(
            "В очереди",
            1,
            100,
            # What the person chose, in words. The engine, the ASR backend and
            # the pivot were in here too, but they are not choices any more -
            # there is one engine - and reading them told nobody anything.
            _queued_detail(job, target_lang),
        )
        if status_message is not None:
            await _safe_edit_status(status_message, progress.render())
        else:
            status_message = await context.bot.send_message(chat_id=chat_id, text=progress.render())
        progress_task = asyncio.create_task(_progress_updater(status_message, progress))

        # "Показать текст" stops the pipeline at the point the text is ready and
        # hands the decision back to the author. The machine is released while
        # they think - voicing resumes as a fresh run over the same workdir.
        awaiting_review = (
            str(job.get("review_mode") or "direct") == "review" and not job.get("review_approved")
        )
        if awaiting_review:
            config.preprocess_only = True
            config.resume = True
            await _run_pipeline_isolated("dub", Path(job["input_path"]), config, progress)
            job.setdefault("review_attempt", 1)
            _capture_progress_metrics(job, progress)
            _save_job_snapshot(job_dir, job, status="awaiting_review")
            await _finish_progress(
                progress, progress_task, status_message, "Текст готов", detail="жду решения"
            )
            await _send_text_for_review(context, chat_id, job)
            return

        [result] = await _run_pipeline_isolated(
            "dub",
            Path(job["input_path"]),
            config,
            progress,
        )
        send_path = result

        if job.get("watermark_enabled", True):
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
        _save_job_snapshot(
            job_dir,
            job,
            status="ready",
            proposal_video_path=str(send_path),
            proposal_output_filename=output_filename,
        )
        await _send_video_file(
            context.bot,
            chat_id,
            send_path,
            output_filename,
            reply_markup=_proposal_keyboard(_job_number(job)) if settings.proposal_enabled else None,
        )
        fun_visual_sent = await _send_fun_visual_if_present(
            context.bot,
            chat_id,
            job,
            progress,
        )
        if transcript_text:
            progress.update("Отправляю транскрипт", 99, 100, None)
            transcript_path = _write_transcript_text(
                job_dir,
                job.get("source_title") or Path(job["input_path"]).stem,
                transcript_text,
                job,
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
        sent_items = ["дубляж"]
        if fun_visual_sent:
            sent_items.append("исходный прикольный видеоряд")
        if transcript_text:
            sent_items.append("транскрипт")
        final_detail = "Отправлены: " + ", ".join(sent_items)
        _capture_progress_metrics(job, progress)
        _save_job_snapshot(job_dir, job, status="done")
        await _archive_finished_dub(context, job)
        await _finish_progress(progress, progress_task, status_message, "Готово", detail=final_detail)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        error_log = job_dir / "error.log"
        error_log.write_text(traceback_text, encoding="utf-8")
        _capture_progress_metrics(job, progress)
        _save_job_snapshot(job_dir, job, status="failed", error=details)
        await _finish_progress(progress, progress_task, status_message, "Ошибка", failed=True, detail=details)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Задача упала:\n{details}",
            reply_markup=_resume_keyboard(job_dir),
        )
    finally:
        await asyncio.to_thread(_clear_runtime_model_caches)


def _clear_runtime_model_caches() -> None:
    clear_openai_whisper_cache()
    clear_tts_model_caches()


async def _archive_finished_dub(context: Any, job: dict[str, Any]) -> None:
    """Copies the finished dub into the permanent library so /show keeps
    working after the job's own workdir is swept by the retention cleanup.
    Best-effort - a failure here must not touch the job the user just got."""
    library_store: LibraryStore | None = context.application.bot_data.get("library_store")
    if library_store is None:
        return
    try:
        job_dir = Path(str(job["job_dir"]))
        source_path = _find_proposal_video_path(job_dir, job)
        if source_path is None:
            return
        settings: BotSettings = context.application.bot_data["settings"]
        job_number = _job_number(job)
        settings.library_dir.mkdir(parents=True, exist_ok=True)
        dest_path = settings.library_dir / f"{job_number}{source_path.suffix}"
        await asyncio.to_thread(shutil.copy2, source_path, dest_path)
        await asyncio.to_thread(
            library_store.add,
            job_number=job_number,
            user_id=int(job.get("user_id") or 0),
            source_title=str(job.get("source_title") or ""),
            target_lang=_target_lang_value(job.get("target_lang")),
            video_path=str(dest_path),
            output_filename=str(job.get("proposal_output_filename") or dest_path.name),
        )
    except Exception as exc:
        print(f"Library archive failed for job {job.get('job_dir')}: {type(exc).__name__}: {exc}", flush=True)


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
            _capture_progress_metrics(job, progress)
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
        _save_job_snapshot(
            job_dir,
            job,
            status="ready",
            proposal_video_path=str(video_path),
            proposal_output_filename=output_filename,
        )
        settings: BotSettings = context.application.bot_data["settings"]
        await _send_video_file(
            context.bot,
            item.chat_id,
            video_path,
            output_filename,
            reply_markup=_proposal_keyboard(_job_number(job)) if settings.proposal_enabled else None,
        )
        fun_visual_sent = await _send_fun_visual_if_present(
            context.bot,
            item.chat_id,
            job,
            progress,
        )

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
        _capture_progress_metrics(job, progress)
        _save_job_snapshot(job_dir, job, status="done")
        await _archive_finished_dub(context, job)
        sent_items = ["дубляж"]
        if fun_visual_sent:
            sent_items.append("исходный прикольный видеоряд")
        if transcript_sent:
            sent_items.append("транскрипт")
        detail = "Отправлены: " + ", ".join(sent_items)
        await _finish_progress(progress, progress_task, status_message, "Done", detail=detail)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, flush=True)
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        (job_dir / "error.log").write_text(traceback_text, encoding="utf-8")
        _capture_progress_metrics(job, progress)
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


def _install_preprocess_bundle(archive_path: Path, job_dir: Path) -> None:
    job_dir = job_dir.resolve()
    incoming = job_dir / ".preprocess-incoming"
    backup = job_dir / ".preprocess-backup"
    workdir = job_dir / "work"
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    incoming.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                relative = Path(info.filename.replace("\\", "/"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Unsafe preprocessing archive member: {info.filename}")
                destination = (incoming / relative).resolve()
                try:
                    destination.relative_to(incoming.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"Unsafe preprocessing archive member: {info.filename}") from exc
            archive.extractall(incoming)
        required = (incoming / "translated.srt", incoming / "source_16k.wav", incoming / "resume_state.json")
        missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError("Preprocessing package is incomplete: " + ", ".join(missing))

        if workdir.exists():
            workdir.replace(backup)
        incoming.replace(workdir)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        if not workdir.exists() and backup.exists():
            backup.replace(workdir)
        raise
    finally:
        shutil.rmtree(incoming, ignore_errors=True)


def _job_snapshot_path(job_dir: Path) -> Path:
    return job_dir / "job.json"


def _save_job_snapshot(
    job_dir: Path,
    job: dict[str, Any],
    *,
    status: str,
    error: str | None = None,
    **updates: Any,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    job.update(updates)
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


def _find_resumable_job_by_number(
    settings: BotSettings,
    user_id: int,
    job_number: str,
) -> dict[str, Any] | None:
    if not job_number.isdigit():
        return None
    return _load_job_snapshot(settings.workdir / str(user_id) / job_number)


def _prepare_job_for_resume(job: dict[str, Any]) -> None:
    job["resume"] = "1"
    job["force_resume_requested_at"] = time.time()
    for key in ("error", "finished_at", "worker_id"):
        job.pop(key, None)


def _job_number(job: dict[str, Any]) -> str:
    job_dir = str(job.get("job_dir") or "").strip()
    return Path(job_dir).name if job_dir else "?"


def _resume_keyboard(job_dir: Path) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Продолжить задачу", callback_data=f"resume:{job_dir.name}")]]
    )


def _proposal_keyboard(job_number: str) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Отправить в предложку", callback_data=f"proposal:submit:{job_number}")]]
    )


def _proposal_submitted_keyboard(submission_id: int) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("✅ Отправлено в предложку", callback_data=f"proposal:submitted:{submission_id}")]]
    )


def _remove_reply_keyboard() -> Any:
    from telegram import ReplyKeyboardRemove

    return ReplyKeyboardRemove()


def _language_keyboard(
    prefix: str,
    items: list[tuple[str, str]],
    columns: int = 2,
    *,
    back_callback: str | None = None,
) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    rows = []
    for index in range(0, len(items), columns):
        row_items = items[index : index + columns]
        rows.append([InlineKeyboardButton(label, callback_data=f"{prefix}:{code}") for code, label in row_items])
    if back_callback:
        rows.append([InlineKeyboardButton("◀️ Назад", callback_data=back_callback)])
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


def _censor_replacement_pattern() -> "re.Pattern[str] | None":
    """Matches any phrase the censor swaps words for.

    The censor leaves no marker of its own - it just substitutes - but every
    substitution comes from a known bank, so the bank is the marker.
    """
    global _CENSOR_MARK_PATTERN
    if _CENSOR_MARK_PATTERN is not None:
        return _CENSOR_MARK_PATTERN or None
    try:
        from .censor import _ACTIVE_REPLACEMENTS
    except Exception:
        _CENSOR_MARK_PATTERN = False
        return None
    phrases = sorted({p.strip() for p in _ACTIVE_REPLACEMENTS if p and p.strip()}, key=len, reverse=True)
    if not phrases:
        _CENSOR_MARK_PATTERN = False
        return None
    _CENSOR_MARK_PATTERN = re.compile("|".join(re.escape(p) for p in phrases), re.IGNORECASE)
    return _CENSOR_MARK_PATTERN


def _artifact_texts(job_dir: Path) -> set[str]:
    """Normalised text of every artifact that was actually injected."""
    path = job_dir / "work" / "debug" / "artifact_injected.srt"
    if not path.is_file():
        return set()
    texts: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        texts.add(_transcript_key(stripped))
    return texts


def _transcript_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _mark_transcript_line(line: str, artifacts: set[str]) -> str:
    """Upper-cases whole lines that are artifacts, and censored phrases inside
    the rest, so it is visible at a glance which words are the bot's own and
    which came from the video."""
    if _transcript_key(line) in artifacts:
        return line.upper()
    pattern = _censor_replacement_pattern()
    if pattern is None:
        return line
    return pattern.sub(lambda match: match.group(0).upper(), line)


def _read_transcript_text(srt_path: Path, *, mark: bool = True) -> str:
    if not srt_path.exists():
        return ""
    # work/translated.srt -> the job folder two levels up.
    artifacts = _artifact_texts(srt_path.parent.parent) if mark else set()
    lines: list[str] = []
    for line in srt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        lines.append(_mark_transcript_line(stripped, artifacts) if mark else stripped)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _write_transcript_text(
    job_dir: Path,
    source_title: str,
    transcript_text: str,
    job: dict[str, Any] | None = None,
) -> Path:
    filename = _lalaschool_filename(f"{source_title}_transcript", ".txt")
    path = job_dir / filename
    header = transcript_header(job)
    body = transcript_text.strip()
    separator = "\n\n"
    contents = header + separator + body if header else body
    path.write_text(contents + "\n", encoding="utf-8")
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
        # A forced retry follows Telegram's HTTP 413 response. Never reuse the
        # previous cached file in that case: it may be exactly the oversized
        # file that Telegram has just rejected.
        if not force and output_path.stat().st_size <= TELEGRAM_SAFE_VIDEO_BYTES:
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


def _find_fun_visual(job: dict[str, Any]) -> Path | None:
    job_dir = Path(str(job.get("job_dir") or ""))
    input_path = Path(str(job.get("input_path") or ""))
    candidates = [input_path]
    if job_dir.is_dir():
        candidates.extend(sorted(job_dir.glob("*_fun_visual.mp4")))
    for path in candidates:
        if path.is_file() and path.stem.endswith("_fun_visual") and path.stat().st_size >= 1024:
            return path
    return None


async def _send_fun_visual_if_present(
    bot: Any,
    chat_id: int | str,
    job: dict[str, Any],
    progress: _ProgressState | None,
) -> bool:
    fun_visual = _find_fun_visual(job)
    if fun_visual is None:
        return False
    if progress is not None:
        progress.update("Отправляю прикольный видеоряд", 99, 100, fun_visual.name)
    source_title = job.get("source_title") or fun_visual.stem.removesuffix("_fun_visual")
    filename = _lalaschool_filename(f"{source_title}_fun_visual", fun_visual.suffix)
    try:
        await _send_video_file(
            bot,
            chat_id,
            fun_visual,
            filename,
            caption="Исходный прикольный видеоряд без дубляжа",
        )
        return True
    except Exception as exc:
        print(f"Fun visual send skipped: {type(exc).__name__}: {exc}", flush=True)
        return False


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


async def video_upload_metadata(video_path: Path) -> dict[str, Any]:
    """Duration and frame size to state when uploading a video.

    Telegram shows a video's length from what the upload declares, not by
    reading the file, so leaving these out makes every clip show 00:00 in
    chats and media galleries however well-formed the file is. Probing failures
    are not worth failing a send over - the upload just goes without them.
    """
    metadata: dict[str, Any] = {}
    try:
        duration = await asyncio.to_thread(probe_duration, video_path)
        if duration > 0:
            metadata["duration"] = int(round(duration))
    except Exception as exc:
        print(f"Video duration probe failed for {video_path.name}: {type(exc).__name__}: {exc}", flush=True)
    dimensions = await asyncio.to_thread(probe_video_dimensions, video_path)
    if dimensions is not None:
        metadata["width"], metadata["height"] = dimensions
    return metadata


async def _send_video_file_once(
    bot: Any,
    chat_id: int | str,
    video_path: Path,
    filename: str,
    *,
    caption: str | None = None,
    reply_markup: Any = None,
) -> None:
    metadata = await video_upload_metadata(video_path)
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
            **metadata,
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
