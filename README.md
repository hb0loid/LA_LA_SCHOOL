# LA LA SCHOOL Local Dubber

Увага! Этот бот на 100 процентов сгенерирован нейросетью, автор не то что за кодинг не шарит, а в принципе за гитхаб и всё из него вытекающее не судите строга :3

Локальный конвейер для перевода и дубляжа видео с Telegram-ботом. Бот оставляет два рабочих режима: сырой текст Whisper и полноценный дубляж с переводом, вырезанием оригинального вокала и XTTS voice cloning.

## Возможности

- Локальный ASR через `faster-whisper`.
- Локальный перевод через `argos-translate` или локальный LibreTranslate.
- Локальная озвучка через Windows SAPI, Piper или XTTS voice cloning.
- Вырезание оригинального голоса через Demucs: финальный микс может быть "инструментал + новый голос".
- Сборка финального видео через `ffmpeg`.
- Telegram-бот: пользователь кидает видео или ссылку на видео, выбирает входной язык и один из двух режимов.
- Очередь Telegram-задач: бот ограничивает число одновременных дубляжей и ставит paid-пользователей выше в очереди.
- Режим `Сырой текст`: только оригинальный OpenAI Whisper, без перевода/дубляжа, для ловли ASR-артефактов.
- Режим `Полноценный дубляж`: перевод, сжатие повторов, Demucs, мультиспикерные XTTS-референсы по сегментам и финальный MP4.
- Watermark для free-пользователей; paid-пользователи задаются allowlist по Telegram ID.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[asr,translate,bot,clone,separation]
```

Проверить окружение:

```powershell
laladub doctor
laladub voices
```

`argostranslate` ставит библиотеку, но языковые пакеты Argos нужно установить отдельно. Для первых тестов можно временно поставить `LALADUB_TRANSLATOR=identity`, чтобы проверить бот, ASR, TTS и сборку без реального перевода.

## CLI

Нормальный дубляж без клонирования:

```powershell
laladub dub ".\input\episode.mp4" `
  --output ".\runs\episode_ru.mp4" `
  --source-lang vi `
  --target-lang ru `
  --translator argos `
  --tts sapi `
  --voice "Microsoft Irina Desktop" `
  --glitch-profile clean
```

Дубляж с клонированием голоса и вырезанием оригинального вокала:

```powershell
laladub dub ".\input\episode.mp4" `
  --output ".\runs\episode_ru_clone.mp4" `
  --source-lang vi `
  --target-lang ru `
  --translator argos `
  --tts xtts `
  --separation demucs `
  --audio-bed instrumental `
  --original-volume 1.0 `
  --dub-volume 1.0 `
  --glitch-profile clean
```

Если есть отдельный чистый референс голоса, лучше использовать его:

```powershell
laladub dub ".\input\episode.mp4" `
  --output ".\runs\episode_ru_clone.mp4" `
  --source-lang vi `
  --target-lang ru `
  --tts xtts `
  --speaker-wav ".\voices\speaker.wav" `
  --separation demucs `
  --audio-bed instrumental
```

Если указать `--source-lang auto` в боте или не указывать `--source-lang` в CLI, Whisper попробует определить язык сам, а переводчик получит найденный язык автоматически.

Сырой/сломанный вариант:

```powershell
laladub dub ".\input\episode.mp4" `
  --output ".\runs\episode_ru_ghost.mp4" `
  --source-lang ru `
  --target-lang ru `
  --translator argos `
  --tts sapi `
  --voice "Microsoft Irina Desktop" `
  --glitch-profile ghost `
  --original-volume 0.18
```

Здесь `--source-lang ru` для вьетнамского видео как раз намеренно ломает распознавание. В Telegram-боте это делается выбором любого входного языка, даже неправильного.

## Режимы Бота

- `Сырой текст` - возвращает SRT/TXT и meta JSON из OpenAI Whisper. Неправильный входной язык намеренно усиливает ASR-артефакты.
- `Полноценный дубляж` - переводит и озвучивает видео. Входной язык тоже можно выбрать неверно, чтобы получить ранний AI-dub эффект.

## Telegram-Бот

Создай бота через BotFather и задай токен:

```powershell
$env:LALADUB_BOT_TOKEN="123456:telegram-token"
$env:LALADUB_TRANSLATOR="hybrid"
$env:LALADUB_TTS="xtts"
$env:LALADUB_VOICE="Microsoft Irina Desktop"
$env:LALADUB_SEPARATION="demucs"
$env:LALADUB_AUDIO_BED="instrumental"
$env:LALADUB_ORIGINAL_VOLUME="1.0"
$env:LALADUB_XTTS_DEVICE="cpu"
# Only if you accept Coqui XTTS CPML terms:
# $env:COQUI_TOS_AGREED="1"
$env:LALADUB_WHISPER_DEVICE="cpu"
$env:LALADUB_WHISPER_COMPUTE_TYPE="int8"
$env:LALADUB_PAID_USERS="123456789,987654321"
$env:LALADUB_FREE_MAX_DURATION_SECONDS="180"
$env:LALADUB_PAID_MAX_DURATION_SECONDS="0"
$env:LALADUB_WATERMARK_TEXT="La La Local Dub"
$env:LALADUB_WATERMARK_IMAGE=".\assets\watermark.png"
laladub-bot
```

Пользовательский сценарий:

1. Пользователь отправляет видеофайл или ссылку на видео, например YouTube.
2. Бот просит выбрать входной язык. Для `Сырого текста` можно выбрать неправильный язык, чтобы ломать Whisper; для дубляжа ASR дополнительно автоопределяет язык ради качества.
3. Бот просит выбрать режим: `Сырой текст` или `Полноценный дубляж`.
4. Бот присылает SRT/TXT или готовый MP4. Язык озвучки всегда русский. Если Telegram ID пользователя нет в `LALADUB_PAID_USERS`, на видео добавляется watermark и действует лимит длительности free-версии.

Полезные команды бота:

- `/start` - короткая инструкция.
- `/me` - показать Telegram ID и статус `free`/`paid`.
- `/cancel` - сбросить текущую задачу.

## Настройки Бота

## Запуск: Тест И Релиз

- `Start-Test-Bot.cmd` / `Stop-Test-Bot.cmd` - текущий тестовый бот. Логи: `bot.out.log`, `bot.err.log`; pid: `bot.pid`; задачи: `runs/bot`.
- `Start-Release-Bot.cmd` / `Stop-Release-Bot.cmd` - релизный бот. Логи: `bot.release.out.log`, `bot.release.err.log`; pid: `bot.release.pid`; задачи: `runs/bot-release`.
- Токен релизного бота лежит в `.secrets/Release-Bot-Token.ps1`; папка `.secrets` добавлена в `.gitignore`.

- `LALADUB_BOT_TOKEN` - токен Telegram-бота, обязательный.
- `LALADUB_PAID_USERS` - список paid Telegram ID через запятую.
- `LALADUB_MAX_ACTIVE_JOBS` - общий лимит одновременно выполняемых задач, по умолчанию `2`.
- `LALADUB_MAX_ACTIVE_JOBS_PER_USER` - лимит одновременно выполняемых задач на одного пользователя, по умолчанию `1`.
- `LALADUB_WATERMARK_IMAGE` - PNG watermark для free-пользователей. Если файла нет, используется текстовый fallback.
- `LALADUB_WATERMARK_TEXT` - текстовый fallback watermark для free-пользователей.
- `LALADUB_MAX_FILE_MB` - лимит входного видео, по умолчанию `200`.
- `LALADUB_FREE_MAX_DURATION_SECONDS` - лимит длительности для free-пользователей, по умолчанию `180` секунд.
- `LALADUB_PAID_MAX_DURATION_SECONDS` - лимит длительности для paid-пользователей, по умолчанию `0` без ограничения.
- `LALADUB_BOT_WORKDIR` - папка задач, по умолчанию `runs/bot`.
- `LALADUB_JOB_RETENTION_SECONDS` - сколько хранить завершённые задачи (`done`, `failed`, `rejected`) перед автоочисткой, по умолчанию `2592000` секунд (30 дней). `0` отключает очистку.
- `LALADUB_CLEANUP_INTERVAL_SECONDS` - как часто запускать автоочистку, по умолчанию `3600` секунд.
- `LALADUB_TRANSLATOR` - `hybrid`, `googleweb`, `mymemory`, `argos`, `libretranslate` или `identity`.
- `LALADUB_TTS` - `xtts`, `f5`, `sapi`, `piper` или `none`; `Start-Bot.ps1` по умолчанию ставит `f5`.
- `LALADUB_VOICE` - имя SAPI-голоса.
- `LALADUB_SPEAKER_WAV` - чистый WAV-референс голоса для XTTS. Если не задан, берётся вокальная дорожка после Demucs.
- `LALADUB_MULTI_SPEAKER` - `1` по умолчанию: для каждого сегмента нарезается свой speaker reference из оригинального вокала.
- `LALADUB_SPEAKER_REFERENCE_SECONDS` - длина сегментного speaker reference, по умолчанию `3.5`.
- `LALADUB_SPEAKER_CLUSTERING` - `1` по умолчанию: группирует похожие сегментные speaker references в speaker bank, чтобы разные персонажи чаще сохраняли разные голоса.
- `LALADUB_MAX_SPEAKER_CLUSTERS` - максимум кластеров голосов, по умолчанию `6`.
- `LALADUB_SPEAKER_CLUSTER_THRESHOLD` - порог объединения голосов, по умолчанию `0.08`; ниже = больше разных голосов, выше = агрессивнее склеивает.
- `LALADUB_XTTS_MODEL` - модель XTTS, по умолчанию `tts_models/multilingual/multi-dataset/xtts_v2`.
- `LALADUB_XTTS_DEVICE` - `cpu` по умолчанию. `cuda` только если CUDA/cuBLAS настроены.
- `LALADUB_F5_PYTHON` - Python из отдельного окружения F5-TTS, по умолчанию `.venv-f5tts\Scripts\python.exe`.
- `LALADUB_F5_HF_REPO` - Hugging Face repo F5-модели, по умолчанию `Misha24-10/F5-TTS_RUSSIAN`.
- `LALADUB_F5_HF_CKPT_PATH` - checkpoint внутри repo, по умолчанию `F5TTS_v1_Base_v2/model_last_inference.safetensors`.
- `LALADUB_F5_HF_VOCAB_PATH` - vocab внутри repo, по умолчанию `F5TTS_v1_Base/vocab.txt`.
- `LALADUB_F5_DEVICE` - `auto`, `cpu` или `cuda`; по умолчанию `auto`.
- `LALADUB_F5_SPEED` - скорость F5-TTS, по умолчанию `1.0`.
- `LALADUB_SEPARATION` - `demucs` или `none`.
- `LALADUB_AUDIO_BED` - `instrumental`, `original` или `dub-only`.
- `LALADUB_WHISPER_MODEL` - модель faster-whisper, по умолчанию `small`.
- `LALADUB_ASR_BACKEND` - backend обычного дубляжа, по умолчанию `faster-whisper`.
- `LALADUB_WHISPER_ONLY_MODEL` - модель для режима bug hunt и artifact-hunt, по умолчанию `turbo`.
- `LALADUB_WHISPER_ONLY_DEVICE` - устройство для bug hunt, по умолчанию `cpu`.
- `LALADUB_WHISPER_DEVICE` - `cpu` по умолчанию. Можно поставить `cuda`, если установлены CUDA/cuBLAS DLL.
- `LALADUB_WHISPER_COMPUTE_TYPE` - `int8` по умолчанию для CPU.
- `LALADUB_ORIGINAL_VOLUME` - громкость оригинала в финальном миксе.
- `LALADUB_DUB_VOLUME` - громкость дубляжа.
- `LALADUB_COLLAPSE_REPETITIONS` - `1` по умолчанию: сжимает повторяющиеся слова/фразы перед XTTS.
- `LALADUB_MAX_PHRASE_REPEATS` - сколько раз подряд можно оставить одинаковую фразу, по умолчанию `2`.
- `LALADUB_MAX_WORD_REPEATS` - сколько раз подряд можно оставить одно слово, по умолчанию `3`.

For another PC on the same LAN, unzip `LaLaDubWorker.zip`, run `Start-Worker.cmd`, and leave it open. When the release bot package changes, idle workers download the update from the main PC and restart themselves.
