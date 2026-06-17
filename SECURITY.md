# Security

Do not commit real Telegram bot tokens, worker API tokens, generated videos, logs,
model folders, or runtime job folders.

Local secret/runtime paths are intentionally ignored:

- `.secrets/`
- `worker_config.json`
- `.worker_state.json`
- `runs/`
- `models/`
- `dist/`
- `updates/`
- `*.log`
- `*.pid`

If a real token is accidentally published, revoke it immediately in BotFather or
rotate the worker token and restart the release bot/workers.
