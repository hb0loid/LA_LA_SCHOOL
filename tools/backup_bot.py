"""Makes a restorable snapshot of everything the bot cannot rebuild by itself.

Run it any time - it never touches the running bot, and a snapshot taken while
jobs are in flight is still consistent, because every database is copied
through SQLite's own backup API rather than by copying the file.

    python tools/backup_bot.py

What is deliberately left out is listed in the README written next to the
snapshot, along with how to restore it.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Everything the bot would lose for good. Videos and job folders are excluded
# on purpose - see the README this writes.
DATABASES = [
    ("runs/proposal/proposals.sqlite3", "karma, submissions, scheduled posts"),
    ("runs/premium/subscriptions.sqlite3", "premium subscriptions"),
    ("runs/presets/presets.sqlite3", "per-user /preset answers"),
    ("runs/library/library.sqlite3", "permanent library index"),
    ("runs/reviews/text_reviews.sqlite3", "text-review decisions"),
    ("runs/cache/translations.sqlite3", "translation cache (rebuildable, but slow to refill)"),
]

STATE_FILES = [
    "runs/library/recent_shows.json",
    "runs/proposal/last_update.json",
    "runs/bot-release/last_update.json",
]

CONFIG_FILES = [
    "admins.txt",
    "paid_users.txt",
    "БОТ - ЗАПУСТИТЬ.cmd",
    "БОТ - ОСТАНОВИТЬ.cmd",
]

CONFIG_DIRS = [
    ("tools/runtime", "runtime-scripts"),
    ("tools/admin", "admin-scripts"),
]


def _run(*args: str) -> str:
    try:
        out = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _copy_database(source: Path, target: Path) -> tuple[bool, str]:
    """Copies through SQLite's backup API, which is safe on a live database,
    then reopens the copy and checks it - a snapshot nobody verified is not
    worth having."""
    if not source.is_file():
        return False, "нет файла"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(target)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    try:
        check = sqlite3.connect(target)
        try:
            result = check.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            check.close()
    except Exception as exc:
        return False, f"проверка не прошла: {type(exc).__name__}: {exc}"
    if str(result).lower() != "ok":
        return False, f"проверка целостности: {result}"
    return True, f"{target.stat().st_size / 1024 / 1024:.1f} МБ"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backs up the bot's irreplaceable data.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("F:/LaLaSchoolData/backups"),
        help="Where snapshots are kept (default: F:/LaLaSchoolData/backups)",
    )
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.dest / f"full-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    problems: list[str] = []

    print(f"Бэкап в {out}", flush=True)

    for relative, note in DATABASES:
        ok, detail = _copy_database(ROOT / relative, out / "databases" / Path(relative).name)
        line = f"  {'OK ' if ok else 'НЕТ'} {Path(relative).name}: {detail} ({note})"
        print(line, flush=True)
        report.append(line)
        if not ok:
            problems.append(f"{relative}: {detail}")

    for relative in STATE_FILES:
        source = ROOT / relative
        if not source.is_file():
            continue
        target = out / "state" / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        report.append(f"  OK  {Path(relative).name}")

    secrets = ROOT / ".secrets"
    copied_secrets = 0
    if secrets.is_dir():
        for item in sorted(secrets.iterdir()):
            if not item.is_file():
                continue
            target = out / "secrets" / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied_secrets += 1
    print(f"  секретов скопировано: {copied_secrets}", flush=True)
    if copied_secrets == 0:
        problems.append(".secrets: ни одного файла не скопировано")

    for name in CONFIG_FILES:
        source = ROOT / name
        if source.is_file():
            target = out / "config" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    for relative, folder in CONFIG_DIRS:
        source = ROOT / relative
        if source.is_dir():
            shutil.copytree(source, out / "config" / folder, dirs_exist_ok=True)

    # Uncommitted work exists nowhere else, so it goes in as a patch plus the
    # untracked files themselves.
    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    commit = _run("git", "log", "--oneline", "-1")
    status = _run("git", "status", "--short")
    unpushed = _run("git", "log", "--oneline", f"origin/{branch}..HEAD") if branch else ""
    if status:
        (out / "uncommitted").mkdir(parents=True, exist_ok=True)
        # Written as raw bytes: reading the diff as text lets Python normalise
        # line endings, and a patch whose CRLF became LF is rejected by
        # "git apply" as corrupt - which only shows up when you try to restore.
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True
        ).stdout
        if diff:
            (out / "uncommitted" / "working-tree.patch").write_bytes(diff)
        for line in status.splitlines():
            if not line.startswith("??"):
                continue
            name = line[2:].strip().strip('"')
            source = ROOT / name
            if source.is_file():
                target = out / "uncommitted" / "untracked" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        patch_path = out / "uncommitted" / "working-tree.patch"
        if patch_path.is_file():
            # Checked in reverse: the changes are already in the working tree,
            # so a forward check always fails. If the patch un-applies cleanly
            # here it is an exact record of them, which is what restoring onto
            # a fresh clone needs.
            check = subprocess.run(
                ["git", "apply", "--check", "--reverse", str(patch_path)],
                cwd=ROOT,
                capture_output=True,
            )
            if check.returncode != 0:
                detail = (check.stderr or b"").decode("utf-8", errors="replace").strip()
                problems.append(f"патч незакоммиченного не сходится с рабочей копией: {detail}")
            else:
                report.append("  OK  патч незакоммиченного проверен")
        print(f"  незакоммиченных изменений: {len(status.splitlines())} файл(ов)", flush=True)

    readme = [
        "Бэкап La La School Dubber",
        f"Создан: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Ветка: {branch}",
        f"Коммит: {commit}",
        "",
        "Что внутри:",
        "  databases/   - карма, подписки, пресеты, библиотека, решения по тексту,",
        "                 кэш переводов. Снимок через SQLite backup API, целостность",
        "                 каждой базы проверена при создании.",
        "  state/       - защита от повторной обработки сообщений и от повторной",
        "                 отправки одной работы в чат.",
        "  secrets/     - токены ботов, Worker API, HuggingFace.",
        "  config/      - списки админов и платных, батники, скрипты запуска.",
    ]
    if status:
        readme += [
            "  uncommitted/ - изменения, которых нет ни в одном коммите:",
            "                 working-tree.patch и новые файлы в untracked/.",
        ]
    readme += [
        "",
        "Чего внутри НЕТ (и почему):",
        "  Код     - в git и запушен, отдельная копия не нужна.",
        "  Видео   - runs/library/videos уже на F:, копия на тот же физический",
        "            диск от его отказа не спасёт.",
        "  runs/   - рабочие папки задач, пересоздаются сами.",
        "",
        "Как восстановить:",
        "  1. git clone, переключиться на ветку выше.",
        "  2. secrets/ -> .secrets/, config/*.txt и батники -> корень проекта,",
        "     config/runtime-scripts -> tools/runtime.",
        "  3. databases/*.sqlite3 -> runs/proposal, runs/premium, runs/presets,",
        "     runs/library, runs/reviews, runs/cache соответственно.",
        "  4. state/*.json -> те же папки, откуда взяты (см. tools/backup_bot.py).",
    ]
    if status:
        readme += ["  5. git apply uncommitted/working-tree.patch, файлы из untracked/ на место."]
    readme += ["", "Сделано: python tools/backup_bot.py"]
    if unpushed:
        readme += ["", "Коммиты, которых не было на origin в момент бэкапа:", *[f"  {l}" for l in unpushed.splitlines()]]
    if problems:
        readme += ["", "ПРОБЛЕМЫ ПРИ СОЗДАНИИ:", *[f"  {p}" for p in problems]]

    (out / "README.txt").write_text("\n".join(readme), encoding="utf-8")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nГотово: {out}", flush=True)
    print(f"Размер: {total / 1024 / 1024:.1f} МБ", flush=True)
    if problems:
        print("\nПРОБЛЕМЫ:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
