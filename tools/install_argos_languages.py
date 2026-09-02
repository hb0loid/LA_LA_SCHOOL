"""Installs the Argos translation packages for a list of languages.

Every language needs a pair - X->en and en->X - because Argos routes
everything through English. Already-installed pairs are skipped, so this is
safe to re-run after an interrupted download.

    python tools/install_argos_languages.py bg ca cs
    python tools/install_argos_languages.py --all-whisper

Set ARGOS_PACKAGES_DIR first to install somewhere other than the home
directory (the bot keeps its models on drive F).
"""

from __future__ import annotations

import argparse
import os
import sys

# Argos reads this at import, and the default splitter needs a per-language
# Stanza model that does not exist for every language we install.
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")


def main() -> None:
    parser = argparse.ArgumentParser(description="Installs Argos language packages.")
    parser.add_argument("languages", nargs="*", help="Language codes, e.g. bg ca cs")
    parser.add_argument(
        "--all-whisper",
        action="store_true",
        help="Every language Argos offers a round trip for that Whisper can also transcribe",
    )
    args = parser.parse_args()

    import argostranslate.package as pkg

    print("Обновляю индекс пакетов...", flush=True)
    pkg.update_package_index()
    available = pkg.get_available_packages()
    installed = {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}

    wanted = set(args.languages)
    if args.all_whisper:
        try:
            from whisper.tokenizer import LANGUAGES as whisper_languages
        except Exception:
            print("Не удалось прочитать список языков Whisper.", flush=True)
            raise SystemExit(1)
        to_en = {p.from_code for p in available if p.to_code == "en"}
        from_en = {p.to_code for p in available if p.from_code == "en"}
        # nb/pb/zt are Argos spellings of languages Whisper calls no/pt/zh.
        alias = {"nb": "no", "pb": "pt", "zt": "zh"}
        wanted |= {
            code
            for code in (to_en & from_en)
            if alias.get(code, code) in whisper_languages
        }

    if not wanted:
        parser.error("Укажи языки или --all-whisper")

    todo = [
        p
        for p in available
        if ((p.from_code in wanted and p.to_code == "en") or (p.from_code == "en" and p.to_code in wanted))
        and (p.from_code, p.to_code) not in installed
    ]
    if not todo:
        print("Всё уже установлено.", flush=True)
        return

    print(f"К установке: {len(todo)} пакет(ов) для {len(wanted)} язык(ов)", flush=True)
    failed: list[str] = []
    for index, package in enumerate(todo, start=1):
        label = f"{package.from_code}->{package.to_code}"
        print(f"[{index}/{len(todo)}] {label}", flush=True)
        try:
            pkg.install_from_path(package.download())
        except Exception as exc:
            print(f"    ошибка: {type(exc).__name__}: {exc}", flush=True)
            failed.append(f"{label}: {type(exc).__name__}")

    print("\nГотово.", flush=True)
    if failed:
        print("Не установились:", flush=True)
        for line in failed:
            print(f"  {line}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
