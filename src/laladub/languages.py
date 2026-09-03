"""The language menus, shared by every process that needs to name a language.

Kept out of bot.py so job_runner can use them too: that module deliberately
avoids importing bot.py, which pulls in the whole ML pipeline. Duplicating the
list instead is how settings drift apart - see the artifact ratio, which was
hardcoded in three places and silently ignored the setting.
"""

from __future__ import annotations

SOURCE_LANGS = [
    ("auto", "Авто"),
    # The two familiar La La School choices stay pinned at the top.  The rest
    # are ordered by the number of distinct phrases in the bundled Whisper
    # hallucination catalogue, so the more varied experimental inputs are
    # easier to reach.
    ("vi", "Вьетнамский"),
    ("ru", "Русский"),
    ("en", "Английский"),
    ("ur", "Урду"),
    ("hi", "Хинди"),
    ("bn", "Бенгальский"),
    ("sl", "Словенский"),
    ("da", "Датский"),
    ("eu", "Баскский"),
    ("bg", "Болгарский"),
    ("fa", "Персидский"),
    ("sq", "Албанский"),
    ("et", "Эстонский"),
    ("pt", "Португальский"),
    ("sv", "Шведский"),
    ("th", "Тайский"),
    ("gl", "Галисийский"),
    ("el", "Греческий"),
    ("az", "Азербайджанский"),
    ("hu", "Венгерский"),
    ("tl", "Тагальский"),
    ("sw", "Суахили"),
    ("ca", "Каталанский"),
    ("zh", "Китайский"),
    ("sk", "Словацкий"),
    ("es", "Испанский"),
    ("lt", "Литовский"),
    ("ro", "Румынский"),
    ("pl", "Польский"),
    ("lv", "Латышский"),
    ("ko", "Корейский"),
    ("cs", "Чешский"),
    ("it", "Итальянский"),
    ("fr", "Французский"),
    ("tr", "Турецкий"),
    ("de", "Немецкий"),
    ("ja", "Японский"),
    ("fi", "Финский"),
    ("no", "Норвежский"),
    ("id", "Индонезийский"),
    ("nl", "Нидерландский"),
    ("ar", "Арабский"),
    ("uk", "Украинский"),
    ("ms", "Малайзийский"),
    ("he", "Иврит"),
]

TARGET_LANGS = [
    ("ru", "Русский"),
    ("uk", "Украинский"),
    ("en", "Английский"),
]


def source_lang_label(value: object) -> str:
    code = str(value or "auto").strip()
    if code == "auto":
        return "Любой язык"
    return next((label for item, label in SOURCE_LANGS if item == code), code)


def target_lang_label(value: object) -> str:
    code = str(value or "").strip()
    return next((label for item, label in TARGET_LANGS if item == code), code)


def transcript_header(job: object) -> str:
    """Which languages a job used, for the top of its transcript.

    The bot shows this while a job runs and then it is gone, so afterwards
    there was no way to tell what a given video had been dubbed from.
    """
    if not isinstance(job, dict) or not job:
        return ""
    source = source_lang_label(job.get("source_lang") or "auto")
    target = target_lang_label(job.get("target_lang"))
    if not target:
        return f"[{source}]"
    return f"[{source} → {target}]"
