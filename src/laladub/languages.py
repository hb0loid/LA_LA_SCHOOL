"""The language menus, shared by every process that needs to name a language.

Kept out of bot.py so job_runner can use them too: that module deliberately
avoids importing bot.py, which pulls in the whole ML pipeline. Duplicating the
list instead is how settings drift apart - see the artifact ratio, which was
hardcoded in three places and silently ignored the setting.
"""

from __future__ import annotations

SOURCE_LANGS = [
    ("auto", "Авто"),
    # Ordered by how well each one breaks the meaning while leaving the text
    # intact - measured over one video forced through all 45 of them. The
    # score multiplies how far the meaning moved by how much of the text
    # survived, so the languages that collapse a dub into one repeated
    # phrase sink rather than win: destroying a dub is not distorting it.
    #
    # Russian sits last for a reason that is not about Russian: the test
    # video was Russian, so choosing it changed nothing at all. On any other
    # video it would land somewhere in the middle.
    ("sl", "Словенский"),  # 0.87
    ("cs", "Чешский"),  # 0.80
    ("uk", "Украинский"),  # 0.71
    ("bg", "Болгарский"),  # 0.68
    ("th", "Тайский"),  # 0.51
    ("en", "Английский"),  # 0.35
    ("he", "Иврит"),  # 0.29
    ("es", "Испанский"),  # 0.27
    ("sv", "Шведский"),  # 0.27
    ("de", "Немецкий"),  # 0.24
    ("fr", "Французский"),  # 0.22
    ("da", "Датский"),  # 0.21
    ("ms", "Малайзийский"),  # 0.20
    ("sk", "Словацкий"),  # 0.18
    ("ja", "Японский"),  # 0.17
    ("ur", "Урду"),  # 0.16
    ("sw", "Суахили"),  # 0.15
    ("tl", "Тагальский"),  # 0.14
    ("ar", "Арабский"),  # 0.14
    ("id", "Индонезийский"),  # 0.14
    ("hu", "Венгерский"),  # 0.14
    ("nl", "Нидерландский"),  # 0.13
    ("az", "Азербайджанский"),  # 0.11
    ("ro", "Румынский"),  # 0.11
    ("lv", "Латышский"),  # 0.10
    ("eu", "Баскский"),  # 0.10
    ("it", "Итальянский"),  # 0.10
    ("no", "Норвежский"),  # 0.08
    ("fi", "Финский"),  # 0.07
    ("bn", "Бенгальский"),  # 0.07
    ("hi", "Хинди"),  # 0.07
    ("zh", "Китайский"),  # 0.07
    ("ca", "Каталанский"),  # 0.06
    ("lt", "Литовский"),  # 0.06
    ("ko", "Корейский"),  # 0.05
    ("pt", "Португальский"),  # 0.04
    ("fa", "Персидский"),  # 0.04
    ("sq", "Албанский"),  # 0.03
    ("tr", "Турецкий"),  # 0.03
    ("gl", "Галисийский"),  # 0.01
    ("pl", "Польский"),  # 0.01
    ("el", "Греческий"),  # 0.01
    ("vi", "Вьетнамский"),  # 0.00
    ("ru", "Русский"),  # 0.00
    ("et", "Эстонский"),  # 0.00
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
