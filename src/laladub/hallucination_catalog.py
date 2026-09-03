"""Ready-made Whisper hallucinations, instead of hunting for them each time.

The bot used to find artifacts by running a second Whisper pass over the whole
audio, forced onto the decoy language, and keeping whatever it invented. That
works, but it is the most expensive optional stage there is: a median of 10
seconds, 141 at the 90th percentile, and up to 20 minutes on a long video -
23 hours in total across 1767 jobs.

The catalogue (assets/hallucinations/) is 7889 phrases across 100 languages
collected from exactly that behaviour, so the same material is available for
free. Picking from it also widens the pool: a job is no longer limited to what
this one video happened to provoke.

Phrases carry the language they belong to, and are weighted by how often
Whisper produced them - common ones stay common. A small share is drawn from a
different language on purpose, because that is what the real hunt did too when
the decoy language was wrong.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "assets" / "hallucinations" / "whisper_hallucinations.csv"

# Below this a phrase is a stray fragment ("aah", "the") rather than something
# worth putting in a dub.
MIN_PHRASE_CHARS = 8


@dataclass(frozen=True, slots=True)
class Hallucination:
    lang: str
    phrase: str
    weight: int


class HallucinationCatalog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_CATALOG
        self._by_lang: dict[str, list[Hallucination]] = {}
        self._all: list[Hallucination] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for row in csv.DictReader(text.splitlines()):
            phrase = (row.get("phrase") or "").strip()
            lang = (row.get("lang") or "").strip().casefold()
            if not phrase or not lang or len(phrase) < MIN_PHRASE_CHARS:
                continue
            try:
                weight = max(1, int(row.get("count") or 1))
            except (TypeError, ValueError):
                weight = 1
            item = Hallucination(lang=lang, phrase=phrase, weight=weight)
            self._by_lang.setdefault(lang, []).append(item)
            self._all.append(item)

    @property
    def languages(self) -> list[str]:
        self._load()
        return sorted(self._by_lang)

    def size(self, lang: str | None = None) -> int:
        self._load()
        if lang is None:
            return len(self._all)
        return len(self._by_lang.get((lang or "").strip().casefold(), []))

    def phrases(
        self,
        lang: str | None,
        count: int,
        *,
        seed: str = "",
        cross_language_share: float = 0.15,
    ) -> list[str]:
        """Up to `count` distinct phrases for this language.

        Seeded, so the same job asked twice gets the same phrases rather than a
        fresh set on every retry.
        """
        self._load()
        if count <= 0 or not self._all:
            return []
        code = (lang or "").strip().casefold()
        own = self._by_lang.get(code, [])
        # With no phrases for this language there is nothing to stay faithful
        # to, so the whole draw comes from elsewhere.
        share = 1.0 if not own else min(1.0, max(0.0, cross_language_share))
        if own:
            # A thin section cannot carry a whole dub on its own. Vietnamese has
            # 23 usable phrases and nearly all of them are some wording of
            # "subscribe to the channel", so drawing six from it gives six
            # variations of one line. The fewer a language has relative to what
            # is being asked for, the more is borrowed from the rest.
            variety = len(own) / max(1, count * 3)
            if variety < 1.0:
                share = max(share, 1.0 - variety)

        rng = random.Random(hashlib.sha256(f"{seed}|{code}".encode("utf-8")).hexdigest())
        chosen: list[str] = []
        chosen_words: list[set[str]] = []
        seen: set[str] = set()
        # Bounded rather than "until we have enough": a language with three
        # usable phrases must not spin here forever.
        for attempt in range(count * 20):
            if len(chosen) >= count:
                break
            pool = self._all if (not own or rng.random() < share) else own
            if not pool:
                break
            item = rng.choices(pool, weights=[h.weight for h in pool], k=1)[0]
            key = item.phrase.casefold()
            if key in seen:
                continue
            # Whole sections of the catalogue are near-duplicates - 65% of the
            # Vietnamese entries are some phrasing of "subscribe to the
            # channel" - and picking six of those gives six identical lines in
            # one dub. Near-duplicates are refused while there is still room to
            # find something else; the guard lifts near the end so a language
            # with little variety still fills its quota.
            if attempt < count * 12 and _too_similar(item.phrase, chosen_words):
                continue
            seen.add(key)
            chosen.append(item.phrase)
            chosen_words.append(_word_set(item.phrase))
        return chosen


# Above this share of shared words two phrases say the same thing.
_SIMILARITY_LIMIT = 0.6


def _word_set(phrase: str) -> set[str]:
    return {w for w in phrase.casefold().split() if len(w) > 2}


def _too_similar(phrase: str, existing: list[set[str]]) -> bool:
    words = _word_set(phrase)
    if not words:
        return False
    for other in existing:
        if not other:
            continue
        overlap = len(words & other) / min(len(words), len(other))
        if overlap > _SIMILARITY_LIMIT:
            return True
    return False


_CATALOG: HallucinationCatalog | None = None


def shared_catalog(path: Path | None = None) -> HallucinationCatalog:
    """One parsed copy per process - the file is small, but a job asks for
    phrases once per run and re-parsing 7889 rows each time is pointless."""
    global _CATALOG
    if _CATALOG is None or (path is not None and Path(path) != _CATALOG.path):
        _CATALOG = HallucinationCatalog(path)
    return _CATALOG
