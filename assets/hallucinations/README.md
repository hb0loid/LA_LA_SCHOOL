# Каталог галлюцинаций Whisper

`whisper_hallucinations.csv` — датасет [sachaarbonel/whisper-hallucinations][src]
(лицензия MIT), 7889 фраз на 100 языках: то, что Whisper выдаёт, когда речи в
звуке нет вообще. Колонки: `lang`, `phrase`, `count`.

Скачан 2026-09-02. Пока просто лежит здесь и ни на что не влияет.

## Зачем он тут может пригодиться

Замеры на момент скачивания, по 45 языкам, которые бот принимает на вход
(2485 фраз из датасета):

- **132** фразы — рекламные («подпишись», «спасибо за просмотр», «субтитры
  сделал…»), то есть ровно та категория, которую ловит
  `_looks_like_meta_hallucination`. Из них нынешние правила распознают **95
  (72%)**, пропускают **37** — включая русские «подпишитесь на канал» и «на
  мой канал», датские и итальянские варианты, и семейство `subtitling by
  klages` на албанском.
- **552** фразы — короткие огрызки вроде `bye`, `aah`, `you`. Их ловят другие
  механизмы (короткие сегменты, повторы), отдельные правила им не нужны.

Отсюда два возможных применения:

1. **Добить фильтр** — те 37 непойманных рекламных фраз.
2. **Источник артефактов** — когда второй проход Whisper ничего не нашёл,
   подмешивать готовые фразы на нужном языке. Дало бы артефакты всегда и без
   лишнего прохода, но это уже не «что бот услышал в этом видео», а вставка со
   стороны — смысл механики меняется.

Ни то, ни другое пока не сделано.

## Как пересчитать эти цифры

```
python - <<'PY'
import csv, io, pathlib, re, sys
sys.path.insert(0, "src")
from laladub.pipeline import _looks_like_meta_hallucination
from laladub.bot import SOURCE_LANGS
rows = list(csv.DictReader(io.StringIO(
    pathlib.Path("assets/hallucinations/whisper_hallucinations.csv").read_text(encoding="utf-8"))))
ours = {c for c, _ in SOURCE_LANGS if c != "auto"}
meta_words = re.compile(r"subscrib|подпис|канал|channel|субтитр|subtitl|thanks for watch"
                        r"|спасибо за просмотр|like|лайк|колокольчик|abonn|suscrib|iscriv|abone", re.I)
meta = [r for r in rows if r["lang"] in ours and meta_words.search(r["phrase"])]
caught = sum(1 for r in meta if _looks_like_meta_hallucination(r["phrase"]))
print(f"рекламных: {len(meta)}, ловится: {caught}, пропускается: {len(meta) - caught}")
PY
```

[src]: https://huggingface.co/datasets/sachaarbonel/whisper-hallucinations
