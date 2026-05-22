# Tasks

Подробное описание всех задач проекта — текущих, выполненных и запланированных.

Формат записи задачи:

```
## [статус] Название задачи
- **Зачем:** мотивация
- **Что делаем:** конкретные шаги
- **Артефакты:** какие файлы/датасеты/модели появятся
- **Зависит от:** другие задачи
- **Заметки:** что выяснилось по ходу
```

Статусы: `todo`, `in progress`, `done`, `blocked`, `cancelled`.

---

## [done] Type 1: rule-based span-инжектор (без LLM)

**Результат:** **751 / 1034 (72.6%)** сэмплов покрыты, 0 битых спанов, 0 self-swap, 100% reversion. Файл `data/type1.jsonl` готов. Брейкдаун: in_sample_pool 242 / int 210 / float 119 / date 83 / url 57 / bool 27 / cross_sample_pool 13. Cross-sample покрыл каноничные случаи из условия задачи (sentiment positive→negative, country US→GB, sport football→baseball).

- **Зачем:** датасет с инжектированными «hallucination» (Type 1) — противоречия между ответом и tool output. Точные char-спаны — основа всего пайплайна, RAGTruth-формат.

- **Стратегии замены (в порядке приоритета):**
  1. **In-sample pool swap** — для строк в «чистых» полях: берём другое значение того же поля из текущего tool output. Главная стратегия, покрывает 76% values.
  2. **Type-based int/float swap** — берём число из ответа (если оно есть в tool_output), генерим контрастное (×3..×5 / random shift), подменяем substring.
  3. **Type-based URL swap** — модифицируем path/host.
  4. **Type-based date swap** — пробуем 6-7 паттернов: ISO verbatim, `Month D, YYYY`, `Month Dth, YYYY`, `Month D` без года, `Month YYYY`, ISO prefix, `Dth Month YYYY`. Найдя присутствующий паттерн, генерим новую дату (random shift 7..365 дней) **в том же формате**.
  5. **Cross-sample pool swap** — fallback для скаляров повторяющихся тулов.

- **НЕ делаем rule-based:**
  - bool inversion (94.8% перефразированы — substring swap не работает);
  - длинные строковые цитаты/описания;
  - значения в «грязных» полях (`name, title, id, description, text, symbol, code, condition, type, status, category`) без type-based fallback.

- **План реализации:**
  1. `src/data.py` — извлечение 1 034 триплетов из ToolACE.
  2. `src/pools.py` — построение in-sample и cross-sample value pools, плюс blacklist полей.
  3. `src/injection.py` — стратегии замены (int/float/url/date/string-pool) с возвратом `(новый_текст, span, метаданные)`.
  4. `src/inject_type1.py` — оркестратор: для каждого триплета перебирает кандидатов и применяет первую успешную стратегию.
  5. `scripts/build_type1.py` — конечный пайплайн запуска.
  6. Выход: `data/type1.jsonl` в RAGTruth-формате.
  7. Валидация: визуальная проверка 20–30 случайных примеров.

- **Артефакты:** `data/triples.jsonl` (~1034), `data/type1.jsonl` (~780 ожидаемых).
- **Зависит от:** ничего.
- **Заметки:**
  - Random seed фиксируем для воспроизводимости.
  - In-sample list-swap особенно ценен: подменённое значение есть в tool output, но относится к другой сущности — реалистично, как настоящая LLM-галлюцинация.

## [done] Type 1: LLM-расширение для оставшихся 25%

**Результат:** 281 / 283 непокрытых сэмплов добраны через `qwen/qwen3-235b-a22b-2507` (OpenRouter). Финальный датасет `data/type1_full.jsonl`: **1032 / 1034 = 99.8% покрытия**. Качество ~85% полностью чистые, ~15% с лёгкими грамматическими артефактами (паттерн «has been failed to X»). Цена прогона ~$0.30. Файлы: `src/llm_inject.py`, `scripts/inject_llm_fallback.py`.

## [done] Type 2: overgeneration injection (LLM-based)

**Результат:** train **2500** / val **50** / test **150**, 0 битых спанов. Полностью LLM-генерация через `qwen/qwen3-235b-a22b-2507`. 30/30 на ручной проверке — правдоподобные фабрикации (industry benchmarks, исторический контекст, market trends, статистика по сэмплам). Файлы: `src/llm_type2.py`, `scripts/build_type2.py`, `scripts/test_llm_type2.py`. Цена ~$3.

## [done] Type 3 (rule-based, topic-aware) — оставлен как референс

**Результат:** 1034 / 1034 (100%) сэмплов в `data/type3.jsonl`. Таксономия из 15 action-категорий + topic-aware ранжирование (88% topic_aware, 12% random fallback). При ручной проверке 10 примеров ~50% натянутые из-за keyword false-positives. Файлы: `src/inject_type3.py`, `scripts/build_type3.py`. **Оставлен в репо как baseline для сравнения с LLM-вариантом** — для отчёта показать, что мы сравнили подходы.

## [done] Type 3: missing tool injection (LLM)

**Результат:** train **2502** / val **50** / test **149**, 0 битых спанов. Полностью LLM-генерация через `qwen/qwen3-235b-a22b-2507`. 30/30 на ручной проверке топически идеальны, привязаны к entities из диалога, ловят тонкие gap'ы в API (например, snowboard tools имеют availability но не reservation → предлагает «reserve the gear»). Файлы: `src/llm_type3.py`, `scripts/build_type3_llm.py`. Цена прогона ~$3.

## [done] Type 1: augmentation на train (по 5 вариантов на source)

**Результат:** train 4057 / val 50 / test 150. Сплит по source-id, аугментация только на train. 0 битых спанов, 0 self-swap. Стратегии в train: llm 48% / in_sample_pool 14% / type_based_int 12% / cross_sample_pool 9% / float 7% / url 4% / date 5% / bool 1%. Файлы: `src/augment.py`, `scripts/build_augmented.py`.

## [todo] Сборка финального датасета и публикация на HF

- **Зачем:** требование задания — выложить датасет на Hugging Face.
- **Что делаем:** train/val/test split, метаданные, dataset card.

## [done] Baseline: LettuceDetect off-the-shelf

**Результат span micro F1:** Type 1 = 0.216, Type 2 = 0.841, Type 3 = 0.722. Прогнано на 150/150/149 test-сэмплах через Colab. LettuceDetect силён на Type 2 (его тренинговый домен RAGTruth), но проваливает Type 1 — точечные swap'ы значений (median 9 chars span) для него непривычная задача. Файлы: `src/evaluation.py`, `src/baselines/lettucedetect_runner.py`, `scripts/eval_lettucedetect.py`, `notebooks/lettucedetect_baseline.ipynb`, `lettucedetect_baseline.ipynb` (исполненная копия из Colab).

## [todo] Baseline: LookBackLens

- **Зачем:** второй бейзлайн из условия задачи.
- **Заметки:** нужен Llama-2-7B-chat для извлечения attention; считаем на сервере, в финальный ноутбук кладём предвычисленные фичи.

## [done] Improvement 1: LLM-as-judge детектор (Qwen3-235B через OpenRouter)

**Результат:** Combined span F1 = **0.859** vs baseline 0.660 (**+20 п.п.**). Per-type: Type 1 = 0.315 (low precision из-за неполноты разметки), Type 2 = 0.844, Type 3 = 0.792. Файлы: `src/llm_detector.py`, `scripts/test_llm_detector.py`, `scripts/eval_llm_detector.py`. Цена ~$0.90, ~3 мин. Промпт: few-shot из train (1 на тип + 1 clean) + явное правило TIGHT SPANS для value-level контрадикций.

## [todo] Improvement 2: fine-tune ModernBERT/LettuceDetect на combined_train

- **Зачем:** альтернативный путь — обучить собственный детектор. Сейчас LLM детектор лидирует, но fine-tune может дать дешевле inference (без API), плюс полезен для отчёта как сравнение методов.
- **Что готово:**
  - `src/finetune.py` — encode_sample, dataset, predict_spans.
  - `notebooks/finetune_modernbert.ipynb` — Colab-ноутбук под train (T4-friendly: bf16, gradient checkpointing, batch=2, accum=2, max_len=2048).
- **Заметка:** в Colab T4 не хватило GPU памяти при первой попытке — параметры понижены, ещё не прогнали. Если не приоритетно — отложим.

## [done] Combine splits with clean (с фиксом ID-коллизий)

**Результат:** `scripts/combine_splits.py`. Создаёт per-type balanced и combined splits с unique префиксированными ID (`t1_…`, `t2_…`, `t3_…`, `…_clean`). После фикса 599/599 уникальных в combined_test. Также добавляет `tools_available` ко всем записям (через lookup из triples.jsonl).

## [done] Combine splits with clean

**Результат:** `scripts/combine_splits.py` строит per-type balanced (T1: 4891/100/300, T2: 3334/100/300, T3: 3336/100/299) и combined (9893/200/599, 8.4% / 25% / 25% clean). На combined_test заодно перепрогнали baseline — реалистичная картина: span F1 = 0.660, Ex P = 0.871 (ложно срабатывает на ~13% clean в combined).

## [todo] Improvement 2: rule-based детектор для Type 1 как ensemble компонент

- **Зачем:** structural matching JSON ↔ ответ может давать высокий precision и компенсировать слабые места ML-детектора.

## [todo] Финальный отчёт и сборка ноутбука

- **Зачем:** артефакт сдачи.
- **Заметки:** собирается **в самом конце**, скриптом из `.py` + `.md` исходников.
