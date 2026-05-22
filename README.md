# Hallucination Detection in Tool Calling

NLP course assignment. Цель — построить пайплайн **детекции галлюцинаций на span-уровне** в диалогах с tool calling. На вход модель получает `(query, tool_output, answer)`, на выход — char-level спаны участков ответа, которые являются галлюцинациями.

## Задача

Три типа галлюцинаций для построения датасета:

| Тип | Что это | Пример |
|---|---|---|
| **Type 1 — Hallucination** | финальный ответ противоречит выходу тула | tool: `{weather: "sunny"}` → answer: «The weather is **rainy**» |
| **Type 2 — Overgeneration** | ответ содержит факт, не подтверждённый tool_output | «The weather is sunny, **and has been good for months**» |
| **Type 3 — Missing tool** | ответ предлагает действие, требующее тула не в списке доступных | «The weather is sunny. **Want me to book a ticket?**» (но Booking_API нет) |

Формат разметки — [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed): `{query, context, output, hallucination_labels: [{start, end, text, type}]}`.

Базовый датасет — [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (1 034 чистых триплета user → tool_output → final assistant text).

Бейзлайны для побития:
- [LettuceDetect](https://arxiv.org/abs/2502.17125) (ModernBERT token classifier, обучен на RAGTruth)
- [LookBackLens](https://arxiv.org/abs/2407.07071) (attention-based; пока не прогоняли)

Финальная сдача — Jupyter Notebook в стиле `assignment_template.ipynb` + датасет на HF + сабмит на CodaLab.

## Что сделано

### 1. Построение датасета

Из 11 300 диалогов ToolACE извлекли **1 034 триплета** с полным циклом (user query → tool_call → tool_output → natural-language answer). Сплит по source-id: **834 train / 50 val / 150 test**.

**Type 1 (Hallucination)** — гибрид rule-based + LLM:
- *Rule-based стратегии* (72.6% покрытия source'ов): in-sample pool swap (берём другое значение того же поля из текущего tool_output), cross-sample pool, type-based int/float/url/date/bool с явными паттернами для 6+ форматов дат и hardcoded правилами для bool-полей вроде `is_correct → (Incorrect)`, `is valid → is invalid`.
- *LLM-fallback* через `qwen/qwen3-235b-a22b-2507` — на оставшиеся 28%. С promptом и word-boundary валидацией. 99.8% итогового покрытия (1 032 / 1 034).
- *Augmentation:* до 5 вариантов на train source с round-robin diversity по стратегиям. Train: **4 057**.

**Type 2 (Overgeneration)** — 100% LLM (overgeneration по природе требует естественной генерации текста). LLM добавляет одно declarative предложение с правдоподобной фабрикованной статистикой/контекстом, не упомянутым в tool_output. Train: **2 500** (3 варианта на source).

**Type 3 (Missing Tool)** — сначала rule-based с 15 категориями действий + topic-aware ранжированием (88% topic-aware, 12% random); потом перешли на полностью LLM, потому что 30/30 ручной проверки показали кардинально лучшее качество (LLM привязывается к entities из диалога, ловит тонкие gap'ы в API типа «есть `getSnowboardGearAvailability` но нет reservation»). Train: **2 502** (3 варианта на source). Rule-based код оставлен в репо как референс.

**Финальные размеры:**

| | Hallucinated | Clean | Total |
|---|---|---|---|
| Combined train | 9 059 | 834 | **9 893** (8% clean) |
| Combined val | 150 | 50 | **200** (25% clean) |
| Combined test | 449 | 150 | **599** (25% clean) |

Clean-сэмплы — оригинальные ответы из ToolACE без инжекций. Это критично для измерения precision (без negatives Ex P всегда был = 1.0 и метрика была бессмысленной).

### 2. Baseline: LettuceDetect off-the-shelf

`KRLabsOrg/lettucedect-large-modernbert-en-v1` через Colab T4 на `combined_test`:

| Split | N | Span F1 | Span macro | Ex F1 |
|---|---|---|---|---|
| Combined | 599 | **0.660** | 0.629 | 0.886 |
| Type 1 + clean | 300 | 0.137 | 0.440 | 0.716 |
| Type 2 + clean | 300 | 0.726 | 0.746 | 0.833 |
| Type 3 + clean | 299 | 0.597 | 0.671 | 0.795 |

Type 1 проседает: median gold span 9 chars, а LettuceDetect обучен на RAGTruth, где спаны — целые предложения. Modельная щётка слишком крупная.

### 3. Improvement: LLM-as-judge детектор (Qwen3-235B)

Тот же Qwen3 теперь как **детектор** через OpenRouter. Промпт: описание 3 типов + правило TIGHT SPANS для value-level контрадикций + few-shot из train (1 пример каждого типа + 1 clean).

**Сравнение:**

| Split | LettuceDetect F1 | LLM F1 | Δ |
|---|---|---|---|
| **Combined** | 0.660 | **0.859** | **+0.199** |
| Type 1 | 0.137 | 0.315 | +0.178 |
| Type 2 | 0.726 | 0.844 | +0.118 |
| Type 3 | 0.597 | 0.792 | +0.195 |

Combined Recall 0.992, Ex F1 0.944. Стоимость прогона ~$0.90, ~3 минуты.

Type 1 micro F1 = 0.315 при macro F1 = 0.688 — разрыв означает char-level over-prediction. LLM в Type 1 сэмплах правильно ловит инжектированный swap, но ещё ловит предсуществующие Type 3-style фразы из исходного ToolACE-ответа, которые мы не помечали. Проблема неполноты разметки, не детектора.

## Структура репо

```
src/
  data.py             # extract triples from ToolACE
  pools.py            # value pools for Type 1 rule-based
  dates.py            # date pattern matching (8 formats)
  injection.py        # Type 1 rule-based strategies
  inject_type3.py     # Type 3 rule-based (legacy reference)
  augment.py          # variant selection with strategy diversity
  llm_inject.py       # OpenRouter wrapper + Type 1 LLM fallback
  llm_type2.py        # Type 2 overgeneration via LLM
  llm_type3.py        # Type 3 missing tool via LLM
  llm_detector.py     # LLM-as-judge detector
  finetune.py         # ModernBERT token classifier (data prep + inference)
  evaluation.py       # span/example P/R/F1 metrics
  baselines/
    lettucedetect_runner.py

scripts/
  build_type1.py             # Type 1 rule-based injection
  build_type2.py             # Type 2 LLM injection (full)
  build_type3.py             # Type 3 rule-based (legacy)
  build_type3_llm.py         # Type 3 LLM injection (full)
  build_augmented.py         # Type 1 with 5-variant augmentation
  combine_splits.py          # combine + clean + balanced files
  inject_llm_fallback.py     # Type 1 LLM fallback for uncovered
  test_llm_inject.py / test_llm_type2.py / test_llm_type3.py   # smoke tests
  test_llm_detector.py       # smoke test LLM detector
  eval_llm_detector.py       # full detector eval
  eval_lettucedetect.py      # local CLI eval
  build_kaggle_notebook.py   # assembles .ipynb from cells/ folders

notebooks/
  lettucedetect_baseline/    # cells folder
  lettucedetect_baseline.ipynb
  finetune_modernbert/
  finetune_modernbert.ipynb

data/
  triples.jsonl                       # 1034 clean ToolACE triples
  type{1,2,3}_{train,val,test}.jsonl  # per-type hallucinated
  type{1,2,3}_*_balanced.jsonl        # per-type + clean
  combined_{train,val,test}.jsonl     # all types + clean, prefixed IDs
  results_lettucedetect.json
  results_llm_detector.json
```

## Что осталось

- LookBackLens baseline (второй обязательный baseline по условию задачи).
- Fine-tune ModernBERT на combined_train (для сравнения с LLM detector'ом и для удешевления inference).
- Возможный ensemble: LLM detector + rule-based JSON-value matcher для Type 1.
- Публикация датасета на HuggingFace и финальный submission на CodaLab.
- Финальный отчёт-ноутбук в формате `assignment_template.ipynb`.

## Рабочие документы

- [description.md](description.md) — подробное описание задачи и выжимки из статей.
- [tasks.md](tasks.md) — все задачи проекта (todo / in progress / done).
- [insights.md](insights.md) — наблюдения по экспериментам.
- [rules.md](rules.md) — правила работы (язык, формат финального ноутбука, ресурсы).
