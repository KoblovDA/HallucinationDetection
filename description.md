# Hallucination Detection in Tool Calling — описание задачи

## Общая постановка

Курсовое задание по NLP. Цель — построить пайплайн **детекции галлюцинаций на span-уровне** в диалогах с tool calling. То есть в финальном текстовом ответе модели нужно подсветить **конкретные диапазоны символов**, которые являются галлюцинациями.

Задание делится на три этапа:

1. **Сборка датасета.** Берётся существующий датасет с tool calling (ToolACE), в его ответы автоматически инжектятся галлюцинации трёх типов, спаны фиксируются как ground truth.
2. **Прогон бейзлайнов.** На полученных датасетах оцениваются LettuceDetect и LookBackLens.
3. **Улучшения.** Свободная часть — предложить и сравнить улучшения над бейзлайнами.

## Финальный артефакт

Запускаемый Jupyter Notebook на английском, выглядящий «по-человечески» (не сгенерированный LLM), следующий структуре `assignment_template.ipynb`:
- секция 1 — инфо о работе (имя, ник на Codalab),
- секция 2 — технический отчёт (методология + обсуждение результатов),
- секция 3 — воспроизводимый код (Python 3, скачивание данных, повторение всех экспериментов).

Дополнительно: датасет публикуется на Hugging Face, тренированная модель — тоже на Hugging Face, submission — на CodaLab.

## Оценка (макс 35 баллов)

| Часть | Балл |
|---|---|
| Methodology в отчёте | 5 |
| Discussion of results | 5 |
| Readability кода | 5 |
| Reproducibility | 5 |
| Побит baseline на CodaLab | 5 |
| **Bonus: топ-20% на лидерборде** | +5 |
| **Bonus: топ-1 на лидерборде** | +10 |
| Штраф за опоздание | до −25 (1 день = 1 балл) |

---

## Этап 1: инжекция галлюцинаций

### Базовый датасет: ToolACE

Источник: https://huggingface.co/datasets/Team-ACE/ToolACE
Один файл `data.json`, 35 MB, **11 300 диалогов**.

Структура одной записи:
```
{
  "system": "<...список доступных функций в JSON...>",
  "conversations": [
    {"from": "user",      "value": "..."},
    {"from": "assistant", "value": "[Weather_API(location=\"Beijing\")]"},   // tool call
    {"from": "tool",      "value": "[{\"name\": \"Weather_API\", \"results\": {...}}]"},  // tool output
    {"from": "assistant", "value": "The weather in Beijing is sunny..."}    // финальный ответ
  ]
}
```

Из 11 300 диалогов только **797 имеют хотя бы один tool-turn** (большинство диалогов в ToolACE тренируют модель именно *звать* тулу, без последующего выполнения). Из них извлекается:
- **741** строгих квадруплов `(user → assistant_tool_call → tool_output → assistant_text)`;
- **1034** более слабых пары `(tool_output → assistant_text)` (с любым предшествующим user query).

Длины:
- ответ ассистента: median 468, mean 540, max 4016 символов;
- tool output (JSON): median 341, mean 472, max 5725 символов.

### Три типа галлюцинаций

**Type 1: Hallucination** — финальный ответ **противоречит** выходу тула.
```
Tool answer: {weather: "sunny"}
Answer: "The weather in Beijing is rainy."   ← rainy = галлюцинация
```

**Type 2: Overgeneration** — ответ содержит факты, которых **нет** в выходе тула.
```
Tool answer: {weather: "sunny"}
Answer: "The weather in Beijing is sunny, and the weather has been
        pretty good past few months."   ← вторая часть = галлюцинация
```

**Type 3: Missing tool** — ответ предлагает действие, требующее тула, **отсутствующего** в списке доступных.
```
Tools: [Weather_API]
Tool answer: {weather: "sunny"}
Answer: "The weather in Beijing is sunny, would you like me to book
        a ticket?"   ← предложение книжить = галлюцинация (нет Booking_API)
```

### Формат разметки: RAGTruth

См. https://huggingface.co/datasets/wandb/RAGTruth-processed

Каждая запись:
- `query` — пользовательский запрос (user turn);
- `context` — выход тула;
- `output` — финальный ответ модели (с инжектированной галлюцинацией);
- `hallucination_labels` — список спанов `{start_char, end_char, text, type}`.

На выходе этапа 1 — **три датасета**, по одному на каждый тип галлюцинации.

---

## Этап 2: бейзлайны

### LettuceDetect (Kovács & Recski, 2025, arXiv:2502.17125)

- **Архитектура:** ModernBERT (контекст до 8K токенов) как token classifier.
- **Вход:** конкатенация `[Context] [SEP] [Question] [SEP] [Answer]`; токенам контекста и вопроса ставится `label = -100`, loss считается только на токенах ответа.
- **Метка на токен:** 0 (поддержан контекстом) / 1 (галлюцинация).
- **Спаны:** агрегируются соседние токены с `prob > 0.5`.
- **Тренировка:** AdamW, lr 1e-5, weight decay 0.01, 6 эпох, batch 8, max_len 4096.
- **Готовые модели на HF:**
  - `KRLabsOrg/lettucedetect-base-modernbert-en-v1` (~150M параметров)
  - `KRLabsOrg/lettucedetect-large-modernbert-en-v1` (~396M параметров)
- **Результаты на RAGTruth (overall):**
  - Example-level F1: base = 76.07, large = **79.22** (SOTA на тот момент среди encoder-методов; ближайший конкурент Luna = 65.4).
  - Span-level F1: base = 55.44, large = **58.93**.
- **Скорость:** 30–60 примеров/сек на одной GPU.
- **Лицензия:** MIT, есть пакет `pip install lettucedetect`.

### LookBackLens (Chuang et al., 2024, arXiv:2407.07071)

- **Идея:** контекстуальные галлюцинации связаны с тем, насколько модель «смотрит» на контекст при генерации. Чем меньше attention на контекст и больше на самой себя — тем выше риск галлюцинации.
- **Lookback ratio:** для каждого слоя `l`, head `h`, шага генерации `t`:
  ```
  LR(l,h,t) = A(l,h,t)→context / (A(l,h,t)→context + A(l,h,t)→new_tokens)
  ```
  где `A→context` — средняя attention-вес на токенах контекста, `A→new_tokens` — на уже сгенерированных токенах.
- **Span-feature:** на всём интересующем спане усредняем `LR(l,h,t)` по `t`, получаем вектор размерности `L × H` (для Llama-2-7B это 32 × 32 = 1024).
- **Классификатор:** простая логистическая регрессия на этом векторе.
- **Бэкбон:** в оригинале LLaMA-2-7B-chat, требует gen-time доступ к attention-картам. Можно переносить на 13B через линейное отображение голов.
- **Тренировка:** ~1000–2655 примеров, разметка получена от GPT-4o.
- **Результаты:**
  - На детекции в same-domain: AUROC 97–99% (predefined spans), 86–89% (sliding window).
  - На out-of-domain transfer: AUROC 82–86%, заметно лучше hidden-states-based детекторов.
  - Метод **переносится без переобучения** между задачами и даже между размерами модели.
- **Репо:** https://github.com/voidism/Lookback-Lens

### ToolACE (Liu et al., 2024, arXiv:2409.00920)

Базовый датасет нашей задачи. Ключевое (для нас, не для бейзлайнов):
- Синтетика, сгенерированная мультиагентной системой (user / assistant / tool agents).
- 26 507 API, 390 доменов.
- Включает single, parallel, dependent и non-tool-use диалоги.
- Авторы выложили также fine-tuned модель `Team-ACE/ToolACE-8B` (на Llama-3.1-8B-Instruct).
- BFCL: ToolACE-8B даёт overall 59.22, что сравнимо с GPT-4-turbo.
- Качество данных обеспечивается dual-layer верификацией (rule checker + model checker).

---

## Этап 3: улучшения

Свободный раздел. Что планируем (см. `tasks.md` по мере появления):
- fine-tune LettuceDetect на наших данных,
- rule-based детектор для Type 1 (структурное сравнение JSON ↔ ответ),
- NLI-проверки на atomic claims,
- ensemble подходов,
- LLM-as-judge как teacher для distillation (если будет смысл).

Результат — таблица «бейзлайн vs все наши методы» по трём датасетам.

---

## Метрики

Для каждой модели и каждого датасета считаем:
- **Example-level** F1 / Precision / Recall (по бинарной метке «есть ли в ответе хоть одна галлюцинация»).
- **Span-level** F1 / Precision / Recall по char-overlap (как в RAGTruth).

Главная метрика для CodaLab — обычно span-level F1; уточнить после регистрации.

---

## Ресурсы и ссылки

- Базовый датасет: https://huggingface.co/datasets/Team-ACE/ToolACE
- Формат разметки: https://huggingface.co/datasets/wandb/RAGTruth-processed
- LettuceDetect модели: https://huggingface.co/KRLabsOrg
- LettuceDetect пакет: `pip install lettucedetect`
- LookBackLens код: https://github.com/voidism/Lookback-Lens
- ToolACE-8B (для генерации/перплексити): https://huggingface.co/Team-ACE/ToolACE-8B
