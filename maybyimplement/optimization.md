# Optimization — план к возможной реализации (не начат)

Статус: **план**. Приоритет №2 из discovery/perf-optimization.md — реранкер
(cross-encoder) с конфигурируемым переключением грейдера. Цели из baseline'ов:
**MRR 0.625 → ≥0.8, fact_ok 4/12 → ≥6/12**, gold-preservation ≥ 9/12 (не упасть).

## Идея

Роль «судья релевантности» становится стратегией с выбором через конфиг:

| GRADER | Судья | Свойства |
|---|---|---|
| `llm` (default) | 3b + JSON-вердикты (как сейчас) | без новых зависимостей; дисперсия, LOST-ы |
| `cross-encoder` | bge-reranker-v2-m3, скор пары «запрос+чанк» | детерминированно, калиброванно; +1-2 ГБ образа |

LLM из конвейера не уходит: rewrite / broaden / generate остаются за ней.
Перестаёт быть судьёй.

## Шаги реализации

### Шаг 1. Интерфейс и стратегии (домен, без сети)
1. `src/domain/interfaces.py`: протокол `RelevanceGrader.grade(query, chunks) -> list[ScoredChunk]`.
2. `src/infrastructure/llm_grader.py`: `LLMGrader` — вынести текущую логику
   `_grade_chunks` из rag_graph.py (промпт, json_mode, fail-open, setdefault-True).
3. `src/infrastructure/cross_encoder_grader.py`: `CrossEncoderGrader` —
   sentence-transformers `CrossEncoder(model)`, батчи по 8-16 пар,
   `predict` → скоры; отбор: скор ≥ порога (стартово 0.1) ИЛИ топ-N (стартово 5);
   graceful fallback: отсутствие модели/пакета → LLMGrader + warning.
4. `CorrectiveRAGGraph`: конструктор принимает `grader: RelevanceGrader`;
   `_grade` вызывает `grader.grade(...)`; узел больше не знает про LLM-грейдинг.
5. Config (config.py + load_settings): `GRADER=llm|cross-encoder`,
   `RERANK_MODEL=BAAI/bge-reranker-v2-m3`, `RERANK_THRESHOLD=0.1`.

### Шаг 2. Реранк в find_relevant_docs (двигает MRR)
1. В server.py при `GRADER=cross-encoder`: `retriever.search(query, top_k*4)`
   → `grader.grade(query, pool)` → сортировка по скору → топ-k.
2. При `GRADER=llm` — текущее поведение (MRR не меняется).
3. Поле `reranked: true|false` в ответе инструмента — для прозрачности.

### Шаг 3. Зависимости и сборка
1. pyproject: `[project.optional-dependencies] rerank = ["sentence-transformers>=3"]`.
2. Dockerfile: `ARG EXTRAS=""` → `pip install -e ".[dev${EXTRAS:+,$EXTRAS}]"`.
3. docker-compose.yml: `build.args.EXTRAS: ${EXTRAS:-}`; при cross-encoder —
   `EXTRAS=rerank docker compose build`.
4. Volume `hf_cache` + `HF_HOME=/app/data/hf` — модель (~1-2 ГБ) качается с HF
   один раз при первом старте; интернет обязателен.

### Шаг 4. Тесты (unit, без сети и без модели)
1. FakeGrader в conftest (возврат заданных скоров).
2. Граф с FakeGrader: grade не вызывает LLM (len(prompts) без grade-вызова).
3. CrossEncoderGrader с подменённым predict: порог/топ-N, graceful fallback.
4. Переключение конфига: llm → граф с LLMGrader, cross-encoder → CrossEncoderGrader.
5. Существующие 28 тестов не меняются (default llm).

### Шаг 5. Проверка по baseline'ам (после сборки с EXTRAS=rerank)
1. Локально: скачать модель, `GRADER=cross-encoder` поднять стек.
2. `scripts/eval_retrieval.py` (полный) → MRR ≥ 0.8? (recall@5 не упасть ниже 1.0).
3. `scripts/eval_retrieval.py --grade` → gold-preservation ≥ 9/12, refusal 2/2.
4. `scripts/eval_generation.py` → fact_ok ≥ 6/12.
5. Сравнение per-question с evals/*.json; артефакты в evals/*-rerank.json.
6. Если MRR < 0.8: порог/топ-N перебор (RERANK_THRESHOLD 0.05-0.3, N 3-7).

### Шаг 6. Документация и коммиты
1. README: таблица GRADER-режимов в «Конфигурация», сценарий включения
   (`EXTRAS=rerank GRADER=cross-encoder docker compose up -d --build`).
2. ARCHITECTURE.md: узел grade как стратегия (если там описан грейдер).
3. discovery/perf-optimization.md: раздел «Реранкер — реализовано» с цифрами.
4. fixes-log.md: №16 (по ходу — найденные грабли).
5. Коммиты: (а) домен+конфиг+тесты, (б) сборка/доки, (в) eval-артефакты.
6. CI: default llm → джобы зелёные без правок; отдельная ручная проверка
   cross-encoder вне CI (2 ГБ модели в runner'е — не для CI).

## Риски и честные оговорки

- **CPU-латентность:** cross-encoder ~100-300 мс/пара × 20 пар ≈ 2-6 c —
  не быстрее 3b-грейда; выигрыш в качестве/стабильности, не в скорости.
  GPU (приоритет №3) снимает и это.
- **Кросс-язык:** bge-reranker-v2-m3 мультиязычный — русские вопросы
  против английского корпуса должен матчить лучше 3b; проверить на
  «Кто такая Альбертина?» до фиксации baseline'а.
- **Дисперсия базовой LLM:** сравнение до/после по fact_ok шумит (±1 вопрос);
  при пограничном результате — второй прогон.
- **Отказ (refusal):** cross-encoder не знает про «вне корпуса» — ветка
  лексического затвора и честность генератора остаются линиями обороны;
  refusal-вопросы перепроверить обязательно.

## Что НЕ входит в этот план

- Приоритет №3 (GPU + крупная модель для generate) — отдельный план.
- Обучение/дообучение реранкера на корпусе — не нужно для целей метрик.
