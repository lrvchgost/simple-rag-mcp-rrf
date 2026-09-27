# 02. Граф Corrective RAG — «на пальцах»

## Как разговор зашёл

После серии тестовых вопросов пользователю захотелось понять механику:
«расскажи процесс: как проходят данные от ask_question до возврата ответа в
MCP», затем «объясни на пальцах, как определён граф в коде». Разбирали по
фактическому коду (src/domain/rag_graph.py, src/server.py,
src/domain/retriever.py), не по учебнику.

## Путь запроса: от MCP до ответа

1. **MCP-транспорт.** Клиент шлёт JSON-RPC `tools/call {name: "ask_question"}`
   на POST /mcp (streamable HTTP). FastMCP держит SSE-сессию с ping-ами — мы
   видели их в живом curl-логе (`: ping - 2026-09-27 14:49:21...`).
2. **server.py:195-207.** `_log_call` (строка в логах: `tool=ask_question
   args={...}`) → проверка пустого индекса (`empty_index`) →
   `check_availability()` Ollama → `ctx.graph.run(question)`.
3. **Граф** (см. ниже) → `RAGResult(answer, sources, stats)`.
4. **Назад:** JSON `{status, answer, sources, stats}` → `content[0].text` →
   SSE-событие. Таймауты/ретраи: OLLAMA_TIMEOUT, OLLAMA_RETRIES=2.

## Граф: три сущности

**State — общая тетрадь.** `RAGState` (TypedDict): `question, rewritten,
retrieved, relevant, loop_count, answer`. Узлы не общаются напрямую — каждый
берёт тетрадь и возвращает dict со своими изменениями, LangGraph мержит.
Пример из кода: `_broaden` возвращает
`{"rewritten": ..., "loop_count": state["loop_count"] + 1}`.

**Узлы — обычные методы класса.** Никакой магии: `_rewrite`, `_retrieve`,
`_grade`, `_broaden`, `_generate`.

**Рёбра — `_build` (rag_graph.py:160-176):**

```
START ──> rewrite ──> retrieve ──> grade ──┬─(релевантных >= 2 или циклы кончились)──> generate ──> END
                       ^                   │
                       └───── broaden <────┘
```

- жёсткие `add_edge`: START→rewrite→retrieve→grade, broaden→retrieve (петля!),
  generate→END;
- единственная развилка — `add_conditional_edges("grade", _route, ...)`;
  `_route` (:155) смотрит в тетрадь: `len(relevant) >= min_relevant_chunks`
  или `loop_count >= max_retrieval_loops` → generate, иначе broaden;
- петля broaden→retrieve — тот самый «Corrective» в названии пайплайна;
- компиляция один раз, наружу только `run(question)`.

## Уточнение числа LLM-вызовов

Пользователь сформулировал гипотезу: «LLM — это только сравнение чанков и
генерация». Проверили по коду — **вызовов четыре**, не два:

1. **rewrite** — вопрос → поисковый запрос (без него гибридный поиск по
   «болтливому» вопросу хуже);
2. **grade** — сравнение чанков с вопросом (гипотеза верна здесь);
3. **broaden** — расширение запроса в петле коррекции (опционально);
4. **generate** — генерация ответа (гипотеза верна здесь).

Не-LLM: BM25 (формула), ChromaDB, RRF-слияние, `_route` (if/else). Нюанс,
который обсудили отдельно: эмбеддинги — тоже модель, но не «разговорная»:
nomic только превращает текст в вектор (см. 05-embeddings.md).

## Латентность по узлам (из логов сессии)

rewrite ~1-2 c + retrieve <1 c + grade ~1-2 c (все 5 чанков одним вызовом) +
generate **34-65 c** — генерация доминирует; broaden + ещё цикл, если грейдер
отсеял слишком много. Отсюда весь интерес к GPU (01-ollama-gpu.md).
