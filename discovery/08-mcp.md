# 08. MCP: путь данных, таймауты, подготовка репо к сдаче

## Путь данных ask_question (разобрали по коду)

1. Клиент шлёт JSON-RPC `tools/call {name, arguments}` на POST /mcp;
   streamable HTTP, SSE-сессия с ping-ами (видели их в curl-логе).
2. server.py:195-207: `_log_call` → проверка пустого индекса (empty_index) →
   `check_availability()` Ollama → `ctx.graph.run(question)`.
3. Граф (02-rag-graph.md) → RAGResult → JSON {status, answer, sources, stats}
   → content[0].text → SSE `data:`.
4. Таймауты/ретраи LLM: OLLAMA_TIMEOUT, OLLAMA_RETRIES=2.

## Таймауты — найденная и убранная мина

При обсуждении «поможет ли GPU+фикс таймаута» вскрылось: в opencode.json стоял
**timeout: 20000 мс = 20 секунд** — вот почему индексация обрывалась (в логах
сервера «Request already responded to» + 404 на мёртвой сессии). Подняли до
600000. Позже, проверяя репо к сдаче, нашли ту же болезнь в **README-примерах**
(stdio и remote варианты) — преподаватель скопировал бы 20-секундный таймаут и
получил бы обрывы у себя. Исправлены оба примера (коммит 1b49f50).

## Проверка репо по чек-листу сдачи

Прошлись по процессу сдачи (клон → docker compose up → README → подключение
MCP → вопросы → проверка инструментов):

- ✅ свежий клон: compose config OK, docker build OK; модели тянет ollama-init;
- ✅ `data/chroma/` в .gitignore → на свежем клоне index_status честно пуст;
- ✅ demo_docs и data/proust закоммичены; opencode.json, README, ТЗ на месте;
- ✅ метаданные инструментов — подробные docstring (агент догадается вызвать);
- ✅ CI: ruff + pytest, 25 тестов зелёные;
- ❌ блокер: **не было git remote** — пользователь добавил
  `git@github.com:lrvchgost/simple-rag-mcp-rrf.git`; проверили SSH-аутентификацию
  (`Hi lrvchgost!`), запушили всё (b5f05e6..dbfc11b, затем f22b817).

Замечание по чек-листу: там фигурирует `index_folder("./sample_docs")` — у нас
пути `./demo_docs` и `/data/docs` (контейнер); README это явно проговаривает,
преподаватель идёт по README.

## Полезный паттерн из практики сессии

MCP-вызовы без IDE-клиента делали curl'ом в три шага: initialize (забрать
mcp-session-id из заголовка) → notifications/initialized → tools/call. Это
позволяло замерять длительность, ловить SSE-пинги и не зависеть от UI.
