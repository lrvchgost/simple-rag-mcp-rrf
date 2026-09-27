# RAG Knowledge Base — MCP-сервер с локальной LLM

MCP-сервер превращает локальную папку с документами в поисковую базу знаний.
Подключи сервер к IDE (Claude Desktop, VS Code Copilot и др.), проиндексируй
документы и задавай вопросы — AI получает ответы, основанные на содержимом файлов.
Внутри — Corrective RAG-пайплайн на LangGraph и локальная LLM через Ollama
(без платных API).

Подробная архитектура: [ARCHITECTURE.md](ARCHITECTURE.md). Техническое задание:
[technical-task.md](technical-task.md).

```
project/
├── src/                     # код сервера (server.py, config.py, domain/, infrastructure/)
├── tests/                   # pytest без сети и без LLM (fake-реализации интерфейсов)
├── demo_docs/               # демо-корпус: 7 технических файлов (6 форматов)
│   └── proust/              # + художественный корпус ~550 КБ (главы + обзоры)
├── scripts/download_corpus.py  # загрузчик глав demo_docs/proust/texts (Project Gutenberg)
├── docker-compose.yml       # ollama + автозагрузка моделей + сервер (одна команда)
├── Dockerfile
├── opencode.json            # готовый конфиг MCP для opencode (remote-вариант)
└── .github/workflows/ci.yml # ruff + pytest (+ сборка образа)
```

## Инструменты MCP

| Инструмент | Что делает | LLM |
|---|---|---|
| `index_folder` | индексирует папку: чанкинг, эмбеддинги в ChromaDB, перестройка BM25 | только эмбеддинги |
| `ask_question` | полный Corrective RAG: гибридный поиск → грейдинг чанков → генерация ответа + источники | да (Ollama) |
| `find_relevant_docs` | гибридный поиск (BM25 + вектор → RRF) без генерации ответа | нет |
| `index_status` | статистика: чанки, файлы, время индексации, настройки | нет |

## Быстрый старт

### Вариант 1 — всё одной командой (Docker Compose)

```bash
docker compose up -d
```

Эта команда поднимает три сервиса (по умолчанию всё на CPU — работает
на любом железе; для NVIDIA-GPU см. раздел «Вычисления на CPU или GPU» ниже):

| Сервис | Что делает |
|---|---|
| `ollama` | локальный LLM-сервер (данные моделей — в volume `ollama_data`) |
| `ollama-init` | скачает `qwen2.5:3b-instruct` и `nomic-embed-text` при первом запуске (несколько минут, дальше мгновенно — модели кэшируются) |
| `rag-mcp` | сам MCP-сервер на `http://localhost:8000/mcp` (streamable HTTP), демо-документы смонтированы в `/data/docs`, векторная база — в volume `chroma_data` |

Проверка и демо после старта:

```bash
docker compose ps                        # все сервисы должны быть Up/Exit(0) у init
docker compose logs -f rag-mcp           # логи сервера (вызовы инструментов, LLM-вызовы)
curl -s http://localhost:8000/mcp | head -c 200   # сервер отвечает
```

Дальше подключи клиент (ниже) и вызови: `index_folder` с `folder_path = /data/docs`,
затем `find_relevant_docs` / `ask_question`.

### Вариант 2 — локально (stdio-транспорт для десктоп-клиентов)

```bash
# 1. Ollama и модели
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text   # только для EMBEDDING_PROVIDER=ollama

# 2. Сервер
pip install -e .
python -m src.server           # stdio; HTTP: MCP_TRANSPORT=http python -m src.server
```

### Подключение к клиенту

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/путь/к/project"
    }
  }
}
```

VS Code / другие клиенты, поддерживающие streamable HTTP (Docker-вариант):

```json
{
  "mcpServers": {
    "rag-knowledge-base": { "url": "http://localhost:8000/mcp" }
  }
}
```

### Настройка в opencode

Конфиг кладётся в `opencode.json` в корне проекта (в репозитории уже лежит рабочий
вариант с локальным stdio-сервером) или в `~/.config/opencode/opencode.json`.

Вариант «local» — opencode сам запускает сервер как подпроцесс (Ollama нужна
только для `ask_question`):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "rag-kb": {
      "type": "local",
      "command": ["/путь/к/project/.venv/bin/python", "-m", "src.server"],
      "cwd": "/путь/к/project",
      "environment": {
        "CHROMA_DIR": "./data/chroma",
        "LOG_LEVEL": "INFO"          // DEBUG — подробные логи сервера
      },
      "timeout": 600000              // 10 мин: индексация и первый старт занимают минуты
    }
  }
}
```

Вариант «remote» — сервер уже запущен через `docker compose up -d`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "rag-kb": {
      "type": "remote",
      "url": "http://localhost:8000/mcp",
      "enabled": true,
      "timeout": 600000
    }
  }
}
```

После перезапуска opencode инструменты доступны с префиксом имени сервера:
`rag-kb_index_folder`, `rag-kb_ask_question`, `rag-kb_find_relevant_docs`,
`rag-kb_index_status` — вызывать можно по имени («проиндексируй demo_docs/proust
через rag-kb») или явно («use the rag-kb_find_relevant_docs tool…»).

## Как пользоваться (демо-сценарий)

В репозитории два тестовых корпуса:

| Корпус | Содержимое | Объём |
|---|---|---|
| `demo_docs/` | технические документы (архитектура AuthService, конфиги, код) — 7 файлов, все 6 форматов из ТЗ | ~7 КБ |
| `demo_docs/proust/` | корпус по «В поисках утраченного времени» Пруста: рукописные обзоры + 13 глав-выдержек томов 1–3 в переводе C. K. Scott Moncrieff (весь текст на английском, public domain) | ~550 КБ, ~820 чанков |

Главы `demo_docs/proust/texts/` скачаны с Project Gutenberg и нарезаны скриптом (воспроизводимо):

```bash
.venv/bin/python scripts/download_corpus.py
```

Сценарий проверки:

1. `index_folder` с `folder_path = ./demo_docs` (или `/data/docs` в контейнере) — технический корпус.
2. `index_folder` с `folder_path = ./demo_docs/proust` (или `/data/docs/proust` в контейнере) — большой англоязычный художественный корпус (проверка латентности и объёма).
3. `index_status` — убедиться, что чанки созданы.
4. `find_relevant_docs` с `query = "TOKEN_EXPIRY_HOURS"` — точечный BM25-поиск по идентификатору.
5. `find_relevant_docs` с `query = "VOLUME_COUNT"` или `query = "Мезеглиз"` — поиск по коду/тексту.
6. `ask_question` с `"Как устроена ротация ключей в AuthService?"` — семантический ответ + источники.
7. `ask_question` с `"Кто такая Альбертина и что с ней случилось?"` — ответ по художественному корпусу.

## Конфигурация (env-переменные)

Полный список — в [ARCHITECTURE.md, раздел 5](ARCHITECTURE.md). Ключевые:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `EMBEDDING_PROVIDER` | `default` | `default` — встроенная модель ChromaDB; `ollama` — внешняя `nomic-embed-text` |
| `LLM_MODEL` | `qwen2.5:3b-instruct` | локальная LLM |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | адрес Ollama |
| `CHROMA_DIR` | `./data/chroma` | каталог векторного хранилища |
| `MCP_TRANSPORT` | `stdio` | `stdio` или `http` |
| `LOG_LEVEL` | `INFO` | уровень логов сервера (`DEBUG` — подробнее) |

**Вычисления на CPU или GPU.** В `docker-compose.yml` Ollama по умолчанию
запускается принудительно на CPU (`CUDA_VISIBLE_DEVICES=-1` в сервисе `ollama`).
Если GPU работает стабильно и нужен он:

```bash
# Требуется: NVIDIA-драйвер + nvidia-container-toolkit на хосте.
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Проверить, где крутится модель: `docker compose exec ollama ollama ps` — в колонке
`PROCESSOR` будет `100% CPU` либо GPU-слои.

> При смене `EMBEDDING_PROVIDER` векторы несравнимы между моделями — очисти `CHROMA_DIR`
> и переиндексируй.

## Логи и отладка

Три поверхности логов — от клиента к LLM.

### 1. Вызовы MCP-инструментов (лог host-агента / opencode)

```bash
tail -f ~/.local/share/opencode/log/*.log   # вызовы инструментов: аргументы и результаты
opencode --log-level DEBUG --print-logs     # или вывести логи в терминал при старте
```

При stdio-транспорте stderr MCP-сервера тоже попадает в этот лог.

### 2. Логи RAG-сервера

Каждый вызов инструмента пишется строкой вида
`tool=index_folder args={'folder_path': 'demo_docs/proust', ...}` (stderr сервера).
Уровень настраивается переменной `LOG_LEVEL` (по умолчанию `INFO`):

```bash
LOG_LEVEL=DEBUG python -m src.server
```

Для opencode добавить в `opencode.json` → `mcp.rag-kb.environment`:
```json
{ "LOG_LEVEL": "DEBUG" }
```

### 3. Вызовы локальной LLM

Со стороны сервера (в логах RAG-сервера) каждый вызов генерации виден вместе с
длительностью — по ним видно, сколько раз граф обратился к LLM (rewrite /
grade / broaden / generate):

```
ollama generate model=qwen2.5:3b-instruct prompt_chars=1842 json=True
ollama done model=qwen2.5:3b-instruct answer_chars=312 elapsed=14.3s
```

Со стороны самой Ollama — полный трейс запросов:

```bash
docker compose logs -f ollama        # контейнер из docker-compose.yml
journalctl --user -u ollama -f       # нативный сервис (или ~/.ollama/logs/server.log)
OLLAMA_DEBUG=1 ollama serve          # максимально подробный режим
```

## Отчёт: стратегия нарезки на чанки

Подход подобран по природе данных; цель — не рвать смысловые единицы.

- **Текст/документация (`.md`, `.txt`)** — `RecursiveCharacterTextSplitter`
  (1000 символов / overlap 150). Обычный разделитель иерархии markdown не учитывает,
  поэтому recursive-подход по абзацам/строкам плюс перекрытие сохраняют контекст
  на стыках: соседние чанки делят общий «хвост», вопрос по стыку находит оба.
- **Код (`.py`, `.js`, `.ts`)** — `RecursiveCharacterTextSplitter.from_language()`
  с грамматикой конкретного языка. Режет по границам функций/классов, а не посреди
  блока: фрагмент кода остаётся синтаксически целым, что важно и для эмбеддингов,
  и для грейдера. Длинные функции дополнительно режутся по строкам с overlap.
- **JSON/YAML (`.json`, `.yaml`)** — файл парсится как объект, текст строится
  по логическим группам верхнего уровня: «путь ключа + полное поддерево значений»
  (`notifications:` с вложенными полями). Это сохраняет иерархию ключ→значение —
  модель видит связь настроек с их секцией. Слишком большие группы режутся
  рекурсивным сплиттером. Номера строк для таких чанков неопределены (`null`).

Метаданные каждого чанка: `source` (файл), `file_type`, `chunk_index`,
`line_number` (приблизительный, для текста/кода — поиск позиции чанка
от курсора предыдущего).

## Тесты и CI

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

Все тесты идут без сети и без LLM: внешние интеграции подменяются fake-реализациями
доменных интерфейсов (в этом и смысл DIP-слоя). CI (GitHub Actions): ruff + pytest,
сборка Docker-образа отдельной job'ой.

## Осознанные отклонения от ТЗ

- Собственная реализация RRF в `HybridRetriever` вместо готового
  `langchain.EnsembleRetriever` — контроль токенизации BM25 (camelCase/snake_case),
  параллельного запуска и детерминизма слияния. Чанкинг — на
  `langchain-text-splitters` из экосистемы LangChain.
- Эмбеддинги вычисляются отдельно и передаются в ChromaDB явно — так
  переключение провайдера (дефолт/Ollama) не требует правок кода.
