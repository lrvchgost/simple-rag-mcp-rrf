# Итоги сессии: производительность и качество RAG-стека

Дата: 2026-09-27. Окружение: ноутбук (Intel UHD 620 + NVIDIA MX150 2 ГБ, 16 ГБ RAM), Docker Compose-стек: ollama + MCP-сервер RAG (Corrective RAG на LangGraph, qwen2.5:3b-instruct + nomic-embed-text, ChromaDB + BM25).

## 1. Segfault при запуске модели (ollama 0.31.1)

**Симптом:** `ollama run qwen2.5:1.5b-instruct` → `500 Internal Server Error: llama-server process has terminated: signal: segmentation fault (core dumped)`.

**Диагностика:** journalctl показал загрузку всех 29 слоёв через Vulkan-бэкенд на Intel UHD 620 (`Vulkan0 model buffer size = 752 MiB`) и падение llama-server на старте. Перекачка модели (`ollama pull`) не помогла — модель целая, виноват бэкенд.

**Решение:** откат на ранее скачанный ollama 0.5.12 (`/tmp/opencode/ollama-0.5.12.tgz`, CUDA v11/v12, без Vulkan):
```bash
sudo systemctl stop ollama
sudo tar -C /usr/local -xzf /tmp/opencode/ollama-0.5.12.tgz
sudo systemctl restart ollama
```
Результат: модель работает, `ollama ps` → `9%/91% CPU/GPU` (CUDA на MX150).

## 2. Переход RAG на qwen2.5:1.5b-instruct — отклонено

- Индексация **не использует LLM** (`index_folder` — только эмбеддинги), так что смена LLM её не ускоряет.
- A/B-качество: Corrective RAG делает несколько LLM-вызовов (грейдинг top_k=5 чанков + генерация); 1.5b слабее в грейдинге и JSON-выводе. Итог — не переходить: узкое место в качестве, а не в скорости.

## 3. Узкие места производительности (по логам)

1. Генерация LLM на CPU в контейнере (`CUDA_VISIBLE_DEVICES=-1`): 34–65 c на ответ ~155 символов — главный тормоз `ask_question`.
2. Мультипликатор Corrective RAG: до 2 циклов (rewrite → retrieve → grade → broaden) на вопрос.
3. Эмбеддинги на CPU: ~0.42 c/чанк (818 чанков ≈ 5.7 мин).
4. MCP-клиент (opencode) обрывал индексацию по таймауту 20 c.

Вывод: железо — фундаментальное ограничение, архитектура (число LLM-вызовов) — множитель.

## 4. GPU в Docker + таймауты (переносимость)

**Проблема:** наивный `gpus: all` в docker-compose.yml ломал `docker compose up` на машинах без NVIDIA.

**Решение:** GPU вынесен в опциональный override:
- `docker-compose.yml` — базовый CPU-вариант (как прежде), работает везде;
- `docker-compose.gpu.yml` — `gpus: all` + `CUDA_VISIBLE_DEVICES=""`; запуск
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
  (требует драйвер NVIDIA + nvidia-container-toolkit на хосте);
- `opencode.json`: MCP-таймаут 20 c → 600 c — индексация больше не обрывается;
- README задокументирован (базовый запуск + GPU-вариант).

Для MX150 2 ГБ qwen2.5:3b целиком не влезает — частичная выгрузка слоёв, ускорение генерации ~x2–4; эмбеддинги (nomic, ~275 МБ) уйдут в GPU целиком.

## 5. Прогресс индексации в MCP

**Проблема:** `index_folder` молчит минуты — «тишина» до финального ответа.

**Реализация:**
- `src/domain/indexer.py`: колбэк `on_progress(done, total, message)` на каждый батч эмбеддингов (по 64 чанка; батчирование уже было в коде), ошибки колбэка не роняют индексацию;
- `src/server.py`: `index_folder` стал async — тяжёлая работа в `asyncio.to_thread`, прогресс мостом `run_coroutine_threadsafe → ctx.report_progress(...)` (MCP progress-нотификации; без `progressToken` от клиента молча пропускаются);
- гарантированный канал прогресса: `docker compose logs -f rag-mcp` (`эмбеддинги N/M`).

Замер полного прогона: 22 файла, 818 чанков, 0 пропущено, **339.78 c**.

## 6. Качество Corrective RAG: фикс грейдера, потолок генератора

**Было:** грейдер с промптом «strict relevance grader» отбрасывал 4 из 5 чанков → лишний цикл поиска → тощий контекст. Слово «strict» малая модель трактует как «проще сказать false».

**Фикс** (src/domain/rag_graph.py, `_GRADE_SYSTEM`): критерий «relevant=true при ЛЮБОЙ полезной информации (даже частичной)», false только при явной другой теме, `when unsure → true`. Цена ошибки асимметрична: false positive отфильтрует генерация, false negative теряет золотой чанк навсегда.

**A/B на вопросе «Who is Odette de Crecy?»:**

| Метрика | Было | Стало |
|---|---|---|
| Релевантных чанков | 1/5 | 2/5 |
| Доп. циклы поиска | 2 | 0 |
| Источники | 1 | 2 (обa реально релевантны) |

**Генератор — 3 итерации промпта безуспешно** (откат к исходному):
1. исходный → ответ тонкий, но верный;
2. анти-галлюцинация → верно, но бесполезно («name of a character mentioned in the context»);
3. экстракция фактов → фактически неверно («wife of Comtesse Swann»).

Вывод: искажение фактов происходит на этапе синтеза — это **потолок 3b-модели**, а не промпта. Грейдер и поиск теперь работают хорошо; рычаг качества — GPU + более крупная модель для генерации (поиск/грейдинг можно оставить на лёгкой), либо 7b-модель при наличии GPU.

## Коммиты

- `59920b2` perf(docker): опциональный GPU через docker-compose.gpu.yml, таймаут MCP-клиента 600s
- `b8170c6` feat(indexer): прогресс индексации — батчевые логи эмбеддингов и MCP progress-нотификации
- `051b677` fix(rag): смягчение критерия грейдера — relevant=true при частичной пользе, unsure->true

## Открытые вопросы

- Установить nvidia-container-toolkit и проверить GPU-проброс контейнера (ускорит и генерацию, и индексацию).
- Рассмотреть более крупную модель только для генерации при GPU.
