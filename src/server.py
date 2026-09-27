"""Интерфейсный слой: FastMCP-сервер с 4 инструментами RAG-базы знаний.

Каждый инструмент снабжён подробным description (docstring): host-агент
самостоятельно решает, когда какой инструмент вызывать, без подсказок пользователя.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

from fastmcp import Context, FastMCP

from src.config import Settings, load_settings
from src.domain.errors import OllamaUnavailableError
from src.domain.indexer import DocumentIndexer
from src.domain.rag_graph import CorrectiveRAGGraph
from src.domain.retriever import HybridRetriever
from src.infrastructure.bm25_storage import BM25SparseStorage
from src.infrastructure.chroma_storage import ChromaVectorStorage
from src.infrastructure.embeddings import DefaultChromaEmbedding, OllamaEmbedding
from src.infrastructure.ollama_service import OllamaService

logger = logging.getLogger(__name__)

mcp = FastMCP("rag-knowledge-base")


@dataclass
class IndexStats:
    """Книжгалтерия интерфейсного слоя: итог последнего index_folder."""

    files_indexed: int | None = None
    last_indexed_at: str | None = None


@dataclass
class AppContext:
    """Собранные зависимости приложения (Dependency Injection)."""

    settings: Settings
    vector: ChromaVectorStorage
    sparse: BM25SparseStorage
    indexer: DocumentIndexer
    retriever: HybridRetriever
    graph: CorrectiveRAGGraph
    llm: OllamaService
    stats: IndexStats


@lru_cache(maxsize=1)
def get_context() -> AppContext:
    """Собирает инфраструктуру и домен один раз за жизнь процесса."""
    settings = load_settings()
    embedder = (
        OllamaEmbedding(
            settings.ollama_base_url, settings.embedding_model, settings.ollama_timeout
        )
        if settings.embedding_is_ollama
        else DefaultChromaEmbedding()
    )
    vector = ChromaVectorStorage(settings.chroma_dir, embedder)
    sparse = BM25SparseStorage()
    indexer = DocumentIndexer(
        vector, sparse, embedder, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    retriever = HybridRetriever(vector, sparse, rrf_k=settings.rrf_k)
    llm = OllamaService(settings)
    graph = CorrectiveRAGGraph(
        retriever,
        llm,
        top_k=settings.top_k,
        min_relevant=settings.min_relevant_chunks,
        max_loops=settings.max_retrieval_loops,
    )

    # BM25 in-memory: восстанавливаем из персистентной ChromaDB без повторных эмбеддингов
    chunks = vector.all_chunks()
    if chunks:
        sparse.build(chunks)
        logger.info("BM25-индекс восстановлен из ChromaDB: %d чанков", len(chunks))

    return AppContext(
        settings=settings,
        vector=vector,
        sparse=sparse,
        indexer=indexer,
        retriever=retriever,
        graph=graph,
        llm=llm,
        stats=IndexStats(),
    )


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _log_call(tool: str, **args) -> None:
    """Лог вызова MCP-инструмента: виден в stderr и в логах host-агента."""
    logger.info("tool=%s args=%s", tool, args)


# ------------------------------- инструменты -------------------------------


@mcp.tool()
async def index_folder(
    folder_path: str, glob_pattern: str = "**/*", ctx: Context | None = None
) -> str:
    """Индексирует локальную папку с документами и строит поисковую базу знаний.

    Сканирует folder_path по glob-паттерну, читает файлы (.md, .txt, .py, .js,
    .ts, .json, .yaml/.yml), разбивает на чанки (свой сплиттер под каждый формат),
    создаёт эмбеддинги, сохраняет в векторное хранилище ChromaDB и перестраивает
    BM25-индекс для гибридного поиска. Повторный вызов по той же папке безопасен:
    чанки перезаписываются, дубликатов не появляется.

    Вызывай этот инструмент ПЕРВЫМ, когда пользователь просит индексировать папку,
    добавить документы в базу знаний, обновить индекс — или до первого
    ask_question/find_relevant_docs, если база ещё пуста. Не вызывай для
    поисковых запросов — для этого есть ask_question и find_relevant_docs.

    Args:
        folder_path: путь к папке с документами (обязателен).
        glob_pattern: glob для отбора файлов, по умолчанию "**/*" (все подпапки);
            например "docs/**/*.md", чтобы взять только Markdown.

    Returns:
        JSON со статистикой: найдено/проиндексировано/пропущено файлов,
        число чанков, время индексации.
    """
    _log_call("index_folder", folder_path=folder_path, glob_pattern=glob_pattern)
    try:
        app = get_context()
        loop = asyncio.get_running_loop()

        def on_progress(done: int, total: int, message: str) -> None:
            """Мост из рабочего потока в MCP progress-нотификации.

            Без progressToken от клиента report_progress молча пропускает отправку.
            """
            if ctx is None:
                return
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(done, total, message), loop
            )

        result = await asyncio.to_thread(
            app.indexer.index_folder, folder_path, glob_pattern, on_progress
        )
        app.stats.files_indexed = result.files_indexed
        app.stats.last_indexed_at = datetime.now(UTC).isoformat()
        return _json(
            {
                "status": "ok",
                "files_found": result.files_found,
                "files_indexed": result.files_indexed,
                "files_skipped": result.files_skipped,
                "skipped_files": result.skipped_files,
                "chunks_created": result.chunks_created,
                "duration_seconds": result.duration_seconds,
                "total_chunks_in_kb": app.vector.count(),
            }
        )
    except (NotADirectoryError, ValueError) as exc:
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def ask_question(question: str) -> str:
    """Отвечает на вопрос пользователя на основе проиндексированных документов.

    Запускает полный Corrective RAG-пайплайн на LangGraph: переписывает запрос,
    выполняет гибридный поиск (BM25 + векторный, слияние через RRF), LLM-оценивает
    релевантность найденных чанков, при нехватке релевантных расширяет запрос и
    повторяет поиск (до 2 циклов), затем генерирует ответ строго по найденному
    контексту и прикладывает список файлов-источников. Требует запущенную Ollama.

    Вызывай, когда пользователю нужен ОТВЕТ на вопрос по содержимому
    проиндексированных документов: «как работает X», «какие есть настройки Y»,
    «что сказано в доках о Z». Не вызывай, если нужен просто список релевантных
    документов (используй find_relevant_docs) или индексация (index_folder).

    Args:
        question: вопрос пользователя естественным языком (любой язык).

    Returns:
        JSON с полем answer (текст ответа) и sources (пути файлов-источников).
    """
    _log_call("ask_question", question=question[:200])
    ctx = get_context()
    if ctx.vector.count() == 0:
        return _json(
            {
                "status": "empty_index",
                "message": "База знаний пуста. Сначала проиндексируйте папку "
                "инструментом index_folder.",
            }
        )
    try:
        ctx.llm.check_availability()
        result = ctx.graph.run(question)
        return _json(
            {
                "status": "ok",
                "answer": result.answer,
                "sources": result.sources,
                "stats": {
                    "chunks_retrieved": result.chunks_retrieved,
                    "chunks_relevant": result.chunks_relevant,
                    "retrieval_loops_used": result.loops_used,
                },
            }
        )
    except OllamaUnavailableError as exc:
        return _json({"status": "error", "message": str(exc)})


@mcp.tool()
def find_relevant_docs(query: str, top_k: int = 5) -> str:
    """Возвращает топ-K релевантных чанков документов БЕЗ генерации ответа.

    Быстрый гибридный поиск: BM25 (точные ключевые слова, идентификаторы кода) плюс
    векторный поиск (смысл запроса), результаты объединяются по Reciprocal Rank
    Fusion. LLM не используется, поэтому работает мгновенно. Требует выполненной
    индексации (index_folder), но не требует Ollama.

    Вызывай, когда пользователю нужно НАЙТИ где что-то упоминается: «в каких файлах
    используется TOKEN_EXPIRY_HOURS», «покажи документы про rate limiter», «найди
    место, где настраивается таймаут». Также полезен, чтобы показать источники или
    оценить покрытие базы знаний до полного вопроса. Для готового ответа используй
    ask_question.

    Args:
        query: поисковый запрос (ключевые слова, идентификаторы, фраза).
        top_k: сколько чанков вернуть, по умолчанию 5.

    Returns:
        JSON-список чанков: rank, score (RRF-балл), source (файл), line_number,
        snippet (начало текста чанка).
    """
    _log_call("find_relevant_docs", query=query, top_k=top_k)
    ctx = get_context()
    if ctx.vector.count() == 0:
        return _json(
            {
                "status": "empty_index",
                "message": "База знаний пуста. Сначала проиндексируйте папку "
                "инструментом index_folder.",
            }
        )
    results = ctx.retriever.search(query, top_k)
    return _json(
        {
            "status": "ok",
            "results": [
                {
                    "rank": i,
                    "score": round(s.score, 4),
                    "source": s.chunk.source,
                    "line_number": s.chunk.line_number,
                    "file_type": s.chunk.file_type,
                    "snippet": s.chunk.text[:300],
                }
                for i, s in enumerate(results, start=1)
            ],
        }
    )


@mcp.tool()
def index_status() -> str:
    """Показывает состояние базы знаний: статистику индексации.

    Возвращает число чанков в векторном хранилище, число документов в BM25-индексе,
    количество файлов последней индексации и её время, а также текущие настройки
    (модель LLM, провайдер эмбеддингов). LLM не требуется.

    Вызывай, когда пользователь спрашивает «что уже проиндексировано», «сколько
    документов в базе», «готова ли база знаний», «какая модель используется» —
    или сам проверь статус перед ask_question/find_relevant_docs, чтобы понять,
    нужна ли индексация.

    Returns:
        JSON со статистикой и настройками базы знаний.
    """
    _log_call("index_status")
    ctx = get_context()
    return _json(
        {
            "total_chunks": ctx.vector.count(),
            "bm25_documents": ctx.sparse.count(),
            "files_indexed_last_run": ctx.stats.files_indexed,
            "last_indexed_at": ctx.stats.last_indexed_at,
            "embedding_provider": ctx.settings.embedding_provider,
            "llm_model": ctx.settings.llm_model,
            "chroma_dir": ctx.settings.chroma_dir,
        }
    )


def main() -> None:
    """Точка входа: выбирает транспорт (stdio по умолчанию, http для Docker)."""
    logging.basicConfig(level=get_context().settings.log_level)
    ctx = get_context()
    if ctx.settings.mcp_transport == "http":
        mcp.settings.host = ctx.settings.mcp_host
        mcp.settings.port = ctx.settings.mcp_port
        mcp.run(transport="http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
