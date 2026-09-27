"""HybridRetriever: параллельный dense+sparse поиск и слияние по RRF."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from src.domain.interfaces import SparseStorage, VectorStorage
from src.domain.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Гибридный поиск: BM25 покрывает точные ключевые слова и токены кода,
    векторный поиск — смысл запроса; результаты сливаются по рангам (RRF)."""

    def __init__(self, vector: VectorStorage, sparse: SparseStorage, rrf_k: int = 60) -> None:
        self._vector = vector
        self._sparse = sparse
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not query.strip() or top_k <= 0:
            return []
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self._safe_dense, query, top_k)
            sparse_future = pool.submit(self._sparse.search, query, top_k)
            dense = dense_future.result()
            sparse = sparse_future.result()

        # rrf_score = 1/(k + r_dense) + 1/(k + r_sparse); отсутствующее слагаемое = 0
        by_id: dict[str, tuple[Chunk, float]] = {}
        for rank, scored in enumerate(dense, start=1):
            _, score = by_id.get(scored.chunk.id, (scored.chunk, 0.0))
            by_id[scored.chunk.id] = (scored.chunk, score + 1.0 / (self._rrf_k + rank))
        for rank, scored in enumerate(sparse, start=1):
            chunk, score = by_id.get(scored.chunk.id, (scored.chunk, 0.0))
            by_id[scored.chunk.id] = (chunk, score + 1.0 / (self._rrf_k + rank))

        # Сортировка стабильна: при равенстве баллов dense-выдача идёт раньше
        fused = [ScoredChunk(chunk=c, score=s) for c, s in by_id.values()]
        fused.sort(key=lambda s: s.score, reverse=True)
        return fused[:top_k]

    def _safe_dense(self, query: str, top_k: int) -> list[ScoredChunk]:
        try:
            return self._vector.search(query, top_k)
        except Exception:  # noqa: BLE001 — отказ одной ветки не должен ронять поиск
            logger.exception("Ветвь dense-поиска упала; продолжаем только с BM25")
            return []
