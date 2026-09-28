"""HybridRetriever: параллельный dense+sparse поиск и слияние по взвешенному RRF."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from src.domain.interfaces import SparseStorage, VectorStorage
from src.domain.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

_ABSENT = float("inf")


class HybridRetriever:
    """Гибридный поиск: BM25 покрывает точные ключевые слова и токены кода,
    векторный поиск — смысл запроса; результаты сливаются по рангам (RRF).

    rrf_score = w_dense/(k + r_dense) + w_sparse/(k + r_sparse);
    отсутствующее слагаемое = 0. Веса компенсируют разницу силы ветвей.

    Тай-брейк при равном балле: выше чанк с лучшим рангом в BM25-ветви,
    затем с лучшим dense-рангом (отсутствующий ранг считается худшим).
    """

    def __init__(
        self,
        vector: VectorStorage,
        sparse: SparseStorage,
        rrf_k: int = 60,
        w_dense: float = 1.0,
        w_sparse: float = 1.0,
    ) -> None:
        self._vector = vector
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._w_dense = w_dense
        self._w_sparse = w_sparse

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not query.strip() or top_k <= 0:
            return []
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self._safe_dense, query, top_k)
            sparse_future = pool.submit(self._sparse.search, query, top_k)
            dense = dense_future.result()
            sparse = sparse_future.result()

        # id -> (chunk, score, rank_dense, rank_sparse)
        by_id: dict[str, tuple[Chunk, float, int | None, int | None]] = {}
        for rank, scored in enumerate(dense, start=1):
            _, score, _, r_sparse = by_id.get(scored.chunk.id, (scored.chunk, 0.0, None, None))
            by_id[scored.chunk.id] = (
                scored.chunk,
                score + self._w_dense / (self._rrf_k + rank),
                rank,
                r_sparse,
            )
        for rank, scored in enumerate(sparse, start=1):
            chunk, score, r_dense, _ = by_id.get(scored.chunk.id, (scored.chunk, 0.0, None, None))
            by_id[scored.chunk.id] = (
                chunk,
                score + self._w_sparse / (self._rrf_k + rank),
                r_dense,
                rank,
            )

        fused = sorted(
            by_id.values(),
            key=lambda e: (
                -e[1],
                e[3] if e[3] is not None else _ABSENT,
                e[2] if e[2] is not None else _ABSENT,
            ),
        )
        return [ScoredChunk(chunk=c, score=s) for c, s, _, _ in fused[:top_k]]

    def _safe_dense(self, query: str, top_k: int) -> list[ScoredChunk]:
        try:
            return self._vector.search(query, top_k)
        except Exception:  # noqa: BLE001 — отказ одной ветки не должен ронять поиск
            logger.exception("Ветвь dense-поиска упала; продолжаем только с BM25")
            return []
