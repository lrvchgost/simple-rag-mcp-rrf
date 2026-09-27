import pytest
from src.domain.models import ScoredChunk
from src.domain.retriever import HybridRetriever

from tests.conftest import FakeSparseStorage, FakeVectorStorage, make_chunk

RANK_BASE = 60


def test_rrf_merges_both_lists():
    a, b, c = make_chunk("a", "a"), make_chunk("b", "b"), make_chunk("c", "c")
    vector = FakeVectorStorage(embedder=None, canned=[[ScoredChunk(a, 0.1), ScoredChunk(b, 0.2)]])
    sparse = FakeSparseStorage(canned=[[ScoredChunk(b, 5.0), ScoredChunk(c, 4.0)]])
    retriever = HybridRetriever(vector, sparse, rrf_k=RANK_BASE)

    fused = retriever.search("q", top_k=3)

    assert [s.chunk.id for s in fused] == ["b", "a", "c"]
    expected_b = 1 / (RANK_BASE + 2) + 1 / (RANK_BASE + 1)  # dense ранг 2, sparse ранг 1
    assert fused[0].score == pytest.approx(expected_b)


def test_rrf_tie_prefers_dense_order():
    """a: (dense 1, sparse 2), b: (dense 2, sparse 1) — баллы равны, dense впереди."""
    a, b = make_chunk("a", "a"), make_chunk("b", "b")
    vector = FakeVectorStorage(embedder=None, canned=[[ScoredChunk(a, 0.1), ScoredChunk(b, 0.2)]])
    sparse = FakeSparseStorage(canned=[[ScoredChunk(b, 1.0), ScoredChunk(a, 0.9)]])
    retriever = HybridRetriever(vector, sparse, rrf_k=RANK_BASE)

    fused = retriever.search("q", top_k=2)

    assert fused[0].score == fused[1].score
    assert [s.chunk.id for s in fused] == ["a", "b"]


def test_dense_failure_does_not_break_search():
    a = make_chunk("a", "a")

    class BrokenVector(FakeVectorStorage):
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            raise RuntimeError("dense down")

    sparse = FakeSparseStorage(canned=[[ScoredChunk(a, 3.0)]])
    retriever = HybridRetriever(BrokenVector(embedder=None), sparse, rrf_k=RANK_BASE)
    fused = retriever.search("q", top_k=3)
    assert [s.chunk.id for s in fused] == ["a"]


def test_empty_query_returns_empty():
    retriever = HybridRetriever(FakeVectorStorage(None), FakeSparseStorage())
    assert retriever.search("   ", top_k=3) == []


def test_fused_list_truncated_to_top_k():
    a, b, c = make_chunk("a", "a"), make_chunk("b", "b"), make_chunk("c", "c")
    vector = FakeVectorStorage(embedder=None, canned=[[ScoredChunk(a, 0.1), ScoredChunk(b, 0.2)]])
    sparse = FakeSparseStorage(canned=[[ScoredChunk(b, 1.0), ScoredChunk(c, 0.9)]])
    retriever = HybridRetriever(vector, sparse, rrf_k=RANK_BASE)

    fused = retriever.search("q", top_k=2)

    assert len(fused) == 2  # объединение даёт 3 уникальных чанка — оставляем 2 лучших
