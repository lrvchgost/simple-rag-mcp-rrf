import pytest
from src.config import load_settings
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


def test_rrf_tie_prefers_better_sparse_rank():
    """a: (dense 1, sparse 2), b: (dense 2, sparse 1) — баллы равны,
    выше чанк с лучшим рангом в BM25-ветви (b)."""
    a, b = make_chunk("a", "a"), make_chunk("b", "b")
    vector = FakeVectorStorage(embedder=None, canned=[[ScoredChunk(a, 0.1), ScoredChunk(b, 0.2)]])
    sparse = FakeSparseStorage(canned=[[ScoredChunk(b, 1.0), ScoredChunk(a, 0.9)]])
    retriever = HybridRetriever(vector, sparse, rrf_k=RANK_BASE)

    fused = retriever.search("q", top_k=2)

    assert fused[0].score == fused[1].score
    assert [s.chunk.id for s in fused] == ["b", "a"]


def test_rrf_tie_present_sparse_beats_absent():
    """Равные баллы (каждый чанк только в одной ветви, ранги 1:1):
    чанк из sparse-ветви выше чанка только из dense."""
    a, b = make_chunk("a", "a"), make_chunk("b", "b")
    vector = FakeVectorStorage(embedder=None, canned=[[ScoredChunk(b, 0.1)]])
    sparse = FakeSparseStorage(canned=[[ScoredChunk(a, 1.0)]])
    retriever = HybridRetriever(vector, sparse, rrf_k=RANK_BASE)

    fused = retriever.search("q", top_k=2)

    assert fused[0].score == pytest.approx(fused[1].score)
    assert [s.chunk.id for s in fused] == ["a", "b"]


def test_weight_flips_fusion_order():
    """Тот же сценарий, что в tie-тесте, но w_sparse=2 ломает симметрию в пользу b."""
    a, b = make_chunk("a", "a"), make_chunk("b", "b")
    vector = FakeVectorStorage(embedder=None, canned=[[ScoredChunk(a, 0.1), ScoredChunk(b, 0.2)]])
    sparse = FakeSparseStorage(canned=[[ScoredChunk(b, 1.0), ScoredChunk(a, 0.9)]])

    even = HybridRetriever(vector, sparse, rrf_k=RANK_BASE).search("q", top_k=2)
    weighted = HybridRetriever(vector, sparse, rrf_k=RANK_BASE, w_sparse=2.0).search("q", top_k=2)

    assert even[0].score == even[1].score  # без весов — ничья
    assert weighted[0].chunk.id == "b"
    assert weighted[0].score > weighted[1].score
    expected_b = 2 / (RANK_BASE + 1) + 1 / (RANK_BASE + 2)
    assert weighted[0].score == pytest.approx(expected_b)


def test_dense_weight_applies_to_score():
    a = make_chunk("a", "a")
    vector = FakeVectorStorage(embedder=None, canned=[[ScoredChunk(a, 0.1)]])
    sparse = FakeSparseStorage(canned=[[]])
    retriever = HybridRetriever(vector, sparse, rrf_k=RANK_BASE, w_dense=2.0)

    fused = retriever.search("q", top_k=1)

    assert fused[0].score == pytest.approx(2 / (RANK_BASE + 1))


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


def test_settings_read_rrf_weights_from_env(monkeypatch):
    monkeypatch.delenv("RRF_W_DENSE", raising=False)
    monkeypatch.delenv("RRF_W_SPARSE", raising=False)
    defaults = load_settings()
    assert defaults.rrf_w_dense == 1.0
    assert defaults.rrf_w_sparse == 1.0

    monkeypatch.setenv("RRF_W_DENSE", "2.5")
    monkeypatch.setenv("RRF_W_SPARSE", "0.5")
    custom = load_settings()
    assert custom.rrf_w_dense == 2.5
    assert custom.rrf_w_sparse == 0.5
