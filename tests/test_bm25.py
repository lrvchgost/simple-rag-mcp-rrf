from src.infrastructure.bm25_storage import BM25SparseStorage, tokenize

from tests.conftest import make_chunk


def test_tokenize_splits_identifiers():
    assert tokenize("TOKEN_EXPIRY_HOURS") == ["token", "expiry", "hours"]
    assert tokenize("getAccessToken") == ["get", "access", "token"]
    assert tokenize("Настройки auth-service") == ["настройки", "auth", "service"]


def test_bm25_finds_exact_token():
    storage = BM25SparseStorage()
    chunks = [
        make_chunk("1", "используется TOKEN_EXPIRY_HOURS из конфига"),
        make_chunk("2", "полностью про графики мониторинга"),
        make_chunk("3", "заметки о деплое и миграциях"),
    ]
    storage.build(chunks)
    results = storage.search("TOKEN_EXPIRY_HOURS", k=2)
    assert results
    assert results[0].chunk.id == "1"


def test_bm25_empty_index_returns_nothing():
    storage = BM25SparseStorage()
    assert storage.search("что угодно", k=3) == []


def test_bm25_zero_scores_filtered():
    storage = BM25SparseStorage()
    storage.build([make_chunk("1", "тема А")])
    assert storage.search("совершенно другое слово", k=3) == []
