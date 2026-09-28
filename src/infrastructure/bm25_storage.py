"""Адаптер sparse-поиска на rank_bm25 (BM25Okapi), индекс в памяти."""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from src.domain.interfaces import SparseStorage
from src.domain.models import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

def _stem(token: str) -> str:
    """Лёгкий стемминг английских окончаний: daughters -> daughter, cities -> city.

    Применяется и к запросу, и к корпусу, поэтому матчинг согласован;
    кириллица не затрагивается (правило смотрит только латинское "s").
    """
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Токены для BM25: нижний регистр, camelCase и snake_case разбираются на слова.

    "TOKEN_EXPIRY_HOURS" -> ["token", "expiry", "hour"],
    "getAccessToken"     -> ["get", "access", "token"],
    "daughters"          -> ["daughter"].
    """
    text = _CAMEL_RE.sub(" ", text)
    text = text.replace("_", " ").replace("-", " ").lower()
    return [_stem(t) for t in _TOKEN_RE.findall(text)]


class BM25SparseStorage(SparseStorage):
    """Точный полнотекстовый поиск по ключевым словам и токенам кода."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize(c.text) for c in chunks]
        with self._lock:
            self._chunks = list(chunks)
            self._index = BM25Okapi(corpus) if chunks else None

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        with self._lock:
            index, chunks = self._index, self._chunks
            if index is None or not chunks:
                return []
            scores = index.get_scores(tokenize(query))
            ranked = sorted(zip(scores, chunks, strict=False), key=lambda pair: pair[0], reverse=True)
            top = [(s, c) for s, c in ranked[:k] if s > 0]
        return [ScoredChunk(chunk=c, score=float(s)) for s, c in top]

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)
