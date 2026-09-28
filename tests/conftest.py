"""Fake-реализации доменных интерфейсов: тесты идут без сети и без LLM."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import pytest
from src.domain.interfaces import EmbeddingService, LLMService, SparseStorage, VectorStorage
from src.domain.models import Chunk, ScoredChunk

_DIM = 64


class FakeEmbeddings(EmbeddingService):
    """Мешок слов, спроецированный в фиксированный вектор (детерминированный)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * _DIM
            for token in re.findall(r"[a-zа-яё0-9]+", text.lower()):
                vec[_stable_hash(token) % _DIM] += 1.0
            vectors.append(vec)
        return vectors


def _stable_hash(token: str) -> int:
    value = 0
    for ch in token:
        value = (value * 31 + ord(ch)) % 1_000_003
    return value


class FakeVectorStorage(VectorStorage):
    """In-memory dense-хранилище с косинусной метрикой (или заранее заданные выдачи)."""

    def __init__(self, embedder: EmbeddingService, canned: list[list[ScoredChunk]] | None = None):
        self._embedder = embedder
        self._chunks: dict[str, tuple[Chunk, list[float]]] = {}
        self._canned = list(canned) if canned else None
        self.search_calls = 0
        self.queries: list[str] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, embeddings, strict=False):
            self._chunks[chunk.id] = (chunk, vector)

    def delete_by_sources(self, sources: list[str]) -> None:
        self._chunks = {
            cid: item for cid, item in self._chunks.items() if item[0].source not in sources
        }

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.search_calls += 1
        self.queries.append(query)
        if self._canned is not None:
            idx = min(self.search_calls, len(self._canned)) - 1
            return self._canned[idx][:k]
        qvec = self._embedder.embed([query])[0]
        scored = []
        for chunk, vec in self._chunks.values():
            sim = _cosine(qvec, vec)
            scored.append(ScoredChunk(chunk=chunk, score=1.0 - sim))  # расстояние
        scored.sort(key=lambda s: s.score)
        return scored[:k]

    def all_chunks(self) -> list[Chunk]:
        return [chunk for chunk, _ in self._chunks.values()]

    def count(self) -> int:
        return len(self._chunks)


class FakeSparseStorage(SparseStorage):
    """Sparse-хранилище с заранее заданными выдачами по номеру вызова."""

    def __init__(self, canned: list[list[ScoredChunk]] | None = None):
        self._canned = list(canned) if canned else None
        self._corpus: list[Chunk] = []
        self.search_calls = 0

    def build(self, chunks: list[Chunk]) -> None:
        self._corpus = list(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.search_calls += 1
        if self._canned is not None:
            idx = min(self.search_calls, len(self._canned)) - 1
            return self._canned[idx][:k]
        return []

    def count(self) -> int:
        return len(self._corpus)


@dataclass
class FakeLLM(LLMService):
    """LLM-заглушка: раздаёт ответы из очереди и запоминает промпты."""

    responses: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    json_flags: list[bool] = field(default_factory=list)

    def generate(
        self, prompt: str, system_prompt: str | None = None, json_mode: bool = False
    ) -> str:
        self.prompts.append(prompt)
        self.json_flags.append(json_mode)
        if not self.responses:
            raise AssertionError("FakeLLM: очередь ответов исчерпана")
        return self.responses.pop(0)

    def check_availability(self) -> None:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def grade_json(verdicts: list[bool]) -> str:
    """JSON-ответ грейдера в формате, который ожидает граф."""
    return json.dumps(
        {"results": [{"index": i, "relevant": v} for i, v in enumerate(verdicts, start=1)]}
    )


def make_chunk(cid: str, text: str, source: str = "f.txt") -> Chunk:
    return Chunk(id=cid, text=text, source=source, file_type="txt")


@pytest.fixture
def embedder() -> FakeEmbeddings:
    return FakeEmbeddings()
