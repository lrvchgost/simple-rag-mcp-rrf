"""Адаптер векторного хранилища на ChromaDB (in-process, персистентный на диске)."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.domain.interfaces import EmbeddingService, VectorStorage
from src.domain.models import Chunk, ScoredChunk

_COLLECTION = "rag_kb"


class ChromaVectorStorage(VectorStorage):
    """ChromaDB не вычисляет эмбеддинги сама: векторы передаются явно.

    Это позволяет подменить провайдера эмбеддингов (default / Ollama) без
    изменения этого адаптера.
    """

    def __init__(self, path: str, embedding_service: EmbeddingService) -> None:
        self._embedder = embedding_service
        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
            embeddings=embeddings,
        )

    def delete_by_sources(self, sources: list[str]) -> None:
        if not sources:
            return
        self._collection.delete(where={"source": {"$in": sources}})

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if self._collection.count() == 0 or k <= 0:
            return []
        query_vector = self._embedder.embed([query])[0]
        res = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = self._rows_to_chunks(res["ids"][0], res["documents"][0], res["metadatas"][0])
        distances = res["distances"][0]
        return [ScoredChunk(chunk=c, score=float(d)) for c, d in zip(chunks, distances, strict=False)]

    def all_chunks(self) -> list[Chunk]:
        if self._collection.count() == 0:
            return []
        res = self._collection.get(include=["documents", "metadatas"])
        return self._rows_to_chunks(res["ids"], res["documents"], res["metadatas"])

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _rows_to_chunks(
        ids: list[str], documents: list[str], metadatas: list[dict]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for cid, text, meta in zip(ids, documents, metadatas, strict=False):
            meta = dict(meta or {})
            line_number = meta.pop("line_number", None)
            source = str(meta.pop("source", ""))
            file_type = str(meta.pop("file_type", ""))
            chunk_index = int(meta.pop("chunk_index", 0))
            chunks.append(
                Chunk(
                    id=cid,
                    text=text or "",
                    source=source,
                    file_type=file_type,
                    chunk_index=chunk_index,
                    line_number=int(line_number) if line_number is not None else None,
                    extra={k: str(v) for k, v in meta.items()},
                )
            )
        return chunks
