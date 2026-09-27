"""Реализации EmbeddingService: дефолт ChromaDB и внешняя модель через Ollama."""

from __future__ import annotations

import requests

from src.domain.errors import OllamaUnavailableError
from src.domain.interfaces import EmbeddingService


class DefaultChromaEmbedding(EmbeddingService):
    """Дефолтная модель ChromaDB (ONNX MiniLM-L6-v2): локально, без внешних API.

    Модель скачивается один раз (~80 МБ) в кэш при первом использовании.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._ef = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._ef(texts)
        return [list(map(float, v)) for v in vectors]


class OllamaEmbedding(EmbeddingService):
    """Эмбеддинги через Ollama REST API (POST /api/embed), например nomic-embed-text."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = requests.post(
                self._url,
                json={"model": self._model, "input": texts},
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Ollama embeddings unavailable at {self._url}: {exc}"
            ) from exc
        data = resp.json()
        return [list(map(float, v)) for v in data["embeddings"]]
