"""Абстрактные контракты домена (DIP). Домен зависит только от этих абстракций."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.models import Chunk, ScoredChunk


class EmbeddingService(ABC):
    """Вычисление плотных векторов текста."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Возвращает вектор на каждый элемент texts (порядок сохраняется)."""


class VectorStorage(ABC):
    """Хранилище плотных векторов (dense retrieval)."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Добавляет чанки с готовыми эмбеддингами."""

    @abstractmethod
    def delete_by_sources(self, sources: list[str]) -> None:
        """Удаляет все чанки, у которых metadata.source входит в sources."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Топ-K релевантных чанков; score — мера близости (чем меньше, тем лучше)."""

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        """Все чанки хранилища (для восстановления BM25-индекса)."""

    @abstractmethod
    def count(self) -> int:
        """Общее число чанков."""


class SparseStorage(ABC):
    """Хранилище лексического (sparse) поиска, например BM25."""

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        """Строит (перестраивает) индекс по корпусу чанков."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Топ-K чанков; score — релевантность (чем больше, тем лучше)."""

    @abstractmethod
    def count(self) -> int:
        """Число документов в индексе."""


class LLMService(ABC):
    """Генерация текста локальной LLM."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Возвращает текст ответа модели; json_mode просит строгий JSON."""

    @abstractmethod
    def check_availability(self) -> None:
        """Бросает OllamaUnavailableError, если сервис недоступен."""
