"""Доменные модели данных. Никаких зависимостей от библиотек инфраструктуры."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """Единица знаний: фрагмент документа с метаданными происхождения."""

    id: str
    text: str
    source: str  # путь к файлу-источнику
    file_type: str  # расширение без точки, например "py"
    chunk_index: int = 0
    line_number: int | None = None  # приблизительный номер строки начала (1-based)
    extra: dict[str, str] = field(default_factory=dict)

    def metadata(self) -> dict[str, str | int]:
        """Плоские метаданные для передачи в хранилище."""
        meta: dict[str, str | int] = {
            "source": self.source,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
        }
        if self.line_number is not None:
            meta["line_number"] = self.line_number
        meta.update(self.extra)
        return meta


@dataclass(frozen=True)
class ScoredChunk:
    """Чанк с оценкой из поиска (расстояние, BM25-скор или RRF-балл)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IndexingResult:
    """Итог одной индексации папки."""

    files_found: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    duration_seconds: float = 0.0
    skipped_files: list[str] = field(default_factory=list)
