"""DocumentIndexer: файлы -> чанки -> хранилища. Выбор сплиттера по формату."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import yaml
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from src.domain.interfaces import EmbeddingService, SparseStorage, VectorStorage
from src.domain.models import Chunk, IndexingResult

logger = logging.getLogger(__name__)

_CODE_LANGUAGES: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
}

# Батч эмбеддингов за один запрос к EmbeddingService
_EMBED_BATCH = 64


class DocumentIndexer:
    """Индексирует папку: чтение, сплиттинг, эмбеддинги, запись в storages.

    Повторная индексация идемпотентна: чанки файлов текущего запуска сначала
    удаляются, затем записываются свежие; BM25 перестраивается целиком.
    """

    def __init__(
        self,
        vector: VectorStorage,
        sparse: SparseStorage,
        embeddings: EmbeddingService,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self._vector = vector
        self._sparse = sparse
        self._embeddings = embeddings
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index_folder(self, folder_path: str, glob_pattern: str = "**/*") -> IndexingResult:
        started = time.monotonic()
        folder = Path(folder_path).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"Путь не существует или не папка: {folder}")

        files = sorted(
            p
            for p in folder.glob(glob_pattern)
            if p.is_file() and p.suffix.lower() in self._extensions()
        )

        chunks: list[Chunk] = []
        skipped: list[str] = []
        indexed = 0
        for path in files:
            try:
                file_chunks = self._split_file(path)
            except Exception as exc:  # noqa: BLE001 — повреждённый файл не должен ронять индексацию
                logger.warning("Пропущен файл %s: %s", path, exc)
                skipped.append(str(path))
                continue
            chunks.extend(file_chunks)
            indexed += 1

        sources = [str(p) for p in files]
        self._vector.delete_by_sources(sources)
        self._add_chunks(chunks)
        self._sparse.build(self._vector.all_chunks())

        return IndexingResult(
            files_found=len(files),
            files_indexed=indexed,
            files_skipped=len(skipped),
            chunks_created=len(chunks),
            duration_seconds=round(time.monotonic() - started, 2),
            skipped_files=skipped,
        )

    def _extensions(self) -> frozenset[str]:
        return frozenset({".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"})

    def _add_chunks(self, chunks: list[Chunk]) -> None:
        for start in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[start : start + _EMBED_BATCH]
            vectors = self._embeddings.embed([c.text for c in batch])
            self._vector.add(batch, vectors)

    def _split_file(self, path: Path) -> list[Chunk]:
        content = path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return []
        ext = path.suffix.lower()
        source = str(path)
        if ext in _CODE_LANGUAGES or ext in {".md", ".txt"}:
            pieces = self._split_text(content, ext)
            return self._to_chunks_with_lines(pieces, content, source, ext)
        return self._split_structured(content, source, ext)

    def _split_text(self, content: str, ext: str) -> list[str]:
        if ext in _CODE_LANGUAGES:
            splitter = RecursiveCharacterTextSplitter.from_language(
                _CODE_LANGUAGES[ext],
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        return splitter.split_text(content)

    def _split_structured(self, content: str, source: str, ext: str) -> list[Chunk]:
        """JSON/YAML: текст строится по группам верхнего уровня с сохранением иерархии."""
        data = json.loads(content) if ext == ".json" else yaml.safe_load(content)
        pieces: list[str] = []
        if isinstance(data, dict):
            for key, value in data.items():
                pieces.append(f"{key}:\n{_dump(value)}")
        elif isinstance(data, list):
            for i, value in enumerate(data):
                pieces.append(f"item[{i}]:\n{_dump(value)}")
        elif data is not None:
            pieces.append(content)

        chunks: list[Chunk] = []
        cursor = 0
        for _piece_index, piece in enumerate(pieces):
            sub_pieces = (
                self._split_text(piece, ".txt")
                if len(piece) > self._chunk_size
                else [piece]
            )
            for sub_index, sub in enumerate(sub_pieces):
                chunks.append(
                    Chunk(
                        id=f"{source}::{cursor + sub_index}",
                        text=sub,
                        source=source,
                        file_type=ext.lstrip("."),
                        chunk_index=len(chunks),
                    )
                )
            cursor += len(sub_pieces)
        return chunks

    def _to_chunks_with_lines(
        self, pieces: list[str], content: str, source: str, ext: str
    ) -> list[Chunk]:
        """Номера строк вычисляются приблизительно: поиск позиции от курсора."""
        chunks: list[Chunk] = []
        cursor = 0
        for index, piece in enumerate(pieces):
            pos = content.find(piece, cursor)
            line: int | None = None
            if pos != -1:
                line = content.count("\n", 0, pos) + 1
                cursor = pos + 1
            chunks.append(
                Chunk(
                    id=f"{source}::{index}",
                    text=piece,
                    source=source,
                    file_type=ext.lstrip("."),
                    chunk_index=index,
                    line_number=line,
                )
            )
        return chunks


def _dump(value: object) -> str:
    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False, default_flow_style=False)
