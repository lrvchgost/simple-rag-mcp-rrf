from pathlib import Path

from src.domain.indexer import DocumentIndexer
from src.infrastructure.bm25_storage import BM25SparseStorage
from src.infrastructure.chroma_storage import ChromaVectorStorage

from tests.conftest import FakeEmbeddings


def build(tmp_path: Path):
    chroma_dir = tmp_path / "chroma"
    vector = ChromaVectorStorage(str(chroma_dir), FakeEmbeddings())
    sparse = BM25SparseStorage()
    indexer = DocumentIndexer(vector, sparse, FakeEmbeddings(), chunk_size=200, chunk_overlap=30)
    return vector, sparse, indexer


def write_docs(folder: Path) -> None:
    (folder / "notes.md").write_text(
        "# Заголовок\n\n" + "Длинный абзац про кэширование и инвалидацию. " * 20,
        encoding="utf-8",
    )
    (folder / "util.py").write_text(
        "def add(a, b):\n    '''Складывает числа.'''\n    return a + b\n\n\nclass Calc:\n    def mul(self, a, b):\n        return a * b\n",
        encoding="utf-8",
    )
    (folder / "settings.json").write_text(
        '{"notifications": {"email": true}, "monitoring": {"dashboard": "auth"}}',
        encoding="utf-8",
    )
    (folder / "cfg.yaml").write_text(
        "service:\n  name: auth\n  replicas: 3\nredis:\n  port: 6379\n",
        encoding="utf-8",
    )
    (folder / "binary.xyz").write_text("неподдерживаемый формат", encoding="utf-8")
    (folder / "broken.json").write_text("{не json", encoding="utf-8")


def test_index_folder_creates_chunks_with_metadata(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    write_docs(folder)
    vector, sparse, indexer = build(tmp_path)

    result = indexer.index_folder(str(folder))

    assert result.files_found == 5  # md, py, json, yaml + broken.json; binary.xyz отфильтрован
    assert result.files_indexed == 4
    assert result.files_skipped == 1  # broken.json
    assert result.chunks_created > 0
    assert vector.count() == result.chunks_created
    assert sparse.count() == vector.count()  # BM25 перестроен из того же корпуса

    sources = {c.source for c in vector.all_chunks()}
    assert str(folder / "notes.md") in sources
    assert str(folder / "binary.xyz") not in sources

    types = {c.file_type for c in vector.all_chunks()}
    assert types == {"md", "py", "json", "yaml"}


def test_text_chunks_have_line_numbers(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "notes.md").write_text("# А\n\nтекст один\n\n## Б\n\nтекст два\n", encoding="utf-8")
    vector, _, indexer = build(tmp_path)

    indexer.index_folder(str(folder))

    md_chunks = [c for c in vector.all_chunks() if c.file_type == "md"]
    assert md_chunks
    assert all(c.line_number is not None for c in md_chunks)


def test_json_yaml_chunks_keep_top_level_keys(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "settings.json").write_text('{"notifications": {"email": true}, "monitoring": {"d": 1}}', encoding="utf-8")
    (folder / "cfg.yaml").write_text("service:\n  name: auth\nredis:\n  port: 6379\n", encoding="utf-8")
    vector, _, indexer = build(tmp_path)

    indexer.index_folder(str(folder))

    texts = [c.text for c in vector.all_chunks()]
    assert any(t.startswith("notifications:") for t in texts)
    assert any(t.startswith("monitoring:") for t in texts)
    assert any(t.startswith("service:") and "name: auth" in t for t in texts)


def test_reindex_is_idempotent(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("первая версия документа", encoding="utf-8")
    vector, _, indexer = build(tmp_path)

    indexer.index_folder(str(folder))
    count_after_first = vector.count()
    result = indexer.index_folder(str(folder))

    assert vector.count() == count_after_first
    assert result.chunks_created == count_after_first


def test_reindex_picks_up_new_content(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    doc = folder / "a.md"
    doc.write_text("версия один", encoding="utf-8")
    vector, sparse, indexer = build(tmp_path)
    indexer.index_folder(str(folder))

    doc.write_text("версия два про rate limiter", encoding="utf-8")
    indexer.index_folder(str(folder))

    all_text = " ".join(c.text for c in vector.all_chunks())
    assert "версия один" not in all_text
    assert "rate limiter" in all_text
    assert sparse.count() == vector.count()


def test_missing_folder_raises(tmp_path):
    _, _, indexer = build(tmp_path)
    import pytest

    with pytest.raises(NotADirectoryError):
        indexer.index_folder(str(tmp_path / "nope"))
