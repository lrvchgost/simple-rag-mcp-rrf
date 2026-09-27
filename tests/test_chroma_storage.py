from src.infrastructure.chroma_storage import ChromaVectorStorage

from tests.conftest import FakeEmbeddings, make_chunk


def test_add_count_all_roundtrip(tmp_path):
    storage = ChromaVectorStorage(str(tmp_path / "db"), FakeEmbeddings())
    chunks = [
        make_chunk("a::0", "текст альфа", "src/a.md"),
        make_chunk("b::0", "текст бета", "src/b.md"),
    ]
    storage.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert storage.count() == 2
    stored = {c.id: c for c in storage.all_chunks()}
    assert stored["a::0"].source == "src/a.md"
    assert stored["a::0"].text == "текст альфа"


def test_search_returns_nearest(tmp_path):
    storage = ChromaVectorStorage(str(tmp_path / "db"), FakeEmbeddings())
    chunks = [make_chunk("a", "текст альфа"), make_chunk("b", "текст бета")]
    storage.add(chunks, FakeEmbeddings().embed([c.text for c in chunks]))

    results = storage.search("поиск про альфа", k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "a"  # общие токены с запросом
    assert results[0].score >= 0.0  # расстояние (cosine): меньше — ближе


def test_delete_by_sources(tmp_path):
    storage = ChromaVectorStorage(str(tmp_path / "db"), FakeEmbeddings())
    storage.add(
        [make_chunk("a", "x", "src/a.md"), make_chunk("b", "y", "src/b.md")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    storage.delete_by_sources(["src/a.md"])
    assert storage.count() == 1
    assert {c.source for c in storage.all_chunks()} == {"src/b.md"}


def test_clear_and_persistence(tmp_path):
    db = str(tmp_path / "db")
    storage = ChromaVectorStorage(db, FakeEmbeddings())
    storage.add([make_chunk("a", "текст")], [[1.0, 0.0]])
    storage.clear()
    assert storage.count() == 0

    # новый клиент над тем же каталогом видит персистентные данные
    reopened = ChromaVectorStorage(db, FakeEmbeddings())
    reopened.add([make_chunk("b", "другой текст")], [[0.0, 1.0]])
    assert reopened.count() == 1
