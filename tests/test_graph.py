import pytest
from src.domain.models import ScoredChunk
from src.domain.rag_graph import CorrectiveRAGGraph
from src.domain.retriever import HybridRetriever

from tests.conftest import FakeLLM, FakeSparseStorage, FakeVectorStorage, grade_json, make_chunk


def build_graph(vector_canned, llm_responses, **kwargs):
    vector = FakeVectorStorage(embedder=None, canned=vector_canned)
    retriever = HybridRetriever(vector, FakeSparseStorage(), rrf_k=60)
    llm = FakeLLM(responses=list(llm_responses))
    graph = CorrectiveRAGGraph(retriever, llm, **kwargs)
    return graph, llm, vector


def test_happy_path_generates_answer_with_sources():
    chunks = [make_chunk("a", "токен живёт 24 часа", "docs/auth.md"), make_chunk("b", "refresh 30 дней", "docs/auth.md")]
    graph, llm, vector = build_graph(
        vector_canned=[[ScoredChunk(chunks[0], 0.1), ScoredChunk(chunks[1], 0.2)]],
        llm_responses=[grade_json([True, True]), "Доступ выдаётся на 24 часа."],
    )
    result = graph.run("Какой срок жизни токена?")

    assert result.answer == "Доступ выдаётся на 24 часа."
    assert result.sources == ["docs/auth.md"]
    assert result.chunks_relevant == 2
    assert result.loops_used == 0
    # первый поиск — по исходному вопросу, без rewrite
    assert vector.queries == ["Какой срок жизни токена?"]
    # вызовы LLM: grade, generate (rewrite на happy path не выполняется)
    assert len(llm.prompts) == 2


def test_few_relevant_triggers_rewrite_and_retry():
    irrelevant = make_chunk("x", "совсем не то", "docs/other.md")
    relevant = [make_chunk("a", "токен 24 часа", "docs/auth.md"), make_chunk("b", "refresh 30 дней", "docs/auth.md")]
    graph, llm, vector = build_graph(
        vector_canned=[
            [ScoredChunk(irrelevant, 0.1)],
            [ScoredChunk(relevant[0], 0.1), ScoredChunk(relevant[1], 0.2)],
        ],
        llm_responses=[
            grade_json([False]),
            "узкий запрос",
            grade_json([True, True]),
            "Итоговый ответ.",
        ],
        min_relevant=2,
    )
    result = graph.run("Сколько живёт токен?")

    assert result.answer == "Итоговый ответ."
    assert result.loops_used == 1
    assert result.chunks_relevant == 2
    # 1-й ретрай — rewrite; поиск идёт по переписанному запросу
    assert vector.queries == ["Сколько живёт токен?", "узкий запрос"]
    # grade, rewrite, grade, generate
    assert len(llm.prompts) == 4


def test_loop_fuse_caps_retry_cycles():
    stuck = make_chunk("x", "совсем не то", "docs/other.md")
    graph, llm, vector = build_graph(
        vector_canned=[[ScoredChunk(stuck, 0.1)]] * 3,
        llm_responses=[
            grade_json([False]),
            "запрос",
            grade_json([False]),
            "расширение",
            grade_json([False]),
        ],
        max_loops=2,
    )
    result = graph.run("вопрос без ответа в базе")

    assert result.loops_used == 2
    assert "Не найдено релевантных фрагментов" in result.answer
    # 1-й ретрай — rewrite, 2-й — broaden
    assert vector.queries == ["вопрос без ответа в базе", "запрос", "расширение"]
    assert len(llm.prompts) == 5  # grade, rewrite, grade, broaden, grade


def test_grade_fallback_all_relevant_on_bad_json():
    chunks = [make_chunk("a", "текст про кэш", "docs/cache.md"), make_chunk("b", "ещё про кэш", "docs/cache.md")]
    graph, _, _ = build_graph(
        vector_canned=[[ScoredChunk(chunks[0], 0.1), ScoredChunk(chunks[1], 0.2)]],
        llm_responses=["это не json", "Ответ по кэшу."],
    )
    result = graph.run("как работает кэш?")

    assert result.answer == "Ответ по кэшу."
    assert result.chunks_relevant == 2


def test_parse_verdicts_tolerates_fences_and_string_bools():
    raw = '```json\n{"results": [{"index": 1, "relevant": "true"}, {"index": 2, "relevant": false}]}\n```'
    verdicts = CorrectiveRAGGraph._parse_verdicts(raw)
    assert verdicts == {1: True, 2: False}


def test_parse_verdicts_raises_without_json():
    with pytest.raises(ValueError):
        CorrectiveRAGGraph._parse_verdicts("вроде релевантно, да")


def test_refusal_message_is_bilingual():
    stuck = make_chunk("x", "совсем не то", "docs/other.md")
    graph, llm, _ = build_graph(
        vector_canned=[[ScoredChunk(stuck, 0.1)]] * 2,
        llm_responses=[
            grade_json([False]),
            "запрос",
            grade_json([False]),
        ],
        max_loops=1,
    )
    result = graph.run("question outside the corpus")

    assert "No relevant fragments found" in result.answer
    assert "Не найдено релевантных фрагментов" in result.answer
    # grade, rewrite, grade — generate не вызывается без чанков
    assert len(llm.prompts) == 3


def test_lexical_gate_refuses_offtopic_english_question():
    noise = make_chunk("n", "пруст и время в Комбре", "docs/proust.md")
    graph, llm, _ = build_graph(
        vector_canned=[[ScoredChunk(noise, 0.1)]] * 2,
        llm_responses=[
            grade_json([True]),
            "unladen swallow",
            grade_json([True]),
        ],
        max_loops=1,
    )
    result = graph.run("What is the airspeed of an unladen swallow?")

    assert "No relevant fragments found" in result.answer
    # затвор сверяет исходный вопрос с чанками, а не переписанный запрос
    assert len(llm.prompts) == 3  # grade, rewrite, grade — generate не вызвал LLM


def test_lexical_gate_passes_on_topic_english_question():
    chunk = make_chunk("a", "Cottard is a physician of the Verdurin circle", "docs/p.md")
    graph, llm, _ = build_graph(
        vector_canned=[[ScoredChunk(chunk, 0.1)]],
        llm_responses=[grade_json([True]), "He is a physician."],
        min_relevant=1,
    )
    result = graph.run("Who is Professor Cottard?")

    assert result.answer == "He is a physician."
    assert result.sources == ["docs/p.md"]
    assert len(llm.prompts) == 2  # grade, generate — без rewrite и broaden
