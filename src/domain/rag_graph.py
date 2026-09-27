"""CorrectiveRAGGraph: оркестрация пайплайна ответа на langgraph.StateGraph.

Узлы: Rewrite -> Retrieve -> Grade -> (Generate | Broaden -> Retrieve, max N циклов).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.domain.interfaces import LLMService
from src.domain.models import ScoredChunk
from src.domain.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You convert a user question into a concise search query for a hybrid "
    "(keyword + semantic) search over project documentation. Keep domain terms "
    "and identifiers as-is. Output ONLY the query, nothing else."
)

_GRADE_SYSTEM = (
    "You grade retrieval chunks for answering the question. Mark relevant=true "
    "if the chunk contains ANY information useful for the answer: facts, names, "
    "definitions, context — even partial. Mark false ONLY if the chunk is clearly "
    "about a different topic. When unsure, answer true. "
    "Answer ONLY with JSON of shape "
    '{"results":[{"index":1,"relevant":true},{"index":2,"relevant":false}]} '
    "with an entry for every chunk."
)

_BROADEN_SYSTEM = (
    "You broaden a search query that returned too few relevant documents. "
    "Add synonyms, related terms and general category words. "
    "Output ONLY the new query, nothing else."
)

_GENERATE_SYSTEM = (
    "You answer the user's question using ONLY the provided context excerpts. "
    "If the context is insufficient, say so honestly and briefly. "
    "Answer in the same language as the question. Be concise and factual."
)


class RAGState(TypedDict, total=False):
    question: str
    rewritten: str
    retrieved: list[ScoredChunk]
    relevant: list[ScoredChunk]
    loop_count: int
    answer: str


@dataclass(frozen=True)
class RAGResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks_retrieved: int = 0
    chunks_relevant: int = 0
    loops_used: int = 0


class CorrectiveRAGGraph:
    """Строит и компилирует граф; единственная точка входа — run(question)."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMService,
        top_k: int = 5,
        min_relevant: int = 2,
        max_loops: int = 2,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._min_relevant = min_relevant
        self._max_loops = max_loops
        self._graph = self._build().compile()

    def run(self, question: str) -> RAGResult:
        final: RAGState = self._graph.invoke(
            {
                "question": question,
                "rewritten": question,
                "retrieved": [],
                "relevant": [],
                "loop_count": 0,
                "answer": "",
            }
        )
        relevant = final.get("relevant", [])
        sources: list[str] = []
        for scored in relevant:
            if scored.chunk.source not in sources:
                sources.append(scored.chunk.source)
        return RAGResult(
            answer=final.get("answer", ""),
            sources=sources,
            chunks_retrieved=len(final.get("retrieved", [])),
            chunks_relevant=len(relevant),
            loops_used=final.get("loop_count", 0),
        )

    # --------------------------- узлы графа ---------------------------

    def _rewrite(self, state: RAGState) -> dict:
        prompt = f"Question: {state['question']}\nSearch query:"
        try:
            rewritten = self._llm.generate(prompt, system_prompt=_REWRITE_SYSTEM)
        except Exception:  # noqa: BLE001 — падение rewrite не должно ломать пайплайн
            logger.exception("Rewrite Query не удался; используем исходный вопрос")
            rewritten = state["question"]
        return {"rewritten": rewritten or state["question"]}

    def _retrieve(self, state: RAGState) -> dict:
        return {"retrieved": self._retriever.search(state["rewritten"], self._top_k)}

    def _grade(self, state: RAGState) -> dict:
        retrieved = state["retrieved"]
        if not retrieved:
            return {"relevant": []}
        return {"relevant": self._grade_chunks(state["rewritten"], retrieved)}

    def _broaden(self, state: RAGState) -> dict:
        prompt = f"Question: {state['question']}\nCurrent query: {state['rewritten']}\nBroader query:"
        try:
            broadened = self._llm.generate(prompt, system_prompt=_BROADEN_SYSTEM)
        except Exception:  # noqa: BLE001
            logger.exception("Broaden Query не удался; используем предыдущий запрос")
            broadened = state["rewritten"]
        return {"rewritten": broadened or state["rewritten"], "loop_count": state["loop_count"] + 1}

    def _generate(self, state: RAGState) -> dict:
        relevant = state["relevant"]
        if not relevant:
            return {
                "answer": "Не найдено релевантных фрагментов в базе знаний. "
                "Попробуйте переформулировать вопрос или переиндексируйте документы."
            }
        context_blocks = "\n\n".join(
            f"[{i}] (source: {s.chunk.source})\n{s.chunk.text}"
            for i, s in enumerate(relevant, start=1)
        )
        prompt = f"Context:\n{context_blocks}\n\nQuestion: {state['question']}\nAnswer:"
        answer = self._llm.generate(prompt, system_prompt=_GENERATE_SYSTEM)
        return {"answer": answer}

    # --------------------------- переходы ---------------------------

    def _route(self, state: RAGState) -> str:
        enough = len(state["relevant"]) >= self._min_relevant
        exhausted = state["loop_count"] >= self._max_loops
        return "generate" if (enough or exhausted) else "broaden"

    def _build(self) -> StateGraph:
        builder: StateGraph = StateGraph(RAGState)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("broaden", self._broaden)
        builder.add_node("generate", self._generate)
        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            self._route,
            {"generate": "generate", "broaden": "broaden"},
        )
        builder.add_edge("broaden", "retrieve")
        builder.add_edge("generate", END)
        return builder

    # --------------------------- утилиты ---------------------------

    def _grade_chunks(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        listing = "\n\n".join(
            f"[{i}]\n{c.chunk.text}" for i, c in enumerate(chunks, start=1)
        )
        prompt = f"Question: {query}\n\nChunks:\n{listing}\n\nJSON verdict:"
        try:
            raw = self._llm.generate(prompt, system_prompt=_GRADE_SYSTEM, json_mode=True)
            verdicts = self._parse_verdicts(raw)
        except Exception:  # noqa: BLE001 — сбой грейдера не должен блокировать генерацию
            logger.exception("Grade Chunks не удался; считаем все чанки релевантными")
            return list(chunks)
        if len(verdicts) < len(chunks):
            # Модель ответила не на все чанки — недостающие считаем релевантными
            for i in range(len(chunks)):
                verdicts.setdefault(i + 1, True)
        return [c for i, c in enumerate(chunks, start=1) if verdicts.get(i, False)]

    @staticmethod
    def _parse_verdicts(raw: str) -> dict[int, bool]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Нет JSON в ответе грейдера: {raw[:200]}")
        data = json.loads(match.group(0))
        verdicts: dict[int, bool] = {}
        for item in data.get("results", []):
            index = int(item.get("index", 0))
            verdicts[index] = _as_bool(item.get("relevant", False))
        return verdicts


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"
