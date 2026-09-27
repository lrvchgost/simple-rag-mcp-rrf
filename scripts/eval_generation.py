"""Оценка качества генерации: fact-check + LLM-judge (faithfulness).

Для каждого вопроса golden-набора:
  1. ask_question (MCP) — реальный ответ системы;
  2. детерминированная проверка: содержит ли ответ опорный факт (fact)
     и нормален ли отказ для expect_refusal;
  3. LLM-judge (host ollama, qwen2.5:3b-instruct): верен ли ответ относительно
     контекста, с обязательной цитатой из контекста на каждое утверждение
     (quote-рубрика — защита от «3b судит 3b»).

Медленно: ~30-70 c на вопрос (CPU). Запускать в фоне.
    .venv/bin/python scripts/eval_generation.py --json-out evals/gen-baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import GOLDEN, call_tool, find_hit, mcp_session  # noqa: E402

JUDGE_MODEL = "qwen2.5:3b-instruct"
OLLAMA = "http://localhost:11434"

_STOP = {"a", "an", "the", "of", "in", "to", "is", "are", "и", "в", "на", "с",
         "для", "же", "или", "как", "что"}


def fact_keywords(fact: str) -> set[str]:
    words = re.findall(r"[a-zа-яё0-9]+", fact.lower())
    return {w for w in words if w not in _STOP and (w.isdigit() or len(w) > 2)}


def fact_covered(fact: str, answer: str) -> float:
    """Доля опорных токенов факта, присутствующих в ответе."""
    if not fact:
        return 0.0
    toks, ans = fact_keywords(fact), answer.lower()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in ans) / len(toks)


def refusal_detected(answer: str) -> bool:
    a = answer.lower()
    return ("не найдено релевантных" in a
            or "no relevant" in a
            or "no information" in a
            or "insufficient" in a
            or "не содерж" in a
            or "does not contain" in a
            or "does not mention" in a)


JUDGE_SYSTEM = (
    "You verify a RAG answer. You get: the question, the CONTEXT excerpt, "
    "and the ANSWER. For EACH factual claim in the ANSWER find the sentence "
    "in CONTEXT that supports it and quote it. Claims with no supporting "
    "sentence in CONTEXT are unsupported. Answer ONLY with JSON: "
    '{"faithful": true|false, "quotes": ["..."], "unsupported": ["..."], '
    '"verdict_reason": "..."} where faithful = the answer makes at least one '
    "supported claim and contains no unsupported factual claims."
)


def judge(question: str, context: str, answer: str, fact: str,
          timeout: int = 120) -> dict:
    prompt = (f"Question: {question}\nExpected key fact: {fact or 'n/a'}\n\n"
              f"CONTEXT:\n{context[:1500]}\n\nANSWER:\n{answer}\n\n"
              "Verify per the system instructions.")
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": JUDGE_MODEL, "prompt": prompt, "system": JUDGE_SYSTEM,
              "format": "json", "stream": False,
              "options": {"temperature": 0}},
        timeout=timeout,
    )
    r.raise_for_status()
    try:
        return json.loads(r.json()["response"])
    except json.JSONDecodeError:
        return {"faithful": None, "quotes": [], "unsupported": [],
                "verdict_reason": "judge returned non-JSON"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000/mcp")
    ap.add_argument("--json-out")
    ap.add_argument("--min-fact-coverage", type=float, default=0.6)
    args = ap.parse_args()

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    session, sid = mcp_session(args.host)

    rows = []
    for i, q in enumerate(golden["questions"], start=1):
        row = {"id": q["id"], "category": q["id"].split("-")[0]}
        try:
            pool = call_tool(session, args.host, sid, "find_relevant_docs",
                             {"query": q["question"], "top_k": 3},
                             call_id=300 + i, timeout=180)
            gold_rank = find_hit(pool.get("results", []),
                                 q["expected_chunks"], None, 30) \
                if q["expected_chunks"] else None
            ctx = ""
            if gold_rank:
                ctx = pool["results"][gold_rank - 1].get("snippet", "")

            answer = call_tool(session, args.host, sid, "ask_question",
                               {"question": q["question"]},
                               call_id=400 + i, timeout=600)
        except requests.RequestException as exc:
            print(f"  !! {q['id']}: сбой ({exc.__class__.__name__})")
            rows.append({**row, "error": True})
            continue

        ans = answer.get("answer", "")
        row["answer"] = ans  # сохраняем для разбора и переоценки без перезапуска
        row["loops"] = answer.get("stats", {}).get("retrieval_loops_used")

        if q.get("expect_refusal"):
            row["refusal_ok"] = refusal_detected(ans)
            row["verdict"] = "refusal OK" if row["refusal_ok"] else "REFUSAL FAIL"
            rows.append(row)
            print(f"{row['id']:30s} {row['verdict']}")
            continue

        row["fact_coverage"] = round(fact_covered(q.get("fact", ""), ans), 2)
        row["fact_ok"] = row["fact_coverage"] >= args.min_fact_coverage
        try:
            j = judge(q["question"], ctx, ans, q.get("fact", ""))
        except requests.RequestException as exc:
            j = {"faithful": None, "verdict_reason":
                 f"judge unreachable: {exc.__class__.__name__}"}
        row["faithful"] = j.get("faithful")
        row["unsupported"] = j.get("unsupported", [])[:3]
        # Вердикт по детерминированному fact-чеку; faithful — advisory
        # (3b-judge ненадёжен на кросс-языковых парах, см. discovery/11).
        row["verdict"] = "OK" if row["fact_ok"] else "FACT MISS"
        rows.append(row)
        print(f"{row['id']:30s} {row['verdict']:15s} "
              f"fact={row['fact_coverage']:.2f} faithful={row['faithful']}")
        time.sleep(0.2)

    graded = [r for r in rows if not r.get("error")
              and not r.get("refusal_ok")]
    refusals = [r for r in rows if r.get("refusal_ok") is not None]
    ok = [r for r in graded if r.get("verdict") == "OK"]
    print("\nИТОГО (генерация):")
    print(f"  fact_ok (OK): {len(ok)}/{len(graded)}")
    print(f"  промах факта: {sum(1 for r in graded if r.get('verdict') == 'FACT MISS')}")
    faithful = [r for r in graded if r.get("faithful") is True]
    print(f"  judge faithful (advisory): {len(faithful)}/{len(graded)}")
    if refusals:
        print(f"  честный отказ: {sum(1 for r in refusals if r['refusal_ok'])}/{len(refusals)}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\nотчёт: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
