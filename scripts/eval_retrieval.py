"""Оценка качества retrieval: recall@k и MRR по golden-набору.

Гоняет find_relevant_docs через реальный MCP-сервер (streamable HTTP)
и сверяет выдачу с evals/golden.json. Без chat-LLM — можно запускать
при каждом изменении чанкинга/RRF/эмбеддера.

Использование:
    .venv/bin/python scripts/eval_retrieval.py                     # таблица
    .venv/bin/python scripts/eval_retrieval.py --json-out out.json # + файл
    .venv/bin/python scripts/eval_retrieval.py --fail-under 0.9    # для CI
    .venv/bin/python scripts/eval_retrieval.py --grade             # режим грейдера

Режим --grade: для каждого вопроса сравнивает пул до грейдера
(find_relevant_docs) с результатом ask_question (sources/stats):
- gold_preserved — дошёл ли золотой чанк до генерации (end-to-end,
  включая retry-петлю rewrite/broaden);
- chunks_relevant — сколько чанков грейдер удержал из top_k;
- refusal-вопросы: честный отказ ("Не найдено релевантных фрагментов").
Медленный режим: ask_question ~30-70 c на вопрос на CPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "golden.json"
DEFAULT_LINE_WINDOW = 30  # чанк может начинаться выше golden-строки


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def mcp_session(base_url: str) -> tuple[requests.Session, str]:
    """initialize → session id → notifications/initialized."""
    s = requests.Session()
    r = s.post(
        base_url,
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                         "clientInfo": {"name": "eval-retrieval", "version": "1.0"}}},
        headers={"Accept": "application/json, text/event-stream"},
        timeout=120,
    )
    r.raise_for_status()
    sid = r.headers.get("mcp-session-id", "")
    s.post(base_url, json={"jsonrpc": "2.0", "method": "notifications/initialized"},
           headers={"Accept": "application/json, text/event-stream",
                    "mcp-session-id": sid}, timeout=30)
    return s, sid


def call_tool(s: requests.Session, base_url: str, sid: str, name: str,
              arguments: dict, call_id: int, timeout: int = 180) -> dict:
    r = s.post(
        base_url,
        json={"jsonrpc": "2.0", "id": call_id, "method": "tools/call",
              "params": {"name": name, "arguments": arguments}},
        headers={"Accept": "application/json, text/event-stream",
                 "mcp-session-id": sid},
        timeout=timeout,
    )
    r.raise_for_status()
    text = r.content.decode("utf-8")  # r.text даёт latin-1 и ломает splitlines на кириллице
    # SSE-событие может быть разрезано на несколько "data:" строк — склеиваем
    for event in re.split(r"\n\s*\n", text):
        data_lines = [ln[6:] for ln in event.splitlines()
                      if ln.startswith("data: ")]
        if not data_lines:
            continue
        try:
            payload = json.loads("".join(data_lines))
        except json.JSONDecodeError:
            continue
        if payload.get("result") and payload.get("id") == call_id:
            content = payload["result"].get("content", [])
            if content:
                return json.loads(content[0]["text"])
    raise RuntimeError(f"нет результата в ответе MCP для {name}; raw={r.text[:400]!r}")


def find_hit(results: list[dict], expected: list[dict], root: str,
             line_window: int) -> int | None:
    """Ранг первого совпавшего чанка (1-based) или None.

    Совпадение: source совпадает по хвосту пути И
    (quote из golden встречается в сниппете — приоритет, ИЛИ golden-строка
    внутри окна от начала чанка). Два прохода: сначала цитаты по всем
    результатам, потом строки — цитата точнее указывает нужный чанк.
    """
    for pass_quote in (True, False):
        for exp in expected:
            exp_src, exp_line = exp["source"], exp.get("line")
            quote = norm(exp["quote"]) if exp.get("quote") else None
            for res in results:
                if not res["source"].rstrip("/").endswith("/" + exp_src.lstrip("/")):
                    continue
                if pass_quote:
                    if quote and quote in norm(res.get("snippet", "")):
                        return res["rank"]
                else:
                    res_line = res.get("line_number")
                    if exp_line is not None and res_line is not None and \
                       abs(int(res_line) - exp_line) <= line_window:
                        return res["rank"]
    return None


def grade_mode(args) -> int:
    """Оценка грейдера: сохранение gold + шум + честный отказ."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    questions = golden["questions"]
    if args.filter:
        questions = [q for q in questions
                     if q["id"].startswith(args.filter + "-")]
    session, sid = mcp_session(args.host)

    rows = []
    for i, q in enumerate(questions, start=1):
        row = {"id": q["id"], "category": q["id"].split("-")[0]}
        pool = None
        for attempt in (1, 2):  # transient-зависания Ollama не должны ронять весь прогон
            try:
                pool = call_tool(session, args.host, sid, "find_relevant_docs",
                                 {"query": q["question"], "top_k": args.top_k},
                                 call_id=100 + i)
                break
            except requests.RequestException as exc:
                print(f"  !! {q['id']}: find_relevant_docs попытка {attempt} "
                      f"не удалась ({exc.__class__.__name__})")
        if pool is None:
            rows.append({"id": q["id"], "category": q["id"].split("-")[0],
                         "gold_in_pool_rank": None, "error": True,
                         "chunks_relevant": None, "loops": None,
                         "gold_preserved": None, "refusal_ok": None})
            continue
        results = pool.get("results", [])
        gold_rank = find_hit(results, q["expected_chunks"], None,
                             args.line_window) \
            if q["expected_chunks"] else None
        row["gold_in_pool_rank"] = gold_rank

        try:
            answer = call_tool(session, args.host, sid, "ask_question",
                               {"question": q["question"]}, call_id=200 + i,
                               timeout=600)
        except requests.RequestException as exc:
            print(f"  !! {q['id']}: ask_question не ответил ({exc.__class__.__name__})")
            rows.append({"id": q["id"], "category": q["id"].split("-")[0],
                         "gold_in_pool_rank": gold_rank, "error": True,
                         "chunks_relevant": None, "loops": None,
                         "gold_preserved": None, "refusal_ok": None})
            continue
        stats = answer.get("stats", {})
        sources = answer.get("sources", [])
        row["chunks_relevant"] = stats.get("chunks_relevant")
        row["loops"] = stats.get("retrieval_loops_used")

        if q.get("expect_refusal"):
            row["refusal_ok"] = "Не найдено релевантных" in answer.get("answer", "")
            row["gold_preserved"] = None
        elif gold_rank is None:
            row["gold_preserved"] = None  # gold не найден поиском — грейдер не виноват
        else:
            exp_srcs = {ec["source"].split("/")[-1] for ec in q["expected_chunks"]}
            row["gold_preserved"] = any(
                src.split("/")[-1] in exp_srcs for src in sources)
        rows.append(row)

    graded = [r for r in rows if r["gold_preserved"] is not None]
    in_pool = [r for r in graded if r["gold_in_pool_rank"] is not None]
    preserved = [r for r in graded if r["gold_preserved"]]
    refusals = [r for r in rows if r.get("refusal_ok") is not None]
    ref_ok = [r for r in refusals if r["refusal_ok"]]

    print(f"Режим грейдера: {len(questions)} вопросов\n")
    print(f"{'id':30s} {'pool':>4s} {'kept':>4s} {'loops':>5s}  verdict")
    for r in rows:
        if r.get("refusal_ok") is not None:
            verdict = "refusal OK" if r["refusal_ok"] else "REFUSAL FAIL"
        elif r["gold_in_pool_rank"] is None:
            verdict = "gold вне пула (не вина грейдера)"
        else:
            verdict = "preserved" if r["gold_preserved"] else "LOST"
        kept = r["chunks_relevant"] if r["chunks_relevant"] is not None else "-"
        print(f"{r['id']:30s} {str(r['gold_in_pool_rank'] or '-'):>4s} "
              f"{str(kept):>4s} {str(r['loops']):>5s}  {verdict}")

    print("\nИТОГО:")
    print(f"  gold в пуле поиска:   {len(in_pool)}/{len(graded)}")
    print(f"  gold дошёл до генерации: {len(preserved)}/{len(in_pool)} "
          f"(grader gold-preservation)")
    kept_vals = [r["chunks_relevant"] for r in rows
                 if r.get("chunks_relevant") is not None]
    if kept_vals:
        print(f"  средне чанков удержано из top_k: "
              f"{sum(kept_vals)/len(kept_vals):.1f}")
    if refusals:
        print(f"  честный отказ: {len(ref_ok)}/{len(refusals)}")

    if args.json_out:
        report = {"mode": "grade", "golden_version":
                  golden["_meta"]["version"], "rows": rows}
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nотчёт: {args.json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000/mcp")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--line-window", type=int, default=DEFAULT_LINE_WINDOW)
    ap.add_argument("--json-out", help="сохранить отчёт в файл (baseline)")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="минимальный recall@k для exit code 0 (CI)")
    ap.add_argument("--filter", default=None,
                    help="категория вопросов по префиксу id (например tech)")
    ap.add_argument("--grade", action="store_true",
                    help="режим оценки грейдера (медленный: ask_question)")
    args = ap.parse_args()

    if args.grade:
        return grade_mode(args)

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    root = golden["_meta"]["source_root"]
    questions = [q for q in golden["questions"]
                 if not q.get("expect_refusal")]
    refusals = [q for q in golden["questions"] if q.get("expect_refusal")]
    if args.filter:
        questions = [q for q in questions
                     if q["id"].startswith(args.filter + "-")]
        refusals = []

    session, sid = mcp_session(args.host)
    rows = []
    for i, q in enumerate(questions, start=1):
        res = call_tool(session, args.host, sid, "find_relevant_docs",
                        {"query": q["question"], "top_k": args.top_k},
                        call_id=i + 10)
        results = res.get("results", [])
        rank = find_hit(results, q["expected_chunks"], root, args.line_window)
        rows.append({
            "id": q["id"], "category": q["id"].split("-")[0],
            "hit_rank": rank,
            "matched": (f"{results[rank-1]['source'].rsplit('/', 1)[-1]}:"
                        f"{results[rank-1]['line_number']}") if rank else "-",
        })
        time.sleep(0.1)

    def agg(rows_subset: list[dict]) -> dict:
        n = len(rows_subset)
        hits = [r for r in rows_subset if r["hit_rank"]]
        return {
            "n": n,
            "recall@k": round(len(hits) / n, 3) if n else None,
            "mrr": round(sum(1 / r["hit_rank"] for r in hits) / n, 3) if n else None,
        }

    total = agg(rows)
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        by_cat[cat] = agg([r for r in rows if r["category"] == cat])

    print(f"recall@{args.top_k} / MRR по golden-набору ({len(questions)} вопросов"
          f", refusal-вопросов исключено: {len(refusals)})\n")
    print(f"{'id':30s} {'rank':>4s}  matched")
    for r in rows:
        mark = f"{r['hit_rank']}" if r["hit_rank"] else "MISS"
        print(f"{r['id']:30s} {mark:>4s}  {r['matched']}")
    print(f"\nИТОГО: recall@{args.top_k}={total['recall@k']}, MRR={total['mrr']}")
    for cat, a in by_cat.items():
        print(f"  {cat:8s} n={a['n']:2d}  recall@{args.top_k}={a['recall@k']}  MRR={a['mrr']}")

    if args.json_out:
        report = {"golden_version": golden["_meta"]["version"],
                  "top_k": args.top_k, "line_window": args.line_window,
                  "aggregate": total, "by_category": by_cat, "rows": rows}
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nотчёт: {args.json_out}")

    if args.fail_under is not None and total["recall@k"] is not None:
        if total["recall@k"] < args.fail_under:
            print(f"FAIL: recall@{args.top_k} {total['recall@k']} < {args.fail_under}",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
