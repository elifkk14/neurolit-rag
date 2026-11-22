"""Measure retrieval quality on the hand-written question set.

Each question in eval/questions.jsonl was written from one specific abstract (known-item
evaluation): the question is answered by that article, so a good retriever ranks it high.

  python -m eval.run                 # retrieval only: Recall@1/5/10, MRR@10, latency
  python -m eval.run --answers       # also run the LLM on the top-5 of the default mode

Limitation: other abstracts may answer a question too, but only the source article counts
as relevant, so the numbers are a lower bound on real quality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL, RETRIEVAL_MODE
from app.embeddings import embed_query
from app.retrieve import MODES, retrieve

HERE = Path(__file__).parent
DEPTH = 10  # rank positions we look at


def load_questions(path: Path = HERE / "questions.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_relevant_rank(ranked_pmids: list[str], relevant: set[str]) -> int | None:
    """1-based position of the first relevant article, or None if not in the list."""
    return next((i for i, pmid in enumerate(ranked_pmids, start=1) if pmid in relevant), None)


def metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "recall@10": sum(r is not None and r <= 10 for r in ranks) / n,
        # MRR: average of 1/rank (0 if not found). Rewards putting the right article first.
        "mrr@10": sum(1 / r for r in ranks if r is not None) / n,
    }


def evaluate_retrieval(conn, questions):
    # Embed each question once and reuse it in every mode: the comparison is then about
    # the search method only, and embedding time is measured separately.
    t = time.perf_counter()
    vectors = [embed_query(q["question"]) for q in questions]
    embed_ms = 1000 * (time.perf_counter() - t) / len(questions)

    results, per_question = {}, {}
    for mode in MODES:
        ranks, latencies = [], []
        for q, vec in zip(questions, vectors, strict=True):
            t = time.perf_counter()
            passages = retrieve(conn, q["question"], k=DEPTH, mode=mode, query_vector=vec)
            latencies.append(1000 * (time.perf_counter() - t))
            rank = first_relevant_rank([p.pmid for p in passages], set(q["relevant"]))
            ranks.append(rank)
            per_question.setdefault(q["id"], {})[mode] = rank
        results[mode] = {**metrics(ranks), "median_ms": statistics.median(latencies)}
    return results, per_question, embed_ms


def evaluate_answers(conn, questions, mode):
    from app.answer import AnswerError, answer_question
    from app.llm import get_llm

    llm = get_llm()
    rows = []
    for q in questions:
        passages = retrieve(conn, q["question"], mode=mode)
        retrieved = {p.pmid for p in passages}
        t = time.perf_counter()
        try:
            a = answer_question(q["question"], passages, llm)
            cited = {p.pmid for p in a.sources}
            rows.append(
                {
                    "id": q["id"],
                    "status": "answered" if a.found else "not_found",
                    "attempts": a.attempts,
                    "relevant_retrieved": bool(retrieved & set(q["relevant"])),
                    "relevant_cited": bool(cited & set(q["relevant"])),
                    "seconds": time.perf_counter() - t,
                }
            )
        except AnswerError:
            rows.append(
                {"id": q["id"], "status": "refused", "attempts": 2, "seconds": time.perf_counter() - t}
            )
        time.sleep(20)  # free tier: 8k tokens/min ~ 3 questions/min
    n = len(rows)
    answered = [r for r in rows if r["status"] == "answered"]
    with_relevant = [r for r in rows if r.get("relevant_retrieved")]
    return {
        "model": llm.name,
        "mode": mode,
        "answered": len(answered) / n,
        "not_found": sum(r["status"] == "not_found" for r in rows) / n,
        "refused": sum(r["status"] == "refused" for r in rows) / n,
        "first_attempt_valid": sum(r["attempts"] == 1 and r["status"] != "refused" for r in rows) / n,
        # when the source article was in the top 5, did the answer actually use it?
        "cites_source_when_retrieved": (
            sum(r["relevant_cited"] for r in with_relevant) / len(with_relevant) if with_relevant else None
        ),
        "median_s": statistics.median(r["seconds"] for r in rows),
        "rows": rows,
    }


def markdown(results, embed_ms, n):
    lines = [
        f"Known-item retrieval, {n} questions, {date.today()}",
        "",
        "| Mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Search latency (median) |",
        "|---|---|---|---|---|---|",
    ]
    for mode, m in results.items():
        lines.append(
            f"| {mode} | {m['recall@1']:.2f} | {m['recall@5']:.2f} | {m['recall@10']:.2f} "
            f"| {m['mrr@10']:.3f} | {m['median_ms']:.0f} ms |"
        )
    lines.append(f"\nQuestion embedding: {embed_ms:.0f} ms per question (CPU), added to vector and hybrid.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--answers", action="store_true", help="also evaluate LLM answers (uses the API)")
    args = parser.parse_args()

    questions = load_questions()
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        results, per_question, embed_ms = evaluate_retrieval(conn, questions)
        report = {
            "date": str(date.today()),
            "n": len(questions),
            "retrieval": results,
            "per_question": per_question,
        }
        print(markdown(results, embed_ms, len(questions)))

        if args.answers:
            ans = evaluate_answers(conn, questions, RETRIEVAL_MODE)
            report["answers"] = ans
            print(
                f"\nAnswers ({ans['model']}, {ans['mode']} top-5): answered {ans['answered']:.0%} · "
                f"not found {ans['not_found']:.0%} · refused {ans['refused']:.0%} · "
                f"valid on 1st attempt {ans['first_attempt_valid']:.0%} · "
                f"cites source article when retrieved {ans['cites_source_when_retrieved']:.0%} · "
                f"median {ans['median_s']:.1f} s"
            )

    (HERE / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
