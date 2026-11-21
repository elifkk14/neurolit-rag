"""Ask a question from the terminal.

python -m app.ask "How does tau PET differ from amyloid PET?"
"""

from __future__ import annotations

import sys
import time

import psycopg
from pgvector.psycopg import register_vector

from app.answer import answer_question
from app.config import DATABASE_URL
from app.embeddings import embed_query
from app.llm import get_llm
from app.retrieve import retrieve


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit(__doc__)

    t_load = time.perf_counter()
    embed_query("warm-up")  # loads the embedding model once (an API server does this at startup)
    t_load = time.perf_counter() - t_load

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        t0 = time.perf_counter()
        passages = retrieve(conn, question)
        t1 = time.perf_counter()
    result = answer_question(question, passages, get_llm())
    t2 = time.perf_counter()

    print(f"\nQ: {question}\n")
    print(result.answer, "\n")
    for p in result.sources:
        print(f"  [{p.pmid}] {p.title} ({p.journal}, {p.year})\n           {p.url}")
    print(
        f"\n{result.model} · attempts {result.attempts} · model load {t_load:.1f} s (once) · "
        f"retrieval {1000 * (t1 - t0):.0f} ms · LLM {t2 - t1:.1f} s"
    )


if __name__ == "__main__":
    main()
