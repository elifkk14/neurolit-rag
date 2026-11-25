# NeuroLit

[![CI](https://github.com/elifkk14/neurolit-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/elifkk14/neurolit-rag/actions/workflows/ci.yml)

Question answering over neuroimaging literature with retrieval-augmented generation (RAG).
Ask a question about Alzheimer's disease imaging, get an answer built only from PubMed
abstracts, with every claim linked to its source (PMID).

> 🚧 Work in progress: deployment is next.

## Status

- [x] PubMed ingestion (E-utilities `esearch` + `efetch`)
- [x] Embeddings (`bge-small-en-v1.5`) + PostgreSQL/pgvector with an HNSW index
- [x] Retrieval (one best chunk per article) + cited answers, schema-validated, invented citations rejected
- [x] FastAPI service (`POST /ask`, `GET /health`, Swagger at `/docs`), web page, Docker Compose
- [x] Tests: 54 (unit + PostgreSQL integration), 83% coverage
- [x] Evaluation: 30 known-item questions, vector vs keyword vs hybrid (Recall@k, MRR)
- [x] CI: GitHub Actions runs lint and all tests (with a pgvector service container), then builds the Docker image and publishes it to GHCR
- [ ] Deployment, agent with tool calling, Kubernetes manifests

## Quick start

Requires Docker and a free [Groq](https://console.groq.com) API key.

```bash
cp .env.example .env          # put GROQ_API_KEY in .env
docker compose up -d          # PostgreSQL + pgvector and the API
```

Load the data once (from the host, into the database on port 5433):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m ingest.pubmed       # 5,000 abstracts, equal share per year 2015-2026
python -m ingest.load data/raw/pubmed_YYYYMMDD.jsonl   # chunk, embed, store (~4 min on CPU)
```

Then open http://localhost:8000 (API docs: http://localhost:8000/docs).

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "Can FDG-PET distinguish Alzheimer disease from frontotemporal dementia?"}'
python -m eval.run            # retrieval metrics (add --answers to also score LLM answers)
pytest                        # unit tests use fakes; integration tests need the compose db (else skipped)
```

## Architecture

```
browser ──> FastAPI /ask ──> embed question (bge-small, CPU)
                         ──> PostgreSQL + pgvector: HNSW top chunks, best chunk per article
                         ──> LLM (Groq or Ollama, behind one interface) with the 5 abstracts
                         ──> checks: JSON schema · every sentence cited · cited PMIDs were retrieved
                         <── answer + sources (PubMed links)
```

## How an answer is checked

1. The model must return JSON matching a Pydantic schema (`found`, `answer`, `citations`).
2. Every sentence must carry a `[PMID]` citation.
3. Every cited PMID must be one of the abstracts that were retrieved for this question.

If a check fails, the model gets the reason and one more try; otherwise the answer is refused.
Questions outside the corpus get "I could not find an answer", not a guess.

## Evaluation

**Method.** 30 abstracts were drawn at random from the corpus (`random.seed(42)`). For each one,
a question was written that the abstract answers, deliberately paraphrased instead of reusing
title words (`eval/questions.jsonl`). A retriever is scored on where it ranks that source
article (known-item evaluation).

**Retrieval** (`python -m eval.run`):

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Search latency (median) |
|---|---|---|---|---|---|
| **vector** (default) | **0.73** | **0.97** | **1.00** | **0.831** | 3 ms |
| keyword (IDF-weighted) | 0.50 | 0.73 | 0.80 | 0.588 | 39 ms |
| hybrid (RRF) | 0.70 | 0.93 | 0.93 | 0.800 | 43 ms |

Question embedding adds ~240 ms on CPU to vector and hybrid.

**What happened along the way.** The plan assumed hybrid search would win. The first
measurement said otherwise: keyword search reached only 0.37 Recall@5 and pulled hybrid down to
0.80. The cause was PostgreSQL's `ts_rank`, which has no IDF. In a corpus where every abstract
is about Alzheimer's, "disease" occurs in 91% of chunks and "alzheim" in 90%, yet they counted
as much as a decisive word like "gadolinium" (10 chunks). Replacing `ts_rank` with an
IDF-weighted score (the core of BM25) and dropping words found in over half of all chunks raised
keyword Recall@5 from 0.37 to 0.73 and hybrid from 0.80 to 0.93.

**Decision.** Vector search stays the default: it is the most accurate here, the simplest and
over 10× faster. The 0.97 vs 0.93 gap is one question out of 30, so it is not evidence that
hybrid is worse in general. Iteration stopped there on purpose, to avoid tuning to the same
30 questions.

**Limitations.** One relevant article per question (other abstracts may also answer it, so
scores are a lower bound). Questions were paraphrased, which disadvantages keyword search;
queries with exact rare terms (gene or tracer names) would favour it. The questions were written
by the developer of the system; a held-out set written by someone else is the next step.

**Answers** (`python -m eval.run --answers`, Groq `gpt-oss-120b`, vector top-5, same 30 questions):

| | |
|---|---|
| Answered with citations | 28 / 30 (93%) |
| "Not found in the abstracts" | 1 / 30 |
| Refused (no properly cited answer after a retry) | 1 / 30 |
| Passed all checks on the first attempt | 27 / 30 (90%) |
| Cited the source article when it was retrieved | 27 / 28 (96%) |
| Median answer time | 1.5 s |

No answer reached the user with an uncited sentence or a PMID that was not retrieved: those
are blocked by the checks, which is why "refused" exists. Free-tier limit: 8,000 tokens/minute
(about 3 questions/minute); the Groq client honours `Retry-After` on HTTP 429.

## Data

| | |
|---|---|
| Source | PubMed, MeSH `Alzheimer Disease` AND (`Positron-Emission Tomography` OR `Magnetic Resonance Imaging`), English, with abstract |
| Sample | 5,000 abstracts, 2015-2026, stratified by publication year |
| Chunks | 6,007 (80% of abstracts fit in one chunk of ≤300 words; long ones are split on sentence boundaries with one-sentence overlap) |

Why stratified: PubMed returns newest records first, so the first 5,000 hits were almost all
from 2024-2026 and the field's earlier key papers were missing.

Data comes only from public PubMed abstracts. This tool is for literature search, not medical advice.
