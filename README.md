# NeuroLit

Question answering over neuroimaging literature with retrieval-augmented generation (RAG).
Ask a question about Alzheimer's disease imaging, get an answer built only from PubMed
abstracts, with every claim linked to its source (PMID).

> 🚧 Work in progress. Day 5 of 10: tests.

## Status

- [x] PubMed ingestion (E-utilities `esearch` + `efetch`)
- [x] Embeddings (`bge-small-en-v1.5`) + PostgreSQL/pgvector with an HNSW index
- [x] Retrieval (one best chunk per article) + cited answers, schema-validated, invented citations rejected
- [x] FastAPI service (`POST /ask`, `GET /health`, Swagger at `/docs`), web page, Docker Compose
- [x] Tests: 44 (unit + PostgreSQL integration), 83% coverage
- [ ] Evaluation (Recall@5, MRR), CI
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

## Data

| | |
|---|---|
| Source | PubMed, MeSH `Alzheimer Disease` AND (`Positron-Emission Tomography` OR `Magnetic Resonance Imaging`), English, with abstract |
| Sample | 5,000 abstracts, 2015-2026, stratified by publication year |
| Chunks | 6,007 (80% of abstracts fit in one chunk of ≤300 words; long ones are split on sentence boundaries with one-sentence overlap) |

Why stratified: PubMed returns newest records first, so the first 5,000 hits were almost all
from 2024-2026 and the field's earlier key papers were missing.

Data comes only from public PubMed abstracts. This tool is for literature search, not medical advice.
