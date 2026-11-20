# NeuroLit

Question answering over neuroimaging literature with retrieval-augmented generation (RAG).
Ask a question about Alzheimer's disease imaging, get an answer built only from PubMed
abstracts, with every claim linked to its source (PMID).

> 🚧 Work in progress. Day 2 of 10: embeddings and vector search.

## Status

- [x] PubMed ingestion (E-utilities `esearch` + `efetch`)
- [x] Embeddings (`bge-small-en-v1.5`) + PostgreSQL/pgvector with an HNSW index
- [ ] Retrieval + cited answers (schema-validated)
- [ ] FastAPI service + Docker
- [ ] Tests, evaluation (Recall@5, MRR), CI
- [ ] Deployment, agent with tool calling, Kubernetes manifests

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # add your NCBI contact email
docker compose up -d db       # PostgreSQL 17 + pgvector on localhost:5433
python -m ingest.pubmed       # 5,000 abstracts, equal share per year 2015-2026
python -m ingest.load data/raw/pubmed_YYYYMMDD.jsonl   # chunk, embed, store (~4 min on CPU)
pytest
```

## Data

| | |
|---|---|
| Source | PubMed, MeSH `Alzheimer Disease` AND (`Positron-Emission Tomography` OR `Magnetic Resonance Imaging`), English, with abstract |
| Sample | 5,000 abstracts, 2015-2026, stratified by publication year |
| Chunks | 6,007 (80% of abstracts fit in one chunk of ≤300 words; long ones are split on sentence boundaries with one-sentence overlap) |

Why stratified: PubMed returns newest records first, so the first 5,000 hits were almost all
from 2024-2026 and the field's earlier key papers were missing.

Data comes only from public PubMed abstracts. This tool is for literature search, not medical advice.
