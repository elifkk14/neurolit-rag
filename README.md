# NeuroLit

Question answering over neuroimaging literature with retrieval-augmented generation (RAG).
Ask a question about Alzheimer's disease imaging, get an answer built only from PubMed
abstracts, with every claim linked to its source (PMID).

> 🚧 Work in progress. Day 1 of 10: data ingestion.

## Status

- [x] PubMed ingestion (E-utilities `esearch` + `efetch`)
- [ ] Embeddings + PostgreSQL/pgvector
- [ ] Retrieval + cited answers (schema-validated)
- [ ] FastAPI service + Docker
- [ ] Tests, evaluation (Recall@5, MRR), CI
- [ ] Deployment, agent with tool calling, Kubernetes manifests

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # add your NCBI contact email
python -m ingest.pubmed --max 5000
pytest
```

Data comes only from public PubMed abstracts. This tool is for literature search, not medical advice.
