"""Settings read from environment variables (.env in development, platform secrets in production)."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neurolit:neurolit@localhost:5433/neurolit")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
# CPU by default: free hosting has no GPU, and the small model is fast enough on CPU.
# (On macOS 14 the "mps" backend of recent torch versions also crashes.)
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# LLM: "groq" (hosted, free tier) or "ollama" (local). Switching provider = changing one variable.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty -> provider default (see app/llm.py)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# "vector", "keyword" or "hybrid" (both fused with Reciprocal Rank Fusion). Chosen from eval/run.py results.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "vector")
# Keyword search ignores words that occur in more than this share of chunks (see db/schema.sql).
KEYWORD_MAX_DF = float(os.getenv("KEYWORD_MAX_DF", "0.5"))
