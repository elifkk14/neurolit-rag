"""Settings read from environment variables (.env in development, platform secrets in production)."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neurolit:neurolit@localhost:5433/neurolit")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
# The model runs with ONNX Runtime (fastembed) instead of PyTorch: same vectors (cosine 1.0000
# on our data), but ~10x less memory, so it fits a free 512 MB server.
# 1 thread: free servers get a fraction of a CPU, and extra threads only compete for it.
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "1"))
EMBED_CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH") or None  # None -> fastembed default

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

# Rate limits for the public demo. The free Groq tier allows ~8k tokens/min, about 3 questions/min,
# so without a limit one visitor could use up the quota for everyone. 0 disables a limit.
RATE_LIMIT_PER_CLIENT = int(os.getenv("RATE_LIMIT_PER_CLIENT", "5"))  # questions per client ...
RATE_LIMIT_CLIENT_WINDOW_S = int(os.getenv("RATE_LIMIT_CLIENT_WINDOW_S", "600"))  # ... per 10 minutes
RATE_LIMIT_GLOBAL = int(os.getenv("RATE_LIMIT_GLOBAL", "3"))  # questions for all clients ...
RATE_LIMIT_GLOBAL_WINDOW_S = int(os.getenv("RATE_LIMIT_GLOBAL_WINDOW_S", "60"))  # ... per minute
# Behind a hosting proxy every request comes from the proxy's address; the visitor's address is
# in X-Forwarded-For. Only trust that header when a proxy you control sets it.
TRUST_PROXY = os.getenv("TRUST_PROXY", "0") == "1"
