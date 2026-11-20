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
