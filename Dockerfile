FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /srv

# 1. CPU-only torch first: the default wheel bundles CUDA and makes the image ~4 GB larger.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 2. The rest of the dependencies. Copied before the code so this layer is cached
#    and only rebuilt when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. Download the embedding model at build time, so the container starts without internet
#    access to Hugging Face and every start is equally fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 4. Application code.
COPY app ./app
COPY ingest ./ingest
COPY db ./db

# Do not run as root inside the container.
RUN useradd --create-home --uid 1000 neurolit && chown -R neurolit /srv /opt/hf
USER neurolit

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
