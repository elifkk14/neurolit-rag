FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/models

WORKDIR /srv

# 1. Dependencies. Copied before the code so this layer is cached
#    and only rebuilt when requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Download the embedding model (ONNX) at build time, so the container starts without
#    internet access to Hugging Face and every start is equally fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# 3. Application code.
COPY app ./app
COPY ingest ./ingest
COPY db ./db

# Do not run as root inside the container.
RUN useradd --create-home --uid 1000 neurolit && chown -R neurolit /srv /opt/models
USER neurolit

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=4)"

# Hosting platforms (Render) pass the port in $PORT; 8000 locally.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
