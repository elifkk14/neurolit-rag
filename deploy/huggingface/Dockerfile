# Hugging Face Space: run the image that CI builds and publishes from the main branch.
FROM ghcr.io/elifkk14/neurolit-rag:latest
# Requests reach the app through the Hugging Face proxy, which sets X-Forwarded-For.
ENV TRUST_PROXY=1
