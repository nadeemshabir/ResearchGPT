# Multi-stage build for a single container serving the API and the UI.
#
# The image ships NO papers. It used to: the corpus was fetched and indexed at
# build time and the vector store was baked into the image, because the
# filesystem is ephemeral and a disk-backed ChromaDB would be emptied by every
# restart. That solved the wrong problem -- it made eight fixed papers the only
# thing the deployment could ever answer from, and shipped them to every user
# of every deployment.
#
# The ephemeral filesystem is now handled where it belongs: the vector store is
# remote (CHROMA_MODE=cloud or http) and uploaded PDFs go to a bucket
# (UPLOAD_BUCKET). Both survive a restart and are shared by every instance.
#
# Build:  docker build -t researchgpt .
# Run:    docker run -p 7860:7860 \
#             -e GEMINI_API_KEY=... \
#             -e CHROMA_MODE=cloud -e CHROMA_API_KEY=... \
#             -e CHROMA_TENANT=... -e CHROMA_DATABASE=... \
#             -e UPLOAD_BUCKET=... researchgpt

# --- stage 1: frontend ------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Dependencies are copied first so a source-only change does not reinstall
# 188 packages.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# vite.config.ts writes to ../static, which is outside this WORKDIR.
RUN npm run build && test -f /static/index.html


# --- stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

# Pinned rather than :latest so a rebuild six months from now produces the same
# image. curl is for HEALTHCHECK; the rest are wheels' shared-library needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # The torch CPU wheel is ~200MB. On a flaky connection the default 15s
    # timeout and 5 retries produce a truncated download, which then fails
    # pip's hash check with a message about tampering -- alarming, and
    # nothing to do with the actual cause.
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    # Hugging Face Spaces serve on 7860.
    PORT=7860 \
    # Model cache inside the image, not $HOME, so it survives the USER switch.
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch, installed before anything that would pull the default build.
# The CUDA wheels are ~2.5GB and every byte of it is dead weight on a CPU Space.
#
# Pinned to 2.9.1 to match the development environment. 2.4.1 failed the build:
# `transformers` 4.57 imports `torch.distributed.tensor.DTensor`, which did not
# exist until torch 2.5, so the model download step died with an ImportError.
# Pinning torch without pinning what depends on it is how that happens.
RUN pip install --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model in. Downloading it at boot adds ~90s to a cold start
# and makes startup depend on huggingface.co being reachable.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')" \
    && chmod -R a+rX /opt/models

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY api/ ./api/
COPY --from=frontend /static ./static/

# Non-root. Uploads are streamed through /tmp and then handed to the bucket,
# and the vector store is remote, so nothing durable is written inside the
# container. /app/data exists for the local-fallback modes (CHROMA_MODE=local,
# no UPLOAD_BUCKET), which are for development, not for this deployment.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

EXPOSE 7860

# An empty corpus is healthy: every paper arrives by upload, so a container
# with nothing indexed is a deployment nobody has uploaded to yet. The check
# fails only when a dependency -- the vector store, the model, the bucket --
# cannot be reached.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# One worker. Ingestion now runs inside the upload request and the stores are
# shared, so a second worker would no longer lose uploads -- but `api/jobs.py`
# is still per-process, so a client polling `GET /jobs/{id}` could still reach
# the worker that does not hold its job. Scale with instances, which the shared
# stores now support, rather than with workers.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
