# Multi-stage build for a single container serving the API and the UI.
#
# The corpus is indexed at BUILD time and the vector store ships inside the
# image. A Hugging Face Space has an ephemeral filesystem: anything written at
# runtime is lost on restart, rebuild, or wake from sleep, which would empty a
# disk-backed ChromaDB every time. Baking it in deletes the problem rather than
# solving it with a database.
#
# Build:  docker build -t researchgpt .
# Run:    docker run -p 7860:7860 -e GEMINI_API_KEY=... researchgpt

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

# Fetch and index the corpus. Both steps fail the build rather than producing an
# image that quietly serves fewer papers than the evaluation measured.
RUN python scripts/fetch_corpus.py \
    && python scripts/index_papers.py \
    && python -c "\
import sys; sys.path.insert(0, '.'); \
from src.ingestion.database import VectorDatabase; \
stats = VectorDatabase().get_stats(); \
print('indexed', stats['total_chunks'], 'chunks from', stats['total_papers'], 'papers'); \
sys.exit(0 if stats['total_papers'] == 8 else 1)" \
    # The PDFs are only needed to build the index; ~60MB of them are not.
    && rm -rf data/raw/*.pdf

COPY api/ ./api/
COPY --from=frontend /static ./static/

# Non-root. The vector store is read-only at runtime -- nothing is written
# outside /tmp -- so the app never needs to own anything it can modify.
RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 7860

# Reports `degraded` on an empty corpus rather than failing, so a healthy
# container with nothing indexed is still visible as a problem without being
# restarted for it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# One worker on purpose. Job state in `api/jobs.py` is per-process, so a second
# worker would let a client poll the one that does not hold its job. Multiple
# workers means moving that to shared storage first.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
