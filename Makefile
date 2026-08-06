.PHONY: help install install-dev api api-dev serve frontend frontend-dev frontend-install \
        docker docker-run \
        lint format typecheck check corpus clean reset-db \
        eval eval-quick eval-sweep-weights eval-sweep-rrfk index \
        eval-testset eval-generation eval-generation-quick eval-generation-agents \
        eval-refusal eval-refusal-sweep eval-calibrate \
        test test-cov

help:
	@echo "install      Install runtime dependencies"
	@echo "install-dev  Install runtime + development dependencies"
	@echo "api          Serve the HTTP API (docs at /docs)"
	@echo "api-dev      Serve the API with autoreload"
	@echo "serve        Build the frontend, then serve API + UI on :8000"
	@echo "frontend     Build the React bundle into static/"
	@echo "frontend-dev Vite dev server on :5173, proxying to the API"
	@echo "corpus       Download the 8-paper demo corpus into data/raw/"
	@echo "docker       Build the image (fetches + indexes the corpus)"
	@echo "docker-run   Run the image on :7860"
	@echo "test         Run the test suite (offline, no API calls)"
	@echo "test-cov     Run tests with a coverage report"
	@echo "lint         Run ruff"
	@echo "format       Format with ruff"
	@echo "typecheck    Run mypy over src/"
	@echo "check        lint + typecheck + tests"
	@echo "reset-db     Delete the vector store (re-ingest afterwards)"
	@echo "clean        Remove caches and build artifacts"
	@echo ""
	@echo "eval               Full retrieval ablation on BEIR/SciFact (300 queries)"
	@echo "eval-quick         Same, 50 queries, no reranking - for iteration"
	@echo "eval-sweep-weights Semantic/keyword weight sweep"
	@echo "eval-sweep-rrfk    RRF damping constant sweep"
	@echo ""
	@echo "index                   Index PDFs from data/raw into the vector store"
	@echo "eval-testset            Build the generation question set (one time)"
	@echo "eval-generation         Score answers with RAGAS (single-shot)"
	@echo "eval-generation-quick   Same, first 10 questions only"
	@echo "eval-generation-agents  Score the 4-stage multi-agent pipeline"
	@echo "eval-refusal            Does it decline questions the corpus cannot answer?"
	@echo "eval-refusal-sweep      Sweep min_semantic_similarity"
	@echo "eval-calibrate          Grade answers by hand; report judge agreement"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

# Models load during startup, not on the first request, so the first query is
# not several seconds slower than the rest.
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

api-dev:
	uvicorn api.main:app --reload --port 8000

# --- Frontend --------------------------------------------------------------
# The build lands in static/, which the API mounts. One process serves both, so
# there is no CORS configuration anywhere.
frontend-install:
	cd frontend && npm install

frontend:
	cd frontend && npm run build

# Vite on :5173 proxying /api to :8000. Only for development -- in production
# the frontend is same-origin and there is no proxy.
frontend-dev:
	cd frontend && npm run dev

# Build the UI, then serve everything from one process at :8000.
serve: frontend
	uvicorn api.main:app --host 0.0.0.0 --port 8000

corpus:
	python scripts/fetch_corpus.py

# --- Docker ----------------------------------------------------------------
# The build fetches and indexes the corpus, so the vector store ships inside
# the image. Expect ~10 minutes cold: torch, the embedding model, 28MB of PDFs,
# and the embedding pass over every chunk.
docker:
	docker build -t researchgpt .

docker-run:
	docker run --rm -p 7860:7860 --env-file .env researchgpt

lint:
	ruff check src scripts eval test api

format:
	ruff format src scripts eval test api
	ruff check --fix src scripts eval test api

typecheck:
	mypy src eval api

# No test touches a real API: LLM clients are stubbed and the retry policy's
# sleeps are patched, so this stays fast and runs offline.
test:
	pytest test

test-cov:
	pytest test --cov --cov-report=term-missing

check: lint typecheck test

# Reranking dominates runtime (cross-encoder on CPU), so the quick target
# omits it. Quote full-run numbers, not quick-run ones.
eval:
	python -m eval.retrieval_eval

eval-quick:
	python -m eval.retrieval_eval --limit 50 --configs bm25,dense,hybrid_minmax,hybrid_sum,rrf

eval-sweep-weights:
	python -m eval.sweep weights

eval-sweep-rrfk:
	python -m eval.sweep rrfk

# --- Generation-side evaluation (Milestone 2b) ----------------------------
# Builds the question set once, then scores answers with RAGAS. The judge
# defaults to OpenAI and the generator to Settings.llm_provider: different
# families on purpose, so the judge never marks its own work. Override with
# --generator-provider / --judge-provider.
eval-testset:
	python -m eval.testset --n 48

eval-generation:
	python -m eval.generation_eval

eval-generation-quick:
	python -m eval.generation_eval --limit 10

# Compares the 4-stage agent pipeline against single-shot generation.
eval-generation-agents:
	python -m eval.generation_eval --multi-agent

# --- Refusal and judge calibration ----------------------------------------
# Refusal is measured through both gates: retrieval raises, and the model
# declines in prose. Measuring only the first scores correct refusals as
# fabrications -- see finding #14.
eval-refusal:
	python -m eval.refusal

eval-refusal-sweep:
	python -m eval.refusal --sweep

# Grade answers by hand, then report Cohen's kappa against the LLM judge.
# Every RAGAS number is meaningless without this.
eval-calibrate:
	python -m eval.calibrate

index:
	python scripts/index_papers.py --skip Panipat.pdf test_paper.pdf

reset-db:
	python scripts/reset_db.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./venv/*" -exec rm -rf {} +
