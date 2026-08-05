.PHONY: help install install-dev run lint format typecheck check corpus clean reset-db \
        eval eval-quick eval-sweep-weights eval-sweep-rrfk index \
        eval-testset eval-generation eval-generation-quick eval-generation-agents

help:
	@echo "install      Install runtime dependencies"
	@echo "install-dev  Install runtime + development dependencies"
	@echo "run          Launch the Streamlit app"
	@echo "corpus       Download a small open-access paper corpus into data/raw/"
	@echo "lint         Run ruff"
	@echo "format       Format with ruff"
	@echo "typecheck    Run mypy over src/"
	@echo "check        lint + typecheck"
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

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	streamlit run app.py

corpus:
	python scripts/fetch_corpus.py

lint:
	ruff check src app.py scripts eval

format:
	ruff format src app.py scripts eval
	ruff check --fix src app.py scripts eval

typecheck:
	mypy src eval

check: lint typecheck

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

index:
	python scripts/index_papers.py --skip Panipat.pdf test_paper.pdf

reset-db:
	python scripts/reset_db.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./venv/*" -exec rm -rf {} +
