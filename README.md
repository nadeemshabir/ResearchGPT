# ResearchGPT

Retrieval-augmented question answering over research papers. Upload PDFs, ask
questions in natural language, get answers grounded in the source text with
citations back to the passages they came from.

Python 3.10+ · MIT licensed

---

## What it does

- **Hybrid retrieval** — dense embeddings fused with BM25, so both paraphrases
  and exact terms (model names, metrics, numbers) are matched.
- **Cross-encoder reranking** — a shortlist is rescored by a model that reads
  query and passage together.
- **Section-aware chunking** — headings are detected and no chunk crosses a
  section boundary.
- **Query routing** — comparisons and literature reviews are handled
  differently from plain questions.
- **Grounded refusal** — when nothing relevant is indexed, the system says so
  instead of answering from the model's own memory.
- **Multi-provider LLM support** — Groq, OpenAI, or Gemini.

---

## Results

Measured on [BEIR/SciFact](https://huggingface.co/datasets/BeIR/scifact) — 300
expert-labelled queries over 5,183 scientific abstracts. No hand-written or
LLM-generated labels. Full methodology in [docs/EVALUATION.md](docs/EVALUATION.md);
reproduce with `make eval`.

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|---|
| BM25 only | 0.6969 | 0.7598 | 0.5937 | 0.6301 | 98 ms |
| Dense only (MiniLM) | 0.7346 | 0.7833 | 0.6012 | 0.6422 | 40 ms |
| Hybrid weighted 0.7/0.3 *(old default)* | 0.7523 | 0.8110 | 0.6512 | 0.6873 | 179 ms |
| Hybrid RRF (k=60) | 0.7141 | 0.7924 | 0.6222 | 0.6599 | 178 ms |
| Hybrid weighted 0.7/0.3 + rerank | 0.7416 | 0.8272 | 0.6629 | 0.6948 | 4,998 ms |
| **Hybrid weighted 0.5/0.5 (tuned)** | 0.7505 | **0.8202** | **0.6632** | **0.6972** | **179 ms** |

Four things worth pulling out:

- **Hybrid retrieval earns its complexity** — Recall@5 improves 5.5 points over
  BM25 and 1.8 over dense retrieval. Fusion beats either component on every metric.
- **Tuning one float beat adding a cross-encoder.** A weight sweep found the
  inherited 0.7/0.3 split was suboptimal; 0.5/0.5 matches or beats the reranked
  pipeline on 3 of 4 metrics **at 1/28th the latency**.
- **So the reranker does *not* earn its cost here** — 28× latency for +0.75
  nDCG points, and Recall@5 actually drops 1.1. Likely domain mismatch: the
  `ms-marco` cross-encoder was trained on web search, not scientific claims.
- **RRF underperformed weighted fusion**, by 3.8 recall points — against the usual
  advice, and against my own expectation. It is the default in Elasticsearch and
  Qdrant, but SciFact rewards the confidence signal RRF discards by design.

At the tuned setting nDCG@10 reaches 0.6972, above ColBERTv2's published 0.693,
on CPU. Meanwhile the BM25 component sits 3.5 points *below* the published BEIR
baseline — a tokenisation gap that is a concrete open task.

Generation-side metrics (faithfulness, citation accuracy) are next, and will be
reported alongside a measured judge-agreement figure rather than presented as fact.

---

## Quick start

```bash
git clone https://github.com/nadeemshabir/ResearchGPT.git
cd ResearchGPT

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

make install                       # or: pip install -r requirements.txt
cp .env.example .env               # then add your API key
```

Get a free Groq key at [console.groq.com](https://console.groq.com) — no credit
card required — and put it in `.env`:

```
GROQ_API_KEY=your_key_here
```

Then:

```bash
make corpus     # optional: download 8 open-access papers into data/raw/
make run        # launch the app at http://localhost:8501
```

In the app: click **Initialize system**, upload PDFs in the **Upload** tab, then
ask questions in the **Ask** tab.

---

## Configuration

Every parameter lives in [`src/config.py`](src/config.py) and can be overridden
from `.env`. See [`.env.example`](.env.example) for the full list. The ones
worth knowing:

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Changing this after indexing requires re-ingesting; the app detects the mismatch and tells you. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Measured in tokens, not characters. |
| `SEMANTIC_WEIGHT` / `KEYWORD_WEIGHT` | `0.5` / `0.5` | Tuned by sweep on SciFact. The old 0.7/0.3 cost ~1 nDCG point. |
| `FUSION_METHOD` | `weighted` | `weighted` beat `rrf` by 3.8 Recall@5 points, even after tuning RRF's `k`. |
| `USE_RERANKING` | `false` | Off on evidence: 28× latency for +0.75 nDCG, and Recall@5 *drops*. |
| `USE_MULTI_AGENT` | `true` | ~`chunks + 3` LLM calls vs 1 for single-shot. |
| `TOP_K_RETRIEVE` / `TOP_K_RERANK` | `50` / `10` | Candidates fetched, then kept. |

---

## Project layout

```
src/
  config.py          all tunable parameters
  exceptions.py      typed error hierarchy
  utils/             logging, retry, device selection
  ingestion/         PDF parsing, chunking, embedding, vector store
  retrieval/         dense, BM25, fusion, reranking, query processing
  generation/        LLM client, prompts, agents, routing, citations
api/                 FastAPI app, schemas, background jobs
frontend/            React + Vite UI, built into static/ and served by the API
test/                297 tests, all offline
eval/                BEIR harness, RAGAS, refusal, judge calibration
scripts/             corpus download, indexing, store reset
Dockerfile           multi-stage build; indexes the corpus at build time
docs/                ARCHITECTURE.md, API.md, EVALUATION.md
plan.md              roadmap
```

---

## Development

```bash
make install-dev    # runtime + dev dependencies
make frontend-install
make serve          # build the UI, serve API + UI on :8000

make test           # 297 tests, offline, ~40s
make check          # lint + typecheck + tests
make docker         # build the image (fetches and indexes the corpus)
```

Three conventions hold throughout `src/`:

1. Library code **logs**; it never prints. Entry points (`scripts/`, `eval/`)
   print. `ruff`'s `T20` rule enforces this.
2. Constants live in `src/config.py` and nowhere else, so evaluation sweeps can
   vary them from a single place.
3. No test calls a real provider. Clients are stubbed and the retry policy's
   sleeps are patched, so the suite runs offline with no API keys.

---

## Tech stack

FastAPI · React + Vite · ChromaDB · sentence-transformers · rank-bm25 ·
PyMuPDF · tiktoken · pydantic-settings · KaTeX · Groq / OpenAI / Gemini

The UI was Streamlit until Milestone 3. It was replaced because it capped how
the interface could look and could not express the thing this project is built
around: clicking a claim to see the passage behind it.

---

## Roadmap

See [plan.md](plan.md). In short:

- [x] **M1 — Foundation.** Config layer, typed exceptions, structured logging,
      error handling at every boundary, repo hygiene.
- [x] **M2a — Retrieval evaluation.** BEIR/SciFact harness, four ablations and
      two parameter sweeps over 300 expert-labelled queries. Two shipped
      defaults changed on the evidence. See [docs/EVALUATION.md](docs/EVALUATION.md).
- [ ] **M2b — Generation evaluation.** RAGAS faithfulness, answer relevancy,
      and context precision/recall, reported alongside a measured
      judge-agreement figure. Plus citation accuracy and refusal correctness.
- [ ] **M3 — Productionisation.** Tests, FastAPI with streaming, Docker, CI,
      a deployed URL, and cost/latency tracking.
- [ ] **M4 — Differentiation.** One deeper capability, measured against the
      M2 harness.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nadeem Shabir](https://github.com/nadeemshabir).
