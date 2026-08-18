# Architecture

How ResearchGPT is put together, and why. Where a decision has a real
trade-off, this document states it rather than presenting the current choice as
obviously correct.

## Pipeline

```
PDF ──▶ parse ──▶ chunk ──▶ clean ──▶ embed ──▶ ChromaDB
                                                    │
query ──▶ process ──▶ ┌── dense (cosine) ───────────┤
                      └── BM25 (lexical) ───────────┤
                                                    ▼
                                              weighted fusion
                                                    ▼
                                    cross-encoder rerank   (off by default)
                                                    ▼
                                    dual-gate filter + context assembly
                                                    ▼
                                  single-shot  or  4-stage agents
                                                    ▼
                                      deterministic citations
                                                    ▼
                                        refusal check ──▶ answer
                                                    ▼
                                  FastAPI (JSON + SSE) ──▶ React UI
```

Query routing, when enabled, runs on the raw query *before* retrieval and picks
which retrieval-and-generation strategy the query takes. It is disabled in the
API — see "Routing is rule-based" below.

## Layout

| Path | Responsibility |
|---|---|
| `src/config.py` | Every tunable parameter. Nothing else hardcodes a constant. |
| `src/exceptions.py` | Typed error hierarchy used at all boundaries. |
| `src/utils/` | Logging, retry policy, torch device selection. |
| `src/ingestion/` | PDF parsing, chunking, embedding, vector storage. |
| `src/retrieval/` | Dense search, BM25, fusion, reranking, query processing. |
| `src/generation/` | LLM client, prompts, multi-agent pipeline, routing, citations. |
| `api/` | FastAPI app. Serves the JSON API and the built React bundle from one process. |
| `frontend/` | React + Vite UI, built into `static/`. The only code that formats output for humans. |
| `scripts/` | Operational CLIs (corpus download, indexing, store reset). |
| `eval/` | BEIR harness, RAGAS, refusal, judge calibration. Entry points, so allowed to print. |
| `test/` | 307 tests. No test calls a real provider. |
| `deploy/` | Cloud Run deployment script. |

Two rules keep this honest:

1. **Nothing under `src/` or `api/` writes to stdout.** Library code logs; CLIs
   print. `ruff` enforces this with the `T20` rule, with `scripts/` and `eval/`
   exempted.
2. **Nothing under `src/` reads a constant from anywhere but `src/config.py`.**
   This is what made the evaluation possible: the weight sweep varies
   `semantic_weight` from one place. Parameters scattered across constructor
   defaults cannot be swept, and two shipped defaults changed because they
   could be.

## Design decisions

### Chunking happens before cleaning

Section detection reads line breaks to find headings. Text normalisation
collapses them. Running the normaliser first therefore makes every document
look like one unstructured block, and section-aware chunking silently
degrades to flat chunking.

The pipeline chunks the raw text first, then cleans each chunk
([`pipeline.py`](../src/ingestion/pipeline.py)). This ordering is load-bearing;
reversing it produces no error, just worse retrieval.

### Chunk boundaries are measured in tokens

Character-based splitting lets a chunk overflow the embedding model's context
window, at which point the tail is silently truncated and its content becomes
unretrievable. `TextChunker` counts with `tiktoken`, so `chunk_size` means what
it says.

### Dense and BM25 are fused, not chosen between

They fail differently. Embeddings match paraphrases but blur exact tokens —
model names, metric names, numbers. BM25 nails exact tokens and misses
synonyms entirely.

Fusing them is worth it: on BEIR/SciFact, hybrid retrieval scored Recall@5 of
0.7523 against 0.6969 for BM25 alone and 0.7346 for dense alone, beating both
components on every metric.

The split is **0.5/0.5, tuned by sweeping it from 0.0 to 1.0** across 300
queries. The previous 0.7/0.3 default was inherited from a tutorial and sat on
the shoulder of the curve rather than the peak, costing about 1 nDCG point.
The sweep's endpoints double as a correctness check: at 0.0 and 1.0 the fused
pipeline reproduces standalone BM25 and standalone dense retrieval exactly.

### Two fusion strategies, neither yet chosen on evidence

Merging two rankings is not a solved problem, so both approaches are
implemented and selectable via `FUSION_METHOD` / `HybridSearcher(fusion=...)`.

**Weighted score fusion** (`"weighted"`, current default) normalises each
retriever's scores onto a common scale and takes a weighted sum. It preserves
*confidence*: a retriever that puts one chunk far ahead of the field can say so.

The price is that it needs a normaliser, and the choice matters. `minmax` maps
the worst candidate in every set to exactly `0.0`, erasing the information that
it was retrieved at all. A chunk ranked first by BM25 but last by dense
retrieval loses its entire dense contribution. `sum` normalisation avoids that;
`none` lets unbounded BM25 swamp bounded cosine and exists only for ablations.

**Reciprocal Rank Fusion** (`"rrf"`, Cormack et al. 2009) discards scores and
fuses rank positions:

```
score(d) = Σ_r  w_r / (k + rank_r(d))          k = 60
```

Being scale-free, it removes the normalisation question entirely — which is why
it is the default in Elasticsearch, OpenSearch, Weaviate, and Qdrant. Its
characteristic behaviour is that documents found by *both* retrievers reliably
outrank documents found by one, however confidently.

Its weakness is the mirror of the other's strength: RRF cannot express
confidence. Dense scores of 0.95 and 0.94 fuse identically to 0.95 and 0.42,
because rank 1 is rank 1 either way.

Implementation note: raw RRF scores top out near `1/(k+1) ≈ 0.016`, which would
silently fall below the default `min_hybrid_score` of 0.2 and discard every
result. The implementation rescales by `k + 1` so a chunk ranked first by both
retrievers scores exactly `1.0`. That is a monotonic transform, so the ordering
is identical to raw RRF; it exists purely to keep `hybrid_score` on a stable
0–1 scale across both strategies.

**This has now been measured.** On BEIR/SciFact (300 expert-labelled queries),
weighted fusion beat RRF by 3.8 Recall@5 points and 2.7 nDCG points. Sweeping
RRF's damping constant narrowed the gap but did not close it: even at its best
`k`, RRF trailed by 0.8 Recall@5 and 2.8 nDCG points.

The reading is that score magnitude carries real information on this dataset,
and any purely rank-based method discards it by construction. Weighted fusion
is therefore the default. See [EVALUATION.md](EVALUATION.md) for the full
tables.

### The reranker sees the raw query

Query processing strips stopwords and appends synonyms, which helps BM25. It
hurts the cross-encoder, which was trained on natural-language pairs. So the
retrievers get the processed query and the reranker gets the original
([`retrieval_system.py`](../src/retrieval/retrieval_system.py)).

### Cross-encoder scores are unbounded logits

Roughly −11 to +11 for the ms-marco models. They are not probabilities, and a
threshold of `0.5` would discard nearly everything. `min_rerank_score` defaults
to `−5.0`. Because the scale depends entirely on whether reranking ran,
`get_relevant_chunks` selects the threshold to match
(`min_rerank_score` vs `min_hybrid_score`).

### Empty retrieval raises instead of returning nothing

An LLM handed an empty context window does not decline — it answers from
parametric memory, producing exactly the ungrounded output this system exists
to prevent. `get_relevant_chunks` raises `NoRelevantContextError`, and
`AnswerGenerator` converts that into an explicit refusal.

### Routing is rule-based, and switched off in the API

Regex patterns and keyword counts produce a confidence score per query type;
the first to clear its threshold wins, and anything else falls through to
standard Q&A. This is instant and free where an LLM classifier would add a
round trip to every request.

Only two of the six routes change what the system does. **Comparison** searches
once per item, because a single query for "BERT vs GPT" returns chunks about
whichever name dominates the corpus and leaves the other side unsupported.
**Literature review** groups chunks by paper and spends one LLM call per paper
plus one to synthesise. The extraction, summary and definition routes call
`answer_question` unchanged — their `extract_mode` / `summary_mode` /
`definition_mode` params are set by the router and read by nothing, so those
routes only relabel the response.

**`api/dependencies.py` passes `use_smart_routing=False`.** Routing was never
part of the evaluation, so the API serves the path the numbers describe. The
config default is `true`, which is what any script constructing
`AnswerGenerator` directly will get.

What routing does today, measured on the same 43 questions as the generation
evaluation: 40 route to Q&A, 3 to comparison, none to the other four types. All
three comparisons extracted at least one wrong item — the ReLU-versus-tanh
question came out as `["ReLUs", "CIFAR-10"]`. That matters more than a
misclassification, because `compare_papers` receives only the items:
`build_comparison_prompt` renders `Compare: ReLUs vs CIFAR-10` and the user's
actual sentence never reaches the model. The failure is fluent and silent, and
refusal detection cannot catch it.

The review route is the expensive one and the easiest to trigger: one pattern
match reaches its threshold on its own, so "Review the ablation in Table 3" and
"What is the current research setup in Section 4?" both request a multi-paper
review — up to 11 LLM calls for a question about one section. Since every
family's per-pattern weight now equals its own threshold, the thresholds do no
filtering; the first matching pattern decides.

Before this can be turned on it needs the same treatment as retrieval: a
labelled set of queries, a measured misroute rate, and the original question
carried into the comparison prompt so a bad item split degrades the answer
instead of replacing the question.

### The multi-agent pipeline is optional on purpose

Four stages — analyze, synthesize, cite, critique — cost roughly
`len(chunks) + 3` LLM calls per question, against 1 for single-shot. That is
several times the latency and cost.

Whether it buys enough quality to justify that is still open. The judged
metrics that would answer it are withdrawn (see below), and the one axis that
*was* countable went against it: its citation stage reached a 0.7209 citation
rate with 3 fabrications, where doing the same job in code reached 1.0000 with
0. It stays behind a flag.

### Citations are attached in code, not written by the model

Every chunk already carries `paper_id`, `title` and `section_title`, and the
retrieval layer already maps chunks back to papers. Asking an LLM to redo that
lookup costs a call and invents formats — one run produced both
`[Attention is All you Need, Undated]` and
`[Training language models... | Section: Model, Figure 2]`.

`CitationManager` scores each paragraph against each retrieved chunk by
content-word overlap and appends the sources that clear a threshold. Because
sources come from a fixed list of retrieved chunks, a fabricated citation is
not merely unlikely but impossible. Citations the model wrote anyway are
stripped semantically — a bracketed span is a citation if it names a retrieved
source — because stripping by pattern cannot keep up with the formats models
invent.

The rate of 1.0000 is true by construction and carries no information. The
figures that do are coverage (0.9225 — the *right* sources get named) and zero
fabrications. This is also what makes click-a-claim-to-see-the-passage possible
in the UI: the paragraph-to-chunk link is real data, not parsed out of prose.

### Embedding provenance is recorded

The collection stores which model built it. Changing `EMBEDDING_MODEL` after
indexing otherwise surfaces as an opaque dimensionality error from ChromaDB at
query time. `verify_embedding_model` fails at startup instead, naming the fix.
A same-dimension model swap warns rather than fails, since it is recoverable.

### The BM25 index refreshes itself

It is an in-memory snapshot built from the vector store. Papers ingested after
construction were previously invisible to keyword search for the life of the
process — which meant every paper a user uploaded during a session.
`KeywordSearcher.search` now checks the collection count and rebuilds when it
changes.

### Only transient LLM failures are retried

Rate limits and timeouts back off exponentially with full jitter. A rejected
API key fails immediately, because retrying it four times just delays the same
error. Provider SDKs raise unrelated exception types for identical conditions,
so `LLMClient._translate_error` maps them onto one hierarchy by inspecting
message text — imprecise, but the only portable option.

### Reranking is off by default

The cross-encoder was measured and did not justify itself: 28× the latency
(179 ms → 4,998 ms p95) for +0.75 nDCG@10, while *lowering* Recall@5 by 1.1
points — the metric that most constrains RAG quality, since the generator
cannot use a passage retrieval never returned.

The likely cause is domain mismatch. `ms-marco-MiniLM-L-6-v2` was trained on
short web-search queries against web passages, not scientific claims against
biomedical abstracts.

It remains available via `USE_RERANKING=true`, and a domain-matched or
fine-tuned cross-encoder is worth testing before writing the approach off. But
it does not ship on by default, and the measurement is the reason.

## Known limitations

- **Retrieval is measured; answer *quality* still is not.** Generation was
  evaluated, but the LLM judge failed calibration against hand grades
  (κ = −0.195), so faithfulness and answer relevancy are recorded and withdrawn.
  What stands is counted in code: refusal 23/23, citation coverage 0.9225, zero
  fabricated citations.
- **Evaluated on one dataset.** SciFact is biomedical abstracts. The
  application ingests full-text papers, and results may not transfer.
- **Chunk size and embedding model are inherited, not chosen.** Chunk size
  cannot be meaningfully swept on abstracts, which fit in one chunk each.
- **BM25 trails its published baseline** by 3.5 nDCG points, traceable to
  tokenisation: no stemming, no stopword removal, default `k1`/`b`.
- **Scanned PDFs are rejected**, not OCR'd.
- **Query routing is off and unevaluated.** On the 43-question set it fires on
  3, and all 3 extract at least one wrong comparison item. See the routing
  section above.
- **Comparison item extraction relies on capitalisation**, so
  "compare attention and recurrence" falls back to Q&A — the safe direction.
- **No cost or latency tracking per query.** There is no `/metrics` endpoint;
  latency is only measured inside the evaluation harness.
- **Ingestion jobs are in-process.** They do not survive a restart, and a
  second worker cannot see the first worker's jobs.
