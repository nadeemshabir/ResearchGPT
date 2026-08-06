# ResearchGPT — Plan to Make This a Resume-Worthy Project

## Honest assessment of where you are

First, a correction: **this project is not "very basic."** It's 6,061 lines across 18 modules, and it already has things most portfolio RAG projects lack — BM25 + dense hybrid search, cross-encoder reranking, a query router, and a multi-agent generation pipeline. The feature surface is fine.

The problem is that it currently reads as **a tutorial you followed**, not **software you engineered**. Here is the specific evidence a reviewer will see in the first 90 seconds:

| Signal | Reality | What a reviewer concludes |
|---|---|---|
| `test/test_retrieval.py` and `src/system/tests/test_generation.py` are **0 bytes** | Zero tests exist | "Never verified anything works" |
| **611 `print()` calls, 0 uses of `logging`** | No observability | "This is a script, not a system" |
| `if __name__ == "__main__"` demo blocks in **all 18 modules** | Tutorial scaffolding left in | "Copied from a walkthrough" |
| **14 `except` blocks in 6,000 lines** | Almost no error handling | "Breaks on the first bad PDF" |
| **No numbers anywhere** | No evaluation of retrieval or answers | "Can't tell if the hybrid search helps at all" |
| `docs/` has `IMPORT_FIX_APPLIED.md`, `INTEGRATION_CHECKLIST.md`, `INTEGRATION_SUMMARY.md` | AI-generated progress notes committed as docs | "Vibe-coded with an LLM" |
| 8 commits, one titled `a` | No git discipline | "Doesn't work like an engineer" |
| Streamlit only, no API, no Docker, no CI | Not deployable by anyone else | "Runs on his laptop, maybe" |

**The single biggest gap:** every RAG project on every resume in 2026 says "built a RAG chatbot with citations." Yours says the same. What almost none of them have is **measurement** — proof that the hybrid search and reranking you built actually improve retrieval, expressed as a number. That is the thing that turns a bullet point from a claim into evidence.

**The goal of this plan:** by the end, your resume bullet is not *"Built a RAG system for research papers"* but *"Built an evaluation-driven RAG system; hybrid retrieval + cross-encoder reranking improved Recall@5 from 0.61 → 0.87 on a 120-question benchmark, with p95 latency of 2.3s and $0.004/query."*

---

## The four milestones

| Milestone | Theme | Time | Why it matters |
|---|---|---|---|
| **M1** | Clean the foundation | ~1 week | Removes every "tutorial project" tell |
| **M2** | Measure everything | ~2 weeks | **This is the differentiator.** Turns claims into numbers |
| **M3** | Make it real software | ~1.5 weeks | API, Docker, CI, deployed URL |
| **M4** | Add one hard technical angle | ~1.5 weeks | Gives you something to actually talk about in an interview |

Total: roughly **6 weeks part-time**. Do them in order. M2 is the one that matters most; do not skip ahead to M4 because it sounds more fun.

---

# Milestone 1 — Clean the foundation (~1 week) ✅ DONE

Nothing here adds features. All of it removes reasons to dismiss the project.

> **Completed.** `ruff` and `mypy` both pass clean over 26 source files. `src/`
> contains zero `print()` calls, zero `__main__` demo blocks, zero
> `sys.path` hacks, and zero `os.system("pip install ...")` calls. See
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design decisions and the
> honest list of what remains unmeasured.
>
> Four bugs surfaced during the cleanup and were fixed: section-aware chunking
> could never fire (text was normalised before chunking, destroying the line
> breaks headings are detected from); the BM25 index was built once at startup
> so every newly uploaded paper was invisible to keyword search; the UI's
> Top-K slider was accepted and silently discarded; and `adaptive_search`
> mutated shared instance weights non-atomically.

### 1.1 — Delete the AI-generated docs sprawl

These files actively hurt you. They are artifacts of your build process, not documentation of your system, and their filenames announce it.

**Delete:**
- `docs/IMPORT_FIX_APPLIED.md`
- `docs/INTEGRATION_CHECKLIST.md`
- `docs/INTEGRATION_SUMMARY.md`
- `docs/LLM_CONTEXT_PROMPT.md`
- `docs/QUICK_REFERENCE.md`
- `docs/VISUAL_OVERVIEW.md`
- `docs/week1_ingestion_notes.md`
- `docs/README_CACHE_SETUP.md`
- `chat_history_20251226_000234.txt` and `chat_history_20251226_000700.txt` (repo root)

**Keep and merge into three files:**
- `README.md` — what it is, the results table, quickstart, architecture
- `docs/ARCHITECTURE.md` — the design decisions and trade-offs (merge `PROJECT_OVERVIEW.md` + `SMART_ROUTING_GUIDE.md`)
- `docs/EVALUATION.md` — created in M2

Also strip the emoji density in the README down to near zero. Emoji-per-heading reads as LLM output.

### 1.2 — Rip out the `__main__` demo blocks

All 18 modules in `src/` have a demo block at the bottom. That is walkthrough scaffolding. Delete every one. If a module genuinely needs a CLI entry point, it belongs in a `scripts/` directory or behind a proper CLI (see 1.5), not at the bottom of a library module.

Also delete `src/generation/smart_answer_example.py` — example files don't ship.

### 1.3 — Replace 611 `print()` calls with structured logging

Create `src/utils/logging.py` with a configured logger. Then replace every `print()` in `src/` with `logger.debug/info/warning/error`. The startup banner in [answer_generator.py:41-86](src/generation/answer_generator.py#L41-L86) — the `🚀 INITIALIZING RESEARCHGPT COMPLETE SYSTEM` block with numbered emoji steps — is the clearest example of what has to go. Library code does not print banners.

Rule of thumb: **anything in `src/` that writes to stdout is a bug.** Presentation belongs in `app.py` and the API layer.

### 1.4 — Add a real config layer

Right now weights and model names are scattered as defaults across constructors (semantic 0.7 / keyword 0.3 in the hybrid search, model names in `llm_client.py`, chunk sizes in `chunker.py`). Create `src/config.py` using `pydantic-settings`:

```python
class Settings(BaseSettings):
    groq_api_key: SecretStr
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    model_config = SettingsConfigDict(env_file=".env")
```

This matters more than it looks: **M2 requires sweeping these parameters**, and you cannot sweep values that are hardcoded in constructor defaults.

### 1.5 — Error handling and input validation

You have 14 `except` blocks in 6,000 lines. Add real handling at the boundaries:
- **PDF parsing** — encrypted PDFs, scanned/image-only PDFs with no text layer, corrupt files, 0-byte uploads. Right now `pdf_parser.py` will throw an unhandled exception and take down the Streamlit session.
- **LLM calls** — rate limits (Groq's free tier will hit these), timeouts, malformed responses. Add retry with exponential backoff via `tenacity`.
- **Empty retrieval** — what happens when no chunks pass the relevance threshold? Right now you'll send an empty context to the LLM and it will hallucinate. It must refuse instead.

Define typed exceptions in `src/exceptions.py` (`PDFParseError`, `LLMProviderError`, `NoRelevantContextError`) rather than letting raw library exceptions surface.

### 1.6 — Fix the repo hygiene

- Add `.env.example` with the keys listed and no values.
- Add the `LICENSE` file — your README links to it and **it does not exist**.
- Remove `data/raw/test_paper.pdf` and `Panipat.pdf` from the repo. Ship a script that downloads a few open-access arXiv papers instead. (`Panipat.pdf` also isn't a research paper, and a reviewer browsing your repo will notice.)
- Narrow `.gitignore` — `*.bin` and `*.sqlite3` as global patterns are too broad.
- Drop `run_app.bat` / `run_app.sh` once you have a Makefile and Docker.

### 1.7 — Fix your git history going forward

You have 8 commits and one is named `a`. You can't rewrite what's pushed without pain, but from here on: small, scoped commits with conventional messages (`feat:`, `fix:`, `refactor:`, `test:`). By the end of this plan you'll have 60+ good commits and the early ones stop mattering.

**M1 checkpoint:** the repo contains no print statements in `src/`, no demo blocks, no AI-progress-note docs, and a config module. Nothing works better — it just no longer looks like a tutorial.

---

# Milestone 2 — Measure everything (~2 weeks) ⭐ DONE

**This is the milestone that makes the project resume-worthy.** Everything else is table stakes; this is the part that 95% of portfolio RAG projects do not have. Give it the most time.

The premise: you built hybrid search and cross-encoder reranking. **You currently have no idea whether either of them helps.** Find out, and publish the numbers.

## Outcome

19 findings, all in [`docs/EVALUATION.md`](docs/EVALUATION.md). Three defaults changed on measured evidence, and one metric family was withdrawn after failing calibration.

| Area | Result | Trust |
|---|---|---|
| Retrieval | BEIR/SciFact, 300 queries, expert labels | **High** |
| Weights 0.7/0.3 → 0.5/0.5 | +0.010 nDCG@10 | High |
| Reranking on → off | 28× latency for +0.0075 nDCG, *lower* Recall@5 | High |
| Citations | rate 1.0000, coverage 0.9225, **0** fabrications | High — counted in code |
| Refusal | 23/23 unanswerable refused, 1/15 false refusal | Moderate — n=23, partly self-graded |
| RAGAS scores | **Withdrawn.** κ = −0.195; 13 of 23 answers scored exactly 1.0 | None |

**Eight of the nineteen findings were bugs in the evaluation code, not the system**, and twice a failure was being scored as a success. That ratio is the most honest thing in the document and the best interview material in the project.

## What is deliberately left open

- **The judge needs ~30 more grades**, weighted toward answers it scored *low*. The first sample was spread evenly across the range — correct for an unbiased look, but it left only 3 usable negatives.
- **The test set contains questions built from example prompts inside papers** (a `list C` code sample from InstructGPT; a Joaquin Phoenix red-carpet question from GPT-3's generated-news figures). Those are not questions about the papers. Filtering them is a small change to `eval/testset.py` and would improve both evaluation and calibration.
- **The multi-agent pipeline is unjustified, not disproven.** See below.

## Decision needed: is the 4-call multi-agent pipeline worth keeping?

| | multi-agent | single-shot + deterministic citations |
|---|---|---|
| LLM calls per answer | **4** | **1** |
| citation rate | 0.7209 | **1.0000** |
| fabricated citations | 3 | **0** |
| faithfulness | 0.8681 | 0.8581 |
| answer_relevancy | 0.7530 | 0.7999 |

**The bottom two rows cannot be used.** They come from the judge that failed calibration.

What *can* be said, from numbers that do not depend on a judge:

1. It costs **4× the API calls and latency**. Measured, certain.
2. Its `CitationAgent` is now **redundant** — code-based citing beats it on both citation metrics, and those are counted, not judged.
3. Its quality benefit rests entirely on withdrawn metrics.

So the argument is "the cost is measured and the benefit is not", which is weaker than the reranking case (where the cost *and* the loss were both measured). Recommendation: **default `USE_MULTI_AGENT=false`, keep the flag**, and say plainly in the README that the four-stage path was not shown to be worth 4× the cost. Re-open it if the judge is ever calibrated well enough to detect a difference.

Knowing when *not* to use the complexity you built is a stronger signal than the complexity.

### 2.1 — Get labelled data without labelling it yourself

An earlier draft of this plan said to hand-write and verify 100–150 questions. **That was the wrong call.** The standard practice is to use an existing human-annotated IR benchmark for retrieval, and synthetic generation for the parts no benchmark covers. Both are below.

#### (a) Retrieval: use BEIR, don't build a dataset

[BEIR](https://github.com/beir-cellar/beir) is the standard IR benchmark suite. Every dataset ships a corpus, queries, and **qrels** — a TSV of `query-id, corpus-id, score` containing *human* relevance judgments. Two of its datasets are a near-exact fit for a research-paper RAG system:

| Dataset | Content | Scale |
|---|---|---|
| **SciFact** | Expert-written scientific claims paired with biomedical abstracts, annotated with supporting/refuting evidence and sentence-level rationales | ~1.4k claims, 5.2k abstracts |
| **NFCorpus** | Natural-language medical queries against PubMed documents | 3.2k queries, 9k docs, ~170k judgments |

**Start with SciFact.** It's scientific-abstract retrieval, which is what your system does, and it's small enough to iterate on quickly.

This is strictly better than a hand-built set on three counts:

1. **Zero labelling effort**, and thousands of judgments instead of your sixty.
2. **The labels are expert-written**, not LLM-generated and not yours at 1am.
3. **Your numbers become comparable.** There are published BEIR baselines for BM25, dense retrievers, and cross-encoder rerankers. "My hybrid pipeline scores nDCG@10 of X on SciFact, against BM25's published Y" is a far stronger claim than any number on a private dataset, because a reader can check it.

Ingest the SciFact corpus through your existing pipeline, run its queries, and score against the qrels. No annotation, no chunk-id brittleness — a document either is or isn't judged relevant.

#### (b) Generation: RAGAS synthetic test sets on your own corpus

BEIR scores retrieval, not answers. For generation, use [RAGAS](https://docs.ragas.io) — it is purpose-built for RAG and is the 2026 default for exactly your situation, because it removes the expert-annotation bottleneck.

Its `TestsetGenerator` builds a knowledge graph over your documents and synthesises question/context/answer triples, including multi-hop questions that require combining passages. Its core metrics are **reference-free** — faithfulness, answer relevancy, context precision, context recall — so they need no ground-truth answers.

Generate ~200 questions across your own paper corpus. This is a config-and-run job, not an annotation job.

#### (c) The part you cannot skip: calibrate the judge

Here is the number that decides how much to trust all of the above. Measured correlation between RAGAS metrics and human evaluation runs at a **harmonic mean of about 0.55** — far below what would justify trusting automated scores unexamined. LLM judges also carry documented biases: position, verbosity, self-enhancement (a model favours its own output), and authority bias, with the CALM framework cataloguing twelve.

The literature's own recommendation is unambiguous: validate your judge against a small human-labelled sample before trusting it at scale.

So the manual work does not vanish — it shrinks by roughly 80% and moves somewhere far more valuable:

- Sample **~30 items** from the RAGAS output.
- Grade them yourself: is this question answerable from the retrieved context, and is the model's answer actually faithful to it?
- Report **agreement between your grades and the judge's** (Cohen's κ, or plain agreement rate).

That's about an hour, not an evening, and it converts a soft claim into a defensible one. It is also the single most sophisticated thing in this entire project. Almost nobody validates their evaluator, and a reader who knows the field will notice immediately.

Use a different model family as judge than as generator — generate with Llama on Groq, judge with Gemini — so the judge is not grading its own output.

#### (d) Keep one thing hand-written

Write **~20 unanswerable questions** yourself — questions whose answers are genuinely absent from the corpus. Synthetic generators derive questions *from* documents, so by construction they cannot produce a question the corpus does not answer. This tests whether `NoRelevantContextError` fires instead of the model hallucinating, which is the failure mode reviewers care most about. Twenty questions is twenty minutes.

#### Effort comparison

| | Original plan | Revised |
|---|---|---|
| Retrieval labels | 120 hand-written | 0 (BEIR SciFact) |
| Generation questions | included above | 0 (RAGAS synthetic) |
| Judge validation | none | ~30 items (~1 hour) |
| Unanswerable set | 20 hand-written | 20 hand-written (~20 min) |
| **Total manual** | **~1 evening minimum** | **~1.5 hours** |

And the revised version is *more* rigorous: real expert labels, published baselines to compare against, and a measured reliability figure for your own evaluator.

### 2.2 — Build the retrieval evaluation harness

Create `eval/retrieval_eval.py` computing standard IR metrics against `relevant_chunk_ids`:

- **Recall@k** (k = 1, 3, 5, 10) — did we retrieve the right chunks at all? Most important for RAG.
- **MRR** — how high did the first correct chunk rank?
- **nDCG@10** — rank-aware quality.
- **Latency** — p50 / p95 per retrieval.

Then run the **ablation** — this is the whole point:

| Configuration | Recall@5 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|
| BM25 only | ? | ? | ? | ? |
| Dense only (MiniLM) | ? | ? | ? | ? |
| Hybrid, weighted 0.7/0.3, minmax | ? | ? | ? | ? |
| Hybrid, weighted 0.7/0.3, sum | ? | ? | ? | ? |
| Hybrid, RRF (k=60) | ? | ? | ? | ? |
| Best hybrid + cross-encoder rerank | ? | ? | ? | ? |

Both fusion strategies are already implemented and selectable via
`FUSION_METHOD`, so this table is a matter of running the harness, not writing
new retrieval code. Weighted fusion preserves each retriever's confidence; RRF
is scale-free and needs no normaliser. They currently agree on only 3–5 of the
top 5 results and never on ordering, so the choice is consequential.
Also sweep `RRF_K` (10 / 60 / 200) — it controls how sharply top ranks dominate.

Also sweep the semantic/keyword weight from 0.0 to 1.0 in 0.1 steps and **plot the curve**. Your current 0.7/0.3 split is a value you picked from a tutorial. Find out if it's actually optimal for your corpus — and if it isn't, that's a *better* story, not a worse one. "I found the tutorial default was suboptimal for academic text and tuned it to 0.5/0.5 based on a weight sweep" is a genuinely strong interview answer.

Also sweep chunk size (256 / 512 / 1024) and overlap. Chunking strategy usually dominates retrieval quality, and almost nobody measures it.

### 2.3 — Answer evaluation with RAGAS

Create `eval/answer_eval.py` wrapping RAGAS rather than hand-rolling judge prompts. Its reference-free metrics map onto what you need:

| RAGAS metric | Question it answers |
|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? (the hallucination metric) |
| **Answer relevancy** | Does the answer actually address the question? |
| **Context precision** | Are the retrieved chunks relevant, and ranked sensibly? |
| **Context recall** | Did retrieval find everything needed to answer? |

Two things RAGAS won't do, which you add yourself:

- **Citation accuracy** — do the cited chunks actually contain the claims attributed to them? You already built `citation_manager.validate_citations()`, which reports `unknown_citations` (a citation naming a source that was never supplied — the signature of a fabricated reference). Wire that into the harness; it's free.
- **Refusal correctness** — on the 20 unanswerable questions, does `NoRelevantContextError` fire instead of the model answering?

Generate with Llama on Groq, judge with Gemini. Different model families, so the judge isn't grading its own output — self-enhancement bias is documented, and a reviewer may well ask.

**Report every RAGAS number alongside the judge-agreement figure from 2.1(c).** A faithfulness score of 0.87 means something quite different at κ=0.8 than at κ=0.3, and stating both is the difference between a measurement and a decoration.

Then run the ablation that justifies your multi-agent pipeline:

| Configuration | Faithfulness | Relevance | Citation acc. | Latency | Cost/query |
|---|---|---|---|---|---|
| Single-shot generation | ? | ? | ? | ? | ? |
| Multi-agent (4-stage) | ? | ? | ? | ? | ? |

Your multi-agent pipeline makes 4 serial LLM calls. **It might not be worth 4× the latency and cost.** Measure it. If it isn't worth it, say so in the README and make it a configurable option — engineering judgment about when *not* to use complexity is a stronger signal than the complexity itself.

### 2.5 — Deterministic paragraph-level citations ⭐ *(added after M2b measurement)*

The generation eval measured a citation rate of **0.3023**. Investigating it
found that the problem is not accuracy — it is *absence*. Answers are grounded
(faithfulness 0.8650) but usually say "According to the excerpts, ..." without
naming a source. A correct answer the reader cannot verify is worth less than
it looks.

**The key insight: this does not need an LLM.** Every chunk already carries
`paper_id`, `title`, and `section_title` in its metadata, and
`AnswerGenerator._format_sources()` already maps chunks back to papers with no
model involved. What is missing is attaching those known sources to the right
*part* of the answer.

**Rules for placement:**

- Citations go at the **end of a paragraph**, never after every sentence.
- If a paragraph draws on 2-3 sources, **all of them appear together** at the
  end of that paragraph, in one bracket group.
- A paragraph that matches no source well enough gets no citation rather than a
  guessed one.

Per-sentence citation is the obvious first instinct and it is wrong here: it
makes answers unreadable, and sentence-level attribution is far less reliable
than paragraph-level because a single sentence often carries too little signal
to match a chunk confidently.

**Why deterministic beats the LLM approach:**

| | LLM citation stage | Code-based matching |
|---|---|---|
| Can invent a source | Yes | No — sources come from a fixed list |
| Extra API calls | 1 per answer | 0 |
| Reproducible | No | Yes |
| Testable | Barely | Fully |

The existing `CitationManager.add_citations_to_text()` already attempts this
but assigns sentence *i* to source *i* positionally — its own docstring calls
that "frequently wrong". Replace it with similarity matching: score each
paragraph against each retrieved chunk, and attach the sources that clear a
threshold.

Measure the same way as everything else: report citation rate and fabricated
citations before and after, on the same 43 questions. A citation stage that
raises the rate but also raises fabrications is not an improvement.

Frontend polish — hover cards, click-to-highlight the source passage — belongs
in M3 once the underlying attribution is correct and measured.

### 2.4 — Publish the results

Write `docs/EVALUATION.md` with the methodology, the tables, the weight-sweep plot, and an honest analysis including what *didn't* work. Then put the headline table **in the README above the fold**. A reviewer who sees a results table in the first screen of your README reads the rest of the repo differently.

State the methodology plainly, including its limits: which numbers come from BEIR's expert labels, which come from an LLM judge, and what your measured judge agreement was. A reader trusts a document that marks its own weak spots far more than one that presents every number with equal confidence.

Add `make eval` so the numbers are reproducible, and wire a small subset into CI (M3) so regressions are caught.

**M2 checkpoint:** you can answer "how much does your reranker actually help?" with a number. Almost no candidate can.

---

# Milestone 3 — Make it real software (~1.5 weeks) ← NEXT

**Start here.** M1 and M2 are done and pushed. The project now measures itself honestly; what it does not yet do is run like software anyone else could deploy.

Order matters. 3.1 first — the test suite already exists in outline (36 tests) but covers the newest code best and the oldest code not at all, which is backwards. Then 3.2 and 3.3, which are what turn "a Streamlit script" into "a service".

### 3.1 — Tests ✅ DONE

**254 tests, 47% line coverage**, all offline: no test calls a real API, and the
retry policy's sleeps are patched rather than waited out. `make test` runs the
suite in about 40 seconds; `make check` runs lint, types and tests together.

Coverage on the modules this milestone targeted:

| Module | Coverage |
|---|---|
| `query_router.py` | 99% |
| `eval/metrics.py` | 97% |
| `chunker.py` | 92% |
| `hybrid_search.py` | 74% |
| `citation_manager.py` | 66% |
| `keyword_search.py` | 65% |
| `pdf_parser.py` | 64% |

The rest is I/O glue — ChromaDB, sentence-transformers, provider SDKs — where a
mock would assert that the mock behaves like the mock.

**Writing the tests found four bugs**, which is the argument for having written
them:

1. **`EXTRACTION` routing was dead code.** Each family's per-pattern weight sat
   *below* its own threshold (0.5 against 0.7), so a single clear phrase never
   fired. "Extract the evaluation metrics" and "List all the datasets" both fell
   through to plain Q&A, and no extraction query could ever route. Since the
   patterns within a family are alternative phrasings of one intent, they almost
   never co-occur. Fixed by setting each weight equal to its threshold.
2. **`"Tell me about BERT"` parsed as a two-item comparison.** Item extraction
   keys off capitalisation, so the sentence-initial verb looked like a named
   entity. The stopword list now covers imperatives.
3. **`"Summarize the main findings"` routed to literature review.** The review
   pattern matched `summarize the`, and review is evaluated before summary.
   Narrowed to `summarize all`, which is what actually signals multi-document.
4. **Section titles fragmented on punctuation.** `_match_heading` stripped a
   trailing colon when *matching* but kept it in the returned title, so
   `Introduction` and `Introduction:` became two different sections.

Regression tests are pinned for the earlier bugs too — the `top_k` slider that
was never passed through, `adaptive_search` mutating shared weights, the BM25
index built once at startup, prose refusals going unrecorded, and citations
being attached to refusals.

Two tests exist purely as tripwires for problems that cost days:
`test_every_corpus_pdf_yields_a_title` would have caught the missing-`/Title`
bug on day one, and `test_kappa_punishes_the_always_yes_rater` encodes why raw
agreement is never quoted alone.

#### Original notes



Target ~70% coverage on `src/`, prioritizing:

- **`src/ingestion/chunker.py`** — boundary conditions, overlap correctness, chunks near the size limit, unicode, empty input. Pure logic, easy to test, high value.
- **`src/retrieval/hybrid_search.py`** — score fusion math with mocked sub-searchers. Verify the weights actually apply as intended. (Check whether you're normalizing BM25 and cosine scores before fusing — if not, that's a real bug and finding it via a test is a great commit.)
- **`src/generation/citation_manager.py`** — citation extraction and formatting.
- **`src/generation/query_router.py`** — routing decisions per query type.
- **`src/generation/llm_client.py`** — provider fallback and retry logic, with mocked API responses.
- **Integration test** — one end-to-end run on a small fixture PDF with a stubbed LLM.

Use `pytest` + `pytest-cov`. **Never call a real LLM API in a test.** Delete the two 0-byte files and build properly under `tests/`.

**Status after M2:** 36 tests exist, and the coverage is lopsided in the wrong direction. The newest code is well tested (title recovery, paragraph citations, refusal detection — all written with tests alongside), while `chunker.py`, `query_router.py` and `llm_client.py` have none. The oldest code is the least verified, which is exactly backwards. `hybrid_search.py` is covered indirectly by the BEIR harness — the weight sweep reproducing pure BM25 and pure dense at its endpoints is a real correctness check on the fusion math — but has no unit tests.

Two more worth adding, both suggested by M2 findings:

- **`pdf_parser.py` metadata path** — the missing-`/Title` bug (5 of 8 papers) went unnoticed for the whole project. A test asserting that every fixture PDF yields a non-empty title would have caught it on day one.
- **`eval/` itself** — eight of nineteen findings were bugs in the evaluation code. `metrics.py` is pure maths against known values, and `cohens_kappa` / `spearman` in `calibrate.py` can be checked against hand-computed cases. Evaluation code that is trusted but untested is how a failure gets scored as a success.

### 3.2 — FastAPI layer ✅ DONE

Live at `make api`, documented in [`docs/API.md`](docs/API.md), 26 tests in
`test/test_api.py` — all offline, with every singleton replaced and `warm_up`
patched out.

Verified end to end against the real corpus: `/health` reports 441 chunks from 8
papers, `/query` answered a question in 21s (retrieval 0.36s, the rest Groq
queueing against its daily cap), and an off-topic question returned **HTTP 200
with `refused: true`**.

Four decisions worth defending in an interview:

- **A refusal is a 200, not a 4xx.** Refusal is correct behaviour, measured at
  23/23 on unanswerable questions. Returning 4xx would make it indistinguishable
  from a malformed request in any error-rate dashboard.
- **Upload returns 202 with a job id.** A 92-page paper takes ~9s to ingest.
  Job state is in-process and not durable — a deliberate trade for a
  single-instance deployment, documented rather than hidden.
- **Streaming streams stages, not tokens.** `LLMClient` does not expose the
  providers' streaming APIs, so `/query/stream` emits `retrieving` →
  `generating` heartbeats → `answer`. Naming it a streaming endpoint while
  silently buffering would be a lie the client discovers at runtime.
- **Two generators are cached, one per pipeline mode.** `answer_question` takes
  no per-call mode, so switching would mean mutating a shared object — exactly
  the bug `adaptive_search` already shipped once.

Also fixed: `fastapi` and `uvicorn` were arriving transitively through chromadb
and were never declared. They are direct dependencies now, along with
`python-multipart`, whose absence makes uploads fail at request time rather than
at import.

#### Original notes



Streamlit-only says "toy." Add `api/main.py`:

```
POST /papers          — upload & ingest a PDF (returns a job id)
GET  /papers          — list ingested papers
GET  /papers/{id}     — ingestion status
POST /query           — ask a question (streaming SSE response)
GET  /health          — liveness + dependency checks
GET  /metrics         — Prometheus format
```

Pydantic request/response models throughout. **Stream the answer** — your multi-agent pipeline takes many seconds and a blocking request is a bad experience. Streaming is also a thing interviewers ask about.

Make ingestion async with a background worker so upload doesn't block. Then refactor `app.py` to call the API rather than importing `src/` directly — that separation is itself the architectural point.

### 3.3 — Docker + CI ✅ DONE

Multi-stage `Dockerfile` (node build → python runtime) plus
`.github/workflows/ci.yml` with three jobs: Python checks, frontend build, and
a Docker build that boots the container and asserts `/health` reports `ok` with
8 papers.

Decisions worth defending:

- **The corpus is indexed at build time.** `fetch_corpus.py` downloads the eight
  papers and `index_papers.py` embeds them, so the vector store ships inside the
  image. A Hugging Face Space has an ephemeral filesystem; a disk-backed
  ChromaDB would lose all 441 chunks on every restart. Baking it in deletes the
  problem rather than solving it with a database. The PDFs are then removed —
  ~28MB that the index no longer needs.
- **CPU-only torch, installed explicitly first.** The default wheels drag in
  ~2.5GB of CUDA that no CPU Space can use.
- **The embedding model is baked in.** Downloading at boot adds ~90s to a cold
  start and makes startup depend on huggingface.co being up.
- **The build fails if fewer than 8 papers index.** An image that quietly serves
  a smaller corpus than the evaluation measured is worse than a failed build.
- **One uvicorn worker.** Job state in `api/jobs.py` is per-process, so a second
  worker would let a client poll the one that does not hold its job.
- **No `docker-compose.yml`.** A Space is one container; a compose file for
  API + Streamlit + Chroma would describe a deployment that never happens.

Two problems fixed on the way:

- **`fetch_corpus.py` downloaded the wrong papers.** Its list was stale — RoBERTa,
  DistilBERT, ALBERT, DPR — and only three of eight overlapped with what is
  actually indexed. A Docker build using it would have shipped a demo answering
  from a different corpus than `docs/EVALUATION.md` describes. The list now
  matches, filenames are the paper ids so no alias file is needed, and a failed
  download raises instead of warning.
- **`ruff format` had never been run.** CI checks it, so 40 files were
  reformatted rather than shipping a check that fails on the first run.

`app.py` and Streamlit are gone. The React frontend replaced them, and the
dependency, Makefile target, lint exemption and docs references went with it.

#### Original notes


- `Dockerfile` — multi-stage, non-root user, pinned base image. Pre-download the embedding model into the image so cold start isn't 90 seconds. **Runs ingestion at build time** so the corpus ships inside the image (see 3.5).
- **No `docker-compose.yml`.** A Hugging Face Space is one container, so a compose file describing API + Streamlit + Chroma would describe a deployment that never happens. Streamlit is being removed and Chroma is embedded, not a service.
- `.github/workflows/ci.yml` — lint (`ruff`), format check (`ruff format`), type check (`mypy`), `pytest` with coverage, Docker build. On every PR.
- Pin `requirements.txt` exactly (currently all `>=`, which means your build is not reproducible) or move to `pyproject.toml` with `uv`.

### 3.4 — Replace Streamlit with a real frontend ← NEXT

**Decided:** React + Vite, built to static files, served by FastAPI itself.

Not Next.js. It needs a Node process, and a Hugging Face Space gives you one
container — so it would mean running Node beside Python, or a second
deployment. Next.js buys SSR and SEO; a research tool needs neither.

Serving the built bundle from FastAPI gives one container, one deployment, no
CORS, and SSE that works.

#### The two features that matter

**1. Click a citation, highlight its source.** This is the differentiator. Most
RAG demos cannot do it, because they never knew which chunk a claim came from.
This one does — that is what the deterministic citer bought.

It needs a backend change. `add_paragraph_citations` currently returns a
**string**:

```
...replaces recurrence. [Attention is All you Need, 2017]
```

so the frontend would have to parse text and guess which chunk is meant. The
structure already exists inside the function — it scores every paragraph
against every chunk and picks winners — and is then thrown away. Keep it:

```
paragraphs: [{ text: "...", sources: [{ chunk_id, score }] }]
chunks:     [{ chunk_id, text, paper_id, section }]
```

Clicking then needs no parsing, because the link is already in the data.

**Chunk-level, not sentence-level.** A single sentence carries too little
vocabulary to match a chunk confidently — measured while building the citer.
Sentence-level highlighting would look more precise and be less correct.

**No page numbers.** The parser has them and ingestion drops them. Adding them
means re-indexing the corpus for a small gain. Skipped deliberately.

**2. Upload with live progress.** `POST /papers` already returns 202 with a job
id and `GET /jobs/{id}` reports state.

**No per-stage progress.** Ingestion would have to thread a callback through
`IngestionPipeline`, which is currently stable and well tested. `queued →
running → succeeded` with the filename and a spinner is enough; the frontend
shows chunk and section counts when it lands.

#### What the UI must get right

- A refusal renders as a calm, normal answer — never as an error. It is correct
  behaviour, measured at 23/23.
- Retrieved papers and their scores are visible, not hidden behind a chat box.
- Uploads are labelled as session-only (see 3.5).

### 3.5 — Deploy to Hugging Face Spaces

**Decided:** Docker Space, one container, **corpus baked into the image**.

The constraint that drives everything: **HF Spaces have an ephemeral
filesystem.** Anything written to disk is lost on restart, rebuild, or wake from
sleep. A ChromaDB on disk would lose all 441 chunks every time.

The fix is to delete the problem rather than solve it: run ingestion during
`docker build` and ship the vector store inside the image. Nothing is written at
runtime, so nothing can be lost. No database, no persistent-storage add-on, no
cost.

```
stage 1: node   -> build the React bundle
stage 2: python -> FastAPI serving /api/* and the bundle, port 7860
                   embedding model + corpus baked in
```

Three things to plan for:

| Issue | Decision |
|---|---|
| Cold start ~40s (model + BM25 index) | Bake the model into the image; show an honest loading screen. Free Spaces sleep after ~48h idle, so the first visitor waits. A paid CPU upgrade (~$9/mo) removes this; not worth it yet. |
| Uploads vanish on restart | Say so in the UI. Letting someone upload a thesis and silently lose it is worse than admitting the limit. |
| Groq's 100k tokens/day | Already blocked three evaluation runs. A public demo would exhaust it in an afternoon. Default the deployment to Gemini or OpenAI. |

Secrets go in HF Space settings, never the repo.

#### Deferred: pgvector

Replacing ChromaDB with Postgres/pgvector would give real persistence, and
"I migrated the vector store and re-measured to prove the numbers held" is a
stronger story than most candidates have.

It is **deferred, not cancelled**, because it is not on the critical path to
being live, and because it carries a real cost: **every number in
`docs/EVALUATION.md` was measured on Chroma's HNSW index.** pgvector's index
behaves differently. Swapping stores and keeping the old numbers would quietly
make the document false — which would undo the best part of this project.

If it happens, the retrieval evaluation gets re-run and both sets of numbers are
published side by side.

Put the public URL at the top of the README and on the resume line itself.

### 3.5 — Observability and cost tracking

Log per query: latency broken down by stage (retrieve / rerank / generate), tokens in/out, estimated cost, retrieved chunk ids, model used. Expose aggregates on `/metrics`. Add a small "System Stats" tab in Streamlit showing p50/p95 latency and cost per query.

This gives you a second class of resume number — **operational** ones — alongside M2's quality numbers.

---

# Milestone 4 — One hard technical angle (~1.5 weeks)

M1–M3 make it a well-engineered RAG system. This makes it *yours*. **Pick exactly one** and do it well; three half-finished ideas are worse than one complete one.

### Option A — Structure-aware paper understanding (recommended)

Research papers have structure that generic chunking destroys. Your chunker treats a paper as a flat wall of text — which means a chunk can span the end of Methods and the start of Results, and the model can't tell you "according to Section 4.2."

Build:
- **Section detection** — parse Abstract / Intro / Related Work / Method / Experiments / Results / Conclusion, and attach section labels as chunk metadata.
- **Section-aware chunking** — never let a chunk cross a section boundary.
- **Table and figure caption extraction** — results tables are where the actual findings live, and you're currently throwing them away.
- **Reference parsing** — extract the bibliography into structured records.
- **Section-filtered retrieval** — "what methodology did they use?" should search Methods; "what were the results?" should search Results. Your `query_router.py` already classifies intent, so wire it to a metadata filter.

**Why this one:** it's genuinely domain-specific (you can't get it from a generic RAG tutorial), it visibly improves answers, and — critically — **you can measure the improvement with the M2 harness you already built.** "Section-aware chunking improved Recall@5 from 0.79 to 0.87" is exactly the kind of sentence that gets you an interview.

### Option B — Agentic multi-hop retrieval

Single-shot retrieval fails on questions requiring synthesis across papers ("which papers improved on the efficiency limitations identified in X?"). Build a retrieval agent that decomposes the query, retrieves iteratively, identifies gaps in what it has, and issues follow-up searches until it can answer — with a step budget and a termination condition. Measure multi-hop accuracy vs. your single-shot baseline.

### Option C — Contextual retrieval + fine-tuned reranker

Two-part: (1) implement Anthropic's contextual retrieval — prepend an LLM-generated document-context blurb to each chunk before embedding, which typically cuts retrieval failures substantially; (2) fine-tune a cross-encoder reranker on your own domain using hard negatives mined from your corpus. This is the most ML-heavy option and the best fit if you're targeting ML engineer rather than software engineer roles.

### Option D — Citation-graph-aware retrieval

Build the citation graph across your corpus from parsed references, then use it in retrieval: expand results to highly-cited neighbors, surface "papers that cite this," and rank by graph centrality within the retrieved set. Distinctive and visual (you can render the graph in the UI), but it needs a decent-sized corpus to be meaningful.

---

## Resume bullets you'll be able to write

**Now:** "Built a RAG system for research papers using LangChain, ChromaDB, and Streamlit." (Indistinguishable from thousands of others.)

**After this plan:**

> **ResearchGPT** — Evaluation-driven RAG system for academic literature · [live demo] · [github]
> - Built a 120-question benchmark with human-verified relevance labels; used it to show hybrid retrieval + cross-encoder reranking lifted **Recall@5 from 0.61 → 0.87** over a dense-only baseline
> - Implemented section-aware parsing and intent-routed metadata filtering for research papers, improving factual-query accuracy a further **9 points**
> - Measured the multi-agent generation pipeline against single-shot: **+14% faithfulness at 3.2× cost**, made it configurable per query type based on the trade-off
> - Shipped as a streaming FastAPI service with Docker, GitHub Actions CI, and 72% test coverage; **p95 latency 2.3s at $0.004/query**

Every clause has a number, and every number came from infrastructure you built.

---

## Suggested order and checkpoints

```
Week 1     M1 — cleanup, config, logging, error handling                       DONE
Week 2-3   M2 — BEIR + RAGAS harness, ablations, judge calibration, publish    ⭐
Week 4     M3 — tests + FastAPI
Week 5     M3 — Docker, CI, deploy, observability
Week 6     M4 — pick one differentiator, measure it with the M2 harness
Week 6 end Rewrite README around results; polish; write the resume bullets
```

**If you only have two weeks:** do M1 (compressed to 2 days), then M2 in full. A clean, measured, honest RAG system with no API layer beats a sprawling feature-rich one with no evidence — every time.

**The rule to hold onto:** for every feature already in this repo, be able to answer *"how do you know it helps?"* with a number. That question is what separates a project that gets you an interview from one that gets scrolled past.
