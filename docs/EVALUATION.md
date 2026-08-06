# Evaluation

Retrieval quality measured on [BEIR/SciFact](https://huggingface.co/datasets/BeIR/scifact),
a standard information-retrieval benchmark of expert-written scientific claims
paired with biomedical abstracts.

**Reproduce with `make eval`.** Raw output is committed under
[`eval/results/`](../eval/results/).

---

## Summary of findings

Sixteen findings, each with the measurement behind it. Findings 1-8 come from
the retrieval benchmark, 9-16 from the generation and refusal evaluation. Full
detail in the sections that follow.

Worth noting up front: **eight of the sixteen were bugs in the evaluation code,
not the system.** Twice, a failure was being scored as a success.

| # | Finding | Evidence | Verdict |
|---|---|---|---|
| 1 | Hybrid retrieval beats either component alone | Recall@5 0.7523 vs 0.6969 (BM25) / 0.7346 (dense) | Kept |
| 2 | The inherited 0.7/0.3 weight was suboptimal | Sweep peaked at 0.5/0.5: nDCG@10 0.6972 vs 0.6873 | **Default changed** |
| 3 | Cross-encoder reranking does not earn its cost | 28× latency for +0.75 nDCG; Recall@5 *drops* 1.1 | **Default changed** |
| 4 | Tuning one float beat adding a reranker | 0.5/0.5 no-rerank ≥ 0.7/0.3 + rerank on 3 of 4 metrics | Informs #2, #3 |
| 5 | RRF underperformed weighted fusion | 3.8 Recall@5 points behind at k=60 | Weighted kept |
| 6 | RRF's standard k=60 is too high here | k<20 gains ~1 nDCG point, but noise is ±0.5 | Documented, not changed |
| 7 | Our BM25 trails the published BEIR baseline | 0.6301 vs 0.665 nDCG@10 | Open task |
| 8 | The refusal gate was silently broken without reranking | Off-topic query scored 0.82 hybrid vs 0.91 on-topic | **Bug fixed** |
| 9 | A 4-question sample scored two metrics at a perfect 1.0000 | Same metrics at n=43: 0.8711 and 0.9147 | Sample size raised |
| 10 | Generator choice moves generation metrics, not retrieval ones | `context_recall` identical at 0.9225 across two models | Validates the metrics |
| 11 | The citation audit both under- and over-counted | Missed `[Source N:]`; scored quoted references as fabricated | **Bug fixed** |
| 12 | 5 of 8 papers carried no title, so nothing was citable | Citation rate 0.1628 → 0.3023 after recovering titles | **Bug fixed** |
| 13 | Citing in code beat citing with an LLM, on every axis | Rate 1.0000 vs 0.7209, 0 fabrications vs 3, 1 API call vs 4 | **Default changed** |
| 14 | Refusals written in prose were counted as fabrications | 5 of 5 correct refusals scored as failures | **Bug fixed** |
| 15 | The prompt was fine; the refusal *measurement* was broken | 23/23 unanswerable refused once both gates were counted | `0.40` kept |
| 16 | The harness scored API failures as successful answers | Two runs minutes apart read 1/15 and 0/15 on identical code | **Bug fixed** |
| 17 | The judge barely discriminates: 13 of 23 answers scored exactly 1.0 | κ = −0.195 at every cut-off tried, robust to dropping any single row | **Scores withdrawn** |
| 18 | Two of five human "unsupported" verdicts did not survive review | Removing them moved ρ from +0.141 to +0.365 | Calibration set too small |
| 19 | The test set contains questions built from example prompts | A question about "list C" comes from a code sample inside InstructGPT | Open |

### Every score, in one place

All figures: BEIR/SciFact test split, 300 queries, 5,183 documents, 50
candidates retrieved, top 10 scored.

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|---|---|
| BM25 only | — | 0.6969 | 0.7598 | 0.5937 | 0.6301 | 98 ms |
| Dense only (MiniLM) | — | 0.7346 | 0.7833 | 0.6012 | 0.6422 | 40 ms |
| Hybrid 0.7/0.3 minmax | — | 0.7523 | 0.8110 | 0.6512 | 0.6873 | 179 ms |
| Hybrid 0.7/0.3 sum | — | 0.7405 | 0.7974 | 0.6583 | 0.6888 | 179 ms |
| Hybrid RRF k=60 | — | 0.7141 | 0.7924 | 0.6222 | 0.6599 | 178 ms |
| Hybrid 0.7/0.3 + rerank | — | 0.7416 | 0.8272 | 0.6629 | 0.6948 | 4,998 ms |
| Hybrid RRF k=60 + rerank | — | 0.7389 | 0.8306 | 0.6594 | 0.6935 | 9,882 ms † |
| **Hybrid 0.5/0.5 minmax (shipped)** | — | **0.7505** | **0.8202** | **0.6632** | **0.6972** | **179 ms** |

† Latency inflated by CPU contention; see the caveat under Reranking.

Recall@1 and Precision@k are recorded in the raw JSON under
[`eval/results/`](../eval/results/) but omitted here: with 1.13 relevant
documents per query, Precision@k is close to Recall@k/k and adds no signal.

### Decisions taken from these measurements

| Setting | Was | Now | Reason |
|---|---|---|---|
| `SEMANTIC_WEIGHT` / `KEYWORD_WEIGHT` | 0.7 / 0.3 | **0.5 / 0.5** | Finding #2: +1.0 nDCG@10, free |
| `USE_RERANKING` | `true` | **`false`** | Finding #3: 28× latency, negative Recall@5 |
| `RRF_K` | 60 | 60 *(unchanged)* | Finding #6: gain is real but within ~2× the noise floor |
| `FUSION_METHOD` | `weighted` | `weighted` *(unchanged)* | Finding #5: RRF lost even when tuned |

Two defaults changed on evidence. Two were left alone because the evidence was
not strong enough. The second half matters as much as the first.

### The final configuration

This is what the system ships with today, and why. The last column is the
important one: it says whether a value was tested or just inherited.

| Setting | Value | Why this value | Tested? |
|---|---|---|---|
| `fusion_method` | `weighted` | RRF lost by 3.8 Recall@5 points, and still lost after tuning its `k` | **Yes** |
| `semantic_weight` | `0.5` | Swept 0.0–1.0. Best nDCG@10 at 0.5 (0.6972 vs 0.6873 at the old 0.7) | **Yes** |
| `keyword_weight` | `0.5` | Same sweep | **Yes** |
| `normalisation` | `minmax` | Beat `sum` on Recall@5 (0.7523 vs 0.7405). Very close on other metrics | **Yes** |
| `use_reranking` | `false` | 28× slower, and Recall@5 dropped 1.1 points | **Yes** |
| `min_semantic_similarity` | `0.40` | Stops off-topic answers. Sits between an on-topic score (~0.49) and an off-topic one (~0.35) | Weakly — 6 probes only |
| `rrf_k` | `60` | Unused while fusion is `weighted`. Left at the literature standard | Yes, but not changed |
| `top_k_retrieve` | `50` | How many candidates to fetch. Held fixed across all runs so results stay comparable | No |
| `top_k_rerank` | `10` | How many results to keep | No |
| `min_hybrid_score` | `0.2` | Filters weak results *after* relevance is confirmed | No |
| `chunk_size` / `chunk_overlap` | `1000` / `200` | Inherited. SciFact abstracts fit in one chunk, so this could not be tested here | **No** |
| `embedding_model` | `all-MiniLM-L6-v2` | Inherited. Fast and small | **No** |
| `use_query_processing` | `true` | Inherited | **No** |

Four settings rest on measurements. One rests on six probes. The rest are
inherited defaults that have not been tested yet, and are marked as such rather
than presented as choices.

Chunk size and embedding model are the two biggest untested settings. Testing
them needs a full-text corpus, because abstracts are too short for chunking to
matter. That is a known gap, not an oversight.

---

## Why this dataset

Retrieval metrics need relevance judgments: for each query, which documents are
actually correct. Three options exist, and only one of them is honest at this
scale.

| Approach | Problem |
|---|---|
| Hand-label a private set | Slow, small, and unverifiable by a reader |
| LLM-generated labels | Measures the labelling model's opinion, not retrieval |
| **An existing expert-labelled benchmark** | **None of the above** |

SciFact ships 300 test queries over 5,183 abstracts with 339 human relevance
judgments. Nothing here was labelled by me or by a model, and the numbers are
directly comparable to published baselines — a reader can check them.

| Property | Value |
|---|---|
| Documents | 5,183 abstracts |
| Queries (test split) | 300 |
| Relevance judgments | 339 (1.13 per query, binary) |
| Indexed as | 5,208 chunks (abstracts rarely exceed one chunk) |

## Setup

| Component | Setting |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunk size / overlap | 1000 / 200 tokens |
| Candidates retrieved | 50 per query |
| Results scored | top 10 |
| Hardware | CPU only |

Chunks are collapsed to documents before scoring — a document takes the rank of
its best-scoring chunk — because BEIR judges documents while this system
retrieves chunks. Without that step a document split across three chunks would
occupy three of the top ten slots and inflate precision.

---

## Retrieval ablation

300 queries, identical index and judgments across every row. Each row changes
exactly one thing.

This was the *first* ablation, run at the then-current 0.7/0.3 weight. The
weight sweep further down improved on every row here; the shipped configuration
is 0.5/0.5, listed in the summary table above.

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|---|
| BM25 only | 0.6969 | 0.7598 | 0.5937 | 0.6301 | 98 ms |
| Dense only (MiniLM) | 0.7346 | 0.7833 | 0.6012 | 0.6422 | 40 ms |
| Hybrid weighted 0.7/0.3, minmax | 0.7523 | 0.8110 | 0.6512 | 0.6873 | 179 ms |
| Hybrid weighted 0.7/0.3, sum | 0.7405 | 0.7974 | 0.6583 | 0.6888 | 179 ms |
| Hybrid RRF (k=60) | 0.7141 | 0.7924 | 0.6222 | 0.6599 | 178 ms |

### What this shows

**Hybrid retrieval earns its complexity.** Recall@5 rises from 0.6969 (BM25) and
0.7346 (dense) to 0.7523 when the two are fused — **+5.5 points over BM25 and
+1.8 over dense**. nDCG@10 improves by 5.7 and 4.5 points respectively. Combining
the two signals beats either alone on every metric, which is the result the
architecture assumed but had never verified.

**RRF loses to weighted fusion here, which contradicts the usual advice.**
Reciprocal Rank Fusion is the default in Elasticsearch, OpenSearch, Weaviate,
and Qdrant, and it is scale-free, so it avoids the normalisation problem
entirely. It still comes in **3.8 recall@5 points and 2.7 nDCG points behind**
weighted fusion on this corpus.

The likely reason is visible in the formula. With `k=60`, a document ranked 1st
contributes `1/61` and one ranked 50th contributes `1/110` — less than a 2×
spread across the entire candidate list. RRF deliberately flattens confidence,
and SciFact rewards confidence: most queries have exactly one relevant document,
and the dense retriever is often decisively right about which. A sweep over `k`
is the obvious follow-up.

This is precisely why the ablation exists. The popular choice is not the right
choice here, and no amount of reasoning would have established that.

**minmax versus sum is close to a tie.** minmax wins on Recall@5 by 1.2 points.
sum wins on MRR by 0.8 and on nDCG@10 by 0.2. Those last two are inside the
noise floor estimated later from the RRF sweep (~±0.5 points), so the only
difference worth acting on is the recall gap.

minmax stays the default on that basis. Recall matters most for RAG, because
the generator cannot use a passage that retrieval never returned.

### Against published baselines

| System | nDCG@10 on SciFact |
|---|---|
| BM25 (published BEIR baseline) | 0.665 |
| **This system, BM25 only** | **0.6301** |
| **This system, best hybrid** | **0.6888** |
| ColBERTv2 (published) | 0.693 |

Two honest observations.

**The BM25 implementation underperforms the published baseline by 3.5 points.**
That gap is real and worth naming: this system tokenises with a simple
lowercase regex, with no stemming, no stopword removal, and default `k1`/`b`
parameters, whereas the BEIR baseline uses a tuned Anserini/Lucene pipeline.
Improving the analyzer is a concrete open task, not a mystery.

**The hybrid configuration lands within half a point of ColBERTv2** (0.6888 vs
0.693) while running a 384-dimension bi-encoder plus BM25 on CPU, against a
late-interaction model that stores per-token embeddings. That is a favourable
cost/quality position, not a claim of superiority.

---

## Reranking

Cross-encoder reranking over the top 50 candidates, same 300 queries.

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|---|
| Hybrid weighted 0.7/0.3 | 0.7523 | 0.8110 | 0.6512 | 0.6873 | 179 ms |
| Hybrid weighted 0.7/0.3 **+ rerank** | 0.7416 | 0.8272 | 0.6629 | 0.6948 | 4,998 ms |
| Hybrid RRF (k=60) | 0.7141 | 0.7924 | 0.6222 | 0.6599 | 178 ms |
| Hybrid RRF (k=60) **+ rerank** | 0.7389 | 0.8306 | 0.6594 | 0.6935 | 9,882 ms |

### The reranker does not earn its cost here

This is the headline negative result, and it is worth stating plainly.

Reranking the weighted-hybrid output changes quality by:

| Metric | Change |
|---|---|
| Recall@5 | **−1.1 points** (0.7523 → 0.7416) |
| Recall@10 | +1.6 points |
| MRR | +1.2 points |
| nDCG@10 | **+0.75 points** (0.6873 → 0.6948) |

For that, latency rises from 179 ms to 4,998 ms — roughly **28×**. Recall@5, the
metric that matters most for RAG because it bounds what the generator can use,
actually gets *worse*.

A plausible mechanism is domain mismatch. `ms-marco-MiniLM-L-6-v2` was trained
on MS MARCO — short web-search queries against web passages. SciFact queries are
expert-written scientific claims against biomedical abstracts. The cross-encoder
reorders confidently, but not always correctly, so it shuffles some genuinely
relevant abstracts out of the top 5 while pulling others into the top 10.

**Conclusion: reranking stays off by default on this corpus.** It remains
available via `USE_RERANKING=true` for corpora where it does help, and a
domain-appropriate cross-encoder (or a fine-tuned one) is the obvious thing to
test before discarding the approach entirely.

This is precisely the kind of finding the harness exists to produce. Reranking
was a headline feature of this system, added because it is standard practice.
It is standard practice, and on this data it costs 28× the latency for a change
within noise on the metric that counts.

### Reranking erases the fusion-strategy gap

Without reranking, weighted fusion leads RRF by 3.8 recall@5 points. With
reranking, the gap collapses to 0.27 points (0.7416 vs 0.7389), and RRF is
marginally *ahead* on recall@10 (0.8306 vs 0.8272).

That follows from what reranking does: it rescores the whole 50-candidate pool
from scratch, so the order fusion produced matters only insofar as it determines
pool membership — and both strategies retrieve nearly the same 50 documents.

The practical reading: **if you rerank, fusion choice barely matters; if you do
not, weighted fusion wins clearly.** Since the measurements say not to rerank,
the fusion choice does matter, and weighted stays the default.

### A caveat on the two reranked latency figures

The reranked rows show 4,998 ms and 9,882 ms, but that ~2× gap is almost
certainly **measurement artifact, not a property of RRF**. Both configurations
rerank the same number of candidates with the same model, and there is no
mechanism by which rank-based fusion would make cross-encoding slower. The two
configurations ran sequentially over roughly 50 minutes with other work
happening on the same machine, so CPU contention is the likely explanation.

Treat the reranked latency as "seconds, not milliseconds" and do not read the
difference between them as meaningful. Isolating latency measurement on a quiet
machine is an open task.

---

## Weight sweep: the default was wrong

Semantic weight from 0.0 (pure BM25) to 1.0 (pure dense), in 0.1 steps, 300
queries each. Keyword weight is the complement.

| Semantic weight | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| 0.0 (pure BM25) | 0.6969 | 0.7598 | 0.5937 | 0.6301 |
| 0.1 | 0.7036 | 0.7772 | 0.6163 | 0.6512 |
| 0.2 | 0.7286 | 0.7738 | 0.6282 | 0.6604 |
| 0.3 | 0.7304 | 0.7791 | 0.6424 | 0.6721 |
| 0.4 | 0.7523 | 0.8005 | 0.6566 | 0.6874 |
| **0.5** | 0.7505 | **0.8202** | **0.6632** | **0.6972** |
| 0.6 | **0.7574** | 0.8143 | 0.6611 | 0.6956 |
| 0.7 *(previous default)* | 0.7523 | 0.8110 | 0.6512 | 0.6873 |
| 0.8 | 0.7431 | 0.7983 | 0.6351 | 0.6722 |
| 0.9 | 0.7379 | 0.7900 | 0.6206 | 0.6591 |
| 1.0 (pure dense) | 0.7346 | 0.7833 | 0.6012 | 0.6422 |

### The endpoints validate the fusion code

At 0.0 the sweep reproduces standalone BM25 exactly (0.6969 / 0.6301), and at
1.0 it reproduces standalone dense retrieval exactly (0.7346 / 0.6422). Those
are controls, not results: they confirm that weighted fusion degenerates
correctly to each component, so the middle of the curve can be trusted.

### 0.7/0.3 was inherited, and it is not optimal

The default came from a tutorial. The peak is at **0.5/0.5** (nDCG@10 0.6972,
+1.0 point) with the best Recall@5 at 0.6/0.4 (0.7574). The curve is a broad
plateau from 0.4 to 0.6 and falls off on both sides, so the exact peak is less
important than the finding that the previous default sat on the shoulder rather
than the top.

### Tuning the weight beats reranking, for free

Putting the sweep next to the reranking ablation:

| Configuration | Recall@5 | MRR | nDCG@10 | p95 latency |
|---|---|---|---|---|
| 0.7/0.3 + cross-encoder rerank | 0.7416 | 0.6629 | 0.6948 | 4,998 ms |
| **0.5/0.5, no reranking** | **0.7505** | **0.6632** | **0.6972** | **~179 ms** |

Changing one configuration value matches or beats a cross-encoder reranker on
three of four metrics **at roughly 1/28th the latency**. The reranker is a
model download, seconds of compute per query, and a meaningful chunk of the
system's complexity. The weight is a float.

That comparison is the single most useful thing this evaluation produced, and
no amount of reasoning about architecture would have surfaced it.

## RRF damping sweep: hypothesis half-confirmed

The main ablation left RRF 3.8 recall@5 points behind weighted fusion. The
proposed explanation was that `k=60` flattens rank differences so severely
(rank 1 contributes `1/61`, rank 50 contributes `1/110`) that RRF discards the
confidence signal SciFact rewards. If so, smaller `k` should recover ground.

| rrf_k | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| 5 | **0.7423** | 0.8027 | 0.6275 | 0.6679 |
| 10 | 0.7339 | **0.8043** | **0.6288** | **0.6690** |
| 20 | 0.7339 | 0.7975 | 0.6233 | 0.6627 |
| 40 | 0.7283 | 0.7931 | 0.6229 | 0.6608 |
| 60 *(standard)* | 0.7141 | 0.7924 | 0.6222 | 0.6599 |
| 100 | 0.7158 | 0.8029 | 0.6250 | 0.6632 |
| 200 | 0.7147 | 0.7903 | 0.6232 | 0.6588 |

**The direction was right; the magnitude was not enough.** Dropping `k` from 60
to 10 gains 0.9 nDCG points, and `k=5` gains 2.8 recall@5 points. The
literature-standard 60 is a poor choice on this corpus.

But even tuned, RRF still loses to weighted fusion:

| | Recall@5 | nDCG@10 |
|---|---|---|
| Best RRF (k=5 / k=10) | 0.7423 | 0.6690 |
| Weighted 0.5/0.5 | **0.7505** | **0.6972** |
| Gap | 0.8 points | 2.8 points |

So the flattening hypothesis explains part of RRF's deficit but not all of it.
Weighted fusion wins on SciFact whether or not RRF is tuned, and the honest
conclusion is that score magnitude carries real information on this dataset that
any purely rank-based method throws away.

**A note on noise.** The curve is not monotonic — `k=100` (0.6632) scores above
`k=60` (0.6599), which no mechanism predicts. That implies roughly ±0.5 points
of run-to-run variation on 300 queries, so the ordering *among* the low-`k`
values is not reliable. What survives the noise is the coarse finding: `k` below
about 20 beats `k=60` by around one nDCG point. Treat "k=10 is optimal" as
unsupported; treat "60 is too high here" as supported.

### Against published baselines, revisited

At the tuned 0.5/0.5 setting, nDCG@10 reaches **0.6972**, which is above
ColBERTv2's published **0.693** on SciFact — using a 384-dimension bi-encoder
plus BM25 on CPU, against a late-interaction model storing per-token
embeddings. Single dataset, so this is a favourable data point rather than a
general claim.

## Finding #8: turning reranking off exposed a broken refusal gate

Disabling the reranker surfaced a latent bug that had been masked for as long
as reranking was on by default.

`get_relevant_chunks` refuses to answer when nothing clears a relevance
threshold. With reranking on it compared against the cross-encoder logit, which
is an absolute score. With reranking off it fell back to the *fused hybrid
score* — and that score cannot judge relevance at all.

Min-max normalisation rescales scores **within the retrieved set**. The best
result therefore approaches 1.0 regardless of how poor the match is. Measured:

| Query | Hybrid score (top result) | Raw cosine (top result) |
|---|---|---|
| "What is multi-head attention?" (on topic) | 0.9062 | **0.487** |
| "What is the boiling point of liquid helium?" (off topic) | 0.8161 | **0.3457** |

The fused score separates these by 0.09; the raw cosine separates them by 0.14
on a much tighter scale, and unlike the fused score it means the same thing
from one query to the next. A corpus of transformer papers was cheerfully
answering questions about liquid helium.

**Fix.** Relevance and ranking are now two separate gates:

1. **Is anything relevant?** Compared against an absolute score — raw cosine
   similarity (`min_semantic_similarity`, default 0.40), or the cross-encoder
   logit when reranking is on.
2. **Which results are worth including?** Compared against the fused score,
   which is the right tool for ordering.

Verified on six probes, three on-topic and three off-topic: 6/6 correct.

**This threshold is under-measured** and rests on a handful of probes rather
than a benchmark. Calibrating it properly is exactly what the 20 hand-written
unanswerable questions in Milestone 2b are for — this bug is the concrete
argument for building that set.

The general lesson generalises past this codebase: *a normalised score can rank,
but it cannot decide whether anything is worth ranking.*

---

# Generation quality

Everything above measures retrieval against expert labels. This half measures
what the system *writes*, scored by [RAGAS](https://docs.ragas.io) on 43
questions generated from the indexed corpus of 8 papers.

**Reproduce with `make eval-generation`.**

## Setup

| | |
|---|---|
| Questions | 43, generated from chunks, round-robin across all 8 papers |
| Generator | varies per run (recorded in every results file) |
| Judge | `openai/gpt-4o-mini` — a different family from either generator |
| Retrieval | the shipped config: hybrid 0.5/0.5 minmax, no reranking, top_k=5 |

The judge is deliberately a different provider from the generator.
Self-enhancement bias in LLM judges is well documented, and a model marking its
own work inflates every score.

## Finding #9: a four-question sample scored a perfect 1.0000

The first smoke run used 4 questions and produced this:

| Metric | n=4 | n=43 |
|---|---|---|
| faithfulness | 0.9306 | 0.8650 |
| answer_relevancy | 0.8528 | 0.8026 |
| context_precision | **1.0000** | 0.8711 |
| context_recall | **1.0000** | 0.9147 |

Two metrics were flagged as untrustworthy *before* the full run, on the
reasoning that each question is written *from* a chunk, so retrieval finds that
chunk with no effort — a perfect score was more likely to mean the test was too
easy than that the system was flawless.

The full run brought both down by 9-13 points. The concern was right in
direction but the effect was smaller than feared: 0.87 and 0.91 are believable
numbers, not artefacts. The test set is somewhat easy, not broken.

**A metric that reads 1.0 should be treated as a bug report until n is large
enough to rule that out.**

## Finding #10: the generator moves three metrics and leaves one untouched

Two runs, identical retrieval, different generator:

| Metric | groq/llama-3.1-8b-instant | gemini/2.5-flash | Δ |
|---|---|---|---|
| faithfulness | 0.8262 | 0.8726 | +0.046 |
| answer_relevancy | 0.7792 | 0.8133 | +0.034 |
| context_precision | 0.8161 | 0.8648 | +0.049 |
| **context_recall** | **0.9225** | **0.9225** | **0.000** |
| Wall time, 43 questions | 18 min 14 s | 2 min 47 s | 6.5× faster |

`context_recall` asks whether retrieval found everything the reference answer
needs. It does not read the generated answer at all. Swapping the generator
therefore *must* leave it unchanged — and it does, to four decimal places.

This is a free correctness check on the harness. If `context_recall` had moved,
something would be leaking generator output into a retrieval metric.

The 18-minute figure is not a property of the 8B model, which normally answers
in 1-3 seconds. It is Groq queueing requests against an exhausted daily token
budget (100,000 tokens/day). Retrieval was 18-111 ms on every single query;
the entire wait was the API.

## Findings #11 and #12: chasing a low citation rate through four bugs

The first full run reported a citation rate of 0.2558 and 3 fabricated
citations. Investigating that number found four defects, three of them in the
measurement code and one in ingestion.

### The audit missed the format models actually use

[`RetrievalSystem`](../src/retrieval/retrieval_system.py) labels each chunk in
the context block as `[Source 1: Title | Section: X]`. Models copy that header
when they cite, far more often than they follow the `[Title, Year]` the prompt
requests — measured on 43 answers, 8 used the header form and 2 used the
requested form. `CitationManager` recognised only the requested form.

### The audit counted things that were not citations

The APA pattern `(Author, 2017)` matched `(Lu et al., 2023)` — a reference
printed *inside the paper text the model had quoted*, not a citation the model
emitted. Those were being reported as fabricated references.

Auditing now applies only two patterns: the header format, and the one matching
the configured citation style. Scanning for every style at once cannot
distinguish a citation from quoted source text.

### The eval compared strings that could never match

`eval/generation_eval.py` passed `paper_id` (`alexnet_2012`) as the source
title, while the model cites the real title
(`ImageNet Classification with...`). Nothing ever matched, so
`citation_coverage` read **0.0 on every answer of every run**. A metric that is
constant is not a metric.

### The root cause: 5 of 8 papers had no title

The parser read the PDF's embedded `/Title` field. LaTeX writes one only if the
author loads `hyperref` with `pdftitle`, so most arXiv preprints leave it empty:

| Paper | Embedded `/Title` |
|---|---|
| alexnet_2012, attention_2017, bert_2019 | present |
| deepseek_r1_2025, gpt3_2020, instructgpt_2022, llama3_2024, rag_survey_2023 | **empty** |

Five of eight papers reached the model as `[Source 1: Unknown | Section: ...]`.
**The model had nothing to cite.** The low citation rate was largely a
consequence of empty metadata, not of the prompt.

`PDFParser` now recovers the title from page 1 when `/Title` is blank, skipping
arXiv stamps and venue banners and stopping at the author block. All 8 papers
resolve.

### Effect of the fixes

Same generator, same judge, same questions — only the title recovery and the
audit fixes differ:

| | Before | After |
|---|---|---|
| citation rate | 0.1628 | **0.3023** |
| fabricated citations | 7 | **2** |
| `citation_coverage` | 0.0000 (constant) | 0.2151, non-zero on 11/43 |

The two remaining flags are worth reading individually: one is the model citing
`DeepSeek-R1` where the stored title is the full
`DeepSeek-R1: Incentivizing Reasoning Capability...`, which is a naming
mismatch rather than a fabrication. The other is a citation of *TriviaQA*, a
paper referenced by BERT but never retrieved — a correct catch.

### Where this leaves the citation rate

**30% is still low and is not yet explained by a bug.** The model usually
writes prose — *"According to the excerpts, ..."* — instead of a bracketed
citation. That is a prompt problem and remains open.

## Finding #13: citing in code beat citing with an LLM

The 30% citation rate above was measured on the single-shot path, which has no
citation stage at all -- those were citations the base model happened to write
unprompted. That was the wrong thing to measure, and the multi-agent number
(0.7209) is the fair comparison for the LLM approach.

The question that reframed the problem: *every chunk already stores
`paper_id`, `title` and `section_title`, and `_format_sources()` already maps
chunks back to papers with no model involved -- so why is an LLM being asked to
do a lookup?*

It should not be. `CitationManager.add_paragraph_citations()` now scores each
paragraph against each retrieved chunk by content-word overlap and appends the
sources that clear a threshold.

| | LLM single-shot | LLM multi-agent | **Deterministic** |
|---|---|---|---|
| citation rate | 0.3023 | 0.7209 | **1.0000** |
| fabricated citations | 2 | 3 | **0** |
| citation coverage | 0.2151 | — | **0.9225** |
| extra LLM calls | 0 | 3 | **0** |
| median latency | 3.2 s | — | **2.7 s** |
| faithfulness | 0.8726 | 0.8681 | 0.8581 |
| answer_relevancy | 0.8133 | 0.7530 | 0.7999 |

**The 1.0000 is guaranteed by construction, not discovered.** Every paragraph
matching a chunk gets a citation, and retrieval always returns chunks, so the
rate cannot be anything else. The numbers that carry information are coverage
(0.9225 -- the *right* sources get named) and fabrications (0 -- impossible,
since sources come from a fixed list of retrieved chunks).

Placement rules, chosen for readability:

- Citations go at the **end of a paragraph**, never after each sentence.
- A paragraph drawing on several papers gets one group: `[A, 2017; B, 2012]`.
- One paper split across two chunks is cited once.
- Headings and bullet lists are never cited.
- A paragraph matching nothing above the threshold is left uncited.

Three problems surfaced while building it:

**The model wrote its own citations too**, producing visible duplicates.
Stripping them by pattern failed, because models invent formats freely -- the
same model produced `[Attention is All you Need, Undated]` and
`[Training language models... | Section: Model, Figure 2]` in one run. The fix
is semantic rather than syntactic: **a bracketed span is a citation if it names
a source that was retrieved.** The titles are known, so this catches every
variant. The prompt now also asks the model not to cite when citations are
attached afterwards.

**The year read `n.d.`** because PDF metadata rarely carries one -- but the
paper ids do (`attention_2017`), so it is parsed from there.

**The audit undercounted merged groups.** `[A, 2017; B, 2012]` was split on
`,` only, so the second title was invisible and coverage read 0.5 on answers
that had cited everything.

## Findings #14 and #15: the refusal measurement, not the prompt

Refusal happens in two places, and only one was being counted:

1. **Retrieval** raises `NoRelevantContextError` when nothing clears
   `min_semantic_similarity`. Loud, easy to detect.
2. **The model** writes *"the provided excerpts do not contain information
   about X"* when retrieval passed weak context through. Silent.

`metadata["refused"]` was set only by case 1. Case 2 -- a correct refusal --
left the flag at `False`, so evaluation scored it as a fabricated answer.

Measured on five hard questions, **all five were refused correctly in prose and
all five were counted as failures.** The first run of `eval/refusal.py`
therefore reported "4 of 5 unanswerable questions were answered; the system made
something up", which was the opposite of what happened.

`src/generation/refusal_detection.py` now reads prose refusals. Detection is
deliberately narrow:

- A phrase must reference the *sources*. "There is no evidence that X causes Y"
  is a research finding, not a refusal.
- A disclaimer followed by a real answer stays an answer, so a genuine partial
  hallucination is not hidden behind a refusal label.
- Pronouns count when a source word appears elsewhere: *"The excerpts discuss
  SuperGLUE. However, **they** do not provide a specific score."*

Citations were also being attached to refusals -- *"the excerpts do not contain
X. [Llama 3, 2024; GPT-3, 2020]"* cites three papers for a statement that they
say nothing. Now skipped.

### Refusal, measured through both gates

23 hand-written unanswerable questions against 15 known-answerable ones:

| | First run | After fixes |
|---|---|---|
| Unanswerable correctly refused | 21 / 23 (91.3%) | **23 / 23 (100%)** |
| Answered anyway | 2 | **0** |
| Answerable wrongly refused | 1 / 15 (6.7%) | 1 / 15 (6.7%) |

Both first-run survivors were examined, and neither was a hallucination:

- One was **not a failure**: the answer correctly separated GLUE (80.5) from
  SuperGLUE and stated the SuperGLUE score was absent. The detector missed it
  because the refusal attached to a pronoun -- *"however, **they** do not
  provide a specific score"*.
- One was a **badly written test question**. It asked about "the Transformer's
  24-layer configuration", but BERT-Large in the same corpus *is* a 24-layer
  transformer with 16 heads, so answering it was correct behaviour. Replaced
  with a WMT 2022 BLEU question, which cannot appear in a 2017 paper.

**Do not read 100% as "solved".** Three reasons it overstates the system:

1. **n = 23.** One question moves the rate by four points.
2. **The detector was tuned on this set.** The pronoun pattern was added after
   watching it miss a specific answer. Fitting a detector to the test set it is
   scored on inflates the result, and the honest fix is a second question set
   written without looking at the failures.
3. **Nine of the questions were written by the same person who wrote the
   detector.** An independently written set is worth more than a larger one.

What the number does support: every `off_topic` and `unknowable` question was
caught easily, and the only questions that ever threatened the gate were
`near_miss` ones built from the corpus's own vocabulary. A question set of
obvious off-topic questions would have scored 100% from the start and proved
nothing.

The one persistent false refusal is a real answerable question about the CoLA
task, scoring 0.436 against a 0.40 floor. It sits close to the boundary, which
is what a borderline case is supposed to look like.

## Finding #16: the harness scored API failures as successful answers

Two runs of identical code, four minutes apart, disagreed:

| Run | Unanswerable refused | Answerable wrongly refused |
|---|---|---|
| 04:13 | 23 / 23 | **1 / 15** |
| 04:18 | 23 / 23 | **0 / 15** |

Nothing had changed. The CoLA question had hit a Gemini 503 and timed out after
60 seconds, and `refuses()` handled `LLMError` by returning `False` -- "did not
refuse", which the harness counted as a successful answer.

The consequence is worse than one flipped row: **a provider outage would have
driven the false-refusal rate to zero and made the system look perfect.** The
worse the infrastructure behaved, the better the score would read.

`refuses()` now returns a third state. Errored questions are excluded from both
denominators and reported separately, and the report prints a warning telling
the reader to rerun before quoting the rates.

The re-run with error tracking reproduces 23/23 and 1/15 with zero errors, which
is the number recorded above.

This is the second time in this evaluation that a failure mode was being scored
as a success -- the first was prose refusals counted as fabrications (finding
#14). Both were caught only by looking at two results that should have matched
and did not.

### The threshold stays at 0.40

The sweep favours 0.50 when only the retrieval gate is measured:

| threshold | unanswerable refused | false refusals |
|---|---|---|
| 0.40 (shipped) | 34.8% | 0 |
| 0.50 | 73.9% | 5 |
| 0.60 | 100.0% | 11 |

But that table describes one gate in isolation. End to end the model gate
already catches what retrieval passes through, reaching 91.3% at 0.40 with a
single false refusal. Raising the threshold to 0.50 would cost four more real
answers to fix a problem that is already handled downstream.

**Tuning a component against a component-level metric would have made the
system worse.** The number to optimise is the one the user experiences.

## Finding #17: the judge does not agree with a human, so its scores are withdrawn

Every RAGAS number above was an LLM marking another LLM's work. 23 answers were
then graded by hand -- *"is every claim backed by the sources shown?"* -- and
compared against the judge.

| | faithfulness |
|---|---|
| n | 23 |
| human said "supported" | 18 / 23 |
| raw agreement at 0.50 | 65.22% |
| **Cohen's κ at 0.50** | **−0.195** (worse than chance) |
| Cohen's κ at best cut-off (0.80) | +0.265 |
| Spearman ρ | +0.141 |
| **mean judge score, human said yes** | **0.781** |
| **mean judge score, human said no** | **0.783** |

Raw agreement of 65% looks respectable and is the trap: 18 of 23 answers were
"supported", so a judge that said "yes" every time would score 78%. That is why
κ is reported -- it subtracts the agreement chance alone would produce, and
here it goes negative.

**The judge barely discriminates.** 13 of the 23 graded answers received
*exactly* 1.0. A scorer that returns the top mark to more than half its inputs
cannot separate them, whatever the underlying answers look like.

### Reading the group means honestly

At face value the two group means are identical -- 0.781 versus 0.783 -- which
would say the judge carries no signal at all. That reading does not survive
inspection of the five negative grades.

| Rows included | n | negatives | κ | ρ | mean(yes) | mean(no) |
|---|---|---|---|---|---|---|
| all grades | 23 | 5 | −0.195 | +0.141 | 0.781 | 0.783 |
| minus `q023` | 22 | 4 | −0.185 | +0.234 | 0.781 | 0.729 |
| minus `q023`, `q005` | 21 | 3 | −0.167 | +0.365 | 0.781 | **0.639** |

Two of the five negatives do not hold up:

- **`q023`** -- the answer states that LLMs "can effortlessly manage contexts
  exceeding 200,000 tokens". The retrieved chunk from the RAG survey reads
  *"Presently, LLMs can effortlessly manage contexts exceeding 200,000
  tokens"*. The answer is a near-verbatim quote of its source. **The judge's
  1.0 was correct and the human grade was not.**
- **`q005`** -- asks the purpose of "the list C" in some code. The code is an
  *example prompt* printed inside the InstructGPT paper to compare GPT-3 and
  InstructGPT completions, not a claim the paper makes. Whether an answer
  describing that code counts as "supported" is genuinely ambiguous, so the
  disagreement says more about the question than about either rater.

Remove those two and ρ more than doubles, and the group means separate by 0.14.
**The judge is weakly informative, not blind** -- which is a materially
different conclusion from the one the headline numbers suggested.

What does *not* change is κ. It stays between −0.167 and −0.195 no matter which
single row is dropped. The judge's threshold is simply in the wrong place: it
marks almost everything as passing.

`answer_relevancy` could not be assessed at all: the human accepted all 23
answers, leaving no variance to correlate against. That is not a judge failure,
it is a test-set limitation -- a calibration set needs bad answers in it.

### What this means for every generation number in this document

**Faithfulness 0.8650, answer_relevancy 0.8026, context_precision 0.8711 and
context_recall 0.9147 cannot be quoted as measurements of quality.** They are
recorded here because they were run, not because they are trusted.

The retrieval numbers are unaffected -- those come from BEIR's expert labels,
not from a model.

### Caveats

- **n = 23, with 5 negatives -- of which 3 survive review.** Nothing about a
  three-item class is stable. Every number in this section should be treated as
  a direction, not a value.
- **The rater confirmed they graded attribution, not truth**, which is what
  faithfulness claims to measure, so the two tests were aligned. The `q023`
  error was a missed line in the source, not a different standard.
- **One judge, one corpus.** gpt-4o-mini on 8 papers. Not a general claim about
  RAGAS.

### What would resolve it

Grade another 30, weighted toward answers the judge scored **low**, so the
negative class is large enough for κ to mean anything. The current sample was
drawn evenly across the score range, which is right for an unbiased first look
but leaves too few negatives to measure agreement on.

### Why the scores stay withdrawn anyway

Even on the most favourable reading -- ρ = +0.365 after removing two contested
rows -- a judge that hands 1.0 to 13 of 23 answers cannot support a claim like
"faithfulness 0.8650". The score would be quoting a scale that does not
discriminate at the top end, where nearly all the mass sits.

**This is what the calibration step is for.** Without it this document would
have printed faithfulness 0.8650 as a result. With it, the honest statement is
narrower and more useful: the harness works, the judge is weakly informative and
badly calibrated, and 23 grades are not enough to say more.

## Generation scores, final

Generator `gemini/2.5-flash`, judge `openai/gpt-4o-mini`, 43 questions,
corpus re-indexed with recovered titles.

| Metric | Score |
|---|---|
| faithfulness | 0.8650 |
| answer_relevancy | 0.8026 |
| context_precision | 0.8711 |
| context_recall | 0.9147 |
| citation rate | 0.3023 |
| fabricated citations | 2 |
| refused | 0 |
| errors | 0 |
| median latency per question | 3.2 s |

**These numbers are recorded, not trusted.** Calibration against 23 hand-graded
answers returned κ = −0.195 for faithfulness, with the judge scoring 0.781 on
answers a human called supported and 0.783 on ones a human rejected. See finding
#17. Do not present any row of this table as a measurement of answer quality.

The citation and latency rows are unaffected -- those are counted in code, not
judged by a model.

---

## Limitations

- **Single dataset.** SciFact is biomedical abstracts. Results may not transfer
  to full-text papers, which is what the application actually ingests. NFCorpus
  or a full-text set would test that.
- **Binary relevance.** SciFact judgments are relevant/not-relevant, so nDCG
  cannot distinguish degrees of usefulness.
- **Short documents.** Abstracts fit in roughly one chunk each, so this run says
  almost nothing about chunking strategy. A chunk-size sweep needs a full-text
  corpus to be meaningful.
- **CPU only.** Latency figures are not representative of GPU deployment; the
  ranking of configurations by *quality* is unaffected.
- **The judge is uncalibrated.** Every generation score is an LLM's opinion.
  RAGAS correlates with human judgement at roughly 0.55; without the agreement
  figure from `eval/calibrate.py` these numbers cannot be quoted as fact.
- **The generation test set is model-written.** A model wrote both the
  questions and the reference answers, from the chunks themselves. That makes
  retrieval easier than real use and is the main reason `context_recall` sits
  above 0.91.
- **Refusal rests on 23 questions.** Enough to show the shape, not enough for a
  confident rate: one question moves it by four points. Two of the 23 turned
  out to be badly written tests, which is a fair warning about any hand-built
  set, including this one.
- **Twelve findings came from evaluation code, not the system.** Findings 9-15
  are mostly bugs in the measurement rather than in the pipeline. That ratio is
  worth stating plainly: an evaluation harness needs the same scrutiny as the
  thing it measures, and a metric that reads 0.0 or 1.0 on every row is a bug
  report until proven otherwise.

## Reproducing

```bash
make install-dev
make eval                  # full ablation, 300 queries
make eval-quick            # 50 queries, no reranking
make eval-sweep-weights    # semantic/keyword split sweep
make eval-sweep-rrfk       # RRF damping constant sweep

make index                 # index data/raw into the app's own store
make eval-testset          # build the 43-question generation set (one time)
make eval-generation       # RAGAS scores for the shipped pipeline
```

Provider and judge are selectable, and both are recorded in every results file
so runs stay comparable:

```bash
python -m eval.generation_eval \
    --generator-provider gemini --generator-model gemini-2.5-flash \
    --judge-provider openai
```

`--generator-model` alone does not switch providers; pass
`--generator-provider` with it or the request goes to the default provider with
a model name it does not recognise.

Refusal and judge calibration:

```bash
python -m eval.refusal --template     # starter file for unanswerable questions
python -m eval.refusal --sweep        # threshold sweep, retrieval gate only
python -m eval.refusal --provider gemini --model gemini-2.5-flash
python -m eval.calibrate              # grade 30 answers by hand
python -m eval.calibrate --report     # agreement stats only
```

`eval.calibrate` samples answers across the judge's score range rather than
from the top, so the disagreements are actually present, and saves after every
verdict so a session can be resumed.

The SciFact index is built once per (chunk size, embedding model) pair into
`data/eval_chroma/`, kept separate from the application's own collection.
