---
title: ResearchGPT
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Question answering over research papers, with every claim traceable to its passage
---

# ResearchGPT

Retrieval-augmented question answering over eight foundational ML papers —
AlexNet, Transformer, BERT, GPT-3, InstructGPT, the RAG survey, Llama 3, and
DeepSeek-R1.

**Click any citation to see the exact passage it came from.**

## What is different about this one

Citations are attached **in code**, not written by the model. Each paragraph is
matched against the retrieved passages by content overlap, so a citation cannot
name a paper that was never retrieved. Measured on 43 questions: citation rate
1.000, fabricated citations 0. The LLM-written alternative scored 0.72 with 3
fabrications and cost four times the API calls.

It also **declines** questions the corpus cannot answer — 23 of 23 unanswerable
questions refused, against 1 of 15 answerable ones wrongly refused.

Retrieval was tuned on BEIR/SciFact against expert relevance labels, not
guessed. Two defaults changed on the evidence: the weight split moved from
0.7/0.3 to 0.5/0.5, and cross-encoder reranking was turned **off** after it
cost 28x the latency for +0.0075 nDCG while *lowering* Recall@5.

Full numbers, including the ones that did not support the change expected:
[docs/EVALUATION.md](https://github.com/nadeemshabir/ResearchGPT/blob/main/docs/EVALUATION.md).

## Notes on this demo

- **First load takes ~40 seconds.** The embedding model loads and a BM25 index
  is built over 441 chunks before the app serves.
- **Uploaded papers last for this session only.** The Space has an ephemeral
  filesystem, so the eight papers are baked into the image and anything you add
  is cleared on restart.

## Source

<https://github.com/nadeemshabir/ResearchGPT>
