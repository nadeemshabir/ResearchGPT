# Deploying to Hugging Face Spaces

The image is self-contained: the corpus is indexed at build time and the vector
store ships inside it. There is no database to provision and no volume to
mount.

---

## Why the corpus is baked in

**Hugging Face Spaces have an ephemeral filesystem.** Anything written at
runtime is lost when the Space restarts, rebuilds, or wakes from sleep. A
disk-backed ChromaDB would therefore be empty every time a visitor arrives
after an idle period.

Three ways to solve that:

| Option | Cost | Verdict |
|---|---|---|
| **Index at build time** | Free | **Chosen.** Nothing is written at runtime, so nothing can be lost. |
| HF persistent storage | ~$5/month | Works, but pays for a problem that can be deleted instead. |
| External Postgres + pgvector | Free tier exists | Real persistence, but every number in `EVALUATION.md` was measured on Chroma's HNSW index. Swapping stores without re-measuring would quietly make that document false. Deferred, not cancelled. |

The consequence is that **papers a visitor uploads last only for that session**.
The UI says so rather than letting someone upload a thesis and silently lose it.

---

## One-time setup

### 1. Create the Space

At <https://huggingface.co/new-space>:

- **SDK:** Docker → *Blank*
- **Hardware:** CPU basic (free) is enough
- **Visibility:** Public

### 2. Add the Space README

A Docker Space is configured by YAML frontmatter in its `README.md`. That
frontmatter is not in this repository's README, because GitHub renders it as a
stray table at the top of the page. Copy
[`deploy/README.space.md`](../deploy/README.space.md) into the Space as
`README.md`.

### 3. Set the API key

Space **Settings → Variables and secrets**:

| Kind | Name | Value |
|---|---|---|
| Secret | `GEMINI_API_KEY` | your key |
| Variable | `LLM_PROVIDER` | `gemini` |
| Variable | `LLM_MODEL` | `gemini-2.5-flash` |

**Do not use Groq for a public demo.** Its free tier caps at 100,000 tokens per
day, which has already blocked three evaluation runs on this project; a demo
would exhaust it in an afternoon. When it runs out Groq *queues* rather than
rejecting, so the symptom is answers taking 30 seconds instead of an error.

### 4. Push

```bash
git remote add space https://huggingface.co/spaces/<user>/researchgpt
git push space m1-m2-evaluation-and-citations:main
```

The Space builds the Dockerfile. First build takes roughly 10-15 minutes:
torch, the embedding model, eight PDFs, and an embedding pass over 441 chunks.

---

## What to expect once it is live

**Cold start is about 40 seconds.** The app loads the embedding model and
builds a BM25 index over every chunk before serving. The UI shows a loading
screen saying so rather than a blank page.

**Free Spaces sleep after roughly 48 hours idle.** The first visitor after that
waits for a full container start. A paid CPU upgrade (~$9/month) removes it.
For a portfolio link, an honest loading screen is a reasonable trade.

**`/health` reports three states.** `ok` is everything reachable with a
populated corpus; `degraded` means the service works but nothing is indexed;
`unhealthy` means a dependency failed. An empty corpus is deliberately *not*
unhealthy — a load balancer should keep sending traffic to a working instance
that simply has nothing to search.

---

## Verifying a deployment

```bash
curl -fsS https://<space>.hf.space/health | jq
```

Expect `"status": "ok"` and `"total_papers": 8`. Anything less means the build
shipped a smaller corpus than `docs/EVALUATION.md` describes — though the
Dockerfile fails the build in that case, so it should not reach a Space.

```bash
curl -fsS https://<space>.hf.space/ | grep -c 'id="root"'
```

Confirms the frontend bundle is being served rather than the JSON fallback.

---

## Running the image locally

```bash
make docker                                   # build
docker run --rm -p 7860:7860 --env-file .env researchgpt
```

Then <http://localhost:7860>. Same image, same corpus, same behaviour as the
Space.

---

## Updating the corpus

The paper list lives in [`scripts/fetch_corpus.py`](../scripts/fetch_corpus.py)
and **is** the deployed corpus. Add an entry, rebuild, redeploy.

Keep it in step with what the evaluation measured. An earlier version of that
file listed RoBERTa, DistilBERT, ALBERT and DPR while the indexed corpus was
eight entirely different papers — a Docker build from it would have shipped a
demo answering from a corpus the evaluation never touched.
