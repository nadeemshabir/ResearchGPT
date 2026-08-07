# Deploying

The image is self-contained: the corpus is fetched and indexed at build time,
so the vector store ships inside it. There is no database to provision and no
volume to mount.

**Target: Google Cloud Run.** It takes the Dockerfile unchanged, its free tier
covers a portfolio demo, and the service scales to zero when nobody is using
it.

---

## Why the corpus is baked into the image

Cloud Run containers have an **ephemeral filesystem** and scale to zero.
Anything written at runtime disappears when the instance does. A disk-backed
ChromaDB would therefore be empty for most visitors.

Three ways to handle that:

| Option | Cost | Verdict |
|---|---|---|
| **Index at build time** | Free | **Chosen.** Nothing is written at runtime, so nothing can be lost. |
| Cloud Storage + FUSE mount | Small | Adds a dependency and a failure mode to solve a problem that can be deleted. |
| Postgres + pgvector | Free tier exists | Real persistence, but every number in `EVALUATION.md` was measured on Chroma's HNSW index. Swapping stores without re-measuring would quietly make that document false. Deferred, not cancelled. |

The consequence: **papers a visitor uploads last only for that instance's
lifetime.** The UI says so rather than letting someone upload a thesis and
silently lose it.

---

## Why not Hugging Face Spaces

It was the original plan. **Gradio and Docker Spaces now require a paid PRO
plan** — only Static Spaces are free, and a static host cannot run a Python
backend with an embedding model.

If you do subscribe to PRO, the Dockerfile works there unchanged: create a
Docker Space, add YAML frontmatter to its `README.md` (`sdk: docker`,
`app_port: 7860`), set `GEMINI_API_KEY` as a secret, and push.

---

## One-time setup

You need the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and a
project with billing enabled. Billing must be on even to use the free tier;
you are not charged unless you exceed it.

### 1. Store the API key

```bash
gcloud secrets create researchgpt-gemini-key --replication-policy=automatic
printf '%s' 'YOUR_GEMINI_KEY' | gcloud secrets versions add researchgpt-gemini-key --data-file=-
```

The key lives in Secret Manager, not in an environment variable, so it never
appears in `gcloud run services describe` or the Console UI.

### 2. Grant Cloud Run access to it

```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding researchgpt-gemini-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

### 3. Deploy

```bash
./deploy/cloudrun.sh YOUR_PROJECT us-central1
```

First run takes about 15 minutes: torch, the embedding model, eight PDFs, and
an embedding pass over ~600 chunks all happen inside the build. Subsequent
deploys reuse cached layers.

**Do not point the demo at Groq.** Its free tier caps at 100,000 tokens per
day, which has already blocked three evaluation runs on this project. When it
runs out Groq *queues* rather than rejecting, so the symptom is answers taking
30 seconds, not a visible error.

---

## Settings, and why

| Setting | Value | Reason |
|---|---|---|
| `--memory` | 2 GiB | torch plus the embedding model sit around 1.5 GB resident. 1 GiB gets the container OOM-killed under load. |
| `--cpu` | 2 | Embedding a query is CPU-bound. One vCPU roughly doubles retrieval latency. |
| `--concurrency` | 8 | The app answers on a thread pool sized from the CPU count. Cloud Run's default of 80 would queue far more than one container can do, turning a busy moment into timeouts instead of a second instance. |
| `--min-instances` | 0 | What keeps it inside the free tier. Also what causes cold starts. |
| `--max-instances` | 3 | A runaway crawler should not be able to run up a bill. |
| `--timeout` | 300 s | Generation can take 20-30 s; a cold start plus a first query needs headroom. |

---

## Cold starts

**Expect 60-90 seconds for the first request after an idle period.** Two things
add up:

1. Cloud Run pulls a 2.95 GB image — cached after the first pull, but not free.
2. The app loads the embedding model and builds a BM25 index over every chunk,
   which is ~40 s.

The UI shows a loading screen explaining this rather than a blank page.

To remove it, set `--min-instances 1`. That keeps one instance warm and **leaves
the free tier** — budget roughly $10-15/month. For a portfolio link, an honest
loading screen is a reasonable trade.

---

## Verifying a deployment

```bash
URL=$(gcloud run services describe researchgpt --region us-central1 --format 'value(status.url)')

curl -fsS "$URL/health" | python -m json.tool
```

Expect `"status": "ok"` and `"total_papers": 8`. Anything less means the image
shipped a smaller corpus than `docs/EVALUATION.md` describes — though the
Dockerfile fails the build in that case, so it should never reach a deployment.

```bash
curl -fsS "$URL/" | grep -c 'id="root"'
```

Confirms the frontend bundle is served rather than the JSON fallback.

---

## Running the image locally

```bash
make docker
docker run --rm -p 7860:7860 --env-file .env researchgpt
```

Then <http://localhost:7860>. Same image, same corpus, same behaviour.

One gotcha worth knowing: **`docker run --env-file` does not parse `.env` the
way `python-dotenv` does.** It keeps inline comments and surrounding
whitespace. `Settings` strips both so the two agree, but a `.env` written for
one is not automatically valid for the other.

---

## Updating the corpus

The paper list in [`scripts/fetch_corpus.py`](../scripts/fetch_corpus.py) **is**
the deployed corpus. Add an entry, rebuild, redeploy.

Keep it in step with what the evaluation measured. An earlier version listed
RoBERTa, DistilBERT, ALBERT and DPR while the indexed corpus was eight entirely
different papers — a build from it would have shipped a demo answering from a
corpus the evaluation never touched.
