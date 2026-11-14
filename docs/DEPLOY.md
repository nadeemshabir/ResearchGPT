# Deploying

**The deployed service ships no papers.** It starts with an empty library and
answers from whatever its users upload.

That is a change. Until now the corpus was fetched and indexed at *build* time
and the vector store was baked into the image, so every deployment shipped the
same eight papers to everyone who opened it — including the eight that were
put in front of users of the live service by accident. The section below says
why that design existed and what replaced it.

**Target: Google Cloud Run.** It takes the Dockerfile unchanged, its free tier
covers a portfolio demo, and the service scales to zero when nobody is using
it.

---

## What has to outlive the container

Cloud Run containers have an **ephemeral filesystem** and scale to zero.
Anything written to local disk disappears when the instance does, and is
invisible to the instance beside it in the meantime. For a service whose entire
library arrives by upload, that is fatal: a paper someone uploads would be gone
by their next question.

So two things move off the container's disk:

| What | Setting | Why |
|---|---|---|
| The vector store (chunks + embeddings) | `CHROMA_MODE=cloud` or `http` | Where answers are retrieved from. On local disk it is emptied by every restart and not shared between instances. |
| The uploaded PDFs | `UPLOAD_BUCKET` | Needed to re-embed the corpus after a model change, and to show a reader the source. The chunks cannot be turned back into a PDF. |

Both have local fallbacks (`CHROMA_MODE=local`, no `UPLOAD_BUCKET`). Those are
for development. `deploy/cloudrun.sh` refuses to deploy with them rather than
producing a service that loses every upload.

### Why not keep baking the corpus in

Baking it in did solve the ephemeral-filesystem problem — nothing was written
at runtime, so nothing could be lost. It solved it by removing the feature:
the deployment could only ever answer from a fixed list that the person
deploying it chose, and uploads lasted until the next restart. For a demo
answering questions about eight well-known papers, that was a fair trade. For a
tool people put their own papers into, it is not.

| Option | Cost | Verdict |
|---|---|---|
| **Chroma Cloud + GCS bucket** | Free tier, then usage | **Chosen.** Nothing to run, survives restarts, shared by every instance. |
| Chroma server on a GCE e2-micro | ~free, but yours to maintain | Same model, one more thing to patch and monitor. Reasonable if you would rather not depend on a hosted service. |
| Cloud SQL + pgvector | Real money | Most solid, and the biggest change: every number in `EVALUATION.md` was measured on Chroma's HNSW index, so swapping stores without re-measuring would quietly make that document false. |
| Index at build time | Free | What this replaced. No uploads, fixed corpus. |

---

## Why not Hugging Face Spaces

It was the original plan. **Gradio and Docker Spaces now require a paid PRO
plan** — only Static Spaces are free, and a static host cannot run a Python
backend with an embedding model.

If you do subscribe to PRO, the Dockerfile works there unchanged: create a
Docker Space, add YAML frontmatter to its `README.md` (`sdk: docker`,
`app_port: 7860`), set the secrets below, and push. The same two rules apply —
a Space's filesystem is ephemeral too, so it needs the remote store and the
bucket just as much.

---

## One-time setup

You need the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and a
project with billing enabled. Billing must be on even to use the free tier;
you are not charged unless you exceed it.

Set these once so the commands below can be pasted as they are:

```bash
PROJECT=your-project-id
REGION=us-central1
BUCKET=${PROJECT}-researchgpt-papers
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
```

### 1. Store the LLM API key

```bash
gcloud secrets create researchgpt-gemini-key --replication-policy=automatic
printf '%s' 'YOUR_GEMINI_KEY' | gcloud secrets versions add researchgpt-gemini-key --data-file=-
```

The key lives in Secret Manager, not in an environment variable, so it never
appears in `gcloud run services describe` or the Console UI.

### 2. Create the vector store

Sign up at [trychroma.com](https://www.trychroma.com/) and create a database.
The dashboard gives you a tenant id, a database name, and an API key. Store
the key the same way:

```bash
gcloud secrets create researchgpt-chroma-key --replication-policy=automatic
printf '%s' 'YOUR_CHROMA_API_KEY' | gcloud secrets versions add researchgpt-chroma-key --data-file=-
```

Running your own Chroma server instead? Deploy it, then use
`CHROMA_MODE=http CHROMA_HOST=... CHROMA_PORT=... CHROMA_SSL=true` and skip
this secret.

### 3. Create the bucket for uploaded PDFs

```bash
gcloud storage buckets create "gs://${BUCKET}" \
  --project="$PROJECT" \
  --location="$REGION" \
  --uniform-bucket-level-access \
  --public-access-prevention
```

`--public-access-prevention` matters: these are documents users uploaded, and
nothing in this app serves them over a public URL.

### 4. Grant the service account access

```bash
# The uploaded PDFs: read, write and delete.
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/storage.objectAdmin

# The two secrets.
for secret in researchgpt-gemini-key researchgpt-chroma-key; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role=roles/secretmanager.secretAccessor
done
```

### 5. Deploy

```bash
UPLOAD_BUCKET="$BUCKET" \
CHROMA_MODE=cloud \
CHROMA_TENANT=your-tenant-id \
CHROMA_DATABASE=researchgpt \
./deploy/cloudrun.sh "$PROJECT" "$REGION"
```

The build is now roughly 10-15 minutes: torch and the embedding model, and no
papers. It used to be 25-35, because it also downloaded eight PDFs and
embedded ~600 chunks.

It runs on Cloud Build's **default machine**, which is what the free allowance
covers. A bigger machine roughly halves the time and is billed from the first
minute:

```bash
BUILD_MACHINE=e2-highcpu-8 UPLOAD_BUCKET="$BUCKET" CHROMA_MODE=cloud \
CHROMA_TENANT=your-tenant-id CHROMA_DATABASE=researchgpt \
./deploy/cloudrun.sh "$PROJECT"
```

**Do not point the demo at Groq.** Its free tier caps at 100,000 tokens per
day, which has already blocked three evaluation runs on this project. When it
runs out Groq *queues* rather than rejecting, so the symptom is answers taking
30 seconds, not a visible error.

---

## Settings, and why

| Setting | Value | Reason |
|---|---|---|
| `--memory` | 2 GiB | torch plus the embedding model sit around 1.5 GB resident. 1 GiB gets the container OOM-killed under load. |
| `--cpu` | 2 | Embedding a query is CPU-bound, and so is embedding an upload. One vCPU roughly doubles retrieval latency. |
| `--concurrency` | 8 | The app answers on a thread pool sized from the CPU count. Cloud Run's default of 80 would queue far more than one container can do, turning a busy moment into timeouts instead of a second instance. |
| `--min-instances` | 0 | What keeps it inside the free tier. Also what causes cold starts. |
| `--max-instances` | 3 | A runaway crawler should not be able to run up a bill. |
| `--timeout` | 300 s | Generation can take 20-30 s, and ingestion now happens inside the upload request — a 92-page paper takes about nine seconds. |

### Why ingestion runs inside the request

It used to return `202 Accepted` and ingest in a `BackgroundTasks` callback.
On Cloud Run that is a trap twice over:

- **CPU is throttled to near zero once the response is sent.** A nine-second
  ingest could take minutes, or be killed outright at the next scale-down.
- **The job registry is per-process.** With more than one instance, the client
  polling `GET /jobs/{id}` reaches whichever instance the load balancer picks,
  which is usually not the one holding its job.

So `POST /papers` now holds the connection until the paper is searchable and
returns `201` with a finished job. Slower to respond, and it actually happens.

---

## Cold starts

**Expect 60-90 seconds for the first request after an idle period.** Two things
add up:

1. Cloud Run pulls a ~2.9 GB image — cached after the first pull, but not free.
2. The app loads the embedding model, opens the remote store, and builds a BM25
   index over whatever is in it.

The second part now scales with how much your users have uploaded rather than
being fixed at eight papers. A large shared library will make startup slower.

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

Expect `"status": "ok"` and **`"total_papers": 0`** on a fresh deployment. Zero
is correct: nobody has uploaded anything yet. What matters is the
`dependencies` list — `vector_store`, `paper_storage`, `llm` and `api_key` must
all report `ok`. An empty corpus is no longer reported as `degraded`; it is the
normal first state of every install, and flagging it would train you to ignore
the field.

Then upload something and check it survives:

```bash
curl -fsS -F file=@some-paper.pdf "$URL/papers" | python -m json.tool
curl -fsS "$URL/papers" | python -m json.tool
```

The real test of the setup is that the paper is **still there after a restart**.
Force one and look again:

```bash
gcloud run services update researchgpt --region us-central1 --no-traffic --tag probe
curl -fsS "$URL/papers" | python -m json.tool
```

If the list is empty, the store is on local disk — check `CHROMA_MODE` in
`gcloud run services describe researchgpt --region us-central1`.

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

Then <http://localhost:7860>. With a `.env` using the local defaults this keeps
its store inside the container, so it starts empty and empties again when the
container goes — fine for a look around, not a test of the deployment.

To exercise what actually ships, point it at the same remote store and bucket:

```bash
docker run --rm -p 7860:7860 \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e CHROMA_MODE=cloud -e CHROMA_API_KEY="$CHROMA_API_KEY" \
  -e CHROMA_TENANT=your-tenant-id -e CHROMA_DATABASE=researchgpt \
  -e UPLOAD_BUCKET="$BUCKET" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json \
  -v "$HOME/creds.json:/creds.json:ro" \
  researchgpt
```

That writes to the **same** store the deployment uses. Use a separate Chroma
database and a separate bucket prefix (`UPLOAD_PREFIX`) if you do not want your
local tinkering in the live library.

One gotcha worth knowing: **`docker run --env-file` does not parse `.env` the
way `python-dotenv` does.** It keeps inline comments and surrounding
whitespace. `Settings` strips both so the two agree, but a `.env` written for
one is not automatically valid for the other.

---

## Multiple instances

The stores are shared now, so `--max-instances 3` is a real option rather than
a formality. Two things are still per-process:

- **The job registry** (`api/jobs.py`). `GET /jobs/{id}` can 404 on an instance
  that never saw that job. Harmless in practice, because `POST /papers` now
  returns the finished job directly.
- **The BM25 index**, rebuilt per instance when the chunk count changes. Two
  instances can briefly disagree about what is searchable by keyword — seconds,
  not minutes.

Scale with instances rather than with `--workers`: a second uvicorn worker
inside one container gets you the same two problems with none of the
redundancy.

---

## The evaluation corpus

[`scripts/fetch_corpus.py`](../scripts/fetch_corpus.py) still lists eight
papers, and `scripts/index_papers.py` still bulk-loads a directory. Those are
local tools now, not part of any deployment: the question sets, hand grades and
unanswerable set in `eval/data/` were all written about those eight papers, so
reproducing a number from `docs/EVALUATION.md` means indexing exactly that
list.

```bash
make corpus        # fetch the eight PDFs into data/raw/
make index         # index them into whatever CHROMA_MODE selects
```

**Check which store your environment selects before running `index`.** With a
cloud configuration in the shell, `scripts/index_papers.py` writes to the live
store — and `--reset` wipes it.
