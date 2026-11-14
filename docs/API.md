# HTTP API

```bash
make api        # serve on :8000
make api-dev    # with autoreload
```

Interactive docs at `/docs`, OpenAPI schema at `/openapi.json`.

Models load during **startup**, not on the first request. Boot takes about 40
seconds — the embedding model, the vector store, and a BM25 index over every
chunk. Doing it lazily would make the first user pay for all of it.

---

## Design decisions worth knowing

### A refusal is not an error

When the corpus cannot support a question, the system says so and that comes
back as **HTTP 200 with `refused: true`**.

```json
{
  "answer": "I could not find anything relevant to that question in the papers you have uploaded.",
  "refused": true,
  "sources": [],
  "generation_method": "refused_at_retrieval"
}
```

Mapping it to 4xx would make correct behaviour look like a client mistake, and
would bury it in any error-rate dashboard. Refusal is a measured feature — 23 of
23 unanswerable questions are declined, see [EVALUATION.md](EVALUATION.md).

### Upload waits for ingestion

`POST /papers` responds **201** once the paper is searchable, carrying a
finished job. A 92-page paper takes about nine seconds.

```
POST /papers            -> 201 {"state": "succeeded", "num_chunks": 160, ...}
GET  /jobs/{job_id}      -> the same record, for a client that lost the response
```

It used to return **202** and ingest in a background task. On Cloud Run that
does not work: CPU is throttled to near zero once the response has been sent,
so the task could take minutes or be killed at the next scale-down, and the
per-process job registry meant a client polling for its job usually reached an
instance that had never seen it.

Job state is still **in-process and not durable**. `GET /jobs/{id}` can 404 on
another instance. That no longer matters much, since the outcome comes back in
the upload response.

### Uploading an id that is already indexed is a 409

The paper id is derived from the filename, so two people uploading `paper.pdf`
would land on the same id. Overwriting silently would lose somebody's work, so
the second upload is refused:

```
POST /papers                    -> 409 {"error": "paper_exists", ...}
POST /papers?replace=true       -> 201, the old chunks and PDF replaced
```

`replace=true` deletes the old chunks before indexing, rather than upserting
over them: chunk ids are `<paper_id>_chunk_<n>`, so a shorter re-index would
otherwise leave the surplus chunks of the old version searchable and citable.

Ids are cleaned before use — `Attention Is All You Need (1).pdf` becomes
`attention_is_all_you_need`. The id is the citation key a reader sees, half of
every chunk id, and the object name of the stored PDF. Pass `paper_id=` to
choose your own.

### Papers are stored, not just indexed

The PDF is kept after it is indexed, in a GCS bucket (`UPLOAD_BUCKET`) or a
local directory. `DELETE /papers/{id}` removes both. Deleting only the chunks
used to leave the file behind for good: a surprise, since the UI says the paper
is gone, and a bill that grows with every delete.

### Streaming streams stages, not tokens

`POST /query/stream` emits server-sent events:

```
event: retrieving   {"question": "...", "top_k": 5}
event: generating   {"elapsed": 2.0}          <- heartbeat every 2s
event: answer       {...full QueryResponse...}
```

**There is no token-level streaming.** `LLMClient` does not expose the
providers' streaming APIs, so the answer arrives whole. Stage events are still
useful — generation is the slow part, and heartbeats stop proxies closing an
idle connection — but the endpoint does not pretend to be something it is not.

### Requests run in worker threads

Retrieval and generation are synchronous and CPU/network bound. Running them on
the event loop would serialise every concurrent request behind the slowest one,
so each request is handed to `asyncio.to_thread`.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | Answer a question |
| `POST` | `/query/stream` | The same, as SSE |
| `GET` | `/papers` | List indexed papers |
| `GET` | `/papers/{id}` | One paper |
| `POST` | `/papers` | Upload and index a PDF (201 + finished job) |
| `DELETE` | `/papers/{id}` | Remove a paper: its chunks and its PDF |
| `GET` | `/jobs/{id}` | Ingestion status |
| `GET` | `/jobs` | Recent jobs |
| `GET` | `/health` | Liveness and dependency checks |
| `GET` | `/` | The built React bundle, or a pointer to `/docs` if it is absent |

### POST /query

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is scaled dot-product attention?", "top_k": 3}'
```

| Field | Default | Notes |
|---|---|---|
| `question` | required | 1–2000 characters |
| `top_k` | 5 | 1–50 |
| `multi_agent` | server default | Four-stage pipeline. **4× the API calls**, and not shown to be worth it — see finding #13 |

Two generators are cached, one per pipeline mode, because `answer_question`
takes no per-call mode. Mutating a shared generator would leak one request's
configuration into the next — a bug this project already shipped once, in
`adaptive_search`.

**Every request takes the plain Q&A path.** Both cached generators are built
with `use_smart_routing=False`, so comparison and literature-review queries are
answered the same way as any other question. Routing was never covered by the
evaluation and it over-triggers on the expensive review path, so the API serves
the measured behaviour. There is no request field to switch it on.

### GET /health

```json
{
  "status": "ok",
  "version": "0.2.0",
  "dependencies": [
    {"name": "vector_store",  "ok": true, "detail": "0 chunks from 0 papers (cloud: chroma-cloud/tenant/researchgpt)"},
    {"name": "paper_storage", "ok": true, "detail": "gcs: gs://my-bucket/papers"},
    {"name": "llm",           "ok": true, "detail": "gemini/gemini-2.5-flash"},
    {"name": "api_key",       "ok": true, "detail": "provider=gemini"}
  ],
  "total_papers": 0,
  "total_chunks": 0
}
```

| `status` | Meaning |
|---|---|
| `ok` | Every dependency reachable |
| `unhealthy` | A dependency failed |

**An empty corpus is `ok`.** Every paper in this service arrives by upload, so
a store with nothing in it is a deployment nobody has uploaded to yet — the
normal first state of every install. It used to report `degraded`, from when
the corpus was baked into the image and an empty one meant the build had gone
wrong. Keeping that would have meant every fresh install started life looking
broken, which is how a status field stops being read.

`total_papers` is still there, for a client that wants to show an
empty-library prompt.

The `api_key` check verifies a key exists **for the provider actually
selected** — having an OpenAI key configured does not help when
`LLM_PROVIDER=groq`.

---

## Error responses

Every failure returns the same shape:

```json
{"error": "invalid_pdf", "detail": "paper.pdf is password protected."}
```

| Status | `error` | Cause |
|---|---|---|
| 404 | — | Unknown paper or job |
| 413 | — | Upload exceeds `MAX_PDF_SIZE_MB` |
| 415 | — | Not a PDF |
| 422 | `invalid_pdf` | Encrypted, corrupt, or no text layer |
| 422 | `ingestion_failed` | Chunking, embedding, or storage failed |
| 502 | `llm_unavailable` | Provider timed out, rate-limited, or rejected the key |
| 500 | `internal_error` | Anything else |

Uploads are streamed to a temp file in 1 MB pieces and the size limit is
enforced as they arrive, so an oversized file is rejected before it lands.

---

## Testing

29 API tests in [`test/test_api.py`](../test/test_api.py), all offline: every
singleton is replaced and `warm_up` is patched out, so no model loads and no
provider is called.
