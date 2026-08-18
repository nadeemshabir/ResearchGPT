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

### Upload returns before ingestion finishes

`POST /papers` responds **202** with a job id. A 92-page paper takes about nine
seconds to ingest, which is too long to hold a request open.

```
POST /papers            -> 202 {"job_id": "...", "state": "queued"}
GET  /jobs/{job_id}      -> {"state": "succeeded", "num_chunks": 160}
```

Job state is **in-process and not durable**. Jobs are lost on restart, and with
more than one worker a client can poll the worker that does not hold its job.
That is a deliberate trade for a single-instance deployment; multiple workers
means moving `api/jobs.py` to shared storage first.

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
| `POST` | `/papers` | Upload a PDF (202 + job id) |
| `DELETE` | `/papers/{id}` | Remove a paper and its chunks |
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
    {"name": "vector_store", "ok": true, "detail": "597 chunks from 8 papers"},
    {"name": "llm",          "ok": true, "detail": "gemini/gemini-2.5-flash"},
    {"name": "api_key",      "ok": true, "detail": "provider=gemini"}
  ],
  "total_papers": 8,
  "total_chunks": 597
}
```

| `status` | Meaning |
|---|---|
| `ok` | Everything reachable, corpus populated |
| `degraded` | Working, but **nothing indexed** |
| `unhealthy` | A dependency failed |

An empty corpus is `degraded`, not `unhealthy`, on purpose: the service is
working correctly and a load balancer should keep sending traffic. Only an
operator can fix an empty index.

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
