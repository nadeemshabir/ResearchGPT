"""In-memory registry for background ingestion jobs.

Ingestion runs for seconds to minutes -- 92 pages of Llama 3 takes about nine --
so the upload endpoint returns a job id immediately rather than holding the
connection open.

**This is per-process and not durable.** Jobs are lost on restart, and with more
than one uvicorn worker a client can poll the worker that does not hold its job.
That is an acceptable trade for a single-instance deployment and a deliberate
one: the alternative is a Redis or database dependency that this project does
not otherwise need. Moving to multiple workers means moving this to shared
storage first.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any

from api.schemas import IngestionJob, JobState

#: Completed jobs are kept so a client that polls late still sees the outcome.
#: Bounded because nothing else evicts them.
_MAX_JOBS = 200


class JobRegistry:
    """Thread-safe store of ingestion jobs, oldest evicted first."""

    def __init__(self, max_jobs: int = _MAX_JOBS):
        self._jobs: OrderedDict[str, IngestionJob] = OrderedDict()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(self, filename: str, paper_id: str) -> IngestionJob:
        """Register a queued job and return it."""
        job = IngestionJob(
            job_id=uuid.uuid4().hex,
            state=JobState.QUEUED,
            filename=filename,
            paper_id=paper_id,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            while len(self._jobs) > self._max_jobs:
                self._jobs.popitem(last=False)
        return job

    def update(self, job_id: str, **fields: Any) -> IngestionJob | None:
        """Apply ``fields`` to a job. Returns the updated job, or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = job.model_copy(update=fields)
            self._jobs[job_id] = updated
            return updated

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[IngestionJob]:
        """Every tracked job, newest first."""
        with self._lock:
            return list(reversed(self._jobs.values()))
