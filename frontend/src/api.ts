/**
 * Typed client for the ResearchGPT API.
 *
 * Types mirror `api/schemas.py`. They are hand-written rather than generated
 * from the OpenAPI schema: the surface is nine endpoints, and a generator would
 * add a build step for something that fits on one screen.
 */

/** In production the API is same-origin. In dev, Vite proxies /api. */
const BASE = import.meta.env.DEV ? "/api" : "";

export interface CitedSource {
  chunk_id: string;
  paper_id: string;
  title: string;
  year: string;
  section: string;
  /** Content-word overlap with the paragraph, 0-1. */
  score: number;
}

export interface AttributedParagraph {
  text: string;
  sources: CitedSource[];
  /** Headings and bullet lists, which are never cited. */
  structural: boolean;
}

export interface Chunk {
  chunk_id: string;
  paper_id: string;
  title: string;
  section: string;
  text: string;
  relevance_score: number;
}

export interface Source {
  paper_id: string;
  title: string;
  author: string;
  section: string;
  relevance_score: number;
  chunk_preview: string;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: Source[];
  paragraphs: AttributedParagraph[];
  chunks: Chunk[];
  /** True when the corpus could not support the question. Not an error. */
  refused: boolean;
  num_sources: number;
  processing_time: number;
  retrieval_time: number;
  generation_time: number;
  model: string;
  generation_method: string;
}

export interface Paper {
  paper_id: string;
  title: string;
  author: string;
  num_chunks: number;
}

export interface PaperList {
  papers: Paper[];
  total_papers: number;
  total_chunks: number;
}

export type JobState = "queued" | "running" | "succeeded" | "failed";

export interface IngestionJob {
  job_id: string;
  state: JobState;
  filename: string;
  paper_id: string | null;
  detail: string | null;
  num_chunks: number | null;
  num_sections: number | null;
  num_pages: number | null;
  seconds: number | null;
}

export interface Health {
  status: "ok" | "degraded" | "unhealthy";
  version: string;
  dependencies: { name: string; ok: boolean; detail: string | null }[];
  total_papers: number;
  total_chunks: number;
}

/** Thrown for any non-2xx response, carrying the API's error body. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly kind: string,
    detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) {
    // FastAPI validation errors use `detail`; ours adds `error`. Handle both
    // so a 422 from either source reads sensibly.
    const body = await response.json().catch(() => ({}));
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : (body.detail?.[0]?.msg ?? response.statusText);
    throw new ApiError(response.status, body.error ?? "error", detail);
  }
  return response.json() as Promise<T>;
}

export function ask(
  question: string,
  topK: number,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
    signal,
  });
}

export function listPapers(): Promise<PaperList> {
  return request<PaperList>("/papers");
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

/**
 * Upload and index a PDF.
 *
 * The request returns once the paper is searchable, carrying a finished job.
 * Ingestion used to run in a background task, which on Cloud Run meant it ran
 * with the CPU throttled after the response and could be killed outright.
 *
 * A 409 means a paper with that id is already indexed; retry with
 * `replace: true` to overwrite it.
 */
export function uploadPaper(
  file: File,
  options: { replace?: boolean; paperId?: string } = {},
): Promise<IngestionJob> {
  const form = new FormData();
  form.append("file", file);
  const query = new URLSearchParams();
  if (options.replace) query.set("replace", "true");
  if (options.paperId) query.set("paper_id", options.paperId);
  const suffix = query.toString() ? `?${query}` : "";
  return request<IngestionJob>(`/papers${suffix}`, {
    method: "POST",
    body: form,
  });
}

export function deletePaper(
  paperId: string,
): Promise<{ paper_id: string; chunks_deleted: number }> {
  return request(`/papers/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  });
}

export function getJob(jobId: string): Promise<IngestionJob> {
  return request<IngestionJob>(`/jobs/${jobId}`);
}

/**
 * Poll a job until it finishes.
 *
 * Rarely needed now that `uploadPaper` returns a finished job: this is the
 * recovery path for a client that lost the upload response. Note that the job
 * registry is per-instance, so a poll can land on an instance that never saw
 * the job and get a 404.
 */
export async function pollJob(
  jobId: string,
  onUpdate: (job: IngestionJob) => void,
  intervalMs = 700,
): Promise<IngestionJob> {
  for (;;) {
    const job = await getJob(jobId);
    onUpdate(job);
    if (job.state === "succeeded" || job.state === "failed") return job;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
