import { useCallback, useRef, useState } from "react";

import { ApiError, type IngestionJob, pollJob, uploadPaper } from "../api";
import { Info, UploadCloud } from "./icons";

/**
 * Drag-and-drop PDF upload with job polling.
 *
 * The API returns 202 with a job id and ingests in the background -- a 92-page
 * paper takes about nine seconds, too long to hold a request open. Ingestion
 * reports `queued -> running -> succeeded` with no intermediate stages, so this
 * shows an indeterminate state rather than a fake percentage.
 */

interface Props {
  onIngested: () => void;
  /** True when uploads do not survive a restart, so the UI can say so. */
  ephemeral: boolean;
}

export function Upload({ onIngested, ephemeral }: Props) {
  const [job, setJob] = useState<IngestionJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const send = useCallback(
    async (file: File) => {
      setError(null);
      setJob(null);
      try {
        const started = await uploadPaper(file);
        setJob(started);
        const finished = await pollJob(started.job_id, setJob);
        if (finished.state === "succeeded") onIngested();
      } catch (exc) {
        setError(
          exc instanceof ApiError ? exc.message : "Upload failed. Try again.",
        );
        setJob(null);
      }
    },
    [onIngested],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files[0];
      if (file) void send(file);
    },
    [send],
  );

  const busy = job?.state === "queued" || job?.state === "running";

  return (
    <div className="upload">
      <div
        className={dragging ? "dropzone dropzone--over" : "dropzone"}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void send(file);
            event.target.value = "";
          }}
        />
        {busy ? (
          <>
            <span className="spinner" aria-hidden />
            <span>
              {job?.state === "queued" ? "Queued" : "Reading"} {job?.filename}…
            </span>
            <span className="dropzone__sub">
              Parsing, chunking and embedding. Long papers take a few seconds.
            </span>
          </>
        ) : (
          <>
            <span className="dropzone__icon">
              <UploadCloud />
            </span>
            <span>Drop a PDF here, or click to choose one</span>
            <span className="dropzone__sub">Research papers work best</span>
          </>
        )}
      </div>

      {ephemeral && (
        <p className="upload__note">
          <Info />
          <span>
            Uploaded papers last for this session only. The demo runs on
            ephemeral storage, so anything you add is cleared when it restarts.
          </span>
        </p>
      )}

      {job?.state === "succeeded" && (
        <p className="upload__ok">
          Indexed <strong>{job.paper_id}</strong> — {job.num_chunks} chunks
          across {job.num_sections} sections from {job.num_pages} pages
          {job.seconds !== null && ` in ${job.seconds}s`}.
        </p>
      )}

      {job?.state === "failed" && (
        <p className="upload__error">{job.detail ?? "Ingestion failed."}</p>
      )}

      {error && <p className="upload__error">{error}</p>}
    </div>
  );
}
