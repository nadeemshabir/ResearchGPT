import { useCallback, useRef, useState } from "react";

import { ApiError, type IngestionJob, pollJob, uploadPaper } from "../api";
import { Info, UploadCloud } from "./icons";

/**
 * Drag-and-drop PDF upload.
 *
 * The request is held open until the paper is indexed -- a 92-page paper takes
 * about nine seconds -- and comes back with a finished job. There are no
 * intermediate stages to report, so this shows an indeterminate spinner rather
 * than a fake percentage. It falls back to polling if the response ever
 * arrives unfinished.
 *
 * A 409 means that id is already indexed. Rather than deciding for the user,
 * this offers to replace it: overwriting somebody else's upload silently is
 * how a shared library loses work.
 */

interface Props {
  onIngested: () => void;
  /** True when uploads do not survive a restart, so the UI can say so. */
  ephemeral: boolean;
}

export function Upload({ onIngested, ephemeral }: Props) {
  const [job, setJob] = useState<IngestionJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  /** Set when the server reports this paper is already indexed (409). */
  const [conflict, setConflict] = useState<{
    file: File;
    detail: string;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const send = useCallback(
    async (file: File, replace = false) => {
      setError(null);
      setConflict(null);
      setJob(null);
      setSending(true);
      try {
        let result = await uploadPaper(file, { replace });
        setJob(result);
        // The upload already waited for ingestion. Polling is the recovery
        // path for a response that somehow came back unfinished.
        if (result.state === "queued" || result.state === "running") {
          result = await pollJob(result.job_id, setJob);
        }
        if (result.state === "succeeded") onIngested();
      } catch (exc) {
        if (exc instanceof ApiError && exc.status === 409) {
          setConflict({ file, detail: exc.message });
        } else {
          setError(
            exc instanceof ApiError ? exc.message : "Upload failed. Try again.",
          );
        }
        setJob(null);
      } finally {
        setSending(false);
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

  const busy = sending || job?.state === "queued" || job?.state === "running";

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
            <span>Reading {job?.filename ?? "your PDF"}…</span>
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

      {conflict && (
        <p className="upload__error">
          {conflict.detail}{" "}
          <button
            type="button"
            className="upload__replace"
            onClick={() => void send(conflict.file, true)}
          >
            Replace it
          </button>
        </p>
      )}

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
