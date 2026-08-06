import { useEffect, useRef, useState } from "react";

import { ApiError, type IngestionJob, pollJob, uploadPaper } from "../api";
import { ArrowUp, Paperclip } from "./icons";

/**
 * The message box: a growing textarea with send and attach.
 *
 * **Enter sends, Shift+Enter inserts a newline.** That is the convention every
 * chat interface uses, and questions are usually one line, so requiring a
 * modifier to send would put a modifier on the common case.
 *
 * Upload lives here rather than in its own section: attaching a paper is part
 * of asking about it, and a separate upload panel would sit unused most of the
 * time while taking permanent space.
 */

interface Props {
  onSend: (question: string) => void;
  onUploaded: () => void;
  busy: boolean;
  /** True when uploads do not survive a restart, so the UI can say so. */
  ephemeral: boolean;
}

const MAX_ROWS_PX = 200;

export function Composer({ onSend, onUploaded, busy, ephemeral }: Props) {
  const [value, setValue] = useState("");
  const [job, setJob] = useState<IngestionJob | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Grow with the content, up to a cap, then scroll internally.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, MAX_ROWS_PX)}px`;
  }, [value]);

  const send = () => {
    const trimmed = value.trim();
    if (!trimmed || busy) return;
    onSend(trimmed);
    setValue("");
  };

  const upload = async (file: File) => {
    setUploadError(null);
    setJob(null);
    try {
      const started = await uploadPaper(file);
      setJob(started);
      const finished = await pollJob(started.job_id, setJob);
      if (finished.state === "succeeded") onUploaded();
      // Leave the result visible briefly, then clear so the bar does not
      // become permanent furniture.
      setTimeout(() => setJob(null), 6000);
    } catch (exc) {
      setUploadError(
        exc instanceof ApiError ? exc.message : "Upload failed. Try again.",
      );
      setJob(null);
    }
  };

  const uploading = job?.state === "queued" || job?.state === "running";

  return (
    <div className="composer">
      {(job || uploadError) && (
        <div className="composer__upload">
          {uploading && (
            <>
              <span className="spinner" aria-hidden />
              Reading {job?.filename}…
            </>
          )}
          {job?.state === "succeeded" && (
            <span className="composer__upload--ok">
              Indexed <strong>{job.paper_id}</strong> — {job.num_chunks} chunks
              from {job.num_pages} pages
            </span>
          )}
          {job?.state === "failed" && (
            <span className="composer__upload--error">
              {job.detail ?? "Ingestion failed."}
            </span>
          )}
          {uploadError && (
            <span className="composer__upload--error">{uploadError}</span>
          )}
        </div>
      )}

      <div className="composer__box">
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void upload(file);
            event.target.value = "";
          }}
        />
        <button
          type="button"
          className="composer__attach"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          title={
            ephemeral
              ? "Attach a PDF (kept for this session only)"
              : "Attach a PDF"
          }
          aria-label="Attach a PDF"
        >
          <Paperclip />
        </button>

        <textarea
          ref={textareaRef}
          className="composer__input"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
          placeholder="Ask about the indexed papers…"
          rows={1}
        />

        <button
          type="button"
          className="composer__send"
          onClick={send}
          disabled={busy || value.trim().length === 0}
          aria-label="Send"
        >
          {busy ? <span className="spinner spinner--on-accent" /> : <ArrowUp />}
        </button>
      </div>

      <p className="composer__note">
        {ephemeral
          ? "Attached papers are kept for this session only."
          : "Shift + Enter for a new line."}
      </p>
    </div>
  );
}
