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
  const [sending, setSending] = useState(false);
  /** Set when the server reports this id is already indexed (409). */
  const [conflict, setConflict] = useState<{ file: File; detail: string } | null>(
    null,
  );
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

  const upload = async (file: File, replace = false) => {
    setUploadError(null);
    setConflict(null);
    setJob(null);
    setSending(true);
    try {
      // The request returns once the paper is indexed, so this already holds
      // the outcome. Polling stays as the recovery path for a response that
      // comes back unfinished.
      let result = await uploadPaper(file, { replace });
      setJob(result);
      if (result.state === "queued" || result.state === "running") {
        result = await pollJob(result.job_id, setJob);
      }
      if (result.state === "succeeded") onUploaded();
      // Leave the result visible briefly, then clear so the bar does not
      // become permanent furniture.
      setTimeout(() => setJob(null), 6000);
    } catch (exc) {
      // 409: that id is already indexed. Offer to replace rather than
      // overwriting silently -- in a shared library the paper under that id
      // may be somebody else's.
      if (exc instanceof ApiError && exc.status === 409) {
        setConflict({ file, detail: exc.message });
      } else {
        setUploadError(
          exc instanceof ApiError ? exc.message : "Upload failed. Try again.",
        );
      }
      setJob(null);
    } finally {
      setSending(false);
    }
  };

  const uploading =
    sending || job?.state === "queued" || job?.state === "running";

  return (
    <div className="composer">
      {(job || uploadError || conflict) && (
        <div className="composer__upload">
          {uploading && (
            <>
              <span className="spinner" aria-hidden />
              Reading {job?.filename ?? "your PDF"}…
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
          {conflict && (
            <span className="composer__upload--error">
              {conflict.detail}{" "}
              <button
                type="button"
                className="composer__replace"
                onClick={() => void upload(conflict.file, true)}
              >
                Replace it
              </button>
            </span>
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
              : "Attach a PDF to your library"
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
