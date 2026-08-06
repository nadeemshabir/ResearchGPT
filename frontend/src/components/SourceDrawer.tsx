import { useEffect, useRef } from "react";

import type { Chunk } from "../api";
import { Close } from "./icons";

/**
 * Slide-in panel showing the passages retrieved for one message.
 *
 * Passages are behind a drawer rather than always on screen: in a chat the
 * answer is the thing being read, and a permanent side panel of raw chunks
 * competes with it. They stay one click away instead -- from the button under
 * each answer, or from any citation inside it.
 *
 * Uncited passages are listed too, dimmed. A chunk that was retrieved and *not*
 * used is often the more interesting fact, and hiding it would misrepresent
 * what retrieval actually did.
 */

interface Props {
  chunks: Chunk[];
  citedChunkIds: Set<string>;
  focusChunkId: string | null;
  onClose: () => void;
}

export function SourceDrawer({
  chunks,
  citedChunkIds,
  focusChunkId,
  onClose,
}: Props) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const cited = chunks.filter((c) => citedChunkIds.has(c.chunk_id));

  return (
    <>
      <div className="drawer__backdrop" onClick={onClose} aria-hidden />
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Retrieved passages"
      >
        <header className="drawer__head">
          <div>
            <h2 className="drawer__title">Retrieved passages</h2>
            <p className="drawer__sub">
              {chunks.length} retrieved · {cited.length} cited in the answer
            </p>
          </div>
          <button
            type="button"
            className="drawer__close"
            onClick={onClose}
            aria-label="Close"
          >
            <Close />
          </button>
        </header>
        <div className="drawer__body">
          {chunks.map((chunk) => (
            <ChunkCard
              key={chunk.chunk_id}
              chunk={chunk}
              cited={citedChunkIds.has(chunk.chunk_id)}
              focused={chunk.chunk_id === focusChunkId}
            />
          ))}
        </div>
      </aside>
    </>
  );
}

function ChunkCard({
  chunk,
  cited,
  focused,
}: {
  chunk: Chunk;
  cited: boolean;
  focused: boolean;
}) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    if (focused) {
      // `block: center` rather than `nearest`: the drawer opens scrolled to
      // top, so a passage far down needs to be brought into the middle to read.
      ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [focused]);

  const className = [
    "passage",
    focused ? "passage--focused" : "",
    cited ? "" : "passage--uncited",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article ref={ref} className={className}>
      <header className="passage__head">
        <h3 className="passage__title">{chunk.title}</h3>
        <span className="passage__score" title="Retrieval score">
          {chunk.relevance_score.toFixed(3)}
        </span>
      </header>
      <p className="passage__section">{chunk.section}</p>
      <div className="passage__text">{chunk.text}</div>
      {!cited && (
        <p className="passage__note">
          Retrieved, but no paragraph drew on it strongly enough to cite.
        </p>
      )}
    </article>
  );
}
