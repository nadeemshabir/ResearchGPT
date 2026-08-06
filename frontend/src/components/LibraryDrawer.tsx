import { useEffect } from "react";

import type { Health, PaperList } from "../api";
import { Close, Info } from "./icons";

/**
 * The indexed corpus, behind a drawer.
 *
 * Which papers are loaded matters before the first question and rarely after,
 * so it is one click away rather than permanently on screen.
 */

interface Props {
  papers: PaperList | null;
  health: Health | null;
  ephemeral: boolean;
  onClose: () => void;
}

export function LibraryDrawer({ papers, health, ephemeral, onClose }: Props) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="drawer__backdrop" onClick={onClose} aria-hidden />
      <aside
        className="drawer drawer--library"
        role="dialog"
        aria-modal="true"
        aria-label="Library"
      >
        <header className="drawer__head">
          <div>
            <h2 className="drawer__title">Library</h2>
            <p className="drawer__sub">
              {papers
                ? `${papers.total_papers} papers · ${papers.total_chunks.toLocaleString()} passages`
                : "Loading…"}
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
          {ephemeral && (
            <p className="drawer__note">
              <Info />
              <span>
                This demo runs on ephemeral storage. Papers you attach are
                cleared when it restarts; the ones below ship with it.
              </span>
            </p>
          )}

          <ul className="library-list">
            {papers?.papers.map((paper) => (
              <li key={paper.paper_id} className="library-item">
                <span className="library-item__title" title={paper.title}>
                  {paper.title}
                </span>
                <span className="library-item__meta">
                  {paper.author !== "Unknown" && (
                    <span className="library-item__author">{paper.author}</span>
                  )}
                  <span className="library-item__chunks">
                    {paper.num_chunks} passages
                  </span>
                </span>
              </li>
            ))}
          </ul>

          {health && (
            <dl className="drawer__health">
              {health.dependencies.map((dependency) => (
                <div key={dependency.name}>
                  <dt>
                    <span
                      className={
                        dependency.ok ? "dot dot--ok" : "dot dot--bad"
                      }
                    />
                    {dependency.name.replace(/_/g, " ")}
                  </dt>
                  <dd>{dependency.detail}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      </aside>
    </>
  );
}
