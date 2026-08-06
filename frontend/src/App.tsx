import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  ask,
  getHealth,
  listPapers,
  type Health,
  type PaperList,
  type QueryResponse,
} from "./api";
import { Answer } from "./components/Answer";
import { Composer } from "./components/Composer";
import { Books, Layers, Plus, Sparkle } from "./components/icons";
import { LibraryDrawer } from "./components/LibraryDrawer";
import { preloadProse } from "./components/Prose";
import { SourceDrawer } from "./components/SourceDrawer";

/**
 * A turn in the conversation.
 *
 * Assistant turns keep the whole `QueryResponse`, so the passages behind an
 * answer stay available after later questions have been asked. Dropping it
 * would mean an answer's sources become unviewable the moment you move on.
 */
interface Turn {
  id: number;
  question: string;
  response: QueryResponse | null;
  error: string | null;
}

const EXAMPLES = [
  "What is scaled dot-product attention and why divide by the square root of d_k?",
  "How does RLHF differ from supervised fine-tuning?",
  "What are the three developmental paradigms of RAG?",
  "What top-5 error rate did AlexNet reach on ImageNet?",
];

/** Uploads do not survive a restart on the deployed demo. */
const EPHEMERAL = true;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [papers, setPapers] = useState<PaperList | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [sourcesFor, setSourcesFor] = useState<{
    turnId: number;
    chunkId: string | null;
  } | null>(null);
  const [libraryOpen, setLibraryOpen] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  const refresh = useCallback(() => {
    void Promise.allSettled([listPapers(), getHealth()]).then(
      ([paperResult, healthResult]) => {
        if (paperResult.status === "fulfilled") setPapers(paperResult.value);
        if (healthResult.status === "fulfilled") setHealth(healthResult.value);
        setLoaded(true);
      },
    );
  }, []);

  useEffect(refresh, [refresh]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, busy]);

  const send = useCallback(
    async (question: string) => {
      const id = nextId.current++;
      setTurns((current) => [
        ...current,
        { id, question, response: null, error: null },
      ]);
      setBusy(true);
      // Fetch the Markdown/KaTeX chunk now, so it lands while the model is
      // still generating rather than after the answer arrives.
      preloadProse();

      try {
        const response = await ask(question, 5);
        setTurns((current) =>
          current.map((turn) => (turn.id === id ? { ...turn, response } : turn)),
        );
      } catch (exc) {
        const message =
          exc instanceof ApiError
            ? exc.message
            : "Could not reach the server. It may still be starting up.";
        setTurns((current) =>
          current.map((turn) =>
            turn.id === id ? { ...turn, error: message } : turn,
          ),
        );
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const openSources = useCallback((turnId: number, chunkId: string | null) => {
    setSourcesFor({ turnId, chunkId });
  }, []);

  const activeTurn = turns.find((turn) => turn.id === sourcesFor?.turnId);

  // The backend is still loading its models. Say so rather than showing an
  // empty page: a cold Space takes about 40 seconds to wake.
  if (!loaded) {
    return (
      <div className="booting">
        <div className="booting__mark">R</div>
        <h1>Starting ResearchGPT</h1>
        <p>Loading the embedding model and indexing the corpus. About 40 seconds.</p>
        <div className="booting__bar" />
      </div>
    );
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar__brand">
          <div className="topbar__mark">R</div>
          <span className="topbar__name">ResearchGPT</span>
        </div>
        <div className="topbar__actions">
          {turns.length > 0 && (
            <button
              type="button"
              className="ghost-button"
              onClick={() => setTurns([])}
            >
              <Plus />
              New chat
            </button>
          )}
          <button
            type="button"
            className="ghost-button"
            onClick={() => setLibraryOpen(true)}
          >
            <Books />
            {papers ? `${papers.total_papers} papers` : "Library"}
          </button>
        </div>
      </header>

      <main className="thread">
        {turns.length === 0 ? (
          <Welcome papers={papers} onPick={send} />
        ) : (
          <div className="thread__inner">
            {turns.map((turn) => (
              <TurnView
                key={turn.id}
                turn={turn}
                onOpenSources={openSources}
              />
            ))}
            {busy && (
              <div className="msg msg--assistant">
                <div className="msg__avatar">
                  <Sparkle />
                </div>
                <div className="msg__body">
                  <div className="thinking">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      <div className="dock">
        <Composer
          onSend={send}
          onUploaded={refresh}
          busy={busy}
          ephemeral={EPHEMERAL}
        />
      </div>

      {activeTurn?.response && sourcesFor && (
        <SourceDrawer
          chunks={activeTurn.response.chunks}
          citedChunkIds={
            new Set(
              activeTurn.response.paragraphs.flatMap((paragraph) =>
                paragraph.sources.map((source) => source.chunk_id),
              ),
            )
          }
          focusChunkId={sourcesFor.chunkId}
          onClose={() => setSourcesFor(null)}
        />
      )}

      {libraryOpen && (
        <LibraryDrawer
          papers={papers}
          health={health}
          ephemeral={EPHEMERAL}
          onClose={() => setLibraryOpen(false)}
        />
      )}
    </div>
  );
}

function Welcome({
  papers,
  onPick,
}: {
  papers: PaperList | null;
  onPick: (question: string) => void;
}) {
  return (
    <div className="welcome">
      <div className="welcome__mark">R</div>
      <h1 className="welcome__title">Ask about the papers</h1>
      <p className="welcome__sub">
        {papers
          ? `${papers.total_papers} papers, ${papers.total_chunks.toLocaleString()} passages indexed.`
          : "Loading the corpus…"}{" "}
        Every claim links back to the passage it came from.
      </p>
      <div className="welcome__grid">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="suggestion"
            onClick={() => onPick(example)}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}

function TurnView({
  turn,
  onOpenSources,
}: {
  turn: Turn;
  onOpenSources: (turnId: number, chunkId: string | null) => void;
}) {
  return (
    <>
      <div className="msg msg--user">
        <div className="msg__bubble">{turn.question}</div>
      </div>

      <div className="msg msg--assistant">
        <div className="msg__avatar">
          <Sparkle />
        </div>
        <div className="msg__body">
          {turn.error && <div className="msg__error">{turn.error}</div>}

          {turn.response?.refused && (
            // A refusal is correct behaviour, not a failure, so it reads as
            // information. The system declines rather than inventing an answer.
            <div className="msg__refused">
              <strong>No relevant passage found</strong>
              <p>{turn.response.answer}</p>
            </div>
          )}

          {turn.response && !turn.response.refused && (
            <>
              <Answer
                paragraphs={turn.response.paragraphs}
                chunks={turn.response.chunks}
                onOpenSource={(chunkId) => onOpenSources(turn.id, chunkId)}
              />
              <div className="msg__foot">
                <button
                  type="button"
                  className="chip-button"
                  onClick={() => onOpenSources(turn.id, null)}
                >
                  <Layers />
                  {turn.response.chunks.length} passages
                </button>
                <span className="msg__stats">
                  {turn.response.retrieval_time.toFixed(2)}s retrieval ·{" "}
                  {turn.response.generation_time.toFixed(2)}s generation ·{" "}
                  {turn.response.model}
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
