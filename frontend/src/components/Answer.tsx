import type { AttributedParagraph, Chunk, CitedSource } from "../api";
import { Prose } from "./Prose";

/**
 * Renders an answer with its citations linked to the passages behind them.
 *
 * Citations sit at the **end of each paragraph**, never after each sentence.
 * Per-sentence citation makes an answer unreadable, and matching a single
 * sentence to a chunk is unreliable -- so the backend attributes whole
 * paragraphs and this renders what it decided.
 *
 * Nothing here parses citation text. The backend returns paragraph-to-chunk
 * links directly, so clicking is a lookup rather than a guess.
 */

interface Props {
  paragraphs: AttributedParagraph[];
  chunks: Chunk[];
  onOpenSource: (chunkId: string) => void;
}

export function Answer({ paragraphs, chunks, onOpenSource }: Props) {
  const known = new Set(chunks.map((chunk) => chunk.chunk_id));

  return (
    <div className="answer">
      {paragraphs.map((paragraph, index) =>
        paragraph.structural ? (
          <div key={index} className="answer__structural">
            <Prose>{paragraph.text}</Prose>
          </div>
        ) : (
          <div key={index} className="answer__paragraph">
            <Prose>{paragraph.text}</Prose>
            {paragraph.sources.map((source) => (
              <Citation
                key={source.chunk_id}
                source={source}
                known={known.has(source.chunk_id)}
                onOpen={onOpenSource}
              />
            ))}
          </div>
        ),
      )}
    </div>
  );
}

function Citation({
  source,
  known,
  onOpen,
}: {
  source: CitedSource;
  known: boolean;
  onOpen: (chunkId: string) => void;
}) {
  const label = shortTitle(source.title);

  // A citation whose chunk was not returned cannot be opened. A test asserts
  // this never happens, but rendering a dead button would be worse than
  // rendering plain text.
  if (!known) {
    return <span className="cite cite--plain">{label}</span>;
  }

  return (
    <button
      type="button"
      className="cite"
      onClick={() => onOpen(source.chunk_id)}
      title={`${source.title} — ${source.section} · ${(source.score * 100).toFixed(0)}% overlap`}
    >
      {label}
      <span className="cite__year">{source.year}</span>
    </button>
  );
}

/**
 * Papers have long titles. "Language Models are Few-Shot Learners" reads fine
 * inline; "BERT: Pre-training of Deep Bidirectional Transformers for Language
 * Understanding" does not, so it is cut at the subtitle and the full title
 * stays in the tooltip.
 */
function shortTitle(title: string): string {
  const beforeColon = title.split(":")[0] ?? title;
  const words = beforeColon.split(/\s+/);
  return words.length > 5 ? `${words.slice(0, 5).join(" ")}…` : beforeColon;
}
