import "katex/dist/katex.min.css";

import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

/**
 * The heavy half of `Prose`, in its own chunk.
 *
 * Everything here -- react-markdown, KaTeX, and KaTeX's fonts -- is loaded
 * lazily. See `Prose.tsx` for why.
 *
 * Raw HTML is deliberately **not** enabled. The text comes from an LLM
 * summarising arbitrary PDFs, so allowing HTML through would be an injection
 * path for whatever a paper happens to contain.
 */

const REMARK = [remarkGfm, remarkMath];

// `throwOnError: false` leaves an unparseable formula visible as red source
// text instead of crashing the render. Extraction mangles maths often enough
// that a strict parser would take the whole answer down with it.
const REHYPE = [[rehypeKatex, { throwOnError: false, strict: false }]];

export function ProseRenderer({ children }: { children: string }) {
  return (
    <Markdown
      remarkPlugins={REMARK}
      rehypePlugins={REHYPE as never}
      components={{
        // The backend already splits the answer into paragraphs, and each is
        // rendered inside one container with its citations appended. Emitting
        // nested <p> tags would break that inline flow.
        p: ({ children }) => <>{children}</>,
      }}
    >
      {children}
    </Markdown>
  );
}
