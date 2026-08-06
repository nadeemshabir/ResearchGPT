import { Suspense, lazy } from "react";

/**
 * Renders one paragraph of an answer as Markdown with LaTeX maths.
 *
 * Three things were shown raw before this existed: `**bold**`, numbered lists,
 * and equations. Papers are full of the third, so a research tool that prints
 * `\sqrt{d_k}` as literal backslashes is not doing its job.
 *
 * **What this cannot fix:** PDF extraction destroys maths structure before it
 * ever reaches here. The Transformer paper's equation is stored as
 * `Attention(Q, K, V ) = softmax(QKT √dk )V` -- the superscript T, the fraction
 * bar and the subscript k are all gone. Rendering cannot recover information
 * that extraction threw away; only a maths-aware parser (Nougat, GROBID,
 * Mathpix) would. The prompt asks the model to restore formatting the notation
 * implies, and explicitly forbids adding terms that were not there.
 *
 * **Why it is lazy:** react-markdown plus KaTeX and its fonts is ~130KB
 * gzipped, more than twice the rest of the app. Loading it up front would slow
 * the first paint of a page whose first view has no maths on it at all. The
 * chunk is fetched when a question is sent, so it arrives while the model is
 * still generating and costs nothing visible.
 */

const Renderer = lazy(async () => {
  const module = await import("./ProseRenderer");
  return { default: module.ProseRenderer };
});

/** Warm the chunk before it is needed. Called when a question is sent. */
export function preloadProse(): void {
  void import("./ProseRenderer");
}

export function Prose({ children }: { children: string }) {
  return (
    // The fallback is the raw text: unstyled but readable, so a slow chunk
    // shows the answer rather than a blank space.
    <Suspense fallback={<span className="prose-fallback">{children}</span>}>
      <Renderer>{children}</Renderer>
    </Suspense>
  );
}
