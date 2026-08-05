"""Streamlit interface for ResearchGPT.

This is the presentation layer: it is the only place that formats output for a
human. Everything under ``src/`` logs instead of printing, so this module owns
all user-facing rendering.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import DEFAULT_MODELS, get_settings
from src.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMError,
    PDFParseError,
    ResearchGPTError,
)
from src.generation.answer_generator import AnswerGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retrieval_system import RetrievalSystem
from src.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_CHOICES: dict[str, list[str]] = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
}

for key, default in (
    ("generator", None),
    ("pipeline", None),
    ("chat_history", []),
    ("papers", []),
    ("initialized", False),
    ("pending_query", None),
):
    st.session_state.setdefault(key, default)


def render_error(exc: Exception) -> None:
    """Show an actionable message for a known failure, generic text otherwise."""
    if isinstance(exc, LLMAuthenticationError):
        st.error(f"**API key rejected.** {exc}")
    elif isinstance(exc, ConfigurationError):
        st.error(f"**Configuration problem.** {exc}")
    elif isinstance(exc, PDFParseError):
        st.error(f"**Could not read that PDF.** {exc}")
    elif isinstance(exc, (LLMError, ResearchGPTError)):
        st.error(f"**{type(exc).__name__}.** {exc}")
    else:
        st.error(f"**Unexpected error.** {exc}")
        logger.exception("Unhandled error in UI")


def sync_papers_from_database(pipeline: IngestionPipeline) -> None:
    """Populate the paper list from the store.

    Without this, restarting the app shows an empty library even though the
    ChromaDB collection is fully populated.
    """
    try:
        stored = pipeline.database.list_papers()
    except ResearchGPTError:
        logger.warning("Could not read paper list from the store", exc_info=True)
        return

    known = {paper["paper_id"] for paper in st.session_state.papers}
    for paper in stored:
        if paper["paper_id"] not in known:
            st.session_state.papers.append(
                {
                    "paper_id": paper["paper_id"],
                    "filename": paper.get("title") or paper["paper_id"],
                    "upload_time": "previously indexed",
                    "stats": {"num_chunks": paper.get("num_chunks", 0)},
                }
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
settings = get_settings()

with st.sidebar:
    st.title("ResearchGPT")
    st.caption("RAG over your research papers")
    st.divider()

    st.subheader("Model")
    provider_options = list(DEFAULT_MODELS)
    provider = st.selectbox(
        "Provider",
        provider_options,
        index=provider_options.index(settings.llm_provider),
    )
    if provider is None:
        provider = settings.llm_provider
    model = st.selectbox("Model", MODEL_CHOICES[provider])

    if not settings.api_key_for(provider):
        st.warning(f"No API key found for {provider}. Add it to your `.env` file.")

    st.divider()
    st.subheader("Generation")
    use_multi_agent = st.checkbox(
        "Multi-agent pipeline",
        value=settings.use_multi_agent,
        help=(
            "Four stages: analyze, synthesize, cite, critique. Higher quality "
            "in principle, but roughly one LLM call per retrieved chunk plus "
            "three, so several times slower and costlier than single-shot."
        ),
    )
    use_citations = st.checkbox("Add citations", value=settings.use_citations)
    use_smart_routing = st.checkbox(
        "Smart routing",
        value=settings.use_smart_routing,
        help="Detect comparisons and literature reviews and handle them differently.",
    )

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Chunks retrieved", 3, 20, settings.top_k_rerank)
    max_context_tokens = st.slider(
        "Context budget (tokens)", 1000, 8000, settings.max_context_tokens, step=500
    )

    st.divider()
    if st.button("Initialize system", type="primary", use_container_width=True):
        with st.spinner("Loading models and opening the vector store..."):
            try:
                pipeline = IngestionPipeline()
                # Reuse the pipeline's embedding model and open collection so
                # the model is loaded once, not twice.
                retrieval = RetrievalSystem(
                    embedder=pipeline.embedder, database=pipeline.database
                )
                generator = AnswerGenerator(
                    use_multi_agent=use_multi_agent,
                    use_citations=use_citations,
                    llm_provider=provider,
                    llm_model=model,
                    use_smart_routing=use_smart_routing,
                    retrieval_system=retrieval,
                )
                st.session_state.pipeline = pipeline
                st.session_state.generator = generator
                st.session_state.initialized = True
                sync_papers_from_database(pipeline)
                st.success("Ready.")
                time.sleep(0.5)
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                render_error(exc)

    st.divider()
    st.subheader("Status")
    if st.session_state.initialized:
        st.success("Ready")
        stats = st.session_state.generator.get_stats()
        col_a, col_b = st.columns(2)
        col_a.metric("Papers", stats["retrieval"]["num_papers"])
        col_b.metric("Chunks", stats["retrieval"]["num_chunks"])
        st.caption(
            f"{stats['llm']['provider']} / {stats['llm']['model']} - "
            f"{'hybrid + rerank' if stats['retrieval']['use_reranking'] else 'hybrid'}"
        )
    else:
        st.info("Not initialized")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("ResearchGPT")
st.caption("Ask questions about your papers and get answers grounded in their text.")

if not st.session_state.initialized:
    st.info("Configure the sidebar and click **Initialize system** to begin.")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Upload papers**\n\nIndex PDFs into a searchable vector store.")
    col2.markdown("**Ask questions**\n\nHybrid retrieval plus cross-encoder reranking.")
    col3.markdown("**Get citations**\n\nAnswers point back to the source passages.")
    st.stop()

upload_tab, chat_tab, library_tab = st.tabs(["Upload", "Ask", "Library"])

# --- Upload ---------------------------------------------------------------
with upload_tab:
    st.subheader("Upload research papers")
    uploaded_files = st.file_uploader(
        "PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process papers", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()
        succeeded = 0

        for index, uploaded_file in enumerate(uploaded_files, 1):
            status.text(f"Processing {uploaded_file.name} ({index}/{len(uploaded_files)})")
            progress.progress(index / len(uploaded_files))

            # Write to a temp file rather than into data/raw: uploads are
            # inputs to indexing, not artifacts that belong in the repo.
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as handle:
                    handle.write(uploaded_file.getbuffer())
                    temp_path = Path(handle.name)

                stats = st.session_state.pipeline.process_paper(
                    temp_path, paper_id=Path(uploaded_file.name).stem
                )
                st.session_state.papers.append(
                    {
                        "paper_id": stats["paper_id"],
                        "filename": uploaded_file.name,
                        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stats": stats,
                    }
                )
                succeeded += 1
                st.success(
                    f"{uploaded_file.name}: {stats['num_chunks']} chunks across "
                    f"{stats['num_sections']} sections in {stats['processing_time_seconds']}s"
                )
            except Exception as exc:  # noqa: BLE001 - per-file, keep going
                render_error(exc)
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        status.text(f"Done: {succeeded}/{len(uploaded_files)} indexed.")
        if succeeded:
            time.sleep(1)
            st.rerun()

# --- Ask ------------------------------------------------------------------
with chat_tab:
    st.subheader("Ask a question")

    if not st.session_state.papers:
        st.warning("No papers indexed yet. Upload some in the **Upload** tab.")
    else:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])
                meta: dict[str, Any] = entry.get("metadata", {})

                if meta.get("refused"):
                    st.caption("No relevant content was found, so no answer was generated.")
                else:
                    cols = st.columns(4)
                    cols[0].metric("Sources", meta.get("num_sources", 0))
                    cols[1].metric("Time", f"{meta.get('processing_time', 0):.1f}s")
                    cols[2].metric("Type", meta.get("query_type_label", "Q&A"))
                    cols[3].metric("Tokens", meta.get("total_tokens", 0))

                if entry.get("sources"):
                    with st.expander(f"Sources ({len(entry['sources'])})"):
                        for source in entry["sources"]:
                            st.markdown(
                                f"**{source.get('title', 'Unknown')}** "
                                f"- {source.get('section', 'Unknown')} "
                                f"(score {source.get('relevance_score', 0):.2f})"
                            )
                            st.caption(source.get("chunk_preview", ""))

        if st.session_state.chat_history and st.button("Clear history"):
            st.session_state.chat_history = []
            st.rerun()

        question = st.chat_input("Ask about your papers...")
        if question:
            st.session_state.pending_query = question
            st.rerun()

        if st.session_state.pending_query:
            pending = st.session_state.pending_query
            st.session_state.pending_query = None

            with st.spinner("Retrieving and generating..."):
                try:
                    response = st.session_state.generator.smart_answer(
                        user_query=pending,
                        top_k=top_k,
                        max_context_tokens=max_context_tokens,
                    )
                    st.session_state.chat_history.append(
                        {
                            "question": pending,
                            "answer": response["answer"],
                            "sources": response.get("sources", []),
                            "metadata": response.get("metadata", {}),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    render_error(exc)

# --- Library --------------------------------------------------------------
with library_tab:
    st.subheader("Indexed papers")

    if not st.session_state.papers:
        st.info("Nothing indexed yet.")
    else:
        st.caption(f"{len(st.session_state.papers)} paper(s) in the collection")

        for paper in list(st.session_state.papers):
            with st.expander(paper["filename"]):
                stats = paper.get("stats", {})
                col1, col2 = st.columns(2)
                col1.markdown(f"**ID:** `{paper['paper_id']}`")
                col1.markdown(f"**Indexed:** {paper['upload_time']}")
                col2.markdown(f"**Pages:** {stats.get('num_pages', 'n/a')}")
                col2.markdown(f"**Chunks:** {stats.get('num_chunks', 'n/a')}")

                if st.button("Remove from index", key=f"del_{paper['paper_id']}"):
                    try:
                        removed = st.session_state.pipeline.database.delete_paper(
                            paper["paper_id"]
                        )
                        st.session_state.papers.remove(paper)
                        st.success(f"Removed {removed} chunks.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        render_error(exc)
