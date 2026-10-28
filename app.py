"""JobMatch Streamlit UI.

This file only handles presentation: input widgets, progress feedback, and
rendering results. All retrieval/AI logic lives in the jobmatch package
(find_similar_jobs, analyze_cv) - this file just calls into it.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from jobmatch.analysis import CVAnalysis, analyze_cv
from jobmatch.cv_parser import CVParseError, extract_text, normalize_whitespace
from jobmatch.errors import ConfigurationError
from jobmatch.retrieval import MatchedJob, find_similar_jobs

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_CV_PATH = PROJECT_ROOT / "data" / "cv.txt"
MIN_WORDS = 50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jobmatch.app")

CUSTOM_CSS = """
<style>
.jm-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0 1rem 0;
}
.jm-chip {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.45);
    font-size: 0.85rem;
    font-weight: 600;
    white-space: nowrap;
}
.jm-before, .jm-after {
    padding: 0.65rem 0.85rem;
    border-radius: 0.6rem;
    font-size: 0.92rem;
    line-height: 1.45;
    min-height: 3.5rem;
}
.jm-before {
    background: rgba(148, 163, 184, 0.16);
    border: 1px solid rgba(148, 163, 184, 0.4);
}
.jm-after {
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(34, 197, 94, 0.45);
}
.jm-section-gap {
    margin-top: 1.5rem;
}
</style>
"""


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def _warm_embedding_model():
    # Thin UI-layer cache around jobmatch's own cached loader: this just
    # ensures Streamlit doesn't treat the (already-cheap-after-first-load)
    # call as something to repeat on every rerun/widget interaction.
    from jobmatch.embeddings import get_embedding_model

    return get_embedding_model()


def _load_sample_cv() -> None:
    if SAMPLE_CV_PATH.exists():
        st.session_state["cv_paste_text"] = SAMPLE_CV_PATH.read_text(encoding="utf-8")
        st.session_state["active_cv_source"] = "paste"
    else:
        st.session_state["_sample_missing"] = True


def _mark_paste_active() -> None:
    st.session_state["active_cv_source"] = "paste"


def _mark_upload_active() -> None:
    st.session_state["active_cv_source"] = "upload"


def _render_cv_input() -> str:
    """Render the upload/paste tabs and return whichever text the user last worked with."""
    upload_tab, paste_tab = st.tabs(["📄 Upload file", "✍️ Paste text"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload your CV",
            type=["pdf", "docx", "txt"],
            key="cv_file_uploader",
        )
        if uploaded is not None:
            file_identity = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("_processed_file_id") != file_identity:
                st.session_state["_processed_file_id"] = file_identity
                try:
                    # .getvalue() reads the in-memory buffer Streamlit already
                    # holds for the upload - the file is never written to disk.
                    extracted = extract_text(uploaded.getvalue(), uploaded.name)
                except CVParseError as exc:
                    st.error(
                        f"Couldn't read **{uploaded.name}**: {exc} "
                        "You can also switch to the 'Paste text' tab and paste it directly."
                    )
                else:
                    st.session_state["cv_upload_text"] = extracted
                    st.session_state["active_cv_source"] = "upload"

        st.text_area(
            "Review extracted text (edit if needed)",
            key="cv_upload_text",
            height=350,
            placeholder="Upload a PDF, DOCX, or TXT file above to extract its text here.",
            on_change=_mark_upload_active,
        )

    with paste_tab:
        if st.button("📄 Load sample from data/cv.txt"):
            _load_sample_cv()
        if st.session_state.pop("_sample_missing", False):
            st.warning(f"No sample CV found at {SAMPLE_CV_PATH}.")

        st.text_area(
            "Paste your CV",
            key="cv_paste_text",
            height=350,
            placeholder=(
                "Paste the full text of your CV here (summary, experience, education, skills)..."
            ),
            on_change=_mark_paste_active,
        )

    active_source = st.session_state.get("active_cv_source", "paste")
    key = "cv_upload_text" if active_source == "upload" else "cv_paste_text"
    return st.session_state.get(key, "")


def _show_error(status, status_label: str, friendly_message: str, exc: Exception) -> None:
    """Show a friendly error plus a 'Technical details' expander, log the traceback, and stop.

    Must be called from inside the `except ... as exc:` block it's reporting,
    so logger.exception() can still capture the active traceback.
    """
    status.update(label=status_label, state="error")
    logger.exception("JobMatch pipeline error (%s): %s", status_label, friendly_message)
    st.error(friendly_message)
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


def _run_analysis(cv_text: str, match_count: int) -> None:
    with st.status("Analyzing your CV...", expanded=True) as status:
        status.update(label="Finding matching jobs...")
        _warm_embedding_model()
        try:
            jobs = find_similar_jobs(cv_text, match_count=match_count)
        except ConfigurationError as exc:
            _show_error(status, "Configuration error", f"Configuration error: {exc}", exc)
        except Exception as exc:
            _show_error(
                status,
                "Could not reach Supabase",
                "Could not fetch matching jobs from Supabase. Please check your Supabase "
                "project/credentials and try again.",
                exc,
            )

        if not jobs:
            status.update(label="No matches found", state="error")
            st.warning(
                "No matching jobs were found. Make sure scripts/ingest_jobs.py "
                "has been run to populate the jobs table."
            )
            st.stop()

        status.update(label="Analyzing with AI...")
        try:
            analysis = analyze_cv(cv_text, jobs)
        except ConfigurationError as exc:
            _show_error(status, "Configuration error", f"Configuration error: {exc}", exc)
        except Exception as exc:
            _show_error(
                status,
                "AI analysis failed",
                "The AI analysis step failed, possibly due to a temporary issue with the "
                "AI provider. Please wait a moment and try again.",
                exc,
            )

        status.update(label="Analysis complete", state="complete")

    st.session_state["matched_jobs"] = jobs
    st.session_state["cv_analysis"] = analysis


def _render_matched_jobs(jobs: list[MatchedJob]) -> None:
    st.subheader("Matched Jobs")
    for job in jobs:
        with st.container(border=True):
            st.markdown(f"#### {job.title}")
            st.caption(f"{job.company or 'Unknown company'} · {job.location or 'Unknown location'}")
            st.progress(
                min(max(job.similarity, 0.0), 1.0),
                text=f"{job.similarity * 100:.1f}% match",
            )
            if job.url:
                st.link_button("View original posting", job.url)
            with st.expander("Full description"):
                st.text(job.description)


def _render_missing_technologies(analysis: CVAnalysis) -> None:
    st.subheader("Missing Technologies")
    techs = sorted(analysis.missing_technologies, key=lambda t: t.mentioned_in, reverse=True)

    if not techs:
        st.info("No clear technology gaps found between your CV and the matched jobs.")
        return

    chips_html = "".join(
        f'<span class="jm-chip">{html.escape(tech.name)} · {tech.mentioned_in}</span>'
        for tech in techs
    )
    st.markdown(f'<div class="jm-chip-row">{chips_html}</div>', unsafe_allow_html=True)

    chart_df = pd.DataFrame(
        {"Technology": [t.name for t in techs], "Jobs mentioning": [t.mentioned_in for t in techs]}
    ).set_index("Technology")
    st.bar_chart(chart_df, horizontal=True)

    st.dataframe(
        [
            {
                "Technology": t.name,
                "Mentioned in": t.mentioned_in,
                "Job titles": ", ".join(t.job_titles),
            }
            for t in techs
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_wording_improvements(analysis: CVAnalysis) -> None:
    st.subheader("Wording Improvements")
    for i, improvement in enumerate(analysis.wording_improvements, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}.**")
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("**Before**")
                st.markdown(
                    f'<div class="jm-before">{html.escape(improvement.original)}</div>',
                    unsafe_allow_html=True,
                )
            with col_after:
                st.markdown("**After**")
                st.markdown(
                    f'<div class="jm-after">{html.escape(improvement.improved)}</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"💡 {improvement.reason}")


def _render_results(jobs: list[MatchedJob], analysis: CVAnalysis) -> None:
    st.markdown('<div class="jm-section-gap"></div>', unsafe_allow_html=True)
    st.header("Results")

    avg_similarity = sum(job.similarity for job in jobs) / len(jobs)
    col1, col2, col3 = st.columns(3)
    col1.metric("Jobs matched", len(jobs))
    col2.metric("Avg. similarity", f"{avg_similarity * 100:.1f}%")
    col3.metric("Missing technologies", len(analysis.missing_technologies))

    st.markdown('<div class="jm-section-gap"></div>', unsafe_allow_html=True)
    _render_matched_jobs(jobs)

    st.markdown('<div class="jm-section-gap"></div>', unsafe_allow_html=True)
    _render_missing_technologies(analysis)

    st.markdown('<div class="jm-section-gap"></div>', unsafe_allow_html=True)
    _render_wording_improvements(analysis)


def main() -> None:
    st.set_page_config(page_title="JobMatch", page_icon="🎯", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🎯 JobMatch")
    st.caption(
        "Paste your CV to find the junior dev jobs you're the best semantic match for, "
        "and get AI-powered feedback on skill gaps and wording."
    )

    # Normalizing here (not in the input widgets themselves) keeps the text
    # areas showing exactly what the user typed/edited, while the pipeline
    # always analyzes whitespace-cleaned text regardless of upload vs. paste.
    cv_text = normalize_whitespace(_render_cv_input())

    match_count = st.slider("Number of matched jobs", min_value=3, max_value=10, value=5)

    analyze_clicked = st.button("🔍 Analyze my CV", type="primary", use_container_width=True)

    if analyze_clicked:
        word_count = len(cv_text.split())
        if not cv_text.strip() or word_count < MIN_WORDS:
            st.warning(
                f"Please upload or paste your full CV (at least ~{MIN_WORDS} words) before "
                f"analyzing. Currently: {word_count} word(s)."
            )
        else:
            _run_analysis(cv_text, match_count)

    if "matched_jobs" in st.session_state and "cv_analysis" in st.session_state:
        _render_results(st.session_state["matched_jobs"], st.session_state["cv_analysis"])


if __name__ == "__main__":
    main()
