# =============================================================
# File: src/verses_rag/web/app.py
# =============================================================
"""
Streamlit UI for the Verses RAG application (SPEC §4.10).

Connects to the FastAPI backend via HTTP. Streams node-progress events
and the final answer using the /query/stream SSE endpoint.

Run (with the FastAPI server already running on port 8000):
    uv run streamlit run src/verses_rag/web/app.py

Or with a custom API URL:
    API_URL=http://localhost:8000 uv run streamlit run src/verses_rag/web/app.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterator

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- page config -------------------------------------------------------------

st.set_page_config(
    page_title = "Verses RAG",
    page_icon  = "📖",
    layout     = "wide",
)


# --- API helpers -------------------------------------------------------------

def _get_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def _stream_query(query: str, use_decomposition: bool) -> Iterator[dict]:
    """Yield parsed SSE event dicts from the /query/stream endpoint."""
    try:
        with requests.post(
            f"{API_URL}/query/stream",
            json    = {"query": query, "use_decomposition": use_decomposition},
            stream  = True,
            timeout = 120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    yield json.loads(line[6:])
    except requests.RequestException as e:
        yield {"event": "error", "data": {"detail": str(e)}}


# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("📖 Verses RAG")
    st.caption("Scripture & article search powered by RAG")
    st.divider()

    # Health status
    st.subheader("API Status")
    health = _get_health()
    if health is None:
        st.error("API unreachable — is the server running?")
    else:
        status_colour = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}.get(
            health.get("status", "down"), "🔴"
        )
        st.write(f"{status_colour} **{health.get('status', 'unknown').upper()}**")

        if p := health.get("primary"):
            ms = f"{p['latency_ms']:.0f}ms" if p.get("latency_ms") else "—"
            st.caption(f"Primary: {p['provider']}  ({ms})")
        if b := health.get("backup"):
            ms = f"{b['latency_ms']:.0f}ms" if b.get("latency_ms") else "—"
            st.caption(f"Backup:  {b['provider']}  ({ms})")

    st.divider()

    # Query options
    st.subheader("Options")
    use_decomposition = st.toggle(
        "Auto-decompose multi-part queries",
        value   = True,
        help    = "Splits 'Compare Genesis and Romans on sin' into two sub-queries.",
    )

    st.divider()

    # History
    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("Recent queries")
    if st.session_state.history:
        for past in reversed(st.session_state.history[-8:]):
            verdict_icon = "✓" if past["verdict"] == "pass" else "—"
            if st.button(
                f"{verdict_icon} {past['query'][:40]}",
                key     = f"hist_{past['ts']}",
                use_container_width = True,
            ):
                st.session_state.rerun_query = past["query"]
    else:
        st.caption("No queries yet.")

    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# --- main area ---------------------------------------------------------------

st.header("Ask a question")

# Example dropdown — selecting one fills the text box on the next rerun.
example_queries = [
    "What does the Bible say about God's love?",
    "The LORD is my shepherd",
    "Compare what Genesis and Romans say about sin",
    "What does Proverbs say about wisdom?",
]

def _on_example_change():
    """Copy the chosen example into the query box, then reset the dropdown."""
    chosen = st.session_state.get("example_select", "— pick one —")
    if chosen != "— pick one —":
        st.session_state.query_input = chosen
        st.session_state.example_select = "— pick one —"

st.selectbox(
    "Try an example",
    ["— pick one —"] + example_queries,
    key       = "example_select",
    on_change = _on_example_change,
)

# History click pre-fills the box too (handled before the widget renders).
if "rerun_query" in st.session_state:
    st.session_state.query_input = st.session_state.pop("rerun_query")

query = st.text_input(
    "Query",
    key         = "query_input",
    placeholder = "e.g. What does Romans say about grace?",
)

ask = st.button("Ask", type="primary")

st.divider()

# --- run query ---------------------------------------------------------------

if ask and query.strip():
    status_box  = st.empty()
    answer_box  = st.empty()
    verdict_box = st.empty()
    cite_box    = st.empty()

    citations: list[dict] = []
    answer    = ""
    verdict   = "abstain"
    latency_s = 0.0
    sub_queries: list[str] = []

    for event in _stream_query(query.strip(), use_decomposition):
        if not isinstance(event, dict):
            continue
        etype = event.get("event")
        data  = event.get("data")

        if etype == "status":
            status_box.info(f"⏳ {data}")

        elif etype == "answer":
            answer = data or ""
            status_box.empty()
            if answer:
                answer_box.markdown(answer)

        elif etype == "citation":
            if isinstance(data, dict):
                citations.append(data)

        elif etype == "done":
            if isinstance(data, dict):
                verdict   = data.get("verdict", "abstain")
                latency_s = data.get("latency_s", 0.0)
            status_box.empty()

            # Verdict badge
            if verdict == "pass":
                verdict_box.success(f"✓ Answer verified  ·  {latency_s:.1f}s")
            else:
                verdict_box.warning(f"— No sufficient context found  ·  {latency_s:.1f}s")

            # Citations
            if citations:
                with cite_box.expander(f"📚 {len(citations)} source(s)", expanded=False):
                    for c in citations:
                        score = c.get("rerank_score", 0.0)
                        src   = "📖" if c.get("source_type") == "bible" else "📄"
                        st.markdown(f"**{src} {c.get('ref', '')}**  `score {score:.2f}`")
                        st.caption(c.get("content_snippet", ""))
                        st.divider()

        elif etype == "error":
            status_box.empty()
            detail = data.get("detail", "unknown error") if isinstance(data, dict) else str(data)
            st.error(f"Error: {detail}")
            break

    # Save to history
    if query.strip():
        st.session_state.history.append({
            "query":   query.strip(),
            "verdict": verdict,
            "ts":      time.time(),
        })

elif ask and not query.strip():
    st.warning("Please enter a question.")


# --- empty state -------------------------------------------------------------

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem; color: #888;">
            <h3>Ask anything about scripture or the article corpus</h3>
            <p>Type a question above, pick an example, or click a recent query in the sidebar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )