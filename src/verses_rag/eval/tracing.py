# =============================================================
# File: src/verses_rag/eval/tracing.py
# =============================================================
"""
LangSmith tracing integration (SPEC §4.9, §4.1).

LangChain auto-traces everything — every LLM call, every graph node,
every retrieval step — when four environment variables are set. This
module sets them from settings so tracing is:
  - opt-in (disabled unless langsmith.enabled = True in .env)
  - configured in one place
  - a graceful no-op in CI or environments without a key

Required .env additions to enable:
    LANGSMITH__ENABLED=true
    LANGSMITH__API_KEY=ls__...
    LANGSMITH__PROJECT=verses-rag          # optional, default "verses-rag"

Usage — call once at app startup or before run_eval():
    from verses_rag.eval.tracing import configure_tracing
    configure_tracing()   # reads settings, sets env vars, returns bool

Usage — tag individual runs for grouping in the LangSmith UI:
    from verses_rag.eval.tracing import make_run_config
    graph.invoke(state, config=make_run_config("eval-S01", tags=["eval"]))
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("eval.tracing")


def configure_tracing(settings=None) -> bool:
    """Set LangChain env vars from settings to activate LangSmith tracing.

    Returns True if tracing is now active, False if disabled or misconfigured.
    Safe to call multiple times — uses setdefault so existing env vars win.
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    ls = settings.langsmith
    if not ls.enabled:
        return False

    if not ls.api_key:
        log.warning(
            "LangSmith enabled but LANGSMITH__API_KEY not set — tracing inactive"
        )
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY",    ls.api_key.get_secret_value())
    os.environ.setdefault("LANGCHAIN_PROJECT",    ls.project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT",   ls.endpoint)

    log.info(
        "LangSmith tracing active → project=%r  endpoint=%s",
        ls.project, ls.endpoint,
    )
    return True


def make_run_config(
    run_name: str,
    tags:     list[str] | None       = None,
    metadata: dict[str, Any] | None  = None,
) -> dict[str, Any]:
    """Build a LangChain RunnableConfig dict for a single traced run.

    Pass the result as `config=` to graph.invoke() or any LangChain call.

    Args:
        run_name: Human-readable name shown in LangSmith (e.g. "eval-S01").
        tags:     String tags for filtering in the UI (e.g. ["eval", "sample"]).
        metadata: Arbitrary key-value pairs attached to the trace.

    Example:
        graph.invoke(state, config=make_run_config(
            "eval-K06", tags=["eval", "kjv"], metadata={"threshold": -8.0}
        ))
    """
    cfg: dict[str, Any] = {"run_name": run_name}
    if tags:
        cfg["tags"] = tags
    if metadata:
        cfg["metadata"] = metadata
    return cfg


# --- self-check --------------------------------------------------------------

def main():
    from verses_rag.config.settings import get_settings
    s = get_settings()

    print("=== tracing self-check ===\n")
    print(f"langsmith.enabled  = {s.langsmith.enabled}")
    print(f"langsmith.project  = {s.langsmith.project}")
    print(f"langsmith.endpoint = {s.langsmith.endpoint}")
    print(f"langsmith.api_key  = {'set' if s.langsmith.api_key else 'not set'}")

    active = configure_tracing(s)
    print(f"\nconfigure_tracing() → {active}")
    if active:
        print(f"  LANGCHAIN_PROJECT  = {os.environ.get('LANGCHAIN_PROJECT')}")
        print(f"  LANGCHAIN_ENDPOINT = {os.environ.get('LANGCHAIN_ENDPOINT')}")

    cfg = make_run_config("eval-S01", tags=["eval", "sample"],
                          metadata={"threshold": -8.0})
    print(f"\nmake_run_config() → {cfg}")


if __name__ == "__main__":
    main()