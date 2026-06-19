# =============================================================
# File: src/verses_rag/llm/router.py
# =============================================================
"""
Per-role LLM router (SPEC §4.8, D5, D7).

Single entry point for all LLM access:

    llm = get_llm("grade")      # returns OpenAI with Anthropic fallback
    llm = get_llm("verify")     # same
    llm = get_llm("generation") # returns local Ollama, no fallback

Two reliability layers for judge roles:
  1. Reactive fallback  — LangChain's .with_fallbacks() catches any exception
                          from OpenAI and routes the same call to Anthropic.
                          Callers never see the switch.
  2. Proactive health check — check_judge_health() makes a cheap probe call and
                              returns HealthStatus for each provider. Call at
                              startup or on a schedule to detect degradation
                              before it hits a real query.

Roles (§4.8.1):
    "grade"      — relevance grading (CRAG) — judge
    "verify"     — faithfulness check       — judge
    "route"      — query routing/decomp     — judge
    "classify"   — doc classification LLM fallback — judge
    "generation" — grounded answer drafting — local Ollama

Deps:
    uv add langchain-openai langchain-anthropic langchain-ollama

API keys (via .env or environment):
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...

Run self-check:
    uv run python -m verses_rag.llm.router
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal, Optional

from pydantic import SecretStr

log = logging.getLogger("llm.router")

# Roles that go through the judge (OpenAI primary / Anthropic backup).
_JUDGE_ROLES = {"grade", "verify", "route", "classify"}

Role = Literal["grade", "verify", "route", "classify", "generation"]


# ---------------------------------------------------------------------------
# Health check types
# ---------------------------------------------------------------------------

class ProviderStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"   # responded, but latency > threshold
    DOWN = "down"           # exception or timeout


@dataclass
class ProviderHealth:
    provider: str
    status: ProviderStatus
    latency_ms: Optional[float]   # None if DOWN
    error: Optional[str]          # None if OK/DEGRADED

    def __str__(self) -> str:
        if self.status == ProviderStatus.OK:
            return f"{self.provider}: OK ({self.latency_ms:.0f}ms)"
        if self.status == ProviderStatus.DEGRADED:
            return f"{self.provider}: DEGRADED ({self.latency_ms:.0f}ms — above threshold)"
        return f"{self.provider}: DOWN — {self.error}"


@dataclass
class JudgeHealth:
    primary: ProviderHealth
    backup: ProviderHealth

    @property
    def any_available(self) -> bool:
        return any(
            p.status != ProviderStatus.DOWN
            for p in (self.primary, self.backup)
        )

    def __str__(self) -> str:
        return f"  primary  {self.primary}\n  backup   {self.backup}"


# ---------------------------------------------------------------------------
# Internal: build LLM instances
# ---------------------------------------------------------------------------

def _make_openai(model: str, timeout: float, max_tokens: int, api_key: Optional[SecretStr] = None):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=0,
        max_completion_tokens=max_tokens,
        timeout=timeout,
        max_retries=1,
        api_key=api_key,                    # SecretStr | None; None -> reads OPENAI_API_KEY from env
    )


def _make_anthropic(model: str, timeout: float, max_tokens: int, api_key: Optional[SecretStr] = None):
    from langchain_anthropic import ChatAnthropic
    kwargs: dict = dict(
        model_name=model,
        temperature=0,
        max_tokens_to_sample=max_tokens,
        timeout=timeout,
        max_retries=1,
        stop=None,
    )
    if api_key is not None:
        kwargs["api_key"] = api_key         # only pass if present; else reads ANTHROPIC_API_KEY from env
    return ChatAnthropic(**kwargs)


def _make_ollama(model: str, base_url: str, temperature: float, max_tokens: int):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llm(role: Role, settings=None):
    """Return a LangChain chat model for the given role.

    Judge roles (grade/verify/route/classify): returns OpenAI with Anthropic
    wired as a fallback via .with_fallbacks(). Any exception from OpenAI —
    rate limit, 500, timeout — automatically retries the same call against
    Anthropic. The returned object is a standard LangChain Runnable.

    Generation role: returns local Ollama directly; no fallback (if Ollama
    is unreachable, that's an infrastructure problem, not a provider issue).

    Args:
        role:     one of the Role literals above.
        settings: optional Settings override; defaults to get_settings().
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    if role == "generation":
        cfg = settings.generation
        return _make_ollama(cfg.model, cfg.base_url, cfg.temperature, cfg.max_tokens)

    if role in _JUDGE_ROLES:
        cfg = settings.judge
        primary = _make_openai(cfg.primary_model, cfg.timeout, cfg.max_tokens, settings.openai_api_key)
        backup = _make_anthropic(cfg.backup_model, cfg.timeout, cfg.max_tokens, settings.anthropic_api_key)
        # .with_fallbacks() is a standard LangChain Runnable method.
        # On any exception from primary it invokes backup with the same input.
        return primary.with_fallbacks(
            [backup],
            exceptions_to_handle=(Exception,),   # broad: covers rate limits, 500s, timeouts
        )

    raise ValueError(f"Unknown role {role!r}. Valid roles: {_JUDGE_ROLES | {'generation'}}")


def check_judge_health(settings=None) -> JudgeHealth:
    """Probe both judge providers with a cheap single-token call.

    Returns a JudgeHealth with status OK / DEGRADED / DOWN for each.
    DEGRADED means the provider responded but latency exceeded
    settings.judge.degraded_latency_ms.

    Typical use: call at app startup and log the result. If primary is DOWN,
    the reactive fallback in get_llm() still works — this gives you early
    warning before a real query fails.
    """
    if settings is None:
        from verses_rag.config.settings import get_settings
        settings = get_settings()

    cfg = settings.judge
    threshold_ms = cfg.degraded_latency_ms

    def _probe(make_fn, label: str) -> ProviderHealth:
        try:
            llm = make_fn()
            start = time.perf_counter()
            # Minimal: 1 token max, trivial prompt — just checking the API responds.
            llm.invoke("Reply with the single word: ok", config={"max_tokens": 5})
            latency_ms = (time.perf_counter() - start) * 1000
            status = (
                ProviderStatus.OK
                if latency_ms <= threshold_ms
                else ProviderStatus.DEGRADED
            )
            return ProviderHealth(label, status, latency_ms, None)
        except Exception as e:
            log.warning("health probe failed for %s: %s", label, e)
            return ProviderHealth(label, ProviderStatus.DOWN, None, str(e))

    primary_health = _probe(
        lambda: _make_openai(cfg.primary_model, cfg.timeout, cfg.max_tokens, settings.openai_api_key),
        f"openai/{cfg.primary_model}",
    )
    backup_health = _probe(
        lambda: _make_anthropic(cfg.backup_model, cfg.timeout, cfg.max_tokens, settings.anthropic_api_key),
        f"anthropic/{cfg.backup_model}",
    )

    return JudgeHealth(primary=primary_health, backup=backup_health)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def main():
    print("=== LLM router self-check ===\n")

    # 1. Health check — load via settings so .env is respected
    from verses_rag.config.settings import get_settings
    settings = get_settings()
    has_openai = settings.openai_api_key is not None
    has_anthropic = settings.anthropic_api_key is not None

    if has_openai or has_anthropic:
        print("Probing judge providers…")
        health = check_judge_health(settings)
        print(health)
        if not health.any_available:
            print("\nWARNING: both judge providers are DOWN — grading/verify will fail.")
    else:
        print("No API keys found in settings — skipping live health check.")
        print("Add OPENAI_API_KEY and ANTHROPIC_API_KEY to your .env file.\n")

    # 2. Verify get_llm() constructs without error (no API call made here)
    print("\nBuilding LLM handles…")
    for role in ("grade", "verify", "route", "classify"):
        llm = get_llm(role, settings)
        print(f"  {role:12} -> {type(llm).__name__}")

    # Ollama check — only if running
    try:
        llm = get_llm("generation", settings)
        print(f"  {'generation':12} -> {type(llm).__name__}")
    except Exception as e:
        print(f"  generation   -> skipped (Ollama not running? {e})")

    print("\nDone.")


if __name__ == "__main__":
    main()