# =============================================================
# File: Dockerfile
# =============================================================
# Single image that runs either the FastAPI backend or the Streamlit UI.
# Which one is selected by the `command:` in docker-compose.
#
# Build context is the repo root. Uses uv for fast, reproducible installs.
#
# Models (BGE-large, cross-encoder) are NOT baked in — they download to a
# mounted HF cache volume on first run (see docker-compose). Keeps the image
# lean (~1.5GB vs ~3GB) and avoids re-downloading on rebuild.

FROM python:3.11-slim

# System deps: build tools for any wheels that need compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- dependency layer (cached unless lockfile changes) ---
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# --- project source ---
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# HF models cache to this dir; mounted as a volume so downloads persist.
ENV HF_HOME=/app/.hf_cache
ENV PYTHONUNBUFFERED=1

# Default port (overridden per-service in compose).
EXPOSE 8000 8501

# No default CMD — compose specifies the command per service.