# syntax=docker/dockerfile:1.6
ARG INCLUDE_MODEL=false

# ─── Builder stage: install Python deps via uv ──────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev || uv sync --no-dev


# ─── Runtime stage: slim Python + venv from builder ─────────────────
FROM python:3.12-slim-bookworm AS runtime
ARG INCLUDE_MODEL
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    DEMO_MODE=true APP_ENV=production \
    CORS_ORIGINS="http://localhost,http://localhost:8000,http://localhost:7860"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 libgeos-c1v5 libproj25 \
    && rm -rf /var/lib/apt/lists/*

# Bring the resolved venv from the builder
COPY --from=builder /app/.venv /app/.venv

# Application code
COPY src ./src
COPY ml/inference ./ml/inference
COPY ml/models ./ml/models

# Snapshot — copied if present (allowed to be missing in early CI builds).
# The trailing slash and absent files are tolerated by buildkit.
COPY data ./data

# Checkpoint metadata + the heavy weights file. We always copy the
# directory to keep the JSON metrics, then conditionally strip the
# 57 MB checkpoint when INCLUDE_MODEL is not "true".
RUN mkdir -p ml/training/checkpoints
COPY ml/training/checkpoints/ ml/training/checkpoints/
RUN if [ "$INCLUDE_MODEL" != "true" ]; then \
        rm -f ml/training/checkpoints/best_model.pth; \
    fi

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
