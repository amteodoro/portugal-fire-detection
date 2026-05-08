# syntax=docker/dockerfile:1.6
ARG INCLUDE_MODEL=false
ARG INCLUDE_SNAPSHOT=true

# ─── Optional snapshot stage ─────────────────────────────────────────
# `INCLUDE_SNAPSHOT=true` (the default, used by the Hugging Face Space
# build) bundles `data/` and `ml/training/checkpoints/`. CI sets it to
# `false` so origin builds — where those directories are gitignored —
# still pass. BuildKit only materialises the reachable stage, so the
# `snapshot-true` COPYs never run when the arg is `false`.
FROM busybox AS snapshot-false
RUN mkdir -p /snapshot/data /snapshot/checkpoints

FROM busybox AS snapshot-true
RUN mkdir -p /snapshot/data /snapshot/checkpoints
COPY data /snapshot/data
COPY ml/training/checkpoints /snapshot/checkpoints

FROM snapshot-${INCLUDE_SNAPSHOT} AS snapshot

# ─── Builder stage: install Python deps via uv ──────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev || uv sync --no-dev


# ─── Test stage: same base as builder but with dev deps + tests ─────
# Used by CI only — never reaches the runtime image.
FROM builder AS test
RUN uv sync --frozen --group dev
COPY src ./src
COPY ml ./ml
COPY tests ./tests
COPY alembic ./alembic
COPY alembic.ini ./
ENV PYTHONPATH=/app
CMD ["uv", "run", "pytest", "--tb=short", "-q"]


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

COPY --from=builder /app/.venv /app/.venv

COPY src ./src
COPY ml/inference ./ml/inference
COPY ml/models ./ml/models

COPY --from=snapshot /snapshot/data ./data
COPY --from=snapshot /snapshot/checkpoints ./ml/training/checkpoints

RUN if [ "$INCLUDE_MODEL" != "true" ]; then \
        rm -f ml/training/checkpoints/best_model.pth; \
    fi

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
