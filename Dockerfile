FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies for geospatial ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Install deps first to leverage Docker layer cache
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --group dev || uv sync --group dev

# Ensure torch and ML dependencies are available in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source, tests, and ml Python modules (data/training dirs excluded via .dockerignore)
COPY src ./src
COPY tests ./tests
COPY ml ./ml
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
