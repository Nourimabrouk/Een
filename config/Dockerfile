# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Set up working directory
WORKDIR /app

# --- Builder stage ---
FROM base AS builder

# Install build dependencies (if any needed for pip install, e.g. numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt if present, else fallback to requirements.in or pyproject.toml if used
# (Assume requirements.txt is in the parent directory if present, else skip)
# If not present, skip this step (no requirements file in ./config, so just install minimal deps)

# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --upgrade pip && \
    pip install numpy

# Copy the service code (from ./config context)
COPY --link . /app

# --- Final stage ---
FROM base AS final

# Create non-root user
RUN addgroup --system een && adduser --system --ingroup een eenuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code from builder
COPY --from=builder /app /app

# Set environment so venv is used
ENV PATH="/app/.venv/bin:$PATH"

USER eenuser

# Default command (can be overridden)
CMD ["python", "mcp_consciousness_server.py"]
