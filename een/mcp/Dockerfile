# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base

ENV POETRY_VERSION=1.8.5 \
    POETRY_CACHE_DIR=/root/.cache/pypoetry \
    PIP_CACHE_DIR=/root/.cache/pip

WORKDIR /app

# --- Builder stage: install dependencies and build venv ---
FROM base AS builder

# Install build dependencies
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install "poetry==${POETRY_VERSION}"

# Copy only dependency files first for better cache utilization
COPY --link ../../pyproject.toml ../../poetry.lock ./

# Install dependencies into a virtual environment in-project
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

# Copy the service code (een/mcp) and the een package code
COPY --link . ./een/mcp
COPY --link ../../een ./een

# --- Final stage: minimal runtime image ---
FROM base AS final

WORKDIR /app

# Create a non-root user
RUN groupadd -r appuser && useradd -m -g appuser appuser

# Copy the built app and venv from builder
COPY --link --from=builder /app /app

# Set permissions
RUN chown -R appuser:appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

# Default entrypoint (can be overridden)
# Example: python -m een.mcp.unity_server
ENTRYPOINT ["python", "-m", "een.mcp.unity_server"]

# To run a different server, override the entrypoint or CMD as needed.
