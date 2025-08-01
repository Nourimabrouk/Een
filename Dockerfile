# syntax=docker/dockerfile:1

# ---- Base image (slim, for both build and final stages) ----
FROM python:3.11-slim AS base

# ---- Builder stage: install dependencies in a venv ----
FROM base AS builder
WORKDIR /app

# System dependencies for numpy, pandas, matplotlib, seaborn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libatlas-base-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and utilities
COPY --link requirements.txt ./
COPY --link src/utils/utils_helper.py ./utils_helper.py

# Create venv and install dependencies using pip cache
RUN python -m venv .venv \
    && .venv/bin/pip install --upgrade pip \
    && .venv/bin/pip install -r requirements.txt

# Copy the rest of the app (excluding files via .dockerignore)
COPY --link scripts/run_demo.py ./run_demo.py

# ---- Final stage: minimal runtime image ----
FROM base AS final
WORKDIR /app

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy venv and app code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/utils_helper.py ./
COPY --from=builder /app/run_demo.py ./

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

# Default command: run the demo script
CMD ["python", "run_demo.py"]
