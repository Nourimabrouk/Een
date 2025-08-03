# syntax=docker/dockerfile:1

# Multi-stage production-ready Dockerfile for Een Unity Mathematics

# ---- Base image with common dependencies ----
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ---- Builder stage: compile dependencies ----
FROM base AS builder

WORKDIR /app

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    # Scientific libraries
    libatlas-base-dev \
    liblapack-dev \
    libopenblas-dev \
    # Graphics libraries
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # Additional dependencies
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -e .

# ---- Development stage ----
FROM builder AS development

WORKDIR /app

# Install development dependencies
RUN pip install -e ".[dev,docs]"

# Copy source code
COPY . .

# Create directories
RUN mkdir -p logs data .cache

# Development entrypoint
CMD ["python", "-m", "uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# ---- Test stage ----
FROM builder AS test

WORKDIR /app

# Install test dependencies
RUN pip install -e ".[dev]"

# Copy source code and tests
COPY . .

# Run tests
RUN python -m pytest tests/ -v --cov=core --cov=src --cov-report=xml --cov-report=term

# ---- Production stage: minimal runtime ----
FROM base AS production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required runtime libraries
    libatlas-base-dev \
    liblapack-dev \
    libopenblas-dev \
    libfreetype6 \
    libpng16-16 \
    libjpeg62-turbo \
    libgomp1 \
    # Security updates
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID
RUN groupadd -g 1001 een && \
    useradd -r -u 1001 -g een -s /bin/false een && \
    mkdir -p /app/logs /app/data /app/.cache && \
    chown -R een:een /app

# Copy virtual environment from builder
COPY --from=builder --chown=een:een /opt/venv /opt/venv

# Copy application code
COPY --chown=een:een . .

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    ENVIRONMENT="production"

# Security: Set read-only root filesystem
RUN chmod -R 755 /app && \
    find /app -type d -exec chmod 755 {} + && \
    find /app -type f -exec chmod 644 {} +

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Switch to non-root user
USER een

# Expose ports
EXPOSE 8000 8050

# Production entrypoint with proper signal handling
ENTRYPOINT ["python", "-m", "gunicorn"]
CMD ["api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5"]
