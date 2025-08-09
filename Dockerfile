# 🌟 Een Unity Mathematics API - Production Docker Container 🌟
# Multi-stage optimized build for production deployment

# Build stage
FROM python:3.11-slim-bullseye as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETARCH

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create build user
RUN useradd --create-home --shell /bin/bash builder

# Set work directory
WORKDIR /build

# Copy requirements and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim-bullseye as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/home/unity/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=4

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create unity user
RUN useradd --create-home --shell /bin/bash --uid 1000 unity

# Set work directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/unity/.local

# Copy application code
COPY --chown=unity:unity api/ ./api/
COPY --chown=unity:unity core/ ./core/
COPY --chown=unity:unity dashboards/ ./dashboards/
COPY --chown=unity:unity monitoring/ ./monitoring/
COPY --chown=unity:unity website/ ./website/

# Create required directories
RUN mkdir -p logs data static cache \
    && chown -R unity:unity logs data static cache

# Switch to unity user
USER unity

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "api.unity_api_production"]

# Development stage (optional)
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy

USER unity

# Development command
CMD ["python", "-m", "api.unity_api_production"]