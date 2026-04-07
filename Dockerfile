# ──────────────────────────────────────────────────────────────
# SentinelOps — Hackathon-Winning Multi-Stage Dockerfile
# Optimized for Hugging Face Spaces & OpenEnv Compliance
# ──────────────────────────────────────────────────────────────

# --- Stage 1: Builder -----------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Avoid writing .pyc files and enable caching
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install requirements to a local directory for copying
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# --- Stage 2: Runtime -----------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-interactive environment and path configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Install minimal runtime dependencies (curl for health check)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source code
COPY --chown=appuser:appuser . .

USER appuser

# Expose the standard Hugging Face Spaces port
EXPOSE 7860

# Metadata labels for OpenEnv discovery
LABEL org.openenv.environment.name="sentinelops" \
      org.openenv.environment.version="1.1.0" \
      org.openenv.environment.type="surveillance"

# Health check ensures the space is marked as 'Running' correctly
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch the FastAPI app using Uvicorn
# We use 1 worker to ensure deterministic environment state within a single session
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
