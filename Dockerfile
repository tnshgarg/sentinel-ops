# ──────────────────────────────────────────────────────────────
# SentinelOps — Dockerfile
# Multi-stage build for production deployment on Hugging Face Spaces
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Install system deps ──────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python deps ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
COPY . .

# ── Create non-root user (HF Spaces requirement) ─────────────
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Expose port ──────────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Launch ───────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
