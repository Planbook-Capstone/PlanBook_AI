# Multi-stage build for PlanBook AI
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

# Set environment variables for build stage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache and temporary files
    pip cache purge && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    NLTK_DATA=/usr/local/nltk_data \
    PATH="/opt/venv/bin:$PATH"

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y \
    # Minimal image processing (only what's needed)
    libglib2.0-0 \
    libgomp1 \
    # PDF processing
    poppler-utils \
    # Network tools for health check
    curl \
    # Clean up aggressively
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && rm -rf /var/cache/apt/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Download only essential NLTK data
RUN python -c "import nltk; import ssl; \
    try: _create_unverified_https_context = ssl._create_unverified_context; \
    except AttributeError: pass; \
    else: ssl._create_default_https_context = _create_unverified_https_context; \
    nltk.download('punkt', download_dir='/usr/local/nltk_data'); \
    nltk.download('stopwords', download_dir='/usr/local/nltk_data')" \
    && rm -rf /tmp/* /var/tmp/*

# Copy application code selectively (excluding files in .dockerignore)
COPY app/ ./app/
COPY google-credentials.json* ./

# Create necessary directories with proper permissions
RUN mkdir -p temp_uploads logs data exports && \
    chmod 755 temp_uploads logs data exports && \
    # Clean up any Python cache files
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + || true

# Create non-root user for security
RUN groupadd -r planbook && useradd -r -g planbook planbook && \
    chown -R planbook:planbook /app
USER planbook

# Expose ports
EXPOSE 8000 5555

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - can be overridden
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
