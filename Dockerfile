# Multi-stage build for PlanBook AI
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    NLTK_DATA=/usr/local/nltk_data

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # Image processing
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PDF processing
    poppler-utils \
    # Network tools
    curl \
    wget \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create NLTK download script and run it
RUN echo "import nltk\nimport ssl\ntry:\n    _create_unverified_https_context = ssl._create_unverified_context\nexcept AttributeError:\n    pass\nelse:\n    ssl._create_default_https_context = _create_unverified_https_context\nnltk.download('punkt', download_dir='/usr/local/nltk_data')\nnltk.download('punkt_tab', download_dir='/usr/local/nltk_data')\nnltk.download('stopwords', download_dir='/usr/local/nltk_data')" > /tmp/download_nltk.py && \
    python /tmp/download_nltk.py && \
    rm /tmp/download_nltk.py

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p temp_uploads logs data exports && \
    chmod 755 temp_uploads logs data exports

# Expose ports
EXPOSE 8000 5555

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - can be overridden
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
