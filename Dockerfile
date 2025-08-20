# ===== Stage 1: builder =====
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/opt/nltk_data

RUN apt-get update && apt-get install -y build-essential gcc g++ \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Tải sẵn NLTK data vào /opt/nltk_data (không cần try/except)
RUN mkdir -p /opt/nltk_data \
 && python -m nltk.downloader -d /opt/nltk_data punkt stopwords

# (tuỳ chọn) dọn rác
RUN find /opt/venv -name "*.pyc" -delete \
 && find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true

# ===== Stage 2: runtime =====
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive NLTK_DATA=/usr/local/nltk_data \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y libglib2.0-0 libgomp1 poppler-utils curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/*

# Copy venv + NLTK data đã tải sẵn
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /usr/local/nltk_data

WORKDIR /app

# KHÔNG cần bất kỳ lệnh python -c nào nữa
# (loại bỏ hoàn toàn bước RUN python -c ... gây SyntaxError)

COPY app/ ./app/
COPY google-credentials.json* ./

RUN mkdir -p temp_uploads logs data exports \
 && chmod 755 temp_uploads logs data exports \
 && find /app -name "*.pyc" -delete \
 && find /app -name "__pycache__" -type d -exec rm -rf {} + || true

RUN groupadd -r planbook && useradd -r -g planbook planbook \
 && chown -R planbook:planbook /app /usr/local/nltk_data
USER planbook

EXPOSE 8000 5555
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
