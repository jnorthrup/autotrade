# HRM Trading System - Multi-Service Container
# Services: Paper Trading (live) + Training Workers (concurrent)

FROM python:3.11-slim-bookworm

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py /app/
COPY conductor/ /app/conductor/
COPY tests/ /app/tests/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Environment defaults (override with -e at runtime)
ENV DB_PATH=/app/data/candles.duckdb
ENV BAG_PATH=/app/data/bag.json
ENV CHECKPOINT_DIR=/app/checkpoints
ENV LOG_DIR=/app/logs
ENV COINBASE_API_KEY=""
ENV COINBASE_API_SECRET=""
ENV DEFAULT_GRANULARITY=300
ENV USE_WS_ONLY=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
