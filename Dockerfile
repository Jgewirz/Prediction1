# Kalshi Deep Trading Bot - Production Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command - run dashboard API
CMD ["uv", "run", "uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
