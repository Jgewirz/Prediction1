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

# Copy dependency files first (for layer caching)
COPY pyproject.toml ./

# Install dependencies using pip (more compatible than uv)
RUN pip install --no-cache-dir \
    httpx>=0.25.0 \
    websockets>=11.0 \
    pydantic>=2.4.0 \
    pydantic-settings>=2.0.0 \
    python-dotenv>=1.0.0 \
    loguru>=0.7.0 \
    rich>=13.0.0 \
    cryptography>=41.0.0 \
    openai>=1.0.0 \
    aiosqlite>=0.19.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    apscheduler>=3.10.0 \
    asyncpg>=0.29.0 \
    "passlib[bcrypt]>=1.7.4" \
    "python-jose[cryptography]>=3.3.0" \
    python-multipart>=0.0.6 \
    email-validator>=2.0.0

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
CMD ["python", "-m", "uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
