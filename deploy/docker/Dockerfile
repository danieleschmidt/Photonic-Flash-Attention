# Multi-stage production Dockerfile for Photonic Flash Attention
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r photonicuser && useradd -r -g photonicuser photonicuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Development stage
FROM base as development
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
RUN chown -R photonicuser:photonicuser /app
USER photonicuser
CMD ["python", "-m", "photonic_flash_attention.cli", "--dev"]

# Production stage
FROM base as production

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install -e .

# Copy configuration files
COPY deploy/config/ ./config/

# Set ownership and permissions
RUN chown -R photonicuser:photonicuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER photonicuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import photonic_flash_attention; print('OK')" || exit 1

# Expose port (if needed for API)
EXPOSE 8080

# Default command
CMD ["python", "-m", "photonic_flash_attention.cli", "--production"]