# Multi-stage Dockerfile for Photonic Flash Attention
FROM nvidia/cuda:12.0-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash photonic && \
    usermod -aG sudo photonic

WORKDIR /app

# Install Python dependencies first (for better layer caching)
COPY pyproject.toml setup.py ./
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -e .[benchmark,hardware]

# Production stage
FROM base as production

# Copy application code
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY README.md LICENSE ./

# Set ownership
RUN chown -R photonic:photonic /app

# Switch to non-root user
USER photonic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import photonic_flash_attention; print('OK')" || exit 1

# Default command
CMD ["python3", "-c", "import photonic_flash_attention; print('Photonic Flash Attention ready')"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip3 install --no-cache-dir -e .[dev,benchmark,hardware]

# Install additional development tools
RUN pip3 install --no-cache-dir \
    jupyter \
    ipython \
    jupyterlab \
    notebook

# Copy application code
COPY . /app/

# Set ownership
RUN chown -R photonic:photonic /app

# Switch to non-root user
USER photonic

# Expose ports for Jupyter
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Testing stage
FROM development as testing

# Run tests
RUN python3 -m pytest tests/unit/ -v
RUN python3 -m pytest tests/security/ -v
RUN PHOTONIC_SIMULATION=1 python3 -m pytest tests/integration/ -v

# Benchmark stage
FROM production as benchmark

# Install benchmark dependencies
USER root
RUN pip3 install --no-cache-dir -e .[benchmark]
USER photonic

# Run benchmarks
RUN PHOTONIC_SIMULATION=1 python3 -m pytest tests/performance/ -v -m "performance and not slow"