# ============================================================
# PhishNet Docker Image
# Sandesh Poudel | B01818884 | UWS MSc Cybersecurity
# ============================================================

FROM python:3.10-slim

LABEL maintainer="b01818884@studentmail.uws.ac.uk"
LABEL project="PhishNet - MSc Cybersecurity"
LABEL version="1.0.0"

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root for security)
RUN useradd -m -u 1000 phishnet
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=phishnet:phishnet . .

# Create result directories
RUN mkdir -p results/models results/plots results/metrics results/logs \
    data/raw data/processed

USER phishnet

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
