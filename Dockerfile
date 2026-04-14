# Using slim to keep the image small — the full python image is 1GB+
# and we don't need any of the extra stuff it comes with.
FROM python:3.10-slim

LABEL maintainer="Mayur Patil <mayur.patil@gwmail.gwu.edu>"
LABEL description="Clinical Lab Abnormality Predictor — API + Dashboard"
LABEL version="2.0.0"

# Prevents Python from writing .pyc files and buffers stdout/stderr
# so logs show up in real time in Docker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system packages needed to build some Python wheels.
# --no-install-recommends keeps things lean.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache the layer.
# If requirements.txt doesn't change, this layer is reused on rebuild.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy the rest of the project
COPY src/     ./src/
COPY api/     ./api/
COPY tests/   ./tests/
COPY docs/    ./docs/
COPY models/  ./models/ 2>/dev/null || true   # models might not exist at build time
COPY data/    ./data/   2>/dev/null || true

# Create directories the app writes to at runtime
RUN mkdir -p monitoring mlruns

# FastAPI (8000), Streamlit (8501), MLflow UI (5000)
EXPOSE 8000 8501 5000

# Health check — pings the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start the API. Override in docker-compose for other services.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
