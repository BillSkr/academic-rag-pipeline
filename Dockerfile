# RAG Pipeline - Optimized Multi-stage build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pip packages incrementally to leverage Docker cache
COPY requirements.txt .

# Split installation into smaller groups for better caching
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies in order of change frequency (least to most)
RUN pip install --no-cache-dir requests chromadb ollama || true

RUN pip install --no-cache-dir langgraph litellm langchain-core langchain-community langchain-ollama || true

RUN pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings python-dotenv || true

RUN pip install --no-cache-dir pymupdf tqdm tiktoken sentence-transformers || true

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt || true

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create vectordb directory
RUN mkdir -p src/vectordb/chroma_collection

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
