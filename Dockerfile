FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p src/vectordb/chroma_collection

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Render sets PORT automatically; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
