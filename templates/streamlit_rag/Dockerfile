# RAG chat UI image. Build the index first, then mount it in (compose does this).
#   docker build -f templates/streamlit_rag/Dockerfile -t rag-chat .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY templates/streamlit_rag/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY templates/streamlit_rag/app.py templates/streamlit_rag/ingest.py ./
ENV PYTHONPATH=/app:/app/src

# Bake the sample index in so the image answers on boot; mount /app/index
# with your own to override.
COPY data/samples/rag_docs/ ./seed_docs/
RUN python ingest.py --docs ./seed_docs --index ./index/rag

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--", "--index", "./index/rag"]
