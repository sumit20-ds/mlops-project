# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/app

# --- deps layer (cached separately from source for fast rebuilds) -------
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# --- app layer ------------------------------------------------------------
COPY src ./src
COPY app ./app
COPY models ./models
COPY monitoring/reference_stats.json ./monitoring/reference_stats.json

# Non-root runtime user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /opt/app/monitoring \
    && chown -R appuser:appuser /opt/app
USER appuser

ENV MODEL_PATH=/opt/app/models/Mental_Health_Model.pkl \
    REFERENCE_STATS_PATH=/opt/app/monitoring/reference_stats.json \
    PREDICTION_LOG_PATH=/opt/app/monitoring/prediction_log.jsonl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else sys.exit(1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
