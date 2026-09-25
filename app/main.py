"""FastAPI serving layer for the Mental Health Score model."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src import monitoring
from src.inference import ModelNotLoadedError, load_model, predict
from src.schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    StudentFeatures,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mental_health_api")

MODEL_VERSION = "rf-v1"  #wire to MLflow run id or SageMaker model package ARN in CI

REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests", ["status"])
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Prediction latency in seconds")
PREDICTION_VALUE = Histogram(
    "predicted_mental_health_score", "Distribution of predicted scores",
    buckets=[3, 4, 5, 6, 7, 8, 9, 10],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        logger.info("Model loaded successfully at startup.")
    except ModelNotLoadedError:
        logger.exception("Model failed to load at startup.")
        raise
    yield


app = FastAPI(  #runingby mlflow model
    title="Mental Health Score API",
    description="Predicts a student's mental health score from lifestyle & social-media usage features.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s path=%s status=%s duration_ms=%.1f",
                request_id, request.url.path, response.status_code, duration * 1000)
    return response


@app.get("/health")
def health():
    """Liveness/readiness probe used by Docker HEALTHCHECK, ECS/K8s, and
    the SageMaker /ping contract."""
    try:
        load_model()
        return {"status": "healthy", "model_version": MODEL_VERSION}
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/drift")
def drift_report(window: int = 500):
    """On-demand drift report comparing recent traffic to the training
    distribution. Intended to be polled by a scheduler (cron / Airflow /
    SageMaker Monitor / CloudWatch Synthetics) rather than end users."""
    try:
        return monitoring.compute_drift_report(window=window)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict_one(features: StudentFeatures, request: Request):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    try:
        row = features.to_model_row()
        score = predict([row])[0]
    except ModelNotLoadedError as exc:
        REQUEST_COUNT.labels(status="model_error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    REQUEST_COUNT.labels(status="success").inc()
    REQUEST_LATENCY.observe(time.perf_counter() - start)
    PREDICTION_VALUE.observe(score)
    monitoring.log_prediction(row, score, MODEL_VERSION)

    return PredictionResponse(mental_health_score=score, model_version=MODEL_VERSION, request_id=request_id)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, request: Request):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    try:
        rows = [f.to_model_row() for f in payload.instances]
        scores = predict(rows)
    except ModelNotLoadedError as exc:
        REQUEST_COUNT.labels(status="model_error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}") from exc

    REQUEST_COUNT.labels(status="success").inc(len(rows))
    for row, score in zip(rows, scores):
        PREDICTION_VALUE.observe(score)
        monitoring.log_prediction(row, score, MODEL_VERSION)

    predictions = [
        PredictionResponse(mental_health_score=s, model_version=MODEL_VERSION, request_id=request_id)
        for s in scores
    ]
    return BatchPredictionResponse(predictions=predictions)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
