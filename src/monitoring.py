"""
Lightweight model-monitoring utilities.

Two things are tracked:

1. Prediction/feature logging — every request+prediction is appended to a
   JSONL log (`monitoring/prediction_log.jsonl`). In production this file
   path is swapped for S3 (via SageMaker Data Capture, see
   sagemaker/deploy.py) or a real log sink (CloudWatch/ELK); locally it's
   just a file so the project runs with zero extra infra.

2. Drift detection — `compute_drift_report()` compares a recent window of
   logged requests against `monitoring/reference_stats.json` (the training
   distribution) using:
     - Population Stability Index (PSI) for categorical features
     - Standardized mean shift ("drift z-score") for numeric features
   This is intentionally dependency-light (no evidently/whylogs required)
   so it runs the same way in CI, Docker, and a SageMaker Monitor job.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime

from src import config

# Thresholds are the common industry rules of thumb for PSI:
#   < 0.1  -> no significant shift
#   0.1-0.25 -> moderate shift, worth a look
#   > 0.25 -> significant drift, investigate / retrain
PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25
# For numeric features: |mean_shift| in units of the reference std-dev.
ZSHIFT_WARN_THRESHOLD = 0.5
ZSHIFT_ALERT_THRESHOLD = 1.0


def log_prediction(request_row: dict, prediction: float, model_version: str) -> None:
    config.PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "prediction": prediction,
        **request_row,
    }
    with open(config.PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _read_recent_logs(limit: int | None = None) -> list[dict]:
    if not config.PREDICTION_LOG_PATH.exists():
        return []
    lines = config.PREDICTION_LOG_PATH.read_text().strip().splitlines()
    if limit:
        lines = lines[-limit:]
    return [json.loads(line) for line in lines if line]


def _psi(reference_dist: dict[str, float], live_counts: Counter, live_total: int) -> float:
    """Population Stability Index between a reference proportion dict and
    a live sample's counts. Categories unseen in either side are floored
    to avoid divide-by-zero / log(0)."""
    eps = 1e-4
    categories = set(reference_dist) | set(live_counts)
    psi = 0.0
    for cat in categories:
        ref_pct = max(reference_dist.get(cat, 0.0), eps)
        live_pct = max((live_counts.get(cat, 0) / live_total) if live_total else 0.0, eps)
        psi += (live_pct - ref_pct) * math.log(live_pct / ref_pct)
    return round(psi, 4)


def _severity(value: float, warn: float, alert: float) -> str:
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def compute_drift_report(window: int = 500) -> dict:
    """Compare the most recent `window` logged predictions against the
    training-time reference distribution. Returns a JSON-serializable
    report; also usable as the payload for a scheduled SageMaker Model
    Monitor job or a CloudWatch custom metric publisher."""
    if not config.REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {config.REFERENCE_STATS_PATH}. "
            "Run scripts/build_reference_stats.py after (re)training."
        )
    reference = json.loads(config.REFERENCE_STATS_PATH.read_text())
    recent = _read_recent_logs(limit=window)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_size": len(recent),
        "reference_size": reference.get("n_reference_rows"),
        "numeric_drift": {},
        "categorical_drift": {},
        "status": "ok",
    }

    if not recent:
        report["status"] = "no_data"
        return report

    worst = "ok"

    for feat, ref_stats in reference.get("numeric", {}).items():
        values = [row[feat] for row in recent if feat in row and row[feat] is not None]
        if not values:
            continue
        live_mean = sum(values) / len(values)
        ref_std = ref_stats["std"] or 1e-6
        z_shift = abs(live_mean - ref_stats["mean"]) / ref_std
        sev = _severity(z_shift, ZSHIFT_WARN_THRESHOLD, ZSHIFT_ALERT_THRESHOLD)
        report["numeric_drift"][feat] = {
            "reference_mean": ref_stats["mean"],
            "live_mean": round(live_mean, 4),
            "drift_zscore": round(z_shift, 4),
            "severity": sev,
        }
        worst = _worse(worst, sev)

    for feat, ref_dist in reference.get("categorical", {}).items():
        values = [row[feat] for row in recent if feat in row and row[feat] is not None]
        if not values:
            continue
        live_counts = Counter(values)
        psi = _psi(ref_dist, live_counts, len(values))
        sev = _severity(psi, PSI_WARN_THRESHOLD, PSI_ALERT_THRESHOLD)
        report["categorical_drift"][feat] = {"psi": psi, "severity": sev}
        worst = _worse(worst, sev)

    report["status"] = worst
    return report


def _worse(a: str, b: str) -> str:
    order = {"ok": 0, "warn": 1, "alert": 2}
    return a if order[a] >= order[b] else b
