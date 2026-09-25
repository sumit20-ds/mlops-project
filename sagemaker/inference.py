"""
SageMaker script-mode inference entrypoint.

Deployed inside AWS's prebuilt SKLearn inference container (via
`sagemaker.sklearn.model.SKLearnModel`, see sagemaker/deploy.py). SageMaker
imports this module and calls the four handlers below; it does NOT run
app/main.py — this file is the SageMaker-side equivalent of that FastAPI
route, reusing the same src.inference / src.schema code so the two serving
paths can never disagree on preprocessing.
"""
from __future__ import annotations

import json
import os
import sys

# When SageMaker packages model.tar.gz (see deploy.py) it lays out:
#   /opt/ml/model/Mental_Health_Model.pkl
#   /opt/ml/model/code/inference.py   <- this file
#   /opt/ml/model/code/src/...        <- our shared src package
# so `src` is importable relative to this file's parent.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.inference import ModelNotLoadedError, load_model, rows_to_dataframe  # noqa: E402
from src.schema import StudentFeatures  # noqa: E402

CONTENT_TYPE_JSON = "application/json"


def model_fn(model_dir: str):
    """Called once per worker at container startup."""
    model_path = os.path.join(model_dir, "Mental_Health_Model.pkl")
    try:
        return load_model(model_path)
    except ModelNotLoadedError as exc:
        # Re-raise so the container fails to come up rather than serving
        # silently broken predictions — SageMaker will surface this in
        # CloudWatch and the endpoint creation will fail fast.
        raise RuntimeError(str(exc)) from exc


def input_fn(request_body: str, request_content_type: str):
    if request_content_type != CONTENT_TYPE_JSON:
        raise ValueError(f"Unsupported content type: {request_content_type}. Use application/json.")
    payload = json.loads(request_body)
    instances = payload["instances"] if "instances" in payload else [payload]
    # Validate + apply the Grouped_country bucketing rule exactly like the
    # FastAPI path does, via the shared Pydantic schema.
    rows = [StudentFeatures(**inst).to_model_row() for inst in instances]
    return rows_to_dataframe(rows)


def predict_fn(input_data, model):
    preds = model.predict(input_data)
    return [round(float(p), 3) for p in preds]


def output_fn(prediction, accept: str):
    if accept != CONTENT_TYPE_JSON and accept != "*/*":
        raise ValueError(f"Unsupported accept type: {accept}")
    body = json.dumps({
        "predictions": [{"mental_health_score": p} for p in prediction]
    })
    return body, CONTENT_TYPE_JSON
