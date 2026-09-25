# Mental Health Score — MLOps Project

Production-style MLOps wrapper around the `Mental_Health_Model.pkl`
RandomForestRegressor (trained in `notebooks/ML_Project.ipynb` on
`data/mental_health.csv`) that predicts a student's mental health score
(0–10) from lifestyle and social-media-usage features.

Includes: MLflow experiment tracking + model registry, a FastAPI serving
layer, a Dockerfile, a GitHub Actions CI/CD pipeline, AWS SageMaker
deployment scripts, and lightweight drift/data-quality monitoring.

> ⚠️ **Version pin, read this first.** The shipped model was trained with
> `scikit-learn==1.6.1`. Loading it with a newer version (confirmed
> failure on 1.8.0) raises `AttributeError`/`UnpicklingError` because
> internal transformer classes changed between releases. Every
> requirements file and the Dockerfile pin `scikit-learn==1.6.1` for
> exactly this reason — don't bump it without re-exporting the `.pkl`
> from a matching environment.

## Architecture

```
                 ┌────────────┐        ┌──────────────────┐
  train.py  ───▶ │   MLflow   │        │   models/*.pkl    │◀── shared by
 (tracking +     │ (tracking +│        │  (joblib artifact) │    FastAPI & SageMaker
  registry)      │  registry) │        └──────────────────┘
                 └────────────┘                 │
                                                 ▼
        ┌───────────────────┐          ┌──────────────────┐
        │ src/inference.py  │◀────────▶│ src/schema.py     │
        │ src/config.py     │          │ (Pydantic + feat. │
        │ src/monitoring.py │          │  vocab, shared)   │
        └───────────────────┘          └──────────────────┘
              ▲                                  ▲
              │                                  │
   ┌──────────┴─────────┐            ┌───────────┴────────────┐
   │  app/main.py        │            │ sagemaker/inference.py │
   │  FastAPI (Docker)    │            │ (SKLearn container)    │
   │  /predict /health     │            │ model_fn/input_fn/...  │
   │  /metrics /monitoring │            └────────────────────────┘
   └──────────────────────┘                        │
              │                                     ▼
              │                         SageMaker endpoint + Data
              ▼                         Capture + Model Monitor
      Prometheus / logs
```

Both serving paths (local Docker/FastAPI and AWS SageMaker) import the
*same* `src/` package for preprocessing, feature validation and country
bucketing, so they can never silently disagree on how a request is
turned into a model input.

## Repo layout

```
src/            shared: config, pydantic schema, training, inference, monitoring
app/            FastAPI serving app (runs in the Docker image)
sagemaker/      SageMaker script-mode inference handler + deploy/teardown scripts
scripts/        one-off ops scripts (rebuild the monitoring baseline)
tests/          pytest suite for the shared inference code and the API
monitoring/     reference_stats.json (training baseline) + prediction_log.jsonl (runtime)
data/           training CSV
models/         model.pkl artifact served by both FastAPI and SageMaker
notebooks/      original exploratory notebook (kept for reference/audit trail)
.github/workflows/ci-cd.yml   lint → test → build/push image → deploy to SageMaker
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Train + track with MLflow

```bash
make train                 # trains with the notebook's default RF hyperparameters
make train-tuned           # RandomizedSearchCV like the notebook, registers the model
make mlflow-ui             # http://localhost:5000 to browse runs/metrics/artifacts
```

Each run logs params, `r2_test`/`r2_train`/`mae_test`/`rmse_test`, the
sklearn/python versions, the fitted pipeline (with signature + input
example), and writes `models/Mental_Health_Model.pkl` for serving.

### 2. Serve locally

```bash
make serve                 # uvicorn --reload, http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Study_Hours": 4.5, "Age": 21, "Avg_Daily_Usage_Hours": 4.0, "Daily_Unlocks": 134,
  "Physical_Activity_Hours": 2.2, "Sleep_Hours_Per_Night": 6.7, "Stress_Level": "Medium",
  "Gender": "Male", "Academic_Level": "Undergraduate", "Most_Used_Platform": "Facebook",
  "Purpose_Of_Use": "Networking", "Country": "India"
}'
```

Endpoints: `POST /predict`, `POST /predict/batch`, `GET /health`,
`GET /metrics` (Prometheus), `GET /monitoring/drift`.

### 3. Docker

```bash
make docker-build
make docker-run            # http://localhost:8000
# or, with MLflow UI alongside it:
make compose-up
```

The image is built and its `/health` endpoint smoke-tested in CI on
every PR; it was not build-tested inside this sandbox (no Docker
daemon available here), but it follows the standard `python:3.11-slim`
+ pinned-requirements + non-root-user pattern and installs cleanly from
the pinned `requirements-serve.txt`.

### 4. Tests

```bash
make test                  # 11 tests: schema validation, inference, all API routes, drift
make lint                  # ruff
```

### 5. Deploy to AWS SageMaker

Requires AWS credentials with SageMaker/S3/IAM access and a SageMaker
execution role ARN.

```bash
pip install boto3 sagemaker
python sagemaker/deploy.py \
  --role-arn arn:aws:iam::<account_id>:role/SageMakerExecutionRole \
  --bucket <your-s3-bucket> \
  --endpoint-name mental-health-score-endpoint
```

This packages `model.tar.gz` (model + `sagemaker/inference.py` + the
shared `src/` package), uploads it to S3, deploys a real-time endpoint
on the prebuilt SKLearn container with **Data Capture** enabled (100%
sampling to S3), and creates an hourly **Model Monitor** data-quality
schedule baselined against `data/mental_health.csv`.

Tear it down when you're done (endpoints bill per instance-hour):

```bash
python sagemaker/teardown.py --endpoint-name mental-health-score-endpoint
```

### 6. CI/CD

`.github/workflows/ci-cd.yml`:
- **Every PR & push:** install deps → `ruff check` → `pytest`.
- **Build job:** builds the Docker image, runs it, curls `/health` as a
  smoke test. Pushes to ECR only on `push` to `main`.
- **Deploy job (main only):** runs `sagemaker/deploy.py` to create/update
  the endpoint, then invokes it once as a post-deploy smoke test.

Required repo secrets: `AWS_DEPLOY_ROLE_ARN` (OIDC role for the Actions
runner — ECR + SageMaker + S3), `SAGEMAKER_EXECUTION_ROLE_ARN` (the role
the endpoint itself runs as), `MODEL_ARTIFACT_BUCKET`.

## Monitoring

Two complementary layers:

1. **`src/monitoring.py` (always on, zero extra infra).** Every
   `/predict` call is appended to `monitoring/prediction_log.jsonl`.
   `GET /monitoring/drift` compares the most recent window against
   `monitoring/reference_stats.json` (the training distribution) using
   **PSI** for categorical features and a **standardized mean-shift
   z-score** for numeric features, with `ok` / `warn` / `alert`
   thresholds per feature and an overall `status`. Run
   `python scripts/build_reference_stats.py` after every retrain to
   refresh the baseline. I validated this against synthetic drifted
   traffic (usage hours pushed to 9–12h/day) and it correctly returned
   `"status": "alert"` with the affected features flagged.
2. **SageMaker Model Monitor (AWS-side, optional).** `deploy.py` turns
   on Data Capture and schedules AWS's own Data Quality monitor against
   the same baseline CSV, so you get CloudWatch alarms/reports even if
   the FastAPI layer isn't in the loop (e.g. traffic hitting the
   endpoint directly).

`GET /metrics` also exposes Prometheus counters/histograms
(`predict_requests_total`, `predict_latency_seconds`,
`predicted_mental_health_score`) for a Grafana dashboard or alerting
rule.

## Known limitations / next steps

- The RandomForest's `mean_squared_error` isn't monotonically tracked
  over time yet — for real production use, add a scheduled job that
  joins predictions back to ground-truth scores (once available) and
  logs realized error to MLflow, not just training-time metrics.
- `src/monitoring.py`'s drift log is a local JSONL file; swap
  `PREDICTION_LOG_PATH` for a real sink (CloudWatch Logs, S3, Kinesis)
  before running multiple replicas, since a local file isn't shared
  across containers/instances.
- `sagemaker/deploy.py` uses `ml.m5.large` / hourly monitoring as
  reasonable defaults — resize the instance type and monitoring cadence
  to your actual traffic and budget.
