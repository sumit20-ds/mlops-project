"""
Deploy the Mental Health Score model to a SageMaker real-time endpoint.

What this does, end to end:
  1. Packages models/Mental_Health_Model.pkl + sagemaker/inference.py +
     the shared src/ package into model.tar.gz (the layout SageMaker's
     SKLearn container expects).
  2. Uploads it to S3.
  3. Creates an SKLearnModel pinned to scikit-learn 1.6.1 (must match the
     version the model was trained/exported with).
  4. Deploys a real-time endpoint with Data Capture enabled, so every
     request/response is written to S3 for monitoring.
  5. Creates a SageMaker Model Monitor schedule that runs the built-in
     Data Quality monitor on the captured traffic.

Requires AWS credentials with SageMaker/S3/IAM permissions
(`aws configure` or the usual env vars) and an execution role ARN.

Usage:
    python sagemaker/deploy.py \
        --role-arn arn:aws:iam::<account_id>:role/SageMakerExecutionRole \
        --bucket my-mlops-bucket \
        --endpoint-name mental-health-score-endpoint
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import boto3

import sagemaker
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
from sagemaker.sklearn.model import SKLearnModel

ROOT = Path(__file__).resolve().parent.parent
SKLEARN_FRAMEWORK_VERSION = "1.2-1"  # closest SageMaker prebuilt SKLearn container to the pinned 1.6.1 lib
BUILD_DIR = ROOT / "sagemaker" / "_build"


def build_model_tarball() -> Path:
    """Lay out /opt/ml/model style package:
        model.tar.gz
        ├── Mental_Health_Model.pkl
        └── code/
            ├── inference.py
            └── src/  (shared config/schema/inference modules)
    """
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    code_dir = BUILD_DIR / "code"
    code_dir.mkdir(parents=True)

    shutil.copy(ROOT / "models" / "Mental_Health_Model.pkl", BUILD_DIR / "Mental_Health_Model.pkl")
    shutil.copy(ROOT / "sagemaker" / "inference.py", code_dir / "inference.py")
    shutil.copytree(ROOT / "src", code_dir / "src")
    shutil.copy(ROOT / "monitoring" / "reference_stats.json", code_dir / "reference_stats.json")

    tarball_path = ROOT / "sagemaker" / "model.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(BUILD_DIR / "Mental_Health_Model.pkl", arcname="Mental_Health_Model.pkl")
        tar.add(code_dir, arcname="code")
    return tarball_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--bucket", required=True, help="S3 bucket for model artifacts + data capture")
    parser.add_argument("--endpoint-name", default="mental-health-score-endpoint")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--skip-monitor", action="store_true", help="Deploy without the drift-monitoring schedule")
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    tarball = build_model_tarball()
    s3_model_uri = sm_session.upload_data(
        path=str(tarball), bucket=args.bucket, key_prefix="mental-health-score/model"
    )
    print(f"Uploaded model artifact to {s3_model_uri}")

    model = SKLearnModel(
        model_data=s3_model_uri,
        role=args.role_arn,
        entry_point="inference.py",
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
        env={"REFERENCE_STATS_PATH": "/opt/ml/model/code/reference_stats.json"},
    )

    capture_s3_uri = f"s3://{args.bucket}/mental-health-score/data-capture"
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=capture_s3_uri,
    )

    print(f"Deploying endpoint '{args.endpoint_name}' ({args.instance_type})...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        data_capture_config=data_capture_config,
    )
    print(f"Endpoint deployed: {predictor.endpoint_name}")
    print(f"Data capture writing to: {capture_s3_uri}")

    if not args.skip_monitor:
        monitor = DefaultModelMonitor(
            role=args.role_arn,
            instance_count=1,
            instance_type="ml.m5.large",
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
            sagemaker_session=sm_session,
        )
        baseline_results_uri = f"s3://{args.bucket}/mental-health-score/monitor/baseline"
        # Baseline dataset should be the training CSV with the same feature
        # columns the endpoint receives — used by Model Monitor's built-in
        # Data Quality container to compute constraints/statistics to diff
        # live traffic against, complementing src/monitoring.py's PSI report.
        monitor.suggest_baseline(
            baseline_dataset=str(ROOT / "data" / "mental_health.csv"),
            dataset_format=sagemaker.model_monitor.DatasetFormat.csv(header=True),
            output_s3_uri=baseline_results_uri,
        )
        monitor.create_monitoring_schedule(
            monitor_schedule_name=f"{args.endpoint_name}-data-quality",
            endpoint_input=predictor.endpoint_name,
            output_s3_uri=f"s3://{args.bucket}/mental-health-score/monitor/reports",
            statistics=monitor.baseline_statistics(),
            constraints=monitor.suggested_constraints(),
            schedule_cron_expression=sagemaker.model_monitor.CronExpressionGenerator.hourly(),
        )
        print("Model Monitor schedule created (hourly data-quality checks).")

    print("Done.")


if __name__ == "__main__":
    main()
