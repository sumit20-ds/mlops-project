"""Delete a deployed endpoint, its config, and any monitoring schedule.

SageMaker real-time endpoints bill per instance-hour until deleted —
run this when you're done testing.

Usage:
    python sagemaker/teardown.py --endpoint-name mental-health-score-endpoint
"""
from __future__ import annotations

import argparse

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", default="mental-health-score-endpoint")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    for schedule in sm.list_monitoring_schedules(
        EndpointName=args.endpoint_name
    ).get("MonitoringScheduleSummaries", []):
        name = schedule["MonitoringScheduleName"]
        print(f"Deleting monitoring schedule {name}")
        sm.delete_monitoring_schedule(MonitoringScheduleName=name)

    try:
        desc = sm.describe_endpoint(EndpointName=args.endpoint_name)
        config_name = desc["EndpointConfigName"]
        print(f"Deleting endpoint {args.endpoint_name}")
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        print(f"Deleting endpoint config {config_name}")
        sm.delete_endpoint_config(EndpointConfigName=config_name)
    except sm.exceptions.ClientError as exc:
        print(f"Endpoint not found or already deleted: {exc}")


if __name__ == "__main__":
    main()
