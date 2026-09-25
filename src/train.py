"""
Train the Mental Health Score regressor and track everything in MLflow.

Reproduces the preprocessing + RandomForestRegressor pipeline built in
notebooks/ML_Project.ipynb, but as a scripted, reproducible, versioned
training job instead of a notebook run.

Usage:
    python -m src.train                     # train with default hyperparams
    python -m src.train --tune              # RandomizedSearchCV like the notebook
    python -m src.train --register          # also push to the MLflow Model Registry
"""
from __future__ import annotations

import argparse
import json
import platform
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src import config


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)
    df = df.drop_duplicates()
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
    df["Grouped_country"] = df["Country"].apply(config.group_country)
    return df


def build_pipeline(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1) -> Pipeline:
    skew_pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    plain_numeric_pipeline = Pipeline([("scale", StandardScaler())])
    ordinal_pipeline = Pipeline([
        ("encode", OrdinalEncoder(categories=[config.STRESS_LEVEL_ORDER])),
    ])
    nominal_pipeline = Pipeline([("encode", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer([
        ("Skewed_Pipeline", skew_pipeline, config.SKEWED_COLS),
        ("Plain_Numeric", plain_numeric_pipeline, config.PLAIN_NUMERIC_COLS),
        ("Ordinal", ordinal_pipeline, config.ORDINAL_COLS),
        ("Normal", nominal_pipeline, config.NOMINAL_COLS),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("random forest", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV like the notebook")
    parser.add_argument("--register", action="store_true", help="Register the model in the MLflow Model Registry")
    parser.add_argument("--n-iter", type=int, default=15)
    args = parser.parse_args()

    df = load_data()
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("tuned", args.tune)

        if args.tune:
            base_pipeline = build_pipeline()
            param_grid = {
                "random forest__n_estimators": [100, 200, 300],
                "random forest__max_depth": [5, 10, 15],
                "random forest__min_samples_split": [2, 5, 10],
                "random forest__min_samples_leaf": [1, 2, 4],
            }
            search = RandomizedSearchCV(
                estimator=base_pipeline,
                param_distributions=param_grid,
                n_iter=args.n_iter,
                cv=5,
                scoring="r2",
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        else:
            pipeline = build_pipeline()
            pipeline.fit(X_train, y_train)

        preds_test = pipeline.predict(X_test)
        preds_train = pipeline.predict(X_train)

        metrics = {
            "r2_test": r2_score(y_test, preds_test),
            "r2_train": r2_score(y_train, preds_train),
            "mae_test": mean_absolute_error(y_test, preds_test),
            "rmse_test": float(np.sqrt(mean_squared_error(y_test, preds_test))),
        }
        mlflow.log_metrics(metrics)
        print(json.dumps(metrics, indent=2))

        signature = infer_signature(X_train, preds_train)
        input_example = X_train.head(3)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=config.REGISTERED_MODEL_NAME if args.register else None,
        )

        # Also save a plain joblib artifact so serving (FastAPI / SageMaker)
        # doesn't need to depend on the MLflow tracking store at inference
        # time — this is what ships inside the Docker image / SageMaker
        # model.tar.gz.
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, config.MODEL_PATH)
        mlflow.log_artifact(str(config.MODEL_PATH))

        print(f"MLflow run: {run.info.run_id}")
        print(f"Model written to: {config.MODEL_PATH}")


if __name__ == "__main__":
    sys.exit(main())
