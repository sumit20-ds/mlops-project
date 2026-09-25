"""
Central configuration for the Mental Health Score MLOps project.

This is the single source of truth for feature names, categorical
vocabularies and paths so that training, FastAPI serving, SageMaker
inference and monitoring can never silently drift apart.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "Mental_Health_Model.pkl"))
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "mental_health.csv"))
REFERENCE_STATS_PATH = Path(
    os.getenv("REFERENCE_STATS_PATH", ROOT_DIR / "monitoring" / "reference_stats.json")
)
PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", ROOT_DIR / "monitoring" / "prediction_log.jsonl")
)

# --------------------------------------------------------------------------
# Model / experiment identity (used for MLflow tracking + registry)
# --------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT_DIR / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mental-health-score")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "mental_health_score_rf")

# --------------------------------------------------------------------------
# Feature schema — MUST match the ColumnTransformer the model was trained
# with (see notebooks/ML_Project.ipynb, cell 49-51).
# --------------------------------------------------------------------------
SKEWED_COLS = ["Study_Hours"]
PLAIN_NUMERIC_COLS = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
]
ORDINAL_COLS = ["Stress_Level"]
NOMINAL_COLS = [
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Grouped_country",
]

FEATURE_COLUMNS = SKEWED_COLS + PLAIN_NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS
NUMERIC_COLUMNS = SKEWED_COLS + PLAIN_NUMERIC_COLS
TARGET_COLUMN = "Mental_Health_Score"

# Ordered categories used by the OrdinalEncoder at training time.
STRESS_LEVEL_ORDER = ["Low", "Medium", "High", "Very High"]

# Observed vocabularies (from the training data) — used for request
# validation and for the "Other" bucketing rule applied to Country.
GENDER_VALUES = ["Female", "Male"]
ACADEMIC_LEVEL_VALUES = ["High School", "Undergraduate", "Graduate"]
PLATFORM_VALUES = [
    "Facebook", "Instagram", "KakaoTalk", "LINE", "LinkedIn", "Snapchat",
    "TikTok", "Twitter", "VKontakte", "WeChat", "WhatsApp", "YouTube",
]
PURPOSE_VALUES = ["Education", "Entertainment", "Networking", "News"]

# Top-10 raw Country values kept as-is by the training notebook; anything
# else was folded into "Other" via `group_countries()`. Serving must
# apply the exact same rule so live traffic matches the training
# distribution the encoder was fit on.
TOP_COUNTRIES = [
    "Other", "India", "USA", "Canada", "Australia", "UK",
    "Germany", "Mexico", "Turkey", "France",
]


def group_country(country: str) -> str:
    """Mirrors the notebook's `group_countries` bucketing rule exactly."""
    return country if country in TOP_COUNTRIES else "Other"


# Sane physical bounds for numeric inputs (drawn from the training data's
# observed min/max — used only for basic request sanity-checking, not
# for silently clipping/altering values).
NUMERIC_BOUNDS = {
    "Study_Hours": (0.0, 15.0),
    "Age": (10, 100),
    "Avg_Daily_Usage_Hours": (0.0, 24.0),
    "Daily_Unlocks": (0, 1000),
    "Physical_Activity_Hours": (0.0, 24.0),
    "Sleep_Hours_Per_Night": (0.0, 24.0),
}
