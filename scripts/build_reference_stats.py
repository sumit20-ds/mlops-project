"""Recompute monitoring/reference_stats.json from the training data.

Run this after every retrain so the drift monitor (src/monitoring.py)
compares live traffic against the distribution the *current* model was
actually trained on, not a stale one.

Usage:
    python scripts/build_reference_stats.py
"""
from __future__ import annotations

import json

import pandas as pd

from src import config


def main() -> None:
    df = pd.read_csv(config.DATA_PATH)
    df = df.drop_duplicates()
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)

    numeric_cols = config.SKEWED_COLS + config.PLAIN_NUMERIC_COLS
    categorical_cols = config.ORDINAL_COLS + [
        c for c in config.NOMINAL_COLS if c != "Grouped_country"
    ]

    stats = {"numeric": {}, "categorical": {}, "target": {}}

    for col in numeric_cols:
        s = df[col]
        stats["numeric"][col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.5)),
            "p75": float(s.quantile(0.75)),
        }

    for col in categorical_cols:
        vc = df[col].value_counts(normalize=True)
        stats["categorical"][col] = vc.round(4).to_dict()

    target = df[config.TARGET_COLUMN]
    stats["target"] = {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
    }
    stats["n_reference_rows"] = int(len(df))

    config.REFERENCE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REFERENCE_STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {config.REFERENCE_STATS_PATH} ({stats['n_reference_rows']} reference rows)")


if __name__ == "__main__":
    main()
