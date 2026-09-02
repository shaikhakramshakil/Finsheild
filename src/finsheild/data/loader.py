"""Loader for Finsheild fraud dataset.

Handles CSV loading, schema validation, missing-value handling, and basic logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Class",
]

TARGET_COL = "Class"


def _load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        candidates = [
            Path("config/dataset.yaml"),
            Path(__file__).parents[3] / "config" / "dataset.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break
        else:
            return {}
    config_path = Path(config_path)
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def validate_schema(df: pd.DataFrame, expected_columns: list[str] | None = None) -> None:
    """Validate DataFrame schema.

    Raises ValueError if columns mismatch or target missing.
    """
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if extra:
        logger.warning("Extra columns found (ignored): %s", sorted(extra))
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not present")
    unique_vals = set(df[TARGET_COL].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Target column has unexpected values: {unique_vals}")


def load_raw(
    csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load raw CSV, validate schema, handle missing values.

    Args:
        csv_path: Path to CSV. If None, reads from config/dataset.yaml.
        config_path: Path to dataset.yaml.
        expected_columns: Override expected columns.

    Returns:
        Cleaned DataFrame.
    """
    config = _load_config(config_path)
    if csv_path is None:
        csv_path = config.get("paths", {}).get("raw_csv", "data/raw/creditcard.csv")
    if expected_columns is None:
        expected_columns = config.get("schema", {}).get("expected_columns", EXPECTED_COLUMNS)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. "
            "Run: python scripts/download_dataset.py  OR  see notebooks/colab/01_dataset.ipynb for Colab acquisition. "
            "For CI/test without credentials: python scripts/download_dataset.py --synthetic"
        )
    logger.info("Loading raw dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    logger.info("Raw shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    validate_schema(df, expected_columns)
    # Handle missing values
    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        logger.info("Missing values detected: %d — dropping rows with NaN", missing_total)
        df = df.dropna()
        logger.info("Rows after dropna: %d (dropped %d)", len(df), initial_rows - len(df))
    else:
        logger.info("No missing values detected")
    dup = int(df.duplicated().sum())
    logger.info("Duplicated rows: %d", dup)
    logger.info("Class distribution: %s", df[TARGET_COL].value_counts().to_dict())
    logger.info("Row count after cleaning: %d", len(df))
    return df
