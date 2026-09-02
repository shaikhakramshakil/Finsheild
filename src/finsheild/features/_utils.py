"""Internal helpers shared by the feature modules.

Keep this small — anything that needs to be public belongs in the top-level
``features`` package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(a: float | np.ndarray,
              b: float | np.ndarray,
              fill: float = 0.0) -> float | np.ndarray:
    """``a / b`` with zero-division protection. Returns ``fill`` where ``b==0``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.where(b == 0, fill, a / b)
    return out


def hour_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour


def day_of_week(ts: pd.Series) -> pd.Series:
    return ts.dt.dayofweek


def is_offhours(hour: pd.Series, low: int = 2, high: int = 5) -> pd.Series:
    return ((hour >= low) & (hour < high)).astype(np.int8)