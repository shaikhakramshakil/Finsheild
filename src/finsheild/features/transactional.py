"""Transaction-level features (no leakage possible by construction).

These are properties of a single transaction in isolation — amount, hour,
day, channel, merchant category. They never need historical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finsheild.features._utils import day_of_week, hour_of_day, is_offhours


def build_transactional_features(tx: pd.DataFrame,
                                  merchants: pd.DataFrame,
                                  high_value_threshold: float) -> pd.DataFrame:
    """Return a per-transaction DataFrame with transaction-level columns.

    Output columns
    --------------
    txn_id, amount_log, hour, day_of_week, is_offhours, is_high_value,
    is_online, is_pos, is_atm, is_mobile, is_high_risk_merchant,
    merchant_risk_band_ord
    """
    df = tx[["txn_id", "amount", "channel", "merchant_id"]].copy()
    df["amount_log"] = np.log1p(df["amount"]).astype("float32")
    df["hour"] = hour_of_day(tx["ts"]).astype("int8")
    df["day_of_week"] = day_of_week(tx["ts"]).astype("int8")
    df["is_offhours"] = is_offhours(df["hour"]).astype("int8")
    df["is_high_value"] = (df["amount"] > high_value_threshold).astype("int8")

    # One-hot channel
    df["is_online"] = (df["channel"] == "online").astype("int8")
    df["is_pos"] = (df["channel"] == "pos").astype("int8")
    df["is_atm"] = (df["channel"] == "atm").astype("int8")
    df["is_mobile"] = (df["channel"] == "mobile").astype("int8")

    # Merchant category (joined)
    merch = merchants[["merchant_id", "category", "risk_band"]].rename(
        columns={"category": "merchant_category", "risk_band": "merchant_risk_band"})
    df = df.merge(merch, on="merchant_id", how="left")
    df["is_high_risk_merchant"] = (
        df["merchant_risk_band"] == "high").astype("int8")
    risk_map = {"low": 0, "medium": 1, "high": 2}
    df["merchant_risk_band_ord"] = df["merchant_risk_band"].map(risk_map) \
        .fillna(0).astype("int8")

    return df.drop(columns=["channel", "merchant_category",
                              "merchant_risk_band"])


FEATURE_COLUMNS = [
    "amount_log", "hour", "day_of_week", "is_offhours", "is_high_value",
    "is_online", "is_pos", "is_atm", "is_mobile",
    "is_high_risk_merchant", "merchant_risk_band_ord",
]