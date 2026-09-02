"""Behavioral features derived from each user's *prior* transaction history.

Leakage rule
------------
For a transaction with ``ts = t``, every behavioral feature is computed
using only the user's transactions with ``ts < t``. The current row is
*never* included in its own history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_behavioral_features(tx: pd.DataFrame,
                                accounts: pd.DataFrame,
                                history_min_events: int) -> pd.DataFrame:
    """Return per-transaction behavioral features.

    Output columns
    --------------
    txn_id, prior_tx_count, prior_total_amount, prior_mean_amount,
    prior_std_amount, amount_zscore, amount_log_ratio,
    is_new_user, prior_unique_merchants, prior_unique_devices,
    prior_unique_locations, prior_unique_countries
    """
    # Need user_id (via account) and ts to sort by history
    user_map = accounts.set_index("account_id")["user_id"]
    tx_sorted = tx[["txn_id", "account_id", "ts", "amount",
                     "device_id", "location_id", "merchant_id"]].copy()
    tx_sorted["user_id"] = tx_sorted["account_id"].map(user_map)
    # Stable sort by user + time so prior-cumulative is well-defined.
    tx_sorted = tx_sorted.sort_values(["user_id", "ts", "txn_id"]).reset_index(drop=True)

    g = tx_sorted.groupby("user_id", sort=False)

    # Per-user prior aggregates
    tx_sorted["prior_tx_count"] = g.cumcount()
    tx_sorted["prior_total_amount"] = g["amount"].cumsum() - tx_sorted["amount"]

    # Mean / std up to (but not including) this row
    cumsum = g["amount"].cumsum() - tx_sorted["amount"]
    cumcount = g.cumcount()
    cum_mean = np.where(cumcount > 0, cumsum / np.maximum(cumcount, 1), np.nan)
    tx_sorted["prior_mean_amount"] = cum_mean

    # std via shifted rolling per group. Using a simple approach:
    # std = sqrt(E[x^2] - (E[x])^2) computed cumulatively, then subtract self.
    sq = (tx_sorted["amount"] ** 2).astype("float64")
    cumsum_sq = g[sq.name].cumsum() - sq
    with np.errstate(invalid="ignore"):
        var = (cumsum_sq - cumcount * (cum_mean ** 2)) / np.maximum(cumcount, 1)
        var = np.where(cumcount >= 2, var, np.nan)
        std = np.sqrt(np.maximum(var, 0.0))
    tx_sorted["prior_std_amount"] = std

    # z-score and log-ratio
    with np.errstate(invalid="ignore", divide="ignore"):
        tx_sorted["amount_zscore"] = (tx_sorted["amount"] - tx_sorted["prior_mean_amount"]) \
            / tx_sorted["prior_std_amount"]
        tx_sorted["amount_log_ratio"] = np.log1p(tx_sorted["amount"]) \
            - np.log1p(tx_sorted["prior_mean_amount"])

    # is_new_user: less than history_min_events prior transactions
    tx_sorted["is_new_user"] = (tx_sorted["prior_tx_count"] < history_min_events) \
        .astype("int8")

    # Unique-prior counts via per-group shift+cumcount trick using nunique
    # For each col, groupby user and use expanding(nunique) excluding self
    def _exp_nunique_prior(series: pd.Series, group_key: pd.Series) -> pd.Series:
        # nunique of (group's values up to but not including this row)
        df_in = pd.DataFrame({"g": group_key.values, "v": series.values})
        out = np.empty(len(df_in), dtype=np.int32)
        seen: dict = {}
        for i, (g, v) in enumerate(zip(df_in["g"], df_in["v"])):
            s = seen.setdefault(g, set())
            out[i] = len(s)
            s.add(v)
        return pd.Series(out, index=series.index)

    for col, out_col in (
        ("merchant_id", "prior_unique_merchants"),
        ("device_id", "prior_unique_devices"),
        ("location_id", "prior_unique_locations"),
    ):
        # We need the value at this row to be excluded — the trick above does that.
        tx_sorted[out_col] = _exp_nunique_prior(tx_sorted[col], tx_sorted["user_id"]) \
            .astype("int16")

    # Unique countries (via location_id → country). We need a country map; we
    # import lazily to avoid a circular dependency with the env module.
    from finsheild.synthetic_env import SCENARIO_NAMES  # noqa: F401  (import probe)

    # The engine computes prior_unique_countries from the locations table.
    # Do not return a placeholder here to avoid duplicate-column merges.
    pass

    return tx_sorted[["txn_id", "prior_tx_count", "prior_total_amount",
                       "prior_mean_amount", "prior_std_amount",
                       "amount_zscore", "amount_log_ratio",
                       "is_new_user", "prior_unique_merchants",
                       "prior_unique_devices", "prior_unique_locations"]]


FEATURE_COLUMNS = [
    "prior_tx_count", "prior_total_amount", "prior_mean_amount",
    "prior_std_amount", "amount_zscore", "amount_log_ratio",
    "is_new_user", "prior_unique_merchants", "prior_unique_devices",
    "prior_unique_locations",
]