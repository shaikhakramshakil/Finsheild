"""Top-level feature orchestrator.

Combines all five feature families into a single per-transaction
DataFrame. Output is indexed by ``txn_id`` and is the canonical input to
the supervised models in Phases 3+ and to the risk-fusion engine in
Phase 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from finsheild.features.behavioral import (
    FEATURE_COLUMNS as BEHAVIORAL_COLS,
    build_behavioral_features,
)
from finsheild.features.config import FeatureConfig
from finsheild.features.device import (
    FEATURE_COLUMNS as DEVICE_COLS,
    build_device_features,
)
from finsheild.features.location import (
    FEATURE_COLUMNS as LOCATION_COLS,
    build_location_features,
)
from finsheild.features.transactional import (
    FEATURE_COLUMNS as TRANSACTIONAL_COLS,
    build_transactional_features,
)
from finsheild.features.velocity import (
    build_velocity_features,
    feature_columns as velocity_feature_columns,
)


@dataclass
class FeatureBuildResult:
    """Output of :func:`build_features`."""

    features: pd.DataFrame
    feature_columns: List[str]
    metadata: Dict = field(default_factory=dict)

    def X(self) -> np.ndarray:
        return self.features[self.feature_columns].to_numpy(dtype="float32")

    def y(self) -> np.ndarray:
        return self.features["label_fraud"].to_numpy(dtype="int8") \
            if "label_fraud" in self.features.columns else None


def build_features(env, config: FeatureConfig | None = None) -> FeatureBuildResult:
    """Build all features for a synthetic environment.

    Parameters
    ----------
    env:
        A :class:`finsheild.synthetic_env.SyntheticEnvironment` (Phase 4
        output) or any object with the same ``tables`` dict
        (``users``, ``accounts``, ``devices``, ``merchants``,
        ``locations``, ``account_devices``, ``transactions``).
    config:
        Optional :class:`FeatureConfig`. Defaults to ``FeatureConfig()``.

    Returns
    -------
    :class:`FeatureBuildResult` with ``.features`` (DataFrame indexed by
    ``txn_id``), ``.feature_columns`` (the ordered list of feature
    columns), and ``.metadata`` (timings + config snapshot).
    """
    if config is None:
        config = FeatureConfig()
    tx = env.transactions
    merchants = env.merchants
    accounts = env.accounts
    users = env.users
    devices = env.devices
    locations = env.locations
    account_devices = env.account_devices

    import time
    timings: Dict[str, float] = {}
    t0 = time.time()

    f_tx = build_transactional_features(tx, merchants,
                                          config.high_value_threshold)
    timings["transactional_s"] = time.time() - t0

    t0 = time.time()
    f_bx = build_behavioral_features(tx, accounts,
                                      config.history_min_events)
    # Add prior_unique_countries via locations join
    loc_country = locations.set_index("location_id")["country"]
    f_bx["tx_country"] = tx.set_index("txn_id").loc[f_bx["txn_id"], "location_id"] \
        .map(loc_country).values
    # Per-account set of countries seen prior to this row
    user_map = accounts.set_index("account_id")["user_id"]
    tx_user = tx[["txn_id", "account_id"]].copy()
    tx_user["user_id"] = tx_user["account_id"].map(user_map)
    country_per_user = tx_user.merge(
        f_bx[["txn_id", "tx_country"]], on="txn_id", how="left")
    country_per_user = country_per_user.sort_values(
        ["user_id", "txn_id"]).reset_index(drop=True)

    counts = np.zeros(len(country_per_user), dtype="int32")
    seen: dict = {}
    for i, row in enumerate(country_per_user.itertuples(index=False)):
        s = seen.setdefault(row.user_id, set())
        if row.tx_country is not None and pd.notna(row.tx_country):
            counts[i] = len(s)
            s.add(row.tx_country)
        else:
            counts[i] = len(s)
    country_per_user = country_per_user.assign(prior_unique_countries=counts.astype("int32"))
    puc_map = dict(zip(country_per_user["txn_id"],
                       country_per_user["prior_unique_countries"]))
    f_bx["prior_unique_countries"] = f_bx["txn_id"].map(puc_map).fillna(0) \
        .astype("int16")
    timings["behavioral_s"] = time.time() - t0

    t0 = time.time()
    f_vx = build_velocity_features(
        tx, tuple(config.velocity_windows_seconds),
        config.high_value_threshold)
    timings["velocity_s"] = time.time() - t0

    t0 = time.time()
    f_dx = build_device_features(tx, account_devices)
    timings["device_s"] = time.time() - t0

    t0 = time.time()
    f_lx = build_location_features(tx, locations, accounts, users)
    timings["location_s"] = time.time() - t0

    # Merge all on txn_id; preserve order of the original transactions.
    out = tx[["txn_id", "account_id", "ts", "amount", config.fraud_label_col,
              "scenario_tag"]].copy()
    out = out.merge(f_tx, on="txn_id", how="left")
    out = out.merge(f_bx, on="txn_id", how="left")
    out = out.merge(f_vx, on="txn_id", how="left")
    out = out.merge(f_dx, on="txn_id", how="left")
    out = out.merge(f_lx, on="txn_id", how="left")

    feature_columns = (TRANSACTIONAL_COLS + BEHAVIORAL_COLS
                       + ["prior_unique_countries"]
                       + velocity_feature_columns(tuple(config.velocity_windows_seconds))
                       + DEVICE_COLS + LOCATION_COLS)

    metadata = {
        "config": config.to_dict(),
        "n_rows": len(out),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "timings_s": timings,
    }
    return FeatureBuildResult(features=out,
                               feature_columns=feature_columns,
                               metadata=metadata)