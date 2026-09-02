"""Location features — country switch indicators and great-circle distance.

Leakage rule
------------
For transaction at time ``t``:
* ``prev_location_id`` is the location_id of the *most recent prior*
  transaction for the same account, where ``prior ts < t``.
* ``country_switch`` is 1 if ``tx.location.country != prev.country``.
* ``distance_km`` is the great-circle distance between the two locations
  (using their lat/lon). Uses ``prev_location_id`` lookup.
* ``is_unusual_location`` is 1 if the location's country is not the
  account's home country (joined from users.home_country).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between two (lat, lon) pairs in degrees."""
    R = 6371.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_location_features(tx: pd.DataFrame,
                              locations: pd.DataFrame,
                              accounts: pd.DataFrame,
                              users: pd.DataFrame) -> pd.DataFrame:
    """Return per-transaction location features.

    Output columns: ``txn_id, prev_location_id, country_switch,
    distance_to_prev_km, is_unusual_location``.
    """
    # Lookups
    loc_country = locations.set_index("location_id")["country"]
    loc_lat = locations.set_index("location_id")["lat"]
    loc_lon = locations.set_index("location_id")["lon"]
    acc_user = accounts.set_index("account_id")["user_id"]
    user_home = users.set_index("user_id")["home_country"]

    # Find previous location_id per account, with ts strictly less than
    # current ts. Vectorised: sort by (account_id, ts), forward-fill prior.
    base = tx[["txn_id", "account_id", "location_id", "ts"]].copy()
    base["user_id"] = base["account_id"].map(acc_user)
    base["home_country"] = base["user_id"].map(user_home)
    base["tx_country"] = base["location_id"].map(loc_country)

    base = base.sort_values(["account_id", "ts", "txn_id"]).reset_index(drop=True)
    base["prev_location_id"] = base.groupby("account_id")["location_id"].shift(1)
    # country_switch: 1 if previous exists and country differs
    base["country_switch"] = (
        (base["prev_location_id"].notna())
        & (base["tx_country"] != base["prev_location_id"].map(loc_country))
    ).astype("int8")
    base["is_unusual_location"] = (
        (base["home_country"].notna())
        & (base["tx_country"] != base["home_country"])
    ).astype("int8")

    # Distance to previous location (NaN if no prior)
    prev_lat = base["prev_location_id"].map(loc_lat)
    prev_lon = base["prev_location_id"].map(loc_lon)
    cur_lat = base["location_id"].map(loc_lat)
    cur_lon = base["location_id"].map(loc_lon)
    base["distance_to_prev_km"] = _haversine_km(
        cur_lat.fillna(0).to_numpy(),
        cur_lon.fillna(0).to_numpy(),
        prev_lat.fillna(0).to_numpy(),
        prev_lon.fillna(0).to_numpy(),
    )
    # Mask distance when no prior (set NaN, not 0)
    base.loc[base["prev_location_id"].isna(), "distance_to_prev_km"] = np.nan
    # Impute "0" (same location) for first transactions
    base["distance_to_prev_km"] = base["distance_to_prev_km"].fillna(0.0) \
        .astype("float32")

    base = base.sort_values("txn_id").reset_index(drop=True)
    return base[["txn_id", "prev_location_id", "country_switch",
                 "distance_to_prev_km", "is_unusual_location"]]


FEATURE_COLUMNS = ["country_switch", "distance_to_prev_km", "is_unusual_location"]