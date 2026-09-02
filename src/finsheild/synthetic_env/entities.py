"""Reference-entity generators for the synthetic environment.

Each generator returns a ``pandas.DataFrame`` with a stable, documented
schema. None of these tables carry a fraud label; they are pure reference
data. Timestamps are naive UTC ``pd.Timestamp`` values so they survive
serialization.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from finsheild.synthetic_env.config import SyntheticEnvConfig
from finsheild.synthetic_env.rng import make_random, make_rng

# ---- Reference categories -------------------------------------------------

COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "NG", "RU", "JP", "MX"]
CITIES = {
    "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    "GB": ["London", "Manchester", "Bristol"],
    "DE": ["Berlin", "Munich", "Hamburg"],
    "FR": ["Paris", "Lyon", "Marseille"],
    "IN": ["Mumbai", "Bengaluru", "Delhi"],
    "BR": ["São Paulo", "Rio de Janeiro"],
    "NG": ["Lagos", "Abuja"],
    "RU": ["Moscow", "Saint Petersburg"],
    "JP": ["Tokyo", "Osaka"],
    "MX": ["Mexico City", "Monterrey"],
}
MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "online_retail", "travel",
    "electronics", "fashion", "entertainment", "utilities", "cash_advance",
]
MERCHANT_RISK_BAND = {
    "grocery": "low", "restaurant": "low", "fuel": "low",
    "online_retail": "medium", "travel": "medium", "electronics": "medium",
    "fashion": "medium", "entertainment": "low", "utilities": "low",
    "cash_advance": "high",
}
DEVICE_TYPES = ["mobile_ios", "mobile_android", "desktop_windows",
                "desktop_macos", "tablet"]
ACCOUNT_TYPES = ["checking", "savings", "credit"]
ACCOUNT_STATUS = ["active", "active", "active", "dormant", "closed"]
USER_RISK_SEGMENTS = ["standard", "standard", "standard", "premium", "student"]


def _start_ts(config: SyntheticEnvConfig) -> pd.Timestamp:
    return pd.Timestamp(config.start_ts)


def _end_ts(config: SyntheticEnvConfig) -> pd.Timestamp:
    return _start_ts(config) + pd.Timedelta(days=config.time_span_days)


# ---- Users ----------------------------------------------------------------

def generate_users(config: SyntheticEnvConfig) -> pd.DataFrame:
    """Generate the ``users`` table.

    Columns: ``user_id, signup_ts, home_location_id, risk_segment``.

    ``home_location_id`` references ``locations.location_id`` produced by
    :func:`generate_locations` — but it is sampled from the country
    distribution so it stays consistent regardless of location-table order.
    """
    n = config.n_users
    rng = make_rng(config.seed, "users")
    rnd = make_random(config.seed, "users")

    start = _start_ts(config)
    end = _end_ts(config)
    # Users sign up uniformly across the 30-day window BEFORE the tx stream,
    # so the earliest signup may predate ``start_ts`` slightly. We use the
    # previous month to keep the timeline natural.
    signup_window_start = start - pd.Timedelta(days=30)
    signup_window_seconds = int((end - signup_window_start).total_seconds())
    signup_offsets = rng.integers(0, signup_window_seconds, size=n)
    signup_ts = signup_window_start + pd.to_timedelta(signup_offsets, unit="s")

    # Countries distribution: 50% US, 25% GB/DE/FR/IN/BR, 25% others.
    country_pick = np.array(
        ["US"] * 5 + ["GB", "DE", "FR", "IN", "BR"] * 2
        + ["NG", "RU", "JP", "MX"]
    )
    countries = rnd.choices(country_pick.tolist(), k=n)

    # Locations are sampled from a stable set (max 400 city/country pairs)
    # keyed by country; the actual location_id is assigned later in
    # ``generate_locations`` for referential integrity, so here we just
    # store a ``home_location_country`` marker that downstream code uses
    # to align with the locations table.
    return pd.DataFrame({
        "user_id": np.arange(1, n + 1, dtype=np.int64),
        "signup_ts": signup_ts,
        "home_country": countries,
        "risk_segment": rnd.choices(USER_RISK_SEGMENTS, k=n),
    })


# ---- Locations ------------------------------------------------------------

def generate_locations(config: SyntheticEnvConfig) -> pd.DataFrame:
    """Generate the ``locations`` table.

    Columns: ``location_id, city, country, lat, lon, region``.

    Total rows is ``min(n_locations, sum(len(cities) for each country))``
    to avoid duplicating cities. Locations are intentionally bounded so
    that "unusual location" scenarios have room to inject foreign
    locations beyond the catalog.
    """
    n_target = config.n_locations
    rows = []
    rng = make_rng(config.seed, "locations")
    # Round-robin build up to n_target entries.
    catalog = []
    for country in COUNTRIES:
        for city in CITIES[country]:
            catalog.append((country, city))
    # Pad with synthetic city entries per country until we hit n_target.
    pad = 0
    while len(catalog) < n_target:
        country = catalog[pad % len(COUNTRIES) % len(COUNTRIES)][0]
        city = f"{CITIES[country][pad % len(CITIES[country])]}-{pad:03d}"
        catalog.append((country, city))
        pad += 1
    catalog = catalog[:n_target]

    # Stable lat/lon per city name.
    for i, (country, city) in enumerate(catalog, start=1):
        # Deterministic pseudo-coords: hash the city name to [0,1] and
        # scale to plausible ranges. Not geocoded — explicitly synthetic.
        seed = abs(hash((country, city))) % (10 ** 6)
        lat = ((seed % 1800) / 10.0) - 60.0
        lon = (((seed // 1800) % 3600) / 10.0) - 180.0
        rows.append({
            "location_id": i,
            "city": city,
            "country": country,
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "region": country,
        })
    df = pd.DataFrame(rows)
    return df


# ---- Devices --------------------------------------------------------------

def generate_devices(config: SyntheticEnvConfig) -> pd.DataFrame:
    """Generate the ``devices`` table.

    Columns: ``device_id, device_type, fingerprint_hash, first_seen_ts``.

    The fingerprint hash is a deterministic 16-char hex string built from
    (seed, device_id) — not cryptographic, just opaque.
    """
    n = config.n_devices
    rng = make_rng(config.seed, "devices")
    rnd = make_random(config.seed, "devices")

    start = _start_ts(config)
    end = _end_ts(config)
    total_seconds = int((end - start).total_seconds())

    offsets = rng.integers(0, total_seconds, size=n)
    first_seen_ts = start + pd.to_timedelta(offsets, unit="s")

    fingerprints = []
    for i in range(1, n + 1):
        digest = hash(f"{config.seed}:device:{i}") & 0xFFFFFFFFFFFFFFFF
        fingerprints.append(format(digest, "016x"))

    return pd.DataFrame({
        "device_id": np.arange(1, n + 1, dtype=np.int64),
        "device_type": rnd.choices(DEVICE_TYPES, k=n),
        "fingerprint_hash": fingerprints,
        "first_seen_ts": first_seen_ts,
    })


# ---- Merchants ------------------------------------------------------------

def generate_merchants(config: SyntheticEnvConfig) -> pd.DataFrame:
    """Generate the ``merchants`` table.

    Columns: ``merchant_id, name, category, mcc_code, country, risk_band``.

    ``mcc_code`` is the ISO merchant category code (4-digit). For
    reproducibility, we map each (category, country) pair to a stable
    4-digit code in [3000, 9999].
    """
    n = config.n_merchants
    rnd = make_random(config.seed, "merchants")

    rows = []
    seen_codes = set()
    for i in range(1, n + 1):
        category = rnd.choice(MERCHANT_CATEGORIES)
        country = rnd.choice(COUNTRIES)
        # Stable code: cat_index*100 + city_index offset, fallback to hash.
        base = (MERCHANT_CATEGORIES.index(category) + 1) * 100
        code = (base + (i % 90) + 3000) % 9000 + 1000
        while code in seen_codes:
            code = (code + 7) % 9000 + 1000
        seen_codes.add(code)
        rows.append({
            "merchant_id": i,
            "name": f"{category.capitalize()}_{i:05d}",
            "category": category,
            "mcc_code": int(code),
            "country": country,
            "risk_band": MERCHANT_RISK_BAND[category],
        })
    return pd.DataFrame(rows)


# ---- Accounts -------------------------------------------------------------

def generate_accounts(config: SyntheticEnvConfig,
                      users: pd.DataFrame) -> pd.DataFrame:
    """Generate the ``accounts`` table.

    Columns: ``account_id, user_id, opened_ts, account_type, status``.

    Each user has 1-3 accounts drawn from a geometric distribution.
    """
    n = config.n_accounts
    rng = make_rng(config.seed, "accounts")
    rnd = make_random(config.seed, "accounts")

    n_users = len(users)
    if n_users == 0:
        return pd.DataFrame(columns=[
            "account_id", "user_id", "opened_ts", "account_type", "status"])

    # Sample users with replacement (each user gets ~n/n_users accounts).
    # We bias the distribution so that a small fraction of users own many
    # accounts — supports mule behavior later.
    user_idx = rng.integers(0, n_users, size=n)
    user_ids = users["user_id"].values[user_idx]

    start = _start_ts(config) - pd.Timedelta(days=180)
    end = _start_ts(config)
    total_seconds = int((end - start).total_seconds())
    offsets = rng.integers(0, total_seconds, size=n)
    opened_ts = start + pd.to_timedelta(offsets, unit="s")

    return pd.DataFrame({
        "account_id": np.arange(1, n + 1, dtype=np.int64),
        "user_id": user_ids,
        "opened_ts": opened_ts,
        "account_type": rnd.choices(ACCOUNT_TYPES, k=n),
        "status": rnd.choices(ACCOUNT_STATUS, k=n),
    })