"""Hard-overlap fraud scenarios.

Each scenario emits a *blend* of weak/moderate signals so that no single
feature is a perfect separator. Amounts, locations, devices, hours and
merchants all overlap with the legitimate background.

All randomness is seeded off the master ``config.seed`` so the result is
reproducible. This module is untouched by the easy `scenarios.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from finsheild.synthetic_env.config import SyntheticEnvConfig
from finsheild.synthetic_env.rng import make_random, make_rng


@dataclass
class HardFraud:
    account_id: int
    device_id: int
    merchant_id: int
    location_id: int
    ts: pd.Timestamp
    amount: float
    channel: str
    scenario_tag: str
    label_fraud: int = 1
    extra: Dict[str, str] = None  # type: ignore


SCENARIO_NAMES_HARD = [
    "hard_moderate_amo_dev",
    "hard_normal_amount_new_device_merchant",
    "hard_high_amount_normal_location_velocity",
    "hard_normal_amount_normal_device_merchant",
    "hard_mixed_signals",
]


def _pick_account(rng, rnd, accounts):
    return int(accounts.iloc[int(rng.integers(0, len(accounts)))]["account_id"])


def _pick_device_for_account(rng, rnd, ad, account_id):
    sub = ad[ad["account_id"] == account_id]
    if len(sub) == 0:
        return int(rng.integers(1, 1))
    return int(rnd.choice(sub["device_id"].tolist()))


def _pick_random_merchant(rng, merchants):
    return int(rng.choice(merchants["merchant_id"].to_numpy()))


def _pick_high_risk_merchant(rng, merchants):
    high = merchants[merchants["risk_band"] == "high"]
    if len(high) == 0:
        return _pick_random_merchant(rng, merchants)
    return int(rng.choice(high["merchant_id"].to_numpy()))


def _pick_random_location(rng, locations):
    return int(rng.choice(locations["location_id"].to_numpy()))


def _pick_foreign_location(rng, locations, user_country):
    foreign = locations[locations["country"] != user_country]
    if len(foreign) == 0:
        return _pick_random_location(rng, locations)
    return int(rng.choice(foreign["location_id"].to_numpy()))


def _pick_near_home_location(rng, locations, user_country):
    pool = locations[locations["country"] == user_country]
    if len(pool) == 0:
        return _pick_random_location(rng, locations)
    return int(rng.choice(pool["location_id"].to_numpy()))


def _random_ts(rng, start, total_seconds):
    return start + pd.Timedelta(seconds=int(rng.integers(0, total_seconds)))


def _random_business_hour_ts(rng, start, total_seconds):
    offset = int(rng.integers(0, total_seconds))
    base = start + pd.Timedelta(seconds=offset)
    return base.replace(hour=int(rng.integers(9, 19)))


def _offhours_ts(rng, start, total_seconds):
    offset = int(rng.integers(0, total_seconds))
    base = start + pd.Timedelta(seconds=offset)
    return base.replace(hour=int(rng.integers(2, 5)))


def hard_moderate_amo_dev(rng, rnd, ctx):
    out = []
    for _ in range(ctx["n_each"]):
        acc_id = _pick_account(rng, rnd, ctx["accounts"])
        user = ctx["users"].set_index("user_id")["home_country"]
        acc_user = ctx["accounts"].set_index("account_id")["user_id"]
        home = user.get(acc_user.get(acc_id, 0), "US")
        known = set(ctx["ad"][ctx["ad"]["account_id"] == acc_id]["device_id"].tolist())
        others = [d for d in ctx["devices"]["device_id"].tolist() if d not in known]
        dev = int(rnd.choice(others)) if others else _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
        amount = float(np.round(rng.lognormal(3.9, 0.6), 2))
        out.append(HardFraud(account_id=acc_id, device_id=dev, merchant_id=_pick_random_merchant(rng, ctx["merchants"]), location_id=_pick_near_home_location(rng, ctx["locations"], home), ts=_random_business_hour_ts(rng, ctx["start"], ctx["total_seconds"]), amount=amount, channel="online", scenario_tag="hard_moderate_amo_dev", extra={"signal": "moderate_amount+new_device"}))
    return out


def hard_normal_amount_new_device_merchant(rng, rnd, ctx):
    out = []
    for _ in range(ctx["n_each"]):
        acc_id = _pick_account(rng, rnd, ctx["accounts"])
        user = ctx["users"].set_index("user_id")["home_country"]
        acc_user = ctx["accounts"].set_index("account_id")["user_id"]
        home = user.get(acc_user.get(acc_id, 0), "US")
        known = set(ctx["ad"][ctx["ad"]["account_id"] == acc_id]["device_id"].tolist())
        others = [d for d in ctx["devices"]["device_id"].tolist() if d not in known]
        dev = int(rnd.choice(others)) if others else _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
        amount = float(np.round(rng.lognormal(3.5, 0.5), 2))
        m = _pick_high_risk_merchant(rng, ctx["merchants"]) if rng.random() < 0.5 else _pick_random_merchant(rng, ctx["merchants"])
        out.append(HardFraud(account_id=acc_id, device_id=dev, merchant_id=m, location_id=_pick_near_home_location(rng, ctx["locations"], home), ts=_random_business_hour_ts(rng, ctx["start"], ctx["total_seconds"]), amount=amount, channel="online", scenario_tag="hard_normal_amount_new_device_merchant", extra={"signal": "normal_amount+new_device+merchant"}))
    return out


def hard_high_amount_normal_location_velocity(rng, rnd, ctx):
    out = []
    for _ in range(ctx["n_each"]):
        acc_id = _pick_account(rng, rnd, ctx["accounts"])
        user = ctx["users"].set_index("user_id")["home_country"]
        acc_user = ctx["accounts"].set_index("account_id")["user_id"]
        home = user.get(acc_user.get(acc_id, 0), "US")
        dev = _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
        amount = float(np.round(rng.lognormal(5.5, 0.5), 2))
        out.append(HardFraud(account_id=acc_id, device_id=dev, merchant_id=_pick_random_merchant(rng, ctx["merchants"]), location_id=_pick_near_home_location(rng, ctx["locations"], home), ts=_random_business_hour_ts(rng, ctx["start"], ctx["total_seconds"]), amount=amount, channel="online", scenario_tag="hard_high_amount_normal_location_velocity", extra={"signal": "high_amount_only"}))
    return out


def hard_normal_amount_normal_device_merchant(rng, rnd, ctx):
    out = []
    for _ in range(ctx["n_each"]):
        acc_id = _pick_account(rng, rnd, ctx["accounts"])
        user = ctx["users"].set_index("user_id")["home_country"]
        acc_user = ctx["accounts"].set_index("account_id")["user_id"]
        home = user.get(acc_user.get(acc_id, 0), "US")
        dev = _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
        ts = _offhours_ts(rng, ctx["start"], ctx["total_seconds"]) if rng.random() < 0.3 else _random_business_hour_ts(rng, ctx["start"], ctx["total_seconds"])
        amount = float(np.round(rng.lognormal(3.7, 0.7), 2))
        loc = _pick_foreign_location(rng, ctx["locations"], home) if rng.random() < 0.2 else _pick_near_home_location(rng, ctx["locations"], home)
        out.append(HardFraud(account_id=acc_id, device_id=dev, merchant_id=_pick_random_merchant(rng, ctx["merchants"]), location_id=loc, ts=ts, amount=amount, channel="online", scenario_tag="hard_normal_amount_normal_device_merchant", extra={"signal": "weak_signals_only"}))
    return out


def hard_mixed_signals(rng, rnd, ctx):
    out = []
    for _ in range(ctx["n_each"]):
        acc_id = _pick_account(rng, rnd, ctx["accounts"])
        user = ctx["users"].set_index("user_id")["home_country"]
        acc_user = ctx["accounts"].set_index("account_id")["user_id"]
        home = user.get(acc_user.get(acc_id, 0), "US")
        roll = rng.random()
        if roll < 0.25:
            amount = float(np.round(rng.lognormal(4.0, 0.4), 2))
            loc = _pick_near_home_location(rng, ctx["locations"], home)
            ts = _random_business_hour_ts(rng, ctx["start"], ctx["total_seconds"])
            dev = _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
            m = _pick_random_merchant(rng, ctx["merchants"])
        elif roll < 0.5:
            amount = float(np.round(rng.lognormal(3.6, 0.5), 2))
            loc = _pick_foreign_location(rng, ctx["locations"], home)
            ts = _random_business_hour_ts(rng, ctx["start"], ctx["total_seconds"])
            dev = _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
            m = _pick_random_merchant(rng, ctx["merchants"])
        elif roll < 0.75:
            amount = float(np.round(rng.lognormal(3.5, 0.6), 2))
            loc = _pick_near_home_location(rng, ctx["locations"], home)
            ts = _offhours_ts(rng, ctx["start"], ctx["total_seconds"])
            known = set(ctx["ad"][ctx["ad"]["account_id"] == acc_id]["device_id"].tolist())
            others = [d for d in ctx["devices"]["device_id"].tolist() if d not in known]
            dev = int(rnd.choice(others)) if others else _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
            m = _pick_random_merchant(rng, ctx["merchants"])
        else:
            amount = float(np.round(rng.lognormal(3.7, 0.6), 2))
            loc = _pick_near_home_location(rng, ctx["locations"], home)
            ts = _random_business_hour_ts(rng, ctx["start"], ctx["total_seconds"])
            dev = _pick_device_for_account(rng, rnd, ctx["ad"], acc_id)
            m = _pick_high_risk_merchant(rng, ctx["merchants"])
        out.append(HardFraud(account_id=acc_id, device_id=dev, merchant_id=m, location_id=loc, ts=ts, amount=amount, channel="online", scenario_tag="hard_mixed_signals", extra={"signal": f"combo_{int(roll*4)}"}))
    return out


HARD_SCENARIO_IMPLEMENTATIONS = [
    hard_moderate_amo_dev,
    hard_normal_amount_new_device_merchant,
    hard_high_amount_normal_location_velocity,
    hard_normal_amount_normal_device_merchant,
    hard_mixed_signals,
]


def build_hard_overlap_events(config, accounts, users, devices, merchants, locations, account_devices, n_per_scenario: int):
    rng = make_rng(config.seed, "hard_overlap_rng")
    rnd = make_random(config.seed, "hard_overlap_random")
    ctx = {"accounts": accounts, "users": users, "devices": devices, "merchants": merchants, "locations": locations, "ad": account_devices, "start": pd.Timestamp(config.start_ts), "total_seconds": config.total_seconds, "n_each": n_per_scenario}
    events = []
    for fn in HARD_SCENARIO_IMPLEMENTATIONS:
        events.extend(fn(rng, rnd, ctx))
    return events
