"""Eight inspectable suspicious scenarios.

Each scenario is a callable:

    def scenario(rng, rnd, ctx: ScenarioContext) -> List[ScenarioEvent]

Where ``ScenarioEvent`` is a small dataclass describing:

    * the original transaction to flag (``label_fraud=1`` and the scenario
      tag appended)
    * any "context" transactions the scenario generates (which remain
      ``label_fraud=0`` but carry a scenario tag so feature engineering can
      observe the surrounding behaviour)

Why context transactions?

Real fraud signals almost always have a "leading edge" of legitimate
behaviour that the attacker copies — high-velocity bursts sit on top of a
normal baseline, account-takeover begins with a normal login, etc. Without
context rows the model would over-fit to the scenario signature and miss
that scenarios occur within a stream of normal traffic.

Public API:

* :data:`SCENARIO_IMPLEMENTATIONS` — ordered list of callables. The order
  is the order in which they run during generation.
* :data:`SCENARIO_NAMES` — human-readable names in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import pandas as pd

from finsheild.synthetic_env.rng import make_random, make_rng

# Ordered list of all eight scenarios.
SCENARIO_NAMES = [
    "account_takeover",
    "unusual_amount_time",
    "transaction_velocity",
    "new_device",
    "unusual_location",
    "device_sharing",
    "mule_behavior",
    "unusual_merchant",
]

#: Default number of "episodes" each scenario produces at dev scale.
#: Tests override these to keep CI fast.
DEFAULT_SCENARIO_COUNTS = {
    "account_takeover": 30,
    "unusual_amount_time": 60,
    "transaction_velocity": 40,
    "new_device": 80,
    "unusual_location": 60,
    "device_sharing": 20,
    "mule_behavior": 15,
    "unusual_merchant": 50,
}


@dataclass
class ScenarioContext:
    """Read-only inputs every scenario receives.

    All big tables live here so scenarios can ``loc[index, ...]`` without
    copying. ``n_target_fraud`` is the budget for the *flagged*
    transactions this scenario should produce; context transactions are
    added on top.
    """

    accounts: pd.DataFrame
    users: pd.DataFrame
    devices: pd.DataFrame
    merchants: pd.DataFrame
    locations: pd.DataFrame
    account_devices: pd.DataFrame
    rng: np.random.Generator
    rnd: random.Random
    config_obj: "SyntheticEnvConfig"  # string annotation to avoid circular import
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp


@dataclass
class ScenarioEvent:
    """A single transaction emitted by a scenario.

    ``scenario_tag`` is the scenario name. ``label_fraud`` is 1 for the
    primary fraud transaction and 0 for surrounding context transactions.
    """

    account_id: int
    device_id: int
    merchant_id: int
    location_id: int
    ts: pd.Timestamp
    amount: float
    currency: str
    channel: str
    status: str
    scenario_tag: str
    label_fraud: int
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_account(ctx: ScenarioContext) -> pd.Series:
    return ctx.accounts.iloc[int(ctx.rng.integers(0, len(ctx.accounts)))]


def _account_devices_for(ctx: ScenarioContext, account_id: int) -> pd.DataFrame:
    return ctx.account_devices[ctx.account_devices["account_id"] == account_id]


def _random_ts(ctx: ScenarioContext) -> pd.Timestamp:
    offset = int(ctx.rng.integers(0, ctx.config_obj.total_seconds))
    return ctx.start_ts + pd.to_timedelta(offset, unit="s")


def _random_location(ctx: ScenarioContext) -> pd.Series:
    return ctx.locations.iloc[int(ctx.rng.integers(0, len(ctx.locations)))]


def _random_merchant(ctx: ScenarioContext) -> pd.Series:
    return ctx.merchants.iloc[int(ctx.rng.integers(0, len(ctx.merchants)))]


def _baseline_event(ctx: ScenarioContext, scenario_tag: str,
                     account_id: int) -> ScenarioEvent:
    """Return a normal-looking transaction under a given scenario tag."""
    dev = _account_devices_for(ctx, account_id)
    device_id = int(dev.iloc[0]["device_id"]) if len(dev) else int(
        ctx.rng.integers(1, len(ctx.devices) + 1))
    m = _random_merchant(ctx)
    loc = _random_location(ctx)
    return ScenarioEvent(
        account_id=int(account_id),
        device_id=int(device_id),
        merchant_id=int(m["merchant_id"]),
        location_id=int(loc["location_id"]),
        ts=_random_ts(ctx),
        amount=float(np.round(ctx.rng.lognormal(3.5, 0.9), 2)),
        currency="USD",
        channel=ctx.rnd.choice(["online", "pos", "atm", "mobile"]),
        status="settled",
        scenario_tag=scenario_tag,
        label_fraud=0,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def account_takeover(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """Account takeover: a few "normal" txns then a flagged large foreign tx."""
    n = DEFAULT_SCENARIO_COUNTS["account_takeover"]
    out: List[ScenarioEvent] = []
    for _ in range(n):
        acc = _random_account(ctx)
        account_id = int(acc["account_id"])
        # 1-2 normal baseline transactions
        for _ in range(int(ctx.rng.integers(1, 3))):
            out.append(_baseline_event(ctx, "account_takeover", account_id))
        # The takeover transaction: high amount, new location, foreign country
        out.append(ScenarioEvent(
            account_id=account_id,
            device_id=int(_account_devices_for(ctx, account_id).iloc[0]["device_id"]),
            merchant_id=int(_random_merchant(ctx)["merchant_id"]),
            location_id=int(_random_location(ctx)["location_id"]),
            ts=_random_ts(ctx),
            amount=float(np.round(ctx.rng.lognormal(7.5, 0.4), 2)),
            currency="USD",
            channel="online",
            status="settled",
            scenario_tag="account_takeover",
            label_fraud=1,
            extra={"signal_high_amount": True, "signal_foreign": True},
        ))
    return out


def unusual_amount_time(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """Late-night, high-value, atypical-amount purchase flagged as fraud."""
    n = DEFAULT_SCENARIO_COUNTS["unusual_amount_time"]
    out: List[ScenarioEvent] = []
    for _ in range(n):
        acc = _random_account(ctx)
        account_id = int(acc["account_id"])
        # baseline
        out.append(_baseline_event(ctx, "unusual_amount_time", account_id))
        # flagged: 2-5 AM, 5-15x baseline amount
        ts = ctx.start_ts + pd.Timedelta(
            days=int(ctx.rng.integers(0, ctx.config_obj.time_span_days)))
        ts = ts.replace(hour=int(ctx.rng.integers(2, 6)),
                         minute=int(ctx.rng.integers(0, 60)))
        out.append(ScenarioEvent(
            account_id=account_id,
            device_id=int(_account_devices_for(ctx, account_id).iloc[0]["device_id"]),
            merchant_id=int(_random_merchant(ctx)["merchant_id"]),
            location_id=int(_random_location(ctx)["location_id"]),
            ts=ts,
            amount=float(np.round(ctx.rng.uniform(800.0, 4500.0), 2)),
            currency="USD",
            channel="online",
            status="settled",
            scenario_tag="unusual_amount_time",
            label_fraud=1,
            extra={"signal_offhours": True, "signal_high_amount": True},
        ))
    return out


def transaction_velocity(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """Burst of 8-12 transactions within ~10 minutes; the burst is fraud."""
    n = DEFAULT_SCENARIO_COUNTS["transaction_velocity"]
    out: List[ScenarioEvent] = []
    for _ in range(n):
        acc = _random_account(ctx)
        account_id = int(acc["account_id"])
        # baseline
        out.append(_baseline_event(ctx, "transaction_velocity", account_id))
        # burst — same anchor time, +0..600s
        anchor = _random_ts(ctx)
        burst_n = int(ctx.rng.integers(8, 13))
        device_id = int(_account_devices_for(ctx, account_id).iloc[0]["device_id"])
        for k in range(burst_n):
            ts = anchor + pd.Timedelta(seconds=int(ctx.rng.integers(0, 600)))
            out.append(ScenarioEvent(
                account_id=account_id,
                device_id=device_id,
                merchant_id=int(_random_merchant(ctx)["merchant_id"]),
                location_id=int(_random_location(ctx)["location_id"]),
                ts=ts,
                amount=float(np.round(ctx.rng.uniform(5.0, 250.0), 2)),
                currency="USD",
                channel="online",
                status="settled",
                scenario_tag="transaction_velocity",
                label_fraud=1 if k < burst_n - 1 else 1,
                extra={"signal_velocity_burst": True},
            ))
    return out


def new_device(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """A transaction from a device this account has never used before."""
    n = DEFAULT_SCENARIO_COUNTS["new_device"]
    out: List[ScenarioEvent] = []
    for _ in range(n):
        acc = _random_account(ctx)
        account_id = int(acc["account_id"])
        # baseline
        out.append(_baseline_event(ctx, "new_device", account_id))
        # Pick a device NOT in this account's known set
        known = set(_account_devices_for(ctx, account_id)["device_id"].tolist())
        all_dev_ids = list(ctx.devices["device_id"].tolist())
        candidates = [d for d in all_dev_ids if d not in known]
        if not candidates:
            continue
        new_device_id = int(ctx.rnd.choice(candidates))
        out.append(ScenarioEvent(
            account_id=account_id,
            device_id=new_device_id,
            merchant_id=int(_random_merchant(ctx)["merchant_id"]),
            location_id=int(_random_location(ctx)["location_id"]),
            ts=_random_ts(ctx),
            amount=float(np.round(ctx.rng.lognormal(5.0, 0.7), 2)),
            currency="USD",
            channel="online",
            status="settled",
            scenario_tag="new_device",
            label_fraud=1,
            extra={"signal_new_device": True},
        ))
    return out


def unusual_location(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """A transaction in a country the user has never transacted from."""
    n = DEFAULT_SCENARIO_COUNTS["unusual_location"]
    out: List[ScenarioEvent] = []
    for _ in range(n):
        acc = _random_account(ctx)
        account_id = int(acc["account_id"])
        home = ctx.users.loc[ctx.users["user_id"] == acc["user_id"]].iloc[0]
        home_country = home["home_country"]
        foreign_locs = ctx.locations[ctx.locations["country"] != home_country]
        if len(foreign_locs) == 0:
            continue
        out.append(_baseline_event(ctx, "unusual_location", account_id))
        loc = foreign_locs.iloc[int(ctx.rng.integers(0, len(foreign_locs)))]
        out.append(ScenarioEvent(
            account_id=account_id,
            device_id=int(_account_devices_for(ctx, account_id).iloc[0]["device_id"]),
            merchant_id=int(_random_merchant(ctx)["merchant_id"]),
            location_id=int(loc["location_id"]),
            ts=_random_ts(ctx),
            amount=float(np.round(ctx.rng.lognormal(5.5, 0.8), 2)),
            currency="USD",
            channel="online",
            status="settled",
            scenario_tag="unusual_location",
            label_fraud=1,
            extra={"signal_foreign_country": True,
                   "home_country": str(home_country),
                   "tx_country": str(loc["country"])},
        ))
    return out


def device_sharing(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """Multiple accounts using one device; one transaction is the mule hit."""
    n = DEFAULT_SCENARIO_COUNTS["device_sharing"]
    out: List[ScenarioEvent] = []
    for _ in range(n):
        # Choose a device that has multiple accounts.
        device_id = int(ctx.rng.integers(1, len(ctx.devices) + 1))
        acc_devs = ctx.account_devices[ctx.account_devices["device_id"] == device_id]
        account_ids = acc_devs["account_id"].unique().tolist()
        if len(account_ids) < 2:
            # fallback: pick two random accounts and force them to share
            acc_a = int(ctx.accounts.iloc[int(ctx.rng.integers(0, len(ctx.accounts)))]["account_id"])
            acc_b = int(ctx.accounts.iloc[int(ctx.rng.integers(0, len(ctx.accounts)))]["account_id"])
            if acc_a == acc_b:
                continue
            account_ids = [acc_a, acc_b]
        # baseline on account_a
        out.append(_baseline_event(ctx, "device_sharing", account_ids[0]))
        # mule hit on account_b: high amount, same device
        out.append(ScenarioEvent(
            account_id=int(account_ids[1]),
            device_id=device_id,
            merchant_id=int(_random_merchant(ctx)["merchant_id"]),
            location_id=int(_random_location(ctx)["location_id"]),
            ts=_random_ts(ctx),
            amount=float(np.round(ctx.rng.lognormal(6.5, 0.6), 2)),
            currency="USD",
            channel="online",
            status="settled",
            scenario_tag="device_sharing",
            label_fraud=1,
            extra={"signal_shared_device": True,
                   "shared_with_accounts": [int(a) for a in account_ids[:5]]},
        ))
    return out


def mule_behavior(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """Fan-in: many small deposits into one mule account, then a big outflow."""
    n = DEFAULT_SCENARIO_COUNTS["mule_behavior"]
    out: List[ScenarioEvent] = []
    for _ in range(n):
        mule = _random_account(ctx)
        mule_id = int(mule["account_id"])
        mule_dev_id = int(
            _account_devices_for(ctx, mule_id).iloc[0]["device_id"])
        # Several inbound transfers (context) — small, fast.
        n_inbound = int(ctx.rng.integers(4, 9))
        anchor = _random_ts(ctx)
        for k in range(n_inbound):
            ts = anchor - pd.Timedelta(seconds=int(ctx.rng.integers(60, 3600)))
            out.append(ScenarioEvent(
                account_id=mule_id,
                device_id=mule_dev_id,
                merchant_id=int(_random_merchant(ctx)["merchant_id"]),
                location_id=int(_random_location(ctx)["location_id"]),
                ts=ts,
                amount=float(np.round(ctx.rng.uniform(20.0, 200.0), 2)),
                currency="USD",
                channel="online",
                status="settled",
                scenario_tag="mule_behavior",
                label_fraud=0,  # inbound leg is the context
                extra={"signal_mule_inbound": True},
            ))
        # The flagged big outbound transfer.
        out.append(ScenarioEvent(
            account_id=mule_id,
            device_id=mule_dev_id,
            merchant_id=int(_random_merchant(ctx)["merchant_id"]),
            location_id=int(_random_location(ctx)["location_id"]),
            ts=anchor,
            amount=float(np.round(ctx.rng.uniform(1500.0, 8000.0), 2)),
            currency="USD",
            channel="online",
            status="settled",
            scenario_tag="mule_behavior",
            label_fraud=1,
            extra={"signal_mule_outbound": True},
        ))
    return out


def unusual_merchant(ctx: ScenarioContext) -> List[ScenarioEvent]:
    """User transacts at a high-risk merchant they have never used."""
    n = DEFAULT_SCENARIO_COUNTS["unusual_merchant"]
    out: List[ScenarioEvent] = []
    high_risk = ctx.merchants[ctx.merchants["risk_band"] == "high"]
    if len(high_risk) == 0:
        return out
    for _ in range(n):
        acc = _random_account(ctx)
        account_id = int(acc["account_id"])
        out.append(_baseline_event(ctx, "unusual_merchant", account_id))
        m = high_risk.iloc[int(ctx.rng.integers(0, len(high_risk)))]
        out.append(ScenarioEvent(
            account_id=account_id,
            device_id=int(_account_devices_for(ctx, account_id).iloc[0]["device_id"]),
            merchant_id=int(m["merchant_id"]),
            location_id=int(_random_location(ctx)["location_id"]),
            ts=_random_ts(ctx),
            amount=float(np.round(ctx.rng.uniform(500.0, 3500.0), 2)),
            currency="USD",
            channel="online",
            status="settled",
            scenario_tag="unusual_merchant",
            label_fraud=1,
            extra={"signal_high_risk_merchant": True,
                   "merchant_risk_band": "high"},
        ))
    return out


# Public ordered mapping.
SCENARIO_IMPLEMENTATIONS: List[Callable[[ScenarioContext], List[ScenarioEvent]]] = [
    account_takeover,
    unusual_amount_time,
    transaction_velocity,
    new_device,
    unusual_location,
    device_sharing,
    mule_behavior,
    unusual_merchant,
]