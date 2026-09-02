"""Config for the synthetic digital-payment environment.

Every numeric default here is a *development* scale chosen so the env
generates quickly enough for CI while still exercising every code path:

* ``n_users=2000`` — plenty of distinct behavioral profiles
* ``n_merchants=500`` — broad enough category mix
* ``n_transactions=50_000`` — ~30 days × ~1700 tx/day, large enough
  for non-trivial velocity signals but small enough to run in <30 s

CI should override these via :class:`SyntheticEnvConfig` to keep tests
fast. The tests in ``tests/test_synthetic_env.py`` use ``CI_DEFAULTS``
which sets roughly 50 users / 5 000 transactions.

Changing the defaults does NOT change architecture — they are passed
through everywhere as parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


#: Defaults used by unit tests so they stay fast.
CI_DEFAULTS = {
    "n_users": 200,
    "n_accounts": 250,
    "n_devices": 220,
    "n_merchants": 80,
    "n_locations": 60,
    "n_transactions": 5_000,
    "time_span_days": 30,
    "fraud_rate": 0.03,
    "seed": 1729,
}


#: Defaults used by ``scripts/generate_synthetic_env.py``.
DEV_DEFAULTS = {
    "n_users": 2_000,
    "n_accounts": 3_000,
    "n_devices": 2_500,
    "n_merchants": 500,
    "n_locations": 400,
    "n_transactions": 50_000,
    "time_span_days": 30,
    "fraud_rate": 0.03,
    "seed": 42,
}


@dataclass(frozen=True)
class SyntheticEnvConfig:
    """All knobs controlling environment generation.

    Attributes
    ----------
    seed:
        Master RNG seed — ``random``, ``numpy`` and ``pandas`` are all
        re-seeded from this so the same config yields identical output.
    n_users, n_accounts, n_devices, n_merchants, n_locations, n_transactions:
        Sizes of the generated entity tables. The generator enforces
        referential integrity (e.g. accounts.users ⊆ users.user_id) but
        does not enforce ``n_accounts >= n_users``.
    time_span_days:
        Window length of the transaction stream, in days, ending at
        ``end_ts = pd.Timestamp("2026-01-01") + time_span_days``. All
        timestamps fall in ``[end_ts - time_span_days, end_ts]``.
    fraud_rate:
        Target fraction of transactions to mark as fraud. Each scenario
        contributes roughly equally; small deviations are expected because
        the scenarios also produce "innocent" context transactions.
    start_ts:
        Wall-clock start of the transaction stream. Pinned so that
        re-generation yields identical timestamps.
    """

    seed: int = 42
    n_users: int = 2_000
    n_accounts: int = 3_000
    n_devices: int = 2_500
    n_merchants: int = 500
    n_locations: int = 400
    n_transactions: int = 50_000
    time_span_days: int = 30
    fraud_rate: float = 0.03
    start_ts: str = "2026-01-01"

    @classmethod
    def ci(cls) -> "SyntheticEnvConfig":
        return cls(**CI_DEFAULTS)

    @classmethod
    def dev(cls) -> "SyntheticEnvConfig":
        return cls(**DEV_DEFAULTS)

    @property
    def end_ts(self) -> str:
        # Documented for readers; the actual computation lives in entities.
        from datetime import datetime, timedelta

        start = datetime.fromisoformat(self.start_ts)
        end = start + timedelta(days=self.time_span_days)
        return end.isoformat()

    @property
    def total_seconds(self) -> int:
        return self.time_span_days * 86_400

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_users": self.n_users,
            "n_accounts": self.n_accounts,
            "n_devices": self.n_devices,
            "n_merchants": self.n_merchants,
            "n_locations": self.n_locations,
            "n_transactions": self.n_transactions,
            "time_span_days": self.time_span_days,
            "fraud_rate": self.fraud_rate,
            "start_ts": self.start_ts,
        }