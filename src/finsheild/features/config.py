"""Feature-engineering config.

All numerical defaults are tuned for the synthetic dev scale; tests override
these. Changing a default does NOT change architecture — every default is
read at runtime from :class:`FeatureConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    """All knobs consumed by :func:`finsheild.features.build_features`.

    Attributes
    ----------
    velocity_windows_seconds:
        Sorted list of time-window widths (in seconds) used for velocity
        features ("how many txns happened in the last W seconds for this
        account?"). Default: 5 min, 1 h, 24 h.
    history_min_events:
        Minimum number of *prior* transactions a user must have for the
        behavioural features to be defined. Below this, behavioural
        features are filled with NaN — a downstream scaler can impute.
    min_user_history_for_deviation:
        Same as :attr:`history_min_events` but specifically for "deviation
        from user mean amount" feature (kept separate so the "new user"
        flag can have its own threshold).
    high_risk_categories:
        Merchant categories flagged as high-risk for the unusual_merchant
        signal. Pulled from the synthetic env's category list.
    high_value_threshold:
        Transactions strictly above this amount get the ``is_high_value``
        flag set (used by rules and the high_value_count velocity).
    fraud_label_col:
        Column name on the input transactions frame that carries the
        binary fraud label. Defaults to ``"label_fraud"``.
    """

    velocity_windows_seconds: tuple = (300, 3600, 86_400)
    history_min_events: int = 5
    min_user_history_for_deviation: int = 3
    high_risk_categories: frozenset = frozenset({"cash_advance"})
    high_value_threshold: float = 1000.0
    fraud_label_col: str = "label_fraud"

    def to_dict(self) -> dict:
        return {
            "velocity_windows_seconds": list(self.velocity_windows_seconds),
            "history_min_events": self.history_min_events,
            "min_user_history_for_deviation":
                self.min_user_history_for_deviation,
            "high_risk_categories": sorted(self.high_risk_categories),
            "high_value_threshold": self.high_value_threshold,
            "fraud_label_col": self.fraud_label_col,
        }