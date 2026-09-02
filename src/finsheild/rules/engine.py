"""Rule engine — Phase 8 deterministic fraud checks.

Eight rules operate on a single *feature row* (one transaction's engineered
features). Each rule is a pure callable ``condition(row) -> bool``; the
engine evaluates all rules and returns a :class:`RuleResult` per rule.

All numeric thresholds are configurable via the ``thresholds`` dict passed
to :class:`RuleEngine`. The global :data:`DEFAULT_RULES` use default
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import math

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

VALID_SEVERITIES = frozenset({"low", "medium", "high", "critical"})

# Canonical thresholds — keys are stable public names.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "high_velocity": 5,  # vel_count_300s >= 5
    "burst_velocity": 8,  # vel_count_300s >= 8
    "new_device_high_value_amount": 500,  # amount > 500
    "unusual_amount_zscore": 3.0,  # abs(amount_zscore) > 3
    "offhours_high_value_amount": 1000,  # amount > 1000
}

# Alias → canonical so callers can use alternative names without breakage.
_THRESHOLD_ALIASES: Dict[str, str] = {
    "high_velocity_threshold": "high_velocity",
    "vel_count_300s": "high_velocity",
    "vel_count_300s_threshold": "high_velocity",
    "burst_velocity_threshold": "burst_velocity",
    "burst_vel_threshold": "burst_velocity",
    "new_device_high_value": "new_device_high_value_amount",
    "new_device_amount": "new_device_high_value_amount",
    "amount_new_device_threshold": "new_device_high_value_amount",
    "high_value_threshold": "new_device_high_value_amount",
    "amount_zscore_threshold": "unusual_amount_zscore",
    "zscore_threshold": "unusual_amount_zscore",
    "zscore": "unusual_amount_zscore",
    "offhours_high_value_threshold": "offhours_high_value_amount",
    "offhours_amount": "offhours_high_value_amount",
    "offhours_threshold": "offhours_high_value_amount",
}


def _normalize_thresholds(user: Dict[str, Any] | None) -> Dict[str, float]:
    """Merge user thresholds into defaults, resolving aliases."""
    out = dict(DEFAULT_THRESHOLDS)
    if not user:
        return out
    for k, v in user.items():
        canonical = _THRESHOLD_ALIASES.get(k, k)
        out[canonical] = float(v) if isinstance(v, float) else v
        # Keep original key as well if it is non-canonical and not alias,
        # so callers can inspect what they passed.
        if k != canonical and k not in out:
            out[k] = v
    return out


def _get(row: Any, key: str, default: Any = 0) -> Any:
    """Safe dict/Series accessor.

    Supports ``dict``, ``pd.Series``, or any mapping with ``.get`` or
    ``__getitem__``. Returns ``default`` when the key is missing or the
    value is ``None``.
    """
    try:
        if hasattr(row, "get"):
            # dict and Series both have .get
            try:
                val = row.get(key, default)  # type: ignore[attr-defined]
            except Exception:
                val = default
            if val is None:
                return default
            return val
        # fallback to mapping protocol
        try:
            val = row[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return default
        if val is None:
            return default
        return val
    except Exception:
        return default


def _is_nan(v: Any) -> bool:
    """Return True when v is NaN (handles None, float('nan'), pd.NA)."""
    if v is None:
        return True
    try:
        # pandas NA / numpy nan
        if pd is not None:
            try:
                if pd.isna(v):  # type: ignore[attr-defined]
                    return True
            except Exception:
                pass
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A single deterministic fraud rule.

    Attributes
    ----------
    rule_id:
        Stable identifier (snake_case).
    name:
        Human-readable short name.
    severity:
        One of ``low``, ``medium``, ``high``, ``critical``.
    description:
        One-sentence explanation.
    condition:
        Callable ``(row) -> bool`` returning True when the rule triggers.
        ``row`` is a dict-like feature row (``dict`` or ``pd.Series``).
    """

    rule_id: str
    name: str
    severity: str
    description: str
    condition: Callable[[Any], bool] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(VALID_SEVERITIES)}; got {self.severity!r}"
            )
        if not callable(self.condition):
            raise TypeError("condition must be callable")


@dataclass(frozen=True)
class RuleResult:
    """Result of evaluating one :class:`Rule` on a row."""

    rule_id: str
    triggered: bool
    severity: str
    description: str


# ---------------------------------------------------------------------------
# Rule factory
# ---------------------------------------------------------------------------


def _build_rules(thresholds: Dict[str, Any]) -> List[Rule]:
    """Create the 8 canonical rules bound to *thresholds*.

    Captures thresholds by value (not by reference) so each engine instance
    can have independent thresholds.
    """
    # Copy thresholds values at build time to freeze them into closures.
    hv = int(thresholds.get("high_velocity", 5))
    burst = int(thresholds.get("burst_velocity", 8))
    ndhv_amt = float(thresholds.get("new_device_high_value_amount", 500))
    z_thr = float(thresholds.get("unusual_amount_zscore", 3.0))
    off_amt = float(thresholds.get("offhours_high_value_amount", 1000))

    def _cond_high_velocity(row: Any) -> bool:
        v = _get(row, "vel_count_300s", 0)
        try:
            return int(v) >= hv
        except Exception:
            return False

    def _cond_burst_velocity(row: Any) -> bool:
        v = _get(row, "vel_count_300s", 0)
        try:
            return int(v) >= burst
        except Exception:
            return False

    def _cond_new_device_high_value(row: Any) -> bool:
        is_new = _get(row, "is_new_device", 0)
        amt = _get(row, "amount", 0)
        try:
            return int(is_new) == 1 and float(amt) > ndhv_amt
        except Exception:
            return False

    def _cond_unusual_location(row: Any) -> bool:
        v = _get(row, "is_unusual_location", 0)
        try:
            return int(v) == 1
        except Exception:
            # also handle truthy non-int
            return bool(v) is True

    def _cond_unusual_amount(row: Any) -> bool:
        v = _get(row, "amount_zscore", 0)
        if _is_nan(v):
            return False
        try:
            return abs(float(v)) > z_thr
        except Exception:
            return False

    def _cond_shared_device(row: Any) -> bool:
        v = _get(row, "device_is_shared", 0)
        try:
            return int(v) == 1
        except Exception:
            return bool(v) is True

    def _cond_offhours_high_value(row: Any) -> bool:
        is_off = _get(row, "is_offhours", 0)
        amt = _get(row, "amount", 0)
        try:
            return int(is_off) == 1 and float(amt) > off_amt
        except Exception:
            return False

    def _cond_new_device(row: Any) -> bool:
        v = _get(row, "is_new_device", 0)
        try:
            return int(v) == 1
        except Exception:
            return bool(v) is True

    return [
        Rule(
            rule_id="high_velocity",
            name="High Velocity",
            severity="medium",
            description="5+ transactions in the last 300 seconds for this account.",
            condition=_cond_high_velocity,
        ),
        Rule(
            rule_id="new_device_high_value",
            name="New Device High Value",
            severity="high",
            description="Transaction on a new device with amount > threshold.",
            condition=_cond_new_device_high_value,
        ),
        Rule(
            rule_id="unusual_location",
            name="Unusual Location",
            severity="medium",
            description="Transaction location differs from account holder home country.",
            condition=_cond_unusual_location,
        ),
        Rule(
            rule_id="unusual_amount",
            name="Unusual Amount",
            severity="high",
            description="Absolute amount z-score exceeds threshold (|z| > 3).",
            condition=_cond_unusual_amount,
        ),
        Rule(
            rule_id="shared_device",
            name="Shared Device",
            severity="medium",
            description="Device is shared across multiple accounts.",
            condition=_cond_shared_device,
        ),
        Rule(
            rule_id="offhours_high_value",
            name="Off-Hours High Value",
            severity="high",
            description="High-value transaction during off-hours.",
            condition=_cond_offhours_high_value,
        ),
        Rule(
            rule_id="burst_velocity",
            name="Burst Velocity",
            severity="critical",
            description="8+ transactions in the last 300 seconds (burst).",
            condition=_cond_burst_velocity,
        ),
        Rule(
            rule_id="new_device",
            name="New Device",
            severity="low",
            description="Transaction on a device never seen for this account.",
            condition=_cond_new_device,
        ),
    ]


# Global default rules (uses DEFAULT_THRESHOLDS).
DEFAULT_RULES: List[Rule] = _build_rules(dict(DEFAULT_THRESHOLDS))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RuleEngine:
    """Evaluate deterministic fraud rules on feature rows.

    Parameters
    ----------
    thresholds:
        Optional overrides for numeric thresholds. Keys are canonical
        threshold names (see :data:`DEFAULT_THRESHOLDS`) or any of the
        supported aliases mapping to them. Unspecified thresholds keep
        their defaults.
    rules:
        Optional custom rule list. When ``None`` (default), the engine
        builds the 8 canonical rules bound to the resolved thresholds.
        Supplying ``rules`` bypasses threshold binding — the caller is
        responsible for rule conditions matching the desired thresholds.

    Examples
    --------
    >>> engine = RuleEngine()
    >>> results = engine.evaluate({"vel_count_300s": 6, "amount": 100, "is_new_device": 0, "is_unusual_location": 0, "amount_zscore": 0.5, "device_is_shared": 0, "is_offhours": 0})
    >>> [r.rule_id for r in results if r.triggered]
    ['high_velocity']
    """

    def __init__(
        self,
        thresholds: Dict[str, Any] | None = None,
        rules: List[Rule] | None = None,
    ) -> None:
        self.thresholds: Dict[str, Any] = _normalize_thresholds(thresholds)
        if rules is not None:
            self.rules: List[Rule] = list(rules)
        else:
            self.rules = _build_rules(self.thresholds)

    def evaluate(self, txn_features_row: Any) -> List[RuleResult]:
        """Evaluate all rules on a single feature row.

        Parameters
        ----------
        txn_features_row:
            Dict-like mapping (``dict``, ``pd.Series``, etc.) with feature
            keys such as ``vel_count_300s``, ``is_new_device``, ``amount``,
            ``is_unusual_location``, ``amount_zscore``, ``device_is_shared``,
            ``is_offhours``. Missing keys are treated as non-triggering.

        Returns
        -------
        list[RuleResult]
            One :class:`RuleResult` per rule in engine order. Always
            returns ``len(self.rules)`` results (8 by default).
        """
        results: List[RuleResult] = []
        for rule in self.rules:
            try:
                triggered = bool(rule.condition(txn_features_row))
            except Exception:
                triggered = False
            results.append(
                RuleResult(
                    rule_id=rule.rule_id,
                    triggered=triggered,
                    severity=rule.severity,
                    description=rule.description,
                )
            )
        return results

    def triggered_rules(self, txn_features_row: Any) -> List[RuleResult]:
        """Convenience: return only triggered rules."""
        return [r for r in self.evaluate(txn_features_row) if r.triggered]

    def __repr__(self) -> str:  # pragma: no cover
        return f"RuleEngine(thresholds={self.thresholds!r}, n_rules={len(self.rules)})"
