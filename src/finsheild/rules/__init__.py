"""Finsheild Rule Engine (Phase 8).

Deterministic fraud checks over engineered feature rows.

Public API
----------
* :class:`Rule` — immutable rule definition
* :class:`RuleResult` — per-rule evaluation result
* :class:`RuleEngine` — evaluates all rules on a feature row
* :data:`DEFAULT_RULES` — the 8 canonical rules with default thresholds
* :data:`DEFAULT_THRESHOLDS` — default numeric thresholds dict

Example
-------
>>> from finsheild.rules import RuleEngine
>>> engine = RuleEngine()
>>> engine.evaluate({"vel_count_300s": 6, "amount": 100, "is_new_device": 0, "is_unusual_location": 0, "amount_zscore": 0.5, "device_is_shared": 0, "is_offhours": 0})
"""

from finsheild.rules.engine import (
    DEFAULT_RULES,
    DEFAULT_THRESHOLDS,
    VALID_SEVERITIES,
    Rule,
    RuleEngine,
    RuleResult,
)

__all__ = [
    "Rule",
    "RuleEngine",
    "RuleResult",
    "DEFAULT_RULES",
    "DEFAULT_THRESHOLDS",
    "VALID_SEVERITIES",
]
