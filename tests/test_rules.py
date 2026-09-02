"""Phase 8 — Rule Engine tests."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from finsheild.rules import DEFAULT_RULES, Rule, RuleEngine, RuleResult, VALID_SEVERITIES
from finsheild.rules.engine import DEFAULT_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_row(**overrides):
    """Clean baseline row that triggers no rules."""
    row = {
        "vel_count_300s": 0,
        "amount": 50.0,
        "is_new_device": 0,
        "is_unusual_location": 0,
        "amount_zscore": 0.0,
        "device_is_shared": 0,
        "is_offhours": 0,
    }
    row.update(overrides)
    return row


def _triggered_ids(results):
    return {r.rule_id for r in results if r.triggered}


# ---------------------------------------------------------------------------
# Schema / engine basics
# ---------------------------------------------------------------------------


def test_engine_returns_all_eight_evaluations():
    engine = RuleEngine()
    results = engine.evaluate(_base_row())
    assert len(results) == 8
    ids = [r.rule_id for r in results]
    assert ids == [r.rule_id for r in DEFAULT_RULES]
    assert len(ids) == len(set(ids)), "rule_ids must be unique"


def test_default_rules_exported():
    assert len(DEFAULT_RULES) == 8
    assert all(isinstance(r, Rule) for r in DEFAULT_RULES)
    assert all(isinstance(r.condition, type(lambda: None)) for r in DEFAULT_RULES)


def test_severity_values_valid_and_expected():
    """Each rule severity in {low,medium,high,critical} and matches expected mapping."""
    expected = {
        "high_velocity": "medium",
        "new_device_high_value": "high",
        "unusual_location": "medium",
        "unusual_amount": "high",
        "shared_device": "medium",
        "offhours_high_value": "high",
        "burst_velocity": "critical",
        "new_device": "low",
    }
    engine = RuleEngine()
    for r in engine.rules:
        assert r.severity in VALID_SEVERITIES, f"{r.rule_id} bad severity {r.severity}"
        assert r.severity == expected[r.rule_id], f"{r.rule_id} severity mismatch"

    # Also severity on results matches rule definition
    results = engine.evaluate(_base_row(vel_count_300s=10, is_new_device=1, amount=2000, is_unusual_location=1, amount_zscore=5, device_is_shared=1, is_offhours=1))
    for res in results:
        rule = next(x for x in engine.rules if x.rule_id == res.rule_id)
        assert res.severity == rule.severity


def test_rule_result_fields():
    engine = RuleEngine()
    for res in engine.evaluate(_base_row()):
        assert isinstance(res, RuleResult)
        assert isinstance(res.rule_id, str)
        assert isinstance(res.triggered, bool)
        assert res.severity in VALID_SEVERITIES
        assert isinstance(res.description, str) and len(res.description) > 0


def test_clean_row_triggers_nothing():
    engine = RuleEngine()
    results = engine.evaluate(_base_row())
    assert _triggered_ids(results) == set()


# ---------------------------------------------------------------------------
# Per-rule trigger tests
# ---------------------------------------------------------------------------


def test_high_velocity_triggers():
    engine = RuleEngine()
    # below threshold
    assert "high_velocity" not in _triggered_ids(engine.evaluate(_base_row(vel_count_300s=4)))
    # at threshold
    assert "high_velocity" in _triggered_ids(engine.evaluate(_base_row(vel_count_300s=5)))
    # above threshold
    assert "high_velocity" in _triggered_ids(engine.evaluate(_base_row(vel_count_300s=10)))
    # burst also triggers high_velocity
    assert "high_velocity" in _triggered_ids(engine.evaluate(_base_row(vel_count_300s=8)))


def test_burst_velocity_triggers():
    engine = RuleEngine()
    assert "burst_velocity" not in _triggered_ids(engine.evaluate(_base_row(vel_count_300s=7)))
    assert "burst_velocity" in _triggered_ids(engine.evaluate(_base_row(vel_count_300s=8)))
    assert "burst_velocity" in _triggered_ids(engine.evaluate(_base_row(vel_count_300s=15)))


def test_new_device_high_value_triggers():
    engine = RuleEngine()
    # new_device=0 even with high amount → no trigger
    assert "new_device_high_value" not in _triggered_ids(engine.evaluate(_base_row(is_new_device=0, amount=1000)))
    # new_device=1 but amount <=500 → no trigger
    assert "new_device_high_value" not in _triggered_ids(engine.evaluate(_base_row(is_new_device=1, amount=500)))
    assert "new_device_high_value" not in _triggered_ids(engine.evaluate(_base_row(is_new_device=1, amount=300)))
    # new_device=1 and amount>500 → triggers
    assert "new_device_high_value" in _triggered_ids(engine.evaluate(_base_row(is_new_device=1, amount=501)))
    assert "new_device_high_value" in _triggered_ids(engine.evaluate(_base_row(is_new_device=1, amount=1000)))


def test_unusual_location_triggers():
    engine = RuleEngine()
    assert "unusual_location" not in _triggered_ids(engine.evaluate(_base_row(is_unusual_location=0)))
    assert "unusual_location" in _triggered_ids(engine.evaluate(_base_row(is_unusual_location=1)))


def test_unusual_amount_triggers():
    engine = RuleEngine()
    # within threshold → no trigger
    assert "unusual_amount" not in _triggered_ids(engine.evaluate(_base_row(amount_zscore=2.9)))
    assert "unusual_amount" not in _triggered_ids(engine.evaluate(_base_row(amount_zscore=-2.9)))
    assert "unusual_amount" not in _triggered_ids(engine.evaluate(_base_row(amount_zscore=0)))
    # exceeds threshold
    assert "unusual_amount" in _triggered_ids(engine.evaluate(_base_row(amount_zscore=3.1)))
    assert "unusual_amount" in _triggered_ids(engine.evaluate(_base_row(amount_zscore=-4.0)))
    # NaN should not trigger
    assert "unusual_amount" not in _triggered_ids(engine.evaluate(_base_row(amount_zscore=float("nan"))))
    assert "unusual_amount" not in _triggered_ids(engine.evaluate(_base_row(amount_zscore=None)))


def test_shared_device_triggers():
    engine = RuleEngine()
    assert "shared_device" not in _triggered_ids(engine.evaluate(_base_row(device_is_shared=0)))
    assert "shared_device" in _triggered_ids(engine.evaluate(_base_row(device_is_shared=1)))


def test_offhours_high_value_triggers():
    engine = RuleEngine()
    # offhours but low amount
    assert "offhours_high_value" not in _triggered_ids(engine.evaluate(_base_row(is_offhours=1, amount=500)))
    assert "offhours_high_value" not in _triggered_ids(engine.evaluate(_base_row(is_offhours=1, amount=1000)))
    # high amount but not offhours
    assert "offhours_high_value" not in _triggered_ids(engine.evaluate(_base_row(is_offhours=0, amount=5000)))
    # both
    assert "offhours_high_value" in _triggered_ids(engine.evaluate(_base_row(is_offhours=1, amount=1001)))
    assert "offhours_high_value" in _triggered_ids(engine.evaluate(_base_row(is_offhours=1, amount=5000)))


def test_new_device_triggers():
    engine = RuleEngine()
    assert "new_device" not in _triggered_ids(engine.evaluate(_base_row(is_new_device=0)))
    assert "new_device" in _triggered_ids(engine.evaluate(_base_row(is_new_device=1)))


# ---------------------------------------------------------------------------
# Threshold configurability
# ---------------------------------------------------------------------------


def test_thresholds_configurable_high_velocity():
    engine_default = RuleEngine()
    engine_custom = RuleEngine(thresholds={"high_velocity": 10})
    row = _base_row(vel_count_300s=6)
    assert "high_velocity" in _triggered_ids(engine_default.evaluate(row))
    assert "high_velocity" not in _triggered_ids(engine_custom.evaluate(row))
    # custom now triggers at 10
    assert "high_velocity" in _triggered_ids(engine_custom.evaluate(_base_row(vel_count_300s=10)))


def test_thresholds_configurable_burst_and_amounts():
    # burst
    e = RuleEngine(thresholds={"burst_velocity": 15})
    assert "burst_velocity" not in _triggered_ids(e.evaluate(_base_row(vel_count_300s=10)))
    assert "burst_velocity" in _triggered_ids(e.evaluate(_base_row(vel_count_300s=15)))

    # new_device_high_value amount
    e2 = RuleEngine(thresholds={"new_device_high_value_amount": 2000})
    assert "new_device_high_value" not in _triggered_ids(e2.evaluate(_base_row(is_new_device=1, amount=1000)))
    assert "new_device_high_value" in _triggered_ids(e2.evaluate(_base_row(is_new_device=1, amount=2001)))

    # unusual_amount zscore
    e3 = RuleEngine(thresholds={"unusual_amount_zscore": 5})
    assert "unusual_amount" not in _triggered_ids(e3.evaluate(_base_row(amount_zscore=4)))
    assert "unusual_amount" in _triggered_ids(e3.evaluate(_base_row(amount_zscore=5.1)))

    # offhours high value
    e4 = RuleEngine(thresholds={"offhours_high_value_amount": 5000})
    assert "offhours_high_value" not in _triggered_ids(e4.evaluate(_base_row(is_offhours=1, amount=2000)))
    assert "offhours_high_value" in _triggered_ids(e4.evaluate(_base_row(is_offhours=1, amount=5001)))


def test_threshold_aliases():
    # using alias keys should also work
    e = RuleEngine(thresholds={"high_velocity_threshold": 10})
    assert "high_velocity" not in _triggered_ids(e.evaluate(_base_row(vel_count_300s=6)))
    e2 = RuleEngine(thresholds={"amount_zscore_threshold": 5})
    assert "unusual_amount" not in _triggered_ids(e2.evaluate(_base_row(amount_zscore=4)))


# ---------------------------------------------------------------------------
# Pandas Series input & edge cases
# ---------------------------------------------------------------------------


def test_evaluate_accepts_pandas_series():
    engine = RuleEngine()
    row = pd.Series(_base_row(vel_count_300s=5, is_new_device=1, amount=600, is_unusual_location=1, amount_zscore=4, device_is_shared=1, is_offhours=1))
    # amount 600 >500 and >? but offhours needs >1000 so offhours not triggered here, adjust
    row2 = pd.Series(_base_row(vel_count_300s=9, is_new_device=1, amount=2000, is_unusual_location=1, amount_zscore=4, device_is_shared=1, is_offhours=1))
    ids = _triggered_ids(engine.evaluate(row2))
    assert ids == {"high_velocity", "burst_velocity", "new_device_high_value", "unusual_location", "unusual_amount", "shared_device", "offhours_high_value", "new_device"}


def test_missing_keys_do_not_crash():
    engine = RuleEngine()
    results = engine.evaluate({})
    assert len(results) == 8
    assert _triggered_ids(results) == set()

    results2 = engine.evaluate({"vel_count_300s": 5})
    assert "high_velocity" in _triggered_ids(results2)


def test_rule_dataclass_validation():
    with pytest.raises(ValueError):
        Rule(rule_id="bad", name="bad", severity="extreme", description="x", condition=lambda row: True)
    with pytest.raises(TypeError):
        Rule(rule_id="bad", name="bad", severity="low", description="x", condition="not callable")  # type: ignore[arg-type]


def test_all_eight_trigger_together():
    """A crafted row triggers all 8 rules simultaneously."""
    engine = RuleEngine()
    row = _base_row(
        vel_count_300s=10,
        is_new_device=1,
        amount=2000,
        is_unusual_location=1,
        amount_zscore=4.5,
        device_is_shared=1,
        is_offhours=1,
    )
    ids = _triggered_ids(engine.evaluate(row))
    assert ids == {
        "high_velocity",
        "new_device_high_value",
        "unusual_location",
        "unusual_amount",
        "shared_device",
        "offhours_high_value",
        "burst_velocity",
        "new_device",
    }
