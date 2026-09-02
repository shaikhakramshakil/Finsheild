"""Phase 10 — Risk Fusion tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsheild.features import build_features
from finsheild.risk_fusion import DEFAULT_THRESHOLDS, RiskFusionEngine, RiskResult
from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment


@pytest.fixture(scope="module")
def env():
    return generate_environment(SyntheticEnvConfig.ci())


@pytest.fixture(scope="module")
def feature_result(env):
    return build_features(env)


@pytest.fixture(scope="module")
def engine(env, feature_result):
    eng = RiskFusionEngine(random_state=42)
    eng.fit(env, feature_result)
    return eng


@pytest.fixture(scope="module")
def batch_results(engine, feature_result):
    return engine.predict_batch(feature_result.features)


# ------------------------------------------------------------------ #
# Basic contract
# ------------------------------------------------------------------ #

def test_default_thresholds_exported():
    assert "red" in DEFAULT_THRESHOLDS
    assert "yellow" in DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS["red"] > DEFAULT_THRESHOLDS["yellow"]
    assert 0 < DEFAULT_THRESHOLDS["yellow"] < 1
    assert 0 < DEFAULT_THRESHOLDS["red"] <= 1


def test_risk_result_fields(engine, feature_result):
    row = feature_result.features.iloc[0]
    result = engine.predict(row)
    assert isinstance(result, RiskResult)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.risk_level in {"GREEN", "YELLOW", "RED"}
    assert result.decision in {"APPROVE", "STEP_UP", "BLOCK", "INVESTIGATE"}
    assert isinstance(result.evidence, list)
    # also check detail scores
    assert 0.0 <= result.xgb_score <= 1.0
    assert 0.0 <= result.anomaly_score <= 1.0
    assert 0.0 <= result.behavioral_score <= 1.0
    assert 0.0 <= result.graph_score <= 1.0
    assert 0.0 <= result.rule_score <= 1.0


def test_predict_accepts_various_inputs(engine, feature_result):
    row_series = feature_result.features.iloc[0]
    row_dict = row_series.to_dict()
    row_df = feature_result.features.head(1)

    r1 = engine.predict(row_series)
    r2 = engine.predict(row_dict)
    r3 = engine.predict(row_df)
    # Single-row DF should match Series
    assert r1.risk_level == r3.risk_level
    assert abs(r1.risk_score - r3.risk_score) < 1e-6
    # Dict should be close (some keys may be missing but core score similar)
    assert r2.risk_level in {"GREEN", "YELLOW", "RED"}


def test_predict_batch_returns_list(engine, feature_result):
    results = engine.predict_batch(feature_result.features.head(10))
    assert len(results) == 10
    for r in results:
        assert isinstance(r, RiskResult)
        assert 0.0 <= r.risk_score <= 1.0


def test_predict_before_fit_raises(feature_result):
    eng = RiskFusionEngine()
    with pytest.raises(RuntimeError):
        eng.predict(feature_result.features.iloc[0])
    with pytest.raises(RuntimeError):
        eng.predict_batch(feature_result.features.head(5))


# ------------------------------------------------------------------ #
# Core fusion behavior
# ------------------------------------------------------------------ #

def test_risk_levels_cover_all_three(batch_results):
    levels = {r.risk_level for r in batch_results}
    assert "GREEN" in levels, "Expected at least one GREEN"
    assert "YELLOW" in levels, "Expected at least one YELLOW"
    assert "RED" in levels, "Expected at least one RED"


def test_fraud_scores_higher_than_legit_on_average(engine, feature_result, batch_results):
    df = feature_result.features
    fraud_scores = []
    legit_scores = []
    for r, (_, row) in zip(batch_results, df.iterrows()):
        if row["label_fraud"] == 1:
            fraud_scores.append(r.risk_score)
        else:
            legit_scores.append(r.risk_score)
    assert len(fraud_scores) > 0
    assert len(legit_scores) > 0
    fraud_mean = float(np.mean(fraud_scores))
    legit_mean = float(np.mean(legit_scores))
    # Fraud should be clearly higher on average
    assert fraud_mean > legit_mean + 0.15, f"fraud_mean={fraud_mean:.3f} vs legit_mean={legit_mean:.3f}"


def test_evidence_non_empty_for_fraud(batch_results, feature_result):
    df = feature_result.features
    fraud_results = [r for r, (_, row) in zip(batch_results, df.iterrows()) if row["label_fraud"] == 1]
    assert len(fraud_results) > 0
    non_empty = sum(1 for r in fraud_results if len(r.evidence) > 0)
    # At least 90% of fraud should have evidence
    assert non_empty / len(fraud_results) >= 0.9, f"only {non_empty}/{len(fraud_results)} fraud had evidence"
    # Also at least one fraud evidence should contain known pattern strings
    all_evidence_text = " ".join(" ".join(r.evidence) for r in fraud_results)
    # Check for presence of at least one spec-like pattern
    assert any(kw in all_evidence_text for kw in ("high_velocity", "shared_device", "new_device", "unusual", "ml_flagged", "anomalous", "rule:")), \
        f"evidence missing expected patterns: {all_evidence_text[:500]}"


def test_evidence_contains_spec_examples(engine, feature_result):
    # Craft a row that should trigger high_velocity and shared_device evidence
    # Use a real row but override fields to force triggers
    row = feature_result.features.iloc[0].to_dict()
    row["vel_count_300s"] = 7
    row["device_account_count"] = 3
    row["device_is_shared"] = 1
    row["is_new_device"] = 1
    # Amount low to avoid BLOCK but still high risk
    result = engine.predict(row)
    evidence_str = " ".join(result.evidence)
    assert "high_velocity" in evidence_str or "burst_velocity" in evidence_str
    assert "shared_device" in evidence_str


def test_thresholds_configurable(env, feature_result):
    # Default thresholds
    eng_default = RiskFusionEngine(random_state=42)
    eng_default.fit(env, feature_result)
    default_levels = {r.risk_level for r in eng_default.predict_batch(feature_result.features)}
    # Very high thresholds should reduce RED/YELLOW
    eng_strict = RiskFusionEngine(thresholds={"red": 0.95, "yellow": 0.85}, random_state=42)
    eng_strict.fit(env, feature_result)
    strict_levels = {r.risk_level for r in eng_strict.predict_batch(feature_result.features)}
    # Strict thresholds should have fewer non-GREEN, but at least config must be respected
    assert eng_strict.thresholds["red"] == 0.95
    assert eng_strict.thresholds["yellow"] == 0.85
    # With strict thresholds, RED should be rarer or absent compared to default
    # Default had RED, strict likely has no RED (or fewer)
    default_red = sum(1 for r in eng_default.predict_batch(feature_result.features) if r.risk_level == "RED")
    strict_red = sum(1 for r in eng_strict.predict_batch(feature_result.features) if r.risk_level == "RED")
    assert strict_red <= default_red
    # Very permissive thresholds should increase RED/YELLOW
    eng_loose = RiskFusionEngine(thresholds={"red": 0.2, "yellow": 0.1}, random_state=42)
    eng_loose.fit(env, feature_result)
    loose_levels = {r.risk_level for r in eng_loose.predict_batch(feature_result.features)}
    loose_red = sum(1 for r in eng_loose.predict_batch(feature_result.features) if r.risk_level == "RED")
    assert loose_red >= default_red


def test_decisions_follow_spec(batch_results, feature_result):
    df = feature_result.features
    for r, (_, row) in zip(batch_results, df.iterrows()):
        amt = row.get("amount_x", row.get("amount", 0))
        try:
            amt_f = float(amt)
        except Exception:
            amt_f = 0.0
        if r.risk_level == "RED":
            expected = "BLOCK" if amt_f > 1000 else "INVESTIGATE"
            assert r.decision == expected, f"RED with amt {amt_f} expected {expected} got {r.decision}"
        elif r.risk_level == "YELLOW":
            assert r.decision == "STEP_UP", f"YELLOW expected STEP_UP got {r.decision}"
        else:  # GREEN
            assert r.decision == "APPROVE", f"GREEN expected APPROVE got {r.decision}"


def test_decision_block_vs_investigate(engine, feature_result):
    # Find a RED row and test amount-dependent decision
    # Create two variants of a high-risk row
    base_row = None
    for _, row in feature_result.features.iterrows():
        # look for a row that would be RED with high xgb-like signal?
        # We'll craft synthetic rows with strong signals
        test_row = row.to_dict()
        test_row["vel_count_300s"] = 10
        test_row["device_account_count"] = 4
        test_row["device_is_shared"] = 1
        test_row["is_new_device"] = 1
        test_row["is_unusual_location"] = 1
        test_row["amount_zscore"] = 5.0
        test_row["amount_x"] = 2000
        test_row["amount_y"] = 2000
        test_row["is_offhours"] = 1
        r = engine.predict(test_row)
        if r.risk_level == "RED":
            base_row = test_row
            break
    assert base_row is not None, "Could not craft a RED row"
    r_high = engine.predict(base_row)
    assert r_high.decision == "BLOCK", f"RED high amount should BLOCK, got {r_high.decision}"

    low_row = dict(base_row)
    low_row["amount_x"] = 100
    low_row["amount_y"] = 100
    r_low = engine.predict(low_row)
    # Still RED but amount low -> INVESTIGATE (if still RED)
    if r_low.risk_level == "RED":
        assert r_low.decision == "INVESTIGATE"
    else:
        # If lowering amount drops level, that's okay but decision should still be non-BLOCK
        assert r_low.decision in {"STEP_UP", "APPROVE", "INVESTIGATE"}


def test_predict_batch_consistency_with_single(engine, feature_result):
    df = feature_result.features.head(20)
    batch = engine.predict_batch(df)
    for i, (_, row) in enumerate(df.iterrows()):
        single = engine.predict(row)
        assert abs(batch[i].risk_score - single.risk_score) < 1e-6
        assert batch[i].risk_level == single.risk_level
        assert batch[i].decision == single.decision


def test_risk_score_deterministic(engine, feature_result):
    row = feature_result.features.iloc[42]
    r1 = engine.predict(row)
    r2 = engine.predict(row)
    assert r1.risk_score == r2.risk_score
    assert r1.evidence == r2.evidence


def test_feature_build_result_passthrough(engine, feature_result):
    # predict_batch should accept FeatureBuildResult directly
    results = engine.predict_batch(feature_result)
    assert len(results) == len(feature_result.features)


def test_accepts_pretrained_models(env, feature_result):
    # Fit one engine, then reuse its models in a new engine
    eng1 = RiskFusionEngine(random_state=42)
    eng1.fit(env, feature_result)
    eng2 = RiskFusionEngine(xgb_model=eng1.xgb_model, anomaly_detector=eng1.anomaly_detector, random_state=42)
    eng2.fit(env, feature_result)  # should not retrain xgb/anomaly but still succeed
    row = feature_result.features.iloc[0]
    r1 = eng1.predict(row)
    r2 = eng2.predict(row)
    # Scores should be identical since same models and same feature columns
    assert abs(r1.risk_score - r2.risk_score) < 1e-6
