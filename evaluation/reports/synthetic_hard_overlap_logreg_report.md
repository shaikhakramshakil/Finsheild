# Synthetic Hard Overlap — logreg

**Variant:** `synthetic_hard_overlap` (new, does not overwrite easy/1% diluted)
**Seed:** 1729 | **Background:** 9000 | **n_per_scenario:** 20 | **Rows:** 9100 | **Fraud:** 100 (1.0989%)

## Methodology
- Background: same legitimate distribution as easy synthetic; legitimate users now travel (30% foreign), visit high-risk merchants (30% high-risk), burst velocity, off-hours — creates overlap.
- Fraud: 5 weak-signal scenarios (moderate amount+new device, normal amount+new device+merchant, high amount only, normal-looking with weak signals, mixed combos). No single feature is a perfect separator.
- Features: 36 engineered cols (transactional, behavioral, velocity, location, device) — same `FeatureConfig` as easy/1%.
- Split: stratified 70/15/15, random_state 42, scaler fit on train only (no leakage).
- Model: logreg fixed config (XGBoost: n_estimators=500 lr=0.05 max_depth=6 subsample 0.8 colsample 0.8 seed 42).
- Threshold: 0.5 for F1/confusion; 1% FPR threshold tuned on val.

## Metrics (holdout n=1365)
- ROC-AUC: 0.9640
- PR-AUC: 0.2200
- F1: 0.1111 (Prec 0.3333 Rec 0.0667)
- Recall @1%FPR: 0.0667 (thr 0.1999)
- Lift vs random (1.0989%): 20.0x

## Confusion @0.5
TN=1348 FP=2 FN=14 TP=1

**Limitations:** still simulated; does not represent real banking behavior. Overlap is heuristic, not calibrated to real ULB.
