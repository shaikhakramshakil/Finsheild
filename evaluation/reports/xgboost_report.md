# xgboost Report — REAL ULB (284k)

Model: `xgboost`
Data: Real Kaggle ULB (284,807 rows, 492 frauds, 0.17% fraud rate)

## Test metrics (holdout n=42722, threshold=0.5)
- Precision: 0.9206
- Recall:    0.7838
- F1:        0.8467
- ROC-AUC:   0.9709
- PR-AUC:    0.8418
- Recall @ FPR=1.00%: 0.8784 (threshold 0.0009 on val)

## Confusion matrix (test, threshold=0.5)
|        | Predicted legit | Predicted fraud |
|--------|-----------------|-----------------|
| Actual legit | 42643 | 5 |
| Actual fraud | 16 | 58 |

Support: 42648 legit, 74 fraud

## Training
- Train seconds: 11.5
- Splits: train 199364, val 42721, test 42722

PR vs random baseline (0.0017): 486.0x lift
**Note:** Synthetic hard (1% fraud) gave PR 0.553 for comparison — real ULB is harder.
