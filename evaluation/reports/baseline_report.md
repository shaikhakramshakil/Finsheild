# logreg Report — REAL ULB (284k)

Model: `logreg`
Data: Real Kaggle ULB (284,807 rows, 492 frauds, 0.17% fraud rate)

## Test metrics (holdout n=42722, threshold=0.5)
- Precision: 0.8197
- Recall:    0.6757
- F1:        0.7407
- ROC-AUC:   0.9495
- PR-AUC:    0.7005
- Recall @ FPR=1.00%: 0.8514 (threshold 0.0055 on val)

## Confusion matrix (test, threshold=0.5)
|        | Predicted legit | Predicted fraud |
|--------|-----------------|-----------------|
| Actual legit | 42637 | 11 |
| Actual fraud | 24 | 50 |

Support: 42648 legit, 74 fraud

## Training
- Splits: train 199364, val 42721, test 42722

PR vs random baseline (0.0017): 404.4x lift
