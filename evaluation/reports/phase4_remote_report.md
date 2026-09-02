# Phase 4 — Remote Colab Validation Report

**Date:** 2026-09-03
**Session:** `finsheild-remote-phase4` (CPU, ephemeral)
**Source repo:** `https://github.com/shaikhakramshakil/Finsheild` @ commit `42532b4`
**Reproducer script:** `/content/_remote_phase4_full.py`

## Workflow verified

```
local edit → git commit (42532b4) → git push → Colab CLI tarball clone →
pip install pandas+pyarrow+pytest → generate_environment(ci()) →
pytest tests/test_synthetic_env.py → metadata.json retrieved via
colab download
```

## Generation result (CI scale, seed=1729)

| Table            | Rows | Cols |
|------------------|-----:|-----:|
| users            |  200 |   4 |
| accounts         |  250 |   5 |
| devices          |  220 |   4 |
| merchants        |   80 |   6 |
| locations        |   60 |   6 |
| account_devices  |  496 |   5 |
| transactions     | 6158 |  13 |

Generation wall-clock: **4.0 s** on Colab CPU.

## Scenario breakdown (label_fraud counts)

| Scenario              | n_total | n_fraud |
|-----------------------|--------:|--------:|
| account_takeover      |      77 |      30 |
| background            |    5000 |       0 |
| device_sharing        |      40 |      20 |
| mule_behavior         |     107 |      15 |
| new_device            |     160 |      80 |
| transaction_velocity  |     434 |     394 |
| unusual_amount_time   |     120 |      60 |
| unusual_location      |     120 |      60 |
| unusual_merchant      |     100 |      50 |
| **TOTAL**             |   6158 |    709 |

Labelled fraud rate: **11.51%** at CI scale (high because scenario context
transactions dilute the 5 000-row background; at dev scale the labelled
rate drops to ~1.4%).

## Tests

```
25 passed in 16.92s
```

All Phase 4 tests pass on Colab CPU. Identical pass count to local.

## Notes

* `git+https://github.com/...` from inside Colab raises
  `could not read Username for 'https://github.com'`, even though
  `urllib` + `curl` both reach github.com successfully. The CLI
  workflow uses the `https://github.com/.../archive/refs/heads/main.tar.gz`
  tarball endpoint instead. `git push` from local → GitHub works as
  expected.
* No GPU used for Phase 4. CPU is sufficient.
* Colab base image: Python 3.13.15, pandas, scikit-learn, xgboost,
  lightgbm, torch all pre-installed.