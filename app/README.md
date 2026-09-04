# Finsheild Demo Application (GUI + Demo Integration)

Demo-first presentation layer. **Does not touch the ML research pipeline.**

- `backend/` — FastAPI. GUI talks to it via a stable contract (`schemas.py`).
  - `adapters/mock_adapter.py` — deterministic demo scorer (active now).
  - `adapters/real_adapter.py` — read-only probe + seam for Agent 3 (`RealMLAdapter`).
- `frontend/` — React + TS + Vite + Tailwind. Screens: Command Center, Investigation,
  Model Performance (real ULB values read from `evaluation/reports/`), Architecture,
  Privacy Identity.

## Data honesty

| Kind | Source | Label in UI |
|---|---|---|
| Real benchmark | `evaluation/reports/xgboost_metrics.json` | `REAL_BENCHMARK` |
| Demo simulation | `MockMLAdapter` + in-memory store | `DEMO_SIMULATION` / `DEMO_FALLBACK` |

The LLM copilot explains engine evidence only; it never sets the risk score.
Identity screen is prototype tokenization, not a zero-knowledge proof.

## Run

```bash
./start.sh            # backend :8000 + frontend :5173
```

Backend only: `.venv/bin/python -m uvicorn app.backend.main:app --port 8000`
Backend tests: `.venv/bin/python -m pytest app/backend/tests/ -q`
Frontend build: `cd app/frontend && npm run build`
