#!/usr/bin/env bash
# Finsheild demo launcher: backend (FastAPI) + frontend (Vite).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
BACKEND_PORT="${BACKEND_PORT:-8000}"
export VITE_API_URL="${VITE_API_URL:-http://127.0.0.1:$BACKEND_PORT}"

"$PY" -m uvicorn app.backend.main:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACK_PID=$!
trap 'kill $BACK_PID 2>/dev/null' EXIT
echo "backend -> http://127.0.0.1:$BACKEND_PORT (pid $BACK_PID)"

cd "$ROOT/app/frontend"
if [ ! -d node_modules ]; then npm install; fi
npm run dev -- --host 127.0.0.1 --port 5173
