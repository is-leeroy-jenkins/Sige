#!/usr/bin/env bash

set -e

APP_FILE="${APP_FILE:-app.py}"
PORT="${PORT:-8501}"
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

if [[ ! -f "${APP_FILE}" ]]; then
    echo "ERROR: Streamlit entry file '${APP_FILE}' was not found in /app."
    exit 1
fi

export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"

exec streamlit run "${APP_FILE}" \
    --server.address="${STREAMLIT_SERVER_ADDRESS}" \
    --server.port="${PORT}" \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true \
    --browser.gatherUsageStats=false