#!/bin/sh
set -eu
PCS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$PCS_ROOT/.venv/bin/python" ]; then
    echo "Please run ./install.sh first." >&2
    exit 1
fi
exec "$PCS_ROOT/.venv/bin/python" -B "$PCS_ROOT/start.py" "$@"
