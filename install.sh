#!/bin/sh
set -eu
PCS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 -B "$PCS_ROOT/install.py" "$@"
