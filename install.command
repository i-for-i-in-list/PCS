#!/bin/sh
PCS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
"$PCS_ROOT/install.sh" "$@"
PCS_STATUS=$?
printf '\nPress Enter to close this window...'
read -r PCS_REPLY
exit "$PCS_STATUS"
