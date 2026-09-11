#!/bin/sh
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
if [ -f "$HOME/.config/nfl_py3/env" ]; then
    set -a
    . "$HOME/.config/nfl_py3/env"
    set +a
fi
export NFL_ATS_SCHEDULER_ROLE="${NFL_ATS_SCHEDULER_ROLE:-capture}"
if ./.tools/uv run --no-sync python scripts/capture_scheduler.py --is-running >/dev/null 2>&1; then
    echo "Refusing to start a second capture scheduler daemon."
    exit 1
fi
mkdir -p data
nohup ./.tools/uv run --no-sync python scripts/capture_scheduler.py >>data/scheduler_stdout.txt 2>&1 &
echo "started pid $! role $NFL_ATS_SCHEDULER_ROLE"
