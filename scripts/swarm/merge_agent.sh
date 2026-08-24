#!/usr/bin/env bash
# Launch the dedicated merge agent over all swarm branches not yet on master.
# Usage: merge_agent.sh [model]
set -uo pipefail
MODEL=${1:-opencode/kimi-k2.6}
REPO=/f/Repos/nfl_py3
SWARM=/f/Repos/nfl_swarm

cd "$REPO"
BRANCHES=$(git branch --list 'swarm/*' --format='%(refname:short)')
[ -z "$BRANCHES" ] && { echo "no swarm branches"; exit 0; }

mkdir -p "$SWARM/logs"
PROMPT="$(cat "$REPO/scripts/swarm/MERGE_PROMPT.md")

=== BRANCHES TO INTEGRATE ===
$BRANCHES

Work sequentially. Keep a ledger at reports/merge_ledger.md."

cd "$SWARM/_merge" 2>/dev/null || { git -C "$REPO" worktree add -b _merge_scratch "$SWARM/_merge" master >/dev/null; cd "$SWARM/_merge"; }
timeout 5400 pi -p --provider opencode --model "$MODEL" --no-session \
  --name "swarm-merger" --mode text "$PROMPT" > "$SWARM/logs/merger.log" 2>&1
tail -5 "$SWARM/logs/merger.log"
