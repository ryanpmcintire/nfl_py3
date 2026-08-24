#!/usr/bin/env bash
# Spawn one headless worker agent for a task.
# Usage: spawn.sh <task-id> [model]
#   Task spec:  scripts/swarm/tasks/<task-id>.md  (required)
#   Worktree:   /f/Repos/nfl_swarm/<task-id>
#   Branch:     swarm/<task-id>
#   Log:        /f/Repos/nfl_swarm/logs/<task-id>.log
set -uo pipefail
TASK_ID=$1
MODEL=${2:-opencode/nemotron-3-ultra-free}
REPO=/f/Repos/nfl_py3
SWARM=/f/Repos/nfl_swarm
WT=$SWARM/$TASK_ID

[ -f "$REPO/scripts/swarm/tasks/$TASK_ID.md" ] || { echo "no task file $TASK_ID"; exit 2; }

cd "$REPO"
if ! git worktree list | grep -q "$WT"; then
  git branch -D "swarm/$TASK_ID" 2>/dev/null
  git worktree add -b "swarm/$TASK_ID" "$WT" master >/dev/null || { echo "worktree add failed"; exit 2; }
fi
mkdir -p "$SWARM/logs"

PROMPT="# Your task id: $TASK_ID
$(cat "$REPO/scripts/swarm/WORKER_PREAMBLE.md")

=== YOUR TASK ===
$(cat "$REPO/scripts/swarm/tasks/$TASK_ID.md")

Write your report to reports/wave1/${TASK_ID}.md inside the repo, commit on
branch swarm/${TASK_ID}, then end with TASK_COMPLETE ${TASK_ID}."

cd "$WT"
echo "[$(date +%H:%M:%S)] spawning $TASK_ID on $MODEL"
timeout 2700 pi -p --provider opencode --model "$MODEL" --no-session \
  --name "swarm-$TASK_ID" --mode text "$PROMPT" > "$SWARM/logs/$TASK_ID.log" 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "TASK_COMPLETE $TASK_ID" "$SWARM/logs/$TASK_ID.log"; then
  echo "[$(date +%H:%M:%S)] DONE  $TASK_ID"
else
  echo "[$(date +%H:%M:%S)] FAIL  $TASK_ID (exit=$rc) — see logs"
fi
