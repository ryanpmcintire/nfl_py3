#!/usr/bin/env bash
# Resilient wave supervisor: caps concurrency, rotates free models,
# retries rate-limited workers with backoff.
# Usage: supervise_wave.sh <wave> [max-parallel]
set -o pipefail
WAVE=${1:-wave1}
MAXPAR=${2:-6}
DIR=/f/Repos/nfl_py3/scripts/swarm
LOGS=/f/Repos/nfl_swarm/logs
MANIFEST=$DIR/${WAVE}_manifest.txt

# Lead with x-preview-f-free (the owner-session model; verified spawner-accessible
# and 6-way parallel at 2026-08-24). Two free models retained as overflow buckets
# so fleet load cannot exhaust the interactive session's quota.
MODELS=(opencode/x-preview-f-free opencode/x-preview-f-free opencode/x-preview-f-free opencode/nemotron-3-ultra-free opencode/hy3-free)

declare -A ATTEMPTS PID_TASK READY
QUEUE=()
while read -r task _m; do [ -n "$task" ] && QUEUE+=("$task"); done < "$MANIFEST"
READY_NOW=0

DONE=(); FAILED=()
mi=0

# A task is only done if it left real evidence: a report file AND commits
# on its branch. Log text alone is untrustworthy: pi echoes the prompt,
# which contains the literal TASK_COMPLETE instruction.
evidence_ok() {
  local task=$1
  local wt=/f/Repos/nfl_swarm/$task
  [ -f "$wt/reports/wave1/$task.md" ] || return 1
  [ "$(git -C "$wt" rev-list --count master..HEAD 2>/dev/null || echo 0)" -gt 0 ]
}

launch() {
  local task=$1
  if evidence_ok "$task"; then
    DONE+=("$task"); echo "[$(date +%H:%M:%S)] skip $task (evidence already present)"
    return
  fi
  local model=${MODELS[$((mi % ${#MODELS[@]}))]}
  mi=$((mi+1))
  # reset any partial work from a previous attempt
  local wt=/f/Repos/nfl_swarm/$task
  git -C "$wt" reset --hard >/dev/null 2>&1
  git -C "$wt" clean -fdq >/dev/null 2>&1
  bash "$DIR/spawn.sh" "$task" "$model" > "$LOGS/$task.launch" 2>&1 &
  PID_TASK[$!]=$task
  ATTEMPTS[$task]=$(( ${ATTEMPTS[$task]:-0} + 1 ))
  echo "[$(date +%H:%M:%S)] launch $task (attempt ${ATTEMPTS[$task]}, $model)"
}

echo "supervising ${#QUEUE[@]} tasks at max $MAXPAR parallel"
while [ ${#QUEUE[@]} -gt 0 ] || [ ${#PID_TASK[@]} -gt 0 ]; do
  # top up: only tasks whose backoff ready-time has passed
  while [ ${#QUEUE[@]} -gt 0 ] && [ ${#PID_TASK[@]} -lt "$MAXPAR" ]; do
    now=$(date +%s)
    pick=-1
    for i in "${!QUEUE[@]}"; do
      t=${QUEUE[$i]}
      if [ $(( ${READY[$t]:-0} )) -le "$now" ]; then pick=$i; break; fi
    done
    # nothing ready: wait and re-poll instead of launching a not-ready task
    [ "$pick" -lt 0 ] && break
    launch "${QUEUE[$pick]}"; unset QUEUE[$pick]; QUEUE=("${QUEUE[@]}")
  done
  sleep 20
  # reap
  for pid in "${!PID_TASK[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      task=${PID_TASK[$pid]}; unset PID_TASK[$pid]
      log="$LOGS/$task.log"
      if evidence_ok "$task"; then
        DONE+=("$task"); echo "[$(date +%H:%M:%S)] DONE $task (${#DONE[@]} done)"
      elif grep -qiE 'RateLimit|FreeUsageLimit|UsageLimit' "$log" 2>/dev/null \
           || ! grep -q "TASK_FAILED" "$log" 2>/dev/null; then
        if [ "${ATTEMPTS[$task]}" -lt 10 ]; then
          # backoff without blocking the loop: schedule a ready-time
          READY[$task]=$(( $(date +%s) + 240 * ATTEMPTS[$task] ))
          echo "[$(date +%H:%M:%S)] retry-queue $task (ready $(date -d @${READY[$task]} +%H:%M:%S))"
        else
          FAILED+=("$task"); echo "[$(date +%H:%M:%S)] GIVEUP $task"
        fi
      else
        FAILED+=("$task"); echo "[$(date +%H:%M:%S)] FAILED $task (see $log)"
      fi
    fi
  done
done

echo "== wave $WAVE summary =="
echo "done:   ${DONE[*]:--none-}"
echo "failed: ${FAILED[*]:--none-}"
