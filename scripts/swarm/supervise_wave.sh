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

MODELS=(opencode/nemotron-3-ultra-free opencode/nemotron-3.5-lightning-free opencode/hy3-free opencode/mimo-v2.5-free opencode/big-pickle)

declare -A ATTEMPTS PID_TASK MODEL_IDX
QUEUE=()
while read -r task _m; do [ -n "$task" ] && QUEUE+=("$task"); done < "$MANIFEST"

DONE=(); FAILED=()
mi=0
launch() {
  local task=$1
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
  # top up
  while [ ${#QUEUE[@]} -gt 0 ] && [ ${#PID_TASK[@]} -lt "$MAXPAR" ]; do
    launch "${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
  done
  sleep 20
  # reap
  for pid in "${!PID_TASK[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      task=${PID_TASK[$pid]}; unset PID_TASK[$pid]
      log="$LOGS/$task.log"
      if grep -q "TASK_COMPLETE $task" "$log" 2>/dev/null; then
        DONE+=("$task"); echo "[$(date +%H:%M:%S)] DONE $task (${#DONE[@]} done)"
      elif grep -qiE 'RateLimit|FreeUsageLimit|UsageLimit' "$log" 2>/dev/null \
           || ! grep -q "TASK_FAILED" "$log" 2>/dev/null; then
        if [ "${ATTEMPTS[$task]}" -lt 4 ]; then
          echo "[$(date +%H:%M:%S)] retry-queue $task (rate limit or silent death)"
          QUEUE+=("$task"); sleep $((45 * ATTEMPTS[$task]))
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
