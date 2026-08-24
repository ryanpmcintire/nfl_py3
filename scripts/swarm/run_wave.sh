#!/usr/bin/env bash
# Launch an entire wave from scripts/swarm/waveN_manifest.txt in parallel.
# Usage: run_wave.sh [manifest-name] [max-parallel]
set -uo pipefail
WAVE=${1:-wave1}
MAXPAR=${2:-29}
DIR=/f/Repos/nfl_py3/scripts/swarm
MANIFEST=$DIR/${WAVE}_manifest.txt
RUNNING=$DIR/.${WAVE}_running
: > "$RUNNING"

count=0
while read -r task model; do
  [ -z "$task" ] && continue
  bash "$DIR/spawn.sh" "$task" "$model" >> "/f/Repos/nfl_swarm/logs/${task}.launch" 2>&1 &
  echo "$! $task" >> "$RUNNING"
  count=$((count+1))
  # throttle to MAXPAR concurrent spawns
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
done < "$MANIFEST"
echo "launched $count workers; pids in $RUNNING"
wait
echo "wave $WAVE complete:"
grep -c DONE "/f/Repos/nfl_swarm/logs"/"${WAVE}"_summary 2>/dev/null || true
