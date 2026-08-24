#!/usr/bin/env bash
# Live status board for a swarm wave.
# Usage: status.sh [wave]   (default wave1)
WAVE=${1:-wave1}
DIR=/f/Repos/nfl_py3/scripts/swarm
LOGS=/f/Repos/nfl_swarm/logs

evidence() {
  local wt=/f/Repos/nfl_swarm/$1
  [ -f "$wt/reports/wave1/$1.md" ] && \
    [ "$(git -C "$wt" rev-list --count master..HEAD 2>/dev/null || echo 0)" -gt 0 ]
}

printf '%-34s %-9s %6s  %s\n' TASK STATE ATT last-log-line
printf '%s\n' "$(printf '%.0s-' {1..95})"
done=0; failed=0; other=0
while read -r task _m; do
  [ -z "$task" ] && continue
  log="$LOGS/$task.log"
  att=$(grep -c "launch $task " "$LOGS/${WAVE}_supervisor.log" 2>/dev/null)
  att=${att:-0}
  if evidence "$task"; then
    state=DONE; done=$((done+1))
  elif grep -q "GIVEUP $task" "$LOGS/${WAVE}_supervisor.log" 2>/dev/null; then
    state=GAVEUP; failed=$((failed+1))
  elif [ -n "$(find "$LOGS/../sessions" -name '*.jsonl' -newermt '-100 seconds' -exec grep -l "swarm-$task" {} + 2>/dev/null | head -1)" ]; then
    state=RUNNING; other=$((other+1))
  elif [ -n "$(find "$log" -newermt '-100 seconds' 2>/dev/null)" ]; then
    state=RUNNING; other=$((other+1))
  else
    state=queued; other=$((other+1))
  fi
  last=$(tail -c 400 "$log" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-45)
  printf '%-34s %-9s %6s  %s\n' "$task" "$state" "$att" "$last"
done < "$DIR/${WAVE}_manifest.txt"
printf '%s\n' "$(printf '%.0s-' {1..95})"
echo "DONE: $done   ACTIVE/QUEUED: $other   GAVE UP: $failed"
