#!/usr/bin/env bash
# Safely stop all swarm processes. Kills ONLY processes whose full command
# line references the swarm worktree/log paths. NEVER kills by image name
# or broad pattern - doing that once took down the owner's pi session
# (taskkill /IM node.exe, 2026-08-24). Do not reintroduce that here.
set -uo pipefail
me=$$
killed=0
while read -r pid _ _ _ cmd; do
  [ "$pid" = "$me" ] && continue
  case "$cmd" in
    *nfl_swarm*|*supervise_wave*|*spawn.sh*)
      # double-check the pid is not our own ancestor
      if [ "$pid" != "$PPID" ]; then
        kill "$pid" 2>/dev/null && echo "stopped $pid: ${cmd:0:80}" && killed=$((killed+1))
      fi
      ;;
  esac
done < <(ps -ef 2>/dev/null | grep -iE 'pi -p|supervise_wave|spawn\.sh' | grep -v grep)
echo "stopped $killed swarm processes"
