#!/usr/bin/env bash
# Safely stop ALL swarm processes (workers + supervisors).
# Uses Windows CIM for reliable process enumeration - git-bash `ps` cannot
# see native processes, which is how five concurrent supervisors accumulated
# on 2026-08-24. Kills ONLY PIDs whose command line references swarm paths.
set -uo pipefail
killed=0
pid_self=$$

# Single source of truth: query by command line via PowerShell CIM
pids=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'supervise_wave|spawn\.sh|nfl_swarm' -and \$_.ProcessId -ne $pid_self } | Select-Object -ExpandProperty ProcessId" 2>/dev/null)

for pid in $pids; do
  # resolve what this pid actually is before killing
  cmdline=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"ProcessId=$pid\").CommandLine" 2>/dev/null)
  case "$cmdline" in
    *supervise_wave*|*spawn.sh*|*nfl_swarm*)
      # never kill our own lineage
      if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
        taskkill //PID "$pid" //F >/dev/null 2>&1 && echo "stopped $pid: ${cmdline:0:90}" && killed=$((killed+1))
      fi
      ;;
  esac
done
echo "stopped $killed swarm processes"
