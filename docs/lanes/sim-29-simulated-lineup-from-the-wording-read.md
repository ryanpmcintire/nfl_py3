# SIM-29

## Goal
Simulated lineup availability from the injury wording read: draw each listed starter's play status from his read probability at game start (ROADMAP SIM-29).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: SIM-12 and SIM-16 model in-game exits and pregame attrition from base rates. LEAD-106 measured the wording read as a better play probability than the ladder. Measured mechanism: draw each listed starter's availability from his read probability, apply the measured replacement cost, and check the simulated margin against the real margin by availability bucket. Fitted term: Simulator input from the read probability; the simulated margin shift enters the calibrated probability only as a challenger term.

## Tried
Novelty check only. Nearest: LEAD-108 (read in the PER-10 mixture), SIM-12, SIM-16. Differs: Feeds the read into the simulator's lineup rather than the margin mixture.

## Next
First unit for one agent: For 2026 games with a stored read, run a small fixed set of simulated games per listed-starter draw, compare the simulated margin shift with the realised margin residual in games where a starter was out (replacement cost check), and report games per bucket before any score.

## Open
Rules: measured mechanism only, never a tuned constant or offset; aggregates are checks, not targets; each stage compared with real play-by-play with game-cluster intervals; flag-off hash must equal HEAD before any flag-on run; unresolved results are reported as measured. Look count and family declared before any sign is seen. Data: LEAD-106 reads, snap counts, SIM-12 replacement costs (local).

## 2026-11-11 11:02 ET first unit (sub-agent, no simulation)
- Stated facts: handing the base model true game-day actives is worth +0.73 pick-accuracy points vs the opener (mod57-p2l-availability-ceiling); engine whole-game backup-QB effect -.0571 vs real -.0604 (sim11-strength-transmission).
- measured: forward reads exist for Week 9 only (332 sunday + 28 thursday rows, 15 games), no realised snaps or participation for Week 9 yet (2026 participation absent; snaps stop before Week 9). So the comparison ran on the stored Weeks 2-8 sample: 331 player rows, 151 team-weeks (scripts/sim29_lineup_expectation.py, artifacts/sim29/lineup_expectation.json). Expected missing COUNT per team-week vs realised: read MAE .362, corr .861; ladder (availability.py:434 resolve_unavailability) MAE .400, corr .846. MAE gain .038 [.012,.067] P+ .9985; corr gain .015 [-.004,.037] P+ .93. Count only: the value-weighted version (players.py:1501 _injury_value_features severity x role share x value rate) was not built; needs _update_player_value_state (players.py:1584) state. Sample is stratified, not all starters. Recorded sim29_read_vs_ladder_team_missing_count (shard registry/weak_signals/sim29_lineup.json; the CLI then errored on another unit's mod57 effect_units entry, not mine).
- Hook: draw each listed starter present with prob p_read per game before Engine.game (sim10_drive.py:1549, rng-independent stream); map missing to xq/strength via the generator's per-team strength input (xq[pos] at ~1440) as replacement cost; generator must export per-team starter value shares and SIM-12 replacement cost per position group (OL only exists).
- Adjudication: 15 Week 9 games now; realised snaps after the next snap-count load; a simulation check with per-bucket counts needs several more weeks (about 14 games a week). Verdict: comparison measured, hook specified, adjudication from about Week 12.
