# ENG-48 Anytime-valid evidence monitor for every forward log

## Goal
Anytime-valid evidence monitor for every forward log: a weekly mixture-martingale update on paired log loss against the served model (ROADMAP ENG-48).

## State
Added 2026-11-03, unmeasured. Forward logs are read only once at Week 18; a sequential monitor gives an evidence value after every slate and a pre-registered way to read it, without peeking inflation.

## Tried
Nothing yet. Novelty: docs/lanes/forward-log-audit-2026-11-03.md (coverage only) and the mod23-unit15 lane. Grep for e-process, always-valid and e-value found no sequential-evidence monitor.

## Next
First unit: Implement the monitor as a read-only script, replay it on logged Weeks 4-9, and compare its interval with the plain bootstrap on the same games.

## Open
Data: artifacts/mod24_forward, mod25_forward and armb_forward runs plus the MOD-35, SKY-03, SKY-04 and ENV-02 ledgers. Zero crossing closes nothing; record unresolved results with a closing ground.

## Unit 1 (2026-11-03, scripts/forward_monitor.py)
Betting e-process on paired log-loss gain vs served p, matched ledger (28 graded games, weeks 7-8 only have matched rows; Week 9 matched pending 0). Bet: WSR 2024 predictable plug-in lambda with one-sided log(1/alpha), alpha .05, truncated at 0.5 of the largest adverse log ratio known pre-game (keeps wealth positive). Not the normal-mixture of docs/anytime_valid.md, which excludes log loss (unbounded). Output artifacts/forward_monitor/monitor.json.
Final e (measured): mod24 0.0015, mod25_combined 0.092, lead99 1.64 (max 3.15). Mean gain/game: -0.021, -0.005, +0.020. Games to e=20 at current estimate (inferred, growth mu^2/2s^2): lead99 ~116 more; mod24 and mod25 undefined (negative estimate). Nothing recorded as verdict; e below 20 is unresolved_below_power, not closure.
Weekly command (for a scheduler Job, after the ledger refresh): `.venv/Scripts/python.exe scripts/forward_ledger.py` then `.venv/Scripts/python.exe scripts/forward_monitor.py`
Open: bootstrap-interval comparison, other ledgers (MOD-35, SKY, ENV-02) not yet in matched set.
