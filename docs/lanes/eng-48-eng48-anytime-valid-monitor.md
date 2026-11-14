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

## State (2026-11-14, through Week 9, measured)
- Rerun on the played-card ledger, 43 matched graded games (W7-9). e by week (W7, W8, W9), threshold 20: lead99 0.78, 1.92, 7.74; mod25_combined 1.01, 0.44, 0.50; mod24 1.74, 0.39, 0.33. Mean log-loss gain per game +0.0632, +0.0225, +0.0070. Games to e=20 at the current estimate (inferred, plug-in growth): lead99 about 11, mod25_combined about 361, mod24 about 4108.
- Every ledger candidate on its own games (17 with a probability): artifacts/forward_monitor/per_term.json, table in the ENG-49 lane. Highest e after lead99: mod25_wfj 4.32 (15 games), mod27_pool_move 1.80, draftcap 1.73; lowest mod24 0.09 (88 games). The game bootstrap agrees in sign with every e above or below 1 except mod25_combined, mod54, mod41_avgwin_served (e 0.997), mod41_residual_load_served and surface (positive mean gain, e below 1).
- Control: the card lost to a no-information 0.5 by 0.0373 log loss on the 43 games and 0.0590 on Week 9, so e against the card overstates skill. Against 0.5 lead99 is +0.0259 [+0.0047, +0.0471], probability_positive .992.
- Card: nothing served, nothing closed; all unresolved_below_power. Recorded with the ENG-49 batch (17 cells, family eng49_forward_ledger_vs_card). Looks: 91 across the three lanes (ENG-49 lane has the count).

## Next
- Weekly rerun is scripts/forward_weekly.py (Job text in the ENG-49 lane, not yet in the scheduler).
- Add a second e-process against the no-information 0.5 and write the per-term rows from forward_monitor.py itself.
- Before lead99 can cross 20 (possibly within one slate), write down how a crossing is read: it is sequence-valid evidence against the card's probabilities on those games, and it enters the card only as a fitted term through ENG-50, never as a side change.
