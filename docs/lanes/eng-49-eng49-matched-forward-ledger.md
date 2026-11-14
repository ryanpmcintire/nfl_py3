# ENG-49 Matched-game forward ledger

## Goal
Matched-game forward ledger: one schema carrying every frozen candidate probability on the same game ids (ROADMAP ENG-49).

## State (2026-11-14, through Week 9; Week 10 not scored)
- Ledgers rebuilt: artifacts/forward_ledger_allweeks (Weeks 1-9, 136 graded, strict all-candidate match 0) and forward_ledger_core (mod24, mod25_combined, lead99: 43 matched graded, W7 14, W8 14, W9 15, 0 pending).
- Served = played card (measured): weekly W-L 9-7, 14-2, 6-10, 9-7, 8-7, 6-8, 7-7, 6-8, 5-10 = 70-66, equal in every week to the pool-line grade of the final card side in tests/scratch/fixpicks/root_combined.csv (0 side and 0 grade disagreements on 136 games; card line equals pool line on 136). Weeks 1-8 sides and probabilities identical to the 11-03 ledger.
- Coverage (graded games): mod24 88 (W4-9), mod25_combined 87, lead99 43 (W7-9), draftcap 29 (W7, W9), mod35_cold_exposure 7, and Week 9 only (15 each) mod25_wfj, mod27_pool_move, mod54, env02, sky01, sky01snap, sky03, sky03_separated, surface, three mod41 terms. sky04 0 (null probabilities). artifacts/forward_terms/per13_locked_rest has frozen coefficients but no weekly file and is not in TERM_DIRS.
- Per-term table (measured): artifacts/forward_monitor/per_term.json, built by tests/scratch/eng49/per_term_week9.py. Paired log-loss gain against the card, 95% game bootstrap, probability_positive (P+), decisive W-L, evidence value e (threshold 20):
  - lead99 (declared row: latest read before each pick deadline; corrected 2026-11-14 15:22 ET) 23-20, +0.0432 [+0.0102, +0.0782] P+ .995, 6-1, e 11.60. lead99_first_view (earliest read; named 2026-11-14 after its results were seen, so Weeks 7-9 are exploratory and its forward sample starts at Week 10; monitor only) 28-15, +0.0632 [+0.0192, +0.1101] P+ .998, 13-3, e 7.74. mod25_wfj +0.0986 [+0.0282, +0.1656] P+ .997, 7-1, e 4.32. draftcap +0.0438 [-0.0237, +0.1097] P+ .901, 7-2, e 1.73.
  - mod41_avgwin_served +0.0509 P+ .937, 3-1, e 1.00. mod27_pool_move +0.0506 P+ .859, 5-2, e 1.80. mod41_residual_load_served +0.0276 P+ .740, 2-2, e 0.40. mod54 +0.0264 P+ .703, 5-4, e 0.70. mod25_combined +0.0100 P+ .727, 18-14, e 0.16.
  - sky01 +0.0124 P+ .618, sky03 +0.0116 P+ .611, env02 +0.0115 P+ .611, sky03_separated +0.0108 P+ .613, sky01snap +0.0082 P+ .578, surface +0.0004 P+ .506: all 4-1 decisive, e 0.79 to 1.59.
  - mod24 -0.0000 [-0.0344, +0.0338] P+ .501, 19-15, e 0.09. mod41_residual_load -0.0304 P+ .332, 5-3, e 0.10. mod35_cold_exposure -0.0574 [-0.2048, +0.0694] P+ .219, 1-1, e 0.26.
- Controls (measured): a no-information 0.5 beats the card by +0.0590 log loss on the 15 Week 9 games (card 5-10), +0.0373 on lead99's 43, +0.0150 on mod24's 88. Against 0.5 four of 18 candidates are positive: lead99 as declared +0.0059 [-0.0231, +0.0349] P+ .656, lead99_first_view +0.0259 [+0.0047, +0.0471] P+ .992 (exploratory), mod25_wfj +0.0397 [-0.0082, +0.0874] P+ .949, draftcap +0.0005 P+ .521; the other 14 are negative (P+ .010 to .439). The frozen sky, env02, surface and mod35 terms change no side against their own record-time base (0 of 15, 0 of 7) and move log loss by under 0.012 against it; their 4-1 decisive record is the five games where that base differs from the published card.
- Looks: 17 candidates with a probability x 4 metrics (decisive record, gain against card, e, gain against 0.5) = 68, plus 14 own-base gains, 6 core monitor reads and 3 stacker cells = 91. Family eng49_forward_ledger_vs_card; 13 of its 17 cells share the same 15 Week 9 games and one card, so they are one correlated reading, not 17 votes.
- Card: nothing served, nothing closed. No e reaches 20; most of the apparent gain is the card's own Week 7-9 probabilities being worse than 0.5, which is the fix-picks lane's subject, not evidence for a term.
- Recorded 2026-11-14 (weak-signals record --batch artifacts/forward_monitor/per_term_batch.json --replace): 17 cells eng49_forward_<candidate>_vs_card_log_loss (log_loss_improvement, unresolved_below_power) plus eng50_online_stacker_cumulative_replay. `weak-signals pool --league nfl --effect-units accuracy_points` loads (exit 0).
- Correction 2026-11-14 15:22 ET (measured, forward_weekly.py rerun, exit 0): the ledger had scored lead99 from each game's earliest record; the declaration (lead99-llm-read.md) names the latest before the pick deadline; sides differ on 13 of 43. forward_ledger.py now keeps the latest verified record for lead99 (LATEST_BEFORE_DEADLINE; cutoff min(ledger deadline, pick deadline), equal on all 43) and emits lead99_first_view from the earliest. No other candidate's row or cell moved. Core set and stacker inputs stay mod24, mod25_combined, lead99; lead99_first_view is monitor only (forward_monitor.py MONITOR_ONLY, excluded in eng50_online_stacker.py _terms).
- Stacker with the declared row: coefficients lead99, mod24, mod25_combined W8 0.52, 0.40, 0.59 (penalty 1), W9 3.51, -0.95, 0.87 (penalty 0.1); 29 games, decisive 8-5, +0.0416 [-0.0609, +0.1503] P+ .782. Before: W8 0.43, 0.38, 0.57, W9 0.94, -0.03, 0.34 (penalty 1); decisive 10-5, +0.0493 [-0.0144, +0.1184] P+ .934. The lead99 coefficient is unstable across the two folds.
- Re-recorded with --replace: eng49_forward_lead99_vs_card_log_loss, eng49_forward_lead99_first_view_vs_card_log_loss (new, exploratory), eng50_online_stacker_cumulative_replay; pool loads (exit 0). Looks now 18 candidates x 4 = 72, 14 own-base, 8 core monitor reads, 3 stacker cells = 97, plus the 3 stacker cells re-read with the corrected input. per_term_week9.py labels both rows and reports first-view forward games (Week 10 on) separately; the monitor reports forward_games and forward_final_e.

## Tried
- 2026-11-03 unit 1: scripts/forward_ledger.py; a record counts only if written before the game deadline and, for forward terms, after the freeze; earliest record kept; MOD-27 backfill excluded; POL ids side-only via --pol-ids.
- 2026-11-03 fix: served = last published pick at or before the pick deadline (_played_card); the candidate's record-time base kept as <name>__base_p.
- 2026-11-14: added scripts/forward_weekly.py, because the ledger's --weeks default is fixed at 4-9 and the scheduler Job takes one command. It reads game_features, keeps weeks whose every game has a result, and runs in one process: ledger all weeks, ledger core (forward_monitor.CANDIDATES), forward_monitor, eng50_online_stacker. Verified by running it (exit 0, core 43 matched graded, outputs equal to the three scripts run by hand).

## Next
- Orchestrator adds the Job below. After each run an agent runs tests/scratch/eng49/per_term_week9.py, then the record --batch command above (cells are rolling, replaced weekly), then pool once.
- Add the per-term table to the monitor's own output and a no-information baseline column, so the weekly read does not depend on a scratch script.
- Extend the core set as candidates reach matched games (Week 10 adds the frozen terms' second week). XLG-12 CFB ledgers stay separate.

## Open
- registry/weak_signals_base_tags.csv has no rows for the 18 new cells (base is the served card, forward_ledger.py _played_card); not edited here.
- artifacts/forward_ledger (Weeks 4-9 strict) is superseded by forward_ledger_allweeks and is not refreshed by the runner.
- Zero crossing closes nothing; every cell above stays unresolved_below_power.

## Scheduler Job text (for scripts/capture_scheduler.py, not applied)
```python
    Job(
        "forward_ledger_tue",
        "tue",
        "13:15",
        105,
        [
            str(UV),
            "run",
            "--no-sync",
            "python",
            str(REPO / "scripts" / "forward_weekly.py"),
        ],
        True,
        "Rebuild the matched forward ledger over fully settled weeks, then rerun the "
        "evidence monitor and the online stacker on it, once the weekly lock has loaded "
        "the Monday result. Research only, never served. A partly settled week is skipped.",
        requires=("weekly_lock",),
        added_on="2026-11-14",
    ),
```
Slot and reason: Tuesday 13:15 is the first free slot after the Tuesday forward freezes (12:35, 12:50, 12:55, 13:05), which all require weekly_lock (12:20). Inferred, to confirm before enabling: game_features is rebuilt by the Tuesday pipeline (on 11-14 it held every Week 9 result and no Week 10 Thursday result). If results are late the runner repeats last week's output instead of scoring a partial week. Grace 105 is copied from forward_terms_tue; this job has no deadline, so any grace works.
