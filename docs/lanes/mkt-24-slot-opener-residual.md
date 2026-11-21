# MKT-24

## Goal
Kickoff-slot opener residual: is the opening line systematically off by Thursday, Sunday early/late/night, Monday, Saturday, holiday or international slot (ROADMAP MKT-24). Nothing here is served. Current unit: one declared forward look at 2026 holiday games.

## State
Unit 1 DONE 2026-11-03: no slot mispriced; 20 cells unresolved (scripts/mkt24_slot_residual.py -> artifacts/mkt24_slot_residual/report.json).
Unit 2 DONE 2026-11-14: `--split era parity parity_opener` -> artifacts/mkt24_slot_residual/reliability.json; 38 cells `unresolved_below_power` (family `mkt24_slot_opener_residual`, `mkt24_u2_*`), 113 looks. Residual = final home margin minus the line. Slot pattern reliability: era -0.27 [-0.65,+0.38] P+ .27; parity +0.51 [-0.44,+0.82] P+ .80; not zero-reliable, nothing closed. Thursday constant miss bounded only at about 3 points or more (planted control). Sunday night 2011-19 +3.34 [+1.59,+5.21] absent at the Tuesday opener (-0.56, P+ .24): line source or size, not shown absent.
Holiday prior (**read**, reliability.json): pooled 2011-25 n 61, -3.32 points, home 21-37 at the line (exact p .048, one of 8 slots); 2020-25 Tuesday opener n 29, -2.64 [-6.98,+2.28] P+ .13, home 12-17 (p .46); 2011-19 SBR proxy -3.94 [-7.52,-0.02], 9-20. Residual SD all slots 13.155 points. Thanksgiving hosts fixed (DET, DAL): slot and team confounded (**inferred**).

### Forward declaration, written 2026-11-21 (session start 15:59Z per `date`), before any 2026 holiday game
- Games (history rule `holiday_mask`: Thanksgiving Thursday or 25 December, not neutral; **read** game_features 2026): Week 12 Thu 2026-11-26 `2026_12_CHI_DET`, `2026_12_PHI_DAL`, `2026_12_KC_BUF`; Week 16 Fri 2026-12-25 `2026_16_GB_CHI`, `2026_16_BUF_DEN`, `2026_16_LA_SEA`. Six games. Black Friday `2026_12_DEN_PIT` is logged as descriptive only: history put Fridays in `other_weekday`, so there is no prior; it is not in the test. Christmas Eve `2026_16_HOU_PHI` is a Thursday slot, not logged.
- Quantity: residual = final home margin minus the Tuesday multi-book opener home line (nflverse sign; **measured** margin_vs_open = result - tue_open_home_spread, max diff 0.0), opener from `nfl_ats.clv.live_tuesday_openers(data/market/raw)`, per game the row with most books, ties earliest. Also logged before each pick deadline: Splash pool line, served side and its probability (last published_picks row before the log).
- Test: home cover record at the opener over the six games, pushes reported and dropped, exact binomial one-sided low vs 0.5. Decisive record first, then mean residual with interval from the history SD, probability_positive, sign agreement with the 2020-25 prior (-2.64), then served-side record at pool line and opener (descriptive).
- Graded once, after `2026_16_LA_SEA` settles (2026-12-26), by `score`; Week 12 rows are logged, never graded alone. It is look 114 in family `mkt24_slot_opener_residual`; recorded via `weak-signals record` as its own cell then.
- Detectable size (**inferred**, hand arithmetic from SD 13.155; `declare` writes measured values): 3 games: SE 7.6 pts; one-sided .05 needs mean -12.5; 80% power needs -18.9; power at the -3.32 prior .11; exact binomial cannot reach .05 (best p .125). 6 games: SE 5.4; needs -8.8; 80% power -13.4; power at prior .15; binomial needs 0 home covers of 6 (p .016), power at the 0.362 prior rate .07, 80% power only if true home cover rate is 3.7%. So this look cannot close or promote anything; it is a sign replication that adds six games to the 61-game cell.

## Tried
- Unit 2 recorded slot cells carry the wider of block and game-level intervals. Concordance of two half means spans [-1,+1]; the product test is the usable Thursday statistic.
- 2026-11-21: wrote scripts/mkt24_holiday_forward.py (`declare` / `log --week auto|N` / `score`, writes artifacts/mkt24/forward_2026/), ruff check and format pass. Factored `holiday_mask(day)` out of scripts/mkt24_slot_residual.py and added an `if __name__ == "__main__":` guard (import ran its argparse); behavior meant to be identical, not re-run. `declare` aborted: "a declared game is already past its pick deadline" because `game_deadlines` was called on the whole season at once and returned 2026-09-13 20:00Z for every game (**measured**). Nothing written to artifacts/mkt24 yet. Nothing recorded.

2026-11-21 16:06Z: per-week deadline fix applied (deadline = kickoff for Thanksgiving/Christmas games, **measured** in declaration.json). Re-ran slot_residual `--split era parity parity_opener`: reliability.json byte-identical to the 11-14 file (fc /b). `declare` wrote artifacts/mkt24/forward_2026/declaration.json (6 games; measured SE 5.37, needed -8.83, 80% power -13.35, power at prior .152, binomial best p .0156, power .067: matches inferred). `log --week 12` wrote 2026_week12_20261121T160641Z.json (4 rows, lines null; Tuesday run fills). ruff check/format clean.

## Next
1. Orchestrator applies the scheduler job below (subagents do not edit the scheduler). Score once on 2026-12-26 or later.

Proposed diff, scripts/capture_scheduler.py, next to the `lead108_forward_pregame_*` jobs:
```
+    *(
+        Job(
+            f"mkt24_holiday_forward_log_{day}_{at.replace(':', '')}",
+            day,
+            at,
+            grace,
+            [str(UV), "run", "--no-sync", "python",
+             str(REPO / "scripts" / "mkt24_holiday_forward.py"), "log"],
+            True,
+            "MKT-24 forward log: pool line, Tuesday opener and served side for holiday "
+            "games before each pick deadline; no-op in weeks without one.",
+            added_on="2026-11-21",
+        )
+        for day, at, grace in (("tue", "21:00", 180), ("wed", "20:30", 120))
+    ),
```
Tuesday 21:00 follows the latest Tuesday Splash captures seen (20:05, 20:19 ET); Wednesday 20:30 is the last evening before Thursday kickoffs.

ROADMAP MKT-24 update: 2026-11-21 forward look declared in the lane before any 2026 holiday game: six games (three Thanksgiving, three Christmas Day), home cover record at the Tuesday opener, one-sided exact binomial, graded once after 2026-12-25; prior 21-37 (2020-25 opener 12-17, -2.64 points); six games give power about .15 at the prior size, so it cannot close or promote. Logger scripts/mkt24_holiday_forward.py (deadline fix pending); scheduler job proposed in the lane.

## Open
Zero crossing closes nothing; one fitted probability selects the side; every cell is a look; verdicts go through `nfl-ats weak-signals record`; the 2011-19 line is a proxy and is never called the opener. Week 16 flex cannot move Christmas dates but can move kickoff times; the logger re-reads the schedule each run.
