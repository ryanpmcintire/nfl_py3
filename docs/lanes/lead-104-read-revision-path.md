# LEAD-104

## Goal
View-to-view revision path of the LEAD-99 read: does the change from the Tuesday view to the last deadline view carry information the final level does not (ROADMAP LEAD-104).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: Each revision is the read's response to news. The level is anchored by the line; the revision is the news. Terms: final read, revision, line move since Tuesday, all fitted together; canonical-row-only scoring (latest before deadline) cannot see this. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: LEAD-99 scoring plan (canonical latest row) and ENG-49 (one row per candidate). Grep found no revision-path term in ROADMAP.md, lanes or registry. Differs: uses the multiple views already being paid for.

## Next
First unit: Compute per-game revision (last view minus Tuesday view, in logit) for Weeks 7-9 and its correlation with the line move toward the pick from Tuesday to the deadline; no outcomes needed. Week 8 Tuesday is lost, so Weeks 7 and 9 only.

## Open
Changes: Uses the forward log more fully. Data: Stored runs for all views, Weeks 7-9, then Weeks 10-18. Revision = last deadline view minus Tuesday view, in logit. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 log (Tue Nov 3 2026 4:16 PM ET)
Script scripts/lead104_revision_path.py (ruff clean). Only Week 7 has Tuesday plus deadline views (W8 Tuesday lost, W9 deadline not yet run): 13 games (SF_ATL lacks a market quote). Deadline = latest view with a row; market = packet home_spread_latest (positive = home favored).
Measured: revision vs move corr 0.885 [0.699, 0.988], P+ 1.0; slope 0.138 logit/pt [0.123, 0.340]; R2 0.78 [0.49, 0.98]. Residual SD 0.085 [0.031, 0.120] vs read noise 0.052 (1.64x) and revision noise sqrt2*0.052=0.074 (1.16x): marginally above noise, not distinguishable (n=13). Intercept +0.028.
Outcome look (1 look, W7, 13 decisive): residual sign agrees with cover at pool line 7-6, rate 0.54 [0.23, 0.77], P(>0.5) 0.61. Recorded: lead104_w7_residual_revision_sign_vs_cover (unresolved_below_power).
## Next
Rerun the script after Week 9 deadline views land (adds W9, n~28 total); then Weeks 10+. Fit revision as a term only with LOSO once n grows.
