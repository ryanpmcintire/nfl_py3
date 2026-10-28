# MOD-23 unit 10b v2: starter draft capital, corrected joins

## Goal
Corrected second version of unit 10b (family mod23_unit10b_draft_capital_v2). Same declaration as v1 (prior game top-11 off/def snaps, snap-weighted draft_ovr and years, home minus away; arm A 88+4, arm B 88+off/def draft_ovr; eras 2013-19 proxy, 2020-25 true, base 802-701). Zero crossing closes nothing; one fitted probability picks the side. Cumulative looks with v1: 8 (4 + 4).

## State
Script scripts/mod23_unit10b_v2.py, artifacts artifacts/mod23_unit10b_v2/ (run.log, report_all.json, weak_signals_batch.json). Draft data fetched to data/raw/draft_picks/<stamp>/ (nflreadpy draft_picks + players id map; none existed). Join: snap_counts pfr_player_id to draft_picks pfr id, fallback pfr->gsis via players then draft_picks gsis_id; unmatched = undrafted value max pick + 1. Coefficient mapping via imputer statistics_ (kept from v1 audit). Leak check: corrupt scored-season outcomes, arm B picks must be identical.

## Tried
Run launched in background (log artifacts/mod23_unit10b_v2/run.log).

## Next
Read report_all.json; record cells with weak-signals record --batch artifacts/mod23_unit10b_v2/weak_signals_batch.json (unresolved_below_power).

## Open
Results, match rate, recording status: see below once filled.

## RESULT (done, recorded)
Match 71.8% of snap rows via pfr id (gsis fallback added 0.0%; 2,804 unmatched unique players coded undrafted 337). Base reproduced 802-701, 897-834 (2013-19). d_acc pp (season-block pp): A 2013-19 -0.58 (0.12), A 2020-25 +0.47 (0.75), B 2013-19 +0.46 (0.76), B 2020-25 +0.13 (0.60). Flips W-L: A 73-83, 77-70; B 62-54, 47-45. B ll 0.7246 v 0.7182 (2013-19), 0.6964 v 0.6969 (2020-25). Coefs negative, share_positive 0-0.33. Leak check: weeks <10 picks identical when weeks >=10 outcomes corrupted (134 games). 4 cells recorded, unresolved_below_power. Next: forward log of arm B (v1 decision unchanged).
