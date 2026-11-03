# Week 8 forward scoring 2026-11-03

## Goal
Score Week 8 for every forward-only row after Monday CHI at SEA settled (CHI 34, SEA 17).

## State (measured 2026-11-03)
- Settled: settle_mon 11-02 23:59 OK 8105 settled, 0 pending; scorecard shows 0 pending. game_features.parquet still has NaN score for 2026_08_CHI_SEA (not the scorer input).
- POL-10 scorecard artifacts/prospective_scorecard/20261103T153520Z: frozen card 65-56 (n=121), Wk8 6-8, Wk8 Best Pick lost, Best Pick 4-4.
- prospective-score artifacts/prospective_scoring/20261103T153531Z (decision line, to date; vs coin [95%] P+): active_model 66/121 +0.045 [-0.053,0.164] 0.777; base 63/121 +0.021 0.677; mod07_weak_signal_stack 63/121 (Wk8 4/14) 0.681; weak_stack_deadline_drag 66/121 (Wk8 7/14) 0.895; weak_stack_expected_lineup_loss 64/121 (7/14) 0.839; interim_playcaller (LEAD-29) 61/119 (4/14) 0.595; pbp08_early_window (LEAD-65) 56/104 (4/14) 0.763; inactives_refresh 64/121 (6/14) 0.667; crew_tilt 65/121 (6/14) 0.733; injury_signal_refresh_tilt 73/121 (7/14) 0.999.
- LEAD-53: tuesday_nominee Wk8 CHI AWAY won, sunday_nominee Wk8 TEN at CIN lost; scored 4/7 and 3/6 (P+ 0.669, 0.512).
- POL-11/MKT-08 refresh ledger (graded vs Tuesday line): Wk8 changed-side 8 games, refreshed 4-4, before_refresh 3-5; season refreshed 25-18, before 21-22.
- MOD-27: --grade Wk8 14 games, card 6-8, challenger 7-7, 1 differs (challenger won).
- Card Wk8: final 6-8 (measured); Tuesday card 5-9 (inferred: prior postmortem 4-9 on 13 plus CHI win).

## Tried
Ran only documented scorers once; no ledger edits.

## Next
Re-run after Week 9 settles; scheduler jobs for scorecard Tue 09:30 pending root.

## Open
MOD-24/25, LEAD-75, LEAD-99, draftcap have no Week 8 score step (accumulate; scored after Week 18 or lost).
