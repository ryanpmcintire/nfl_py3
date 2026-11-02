# mod27-u3-beat-reporters

## Goal
Forward-score the declared beat-reporter grade (family mod24_beat_reporter_news) on 2026 with the fit from earlier years, no refit.

## State
Done 2026-11-02 (measured). Capture healthy: scheduler jobs beat_reporters_sat/sun/mon in scripts/capture_scheduler.py:518, last OK 2026-11-02 run 20261102T123006Z, 139,597 posts. 2026 own tagged (player+availability) posts exist every week Aug 3 to Nov 1 (38-155/wk, 1,304 total), so no backfill needed. 2026 games graded: weeks 2-8, 104 (week 1 absent from g26.parquet); coverage 93 with a tagged post, 86 nonzero; lead variant 80/67.
Method: scratchpad agentU3/fwd.py reuses build_feature/fit/metrics/boot from scripts/mod24_u9_beat_grade.py unchanged; fit on 2020-25 population (artifacts/pick_probability/20261020T184404Z/per_game.parquet), scored on 2026 g26.parquet (model_logit = logit of discrete cover prob). Output docs/mod27_u3_beat_forward26.json.
Results (n=104): base 61-43 (58.7%), LL .6827, Brier .2449; +beat 59-45, LL .6724, Brier .2400; +beat_lead 56-48, LL .6840, Brier .2456; 50% market 45.2%.
+beat accuracy -1.92 pts [-9.43,+5.83] p+ 0.312, decisive 11 up/13 down, LL gain +0.0103 p+ 0.90, Brier gain +0.0049 p+ 0.89, 24 flips, coef -7.35.
+beat_lead accuracy -4.81 pts [-9.62,0.0] p+ 0.037, decisive 3 up/8 down, LL -0.0012 p+ 0.36, 11 flips, coef -4.16.
Recorded unresolved_below_power: mod27_beat_reporter_news_pickterm_all_fwd2026, ..._lead_fwd2026 (registry/weak_signals).
Per-fold coefs: LOSO folds from the 2024-25 grade (-11.3..+1.2) stand; the 2026 fit is one full fit, no folds.

## Tried
Declared arms and term definitions unchanged; 2 cells; no tuning. Leakage: build_feature raises if any post is at or after the deadline.

## Next
Rerun after Week 9+ settle. Beat terms stay unserved (sign unstable, lead cell reversed forward). Not pooled with 2024-25 (different fit regime).

## Open
Week 1 absent from g26.parquet. Beat term is zero for 2020-23, so its coefficient is identified from 2024-25 only. True first-official-entry timestamps still absent.
