# Lane: forward-scorable terms, 2026 Weeks 9-18

## Goal
Make SKY-03 (oN x defense completions per target), SKY-04 (sharpness-weighted book mean minus median) and ENV-02 (roof-open probability) genuinely forward-scorable: coefficients frozen on completed seasons, each week's term value and combined home probability written before kickoff to artifacts/forward_terms/<term>/2026_weekNN_<UTC stamp>.json. Never served or published. Scored only after a week settles, against the opener, beside the served card. Zero crossing closes nothing; these stay unresolved_below_power.

## State
Runner: scripts/forward_terms_week.py --week N|auto [--term sky03|sky04|env02|all] [--no-fetch]. Base probability = served card in artifacts/margin_predictions/2026-week-NN-<newest>/pool_card.csv (pick_probability converted to home side); combined = sigmoid(logit(base) + b * term). Refuses without that card, skips games already kicked off, never overwrites a term/week file.
Frozen coefficients (written once, with hash): artifacts/forward_terms/<term>/frozen_coefficients.json.
- sky03: c fit on all 2019-2025 team-game rows (no 2026 outcomes), b = pooled LOSO coefficient from artifacts/sky03/def_traits/term.json. Runner fetches the 2026 PFR defense file itself (data/raw/pfr_advstats/<stamp>/ with manifest).
- sky04: 1/MSE book weights on all of 2023-2025, b from artifacts/sky04/sharpness.json. Reads live 2026 quotes under data/market/raw.
- env02: open-probability logistic and five-term roof beta fit on all completed seasons; delta logit = beta * (term - train mean), only for ARI/ATL/DAL/HOU/IND hosts (others 0). Uses newest cached MEX bulletin under data/raw/forecast_archive/mos_bulletins/MEX.
Deviations from the measured protocol (inferred, state them when scoring): sky03 c is frozen at its end-of-2025 value, not expanding; sky04 cutoff is last book observation at or before Tuesday 12:30 ET (study used noon); env02 forecast is the latest cached MEX run at job time, not the Sunday pool-decision cutoff; env02 combined probability is a delta-logit on the served base, not a refit five-term model.

## Tried
Rehearsal for Week 8 with the clock faked to 2026-10-27 17:00 UTC: sky03 14/14 games with term, env02 14/14 (DAL host nonzero), sky04 0/14 (no 3-book observation inside the game week before the cutoff). Frozen files written for all three terms (hashes in each frozen_coefficients.json).
Measured: sky03 frozen c = -0.191 vs c_last -0.248 in term.json (unexplained gap, reported as measured, not filled); b = -0.105 from term.json. sky04 b = -0.431. env02 roof beta 0.0048 per unit probability (train mean 0.026).

## Added 2026-11-03 (SKY-01 x2, ENV-02 surface switch)
Terms: sky01 (beat-reporter out-count differential, b -0.003925 from artifacts/sky01/fitted_term.json), sky01snap (snap-share weighted, b +0.019904 from artifacts/sky01/drift_weighted.json; lane names no single current variant, both live), surface (ENV-02 point-in-time surface-switch flag, five-term logit beta 0.20717 fit on extended population 2011-2025). Run: --term sky01|sky01snap|surface (all = six terms). Frozen files carry real created_at_utc 2026-11-03T14:35Z; hashes sky01 884e2439..., sky01snap 51c47dbe..., surface d4463691....
Pregame inputs and timestamp sources: sky01/sky01snap use out-class beat posts with created_at before Tue 12:00 ET opener and <= run time (data/processed/beat_reporter_posts.parquet, sha and max created_at recorded; schedule from newest data/raw snapshot); sky01snap adds snap shares strictly before the game week (newest data/players/raw). surface uses the newest schedule snapshot (venue surface published pregame; away-team modal home surface from earlier games only).
Deviations (inferred): combined = sigmoid(logit(served base) + b*term), not a refit; surface fit population is the extended one with a flag from the latest schedule snapshot; sky01 weights depend on 28% unmatched players weight 0.
Rehearsal Week 8, fake clock 2026-10-27 17:00 UTC, output in tests/scratch (deleted): 14/14 games with term for all three. Scheduler Job text needs no change (default --term all, runner argv unchanged); its description names three terms, cosmetic only.

## Cleanup still owed (cap hit before it ran)
Delete rehearsal outputs: artifacts/forward_terms/{sky03,sky04,env02}/2026_week08_20261027T170000Z.json and the fake-stamped fetch dir data/raw/pfr_advstats/20261027T170000Z (a copy of the 2026 PFR defense file named by the faked clock). Keep the three frozen_coefficients.json. Then run: F:\Repos\nfl_py3\.tools\uv.exe run --no-sync python -I F:\Repos\nfl_py3\scripts\forward_terms_week.py --week 9 --no-fetch (expect MISSING INPUT until the Week 9 card exists; never executed for real).

## Missing inputs for Week 9
- All terms: Week 9 served card (artifacts/margin_predictions/2026-week-09-*/pool_card.csv) exists only after weekly_lock (12:20 ET Tue).
- sky03: pbp 2026 stops at week 7 (capture 20261027T220410Z; the weekly pbp capture runs Tue 22:04 UTC, after the lock), so traits are lagged one week (recorded as weeks_lagged_vs_week_minus_1). A pbp refresh after Monday night, before the runner, removes the lag. PFR 2026 defense had no scheduled capture; the runner fetches it from nflverse each run.
- sky04: after 12:10 ET the capture must put at least 3 of the 11 frozen books at one timestamp; recent private captures carry bovada, mybookieag, williamhill_us only (Oct 28 capture had all three; Nov 2 had two). Otherwise term is blank. Study used 11 books.
- env02: no 2026 forecast archive; runner uses the newest cached MEX bulletin (paired wind capture) for ARI/ATL/DAL/HOU/IND hosts.

## Next
Root adds the Job entry below. After each week settles: score each file's combined_home_probability vs base on the opener (Brier, log loss, accuracy, decisive games), pooled across weeks, with probability_positive; record with `nfl-ats weak-signals record`.

## Open
Missing captures for Week 9 are listed in Update. Week 9 files cannot exist until weekly_lock writes the Week 9 card (12:20 ET Tuesday).

## Scheduler Job entry (root adds to scripts/capture_scheduler.py, after weekly_lock; not edited by this lane)
```
    Job(
        "forward_terms_tue",
        "tue",
        "12:35",
        105,
        [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "forward_terms_week.py"), "--week", "auto"],
        True,
        "Freeze the SKY-03, SKY-04 and ENV-02 forward predictions for the coming week after the 12:10 book capture and the weekly lock, before the 14:20 lock window closes. Research only, never served.",
        requires=("odds_private_tue_1210", "weekly_lock"),
        added_on="2026-11-03",
    ),
```
Check that `requires` means "succeeded earlier today" for these two names before relying on it.
