# PER-13 Week 17-18 resting hazard

## Goal
Pregame hazard that a seed-locked or eliminated team rests starters in Weeks 17-18, priced or not at the opener (ROADMAP PER-13).

## State
- Unit 1 (2026-11-03, family per13_locked_seed_rest, 10 looks, all unresolved_below_power): 19 locked team-games 2009-25; starter snap drop +34.1 pp [21.9, 45.8]; opener ATS diff -3.26 [-9.5, +3.3], fade P+ 0.847; open-to-close move against locked team 2.91 [1.26, 4.71] P+ 1.00. Script scripts/per13_locked_seed_rest.py.
- Unit 2 (2026-11-03, family per13_term_design, 5 looks, 4 recorded): no-intercept OLS of move toward team on d_lock, d_elim, Weeks 16-18 2009-25, 1212 rows; lock coef -1.18 [-2.19, -0.22], elim -0.63 [-1.31, +0.03]; LOSO R2 0.0014 (in-sample 0.0116); cover slope on predicted move -1.45 [-5.6, +2.9] (expected +1 if unpriced). Frozen artifacts/forward_terms/per13_locked_rest/frozen_coefficients.json.
- Unit 3 (2026-11-21 11:16 EST) forward runner, nothing recorded:
  - scripts/per13_forward.py: `pin` (once; wrote freeze_manifest.json, sha256 38e3f6b17d9eb10e5174965c3f74f590ffeeb4c91b6bdd542628bf4cf19077b2 = file unchanged since created_at 2026-11-03T20:04:50Z, by mtime), `refresh` (refresh/2026_weekNN_<stamp>.json), `log --week auto` (2026_weekNN_<stamp>.json in the ledger term format; SKIP outside 17-18). Every run verifies the pinned hash and fails closed.
  - Declared 2026-11-21, before any Week 17-18 data: as-of rule (newest data/raw_schedules snapshot at or before run time, scores only with kickoff before run time, log fails closed if an earlier week lacks a final), feature rule (ss.simulate at target week, NSIM 6000, seed [20261103, season, week]), probability rule (served line_sweep method reproducing pool_card, read at the half-point line nearest home pool line minus term).
  - per13_term_design.py main() no longer rewrites the frozen file or procedure.md (it did on every rerun).
  - Ran: pin -> PINNED, MATCH on rerun; log --week auto -> "SKIP: week 11 ...", exit 0; refresh -> refresh/2026_week11_20261121T161519Z.json, 4 s, 151 games, last kickoff 2026-11-20T01:15Z. Scratch log --week 11 --any-week (scratchpad only): 13 games, method market_residual, runs.
  - Refresh (sim output; 600 outer draws, SE about 0.014 at p 0.14): P(locked at Wk17 Tue) SF .138, BUF .070, NYG .037; Wk18 NYG .208, SF .205, BUF .155. P(elim by Wk17) TEN .997, MIA .997, LAC .995, ARI .992, NYJ .985, IND .937, PHI .883.

## Tried
Intercept-only control = unflagged team mean ATS (unit 1). Old lock_probabilities() used the Sept 8 schedule plus game_features results; replaced by the snapshot as-of rule.

## Next
Owner/orchestrator: add the scheduler jobs below and `"per13_locked_rest"` to TERM_DIRS in scripts/forward_ledger.py. Dec 29 (Wk17) and Jan 5 (Wk18): check the log file exists before the first kickoff. Score after 2027-01-11 beside the served card at the opener; record through weak-signals.

```diff
+    Job("per13_lock_refresh_tue", "tue", "14:20", 360,
+        [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "per13_forward.py"), "refresh"],
+        True, "PER-13 weekly as-of lock, clinch and elimination probabilities for the Week 17 and 18 "
+        "Tuesdays; no-op after the regular season.", requires=("weekly_lock",), added_on="2026-11-21"),
+    *(
+        Job(f"per13_forward_log_{day}_{at.replace(':', '')}", day, at, grace,
+            [str(UV), "run", "--no-sync", "python", str(REPO / "scripts" / "per13_forward.py"),
+             "log", "--week", "auto"],
+            True, "PER-13 forward log: frozen locked-seed rest term beside the served card in Weeks "
+            "17-18, before kickoff; SKIP other weeks; never changes a side.",
+            requires=("weekly_lock",) if day == "tue" else (), added_on="2026-11-21")
+        for day, at, grace in (("tue", "14:30", 360), ("sat", "11:00", 240), ("sun", "10:30", 60))
+    ),
```

## Open
- Half-point snap: |term| < 0.25 never moves the read line; declared, not tuned. Serving needs a pinned pick_probability.py edit plus re-enrolled validation (procedure.md step 5).
- 2026 schedule: Week 17 Dec 31-Jan 4, Week 18 Jan 10 (Saturday games TBD; sat 11:00 job covers them).

ROADMAP PER-13 update: Unit 3 2026-11-21: frozen term pinned (sha256 38e3f6b1...), scripts/per13_forward.py refresh/log built and run (refresh written; log SKIP Week 11); weekly refresh guarded from rewriting the frozen file; scheduler entries proposed in lane. Next: add jobs and TERM_DIRS entry; log Weeks 17-18 2026; score after 2027-01-11.
