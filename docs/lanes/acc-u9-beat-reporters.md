# acc-u9-beat-reporters

## Goal
Acquire timestamped NFL beat-reporter Bluesky posts (practice participation, limitations, lineup news) ahead of official reports; backfill plus forward capture. Parent: docs/lanes/accuracy-ceiling-theory.md. Acquisition only, no grading.

## State
Done 2026-10-01 (measured). Script scripts/mod24_u9_beat.py (modes backfill, retag, capture, report, leadtime). Endpoint public.api.bsky.app getAuthorFeed (unauthenticated, 0.3 s sleep). Raw: data/raw/beat_reporters/<run_id>/ (jsonl + manifest sha256). Processed: data/processed/beat_reporter_posts.parquet (131,895 rows, 117,856 own posts, 14,037 reposts flagged), beat_reporter_handles.csv, beat_reporter_leadtime_2025.csv.
Handles: 132 seeded, 114 verified (>=20 own posts and >=15% team-relevant); 27 teams with 2+, 32 with 1+; only 1 verified each for BAL, HOU, JAX, LAC, TEN. Many ESPN reporters cross-post almost nothing (e.g. Hensley 1 post). All accounts start Nov 2024 or later (profile age), so no pre-2024 coverage.
Own posts by season: 2023 312, 2024 33,754, 2025 60,554, 2026 23,236. Availability-tagged 5.1% (6,001); ruled_out 2,060, questionable 1,465, expected 957, DNP 739, first-team 421, full 373, limited 350, absent 315, boot/brace 186. Player-tagged 45.9% (full-name or unique-last-name on handle's team roster).
Lead time 2025 REG (descriptive): 1,038 starter team-weeks flagged (prior-week snaps >=50%; Out/Doubtful or DNP/limited on the final report); 295 have a beat out/limited/DNP/absent/boot post naming the player in the prior 6 days; median lead 65.5 h (IQR 29-74) before the weekly proxy; Out/Doubtful only n=173 median 56 h; 31% before Wed 18:00Z. Caveat: official time is nflverse's Saturday-18:00Z week proxy, not a true first entry, and posts may refer to the prior game's injury, so leads are overstated.
Capture: first run 20261001T184920Z fetched pages, added 2 new posts.

## Tried
First backfill failed (getProfiles key is profiles, not actors); fixed, rerun. Lexicon is fixed in the script (LEXICON_VERSION u9-2026-10-01), loose on purpose; "is out" and "limited" are noisy.

## Next
Add to capture_scheduler (owner/orchestrator, not done here): daily `uv run --no-sync python scripts/mod24_u9_beat.py capture`. Grading unit (not started): restrict to posts after the player's first practice day and before the official Wed report, compare to the opener model with leakage guard; weak-signals record needed before any verdict. Seed more JAX/TEN/HOU/LAC/BAL reporters.

## Grading declaration (written 2026-10-20 BEFORE outcomes read; family mod24_beat_reporter_news)
Script scripts/mod24_u9_beat_grade.py. Per team-game: non-repost posts with player_gsis and availability class, created_at in (team's previous game kickoff, deadline), deadline = min(kickoff, Sunday 16:00 ET of game week); assert every used post < deadline. Per (team,player) the latest post decides: adverse class -> unavailability prob from data/processed/player_availability_rates.parquet (target_season = game season, position __all__; ruled_out/absent = out/out; did_not_practice = none/dnp; limited = none/limited; questionable and boot_brace = observation-weighted pooled questionable rows); positive class (expected_to_play, first_team_reps, full) -> 0. Value = max(0, offense_rating + defense_rating) from player_participation_ratings (season = game season). Team loss = sum p*value; beat_home = away_loss - home_loss. Lead variant: only posts before Wednesday 18:00Z of the game week (convention from the leadtime measure above; not tuned). Population: artifacts/pick_probability/20261020T184404Z/per_game.parquet (2020-2025; 2026 absent, so beat coverage is 2024-2025 only). Arms: base (FIT_FEATURES), +beat, +beat_lead; LOSO with _fit_logit/FIT_RIDGE, scored on held seasons 2024-2025. 2 cells. Zero crossing closes nothing.

## Result (measured 2026-10-20, docs/mod24_u9_beat_grade.json, 533 games 2024-25, LOSO)
Grading unit DONE. Coverage 355/533 games with a tagged post (326 nonzero; lead variant 306/261). All-posts term: accuracy -1.31 pts vs base [-3.88,+1.69], p+ 0.175, decisive 14 up/21 down, LL -0.0048, IS +1.31 (gap 2.63), coef per fold -11.3..+1.2 unstable. Lead term: +0.75 pts [+0.19,+1.51], p+ 0.992 but only 6 decisive games (5-1), LL/Brier ~0 (+1.5e-5/+4.5e-5), IS +1.50. Base acc 57.8, model 52.9, market 50.8. Both recorded (unresolved_below_power, family mod24_beat_reporter_news). Neither served; 2026 wks 1-6 absent from per_game.
Next: rerun when a 2026 per_game exists; lead cell has too few decisive games to read.

## Open
True first-official-entry timestamps are not on disk (injury_news is NFL.com headlines; nflcom_injuries captures began 2026). Verified threshold (20 posts, 15%) is a judgment.
