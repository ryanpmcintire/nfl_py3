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

## Open
True first-official-entry timestamps are not on disk (injury_news is NFL.com headlines; nflcom_injuries captures began 2026). Verified threshold (20 posts, 15%) is a judgment.
