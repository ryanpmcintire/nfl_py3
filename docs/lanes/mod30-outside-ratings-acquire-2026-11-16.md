# MOD-30 outside ratings: acquisition and provenance (2026-11-16)

## Goal
Acquire independent pregame model ratings (college first) so a later declared unit can test their disagreement with the opener. Acquisition and provenance only: no outcomes read, no accuracy computed. ROADMAP MOD-30; prior lane docs/lanes/mod-30-mod30-public-ratings.md.

## State
- Script: scripts/mod30_outside_ratings.py (offline; reads only id, date, team, division and pregame Elo columns; makes zero network requests; writes data/external/outside_ratings/{cfb_pregame_elo.parquet, inventory.json, .gitignore}). The local `.gitignore` (`*`) keeps the data untracked (measured: git status empty for data/external). Run 2026-11-16 20:10 UTC, exit 0.
- College, CFBD Elo via the already-local cfbfastR-data schedule snapshots (sportsdataverse/cfbfastR-data commit 935eed66 and later snapshots under data/cfb/schedules/raw): home_pregame_elo and away_pregame_elo present 2012-2025; join to data/processed/cfb_game_features.parquet game_id = 100% every season (measured, 503 to 762 repo games per season; 704-808 FBS-involved games with both Elo). Not used anywhere else in src/scripts (read, grep pregame_elo), so new to the model.
- As-of for CFBD Elo: inferred pregame by construction (sequential Elo from prior results), NOT a timestamped pre-kickoff publication; values are recomputed retroactively by CFBD. Independence from the market reported (CFBD Elo is results-only), not verified.
- NFL ESPN predictor: NOT fetched. Policy: espn_nfl_injuries red (robots disallows our agent sitewide, no fetches), and data/market/espn_pickcenter_blocked.json records the 2026-09-20 403 stop. Prior lane records root closed this route 2026-11-03. Coverage/join unmeasured. Same policy covers college ESPN predictor.
- 538 NFL Elo: NOT fetched; no config/source_policies.json entry (CC-BY-4.0 licence per prior lane). Archive ends 2021-10 commits for nfl-elo folder; CSV published after the fact, so commit time cannot prove pregame (prior lane, reported).
- nfelo: NOT fetched; no policy entry; repo has no licence; owner approval pending since 2026-11-03 (prior lane has the proposed entry text). Pregame provable only for 2025 via bot commit history.

## Tried
Policy gate read in-script (network_gate in inventory.json: all three fetch_permitted false, requests_made 0). Sagarin NFL 1998-2011 already exists (data/raw/sagarin, MKT-11) and is prior work, not reopened.

## Next
1. College-first screen candidate: CFBD pregame Elo difference vs opener-implied margin on the repo CFB benchmark, frozen definition declared before any grade; must run an intercept-only and opener-only control (opener-grade-needs-opener-input). Declare as a new unit; leave-one-season-out.
2. Owner decision (physical approval, not research): add source_policies entries for fivethirtyeight_nfl_elo (CC-BY-4.0, historical only) and github_nfelo_public_ratings; then extend this script with a polite fetcher (generic UA, 2 s delay, stop on 403/429/5xx).

## Open
- CFBD Elo as-of is construction-based only; a chain check (pregame equals previous postgame) would prove it but reads an outcome-derived column, so it was not run here.
- Whether CFBD Elo carries a preseason prior from recruiting or returning production (would still be pregame) is unverified.
