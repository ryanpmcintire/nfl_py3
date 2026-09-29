# LEAD-80 unit 1: local source inventory

**Measured:** inventory only; no outcome scoring, model fit, field simulation or registry write.
Declaration: `docs/lanes/lead80.md`; complete source: `docs/lead80_protocol.md`.

## Inventory boundary

Inspected file names under configured data/artifact roots, including ignored files.
Candidate names match pool, Splash, tiebreak, opponent, contest, field, card, entrant, entry or guess terms.
Only registered pool-observation season/kind metadata and Python schema declarations were read.
Generic files, opaque archives and external/browser-held sources are not authenticated.
Name matching establishes candidates, not authenticity, historical coverage or absence elsewhere.

| Root | Exists | File names inspected | Candidate files | Links skipped | Read errors |
|---|---|---:|---:|---:|---:|
| `data` | True | 95805 | 545 | 0 | 5 |
| `artifacts` | True | 53551 | 1686 | 0 | 0 |

**Measured:** 0 registered pool-observation files; 0 dated to 2020-2025; 0 metadata errors.
Observation kinds: none.

| Selected candidate basename | Files | Example path (full counts above) |
|---|---:|---|
| `2026_week01_20260908_noon.json` | 1 | `data/splash/2026_week01_20260908_noon.json` |
| `2026_week01_field_distribution.tsv` | 1 | `data/splash/field/2026_week01_field_distribution.tsv` |
| `2026_week02_20260915_1302.json` | 1 | `data/splash/2026_week02_20260915_1302.json` |
| `2026_week02_field_distribution.tsv` | 1 | `data/splash/field/2026_week02_field_distribution.tsv` |
| `2026_week03_20260922_2019.json` | 1 | `data/splash/2026_week03_20260922_2019.json` |
| `2026_week04_20260929_1514.json` | 1 | `data/splash/2026_week04_20260929_1514.json` |
| `tiebreaker.json` | 58 | `artifacts/margin_predictions/2026-week-01-20260905T141453Z/tiebreaker.json` |
| `wp12_pool_rules.md` | 1 | `artifacts/fleet_reports_20260901/wp12_pool_rules.md` |

## Source gates

**Read:** `src/nfl_ats/pool_observables.py:24` `FieldObservation` exposes:
`season`, `week`, `entries`, `paid_places`, `prize_notes`, `observed_at_utc`, `observer`.

**Read:** `src/nfl_ats/pool_observables.py:35` `DistributionObservation` exposes:
`season`, `week`, `game_id`, `home_share`, `away_share`, `unlocked_at_utc`, `observed_at_utc`, `observer`.

**Read:** these schemas supply field size/prizes and aggregate per-game side shares.
They omit entrant-level weekly cards, Best Pick nominations and submitted tiebreak guesses.
`docs/pool_observables.md:31` identifies the registered snapshot location.

**Read:** `docs/tiebreaker.md:3` says final score of the week's last game.
`src/nfl_ats/pool_workbench.py:37` defaults to 1/0 correct/incorrect points, 0.5 push points,
and a +1/0 Best Pick bonus/penalty; line 52 names `final_score_last_game`.
`docs/pool_rules.md:13` delegates pick locking to min(kickoff, Sunday 16:00 ET).
These current descriptions are not authenticated season-specific 2020-2025 rules.
Closest-total versus exact-score loss, residual tie handling and guess deadline are unverified.

**Read:** `docs/prospective_bestpick_tiebreaker.md:26` describes the project's own 2026 guesses.
Those forecasts cannot substitute for historical opponent submissions or field outcomes.

**Inferred:** the replay source gate is not cleared. No conditional field fit is authorized.
Authentic opener/total issuance and ingestion against the deadline remain unchecked
because the rules/field prerequisites are unmet.

## Prespecified results availability

**Measured:** evaluated decisive games = 0; record unavailable (no games scored).
IS/OOS accuracy, Brier, log loss, margin MAE, joint-score log score, actual tiebreak utility,
IS-minus-OOS gaps, season-block 95% intervals and `probability_positive` are not estimable.
Fold coefficients, coefficient stability and reliability bands are unavailable; no fit ran.

| Held-out season | Registered observation files | Fold coefficients / IS / OOS / gap |
|---|---:|---|
| 2020 | 0 | Not fitted: source gate |
| 2021 | 0 | Not fitted: source gate |
| 2022 | 0 | Not fitted: source gate |
| 2023 | 0 | Not fitted: source gate |
| 2024 | 0 | Not fitted: source gate |
| 2025 | 0 | Not fitted: source gate |

**Read:** B=3, F=6, E=2; declared looks = (3+36+16)*7+25+35 = 445.
**Measured:** executed outcome looks = 0; inventory counts are not performance estimates.
**Inferred:** missing-source status leaves the mechanism open; no negative or terminal verdict.
No effect estimate exists to register. The orchestrator has no record command from this unit.

## Next source unit

Obtain season-specific official rules and entrant cards/Best Picks/guesses for 2020-2025,
with field size, contest/week IDs, tiebreak game, effective dates and submission deadlines.
Keep held-out opponent submissions as outcomes. Authenticate frozen opener/total captures
and pre-deadline issuance/ingestion before chronological conditional/unconditional replay.

## Inventory errors

- [WinError 5] Access is denied: 'data\\market\\raw\\20260928T192821Z'
- [WinError 5] Access is denied: 'data\\players\\inactives\\20260928T192923Z'
- [WinError 5] Access is denied: 'data\\raw\\nflverse_injuries\\20260928T192512Z'
- [WinError 5] Access is denied: 'data\\raw\\public_betting_live\\20260928T192818Z'
- [WinError 5] Access is denied: 'data\\raw\\pytest-player-props-final'
