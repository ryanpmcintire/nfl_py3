# Combine ingest (scout v5 Section B #6)

Status: complete, cheap ingest + feasibility readout only. No ATS screen was run
this session and no shared registry was written; the experiment stamp lives under
`artifacts/combine/<ts>/experiment_registry/` per the `arctic_shift_gate.py`
precedent. Owner scope: `scripts/ingest_combine.py`, this doc,
`data/raw/combine/`, `artifacts/combine/`.

## Dataset existence verdict

**exists** — measured 2026-08-22: `nflreadpy` 0.1.5 exposes `load_combine`,
which pulls the `nflverse-data` release asset `combine/combine.parquet`
(HEAD request returned 200, Content-Length 374318). Snapshot captured at
`data/raw/combine/20260822T143152Z/` with sha256 manifest
(`1b6c48a0...f980bf2cf`). The table has **8,968 rows, seasons 2000–2026**,
columns: season, draft_year/team/round/ovr, pfr_id, cfb_id, player_name, pos,
school, ht ("F-II" string), wt, forty, bench, vertical, broad_jump, cone,
shuttle.

**No gsis_id column exists in the source** (measured). Player identity is
carried by PFR id plus a derived gsis_id from the roster join only.

## Tidy output

`artifacts/combine/<ts>/tidy_combine.parquet`: one row per combine player-season
with player_name, name_norm, pfr_id, season, position, position_group,
ht_in, wt_lb, forty_sec, bench_reps, vertical_in, broad_in, shuttle_sec,
cone_sec, speed_score (`200 * wt_lb / forty_sec^4`), roster_name_collisions,
gsis_id (derived), join_method. Measurable null rates in the raw source are
high for bench (~40%), cone/shuttle (~42–43%), vertical/broad (~24–26%);
height parses from the "F-II" string.

## Join feasibility to local rosters/participation

Target: `data/players/raw/20260817T184901Z/weekly_rosters.parquet`
(seasons 2009–2025, weekly). Join tiers, all measured this session:

| tier | matches |
| --- | --- |
| exact `(season, pfr_id)` → roster gsis_id | 2,331 |
| normalized name+season unique fallback | 2,391 |
| pfr_id matched in any roster season → gsis_id | 262 |
| combined reachable | 4,984 / 8,968 = **55.6% overall** |

- **2016–2025 combine rows reach a gsis_id at 91.1%** (3,121/3,425); 2012+
  is 86.1%. The overall rate is dragged down by 2000s classes predating the
  local roster window and the 2026 class (0/319 — no rosters exist yet).
- **name+team is not testable**: combine rows carry no team assignment
  (`draft_team` is post-draft), so team disambiguation at the combine stage is
  impossible; joins are season-keyed only.
- Normalization ambiguity: lowercase, suffix tokens jr/sr/ii/iii/iv/v dropped,
  periods/apostrophes removed, hyphens spaced. Roster-side name collisions are
  rare: 0.11% of non-pfr-matched rows face a >1-player name collision and are
  excluded from the unique-name fallback.
- Local `weekly_rosters.pfr_id` is only partially populated (measured example:
  Jake Andrews has null pfr_id across 2023–2025), which caps the same-season id
  join; the name fallback carries recent seasons to ~90–96%.
- Participation reach (`players_on_play` gsis ids scanned across all seasons of
  the 20260813T131635Z snapshot): 43.0% of all matched combine rows appear;
  restricted to combine classes 2016+ that matched a roster, **62.4%
  (measured)**. Non-reach is dominated by players who never took an NFL snap —
  expected, since most combine invitees never play.

## Speed-score year-over-year stability by position group

Trait: speed score = `200 * weight_lb / forty_sec^4`, filtered 40yd ∈
[4.20, 6.00], weight ∈ [140, 420] lb, group-season n ≥ 10. Measured:
adjacent-season Pearson r of group means averages **0.938** across 21 season
pairs (min 0.829, max 0.994) — position-group means are highly stable
year-over-year. Group grand means span 71.9 (SPEC) to 100.1 (LB); within-group
SD of season means runs 2.2 (DB) to 4.3 (TE) against between-group gaps of
10–30 points, i.e. group membership dominates seasonal drift. Per-group detail
is in `feasibility.json`.

## Screenability through the week-blocked evaluator

Combine traits are static player-level priors fixed before any game is played.
Joined via gsis_id to player-week features they are leak-free by construction
(the trait exists before every week it could be joined to) and screen cleanly
through the existing week-blocked evaluator as time-invariant within-season
covariates — the same handling as other slow-moving preseason features. No new
evaluator machinery is needed; the binding constraint on sample size is the
roster join rate above (91.1% of 2016+ classes), not leakage. Screenable trait
families: speed score / straight-line speed (forty), mass-adjusted speed
(speed score), size (ht/wt), explosion (vertical, broad), agility (shuttle,
cone), and bench strength — each usable alone or pooled into a commensurable
athleticism composite, with the family declared before signs are seen per
AGENTS.md.

## Reproduction

```powershell
.\.tools\uv.exe run python scripts/ingest_combine.py            # fresh download
.\.tools\uv.exe run python scripts/ingest_combine.py --skip-download  # reuse latest snapshot
```

Gates (scoped to owned files): `ruff format --check`, `ruff check`, `mypy` (with
`MYPYPATH=src`) clean; full `pytest --basetemp C:\Users\Ryan\AppData\Local\Temp\opencode\pt_comb`
green after wiring `write_experiment_artifact()` (provenance contract test).
