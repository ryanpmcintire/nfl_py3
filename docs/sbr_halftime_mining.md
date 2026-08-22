# SBR halftime/2H mining — field inventory and late-information sizing

Source-mining pass on ALREADY-IN-HOUSE data: the raw SBR snapshot
(`data/raw/sbr_odds/20260819T192226Z/`, fetched 2026-08-19) carries a `2H`
column plus per-quarter scores (`1st/2nd/3rd/4th`) that
`scripts/ingest_sbr_odds.py` parsed but **dropped** from
`sbr_odds.parquet` (**read**: that script's `build_games()` records none of
them). Code: `scripts/sbr_halftime_mining.py`. Artifacts:
`artifacts/sbr_2h/{field_inventory,analysis}.json`, `records.txt`. No
registry writes; ATS-relevant cells are returned as RECORD lines only.

## 1. Field inventory (**measured** this session, full re-parse of the raw snapshot)

- Every one of the 15 season tables (2007-08 .. 2021-22) carries the `2H`
  column; all 8,050 team-row cells are numeric or `"pk"` — zero blanks,
  zero `"NL"` (**measured**, same scan method as the ingest doc §2).
- The `2H` column interleaves spread and total across a game's two rows
  exactly like Open/Close: smaller magnitude = second-half spread (on the
  favored team's row), larger = second-half total. Verified on hand-checked
  examples (e.g. 2015 opener PIT@NE: cells 27.5/3.0 → NE −3 in 2H against a
  14-3 halftime lead; plausible). Convention reuse is **inferred** from the
  ingest doc's documented convention + spot checks, not site-documented.
- Coverage: **4,025 games** across seasons 2007-2021, every season fully
  populated (**measured**, `field_inventory.json.coverage_per_season`).
- Extracted per game: `two_h_home_spread`, `two_h_total`,
  `h1_margin_home`, `h1_pts_total`, `m2_margin_home` (realized 2H margin),
  joined to `sbr_odds.parquet` on (season, game_date, away_rot, home_rot)
  to inherit `game_id`. Join validated: my independently recomputed close
  spread matches the parquet's `close_home_spread` on all matched rows to
  1e-9, and unmatched rows = exactly the 534 pre-2009 games (**measured**
  via in-script assertion).
- Data defects found and excluded (**measured**, enumerated in
  `field_inventory.json.excluded_games`):
  - **11 games, all on 2018-09-23** (2018 week 3): larger-magnitude cell
    41-56 points — implausible as a 2H total (league 2H totals run ~13.5-32.5
    everywhere else); these look like full-game totals pasted into the 2H
    column for that slate, which also corrupts the min/max split itself.
  - **4 games in 2020**: both rows literally `"pk"` — no usable line.
  - n usable: **4,010**.
- Timestamp caveat: nothing establishes WHEN the 2H number was captured
  (kickoff? halftime? post?) beyond its name; treat as a live-line *proxy*
  (**inferred**, same provenance gap as SBR Open in the ingest doc §3b).

## 2. (a) Info value: does the 2H line know something about the FULL game?

Design (**measured**, `analysis.json.cv` / `.bootstrap`; leave-one-season-out
CV logistic on P(home covers vs close), week-clustered bootstrap ×200):

| Model | Features | CV accuracy | CV log-loss |
|---|---|---|---|
| M0 pregame only | close | 52.87% | 0.6917 |
| M1 + realized H1 score | close, H1 margin | 74.54% | 0.5110 |
| M2 + 2H line | close, H1 margin, 2H line | 74.81% | 0.5106 |

- **Total halftime-information pie: +21.98 accuracy points** over pregame
  alone [boot CI +20.02, +23.85], P+ 1.000 (**measured**) — but read the
  decomposition before quoting it: M1 shows almost ALL of that comes from
  the already-realized halftime score, which is a component of the final
  margin (mechanical, not market skill — **inferred**, arithmetic).
- **Live-market increment beyond the realized score: +0.215 accuracy
  points** [−0.420, +1.138], P+ 0.685 (**measured**) — small in forced-pick
  terms because forced-pick headroom at 74.5% baseline is compressed toward
  the remaining coin-flip mass.
- Continuous form is sharper (**measured**): for the REMAINING half margin,
  m2 ~ close + 2H line gives coefficient **+0.670 per point of live line**
  [+0.527, +0.812]; R² 0.094 → 0.113 vs pregame-only. So each point of live
  revision moves expected remaining margin by about 0.67 points — real
  information, but the live line explains only ~2 points of R² of the
  remaining half. Under full oracle conditioning (close + H1 + line) the
  slope is +0.579 [+0.351, +0.808].
- Framing discipline: this sizes the late-information pie at the HALFTIME
  decision point and connects to the movement channel's "+1.72 measured
  direct" ceiling (docs/ceiling_error_split.md:93, **read**). It CANNOT
  transfer to Tuesday-frozen picks: the 2H line does not exist at lock time,
  and no pregame feature family was added, so no leakage regression test is
  required or applicable (**inferred**; this is an oracle/descriptive lane).

## 3. (b) How sharp are books live?

(**Measured**, `analysis.json.sharpness_overall`, n=4,010.)

- Residual (realized 2H margin − 2H line, home convention): bias **−0.322
  pts** [−0.612, −0.032], P+ 0.015 — a small but interval-excluding road-side
  lean in the 2H number; descriptive, not a wagerable cell.
- Residual SD **9.37 pts** vs realized 2H-margin SD 9.93 → scale-free
  sharpness 0.944: the live line absorbs only ~11% of remaining-outcome
  variance ((1−(9.37/9.93)²)≈0.11, **inferred** arithmetic on measured SDs).
  Books are NOT close to solving the second half.
- MAE 7.43 pts; 27.4% of games within ±3; favorite covers 49.36% of
  non-pushes (124 pushes) — no favorite-longshot-style distortion visible at
  this granularity; mean absolute error flat across |line| buckets
  (7.35 / 7.54 / 7.24).
- 2H total: bias +0.57 pts, residual SD 9.69.

## 4. (c) Era trend: has live sharpness increased?

(**Measured**, `analysis.json.era`; per-season table in analysis.json.)

| Era | n | bias | resid SD | scale-free sharpness |
|---|---|---|---|---|
| 2007-2011 | 1,335 | −0.387 | 9.747 | 0.954 |
| 2012-2016 | 1,335 | −0.068 | 9.347 | 0.938 |
| 2017-2021 | 1,340 | −0.510 | 9.019 | 0.940 |

- Raw residual SD declines at **−0.64 pts/decade** (SE 0.28) — but the
  scale-free ratio declines only **−0.009/decade (SE 0.009)**: once you
  normalize by the shrinking scoring environment, there is NO clear
  increase in live sharpness 2007→2021 (**measured** slopes, **inferred**
  reading). Bias shows no trend (−0.11 ± 0.29 per decade).

## 5. Record lines (returned, not written to registry)

See `artifacts/sbr_2h/records.txt` — five RECORD lines:
`sbr_2h_halftime_oracle_vs_pregame` (+21.978 acc pts, oracle pie),
`sbr_2h_live_market_beyond_realized_score` (+0.215 [−0.420, +1.138] P+0.685),
`sbr_2h_line_efficiency_bias` (−0.3217 ats pts P+0.015),
`sbr_2h_line_remaining_half_info_slope` (+0.6696 [+0.5273, +0.8118]).
All classified `unresolved_below_power`; none gates anything. The
full-margin-slope OLS cell (−3.40 without the H1 term) was computed but
deliberately NOT recorded as a registry-style cell: its sign is a
collinearity artifact of omitting realized H1, not market information
(**inferred**, shown by comparison with the +0.579 oracle-conditioned slope).

## 6. Verification

`--self-test` gate passes (parsing conventions, date parsing, CV/OLS/
sharpness plumbing on synthetic frames). Gates: ruff format/check on the
script, mypy src, pytest scoped with the session basetemp — see session log.
