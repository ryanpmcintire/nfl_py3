# Venue milestone states screen

Written **before** `scripts/venue_milestone_screen.py` scores anything
(predeclaration frozen first, per project convention). Family: **venue
milestone states** — rare, schedule-determined points in a franchise's venue
life cycle where crowd-energy / familiarity mechanisms predictably differ from
a normal home game. Modeled on `docs/travel_rest_battery.md` /
`scripts/nfl_travel_rest_battery_screen.py`: same population construction
(REG 2009-2025, `add_ats_outcomes`, pushes dropped), same joint week-blocked
bootstrap (season-blocked secondary), same full-slate effect scaling, same
`probability_positive` definition, measure-only (never writes either registry
JSON; recording happens via separate explicit `nfl-ats weak-signals record`
calls).

## Point-in-time safety

Every flag in this battery is a **schedule fact**: it is derived solely from
`schedules.parquet` columns known at schedule release (`season`, `week`,
`gameday`, `home_team`, `away_team`, `location`, `stadium`) plus static
reference mappings (which physical venues are renames of each other; which
seasons a franchise changed permanent home venue). Nothing uses scores,
in-game actuals, weather, or market data. **All flags are point-in-time safe
by construction** — no leakage regression caveat applies, same posture as the
travel/rest battery.

## Overlap audit (grep of `registry/weak_signals.json` signal names + docs, done before scoring)

| Existing signal | Why this battery is not a re-measure |
| --- | --- |
| `travel_rest_home_off_bye` / `travel_rest_away_off_bye` (`home_rest>=13` / `away_rest>=13`) | Adjacent, disclosed: those use a rest-days threshold that conflates true byes with MNF/primetime extra rest, and are unconditioned on venue framing. Cells (d1)/(d2) here require a strict >=12-day gap to the team's immediately preceding game (a scheduled bye) AND condition on the game being home vs road. Correlated inputs; do not sign-test-pool together. |
| `bias_battery_extra_rest_edge` | Pooled extra-rest edge, different construct (rest advantage vs opponent), no venue conditioning. |
| `pick_conditioned_off_bye_fade_pre2018` | Pick-conditioned replication against the production model, 2011-2017 close-grade walk-forward — different population, grading, and mechanism framing. |
| `travel_rest_international_game` (`location=='Neutral'`) | Cell (b) here DELIBERATELY EXCLUDES one-off neutral-site internationals (Tottenham etc.), so it does not double-cover that cell; see exclusions below. |
| No prior signal exists for home openers, new-venue debuts, or former-stadium returns (name-level grep of all 341 registry signals for `stadium\|venue\|open\|debut\|swing\|milestone\|relocat` found none in these families). | Clean. |

## Cells (directions frozen before scoring)

Outcome is always game-level `home_cover` (accuracy_points units, full-slate
scaled). Subset-vs-complement on the full REG 2009-2025 slate.

1. **`venue_milestone_home_opener`** — the HOME team's first `location=='Home'`
   game of its season (per-team chronological order by `gameday`). Mechanism:
   crowd energy / ceremony elevation. Predicted direction: **positive**
   home_cover edge.
2. **`venue_milestone_new_stadium_debut`** — a franchise's FIRST regular-season
   home game in a venue that is new to the franchise that season (brand-new
   stadium opening or relocation destination). One flagged game per
   franchise-venue change. Mechanism: unfamiliarity/no established home-field
   routine outweighs novelty energy. Predicted direction: **negative**
   home_cover edge. Inline mapping table (opening/relocation facts are
   *reported* from standard public stadium/franchise histories, then
   *measured*-verified against the modal home strings in
   `data/raw/20260817T235649Z/schedules.parquet`; expected n_flag=12):
   - 2009 DAL — Cowboys Stadium opens (renamed AT&T Stadium 2013; same venue)
   - 2010 NYG, NYJ — New Meadowlands Stadium opens (MetLife from 2011)
   - 2014 SF — Levi's Stadium opens
   - 2014 MIN — TCF Bank Stadium (temporary, 2014-2015, disclosed)
   - 2016 LA — relocation STL→LA, Los Angeles Memorial Coliseum (temporary)
   - 2016 MIN — U.S. Bank Stadium opens
   - 2017 LAC — relocation SD→LAC, StubHub Center (temporary)
   - 2017 ATL — Mercedes-Benz Stadium opens
   - 2020 LA, LAC — SoFi Stadium opens
   - 2020 LV — relocation OAK→LV, Allegiant Stadium opens
   Known exception disclosed before scoring: MIN hosted ONE extra game at
   TCF Bank Stadium in 2010 (Metrodome roof collapse forced relocation); that
   one-off is NOT a new-permanent-home debut and is excluded from cell 2
   (each debut flags only its declared debut season's first home game in the
   new venue).
   Exclusions (disclosed): one-off neutral-site internationals (Tottenham,
   Wembley, Mexico City, Toronto) are NOT counted — they are covered by
   `travel_rest_international_game` and lack the "franchise's new permanent
   home" mechanism. Renames of the same physical venue (e.g. Cowboys→AT&T,
   Dolphin→Sun Life→Hard Rock, Arrowhead→GEHA) are NOT debuts.
3. **`venue_milestone_former_stadium_swing`** — a team playing a REGULAR-
   SEASON road game at a physical venue that was its own home in some EARLIER
   season (relocation revenge; physical identity via a rename-collapsing alias
   map, e.g. Oakland-Alameda County Coliseum ≡ O.co Coliseum ≡ Ring Central
   Coliseum). Mechanism: visiting former tenant outperforms. Predicted
   direction: **negative** home_cover edge. Enumeration duty: the script must
   enumerate every actual case 2009-2025. Expected enumeration (inferred from
   the venue table above, verified programmatically): **the set is expected to
   be EMPTY** — every venue vacated during the window (Qualcomm, Edward Jones
   Dome, Oakland Coliseum, LA Memorial Coliseum post-Rams, TCF Bank Stadium,
   StubHub Center) stopped hosting NFL REG games immediately, and no venue
   vacated pre-2009 hosted REG games in-window. If empty, the cell reports
   `insufficient_data` honestly and nothing is recorded for it.
4. **`venue_milestone_post_bye_home`** — the HOME team's first game after a
   bye (strict definition: >=12 calendar days since that team's immediately
   preceding game of the season; excludes MNF-style short extra rest).
   Venue-conditioned variant of the rest battery, disclosed above. Predicted
   direction: **positive** home_cover edge.
5. **`venue_milestone_post_bye_road`** — mirror of cell 4 for the AWAY team.
   Predicted direction: **negative** home_cover edge. The predeclared
   venue-conditioning contrast is the sign DIFFERENCE between cells 4 and 5
   (bye rest helps more at home than on the road); each cell stands alone for
   recording.

Five cells total (within the 4-6 gate). Directions frozen as stated above
before any scoring run.

## Method

Identical machinery to `scripts/nfl_travel_rest_battery_screen.py`:
week-blocked primary bootstrap (`block = season*100 + week`), season-blocked
secondary, 20,000 resamples, seed **20260821**, full-slate scaled effect =
raw subset-complement gap × fraction of slate, `probability_positive` =
share of raw (unscaled) draws > 0. Population: newest
`data/raw/*/schedules.parquet` snapshot, REG 2009-2025, pushes/missing results
dropped via `add_ats_outcomes`.

## Multiplicity and recording posture

Mined/predeclared 5-cell battery, uncorrected multiplicity: every scoreable
cell is predeclared to record `unresolved_below_power` regardless of interval
shape (an interval crossing zero is the EXPECTED shape for a real small
signal and is never grounds for rejection). `wrong_sign_resolved` would apply
only if a whole week-blocked interval sat entirely below zero on the WRONG
side of its frozen direction; no positive-control bound is run. Cell 3, if
empty, records nothing (nothing measurable exists to record).

## Results (measured this session)

Run: `.\.tools\uv.exe run python scripts/venue_milestone_screen.py --output
artifacts\venue_milestone_screen\predeclared_run` on snapshot
`data/raw/20260817T235649Z/schedules.parquet`; scored population **4,317**
REG 2009-2025 games (4,431 REG games minus 114 pushes dropped by
`add_ats_outcomes`); seed 20260821, 20,000 draws; artifact
`artifacts/venue_milestone_screen/predeclared_run/results.json`; registry
stamp `registry/experiments/venue-milestone-screen/predeclared_run.json`.

| Cell | n_flag | Full-slate effect (pts) | Week-blocked 95% CI | P+ | Season-blocked P+ | Frozen direction |
| --- | --- | --- | --- | --- | --- | --- |
| `venue_milestone_home_opener` | 544 | −0.133 | [−0.667, +0.390] | 0.3115 | 0.2917 | + |
| `venue_milestone_new_stadium_debut` | 12 | +0.026 | [−0.068, +0.115] | 0.7449 | 0.7166 | − |
| `venue_milestone_former_stadium_swing` | 0 | — | — | — | — | − |
| `venue_milestone_post_bye_home` | 344 | −0.232 | [−0.669, +0.196] | 0.1446 | 0.1444 | + |
| `venue_milestone_post_bye_road` | 360 | −0.001 | [−0.453, +0.445] | 0.4995 | 0.5057 | − |

Enumeration findings (**measured**): the new-stadium debut table matched the
schedule exactly — all 12 declared debuts found, each first game falling in
its declared debut season (asserted in-script). The former-stadium swing
enumeration is **EMPTY**: after excluding shared-stadium co-tenancy (NYG at
MetLife hosting NYJ, LA at SoFi hosting LAC — the venue is the visitor's
CURRENT home, not a former one), zero regular-season games 2009-2025 were
played at a visiting franchise's former physical home. Every vacated venue
(Qualcomm, Edward Jones Dome, Oakland Coliseum, LA Memorial Coliseum post-
2019, TCF Bank Stadium, StubHub Center) stopped hosting NFL REG games
immediately on vacatur. This confirms the predeclared expectation above; the
relocation-revenge cell has no measurable population in this window and
records nothing.

Classification: no week-blocked interval sits entirely on the wrong side of
zero, so `wrong_sign_resolved` does not apply to any cell; no positive-control
bound was run. All four scoreable cells are category 3,
`unresolved_below_power`. Two cells observed leans OPPOSITE their frozen
direction (home opener negative-leaning at P+ 0.31 against a frozen +;
new-stadium debut positive-leaning at P+ 0.74 against a frozen −) — both
tiny, both unresolved, reported as `probability_positive`, never as
"contains zero". The venue-conditioning contrast predeclared in cells 4/5:
the bye-rest lean is more negative at home (−0.232 pts) than on the road
(−0.001 pts), i.e. the data lean AGAINST the frozen "bye helps more at home"
contrast, weakly and unresolved.
