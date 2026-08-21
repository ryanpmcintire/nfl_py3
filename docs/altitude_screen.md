# Altitude-adaptation screen — predeclaration

Written **before** `scripts/altitude_screen.py` scores any cover-rate outcome.
Per AGENTS.md this is a mined/exploratory lead-generation family: every cell
here is predeclared to record `unresolved_below_power` regardless of interval
shape (an interval crossing zero is the EXPECTED outcome for a real small
signal at this evaluator's ~2-point resolution, never a rejection ground).
Method, population, thresholds, and blocking are locked before any cell's sign
is seen — only population/flag-size diagnostics (stadium-name counts, season
ranges, n per venue) were examined before freezing this document, never a
cover-rate outcome.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator. Every cell below is recorded regardless
of sign.

## Family and mechanism

**Family: altitude adaptation.** Visiting teams whose chronic training/
living baseline is near sea level face reduced aerobic capacity, faster
fatigue onset, and (for kickers) altered ball flight when playing at Denver's
Mile High venue (1,609 m / 5,280 ft — **read**, uncontroversial documented
figure) and at occasional Mexico City games (Estadio Azteca, 2,241 m /
7,350 ft — **measured**, Wikipedia infobox fetched this session). Chronic
adaptation is a *team trait* (where a franchise trains/lives); acute exposure
is *per-game* (the visitor arrives 1-2 days before kickoff at most). The
market may price this coarsely because high-altitude host games are rare
(only Denver hosts regularly) and the visitor-side deficit varies widely by
opponent.

## Overlap check (measured this session)

- `docs/travel_rest_battery.md` / `scripts/nfl_travel_rest_battery_screen.py`
  use distance, timezone, rest, and neutral-site facts only. Grep for
  `elev|altitude|Mile High` over that doc hits only stadium-rename notes
  (lines 94-95) and timezone text — **no elevation feature has ever been
  built there**.
- The ENV archive (`docs/environmental_exposures.md`) covers air quality
  (AQI) and drought only — no elevation axis.
- A repo-wide search for `ENV-06` returns **zero matches** across `docs/`,
  `registry/`, `scripts/`, `src/`, `artifacts/` (**measured**, this session):
  that ID does not exist in this repo; nearest are ENV-02 precedent mentions
  inside the travel battery doc. Nothing named ENV-06 has ever consumed
  elevation.
- No entry in `registry/weak_signals.json` is keyed on altitude/elevation
  constructs (family name here is new).

## Data source and leakage posture

- Newest snapshot only: `data/raw/20260817T235649Z/schedules.parquet`
  (**read**, same snapshot the travel/weather batteries use). Columns used:
  `game_id`, `season`, `week`, `game_type`, `gameday`, `home_team`,
  `away_team`, `result`, `spread_line`, `location`, `stadium`, `div_game`.
- REG games only, full 2009-2025 window, `add_ats_outcomes` reused verbatim,
  pushes dropped — identical population construction to
  `nfl_travel_rest_battery_screen.py`.
- **Pregame-safe, no leakage caveat**: venue elevation and the visitor's
  modal home-stadium elevation are static reference facts about known,
  scheduled venues; `div_game` is a pregame-known scheduling fact. No
  game-time actuals are used anywhere. (Same posture as the travel battery,
  which had no leakage caveat either.)
- Visitor's own home venue = away team's **modal home stadium that season**
  among `location=='Home'` rows — resolves relocations (STL→LA, SD→LAC,
  OAK→LV) automatically; identical convention to the travel battery's
  `away_modal_stadium`. This uses only where a team hosts, a fact fixed at
  schedule release.

## Elevation reference table (new reference data)

Built this session: `registry/stadium_elevations.json`, keyed by the exact
`stadium` string in `schedules.parquet` (NOT `stadium_id`, which is
unreliable for neutral-site games — see `docs/travel_rest_battery.md`).
Coverage: every distinct `stadium` name appearing in REG games 2009-2025
(**measured**: 49 distinct names, enumerated from the snapshot before this
document was frozen; renamed stadiums at the same physical site get separate
entries with identical elevations).

Provenance, per AGENTS.md's label rule:

- **Measured/fetched this session**: Estadio Azteca 2,241 m (Wikipedia
  infobox); Denver's three names 1,609 m (5,280 ft, documented "Mile High"
  figure).
- **Reported general knowledge, approximate (labelled `inferred: true`)**:
  every other entry — city/area terrain elevation from Wikipedia stadium/city
  articles and map familiarity, NOT surveyed stadium-site values and NOT
  fetched live this session. Approximation error on these is tens of meters;
  the only threshold decision they participate in is the visitor-side
  baseline, where a ±50 m error moves the deficit by ±164 ft against a
  4,000-ft threshold. Borderline cases created by that error band are
  disclosed below rather than tuned away.

Borderline entries under the 4,000-ft (1,219.2 m) deficit threshold versus
Denver (visitor home ≤ 389.8 m qualifies): Glendale AZ venues (~355 m,
deficit ≈ 4,114 ft) qualify but sit within one approximation-error-band of
the line; Las Vegas (~610 m, deficit ≈ 3,277 ft) clearly misses. Versus
Azteca (visitor home ≤ 1,021.8 m): every venue except Denver itself
(deficit ≈ 2,070 ft) qualifies. These classifications are frozen here before
any outcome is scored.

## Predeclared cells (frozen before scoring)

All effects are `home_cover` subset-vs-complement gaps in accuracy points,
scaled to the full slate (`fraction_of_slate` × raw gap), week-blocked
bootstrap primary (block = `season*100+week`), season-blocked secondary,
20,000 samples, seed 20260821 — identical machinery to the travel/weather
batteries.

| # | name | flag definition | predicted direction |
|---|---|---|---|
| 1 | `altitude_deficit_4000ft` | venue_elev_ft − away_modal_home_elev_ft ≥ 4000 (1219.2 m) | positive home_cover edge (acute hypoxic disadvantage for near-sea-level visitors into DEN/Azteca) |
| 2 | `altitude_deficit_4000ft_division` | cell 1 flag AND `div_game` | positive but SMALLER than cell 1 (division visitors face Denver twice/year — chronic/repeat exposure blunts the acute deficit) |
| 3 | `den_home_vs_own_conference` | home_team == DEN AND away team in AFC (Denver's own conference) | negative home_cover edge relative to complement (AFC opponents visit Denver far more often than NFC ones → more chronic adaptation → smaller DEN edge) |
| 4 | `mexico_city_neutral` | stadium == 'Azteca Stadium' | positive home_cover edge for the designated home side (both teams lack chronic adaptation, but the designated away team carries the full acute travel/exposure burden) |
| 5 | `altitude_deficit_4000ft_era_2009_2017` | cell 1 flag restricted to seasons 2009-2017 | positive (early era, plausibly coarser market pricing) |
| 6 | `altitude_deficit_4000ft_era_2018_2025` | cell 1 flag restricted to seasons 2018-2025 | positive but attenuated vs cell 5 if the market learned; era split is descriptive, both windows recorded |

Cell 4 is thin by construction: exactly 4 REG games at Azteca exist in
2009-2025 (**measured** from the snapshot: 2016, 2017, 2019, 2022). It is
reported honestly at that n; a 4-game cell cannot clear any power bar and is
recorded as unresolved regardless of shape. Cell 3's flag size is also small
(DEN hosts ~8 games/year, roughly half vs AFC non-division opponents plus
division games); n is reported per cell in the artifact.

Threshold justification: 4,000 ft is round and externally meaningful —
meaningful hypoxic stress begins around 5,000 ft of venue altitude, so a
≥4,000-ft deficit captures most of the physiological gap for near-sea-level
visitors into Mile High while excluding modest-deficit cases (LV→DEN ≈
3,277 ft). It was chosen before scoring and is not tuned.

## What happens after scoring

Every cell is recorded via `nfl-ats weak-signals record` as
`unresolved_below_power` with its week-blocked interval, standard error,
`probability_positive`, sample-games and sample-blocks passed through
unmodified — including cells whose intervals exclude zero (that would be
surprising, not disqualifying) and cells whose intervals cross zero (the
expected shape). No terminal classification is available to this design: no
positive control exists for altitude effects at this evaluator, and split-half
reliability of the underlying trait (franchise altitude) is trivially 1.0 —
which is why a null-looking result here can never be a refutation.

---

## Results (measured 2026-08-21, `artifacts/altitude_screen/20260821T182533Z/results.json`)

Population: REG 2009-2025, 4,431 games, 114 pushes/missing dropped, **4,317
scored**, 0 unresolved stadium names. Week-blocked primary (294 blocks),
season-blocked secondary (17 blocks), 20,000 samples, seed 20260821. All
numbers below are **measured** from the artifact; every cell is recorded as
`unresolved_below_power` per the predeclaration — every interval here
crosses zero, which at this evaluator's ~2-point resolution is the expected
shape for a real-but-small signal and closes nothing.

| cell | n_flag | full-slate effect | week-blocked 95% | P+ | season-blocked 95% | P+ |
|---|---|---|---|---|---|---|
| `altitude_deficit_4000ft` | 133 | +0.0230 pts | [-0.2443, +0.2889] | 0.5662 | [-0.2195, +0.2484] | 0.5903 |
| `altitude_deficit_4000ft_division` | 44 | -0.0589 pts | [-0.2095, +0.0936] | 0.2213 | [-0.2130, +0.1105] | 0.2438 |
| `den_home_vs_own_conference` | 99 | -0.1046 pts | [-0.3273, +0.1193] | 0.1746 | [-0.3018, +0.0858] | 0.1507 |
| `mexico_city_neutral` | 4 | -0.0222 pts | [-0.0463, +0.0468] | 0.2050 | [-0.0461, +0.0467] | 0.2073 |
| `altitude_deficit_4000ft_era_2009_2017` | 71 | -0.0399 pts | [-0.4114, +0.3332] | 0.4155 | [-0.4259, +0.3039] | 0.4355 |
| `altitude_deficit_4000ft_era_2018_2025` | 62 | +0.0905 pts | [-0.2803, +0.4678] | 0.6873 | [-0.1941, +0.3487] | 0.7431 |

Reading (**inferred**, mechanism-level, not evidence): the primary deficit
cell leans the predicted positive direction (P+ 0.5662) but is far below any
claim-worthy confidence; the two chronic-exposure contrast cells (division,
own-conference) lean in the predicted negative direction (P+ 0.2213 / 0.1746
— i.e., ~78-83% likely that Denver covers LESS often against repeat-exposure
visitors), consistent with the adaptation story but individually unresolved;
the era split leans the predicted direction (early-era negative-ish, late-era
positive P+ 0.6873) though the sign of the raw gap flips between eras rather
than cleanly attenuating. Mexico City is 4 games and says almost nothing
(P+ 0.2050 against the prediction). Category 3 across the board: no
refuted-mechanism ground exists (franchise-altitude trait reliability is
trivially 1.0) and no positive control was run; nothing here is closed.
