# Era-mechanism screens (2026-09-01, WP34)

`docs/era_magnitude_report.md` (WP4, this session) closed by proposing **three
unrun mechanism cells**, one per sign-flipping era-split construct it found.
This document is the predeclaration for those three cells and, below the
predeclaration, the results of running them.

**Section 1-3 (the predeclaration) was written and saved before
`scripts/era_mechanism_screens.py` computed a single cover rate, effect, or
sign.** Everything in section 4 was produced afterwards, by that script, from
the artifacts named there.

## 0. Binding taxonomy (verbatim; restated because a doc and its subagents get
no session context injection)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism — a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

Also binding here:

- **Era magnitude, not presence.** Per-era magnitudes are reported
  separately. A sign flip is two magnitudes, never one average.
- **Within-week correlation is zero** — games inside a week are independent;
  no ICC is estimated or padded anywhere below.
- **Decisions are expected value** (`probability_positive > 0.5`), and a
  promotion bar is not a decision bar. None of these three cells is a play
  decision; all three are screens.
- **Never "needs N more games."** Where an estimate is imprecise, the
  interval and `probability_positive` are given and the reader judges.

### Provenance tags used below

**measured** = run this session, command/artifact given. **read** = file
opened this session, path:line given. **reported** = a doc or subagent says
so and it is NOT verified here. **inferred** = reasoning, not evidence.

### Mined-window discount (applies to all three cells, disclosed up front)

All three cells run on the full local history, which **includes the mined
2018-2025 seasons** that this project's overlay and battery families were
selected on, and cells 1 and 2 additionally reuse the exact rows their parent
batteries already scored. Every number in section 4 therefore carries the
mined-window discount: it is **reinforcing evidence about an already-seen
window, not an independent out-of-sample vote**, and none of the three may be
pooled with its parent battery's entries as an independent input. Each cell
is recorded `unresolved_below_power` unless a terminal ground in section 0 is
literally met — and section 3.4 states, ahead of the numbers, exactly what
would meet one.

---

## 1. Method common to all three cells (frozen)

| Item | Frozen choice |
|---|---|
| Grade | **close** — the pre-release closing `spread_line` recorded in `schedules.parquet` (cells 1, 2) or the same column as the market leg of the divergence (cell 3). Every parent battery here is close-graded; an opener-graded version of any of these is a different, future look. |
| Effect units | `accuracy_points` |
| Primary uncertainty | `nfl_ats.clv.week_blocked_bootstrap(block="week")`, **20,000 draws**, whole (season, week) blocks resampled with replacement |
| Secondary uncertainty | the same function with `block="season"`, reported **informational only**, flagged `DEGENERATE` when fewer than 10 season blocks exist (the project's existing floor, e.g. `bye_overval_home_edge_pre2011`'s own registry note, read) |
| Null | **within-week permutation, 200 draws** — the cell's own label is permuted *within* each (season, week) block, the statistic recomputed, and the null's mean, [2.5%, 97.5%] spread and the observed statistic's percentile within it are reported. The null is NOT assumed to be centred on zero: a within-week permutation preserves each week's own home tilt and slate composition, and this project has already measured such a null to sit off zero by construction (project MEMORY `home-tilt-null-artifact`). One permutation run is a spread, never a test. |
| Positive control | per-cell, section 2; either an external construct with more power (cells 1, 2 — the controls `docs/era_magnitude_report.md` itself predeclares) or an injected effect of a declared size (cells 1, 3) |
| Injection control design | inject a target effect of a declared size into the **real** population and its **real** block structure by flipping the minimum number of the relevant rows' binary response from 0 to 1, then run the identical estimator. **R = 25 seeded injections per target size, 2,000 bootstrap draws each** (a deliberately reduced-cost power probe, distinct from the 20,000-draw primary). Report the mean recovered effect and the **detection rate** = share of the 25 whose 95% week-blocked interval excludes zero. A target that would need more 0-rows than exist is reported `not_achievable` with the achievable ceiling stated — that ceiling is itself the answer to "how large an effect could this instrument even represent." |
| Seeds | each cell inherits its parent battery's own frozen seed: cell 1 → 20260821 (`bye_overvaluation_screen.BOOTSTRAP_SEED`, read), cell 2 → 20260821 (`primetime_cells_screen.BOOTSTRAP_SEED`, read), cell 3 → 20260820 (`sagarin_divergence_battery.BOOTSTRAP_SEED`, read) |
| Flag provenance | **every flag is imported from the battery that owns it, never re-derived.** `scripts/era_mechanism_screens.py` imports `bye_overvaluation_screen.load_population` / `build_bye_maps`, `primetime_cells_screen.load_population` / `build_long_table` / `build_cells`, `era_magnitude_profile._changepoint_grid` / `MIN_SEGMENT_SEASONS` / `build_hc_year_one_fade`, and `sagarin_divergence_battery.build_close_population` / `LARGE_DIVERGENCE_THRESHOLD` / `ERA_SPLITS`. `tests/test_era_mechanism_screens.py` pins the imported builders against a fixture. |
| Artifacts | `artifacts/era_mechanism_screens/<UTC timestamp>/results.json`, written through `nfl_ats.provenance.write_experiment_artifact` (so each run also stamps `registry/experiments/era-mechanism-screens/`) |
| Recording | one `nfl-ats weak-signals record` call per predeclared name, run under this session's cross-process registry lock. The script writes **no** registry JSON. |
| Multiplicity | 3 cells + 1 predeclared companion arm (cell 3's late era), uncorrected. Disclosed, not adjusted. |

---

## 2. The three cells (definitions frozen; transcribed from `docs/era_magnitude_report.md`, not redesigned)

### 2.1 `bye_overval_install_need_moderator`

**What already exists, and what is new.** `docs/era_magnitude_report.md`
§2.1 (read, lines 169-198) establishes that the parent mechanism —
bye-week overvaluation after the 2011 CBA — is **already fully predeclared
and already run** in `docs/bye_overvaluation_screen.md` (read, lines 1-7,
49-56, 115-156), whose five cells include both era arms
(`bye_overval_home_edge_pre2011` +0.271 pts / `_post2011` −0.330 pts). So
this work package **does not re-slice by era**: it runs only the NEW
moderator cell §2.1 defines.

**Script extended (named as required):** `scripts/bye_overvaluation_screen.py`
— its `load_population`, `build_bye_maps`, `POST_BYE_GAP_DAYS`,
`ERA_POST_MIN_SEASON`, `BOOTSTRAP_SAMPLES` and `BOOTSTRAP_SEED` are imported
verbatim by `scripts/era_mechanism_screens.py`. The base flag is therefore
byte-identical to `bye_overval_home_edge_post2011`'s, including the
2026-08-22 cross-season bye-map fix (read,
`docs/bye_overvaluation_screen.md:158-165`).

- **Population**: NFL REG, **seasons 2012-2025**, non-push `home_cover`
  (`ERA_POST_MIN_SEASON = 2012`, read `scripts/bye_overvaluation_screen.py:59`).
  The pre-2011 control is not split further — it holds 58 flagged games.
- **Base flag**: HOME team off strict bye (gap to its own immediately
  preceding game of the same season **>= 12 calendar days**,
  `POST_BYE_GAP_DAYS = 12`) **AND** the opponent NOT off bye.
- **Moderator (subset)**: "install-need" HOME team.
  **Complement**: neither changed.
- **Moderator operationalisation, frozen here because the local data forces
  a choice and the choice is made before any sign is seen:**
  - The **QB leg** is `install_need_qb` = the home team's **starting QB in
    its first REG game of season S differs from its starting QB in its first
    REG game of season S−1**, where "starting QB" is the passer with the most
    `qb_dropback` plays for that team in that game
    (`data/pbp/raw/<latest snapshot>`, `passer_player_id`; team codes
    normalised through `nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`).
    **Point-in-time safety:** both facts are settled before any bye-week
    game is played (a strict >=12-day gap cannot occur before week 3), so
    this is a pregame-known season fact for every row in the population, and
    the leakage regression test in `tests/test_era_mechanism_screens.py`
    pins that no row's own game contributes to its own moderator.
    "First REG game" rather than literally week 1 so the two 2017
    hurricane-postponed team-seasons (TB, MIA — **measured**: 30 of 32 teams
    have a week-1 REG game in 2017) are covered rather than dropped.
  - The **OC leg is NOT available**: **measured** — the repository holds no
    offensive-coordinator source (`find data -iname "*coach*" -o -iname
    "*coordinator*"` returns only `data/raw/interim_coaches`, a HEAD-coach
    capture; `schedules.parquet` carries `home_coach`/`away_coach` only).
    The moderator is therefore run on the **QB leg alone**. This makes the
    install-need subset strictly **narrower** than §2.1's "QB change OR OC
    change" definition: some genuine install-need teams (new OC, same QB)
    land in the complement, which **dilutes the contrast toward zero**. It
    is a power reduction and a conservative one, disclosed in the registry
    note; it is not a change of direction, population, or comparator.
- **Direction (predeclared, one-sided)**: if the CBA practice-cap mechanism
  is real, a bye week's now-capped extra practice time should still carry
  more marginal installation value for install-need teams than for
  continuity teams, so the post-2011 negative home-bye edge should be
  **less negative (or flat) for the install-need subset**:
  **contrast = (install-need arm effect) − (no-need arm effect) > 0.**
- **Comparator / statistic**: difference-in-differences inside the post-2012
  population, using the parent battery's own subset-vs-complement,
  full-slate-scaled convention *within each moderator stratum*:
  `effect_arm = (mean(home_cover | base flag, arm) − mean(home_cover | not
  base flag, arm)) × 100 × (n_flagged_in_arm / n_arm)`; the recorded effect
  is `effect_install_need − effect_no_need`.
- **Grade**: close. **Effect units**: `accuracy_points`.
- **Uncertainty**: section 1's table (20,000-draw week-blocked primary,
  season-blocked secondary, 200-draw within-week permutation null).
- **Positive control (two, both declared now)**:
  1. **External, exactly as `docs/era_magnitude_report.md` §2.1 predeclares
     it**: the identical install-need moderator contrast computed on
     `travel_rest_home_off_bye` (`home_rest >= 13`,
     `scripts/nfl_travel_rest_battery_screen.py`, **full 2009-2025 window**,
     a larger flagged population than the strict-bye cell). If a
     practice-time-driven moderator effect is real it should be visible
     there with more precision even though that population's bye
     identification is looser; if it is absent there despite the extra
     power, that bounds the install-need story.
  2. **Injection**, per section 1's design, targets **{0.25, 0.50, 1.00,
     2.00}** accuracy points.
- **Per-era magnitudes**: reported for the two eras the parent battery
  already uses (2009-2011 pre-CBA control, 2012-2025 post-CBA) for both
  moderator arms, so the reader sees four magnitudes, never an average
  across the flip.
- **Recording rule (frozen before the numbers)**: name
  `bye_overval_install_need_moderator`, league `nfl`, seasons 2012-2025,
  units `accuracy_points`, family `bye_overvaluation` (**correlated with the
  parent battery's own rows — never a second independent vote**), category
  `schedule`, classification `unresolved_below_power`, **unless** the whole
  week-blocked 95% interval sits **below** zero, in which case and only in
  which case `refuted_mechanism` with `--closing-ground wrong_sign_resolved`
  (admissible here precisely because a one-sided positive direction is
  frozen above). A positive-control bound requires control (1) to have
  *detected* a moderator effect of the size claimed and this cell to be
  flat; a bare interval crossing zero never closes anything.

### 2.2 `pt_post_mnf_sunday_changepoint`

**Script extended:** `scripts/primetime_cells_screen.py` (flag builder,
imported) plus `scripts/era_magnitude_profile.py`'s Stage-2a changepoint
machinery (`_changepoint_grid`, `MIN_SEGMENT_SEASONS = 3`, and
`build_hc_year_one_fade` for the control), reused rather than reimplemented.
`docs/era_magnitude_report.md` §2.2 (read, lines 289-316) establishes that
the 2017/2018 split is **the primetime battery's fixed convention applied
uniformly to all seven cells**, with no rule change found at that boundary;
this cell asks whether the data itself locates a break and where.

- **Population**: identical to `pt_post_mnf_sunday` — team-game rows where
  the current game is Sunday and the team's own strictly-prior REG game that
  season was Monday, on the primetime battery's eligible population
  (`has_prior` rows, REG 2009-2025, pushes dropped). Imported, not redefined.
- **Direction**: **none predeclared.** This is a structural/diagnostic cell,
  exactly like `docs/era_magnitude_profile.md`'s Stage 2a (read, lines
  121-128). Two-sided; `wrong_sign_resolved` can therefore never apply.
- **Comparator / statistic**: the real per-season series of the construct's
  full-slate-scaled effect (the battery's own `sign = -1` convention kept),
  then the optimal single-changepoint fit — the break season minimising the
  two-segment total sum of squared deviations over every candidate with **>=
  3 seasons on each side**. Recorded effect =
  **`post_break_mean − pre_break_mean`** in accuracy points at the located
  break. Also reported: the break season's bootstrap distribution (median,
  [2.5%, 97.5%], modal value and modal share, `stable` = modal share >=
  0.25, `era_magnitude_profile`'s own flag), the pre/post means with
  intervals, and — for the direct comparison the cell exists to make — the
  same two-segment statistic evaluated at the battery's **fixed** 2017/2018
  boundary.
- **Grade**: close. **Effect units**: `accuracy_points`.
- **Uncertainty**: section 1's table. The week-blocked primary resamples
  whole (season, week) blocks and recomputes the entire per-season series
  and its changepoint inside each draw (the same design
  `era_magnitude_profile.signal7_slope_and_changepoint` already uses with
  `nfl_ats.clv.week_blocked_bootstrap`, read).
- **Positive control (exactly as §2.2 predeclares it)**: apply the identical
  changepoint estimator to **`hc_year_one_fade`**, a construct with an
  already-known large break (**reported**, unverified here:
  `docs/era_magnitude_profile.md:84` records +0.09 pts 2009-2017 vs −8.08
  pts 2018-2025). If the estimator locates that break at or near where it is
  already known to sit, the machinery is validated before its
  `pt_post_mnf_sunday` answer is trusted; if it does not, the cell's own
  break location is uninterpretable and that is reported as such.
- **Per-era magnitudes**: both the data-located segments' means AND the
  registry's already-recorded fixed-convention arms (+0.255 pts 2009-2017,
  −0.137 pts 2018-2025, read from `docs/era_magnitude_report.md:109-110`)
  are reported side by side. Never averaged.
- **Recording rule**: name `pt_post_mnf_sunday_changepoint`, league `nfl`,
  seasons 2009-2025, units `accuracy_points`, family `primetime_cells`
  (**same rows as `pt_post_mnf_sunday` — correlated, never an independent
  vote**), category `schedule`, classification `unresolved_below_power`.
  Two-sided by construction, so `wrong_sign_resolved` is inadmissible;
  `positive_control_bound` would require the `hc_year_one_fade` control to
  demonstrate the estimator resolving a break of the size claimed here while
  this construct shows none.

### 2.3 `sagarin_battery_large_divergence_coverage_matched_era` (+ its predeclared late-era companion)

**Script extended:** `scripts/sagarin_divergence_battery.py`
(`build_close_population`, `LARGE_DIVERGENCE_THRESHOLD = 3.0`, `ERA_SPLITS`,
`DEFAULT_SAGARIN_SNAPSHOT`), imported.

**This is a NEW look at an already-scored construct, and is declared as its
own family, not a re-score.** `docs/sagarin_backfill.md` §9.4 (read) is
explicit that the seven frozen `sagarin_battery_*` entries were **not**
re-scored on the corrected 2012/2013 coverage and that re-running the frozen
battery against the fixed snapshot would be a new look at the same outcome
data. Accordingly:

- The cells below are recorded under the **new family
  `sagarin_divergence_coverage_matched`**.
- Their **population differs** from the frozen entries': it is
  coverage-matched (per-season usable-Sagarin coverage >= 80%), not the full
  era.
- **No existing `sagarin_battery_*` registry entry is overwritten,
  corrected, replaced, or reclassified by this work package.** No
  `--replace` is used against any of them.
- The corrected-coverage **unmatched** full-era reads are computed and
  printed as run diagnostics only and are **deliberately not recorded**:
  those would be the frozen entries' own identities re-measured, which §9.4
  says needs its own predeclaration and rotation window.

- **Coverage-matching rule (frozen; the RULE is frozen, the season list is
  its consequence)**: keep only seasons whose own within-season usable-Sagarin
  coverage in the screen population is **>= 80%**. `docs/era_magnitude_report.md`
  §2.3 wrote that rule against the pre-fix coverage table (excluding 2012 at
  0.0% and 2013 at 31.2%). WP19's fix (read, `docs/sagarin_backfill.md` §9.3)
  measured 2012 at **47.3%** and 2013 at **85.5%** on close-grade coverage
  rising 2,966 → 3,229 games, so applying the *same* >=80% rule to the
  *corrected* coverage is what this cell runs, and the resulting season list
  is reported in section 4 rather than hardcoded. Coverage is recomputed
  live from the joined population so the filter and the population can never
  disagree.
- **Population (primary cell)**: `|divergence_close| >= 3.0`, close grade,
  seasons **2010-2016** (`ERA_SPLITS[0]`), restricted to that era's
  >=80%-coverage seasons.
- **Population (predeclared companion arm)**: the identical construction on
  seasons **2017-2025** (`ERA_SPLITS[1]`), restricted to that era's
  >=80%-coverage seasons — declared now, before any sign is seen, because
  "era magnitude, not presence" requires the matched comparator arm to be
  reported as its own magnitude rather than inferred from the unmatched one.
  Name: `sagarin_battery_large_divergence_coverage_matched_era_late`.
- **Direction**: **none predeclared** for either arm. This is an
  instrument-composition diagnostic. Report whether the coverage-matched
  subsets' point estimates and intervals materially move from the
  currently-recorded full-window reads (+1.807 [−2.996, +6.686] for
  2010-2016 and −2.299 [−6.052, +1.506] for 2017-2025, read from
  `docs/era_magnitude_report.md:122-123`); if materially unchanged, the
  coverage-composition explanation for the era heterogeneity is disfavoured;
  if they move substantially, coverage composition is implicated.
- **Statistic**: the battery's own single-group metric,
  `(mean(sagarin_side_cover) − 0.5) × 100` accuracy points.
- **Grade**: close. **Effect units**: `accuracy_points`.
- **Uncertainty**: section 1's table. The season-blocked secondary will have
  few blocks and is expected to be flagged `DEGENERATE`; it is reported
  anyway, labelled.
- **Within-week permutation null**: `sagarin_side_home` (which side Sagarin
  favours) is permuted **within** each (season, week) block and the cover
  rate recomputed — a real null for "does Sagarin's side choice carry
  information", since permuting the outcome alone would leave the mean
  unchanged.
- **Positive control (two)**:
  1. **Already in hand, per §2.3** (**reported**, read from
     `docs/sagarin_backfill.md` §9, not re-run here): the Era-B
     coverage-completeness fix already moved this exact cell's point
     estimate +2.926 → +1.807 as 2010/2011 coverage rose — a measured
     instrument-*sensitivity* precedent. Stated plainly: sensitivity is not
     detectability, so it is reported as context, not as a bound.
  2. **Injection**, per section 1's design, targets **{0.5, 1.0, 2.0, 4.0}**
     accuracy points, run on the coverage-matched early-era population — the
     detectability statement this cell actually needs.
- **Per-era magnitudes**: both coverage-matched arms reported separately,
  alongside both unmatched corrected-coverage diagnostics and both
  registry-frozen pre-fix values. Four pairs, no average across the flip.
- **Recording rule**: both names above, league `nfl`, units
  `accuracy_points`, family `sagarin_divergence_coverage_matched`, category
  `market`, classification `unresolved_below_power`. Two-sided, so
  `wrong_sign_resolved` is inadmissible for either arm.

---

## 3. Run interface, files, and what would close anything

### 3.1 Command

```powershell
.\.tools\uv.exe run --no-sync python scripts/era_mechanism_screens.py `
    --cell <bye_overval_install_need_moderator|pt_post_mnf_sunday_changepoint|sagarin_battery_large_divergence_coverage_matched_era> `
    --mode <screen|null|positive-control>
```

`--mode screen` is the predeclared primary read; `--mode null` runs only the
200-draw within-week permutation null; `--mode positive-control` runs only
the controls of section 2. `--cell all` runs every cell. Each invocation
writes one `artifacts/era_mechanism_screens/<UTC timestamp>/results.json`
through `write_experiment_artifact`.

### 3.2 Files this work package owns

`docs/era_mechanism_screens_20260901.md` (this file),
`scripts/era_mechanism_screens.py`, `tests/test_era_mechanism_screens.py`,
`artifacts/era_mechanism_screens/`.

### 3.3 Tests (frozen requirement list)

1. The imported flag builders reproduce the parent batteries' flags on a
   fixture (bye map, post-MNF-Sunday, large-divergence).
2. The changepoint machinery gives the known answer on a synthetic series
   with a planted break.
3. The coverage-matched population construction keeps exactly the seasons
   at or above the threshold and drops the rest.
4. Leakage: the install-need moderator uses only prior/first-game facts —
   no row's own game feeds its own moderator value.

### 3.4 What could close a cell here — stated before the numbers

- Cell 1 (one-sided +): `refuted_mechanism` / `wrong_sign_resolved` **only**
  if the entire week-blocked 95% interval lies below zero.
- Cells 2 and 3: two-sided by construction — `wrong_sign_resolved` is
  **inadmissible** for them under any result.
- Any cell: `bounded_by_control` **only** if its own positive control is
  shown to detect an effect of the size claimed while the cell itself is
  flat. An injection control that fails to detect at the relevant size
  proves the opposite — that the instrument cannot bound anything here.
- Everything else, including every interval that contains zero, is
  `unresolved_below_power` and gets recorded as such.

---

## 4. Results (measured; produced after everything above was frozen)

Commands (**measured**, 2026-09-01):

```powershell
.\.tools\uv.exe run --no-sync python scripts/era_mechanism_screens.py --cell all --mode screen `
    --output artifacts/era_mechanism_screens/20260901T195546Z_screen
.\.tools\uv.exe run --no-sync python scripts/era_mechanism_screens.py --cell all --mode null `
    --output artifacts/era_mechanism_screens/20260901T195940Z_null
```

Artifacts: `artifacts/era_mechanism_screens/20260901T195546Z_screen/results.json`,
`.../20260901T195940Z_null/results.json`,
`.../20260901T193000Z_positive_control_sagarin/results.json`.

### 4.1 `bye_overval_install_need_moderator`

**What this implies for the DECISION, first.** The mechanism's own prediction
is confirmed in direction at `probability_positive` **0.6512**: install-need
home teams off a bye lose **less** against the spread than continuity teams
do. Both moderator arms are still negative — the post-CBA fade the parent
battery found is present in each — but the install-need arm is only about
half as negative. On EV that is a 65/35 read in favour of the practice-cap
mechanism being the right story for the parent construct's era flip, and it
says the fade should be applied more heavily to continuity teams than to
teams with a new quarterback. It is far too imprecise to change a card, and
nothing here proposes one.

Measured, week-blocked primary, 20,000 draws, seed 20260821, 243 week blocks,
3,573 games (0 rows lost to an unknown moderator):

| quantity | effect (pts) | week-blocked 95% | P+ |
|---|---:|---|---:|
| install-need arm | −0.2289 | [−1.0009, +0.5262] | 0.2777 |
| no-need arm | −0.4143 | [−0.9374, +0.1213] | 0.0649 |
| **contrast (recorded)** | **+0.1855** | **[−0.7664, +1.1250]** | **0.6512** |

Season-blocked secondary (informational, 14 blocks, above the floor):
contrast +0.1855 [−1.0284, +1.3405], P+ 0.6298 — the same read.

Per-era magnitudes, four numbers, never averaged:

| era | install-need arm | no-need arm | n flagged |
|---|---:|---:|---:|
| 2009-2011 (pre-CBA control) | −0.3404 | +0.8915 | 18 / 17 |
| 2012-2025 (post-CBA) | −0.2289 | −0.4143 | 87 / 151 |

Within-week permutation null, 200 draws (**not** zero-centred by design):
null mean **+0.1235**, [−0.7002, +1.0221]; **44%** of null draws sit at or
above the observed +0.1855. Read plainly: the shuffled label reproduces this
contrast about as often as not, so the permutation adds no support beyond the
bootstrap's own 0.65.

What is wrong with it, after: the OC leg of the moderator does not exist in
this repository (**measured**), so the subset is narrower than predeclared
and the contrast is diluted toward zero; only 87 flagged install-need games
carry the arm; the rows are the same rows `bye_overval_home_edge_post2011`
already scored, so this is not an independent vote; and the pre-CBA control's
+0.8915 no-need arm rests on 17 games.

**Recorded** `unresolved_below_power` — the interval does not sit entirely
below zero, so `wrong_sign_resolved` is inadmissible against the frozen
one-sided positive direction, and no control has been shown able to detect an
effect this size.

### 4.2 `pt_post_mnf_sunday_changepoint`

**What this implies for the DECISION, first.** The battery's fixed 2017/2018
boundary is **not** in the wrong place: the free search lands on 2019 and the
two segmentations agree to within 0.05 accuracy points (free break −0.4424,
fixed boundary −0.3886). So the recorded era arms can be read as they stand,
and no re-slicing of the primetime battery is warranted. Separately, the
break itself is not distinguishable from what shuffling produces, so the
construct's "era instability" should keep being described as the ordinary
mined-battery pattern `docs/era_magnitude_report.md` §2.2 inferred, not as an
environmental shift.

Measured: 8,090 eligible team-games, 586 flagged, 277 week blocks, seed
20260821. Per-season effect series (accuracy points, battery `sign = −1`,
full-slate scaled): 2009 +0.93, 2010 −0.23, 2011 −0.35, 2012 −0.00, 2013
−0.00, 2014 +1.27, 2015 −0.23, 2016 −0.00, 2017 +0.91, 2018 +0.25, 2019
−0.47, 2020 +0.34, 2021 −1.17, 2022 +0.44, 2023 −0.34, 2024 −0.65, 2025
+0.53.

| quantity | value | week-blocked 95% | P+ |
|---|---:|---|---:|
| located break season | 2019 | bootstrap [2012, 2023], median 2019 | — |
| pre-break mean (2009-2018) | +0.2550 | [−0.1324, +0.6470] | 0.8997 |
| post-break mean (2019-2025) | −0.1874 | [−0.6335, +0.2624] | 0.2084 |
| **break magnitude (recorded)** | **−0.4424** | **[−1.0357, +0.1572]** | **0.0723** |
| fixed 2017/2018 gap | −0.3886 | [−0.9786, +0.2046] | 0.0969 |

Within-week permutation null, 200 draws: the free-break magnitude's null is
centred near zero (−0.0157) with spread [−1.0283, +1.0267], and **65.5%** of
null draws produce a break at least as large in absolute value as the
observed −0.4424. An optimal changepoint estimator always finds *some* break;
on this construct it manufactures one this big from a shuffled label most of
the time. The fixed-convention gap fares better against the same null (21.5%).

**Positive control (partially run — see §5).** The `hc_year_one_fade`
known-answer check ran at reduced bootstrap settings only
(`--samples 200`, scratchpad): the identical estimator locates that
construct's break at **2019** with pre-break mean −0.045 and post-break mean
+2.395 on 3,760 rows — i.e. it recovers a large break within one season of
the boundary `docs/era_magnitude_profile.md:84` reports (**reported**,
unverified: +0.09 pts 2009-2017 vs −8.08 pts 2018-2025; the magnitudes differ
because that figure is on a different scaling convention — what this control
tests is the *location*). The machinery is validated for locating a large
break; it has not been shown able to resolve a break of ~0.44 points, which
is exactly why `positive_control_bound` is inadmissible here.

Also disclosed: the **season-blocked secondary is structurally degenerate for
this metric** and carries no information — a season-block resample of a
per-season-series statistic reproduces the same series in every draw, so the
interval collapses to a point. Reported in the artifact for completeness,
never read as an interval.

**Recorded** `unresolved_below_power`; two-sided by construction, so
`wrong_sign_resolved` is inadmissible under any result.

### 4.3 `sagarin_battery_large_divergence_coverage_matched_era` (+ late companion)

**What this implies for the DECISION, first.** Coverage composition does
**not** explain the Sagarin era heterogeneity — it *widens* it. Matching both
eras on >=80% within-season coverage moves the early era from +1.4548 to
**+2.1515** and the late era from −2.2989 to **−3.5417**, so the era gap goes
from 3.75 points (unmatched) to **5.69 points** (matched). The candidate
mechanism `docs/era_magnitude_report.md` §2.3 proposed is therefore
disfavoured by its own test, and `docs/pool_edge_plan.md`'s standing read —
that pooling these two eras would misrepresent two things that are not
measuring the same quantity — survives on the corrected data. Nothing here
touches the pool-relevant opener population, which WP19 left unchanged.

Coverage recomputed live from the joined population (**measured**, and
reproducing `docs/sagarin_backfill.md` §9.3's corrected table season for
season): 2010 91.8, 2011 89.5, **2012 47.3**, 2013 85.5, 2014 85.5, 2015
90.2, 2016 91.8, 2017 96.9, **2018 78.1**, **2019 71.9**, 2020 100.0, **2021
27.9**, **2022 38.4**, 2023 83.8, 2024 81.2, 2025 82.0. Close population
3,229 games; `|divergence| >= 3.0` subset 1,349 games. The >=80% rule keeps
2010/2011/2013/2014/2015/2016 early (drops 2012) and 2017/2020/2023/2024/2025
late (drops 2018/2019/2021/2022).

| arm | n | seasons | effect (pts) | week-blocked 95% | P+ | recorded |
|---|---:|---|---:|---|---:|---|
| early, coverage-matched | 581 | 2010-16 less 2012 | **+2.1515** | [−2.1960, +6.3964] | 0.8289 | yes |
| early, unmatched (diagnostic) | 653 | 2010-2016 | +1.4548 | [−2.4960, +5.3825] | 0.7614 | **no** |
| late, coverage-matched | 480 | 5 seasons | **−3.5417** | [−8.2782, +1.1408] | 0.0660 | yes |
| late, unmatched (diagnostic) | 696 | 2017-2025 | −2.2989 | [−6.1010, +1.5153] | 0.1130 | **no** |

The unmatched late arm reproduces the frozen registry entry
`sagarin_battery_large_divergence_era_2017_2025` (−2.2989, n=696) to four
decimals — this run's own cross-check that the pipeline reproduces the parent
battery, since WP19 changed no season after 2013.

Season-blocked secondaries (6, 7, 5 and 9 blocks) are all **DEGENERATE below
the 10-block floor** and informational only: early matched +2.1515 [+0.1701,
+4.9815] P+ 0.9810; late matched −3.5417 [−7.1429, +0.4836] P+ 0.0365.

Within-week permutation null on the Sagarin **side assignment**, 200 draws
(not zero-centred): early null mean +0.6695 [−3.0120, +4.9053], 34% of draws
at or above |observed|; late null mean −1.2208 [−4.7917, +2.3021], 14.5% at
or above |observed|.

**Injection positive control** (frozen design: 25 replicates × 2,000 draws
per target, on the coverage-matched early population;
`artifacts/era_mechanism_screens/20260901T193000Z_positive_control_sagarin/results.json`):
detection rate (95% interval excludes zero) **0.00 at +0.5 pts, 0.00 at +1.0,
0.44 at +2.0, 1.00 at +4.0**; achievable injection ceiling 47.8 points. So
this cell resolves effects only around 4 accuracy points and cannot bound
anything at or below 1 point — which is precisely why no result here can be
`bounded_by_control`. (For a single-group cover-rate metric the recovered
point estimate is invariant to *which* zero rows are flipped, by
construction; only the interval varies across replicates.)

**Both arms recorded** `unresolved_below_power` under the new family
`sagarin_divergence_coverage_matched`. **No `sagarin_battery_*` entry was
read back, overwritten, corrected or reclassified**, and no `--replace` was
used against any of them.

### 4.4 Registry (measured)

Four names recorded, each under this session's cross-process registry lock;
registry total 675 → 679:

- `bye_overval_install_need_moderator` (family `bye_overvaluation`)
- `pt_post_mnf_sunday_changepoint` (family `primetime_cells`)
- `sagarin_battery_large_divergence_coverage_matched_era` and
  `..._era_late` (family `sagarin_divergence_coverage_matched`)

All four `unresolved_below_power`, no closing ground named, every one
reported with `probability_positive`.

## 5. What is NOT finished (stated plainly)

- **Cell 1's two positive controls did not run at frozen settings.** The
  external `travel_rest_home_off_bye` replication and the injection control
  are implemented and smoke-tested but were not executed before the session
  budget closed. Command:
  `python scripts/era_mechanism_screens.py --cell bye_overval_install_need_moderator --mode positive-control`
  (~15-20 minutes). Until it runs, nothing about cell 1 can be
  `bounded_by_control` — which does not change its recorded classification,
  since `unresolved_below_power` is the outcome either way.
- **Cell 2's positive control ran only at `--samples 200`** (scratchpad),
  enough to establish the break location but not to quote an interval. Re-run
  with `--mode positive-control` at the frozen 20,000 draws to publish one.
- **No rotation-registry look was declared.** `docs/sagarin_backfill.md` §9.4
  asks for one before the frozen `sagarin_battery_*` identities are
  re-measured; this work package deliberately did not re-measure them, so no
  look was spent. Whether `sagarin_divergence_coverage_matched` should carry
  its own declared rotation family is an open owner decision.
- **No split-half reliability** was computed for any of the four entries
  (their parent battery entries carry none either). A cell without a recorded
  reliability can never be closed on `no_split_half_reliability`.
