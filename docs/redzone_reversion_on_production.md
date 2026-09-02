# Third-down mean-reversion fade, stacked on PRODUCTION: predeclaration

Written **before any ATS number is produced by this comparison**, per the same
rule that governs `docs/illness_on_production.md`,
`docs/graph_team_stat_def_ypp_on_production.md` and
`docs/fluview_on_production.md`. **Sections 1-6 are the predeclaration** and
contain no accuracy, cover-rate or `probability_positive` number against NFL
outcomes from this comparison. **Section 7 was added after the look** and
reports what it found; it changes nothing above it.

One arm, mirroring the single-arm shape of
`docs/graph_team_stat_def_ypp_on_production.md`.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) **refuted mechanism** — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) **bounded by a
positive control** — the instrument was PROVEN able to detect an effect that
size and it was absent. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible closures;
if a record command errors, the verdict is wrong, not the validator. Decisions
are expected value: `probability_positive` above 0.5 favours the candidate;
predeclared thresholds govern only what docs may CLAIM. Grade play/no-play
decisions at the OPENER; a close-graded look settles no play decision and is
recorded `unresolved_below_power` regardless of sign. Never say something
"needs N more games". Within-week game correlation is ZERO by owner mandate.

## 1. What this closes, and the quality-proxy risk stated honestly first

`docs/redzone_reversion_screen.md` predeclared and froze six cells of a
red-zone / third-down mean-reversion battery and scored every one of them
against a **bare market baseline** — a subset cover-rate gap scaled to the
full slate. **Read**, `registry/weak_signals.json`:

| cell | effect (accuracy points) | week-blocked 95% | `probability_positive` | reliability |
|---|---|---|---|---|
| `redzone_reversion_c3_third_down_over_fade` | **+0.3665** | [-0.2587, +0.9990] | **0.8719** | **0.407** |
| `redzone_reversion_c5_hot_offense_vs_stingy_defense` | +0.1610 | [-0.2071, +0.5216] | 0.8045 | 0.201 |
| `redzone_reversion_c1_rz_over_fade` | +0.0302 | [-0.5434, +0.5958] | 0.5375 | 0.141 |
| `redzone_reversion_c2_rz_under_rebound` | -0.1062 | [-0.6985, +0.4953] | 0.3613 | 0.141 |
| `redzone_reversion_c4_third_down_under_rebound` | -0.3564 | [-0.8928, +0.1806] | 0.0989 | 0.407 |

All six are recorded `unresolved_below_power` with `closing_ground: null`, over
`sample_games` 8,634 team-games (4,317 games), `sample_blocks` 294, seasons
2009-2025, `family: null` (**read**, the same registry file). **C3 is the arm
this document tests**: it carries both the highest lean in the battery
(P+ 0.8719) and the highest split-half reliability (0.407, the year-over-year
Pearson of the centered trait, 95% [+0.337, +0.473] on 512 team-season pairs —
**read**, `docs/redzone_reversion_screen.md`, reliability table). AGENTS.md
makes reliability the decisive field: an unreliable trait is refuted because no
sample size rescues it, so the battery's most reliable input is the one worth
spending a window on. The red-zone cells' 0.141 is the battery's thin end and
is not tested here.

**The honest risk, before any result.** The project's own recorded build filter
`team-quality-is-already-priced` says features that only measure team quality
better are bounded near zero, and prior-season third-down conversion rate is
quality-adjacent: good teams convert third downs. If this column is nothing but
a lagged, noisier restatement of team strength, the production chain — which
already carries fifteen rolling offensive/defensive EPA, yards-per-play,
sack-rate, turnover-rate and point-differential states plus Elo — should
already price it, and the marginal should land near zero. That is the
prediction this document is prepared to be wrong about, stated up front rather
than discovered afterwards.

**Three reasons to run it anyway, stated before the outcome:**

1. **The construct is mean REVERSION, not measurement.** It does not say "this
   team is good at third down"; it says "fade last season's over-performer",
   i.e. it bets *against* the persistence of a measured quality. A better
   quality proxy and a fade of a quality extreme are different quantities with
   opposite signs on the same input.
2. **The mirror cell lands in the opposite tail.** `c4_third_down_under_rebound`
   — the same trait, bottom quartile, predicted POSITIVE — sits at P+ 0.0989
   (**read**, `registry/weak_signals.json`). A trait that were purely a quality
   proxy would push both tails the same way once signs are aligned; a reversion
   mechanism predicts exactly this antisymmetry. This is not proof (c3 and c4
   share a trait and are explicitly recorded as "not independent" in c3's own
   registry note), but it is the shape the mechanism predicts and the shape a
   pure quality proxy does not.
3. **Production carries no third-down conversion rate in any form.**
   **Measured** this session:
   `FEATURE_SETS["full_weak_stack"]` (the production feature set,
   `src/nfl_ats/constants.py`) holds **90** columns, and **zero** of them
   contain `third`, `conv`, `redzone` or `rz_` in the name — no third-down
   conversion rate, current-season or prior-season, and no red-zone rate at
   all. Its efficiency states are EPA-per-play (overall/pass/rush), CPOE,
   yards-per-play, turnover rate, sack rate, takeaway rate, point differential
   and ATS residual, all as rolling *current-season* states, plus Elo, QB,
   injury, lineup-continuity and bias families.
   **Correction to the orchestrator's briefing, measured:** that briefing named
   `diff_drive_scoring_rate` and `diff_pbp_off_success_rate` as members of
   `full_weak_stack`. They are **not** — `diff_drive_scoring_rate` appears only
   in the `football_drive` / `full_drive` sets, and neither name is in
   `full_weak_stack`. The load-bearing half of the claim survives and is
   stronger than stated: production has no drive-conversion or success-rate
   state either.

**What this document asks, and it is the marginal question that decides:** does
the C3 fade indicator add anything on top of the full PRODUCTION chain
(**read**, `artifacts/active_ats_model.json`: `feature_profile: "weak_stack"`,
`method: "market_residual"`, `regressor: "ridge"`, `ridge_alpha: 10.0`)? The
project's recorded lesson "composition is not the signal" is that a component
positive alone can go negative once stacked on the chain that is actually
PLAYED, because the played chain already explains some of the variance a
bare-baseline comparison credits to the candidate. The reversion channel has
never been stacked on production: the battery's only look was the 2026-08-21
bare-baseline screen, and its registry rows carry `family: null` and no
rotation family at all (**verified** below, section 5).

**This sweep is not a foregone negative.** Sibling construct #1 in the same
2026-09-01 on-production sweep came back POSITIVE: `illness_away_active_ge1`
scored **+0.804 accuracy points at week-blocked P+ 0.908** on [2011, 2013]
(**reported by the orchestrator, unverified by me**; artifact cited as
`artifacts/illness_on_production/20260901T192918Z/results.json`), against the
two graph constructs at -0.935 (P+ 0.122) and -0.668 (P+ 0.189). Two of four
on-production stackings so far survived. This is a real question with a live
answer either way.

## 2. The candidate column, and both deviations declared

**One column: `redzone_third_down_over_fade_diff`**, taking values in
{-1, 0, +1}:

```
redzone_third_down_over_fade_diff = int(home team flagged) - int(away team flagged)
```

A team is **flagged** when its **prior-season league-centred third-down
conversion rate** (`third_down_conv_rate_centered` from
`build_efficiency_panels`) is **at or above** the top-quartile threshold for
that game's season. NaN where either team has no prior-season panel row, or
where the threshold itself does not exist.

`src/nfl_ats/redzone_reversion_production_feature.py` **imports the frozen
panel construction rather than reimplementing it**: `build_efficiency_panels`,
`OFF_TRAITS`, `_alias_team` and `_prior` come straight from
`scripts/redzone_reversion_screen.py` (read-only, never edited), which itself
reads the local play-by-play snapshot through
`nfl_ats.pbp.latest_pbp_snapshot` / `load_pbp_snapshot` / `analysis_plays` —
the house v1 efficiency filter, REG plays only, franchise aliases applied via
`nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`. `third_down_conv_rate` is
third-down plays converted (`first_down == 1`) over third-down plays, centred
against its own season's unweighted league mean. Team codes on the feature
table are canonicalised with the same `TEAM_ABBREVIATION_ALIASES` map, matching
the screen.

`scripts` is not part of the installed package, so this module puts the
repository root on `sys.path` the same guarded way
`nfl_ats.fluview_production_feature` and `nfl_ats.illness_production_feature`
already do for the same reason.

### Deviation 1 — team-level cell to game-level column

The registered cell is a **team-game** flag scored on a long table of 8,634
team-games (one row per team per game, graded on `team_covered`). A feature
column on the game-level production table must be **game-level**, so the two
team flags combine into one **signed difference**.

This **preserves the construct's direction**. C3's sign convention is -1: a
flagged team is predicted to UNDER-cover. So a flagged HOME team is a reason to
lean away from the home side, and a flagged AWAY team is a reason to lean
toward it. Encoding `home_flag - away_flag` puts a home over-performer to fade
at +1 and an away over-performer to fade at -1; the ridge fit is free to learn
whichever sign the data supports, and the *ordering* of the three states is the
construct's own ordering. Both-flagged and neither-flagged collapse to 0, which
is correct for a fade-the-extreme construct: when both sides are extreme in the
same direction the reversion pressures offset.

It also keeps the profile at **exactly one new column**, the shape every
candidate profile in this family uses (`weak_stack_graph_sack`,
`weak_stack_graph_def_ypp`, `weak_stack_fluview_home`/`_away`,
`weak_stack_illness_home`/`_away`).

**Cost of the deviation, disclosed:** a signed difference cannot distinguish
"neither team flagged" from "both teams flagged". The battery's own
`n_flag`/`n_total` implies roughly a quarter of team-games are flagged, so
both-flagged is the rarer of the two collapsed states, but this is a real loss
of information relative to two separate indicator columns and it can only
attenuate toward the null, never inflate away from it.

### Deviation 2 — expanding, strictly-prior threshold

The original screen computes its `third_down_q75` cut over the **WHOLE
2009-2025 panel** — **read**, `scripts/redzone_reversion_screen.py` line 383:
`"third_down_q75": float(offense["third_down_conv_rate_centered"].quantile(0.75))`,
taken on the full pooled panel before any cell is scored. That is a mild
look-ahead: a 2012 game is flagged against a cut estimated partly from 2020
data. It was defensible in a whole-panel descriptive screen; **a pregame
feature column may not carry it.**

This column's threshold is therefore recomputed **expanding over strictly prior
seasons only**: for a game in season S, the cut is the 75th percentile of
`third_down_conv_rate_centered` across **every team-season strictly before S**.
Games in the panel's first season carry no threshold and no prior-season value,
so the column is NaN there.

**Consequences, declared before scoring:**

- This makes the column a **slightly different quantity** from the registered
  cell. It is not a reproduction of C3; it is C3's mechanism rebuilt to a
  pregame standard, and the two are not expected to agree game-for-game.
- The early-window threshold is estimated from few team-seasons (32 per prior
  season), so it is **noisier** than the pooled cut. A noisier threshold
  misclassifies teams near the boundary in both directions, which **can only
  attenuate the measured effect toward the null**, never manufacture one away
  from it. If this column reads positive, the pooled-cut version would read at
  least as positive; if it reads flat, part of that flatness is threshold noise
  and the reading is correspondingly less informative — which is exactly what
  `unresolved_below_power` means.
- The leakage regression test **pins** both halves: a synthetic panel proves a
  season-S value uses only season < S conversion rates AND a threshold
  estimated from seasons < S, and that injecting an extreme value into season S
  or any later season changes no season-S value.

The column is additively joined back onto the production feature table by
`game_id` with `validate="one_to_one"` — the same additive-merge discipline
`nfl_ats.forecast_weather_features.attach_forecast_weather_features` and every
sibling candidate module already established: every pre-existing column comes
back bit-identical, only the one new column is added. Unmatched games are left
NaN on purpose; imputation belongs to the model's own training-fold median
(`fit_margin_model`), never to a feature builder that can see every season at
once.

## 3. The candidate profile: `weak_stack_redzone_third_down`

A `MarginFeatureProfile` (already wired in `src/nfl_ats/margin.py`) =
production `weak_stack`'s exact feature set plus **exactly one** new column.

**Measured** this session:
`margin_feature_set("market_residual", "weak_stack")` resolves to
`full_weak_stack` (**90** columns) and
`margin_feature_set("market_residual", "weak_stack_redzone_third_down")`
resolves to `full_weak_stack_redzone_third_down` (**91** columns); the set
difference is exactly `["redzone_third_down_over_fade_diff"]` added and
**nothing removed**.

Built on `data/processed/game_features_weak_stack.parquet` — the PRODUCTION
table — **directly**, never on
`weak_stack_v3`/`_surface`/`_v4`/`_graph_*`/`_fluview`/`_illness`, mirroring
`weak_stack_graph_sack`'s own declared reason verbatim: stacking a candidate
onto a profile already refused or still undecided would confound the answer to
"does this add to what is actually played." The widened table is written to
`data/processed/game_features_weak_stack_redzone_third_down.parquet`; the
production table is never touched. Never referenced by the active model. Never
mixed with any other candidate profile.

## 4. The comparison

**Two arms, one evaluator, one window:**

| arm | feature_profile | role |
|---|---|---|
| baseline | `weak_stack` (production, unmodified) | what is actually played |
| candidate | `weak_stack_redzone_third_down` | production + the one fade column |

Both arms hold `regressor="ridge"`, `ridge_alpha=10.0`,
`target="market_residual"` fixed at the active model's own values
(**read**, `artifacts/active_ats_model.json`); only `feature_profile` differs,
isolating the column's marginal contribution against everything the production
chain already explains. Both are fit with `nfl_ats.margin.fit_margin_model` —
the same estimator production itself uses, not a single-feature model — which
is the whole point of "on top of production" rather than "on top of a bare
baseline". Both are fit and scored on the **same games in the same weeks**,
forward-chaining only: each week is predicted from a model trained strictly on
games that kicked off before that week's earliest kickoff.

The primary quantity is the **paired candidate-minus-baseline** forced-pick
accuracy delta, in `accuracy_points` (percentage points), `pick_correct`
against `home_cover_probability >= 0.5` (the same probability rule production
plays), per `nfl_ats.clv.pick_correct`.

## 5. Grade, window, and why a new rotation family

**Grade.** Close-graded, mirroring both sibling on-production documents. Per
the binding "grade the decision at the opener" rule, nothing here may settle a
play/no-play or promotion call; the recorded classification is
`unresolved_below_power` regardless of sign.

**Window and family.** A new rotation family, `redzone_reversion_on_production`,
close-graded, declared with **no `--inherits`**: the red-zone / third-down
reversion battery has never held a rotation family at all. **Verified** this
session by reading `registry/rotation_registry.json` — the 17 declared families
are `best_pick_ranker`, `best_pick_ranker_opener`, `cfb_role_continuity`,
`combined_stacker`, `era_weighting_half_life_8`,
`fluview_elevated_on_production`, `fluview_home_elevated_opener`,
`graph_def_ypp_on_production`, `graph_off_rush_epa_on_production`,
`graph_off_sack_rate_on_production`, `graph_ratings_v2_team_stat`,
`illness_on_production`, `mod07_weak_signal_stack`, `movement_expansion_v1`,
`pbp_drive_bundle`, `player_qb_continuity` — and none of them is a red-zone or
reversion family. The battery's own registry rows likewise carry
`family: null`. There is nothing to inherit.

Declared **without** `--acknowledge-mined`, because the deterministic
earliest-eligible close block is not expected to intersect the 2018-2025 mining
ledger; if the CLI refuses, that refusal is the authority and section 7 records
what actually happened.

The window is **ASSIGNED by `nfl-ats rotation assign`**, never hand-picked
(`src/nfl_ats/rotation.py::assign_window`: the lowest-starting block of the
requested size inside the grade's pool that starts at or after the warm-up
floor — there is no hidden choice and nothing to tune). The assigned block is
confirmed in section 7, not asserted here.

## 6. Uncertainty, instrument checks, and the power caveat stated up front

**Uncertainty.** `nfl_ats.clv.week_blocked_bootstrap`, **week-blocked as the
primary** reference (within-week game correlation is zero by owner mandate) and
**season-blocked reported beside it, never averaged with it**. Same
`BOOTSTRAP_SAMPLES=1000` / `SEED=20260826` constants the sibling on-production
scripts already use, for comparability.

**Within-week permutation null**, **200 permutations**, identical mechanism to
the sibling documents: both arms' models are fit ONCE per week on the REAL
`ats_margin`; only the grading margin is shuffled within week for the null, so
200 draws cost no extra model fits. This null is **not** centred on zero by
design — it preserves each week's realized home-cover rate, and the two arms
may carry different home-pick rates — and is reported ALONGSIDE the
bootstrap-vs-zero interval, never instead of it.

**Positive control, run BEFORE the screen.** The candidate profile's one new
column is temporarily REPLACED by the realized `ats_margin`, a deliberate large
leak, so the harness must show an obvious large effect. This proves the
FULL-PROFILE ridge fit can detect a real effect of meaningful size when one is
actually present, even with the candidate column embedded in 90 other
production features. A "no effect" reading from a blind instrument would mean
nothing; this check exists so that possibility is ruled out first. **If the
positive control does not show a large obvious effect, the run STOPS and the
screen is not read at all.** (The illness sibling's control came back +50.13
points at P+ 1.000 — reported by the orchestrator, unverified by me.)

**Power caveat, stated before any result.** This is a **three-valued** column,
and a three-valued column carries less information than a continuous one: it
can only shift the fit by a constant per state, so the most it can do is nudge
games near the 0.5 probability boundary across it. By construction roughly a
quarter of teams are flagged in any season (a top-quartile cut), so on
independent sides the signed difference would be non-zero on roughly
2 x 0.25 x 0.75 = **37.5%** of games (about 3/8) and zero on the rest; the
column is therefore silent on the majority of the slate. The exact measured
figure is filled in below once the table is built (a feature-table property,
not an outcome), and the in-window distribution is reported in section 7.

Per the binding taxonomy, a wide interval from a coarse column is
`unresolved_below_power`, never a negative, and never "needs more games".

**Measured coverage and value distribution** (built table
`data/processed/game_features_weak_stack_redzone_third_down.parquet`, filled in
after the build, before any ATS scoring):

- **4,902 rows**, seasons 2009-2026. `redzone_third_down_over_fade_diff` is
  non-missing on **94.553%** of rows (4,635 games). The missing 267 are
  **exactly and only the 2009 games** — the panel's first season, which has no
  strictly-prior season and therefore neither a prior-season value nor a
  threshold. Coverage is **100.0% for every season 2010 through 2026**
  (measured per season). The offensive panel itself is 544 team-seasons
  (32 x 17, 2009-2025).
- **Value distribution over covered games:** **-1 on 17.605%** (816 games),
  **0 on 64.078%** (2,970), **+1 on 18.317%** (849). Non-zero on **35.922%**
  of covered games, against the ~37.5% the construction predicts — the
  shortfall is the both-flagged state, which is more common than independence
  would imply because the flags are positively correlated across a slate.
- **Per-side flag rate:** home flagged on **24.06%** of covered games, away on
  **23.34%**; both flagged on **5.74%**, neither on **58.34%**. Both sides sit
  just under the 25% a top-quartile cut implies, which is the reproduction
  check the expanding threshold has to pass — it is slightly under rather than
  exactly at 25% because the strictly-prior cut is estimated on a different
  cohort than the season being flagged.
- **Threshold path** (measured, expanding, strictly prior): 0.0490 for 2010
  (estimated on 2009's 32 team-seasons alone), 0.0403 for 2011, 0.0348 for
  2012, 0.0396 for 2013, 0.0382 for 2014, settling to 0.0349 by 2025. The
  early-window instability that deviation 2 disclosed is visible right there in
  the first four values, and it is exactly the noise that attenuates toward
  the null.

## Decision rule, frozen before scoring

This is a FORCED-PICK pool: 285 cards must be submitted either way. The
decision is expected value, never a 0.90/95% threshold —
`probability_positive` above 0.5 favours playing the candidate over the
baseline (predeclared thresholds govern only what a doc may CLAIM). This run is
close-graded, so it settles no play/no-play decision by itself; what it DOES
settle is whether the battery's screen-stage lean, measured the honest way
(stacked on what is actually played, not a bare baseline, and with the
look-ahead threshold removed), still looks worth an eventual opener-graded
confirmation look.

## Recording

ONE `nfl-ats weak-signals record` entry, `--effect-units accuracy_points`,
`--league nfl`, `--family redzone_reversion_on_production`,
`--classification unresolved_below_power`, NO `--closing-ground`,
`--reliability 0.407` (the battery's own recorded year-over-year Pearson for
`third_down_conv_rate`), `--category onfield`.

**This family is a DIFFERENT pooling bucket from the bare-baseline reversion
battery, and the two are NOT poolable.** Both measure the same underlying
trait, but against **non-commensurable comparators**: the battery scored a
subset cover-rate gap against a bare market baseline, this scores a paired
forced-pick accuracy delta against the full production chain. AGENTS.md's
commensurability rule ("pooled inputs must be commensurable — same units, same
scale, same population") forbids pooling them together, and the expanding
threshold (deviation 2) makes the populations differ as well.

ONE `nfl-ats rotation record --name redzone_reversion_on_production --verdict
unresolved` call spends the assigned window, carrying the paired effect,
interval and `probability_positive`.

## 7. Results (added after the look, 2026-09-01)

Rotation family `redzone_reversion_on_production` was declared close-graded
with **no inheritance** (`inherits: []`) and **without** `--acknowledge-mined`
— the CLI accepted both — then assigned by `nfl-ats rotation assign`, never
hand-picked: the earliest eligible close-pool block is **[2011, 2013]**. It
does not intersect the 2018-2025 mining ledger, so no acknowledgement was
required. The family retains **2** eligible contiguous windows (and 4 eligible
stratified seasons) after this one is spent. All numbers below are **measured**
this session unless labelled otherwise.

**Window counts, coverage and value distribution** (measured, in-window REG
completed games): **768** games, `redzone_third_down_over_fade_diff`
non-missing on **100.000%** of them. Value distribution inside the window:
**-1 on 18.750%** (144 games), **0 on 62.500%** (480), **+1 on 18.750%** (144)
— non-zero on exactly **37.500%**, matching the construction's own prediction
to three decimals. The paired scoring population is **746** games over **51**
weeks — all 51 weeks of the window were fitted, none dropped by
`MIN_FITTABLE_TRAIN_GAMES`. The 22-game gap is **pushes**: measured directly,
exactly 22 of the 768 in-window games settle at `result - spread_line == 0`,
and `nfl_ats.clv.pick_correct` returns NaN for a push rather than scoring it as
a loss (**read**, `src/nfl_ats/clv.py:2007-2023`). The illness sibling reports
the same 746-game paired population on this window. Both instrument checks ran
first, in the declared order.

**Null check** (`--mode null`, 200 within-week permutations, real feature, not
leaked; artifact
`artifacts/redzone_reversion_on_production/20260901T194709Z/results.json`):
mean **+0.172** accuracy points, sd 0.666, 95% [-1.206, +1.344]. A sane,
finite, non-degenerate distribution — the harness produces a null, not a crash
or a spike. The null centres above zero, as section 6 predicted it would: it
preserves each week's realized home-cover rate, and the two arms carry
different home-pick rates.

**Positive control** (`--mode positive-control`, the candidate's one column
replaced by the realized `ats_margin`; artifact
`artifacts/redzone_reversion_on_production/20260901T194731Z/results.json`):
paired delta **+50.134** accuracy points (candidate accuracy 100.000% against
the baseline's 49.866%), week-blocked P+ **1.000**, 95% [+46.428, +53.652],
season-blocked 95% [+48.000, +53.878], sitting at the **100.0th percentile** of
its own null (which itself centres at +2.713 pts under the leak treatment).
The full-profile ridge fit is **not blind** to a real effect of meaningful size
with this column embedded in 90 other production features, so the screen below
means something. The predeclared STOP condition did not fire.

**The real screen** (`--mode screen`, artifact
`artifacts/redzone_reversion_on_production/20260901T194749Z/results.json`), 746
paired games over 51 weeks, 3 seasons. Production `weak_stack` accuracy on this
paired population is **49.866%** (`baseline_accuracy` 0.49865951742627346) —
to the last digit the same paired baseline `docs/illness_on_production.md`
section 7 records for this window (**read**, that file: `baseline_accuracy`
0.49865951742627346), which is the cross-check that both runs scored the same
population with the same evaluator.

| quantity | measured |
|---|---|
| candidate accuracy | **50.536%** (0.5053619302949062) |
| paired delta | **+0.670 accuracy points** (0.006702412868632708) |
| week-blocked 95% CI | **[-0.408, +1.854]** |
| week-blocked `probability_positive` | **0.849** |
| season-blocked 95% CI | [+0.000, +2.000] |
| season-blocked `probability_positive` | 0.672 |
| percentile of its own permutation null | **74.5th** (null mean +0.172) |

**Home-pick rates** (measured): baseline **53.646%**, candidate **56.120%** —
a 2.47-point gap, wider than the illness sibling's 1.3-point gap and much
narrower than the 55-67% home-pick arms that discounted the bare-baseline
screens. This gap is exactly what pushes the permutation null off zero to
+0.172, and the 74.5th-percentile reading is the null-adjusted version of the
+0.670: roughly a quarter of the raw delta is the home-tilt offset the project's
own `home-tilt-null-artifact` note warns about, and about three quarters is not.

The season-blocked lower bound sits at exactly 0.000 with only 3 season blocks;
it is reported beside the week-blocked primary and **never averaged with it**,
and it is read with the same caution both sibling documents give their own
3-season secondaries: with 3 blocks the season-blocked bootstrap has very
little combinatorial diversity, so a bound landing on a round number is a
low-power artifact of block count, not a sharper answer than the week-blocked
primary.

### What this implies for the decision, before what is wrong with it

On EV grounds — `probability_positive` above 0.5 favours playing the candidate,
the only decision rule this project uses — **this measurement favours adding
the third-down fade column over the status quo, at week-blocked P+ 0.849.**
This is a FORCED-PICK pool: 285 cards must be submitted either way, so
declining a candidate that is ~85% likely better is not caution, it is taking
the other side of an 85/15 bet. The point estimate, **+0.670 accuracy points on
top of what is actually played**, is the **second-largest positive
on-production marginal recorded in this line of work**, behind only the illness
away arm measured on this same window today.

| construct | on-production delta | week-blocked P+ |
|---|---|---|
| graph `off_sack_rate` | -0.935 pts | 0.122 |
| graph `def_yards_per_play` | -0.668 pts | 0.189 |
| FluView away elevated | 0.000 pts | 0.403 |
| FluView home elevated | +0.969 pts | 0.792 |
| illness away active ≥1 | +0.804 pts | 0.908 |
| **third-down over-performer fade** | **+0.670 pts** | **0.849** |

(The first five rows are **reported** by the orchestrator's briefing and the
sibling documents and are **unverified by me**; the last row is measured this
session.)

**The result the quality-proxy risk predicted did not happen.** Section 1
stated honestly, before scoring, that prior-season third-down conversion rate
is quality-adjacent and that the `team-quality-is-already-priced` build filter
predicts such a feature should land near zero once stacked on a chain carrying
fifteen rolling efficiency states plus Elo. It landed at +0.670 with 85% of the
week-blocked bootstrap mass above zero, and it did so at the **74.5th
percentile of its own permutation null**, i.e. not merely as a home-tilt
artifact. That is evidence — not proof — that the construct is behaving as the
**reversion** mechanism it was declared to be rather than as another quality
measurement, which is the distinction section 1 said would decide how to read
this. The antisymmetric mirror cell (C4 at P+ 0.0989 against C3's 0.8719 on the
bare baseline) points the same way.

**What this does not settle.** This run is **close-graded**, and per the
binding "grade the decision at the opener" rule a close-graded look settles no
play/no-play or promotion decision regardless of sign. The close is the market
at its sharpest and systematically understates pool-relevant edge, so the
opener-graded number is the one that would decide a card change — and it is not
measured here. The honest next step is an **opener-graded confirmation look on
a disjoint window**; that is a new family, not a re-look, and nothing in this
document authorises a card change.

**Caveats, after the implication and not instead of it.** The week-blocked
interval reaches below zero (-0.408), which per the binding taxonomy is the
expected shape for a real small signal at this evaluator's ~2-point resolution
and is never grounds to close or discount a line of work. The column takes the
value 0 on 62.5% of the window's games (480 of 768) and differs between the
sides on only **288**; it is a ridge fit, so the added column also perturbs the
other 90 coefficients and the two arms' predictions are not identical on the
zero-valued games either — but the column's own direct contribution is confined
to those 288, which is the coarseness section 6 disclosed before scoring. Both
deviations remain in force and both cut toward the null: the signed difference
cannot distinguish both-flagged from neither-flagged (deviation 1), and the
expanding threshold is noisiest exactly in this early window, where the 2011
cut rests on two prior seasons and the 2012 cut on three (deviation 2) — the
pooled-cut version the frozen screen used would, if anything, read at least as
positive. One arm was measured on one window, so no multiplicity correction
arises within this family; but this block is heavily used —
`illness_on_production` drew the identical [2011, 2013] today and
`fluview_elevated_on_production`'s [2011, 2025] window overlaps it (**read**,
`registry/rotation_registry.json`) — and that is disclosed here rather than
corrected.

**This family is NOT poolable with the bare-baseline reversion battery.** Both
measure the same underlying trait, but the comparators are not commensurable —
a subset cover-rate gap scaled to the full slate against a bare market baseline
versus a paired forced-pick accuracy delta against the full production chain —
and deviation 2 makes the populations differ as well. AGENTS.md's
commensurability rule forbids combining them, and the registry entry carries
`family: redzone_reversion_on_production` precisely to keep the two buckets
apart.

The family is **not closed** and the column is **not promoted**: the result is
recorded `unresolved_below_power` with no closing ground, the family stays
**open**, and it retains 2 eligible close-pool windows.
