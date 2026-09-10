# Expected lineup loss (LEAD-62) scored THROUGH the played card

Predeclared 2026-09-10, **before** `scripts/lineup_loss_on_card.py` was run and
before any number in the Results section existed. Provenance tags: **measured**
(run this session, command/path given), **read** (file opened this session),
**reported** (another doc's claim, unverified here), **inferred** (reasoning,
not evidence).

## Binding closing-grounds taxonomy (pasted verbatim, per AGENTS.md)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (a) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (b) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it, report
> `probability_positive`, never "contains zero".

Within-week correlation is ZERO (owner mandate); the week-blocked bootstrap is
the primary uncertainty and no ICC is estimated or padded. Era readings differ
in MAGNITUDE; a weaker era is never absence. The decision is expected value:
forced picks, the card with the higher expected opener accuracy is played. A
0.90-style threshold governs only what these docs may CLAIM, never which card
is played.

## 1. The gap this fills

`weak_stack_expected_lineup_loss` (**read**,
`src/nfl_ats/expected_lineup_loss_challenger.py:48-83`) is a refit of the
active recipe on the production `weak_stack` columns PLUS the three frozen
`diff_expected_lineup_loss_{qb,offense,defense}` columns
(**read**, `src/nfl_ats/expected_lineup_loss_features.py:48-55`). It is a
paired challenger by owner order of 2026-09-05, left registered "for now".

It has exactly one prior look (**reported**, from
`docs/expected_lineup_loss.md:272-284`, not re-verified here): the CX5 opener
screen on the assigned 2020-2021 rotation window, **456** non-push games,
**+0.657895** accuracy points, week-blocked 95% [-1.569507, +3.118040],
`probability_positive` **0.66505**, 28 pick disagreements.

**That look scores the arm STANDALONE, on the bare model.** It has never been
scored through the card that is actually played. That matters for the same
reason `docs/unserved_tilt_marginals.md` gave for the tilt overlays and
`docs/era_weighted_on_card.md` gave for the era-weighted refit: the served card
already flips a large share of picks, so a change to the underlying model can
either add to, or be absorbed by, flips the card already makes. A standalone
number is not the decision number.

The prior look also predates the current served instrument: it used the
2020-2021 window on the then-current model, whereas the active model is
`2e8c616b476dd0d2` (**read**, `artifacts/active_ats_model.json`), serving
`gaussian_median` with the fitted home-side offset, on feature table
`fadeed326dc277416519e75914154c0d7b822b5ef28ba6d19efaa6469cb64e21`.

## 2. Population, instrument, and what is NOT in it

- **Archive**: `artifacts/opener_evaluation/20260910T005854Z` -- the 2020-2025
  Tuesday-opener archive of the ACTIVE model (**read**, its `metadata.json`
  names model `2e8c616b476dd0d2`, feature table `fadeed326...`, 1,537 games).
  34 of those are opener pushes, so **n = 1,503** scored games.
- **Harness**: `scripts/spread_hole_arms.py`'s `walk_forward` (**read**),
  reused unchanged except that the arm's feature profile varies. That harness
  replays the served incumbent on an opener archive with a residual gap of 0.0
  (**reported**, `docs/era_weighted_on_card.md` Section 9.1), which is why it
  is the harness used rather than a fresh reimplementation.
- **Grade**: Tuesday opener (`tue_open`), production probability rule
  (`home_cover_probability >= 0.5`), `gaussian_median`, with the served
  per-arm home-side offset fitted forward-only from that arm's own prior
  weeks. Pushes do not score.
- **Card**: the SERVED NINE-member policy, via
  `nfl_ats.unserved_tilt_marginals.served_card_flip_set(..., card="served")`
  (**read**, `src/nfl_ats/unserved_tilt_marginals.py:418-444`), whose members
  are `four_overlay_composition.COMPOSITION_ORDER`. The flip set is recomputed
  against EACH arm's own predictions, so a card flip that the candidate model
  makes unnecessary is credited to the candidate.
- **Stated limitation, not smoothed over**: the arrest member is
  reconstructed from the incidents table rather than replayed from a
  point-in-time capture, exactly as every prior card study on this archive
  does. The nine-member set is the card served on today's live Week 1 board.

## 3. Arms

Both arms fit the frozen production recipe -- `ridge`, `ridge_alpha=10.0`,
`market_residual` target, `min_train_games=500`, forward-chained weekly refits
on training rows strictly earlier than the scored week's first gameday.
**Only the feature column tuple changes.**

- `incumbent` -- `weak_stack`, `nfl_ats.margin.fit_margin_model`. This is the
  served model.
- `lineup_loss` -- the challenger's own
  `expected_lineup_loss_challenger.candidate_profile()` context and its
  `CANDIDATE_FEATURE_PROFILE` (`weak_stack_expected_lineup_loss`), which is
  the production `weak_stack` tuple plus exactly
  `EXPECTED_LINEUP_LOSS_COLUMNS`. This is the same profile the challenger's
  own weekly recorder refits under (**read**,
  `src/nfl_ats/expected_lineup_loss_challenger.py:246-259`, which calls
  `fit_margin_models_for_week(..., feature_profile=CANDIDATE_FEATURE_PROFILE,
  regressor="ridge", ridge_alpha=10.0, methods=("market_residual",))` inside
  that context).

The three added columns are built by
`expected_lineup_loss_features.attach_expected_lineup_loss_features` on the
CURRENT production feature table and the CURRENT play-probability panel, by
the same `build_features` recipe `scripts/expected_lineup_loss_on_production.py`
uses (**read**, its lines 82-140). The augmented table is written under this
lane's artifact directory and never under `data/processed`.

## 4. Gates, run before any candidate number is read

1. **Archive reproduction**: the `incumbent` arm must reproduce the archive's
   `residual_at_open` and `home_cover_probability_at_open`. The gaps are
   reported; a non-zero gap is a bug in this adaptation, not a finding.
2. **Profile identity**: `margin_feature_columns("market_residual",
   "weak_stack_expected_lineup_loss")` must equal
   `margin_feature_columns("market_residual", "weak_stack")` followed by
   exactly the three named columns, in that order. A mismatch aborts.
3. **Coverage**: every archive game must carry non-null values in all three
   added columns. Any game that does not is reported and dropped from BOTH
   arms, so the pairing stays exact.

## 5. The point-in-time question, asked before the arms are scored

This feature reads two sources whose visibility must be checked against the
pool's actual Tuesday lock, not against its own build cutoff:

- the WEEK-OF depth chart (`depth_rank == 1` rows of
  `data/processed/play_probability_panel.parquet`), and
- the as-of-visible injury designations from the production player snapshot.

Both are gated in code at `decision_at = pool_decision_cutoff(kickoff)` =
`min(kickoff, Sunday 16:00 America/New_York)` (**read**,
`src/nfl_ats/expected_lineup_loss_features.py:90-104`), which is the pool's
PICK deadline, not the Tuesday lock at which the pool's LINE freezes.
`docs/expected_lineup_loss.md:258-261` already states this in words
(**read**): "this existing opener harness substitutes the opener spread while
other production inputs retain their existing build timing. This is
opener-graded pool-deadline feature research, not a claim that these
injury/depth features were already available on Tuesday."

The audit measured this session (`point_in_time_audit.py` in this lane's
artifact directory, numbers in Section 10.0 below) decides which of the two
readings this document reports as its decision number:

- If a material share of team-games carries usable depth and injury
  information before that week's **Tuesday 12:00 ET**, the arm is a
  Tuesday-lock candidate and the 2020-2025 opener read is the decision number.
- If not, the arm is a **refresh-path** candidate only: the pool's line is
  frozen at Tuesday but the pick is due at `min(kickoff, Sunday 16:00 ET)`, so
  a deadline-vintage feature scored against the Tuesday-frozen opener line IS
  the refresh card. In that case the same measurement is reported, but named
  and interpreted as the refresh card, and a restricted **2023-2025** window
  is reported alongside it as the sub-window the task names.

Either way the arithmetic is identical; what changes is which card the number
is allowed to claim, and that claim is fixed here before the numbers exist.

## 6. Endpoints, declared before the run

Names are `lineup_loss_on_card_<arm>_<window>`, `arm` in
{`card`, `standalone`}.

- **Primary**: paired forced-pick opener accuracy improvement, candidate minus
  incumbent, on the **card** surface, overall (`window` = `overall`).
- **Secondary, same footing, reported in full whatever the sign**: the
  `standalone` surface (the served raw model, no overlays); per-season windows
  2020..2025; spread-bucket windows `0-6.5`, `7`, `7.5-10`, `10.5+` (the
  coarse map already in `scripts/spread_hole_arms.py:56`); the
  `refresh_2023_2025` window on both surfaces; Brier and log loss on the
  served opener cover probability, both surfaces; picks changed on every cut.
- **Uncertainty**: paired week-blocked and season-blocked bootstrap, 20,000
  samples, seed 20260821 -- the same machinery and seed the composition study
  and `spread_hole_arms.py` use. Season-blocked on a per-season window is one
  block and therefore degenerate; it is reported as degenerate, never as an
  interval.
- **Family declared before signs are seen**: the commensurable family for any
  pooling is `{card overall, standalone overall}` in `accuracy_points` on this
  archive. The per-season, per-bucket and `refresh_2023_2025` cells are
  decompositions of those same games; they are recorded with the overlap
  stated and must not be pooled as independent inputs.

## 7. Classification rule, mechanical, decided by reading the artifact

- Whole week-blocked primary interval below zero -> `refuted_mechanism` with
  `--closing-ground wrong_sign_resolved`.
- No positive control is run in this document. The CX5 leaky control
  (**reported**, `docs/expected_lineup_loss.md:264-270`: +44.30 points by
  substituting realized ATS margin) establishes sensitivity to a huge injected
  signal and does NOT bound a candidate-sized effect, so
  `bounded_by_control` is unavailable by construction.
- Everything else, including a wholly-positive interval ->
  `unresolved_below_power`, reported with `probability_positive`.
- If a `weak-signals record` call errors, the verdict is wrong, not the
  validator: reclassify as `unresolved_below_power`.

## 8. Decision rule, stated before the numbers

The pool is forced picks. The decision is expected value at the opener,
through the card: if the candidate's card-surface point estimate is positive,
the honest statement is "playing the lineup-loss card is the better side of a
P+ bet", and the write-up says exactly what promoting it would take and which
Week 1 picks move. If it is negative, the same logic runs the other way and
the arm is not proposed for the card -- and either way the result is recorded
as `unresolved_below_power` unless the mechanical rule in Section 7 fires.
**No promotion is performed by this document under any outcome**; the active
manifest, the ledgers, the forecast and the board are not touched.

## 9. Commands

```
uv run --no-sync python scripts/lineup_loss_on_card.py \
  --features data/processed/game_features_weak_stack.parquet \
  --data-root data --market-root data/market/raw \
  --archive artifacts/opener_evaluation/20260910T005854Z \
  --out artifacts/lineup_loss_on_card/20260910T035735Z
```

## Results

**Measured** 2026-09-10, `scripts/lineup_loss_on_card.py`, artifact
`artifacts/lineup_loss_on_card/20260910T035735Z/`. Command in Section 9.

### 10.0 The point-in-time answer: this is a REFRESH-PATH feature, not a Tuesday one

**Measured** this session,
`artifacts/lineup_loss_on_card/20260910T035735Z/point_in_time_audit.py` (output
saved beside it as `point_in_time_audit.json`) over
`data/processed/play_probability_panel.parquet` (355,446 rows) and the
production player snapshot `20260910T004732Z`:

| Question | Measured |
| --- | ---: |
| Panel team-weeks with ANY depth observation before their own week's Tuesday 12:00 ET | **0 of 6,814** |
| Panel team-weeks with a real (non-legacy) depth observation time at all | 544 of 6,814 (2025 only) |
| Daily depth rows: hours between observation and the decision cutoff, median | **10.77 h** |
| Daily depth observation weekday | Sun 31,088 / Thu 2,642 / Sat 765 / Fri 251 |
| Injury rows visible at the pool decision cutoff | 86,474 of 87,220 (99.1%) |
| Injury rows visible at that week's Tuesday 12:00 ET | **328 of 87,220 (0.38%)** |
| Injury rows 2020-2025 visible at Tuesday 12:00 ET | **187 of 33,383 (0.56%)** |
| Team-weeks with any injury row visible at Tuesday 12:00 ET | 129 of 8,856 (1.5%) |

The other 6,270 team-weeks are `legacy_week` rows whose `depth_observed_at` is
NULL for all 320,700 of them; `_visible_panel` lets those through on the
archive's week-labelled pregame assumption, so their real observation time is
not merely late, it is **unknown**. Nothing here is a defect in the feature; it
is what the inputs are. Both sources are gated at
`min(kickoff, Sunday 16:00 ET)` in code and both arrive within roughly half a
day of that instant.

**So the arm is not a Tuesday-lock candidate.** Per Section 5, the decision
number is therefore reported as the **refresh card**: the pool freezes its LINE
on Tuesday but the PICK is due at `min(kickoff, Sunday 16:00 ET)`, so a
deadline-vintage feature scored against the Tuesday-frozen opener line is
exactly the refresh card. Every number below is that reading, on the same
arithmetic, and the `refresh_2023_2025` rows are the restricted window Section 6
declared. The incumbent baseline carries deadline-vintage injury features too,
so the comparison is fair; what would be wrong is calling it a Tuesday-noon
result.

### 10.1 Gates pass before any candidate number was read

- **Archive reproduction**: 1,537 rows, `max_residual_gap` **0.0**,
  `max_probability_gap` **0.0** against
  `artifacts/opener_evaluation/20260910T005854Z/per_game.parquet`.
- **Profile identity**: 90 baseline columns -> **93** candidate columns, the
  added three being exactly `diff_expected_lineup_loss_offense`,
  `diff_expected_lineup_loss_defense`, `diff_expected_lineup_loss_qb`.
- **Coverage**: `archive_games_without_loss_features` = **0**. No game is
  dropped from either arm.
- **Split-half reliability** of the trait, computed in the build before either
  arm was scored: **0.8140 on 384 team-seasons**
  (`artifacts/lineup_loss_on_card/20260910T035735Z/build.json`). Non-zero, so
  `no_split_half_reliability` is unavailable as a closing ground.
- **Served card**: nine members, OR-union **487** flips under the incumbent and
  **488** under the candidate on 1,537 archive games.

### 10.2 The decision number

Paired forced-pick accuracy, candidate minus incumbent, Tuesday-opener grade,
1,503 non-push games, 107 week blocks, 6 season blocks, 20,000 samples, seed
20260821.

| Name | Card % (base -> cand) | Picks changed | Effect (pts) | Week 95% | P+ (week) | Season 95% | P+ (season) |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| `lineup_loss_on_card_card_overall` (PRIMARY) | 56.886 -> 56.154 | 55 | **-0.732** | [-1.651, +0.201] | **0.0631** | [-1.841, +0.465] | 0.1203 |
| `lineup_loss_on_card_standalone_overall` | 54.558 -> 54.092 | 97 | -0.466 | [-1.732, +0.819] | 0.2374 | [-1.991, +1.632] | 0.2801 |
| `lineup_loss_on_card_card_refresh_2023_2025` | 56.070 -> 55.069 | 28 | -1.001 | [-2.233, +0.251] | 0.0563 | [-2.247, -0.376] | 0.0000 |
| `lineup_loss_on_card_standalone_refresh_2023_2025` | 55.069 -> 53.942 | 51 | -1.126 | [-2.771, +0.622] | 0.0953 | [-1.873, -0.376] | 0.0000 |

The primary is **-0.732 accuracy points**, `probability_positive` **0.0631** --
net **11 fewer correct picks** across six seasons out of 55 the refit changed.
The standalone surface moves 97 picks and loses **7**; the played card absorbs
roughly two fifths of the model's flips, so the card surface changes fewer picks
than the raw model does and still lands slightly further behind.

### 10.3 By season (card surface)

Season-blocked uncertainty is one block per row and therefore degenerate; the
week-blocked interval is the readable one. Magnitudes differ by era; no season
is an absence.

| Season | n | Card % (base -> cand) | Picks changed | Effect (pts) | Week 95% | P+ |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| 2020 | 220 | 59.091 -> 56.364 | 10 | -2.727 | [-5.357, -0.441] | 0.0142 |
| 2021 | 236 | 55.508 -> 57.627 | 11 | **+2.119** | [-0.415, +4.938] | 0.9559 |
| 2022 | 248 | 58.871 -> 58.065 | 6 | -0.806 | [-2.767, +1.181] | 0.2084 |
| 2023 | 266 | 59.398 -> 59.023 | 5 | -0.376 | [-1.901, +1.158] | 0.3301 |
| 2024 | 266 | 54.135 -> 53.759 | 9 | -0.376 | [-2.642, +2.273] | 0.3655 |
| 2025 | 267 | 54.682 -> 52.434 | 14 | -2.247 | [-4.478, -0.368] | 0.0134 |

**2021 is the one season that leans to the candidate, and it is the same season
the CX5 screen was assigned.** That screen's window was 2020-2021 and its
headline `+0.658` was 2020's `-1.364` averaged with 2021's `+2.542`
(**reported**, `docs/expected_lineup_loss.md:289-292`, unverified here). On this
instrument the same split reappears -- 2020 `-2.727`, 2021 `+2.119` on the card;
2020 `-3.182`, 2021 `+4.661` standalone -- and the four seasons the CX5 window
never saw all lean the other way. That is the single most decision-relevant fact
in this table: the prior positive look was one season out of six, and the
out-of-window seasons did not confirm it.

### 10.4 By spread bucket (card surface)

| Name | n | Card % (base -> cand) | Picks changed | Effect (pts) | Week 95% | P+ |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `lineup_loss_on_card_card_0-6.5` | 1104 | 58.786 -> 58.605 | 40 | -0.181 | [-1.372, +1.014] | 0.3861 |
| `lineup_loss_on_card_card_7` | 74 | 50.000 -> 44.595 | 4 | -5.405 | [-11.268, -1.266] | 0.0083 |
| `lineup_loss_on_card_card_7.5-10` | 194 | 52.577 -> 48.969 | 7 | -3.608 | [-6.316, -1.257] | 0.0003 |
| `lineup_loss_on_card_card_10.5+` | 131 | 51.145 -> 52.672 | 4 | **+1.527** | [-1.587, +5.674] | 0.7839 |

Standalone, `10.5+` reads **+2.290** [-2.419, +7.407], P+ 0.8136 on 131 games
with 9 picks changed, and `7.5-10` reads -2.062, P+ 0.1026. This is a diagnosis
to publish, not a threshold to bolt on: the refit's damage concentrates in the
7 to 10 zone -- the same zone `docs/spread_hole_diagnosis.md` is about -- while
the largest spreads are the one place it leans the candidate's way on both
surfaces. Four picks in a 74-game bucket is four picks; the `card_7` cell is a
decomposition, not a finding.

### 10.5 Brier and log loss

| Metric | Baseline | Candidate | Improvement | Week 95% | P+ |
| --- | ---: | ---: | ---: | --- | ---: |
| Brier, card overall | 0.246965 | 0.247580 | -0.000615 | [-0.001372, +0.000142] | 0.0542 |
| Log loss, card overall | 0.687158 | 0.688413 | -0.001256 | [-0.002817, +0.000304] | 0.0563 |
| Brier, standalone overall | 0.251581 | 0.251883 | -0.000302 | [-0.001026, +0.000414] | 0.2030 |
| Log loss, standalone overall | 0.696618 | 0.697245 | -0.000627 | [-0.002105, +0.000834] | 0.1989 |
| Brier, card refresh 2023-2025 | 0.248291 | 0.248422 | -0.000132 | [-0.001060, +0.000769] | 0.3942 |
| Log loss, card refresh 2023-2025 | 0.689817 | 0.690083 | -0.000266 | [-0.002174, +0.001586] | 0.3953 |

Accuracy and calibration lean the same way here, both toward the baseline, both
by amounts whose intervals contain zero.

### 10.6 Classification, read off the artifact

Applying Section 7's mechanical rule, unchanged:

- `lineup_loss_on_card_card_overall` (PRIMARY): week-blocked [-1.651, +0.201]
  is **not** wholly below zero, so `wrong_sign_resolved` does not fire.
  Reliability is 0.8140, so `no_split_half_reliability` does not fire. No
  positive control was run, so `bounded_by_control` is unavailable by
  construction. -> **`unresolved_below_power`**, `probability_positive`
  **0.0631**.
- All 24 cells are recorded `unresolved_below_power`. Five decomposition cells
  carry wholly-negative week intervals (`card_7`, `card_7.5-10`, `card_2020`,
  `card_2025`, `standalone_2020`) and one carries a wholly-positive one
  (`standalone_2021`, +4.661 [+0.826, +9.052], P+ 0.9907). **None of them is
  recorded terminal**: Section 7 gives the terminal rule to the primary cell
  only, and AGENTS.md's own decomposition caution -- any real effect can be
  sliced until its pieces resolve -- cuts the same way for negatives as for
  positives. **Nothing is closed by this document.**
- Twenty-four cells recorded with `nfl-ats weak-signals record`, one call at a
  time, family `expected_lineup_loss_on_card`, category `health`, units
  `accuracy_points`; argv and return codes in
  `artifacts/lineup_loss_on_card/20260910T035735Z/record_commands.json`, all
  `rc=0`, all read back from `registry/weak_signals.json`.

### 10.7 The decision

**On expected value, do not put this refit on the card, and keep it a paired
challenger.** Forced picks, opener grade, through the played nine-member card:
the lineup-loss card is **0.73 accuracy points behind** the card that is played
and the probability it is better is **0.0631**. On the refresh window the task
named it is **1.00 points behind** at P+ 0.0563. Declining it is not caution --
taking it would be the 1-in-16 side of the bet.

That is a decision, not a closure. The trait is reliable (0.814), the mechanism
is intact, and one season and one spread bucket lean the candidate's way; the
arm keeps recording paired prospective picks and the registry rows above are its
evidence.

**What promoting it WOULD take, for the record, since the question was asked.**
Nothing below was done.

1. Register `weak_stack_expected_lineup_loss` as a real
   `MarginFeatureProfile` in `src/nfl_ats/margin.py`
   (`MARGIN_FEATURE_PROFILES`, `_MARGIN_PROFILE_FEATURE_SETS`) and its two
   feature sets in `constants.py` -- today it exists only inside the
   challenger's `patch.dict` context, so `weekly-run` cannot name it.
2. Make the three columns part of the built feature table, which means
   `attach_expected_lineup_loss_features` runs inside
   `build-learned-availability-features` (or its own build step) and writes
   `data/processed/game_features_weak_stack.parquet`. That changes the table's
   sha256 and therefore the model id.
3. Recompute everything keyed to the model: `margin-backtest`, a new
   `artifacts/active_ats_model.json`, then `opener-evaluation` and
   `overlay-composition` under the new model id, then `publish-predictions` and
   `publish-board` (which fails closed if any headline number still names the
   old model).
4. **The refresh path is the part that actually matters here**, per 10.0: since
   the feature is only populated near the deadline, promoting it to the Tuesday
   card would serve a column that is empty at Tuesday noon. The honest wiring
   is `scripts/refresh_lineup_forecast.py` / the scheduler's daily `lineups_*`
   job, i.e. a refresh-card model, not the Tuesday-lock model.
5. De-register the paired challenger (`artifacts/prospective/challengers.json`)
   and retire its `settlement.py` ledger spec, or the ledger would pair the
   promoted model against itself.

**Which Week 1 picks it moves** (**measured**, read-only replay of the
challenger's own refit path against the ACTIVE forecast
`2026-week-01-20260910T005452Z`, feature table `fadeed326...`; wrote nothing):
**1 of 16** model picks moves.

| Game | Model p(home), served -> lineup-loss | Model pick | Published pick today |
| --- | --- | --- | --- |
| NE at SEA | 0.496252 -> 0.500304 | NE -> SEA | NE +3.5 |

It lands **0.0003** past the fence. The challenger's own already-recorded Week 1
ledger row agrees (**measured**,
`artifacts/prospective/weak_stack_expected_lineup_loss_paired_decisions.parquet`:
16 rows, 1 differing from its paired baseline, the same NE at SEA game), though
that ledger was frozen against the earlier forecast
`2026-week-01-20260909T220047Z`. **Inferred, not measured:** if the served
rules' flip set were unchanged on that game the published card would read SEA
-3.5 instead of NE +3.5 -- but the archive shows the flip set is not strictly
arm-independent (487 flips under the incumbent, 488 under the candidate), so
that last step is reasoning, not a measurement.

### 10.8 Stated limitations

- The 2013-2024 depth panel is `legacy_week` throughout: 320,700 of 355,446
  rows have no observation timestamp at all, so their pregame status is an
  archive assumption, not a verified fact. Only 2025's 34,746 rows carry real
  times, and those land a median 10.8 hours before the deadline.
- The arrest member of the card is reconstructed from the incidents table
  rather than replayed from a point-in-time capture, as in every prior card
  study on this archive.
- Twenty-four cells over one 1,503-game archive are correlated decompositions,
  not 24 votes. The declared pooling family is the two overall cells.
