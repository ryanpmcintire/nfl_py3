# Promotion evaluation, lane T (2026-09-05): qb_revenge and deadline_drag

**Predeclared before any of the four arms below was scored.**

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. A promotion bar is not a decision bar: the pool is forced picks,
so the decision is expected value, graded at the OPENER; declining a
candidate that is 80% likely better is taking the other side of an 80/20
bet.

## Context

Two one-column candidates were screened overnight on production `weak_stack`,
opener-graded on the rotation-assigned window **[2020, 2021]** (456 paired
games, 35 weeks): `weak_stack_qb_revenge` (`qb_revenge_flag`,
`src/nfl_ats/qb_identity_features.py`, `docs/schedule_flag_battery.md` Wave 5)
read effect **+0.6579** accuracy points, week-blocked 95% **[-0.228, +1.948]**,
`probability_positive` **0.83395**; `weak_stack_deadline_drag`
(`deadline_integration_drag_flag`, `src/nfl_ats/transaction_flag_features.py`,
Wave 6) read effect **+0.6579**, week-blocked 95% **[-0.2198, +1.5945]**,
`probability_positive` **0.8829**, 70/4,902 games flagged full-schedule (25
resolved events). Both were **read** from `docs/schedule_flag_battery.md`
(Wave 5/Wave 6 "Measured results" tables) and are unverified by this lane
beyond that read.

Between that screen and this look, the production `weak_stack` feature table
(`data/processed/game_features_weak_stack.parquet`) was rebuilt with the
ENG-39 injury-timestamp-fallback idempotency fix (**measured**, this
session: `sha256sum data/processed/game_features_weak_stack.parquet` ->
`41a778f26a38e63bede7e7bf01f4a4a30254c09164cae3c5ee2cce87bc2547f6`, matching
the value named in this lane's own task brief and
`C:\...\scratchpad\reports\laneS2_eng39_wiring.md`'s rebuilt table). The
currently-ACTIVE model (`artifacts/active_ats_model.json`) has **not** been
repointed at this rebuilt table yet (its own `feature_table_sha256` still
reads the pre-fix `13644e5c...`, **read** directly from that file) -- this
lane does not change that; no `margin-predict`, no `active_ats_model.json`
write, per this lane's mandate.

## Question

Should `weak_stack_qb_revenge`, `weak_stack_deadline_drag`, or their union be
PROMOTED into the played chain, on top of the rebuilt production table?

## Multiplicity disclosure (binding, stated before scoring)

This is a **promotion look on a REUSED population**, not a fresh rotation
confirmation, exactly like `docs/player_arrests_policy_eval.md`'s precedent
and the MOD-07 promotion note in `AGENTS.md`. All three candidate arms below
(`qb_revenge`, `deadline_drag`, `both`) are graded on the SAME full
2020-2025 Tuesday-opener archive (~1,537 paired games) that:

1. `qb_revenge` and `deadline_drag` were each already screened against
   (albeit on a narrower [2020, 2021] rotation window, a subset of this
   archive), and
2. the two candidates are themselves correlated draws from the same
   overnight lead-generation process (both from `docs/schedule_flag_battery.md`'s
   waves), so selecting among "qb_revenge alone", "deadline_drag alone", and
   "both together" after seeing all three screen results is a multiplicity
   problem, not three independent confirmations.

Per `AGENTS.md`'s pooling discipline, this is disclosed, not hidden, and does
not by itself invalidate acting on the result: the project's decision rule is
expected value on a forced-pick pool, not a multiple-comparisons-corrected
significance test. **No rotation window is spent by this look** (matching
the arrest-policy precedent): this is a promotion evaluation on already-mined
history, not a new confirmation draw.

## Design

Four arms, same estimator as the active model (ridge alpha 10, target
`market_residual`, `min_train_games=500`), fitted via
`nfl_ats.clv.opener_pick_evaluation` (the identical machinery
`scripts/on_production_opener_confirmation.py` and every on-production wave
in `docs/schedule_flag_battery.md` already uses -- imported, not
reimplemented):

| Arm | Feature profile | Added column(s) |
|---|---|---|
| `base` | `weak_stack` | none |
| `qb_revenge` | `weak_stack_qb_revenge` | `qb_revenge_flag` |
| `deadline_drag` | `weak_stack_deadline_drag` | `deadline_integration_drag_flag` |
| `both` | `weak_stack_qb_revenge_deadline_drag` (**new**) | `qb_revenge_flag` + `deadline_integration_drag_flag` |

`weak_stack_qb_revenge_deadline_drag` is a new additive `MarginFeatureProfile`
(`src/nfl_ats/constants.py`, `src/nfl_ats/margin.py`): `weak_stack` plus BOTH
already-declared `FEATURE_FAMILIES` entries `qb_revenge_on_production` and
`deadline_integration_drag_on_production` (no new `FEATURE_FAMILIES` key --
both columns already belong to a family from their own individual
screens). `profile_identity()` (`scripts/on_production_opener_confirmation.py`,
reused unmodified) is asserted for every non-base arm: each candidate profile
is `weak_stack` plus EXACTLY its declared added column(s), nothing else.

Both columns are built onto the rebuilt production table exactly as their
own lanes built them: `nfl_ats.qb_identity_features.attach_qb_revenge_features`
(schedule + combine + weekly-roster snapshots, newest-snapshot convention) and
`nfl_ats.transaction_flag_features.attach_deadline_integration_drag_features`
(schedule + PFR transaction-wire index + snap counts, newest-snapshot
convention) -- both strictly pregame, both already leakage-tested in
`tests/test_qb_identity_features.py` / `tests/test_transaction_flag_features.py`
(unmodified by this lane).

## Metric, controls, grade

Primary: opener-graded, production probability rule
(`home_cover_probability_at_open >= 0.5`), paired candidate-minus-base
forced-pick accuracy, week-blocked bootstrap (20,000 resamples, seed
20260902) plus a season-blocked secondary read, both via
`nfl_ats.clv.week_blocked_bootstrap` -- identical machinery to every
sibling on-production script. Sign-rule and close-graded reads are reported
alongside as secondary, matching every prior wave's convention (AGENTS.md:
grade the decision at the opener).

**Positive control**, run once: for each of the three non-base arms, the
arm's own added column(s) are replaced with the REALIZED `ats_margin`
(deliberately leaky, never promotable) before fitting; this must read
`probability_positive` near 1.0 with a large positive effect, or the harness
cannot be trusted to detect an effect of any size in the screen reads below.

**Card impact** (secondary, descriptive): each arm is additionally fit once
on the FULL current table (all completed games, no rotation/window
restriction -- mirroring `nfl_ats.margin.fit_margin_model`'s own internal
training-row selection, the same call `opener_pick_evaluation` and
production's own weekly forecast both make) and used to `.predict()` the 16
2026-Week-1 games in the table (`probability_method="gaussian"`, matching
`nfl_ats.outcomes.score_outcome_week`'s own production call). Two
comparisons are reported, kept distinct because they are NOT the same
question: (a) each candidate arm's Week-1 pick vs. the `base` arm's Week-1
pick, BOTH freshly fit on the current (ENG-39-fixed) table -- isolates the
candidate's own effect; (b) each arm's Week-1 pick vs. the pick already
recorded in the newest, already-published
`artifacts/margin_predictions/2026-week-01-*/predictions.csv` (read-only,
never regenerated) -- this conflates the candidate effect with the
ENG-39 table fix itself, since that published card was generated before the
production table was rebuilt (its own `feature_table_sha256` is the pre-fix
value, read directly above); disclosed, not hidden.

## Decision rule (stated before results)

Per `AGENTS.md` "a promotion bar is not a decision bar": the card should
play whichever of `{base, qb_revenge, deadline_drag, both}` has the highest
`probability_positive` (opener, production rule, week-blocked) PROVIDED that
arm's point estimate is non-negative; if no candidate arm clears
`probability_positive > 0.5` with a non-negative point estimate, the base
(currently-played) arm remains the recommendation. This is an EV decision,
not a significance test -- an interval crossing zero does not veto a play
with `probability_positive` above 0.5.

## Recording plan

Every non-base arm's result is recorded with
`nfl-ats weak-signals record --league nfl --season-start 2020 --season-end
2025 --family promotion_eval_20260905 --effect-units accuracy_points
--classification unresolved_below_power --category modeling` (unless a
RESOLVED wrong sign, i.e. the WHOLE interval strictly negative, applies --
predeclared as the only admissible closing ground checked below;
`no_split_half_reliability` is inadmissible for the same reason
`docs/schedule_flag_battery.md` gives for every deterministic pregame flag
in this repo: zero measurement noise, so there is nothing for a split-half
read to characterize). `--notes` on every row discloses the reused-population
multiplicity stated above. **No rotation-registry record** is filed (no
rotation window is spent; this is a promotion look on the archive, matching
the `player_arrests_policy_eval.md` precedent, not a fresh confirmation
draw).

## Results (measured, 2026-09-05)

**Screen** (`.\.tools\uv.exe run --no-sync python scripts\promotion_eval_20260905.py --mode screen`,
`artifacts/promotion_eval/20260905T141121Z/results.json`). `profile_identity`
confirmed additive discipline for every arm: base 90 columns; `qb_revenge`
91 (+`qb_revenge_flag`); `deadline_drag` 91 (+`deadline_integration_drag_flag`);
`both` 92 (+both). Population: **1,537 paired games, 107 weeks, 6 seasons
(2020-2025)**, matching the archive size named in this lane's task and in
`AGENTS.md`'s own MOD-07 citation.

Primary read (opener, production rule, week-blocked, 20,000 draws, seed
20260902), candidate minus base, in **accuracy points** (percentage
points):

| Arm | Effect (acc. pts) | Week-blocked 95% CI | P+ (week) | Season-blocked 95% CI | P+ (season) | Differing picks (of 1,537) |
|---|---|---|---|---|---|---|
| `qb_revenge` | -0.0665 | [-0.535, +0.399] | 0.34205 | [-0.527, +0.415] | 0.34195 | 11 |
| `deadline_drag` | -0.1996 | [-0.872, +0.468] | 0.2506 | [-0.967, +0.420] | 0.28505 | 37 |
| `both` (stacked) | +0.0665 | [-0.734, +0.874] | 0.5336 | [-0.950, +0.771] | 0.588 | 45 |

Every interval crosses zero -- the EXPECTED shape at this evaluator's
resolution, per the taxonomy. No admissible closing ground applies to any
arm: no interval sits entirely on the wrong side of zero (`wrong_sign_resolved`
unavailable for all three), and `no_split_half_reliability` is inadmissible
for the same reason `docs/schedule_flag_battery.md` gives each column
individually (deterministic pregame facts, zero measurement noise). All
three are `unresolved_below_power`.

**Both single-column arms FLIPPED SIGN relative to their own narrower
[2020, 2021] rotation-window screens**: `qb_revenge` read P+ 0.834 there,
P+ 0.342 here; `deadline_drag` read P+ 0.883 there, P+ 0.251 here. This is
reported as a measured fact, not smoothed over -- the wider, reused
2020-2025 population (which fully contains the narrower window) and the
ENG-39-fixed table together move the read from "leans favourable" to
"leans unfavourable" for each column alone. The **stacked** arm, built from
two individually-unfavourable-leaning columns, reads P+ 0.5336 (week) /
0.588 (season) -- barely favourable, the mirror image of the usual
"composition is not the signal" warning (here two negatives partially
cancel toward a weak positive rather than a positive canceling toward
negative).

Per-season magnitudes (opener, production rule; `delta` = candidate minus
base accuracy, disagreeing picks in parentheses):

| Season | n | `qb_revenge` delta (flips) | `deadline_drag` delta (flips) | `both` delta (flips) |
|---|---|---|---|---|
| 2020 | 227 | +0.909 pts (2) | +0.455 pts (1) | +0.909 pts (2) |
| 2021 | 239 | 0.000 pts (0) | +0.847 pts (4) | +0.847 pts (4) |
| 2022 | 255 | -0.806 pts (2) | 0.000 pts (8) | +0.806 pts (10) |
| 2023 | 272 | 0.000 pts (2) | -0.376 pts (5) | 0.000 pts (8) |
| 2024 | 272 | -0.752 pts (4) | -1.880 pts (7) | -2.256 pts (8) |
| 2025 | 272 | +0.375 pts (1) | 0.000 pts (12) | +0.375 pts (13) |

No single season dominates any arm's overall read the way 2025 dominated
the ENG-39 injury-fix confirmation; the per-season deltas are small and mixed
in sign for all three arms.

**Positive control** (`--mode positive-control`,
`artifacts/promotion_eval/20260905T142315Z/results.json`): leaking each
arm's own added column(s) to the realized `ats_margin` reads **identically
for all three arms** -- **+42.71 accuracy points**, `probability_positive`
**1.0** at both blockings, candidate accuracy 0.968 vs. base 0.541 (697 of
1,537 picks flip). This is the same mechanical consequence every prior
on-production wave documents (the leaked value is the same real
`ats_margin` regardless of which column(s) it replaces) and confirms the
harness is sensitive to an effect of this size before any of the three
screen reads above is trusted.

**2026 Week-1 card impact** (read-only; `artifacts/margin_predictions/2026-week-01-20260903T143253Z/predictions.csv`,
the newest artifact at the moment this lane's screen ran): **0 of 16**
Week-1 games flip for ANY of the three arms, either against the freshly
recomputed `base` arm or against the published card's own `market_residual`
pick. Every arm's `home_cover_probability` for every Week-1 game differs
from base by well under 0.001 -- neither `qb_revenge_flag` nor
`deadline_integration_drag_flag` is nonzero for any of this week's 16
games (both are low-frequency flags, 1.1%/1.4% of the full schedule).
**Verified separately** (measured, this lane, after the coordinator's
note that a newer forecast `2026-week-01-20260905T141453Z` was published
mid-run from the same, now-activated production table): the two published
predictions.csv files' `market_residual` picks are IDENTICAL for all 16
games (max `home_cover_probability` difference 0.0025), so this card-impact
read is unaffected by which of the two artifacts is treated as "newest."

## EV decision (stated before the caveats, per the predeclared rule above)

Per the predeclared rule, the card should play whichever arm has the
highest `probability_positive` among arms with a non-negative point
estimate; base otherwise. `qb_revenge` (-0.0665 pts) and `deadline_drag`
(-0.1996 pts) both have NEGATIVE point estimates and are disqualified by
the rule. `both` has a positive point estimate (+0.0665 pts) with
`probability_positive` 0.5336 (week) / 0.588 (season) -- mechanically, this
is the one arm the rule selects.

**Coordinator decision: do not promote, on SELECTION grounds, not a
threshold.** A promotion bar is not a decision bar (`AGENTS.md`), so the
reason to hold off here is deliberately NOT "P+ 0.53-0.59 is too low a
number." The actual reason is that this P+ is the output of SELECTING the
best of three correlated arms drawn from one reused window, after seeing
that BOTH of the stack's own components read AGAINST the candidate on that
identical population (`qb_revenge` P+ 0.342, `deadline_drag` P+ 0.251) --
the stack's barely-favourable read is exactly the shape multiplicity
produces, not independent evidence for the mechanism. Combined with the
2026 Week-1 card impact being exactly zero regardless (no live pick
actually changes this week either way), there is nothing here to act on
now: not because the number is too small, but because there is no
uncontaminated number to act on yet. The practical recommendation is: **do
not promote any of the three arms into the played chain now; keep base**.
All three findings are kept (recorded `unresolved_below_power`, not
discarded) for future pooling, exactly as the taxonomy requires -- this is
a "not yet," not a "no." The clean next step, which costs no
rotation-registry window, is prospective 2026 evidence: the stacked
profile (`weak_stack_qb_revenge_deadline_drag`) is registered as a
no-window-cost 2026 prospective challenger
(`artifacts/prospective/challengers.json`,
`src/nfl_ats/qb_revenge_deadline_drag_stack_challenger.py`), exactly the
path `best_pick_nomination_v3` and `mod07_weak_signal_stack` recommend/use
for a confounded-but-not-refuted archive read.

## Verification plan

`ruff format --check` / `ruff check` on every touched file; `mypy src`;
`tests/test_features.py`, `tests/test_market_decomposition.py`,
`tests/test_cli_contract.py`, `tests/test_margin.py`, plus a new
`tests/test_promotion_eval_profiles.py` pinning the stacked profile's exact
column set. The CLI contract fixture is regenerated LAST, after every other
code change, via
`.\.tools\uv.exe run --no-sync python scripts\cli_contract_snapshot.py tests\fixtures\cli_contract.json --normalize-years`.


## 2026-09-05 CX8 rerun: decision-time assignments

Inferred (decision read from the measured results below): the corrected roof
and full-archive QB arms favor the baseline on this frozen screen recipe; the
2020-2021 QB arms make exactly the baseline picks and offer no directional
preference. Measured (`artifacts/cx8_posthoc_assignments/20260905/record_commands.json`):
all six affected records were replaced through `nfl-ats weak-signals record
--replace` as `unresolved_below_power`; no line was closed and no new rotation
window was consumed. Original registry entries remain in
`superseded_registry_entries.json` alongside the rerun outputs.

Measured (`scripts/cx8_posthoc_assignments.py`): all arms use the same frozen
feature table (digest `41a778f26a38e63bede7e7bf01f4a4a30254c09164cae3c5ee2cce87bc2547f6`),
the original weak-stack/ridge screen recipe and opener probability rule,
chronological training, 20,000 week-blocked bootstrap samples, seed 20260902.
The oracle replay holds that table fixed to isolate the assignment correction;
the older recorded screens used a different table. These are reruns of the
original research recipe, not an evaluation of the active card's full overlay
policy. Each table cell is **accuracy points [95% interval]; probability_positive**.
The rotation comparisons have 456 games / 35 weeks; full-archive comparisons
have 1,503 games / 107 weeks. Prediction-level outputs are retained as
`<candidate>_paired.csv` and `oracle_<candidate>_paired.csv` in that directory.

| Screen | Original recorded (read) | Oracle on frozen table (measured) | Decision-time on frozen table (measured) |
|---|---|---|---|
| QB revenge, 2020-2025 | -0.067 [-0.535, +0.399]; 0.34205 | -0.067 [-0.535, +0.399]; 0.34205 | -0.665 [-2.021, +0.522]; 0.12820 |
| QB + deadline, 2020-2025 | +0.067 [-0.734, +0.874]; 0.53360 | +0.133 [-0.594, +0.858]; 0.60675 | -0.798 [-2.257, +0.529]; 0.11105 |

Measured (`audit.json`): timestamped QB sources resolve both starters for all
272 regular-season 2025 games; 47 games differ from the recorded assignment,
49 of 544 team sides. The 2013-2024 sources have no verified pre-cutoff
observations, so their disagreement counts are unavailable, not zero.
Read (`src/nfl_ats/nfl_week.py:22`): the reused pool cutoff is the earlier of
kickoff and Sunday 16:00 Eastern; observations must be strictly earlier.
Opener is the settlement line, not an invented observation timestamp.

Measured (QB rotation JSONs): the corrected 2020-2021 QB columns are missing;
the model reproduces baseline picks exactly. Here `probability_positive=0`
counts strict improvements among identical bootstrap outcomes and is not
wrong-sign evidence. Inferred: the broader QB rerun includes the model's
missing-value handling as well as starter identity, so it does not isolate
starter mechanism reliability.

Measured (`*_paired.csv` and rerun JSONs): the following split-half diagnostic
correlates team-season flag exposure in odd versus even weeks; undefined means
no paired observations or a constant half. This is an exposure-stability
diagnostic, not repeated-observation reliability of a latent trait; that
reliability remains unverified and no `no_split_half_reliability` ground is
claimed.

| Screen | Oracle odd/even exposure r (measured) | Decision-time odd/even exposure r (measured) |
|---|---|---|
| Rookie debut, 2020-2021 | 0.04854 (n=64) | undefined (n=0) |
| QB revenge, 2020-2021 | -0.107041 (n=64) | undefined (n=0) |
| Dome favorite, 2020-2021 | 0.230558 (n=64) | 0.217102 (n=64) |
| September heat, 2020-2021 | -0.000210786 (n=64) | undefined (n=64) |
| QB revenge, 2020-2025 | -0.0635333 (n=192) | -4.73835e-20 (n=32) |
| QB + deadline, 2020-2025 | -0.0635333 (n=192) | -4.73835e-20 (n=32) |


Measured (`artifacts/cx8_posthoc_assignments/20260905/season_stability.json`): decision-time opener effects by season, accuracy points, non-push games:

- rookie_debut: 2020: +0.000 (n=220), 2021: +0.000 (n=236).
- qb_revenge: 2020: +0.000 (n=220), 2021: +0.000 (n=236), 2022: +0.000 (n=248), 2023: +0.000 (n=266), 2024: +0.000 (n=266), 2025: -3.745 (n=267).
- dome_shootout: 2020: +0.000 (n=220), 2021: -0.847 (n=236).
- sept_heat: 2020: -0.455 (n=220), 2021: -0.424 (n=236).
- both: 2020: +0.000 (n=220), 2021: +0.847 (n=236), 2022: +0.403 (n=248), 2023: +0.376 (n=266), 2024: -1.880 (n=266), 2025: -4.120 (n=267).
