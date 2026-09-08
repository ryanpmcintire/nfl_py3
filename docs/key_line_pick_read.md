# The key-line pick read (MOD-18 lane T served, 2026-09-08)

## The contract

Read: `AGENTS.md`, "Football margins are multimodal, not Gaussian" -- "Any
served cover probability ... is computed against the DISCRETE margin
distribution conditional on the line ... A smooth pooled-residual read ...
may run only as a CHALLENGER and must beat the discrete read on the opener
grade through the played card to be served."

Lane K measured that sentence on every game and lane T measured it where
it actually bites. Served everywhere, the discrete read LOSES the side
through the played card (lane K's MP1: -0.399 accuracy points,
`probability_positive` 0.311, `docs/mass_preserving_lattice.md`); served
only where the Tuesday line sits exactly ON a key atom it WINS it:

| Arm (lane T, `artifacts/research/laneT/cells.json`) | Atoms | Games touched | Through the played card | Standalone at the opener |
|---|---|---:|---|---|
| KL1 (`kl1_kl1_overall_card`) | 3, 7, 10, 14 | 307 of 1,537 | +0.133 pts, 95% [-0.396, +0.666], `probability_positive` 0.6456 | +0.133, 0.6162 |
| **KL1b** (`kl1_kl1b_overall_card`) -- **served** | **3, 7** | **272 of 1,537** | **+0.200 pts, 95% [-0.271, +0.726], `probability_positive` 0.7426** | +0.200, 0.688 |

Both clear lane T's predeclared 0.5 bar; KL1b is the stronger of the two
and the coordinator's expected-value decision (AGENTS.md, "A promotion bar
is not a decision bar": the pool is forced picks, and declining a card that
is 74% likely better is taking the other side of a 74/26 bet) serves it
from the 2026-09-08 Week 1 lock.

The mechanism, named (AGENTS.md, "No unexplained threshold flips on the
played card"): a discrete read differs from a smooth read at exactly one
place, the integer the football piles up on. When the quoted line SITS on
that atom, the atom's weight is the push and the cover / loss split either
side of it is decided by mass the smooth read spreads across a continuum,
so the discrete read is answering the question the football actually
asked. Between the atoms (3.5, 4.5, 6.5 ...) the push atom is empty, the
discrete read's only contribution is sampling noise, and the smooth read
was already calibrated. The boundary is the key-number set MOD-05 named and
AGENTS.md makes binding, never a spread value chosen because accuracy
dipped there.

So the served card is now split by LINE as well as by quantity:

1. **The pick and its two-way `home_cover_probability`** come from the
   discrete lattice on games whose line is exactly 3 or 7 (either sign; a
   half-point or quarter-point line such as 6.75 / 7.25 is never on the
   atom), and from the smooth `gaussian_median` read everywhere else. The
   served number on a touched game is lane T's decision number, bit for
   bit: `cover + push / 2` (`MassPreservingRead.home_cover_probability`,
   the frozen research convention; measured on lane T's `scored.parquet`,
   `p_MP1 == cover_MP1 + 0.5 * push_MP1` on all 1,537 rows, and the
   push-renormalised `cover / (cover + loss)` differs from it by up to
   0.018 -- serving that would not be the measured arm).
2. **The override is applied AFTER the served home-side offset**
   (`docs/home_side_offset_promotion.md`, S3): the point the atoms are
   tilted to is the offset-corrected centre plus the residual location --
   exactly the served point the discrete push read already tilts to, so
   the pick and the card's push chance can never disagree on the lattice.
3. **The push probability and every three-way answer** stay on the
   discrete read on every game (`docs/discrete_push_read.md`, unchanged).
4. **The smooth-everywhere two-way read is the paired challenger**,
   `key_line_pick_read_off_incumbent` (below), recorded from the sidecar
   verbatim.

Served flag: `KEY_LINE_PICK_READ_SERVED = True`, policy id
`key_line_pick_read_v1`, atoms `KEY_LINE_ATOMS = (3.0, 7.0)` -- all three
declared once in `src/nfl_ats/key_line_pick_read.py`; every caller imports
them. With the flag `False`, `margin-predict` serves the smooth two-way read
on every game again and writes no sidecar.

Applicability, and telling "did not apply" from "did not run"
(2026-09-08): `key_line_read_applicable(line)` is the named predicate the
served path skips on (the single-line form of lane T's `key_line_mask`),
`key_line_applicability(lines)` summarises it for a week, and the sidecar
and metadata block both carry a `status` plus an `applicability` object:

| `status` | Means | `applicability` | `error` | `fit` |
|---|---|---|---|---|
| `served` | the lattice was fitted and at least one served line sat on an atom | counts, `applicable: true`, `reason: null` | null | present |
| `inapplicable` | the lattice was fitted and the exact-match test matched no served line -- **the steady state on the owner's half-point pool** | counts, `applicable: false`, `reason` in words | null | present |
| `not_run` | the policy could not be built (flag off, or no lattice this week) | `null` -- there is nothing to be applicable about | why | null |

Both of the last two touch zero games; before this they were
indistinguishable from outside. Each game row also carries an
`inapplicable_reason` (`"half-point line, so the exact-match test on 3 and
7 cannot fire; the key-number mass lands wholly on one side here rather
than pushing (MOD-18 candidate C2)"`, or `"whole-number line, but not one
of the declared key numbers"`). `served` stays `true` in the
`inapplicable` case, so every downstream reader -- the refresh, the paired
challenger, the override helpers -- behaves exactly as it did.

Served-line provenance (2026-09-08):
`prediction_safety.validate_pool_lines` fails the served card **closed**
when any decision line is a whole number, because the pool posts only half
points and a whole number therefore proves the line came from the schedule
feed. It is called from `orchestrate_margin_predict` only. It is
deliberately NOT part of `validate_prediction_card` /
`validate_outcome_prediction_card`: the opener archive, every backtest and
every registry cell are graded on whole-number-capable lines and must keep
passing untouched. A fixture that quotes whole numbers on purpose (the ones
that exercise this module) opts out explicitly through
`POOL_QUOTES_HALF_POINT_LINES`; production never takes that path.

## Where it is wired

- `nfl_ats.key_line_pick_read`: `key_line_mask` (lane T's atom selection,
  reproduced), `key_line_decision_probability` (`cover + push / 2`),
  `KeyLinePickRead` (the week's walk-forward lattice reader -- the SAME
  reader the discrete push read fitted -- plus the atom set),
  `apply_key_line_pick_read` (the served override on a `MarginModel.predict`
  frame, only `home_cover_probability`, only on touched games),
  `apply_key_line_pick_read_to_sweep` (the policy at every alternative
  atom line in `line_sweep.parquet`, so the line-0 row equals the card),
  the sidecar / metadata writers and the per-game override helpers.
- `nfl_ats.outcomes.score_outcome_week(key_line_pick_read=...,
  key_line_pick_read_log=...)` and `score_outcome_week_line_sweep(
  key_line_pick_read=...)`: the served `market_residual` rows only, after
  the offset and the push split, before the decision columns; the push
  sidecar's two-way number is synchronised to the served one so the two
  records beside the card never disagree. `None` leaves every caller's
  output bit-for-bit unchanged.
- `nfl_ats.cli_commands.prediction`: `_served_key_line_pick_read` builds
  the policy from the week's discrete reader (never raises; with no lattice
  -- push read off, or its fit failed -- the smooth read serves every game
  and the sidecar says why); `margin-predict` writes
  `key_line_pick_read.json` beside the card (both reads per game, which
  games were touched, which sides changed, the point, atoms, band) and a
  `key_line_pick_read` block in `metadata.json` (provenance, never reader
  text: policy, atoms, every touched game with both reads,
  `sides_changed`), and labels the line sweep's `pick_read`. The
  offset-off arm (`home_side_offset.json`, the
  `home_side_offset_off_incumbent` challenger) keeps the key-line read at
  ITS OWN uncorrected point, so that pair differs by the offset alone and
  this pair differs by the key-line read alone -- one policy per ledger.
- **Every module that refits or replays the served card reproduces the
  served probability on touched games** through the per-game override
  (`served_pick_overrides` from the sidecar, `pick_overrides_from_metadata`
  from the metadata block, `load_pick_overrides` preferring the block),
  the shape `home_side_location.served_center_offsets` /
  `center_offsets_from_metadata` use; a pre-promotion card (no sidecar, no
  block) behaves exactly as today, pinned by tests:
  - the late-week refresh (`pick_refresh.plan_refresh`): the Tuesday
    sidecar's atoms are re-applied at the frozen line through the week's
    lattice rebuilt exactly as `margin-predict` built it, at the refit's
    own point, so new information moves a touched game the same way it
    moves every other game and with nothing new the served number
    reproduces bit for bit; if the lattice cannot be rebuilt the served
    probability is substituted verbatim (the policy degrades to "keep
    Tuesday's number", never to "silently drop it");
  - lane B's `card_refit.CardRefit.predict` (the own-arm refit challengers:
    `expected_lineup_loss`, `deadline_drag`,
    `qb_revenge_deadline_drag_stack`, `era_weighted_half_life_8`);
  - the spread explorer (`compute_spread_explorer_params` /
    `compute_spread_explorer_distribution`, `pick_overrides=`): a touched
    game's params are marked `key_line_pinned`, the payload carries the
    served number as `pinned`, and the build-time guard
    (`public_board.assert_spread_explorer_matches_card`) checks THAT
    number against the card at the quoted line;
  - the mapping-incumbent recorders (`ecdf_mapping_incumbent`,
    `gaussian_mean_mapping_incumbent`, `smooth_cdf_mapping`): the
    reproduction check expects the served number on a touched game; their
    own arms (the mapping on every game) are unchanged;
  - the board (`board_content._load_spread_explorer_params`,
    `public_board.build_public_site`): the cover curve's offset-zero point
    and the line-offset adjuster show the served number at the quoted
    line (`data-pinned-p`, one line of the widget script) and the smooth
    curve at every other line.
- `nfl_ats.card_explanation`: on a touched game the "Why this pick"
  paragraph says so in pool-player words (below); `publishing` passes the
  touched set read from the forecast's metadata.
- `nfl_ats.key_line_pick_read_incumbent_overlay`: the paired challenger
  recorder, wired into `publish-predictions --record-decisions`
  (`key_line_pick_read_off_incumbent_challenger_ledger`, fail-open) and
  `scripts/lockday_rehearsal.py`.

## The paired challenger

`key_line_pick_read_off_incumbent` records the smooth `gaussian_median`
two-way read's forced picks on EVERY game (offset included -- the pick
exactly as it was served before this promotion) from
`key_line_pick_read.json` verbatim, never a refit, beside the published-card
decision recorder in `publish-predictions --record-decisions` and in the
lock-day rehearsal. It refuses a card without a served sidecar and a
configuration-fingerprint drift, exactly like `home_side_offset_off_incumbent`.
On an untouched game the two arms pick identically by construction; the
2026 rows on the atoms settle the arm at no rotation-window cost.

### Registration (to apply in ONE step with the publish result-key entry)

The lock-day audit (`scripts/lockday_contract.py`, `tests/test_cli.py::
test_publish_challenger_result_map_covers_live_active_registry`) refuses a
publish result-key entry without its registration and a registration
without its entry, so the two edits below land together, after the Week 1
lock. Neither is in this patch (the registry is never edited by a lane).

1. Append to `artifacts/prospective/challengers.json` `challengers` (the
   `config_fingerprint` is the active recipe's, identical to
   `home_side_offset_off_incumbent`'s):

```json
{
  "challenger_id": "key_line_pick_read_off_incumbent",
  "status": "ACTIVE_PROSPECTIVE",
  "registered_at_utc": "2026-09-08T17:00:00+00:00",
  "model": {
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "feature_set": "full_weak_stack",
    "feature_table": "data/processed/game_features_weak_stack.parquet",
    "method": "market_residual",
    "min_edge": 0.02,
    "min_train_games": 500,
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "target": "market_residual"
  },
  "config_fingerprint": "bc77638d47e2748c",
  "reader_summary": "Tracks the forecast that reads every game the usual smooth way, so the picks on spreads set exactly on 3 or 7 can be compared with and without the key-number read on the same games.",
  "prospective_protocol": "Score forced-pick ATS accuracy against the recorded (decision) line, paired per game with the active model's own paper ledger (artifacts/clv_ledger/decisions.parquet) over the games both entrants recorded, via nfl-ats prospective-score (decision_line primary, close_line secondary). No promotion decision is implied by a partial season, and no rotation-registry window is spent or implied by this registration.",
  "known_gap": "Forced picks only; no paper bets. Historical evidence is a post-hoc restriction on a mined archive, not independent confirmation; the atom set is being graded out of sample by lane V.",
  "model_note": "Reads the smooth two-way probability the served forecast wrote beside its card (key_line_pick_read.json) for every game; never refits. Refuses a card without that sidecar or a recipe drift.",
  "weekly_recording_command": "nfl-ats publish-predictions --record-decisions",
  "weekly_generation_command": "Uses the synchronized active forecast.",
  "evidence": {
    "classification": "unresolved_below_power",
    "registry_source": "registry/weak_signals.json:mod18_conditional_margin_v1_kl1_kl1b_overall_card_2020_2025",
    "write_up": "docs/key_line_pick_read.md"
  }
}
```

2. Add to `PUBLISH_CHALLENGER_RESULT_KEYS` in
   `src/nfl_ats/cli_commands/publishing.py`, directly under the
   `home_side_offset_off_incumbent` line:

```python
    "key_line_pick_read_off_incumbent": "key_line_pick_read_off_incumbent_challenger_ledger",
```

The recorder call, its skipped-reason block, and the rehearsal bindings
are already in the patch; `scripts/lockday_rehearsal.py` reports the
recorder as an error (`find_challenger` refuses an unregistered id) until
step 1 lands, which is the fail-closed behaviour every recorder shares.

## What a reader sees change

- **Week 1 2026, one side (measured by lane T, `artifacts/research/laneT/
  week1.json`, replayed bit for bit by `tests/test_key_line_pick_read.py`):
  NO at DET.** The line sits on 7. The smooth read had Detroit at 0.5158 to
  cover; the lattice read has 0.4720, so the card says **NO +7** (about
  52.8 in 100) instead of **DET -7** (51.6). It survives the three-member
  card. DEN at KC (line 3) moves from 0.5178 to 0.5152 and keeps Kansas
  City; the other fourteen games are quoted off the atoms and do not move
  at all.
- On NO at DET's deep dive the "Why this pick" paragraph reads, in place of
  the push sentence: "The line sits right on 7, a number games land on a
  lot, so this pick is read off how games with lines like this actually
  finished, and the roughly 4 in 100 that ended exactly there, a push, are
  counted in that chance." DEN at KC gets the same sentence for 3, with the
  card's own push chance at 3 (inferred, not yet measured on a served card:
  the archive average at a line of 3 is 9.1 in 100, lane K's
  `push_calibration.json`).
- The cover curve's quoted-line point and the line-offset adjuster's
  starting number are the card's own (52.8% for NO); dragging the slider
  off the quoted line shows the smooth curve. Inferred, not measured: the
  "Flips at" cell on NO at DET should therefore read the very next
  half-point (DET at -6.5) -- off the number the smooth read still leans
  Detroit, and the pick holds because the line is ON the number. That is
  the served policy stated as a flip line, not a disagreement; the first
  `publish-board` after the lock measures it.
- Nothing else moves: not the push chance, not the three-way split, not
  any game off 3 or 7.

## Scope: the atom-equality test cannot fire on the pool's lines, and the key-number mass matters MORE there (2026-09-08)

**Read this whole section before concluding anything about scope.** It is
the place a future session is most likely to draw the wrong conclusion.

Measured 2026-09-08 on the owner's own pool
(`data/splash/2026_week01_20260908_noon.json`, all sixteen Week 1 games of
the Splash Sports contest this card is played into): **every line the pool
posts is a half point** -- 3.5, 3.5, -2.5, 8.5, 6.5, 3.5, 1.5, 3.5, -3.5,
-1.5, 1.5, 9.5, 3.5, 5.5, -2.5, 2.5 -- and **nine of the sixteen sit within
half a point of 3** (six at |3.5|, three at |2.5|). `KEY_LINE_ATOMS` is
tested by exact equality, so it can never match one of those lines and the
override never fires on the served card.

Until 2026-09-08 the served decision line came from nflverse
`schedules.spread_line`, which does carry whole numbers, and the read
therefore fired on lines the pool does not offer. Measured on the card
regenerated that afternoon
(`artifacts/margin_predictions/2026-week-01-20260908T162923Z/key_line_pick_read.json`):
six games were treated as sitting on an atom (ATL_PIT 3.0, CHI_CAR -3.0,
DAL_NYG -3.0, DEN_KC 3.0, NE_SEA 3.0, NO_DET 7.0) and **three of them
changed sides** (CHI_CAR, DAL_NYG, NO_DET) -- while the pool quoted those
same six at 3.5, -2.5, -2.5, 2.5, 3.5 and 6.5.

### This is a limitation of the exact-match TEST, not of the mechanism

Measured this session on 4,431 completed regular-season games
(`data/processed/game_features.parquet`, 2009-2025): the empirical
home-cover rate at a neutral point, against a normal fitted to the same
games.

| Line | Empirical cover | Gaussian | Gaussian error |
|---:|---:|---:|---|
| 2.5 | 50.85% | 48.71% | **2.14 points TOO LOW** |
| 3.5 | 43.04% | 45.99% | 2.95 points too high |
| 6.5 | 35.32% | 37.98% | 2.66 points too high |
| 7.5 | 30.92% | 35.40% | 4.49 points too high |
| 10.5 | 24.42% | 28.11% | 3.69 points too high |

Crossing the 3 atom from 2.5 to 3.5 the empirical cover chance falls
**7.81 points**; the smooth read says 2.72. It understates the cliff by
**2.87x**, and it errs in **opposite directions** on the two sides of the
atom -- at 2.5 the two reads straddle 0.5 outright, i.e. they pick opposite
sides of a neutral game.

The reason is mechanical: **14.58% of finals land exactly on |3|**. On a
whole-number line that mass is absorbed by the push; on a half-point line
the entire block falls on **one** side of the number. So the discrete
conditional distribution is *more* decisive at the pool's lines than at the
whole numbers this module was measured on, not less.

Precisely, and none of it closes anything:

- **The multimodal premise is untouched and still correct** (AGENTS.md,
  "Football margins are multimodal, not Gaussian"). The table above is
  additional evidence *for* it at half-point lines. Nothing in this
  document may be cited as refuting it.
- **No discrete read is switched off.** The served three-way split
  (`docs/discrete_push_read.md`,
  `mass_preserving_lattice.serve_discrete_three_way`) keeps serving on
  every game and carries the correct conditional mass, which is exactly
  what a half-point line needs.
- **Nothing here closes lane T.** Every lane T cell stays
  `unresolved_below_power` at `probability_positive` 0.7426 through the
  played card. The read remains correct machinery for a whole-number line,
  a registered paired challenger, and the way the 1,537-game archive --
  graded on whole-number-capable lines, where the push is real -- is read.

### TODO, predeclared: generalise the side read to every half-point line (MOD-18 candidate C2)

**Not implemented, not measured, and owned by another lane.** Stated here
with its numbers so the next session finds the direction rather than
finding a gate and concluding the idea was tried and dropped.

The direction: **price the served cover probability off the discrete
conditional margin distribution at EVERY line the pool posts, not only
where the line equals an atom.** The lattice
(`mass_preserving_lattice.band_read`) already answers at any line -- the
served three-way split calls it on every game today -- so the change is to
the SIDE read, not to the construction. At a half-point line `push` is zero
by construction and `cover` already carries the whole key-number block on
its correct side; the candidate two-way number is that `cover` (there is no
push to halve), against the smooth `gaussian_median` read as the paired
incumbent.

What the grading has to answer, declared before the signs are seen:

1. Forced-pick ATS accuracy at the **opener** through the played card, on
   the archive, paired per game against the smooth read -- the same
   protocol lanes K and T used, with the same
   `unresolved_below_power` default and `probability_positive` reported.
2. Whether the gain concentrates in the |2.5| / |3.5| band the table above
   predicts (nine of this week's sixteen games), or is spread across the
   line range. The mechanism predicts the former; a gain that is flat
   across lines is a different effect wearing this one's name.
3. The post-hoc discount this module already pays (below) applies to the
   generalisation as well: the 2.5/3.5 asymmetry was seen on the same
   archive before the arm was declared.

An interval that crosses zero is not a result here. Only a resolved wrong
sign or a positive-control bound closes it; everything else is
`unresolved_below_power`, recorded with `nfl-ats weak-signals record`.

## Known limitations, disclosed

- **The atom set is post hoc.** Lane T's predeclaration says so
  (`docs/key_line_lattice.md`): the restriction was chosen after seeing
  lane K's bucket-7 gain on the same 1,537-game archive lanes K and S were
  selected on; S3 is itself a post-hoc restriction fitted on it; the two
  arms are nested and correlated with lane K's MP1 and lane H's M1. Lane V
  is grading the atom set out of sample (declare {3, 7} on 2020-2023,
  grade on 2024-2026); until then this read pays that discount, and the
  2026 rows on the atoms are the evidence that settles it at no window
  cost. Every cell remains `unresolved_below_power`; nothing here closes
  anything.
- Both losses are fractionally worse on the archive (KL1b Brier
  -0.0000141, `probability_positive` 0.477; log loss -0.0000230, 0.483):
  the arm buys side selection at the 7 atom (+8.571 standalone,
  `probability_positive` 0.934) and pays a little calibration at the 3
  atom (Brier -0.000649, 0.285). Per season through the card: 2020 +0.455,
  2021 no graded pick moved, 2022 -0.403, 2023 -0.376, 2024 +2.256
  [+0.377, +4.044], 2025 -0.749.
- The lattice read at a hypothetical line that is ALSO an atom (line 3
  when the card is on 7) is not carried into the browser widget: the
  slider shows the smooth curve off the quoted line. `line_sweep.parquet`
  does carry the policy at every alternative atom line, and
  `scripts/cover_odds.py`'s discrete three-way already reads the lattice
  at any line.
- The key-line read needs the week's lattice: when the discrete push read
  is off or its fit degrades, the smooth read serves every game and the
  sidecar records `served: false` with the reason; the paired challenger
  then refuses to record (nothing paired to record).

## Verification

```powershell
.\.tools\uv.exe run --no-sync pytest tests/test_key_line_pick_read.py tests/test_discrete_push_read.py tests/test_home_side_offset_promotion.py tests/test_spread_explorer.py tests/test_board_content.py tests/test_pick_refresh.py tests/test_card_explanation.py tests/test_ecdf_mapping_incumbent_overlay.py tests/test_gaussian_mean_mapping_incumbent_overlay.py tests/test_smooth_cdf_mapping_overlay.py tests/test_lockday_contract.py tests/test_deadline_drag_challenger.py tests/test_expected_lineup_loss_challenger.py tests/test_era_weighted_half_life_8_overlay.py tests/test_outcomes.py tests/test_cli.py tests/test_public_board.py -n 2
.\.tools\uv.exe run --no-sync ruff format --check .
.\.tools\uv.exe run --no-sync ruff check .
.\.tools\uv.exe run --no-sync mypy src
```

`tests/test_key_line_pick_read.py::test_lane_t_kl1b_replays_bit_for_bit_through_the_served_read`
replays lane T's `scored.parquet` through the served module (272 touched
games, `p_KL1b` and `touched_KL1b` equal to the recorded arrays with no
tolerance) and pins the Week 1 side change from `week1.json`; it skips
where the research artifacts are absent.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`. Nothing in this document closes anything: every
lane T cell stays `unresolved_below_power`, and the served read is an
expected-value decision on the played card.
