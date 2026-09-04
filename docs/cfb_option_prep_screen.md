# CFB triple-option prep-deficit screen (LEAD-45): predeclaration

Written **before any ATS outcome, cover rate, accuracy delta or sign is
computed for this flag**. Sections 1–8 are the predeclaration. Section 9
will be appended after the look and will report what was found; it will
change nothing above it.

This is a **cross-league screen**, not a new NFL look. It spends **no NFL
evaluation window and no rotation window** — CFB is this project's
sanctioned free replication ground. It is also genuinely new: no
sandwich, lookahead, option, or service-academy screen exists anywhere in
`docs/` (measured 2026-09-04 by full-text search), and the CFB rest/bye
replication (`docs/cfb_rest_bye_replication.md`) tested rest thresholds,
never scheme-prep asymmetry.

## Closing-grounds taxonomy (binding, restated verbatim per AGENTS.md)

An interval or CI that contains zero is **NEVER** grounds to reject, fail,
or close an experiment. At this evaluator's ~2-point resolution,
"contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) **refuted mechanism** — a RESOLVED
wrong sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) **bounded by a positive control** — the instrument was
PROVEN able to detect an effect that size and it was absent. Everything
else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record --league cfb`, report
`probability_positive`, never the binary "contains zero". If a record
command errors, the verdict is wrong, not the validator. A promotion
threshold governs only what the docs may CLAIM; it never governs which
card is PLAYED, which is expected value.

## 1. Mechanism and predeclared direction

Defending a triple-option offense requires a week of practice against cut
blocks, mesh reads, and assignment football that cannot be replicated by
scout-team looks — a preparation deficit that lands entirely on the
non-option opponent. **Predeclared direction: BACK the option-team side**
(the opponent underperforms the market by more than the talent gap
implies). This is a prep-asymmetry mechanism, not a talent claim about
service academies.

## 2. Population

The XLG-03 clean core (2012–2019, 2021–2025 FBS-vs-FBS, ~8,933 games).
Option-team identity: Army, Navy, Air Force in all seasons, plus Georgia
Tech 2008–2018 (Paul Johnson era). Measured 2026-09-04 on
`data/processed/cfb_game_features.parquet`: 534 option-involved games
2006–2025 (~29/season); the clean-core subset is whatever of those fall
in the scored seasons — counted by the script before scoring, never
adjusted after.

## 3. Encoding (frozen)

One signed column, `cfb_option_side`: +1 when the HOME team is the sole
option team, −1 when the AWAY team is, 0 otherwise. Option-vs-option
games (Army–Navy and kin: no prep asymmetry exists) encode 0 and stay in
the population as baseline rows — predeclared, not a post-hoc exclusion.

## 4. Comparator

The frozen XLG-03 benchmark arm (35 columns,
`nfl_ats.cfb_benchmark.fit_cfb_residual_model`, ridge alpha 10.0) vs the
same arm plus `cfb_option_side` — one column, isolating the flag's
marginal contribution against everything the benchmark already explains.
Walk-forward: every scored week's models train on completed table games
kicking off strictly before that week's earliest kickoff, 500-game floor.

## 5. Metric and uncertainty

Paired candidate-minus-baseline forced-pick accuracy delta in
`accuracy_points`, picks at `home_cover_probability >= 0.5`, graded with
`nfl_ats.clv.pick_correct`. Week-blocked bootstrap primary (1,000
samples, seed 20260904), season-blocked secondary, never averaged.

## 6. Controls

- **Positive (leak) control:** the flag column planted with the realized
  `ats_margin` must read hugely positive (P+ 1.000); if it does not, the
  harness is blind and nothing below is trusted.
- **Null:** 200 within-week permutations of the flag; the observed delta
  is reported at its null percentile. A null centered far from zero is a
  home-tilt artifact to disclose (per `docs/graph_ratings_v2_screen.md`
  §6), not a defect.

## 7. Reliability

The flag is a deterministic identity-plus-calendar fact (team names ×
season), with zero measurement error — the same argument
`docs/cfb_rest_bye_replication.md` §5 freezes for calendar facts. No
split-half reliability applies, so `no_split_half_reliability` is
inadmissible for this family either way.

## 8. Decision rule and recording

Expected value, never a threshold. One registry entry
(`cfb_option_side_on_benchmark`, `unresolved_below_power` unless an
admissible ground fires), with the five required disclosures. A
close-graded CFB number settles no NFL play/no-play decision; a positive
lean earns an NFL transfer predeclaration, nothing more.

## 9. Results (appended after the look, 2026-09-04)

Coverage (predictor-only, `--mode coverage`): 9,093 clean-core games,
**405 flagged** (sole option side, ~31/season every season 2012–2025),
39 option-vs-option encoded 0 and kept.

Positive control (`--mode positive-control`): **+48.405 accuracy
points, week-blocked 95% [+47.340, +49.466], P+ 1.000** on 8,933 games /
199 weeks — the instrument sees effects; the harness is not blind. Null
centers +0.746 (home-tilt artifact of the leak arm, disclosed per §6).

Screen (`--mode screen`, artifact
`artifacts/cfb_option_prep/20260904T120205Z/results.json`):

| cut | delta (pts) | week 95% | P+ | n |
|---|---|---|---|---|
| pooled | **-0.034** | [-0.330, +0.270] | **0.381** | 8,933 / 199 |
| era 2012–2019 | +0.000 | [-0.298, +0.314] | 0.489 | — |
| era 2021–2025 | -0.084 | [-0.624, +0.507] | 0.348 | — |

Null centers **+0.023**, observed at the **31st percentile** — clean, no
tilt to discount. Era splits agree with each other (both within ±0.09
of zero): no regime disagreement to carry, unlike the rest/bye battery.

**Decision implication, before caveats:** the predeclared direction
(back the option side) leans wrong at P+ 0.381, but the week interval
contains zero and spans only ±0.3 points — no admissible closing ground
applies (not wholly below zero; deterministic flag, so reliability
cannot refute). Recorded `cfb_option_side_on_benchmark`
(`unresolved_below_power`, registry now 693). No NFL transfer is earned:
a negative-leaning CFB column is not a green light to spend an NFL
window, and the effect is an order of magnitude below the instrument's
own resolving power. LEAD-45's column version is answered-unresolved;
the prep-deficit mechanism stays open only behind a genuinely different
instrument (e.g. LEAD-29-style first-meeting splits, untested).
