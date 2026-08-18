# Player availability and value: inventory, overlap audit, and the 2026 confirmation

Written 2026-08-18. The pool-edge plan names this family as the one place a
measured lean survives the "team quality is already priced" ceiling
(`docs/pool_edge_plan.md`, build filter item 2), and `ROADMAP.md`'s RWB-16 row
records the reason: "The one family where it accumulates is **player
availability and value** — five measurements, all positive, p = 0.0625 within
the family."

This document does three things. It rebuilds that inventory from artifacts
rather than from prose. It audits the independence the p-value assumes. And it
records what the 2026 prospective confirmation actually is, having checked that
it is wired up.

**The one-line verdict: the accumulation is real but much weaker than 0.0625,
and the family splits in two — the value-weighted-injury half passes a mechanism
test that the learned-availability-rate half fails.**

---

## 0. What was run, and why none of it cost a window

Three new scripts, all reading data that has already been looked at:

| script | what it does | window cost |
|---|---|---|
| `scripts/availability_ablation.py` | reproduces the MOD-07 look and splits it into its availability half and its bias half | none — attribution on the already-spent [2020, 2021] block, which `docs/pool_edge_plan.md` states explicitly is free |
| `scripts/availability_overlap_audit.py` | game overlap, error correlation, and a dependence-aware sign test across the whole family | none — re-reads existing artifacts, fits nothing |
| `scripts/availability_mechanism_screen.py` | asks whether the MOD-07 availability advantage sits where its mechanism says it must | none — re-reads the same spent window |

No `rotation assign`, no `rotation record`, no NFL confirmation window scored.
The rotation ledger is byte-identical to how this session found it.

Artifacts: `artifacts/availability_experiments/mod07_ablation_2020_2021.json`,
`overlap_audit.json`, `mechanism_screen.json`, plus the three per-game paired
parquets beside the ablation JSON.

---

## 1. The inventory

### 1.1 First: the headline number had no artifact

`probability_positive` **0.899** is quoted in four places — `ROADMAP.md:459`,
`HANDOFF.md:80`, `docs/pool_edge_plan.md:133` and `:182`, and
`docs/play_level_audit.md:178` — as the single measured justification for the
project's current strategic direction. **No artifact on disk recorded it.**
`artifacts/mod07_weak_stack/opener_2020_2021.json` holds only the two-arm
headline (+1.97, P+ 0.8745); the ablation that produced 0.899 was never written
down.

It has now been reproduced. `scripts/availability_ablation.py` rebuilds the
registry's forward-chained split for the spent window, scores three arms, and
**asserts** that the two recorded ones come back before reporting the third:

| check | result |
|---|---|
| paired games = 456 | ✅ |
| weeks = 35 | ✅ |
| baseline accuracy = 0.5131578947368421 | ✅ exact |
| candidate accuracy = 0.5328947368421053 | ✅ exact |
| delta = +1.9737 points | ✅ exact |
| `probability_positive` = 0.8745 | ✅ exact |

The three arms differ in a way worth stating precisely, because it is what
makes §3 possible. `player` (79 columns) and `player_value` (81 columns) differ
by **exactly two columns** — `diff_injury_skill_epa_value_lost` and
`diff_injury_defense_disruption_value_lost` — plus the fact that the
`weak_stack` table carries **learned** availability semantics in the shared
`*_injury_*_unavailability` and `*_qb_start_probability` columns where the
`player` table carries hand-authored fixed priors. `weak_stack` (90 columns)
adds only the nine bias columns on top.

| arm | profile / table | opener accuracy |
|---|---|---|
| A | `player` on `game_features_player.parquet` | 51.32% |
| B | `player_value` on `game_features_weak_stack.parquet` | 53.07% |
| C | `weak_stack` on `game_features_weak_stack.parquet` | 53.29% |

| contrast | delta | week-blocked 95% | `probability_positive` | picks split |
|---|---|---|---|---|
| C − A (the recorded look) | **+1.97 pts** | [−1.10, +5.00] | 0.8745 | 49 |
| **B − A (availability + value)** | **+1.75 pts** | [−0.69, +4.27] | **0.899** | 34 |
| C − B (the three opener biases) | +0.22 pts | [−2.66, +3.24] | 0.505 | 39 |

**A correction to the prose.** `docs/pool_edge_plan.md:182` says the
availability half "carried the whole +1.97". It carried **+1.75 of the +1.97**;
the bias half carried the remaining +0.22. The qualitative claim (the biases
contributed essentially nothing, availability carried it) is right; the
arithmetic in the sentence is not. See §5 for the exact patch.

### 1.2 Every measurement in the family

Nine measurements exist with a direction. All of them are recomputed in
`artifacts/availability_experiments/overlap_audit.json` from the prediction
artifacts, not copied from prose.

| id | measurement | effect | games | seasons | sign |
|---|---|---|---|---|---|
| M1 | `base → player_injuries` (fixed injury priors) | +0.19 pts | 2,075 | 2018–2025 | + |
| M2 | `base → player_injury_value` (value-weighted injuries) | +0.39 pts | 2,075 | 2018–2025 | + |
| M3 | `base → player_value` (player + value composite) | +1.06 pts | 2,075 | 2018–2025 | + |
| M4 | `fixed → learned` availability, ATS | +0.10 pts | 2,075 | 2018–2025 | + |
| M5 | `fixed → learned` availability, **player-level target** | Brier 0.09500 → 0.09056 | 57,294 player-games | rates target 2014–2026 | + |
| M6 | `player_value → participation` RAPM | **−0.43 pts** | 2,075 | 2018–2025 | **−** |
| M7 | MOD-07 availability half, **at the opener** | +1.75 pts | 456 | 2020–2021 | + |
| M8 | `base → player` (the frozen active profile) | +0.96 pts | 2,075 | 2018–2025 | + |
| M9 | `base → player_injuries_continuity` | +0.77 pts | 2,075 | 2018–2025 | + |
| M10 | `base → player_qb_injuries` | **−0.43 pts** | 2,075 | 2018–2025 | **−** |

Every figure above is recomputed from the prediction parquets, not copied.
The recomputation uses the evaluator's own tie convention — a cover probability
of exactly 0.5 picks HOME (`>= 0.5`, not `> 0.5`) — and was checked against all
eleven recorded `paired_comparisons.csv` estimates before anything here was
reported: **all eleven reproduce to 1e-12.** The convention matters: the wrong
one moves M6 from −0.43 to −0.63 points.

Interval sources: M1–M4, M6, M8–M10 are the week-blocked accuracy rows in
`artifacts/player_experiments/20260813T122348Z/paired_comparisons.csv`,
`artifacts/availability_experiments/20260813T133345Z/paired_comparisons.csv`,
and `artifacts/participation_experiments/20260813T132030Z/paired_comparisons.csv`
— every one of them contains zero. M7's is [−0.69, +4.27]. M5 is the only one
on a different target and it has no interval recorded at all.

Only M4 and M7 grade a change to **availability semantics**. M1, M2, M3, M8, M9
grade *adding injury/value columns at all*; M6 and M10 are the negatives.

**The five RWB-16 counted are not recorded anywhere.** The exact set is
recoverable only by which combination reproduces p = 0.0625, and that is
{M1, M2, M3, M4, M7} — five positives, no negatives. That reconstruction is
used below, but note what it required: the boundary was drawn to exclude M6,
and M6 sits in the same experiment sequence (its own baseline arm *is*
`player_value`), was built as a player-**value** rating, and is the direct
subject of ROADMAP's PER-05 line. Excluding it is a choice, and it was made
after the signs were known.

---

## 2. The overlap audit

### 2.1 The games are the same games

| pair | shared games | share of the smaller set |
|---|---|---|
| any two of the 2018–2025 experiments | 2,075 | **100%** |
| any 2018–2025 experiment vs the MOD-07 window | 453 | **99.3%** |

Every measurement in the family except M5 is scored on the identical 2,075-game
2018–2025 set, and the MOD-07 opener window's 456 games are 453 the same games
again, re-graded at a different line. There is **no independent football** in
this family at all. M5 is the only measurement on different data, and it is on a
different *unit* (player-games) and is **causally upstream** of M4 — better
learned rates produce the features M4 grades — so it is M4's mechanism check,
not a second draw.

### 2.2 But the errors are less correlated than that implies

Sharing games does not automatically mean sharing errors: two different feature
sets flip different picks. Measured on the full 2,075-game set, the per-game
paired-correctness differences correlate like this:

|  | M1 | M2 | M3 | M8 | M9 | M10 | M4 | M6 |
|---|---|---|---|---|---|---|---|---|
| M1 | 1.00 | 0.68 | 0.40 | 0.43 | 0.45 | 0.71 | −0.02 | −0.02 |
| M2 | 0.68 | 1.00 | 0.48 | 0.40 | 0.45 | 0.57 | −0.04 | −0.05 |
| M3 | 0.40 | 0.48 | 1.00 | **0.87** | **0.80** | 0.48 | −0.23 | −0.26 |
| M8 | 0.43 | 0.40 | **0.87** | 1.00 | **0.84** | 0.52 | −0.06 | −0.12 |
| M9 | 0.45 | 0.45 | **0.80** | **0.84** | 1.00 | 0.42 | −0.04 | −0.10 |
| M10 | 0.71 | 0.57 | 0.48 | 0.52 | 0.42 | 1.00 | 0.00 | −0.04 |
| M4 | −0.02 | −0.04 | −0.23 | −0.06 | −0.04 | 0.00 | 1.00 | 0.17 |
| M6 | −0.02 | −0.05 | −0.26 | −0.12 | −0.10 | −0.04 | 0.17 | 1.00 |

The nested contrasts are heavily correlated (M3/M8 0.87, M8/M9 0.84, M3/M9
0.80, M1/M10 0.71, M1/M2 0.68) — as they must be, since they share the `base`
arm and add overlapping column sets. M4, the one measurement that changes
availability *semantics* rather than adding columns, is nearly orthogonal to all
of them, and mildly *anti*-correlated with M3.

Cheverud/Nyholt effective count across all eight: **6.65 of 8**. Across the
reconstructed five: **4.76 of 5**.

### 2.3 The dependence-aware sign test

`M_eff` rounds away the fractional part, so it is too coarse for a five-way
test. The direct measurement is better: centre every contrast's per-game
differences so the true effect is exactly zero by construction, then resample
whole weeks **once per draw** and recompute all five on that same resample. The
fraction of draws where all five land the same sign is the honest analogue of
the sign test's 2 × 0.5⁵.

| quantity | value |
|---|---|
| independent assumption (what 0.0625 asserts) | 0.0625 |
| **measured under the real dependence structure** | **0.0978** |
| inflation factor | **×1.56** |
| draws | 4,000, seed 20260818 |

So the shared-games objection is real but costs less than intuition suggests:
p = 0.0625 should be read as **p ≈ 0.10**.

### 2.4 The family boundary is where the p-value actually goes

| boundary | positives | nominal two-sided p | dependence-adjusted |
|---|---|---|---|
| the reconstructed five {M1, M2, M3, M4, M7} | 5 / 5 | 0.0625 | **0.098** |
| + the same-kind negative M6 (participation RAPM) | 5 / 6 | **0.219** | — |
| every injury-or-value contrast (M1–M4, M6–M10) | 7 / 9 | **0.180** | — |

Any boundary wide enough to admit the extra positives (M8, M9) must also admit
the negatives (M6, M10). There is no defensible definition of "player
availability and value" that keeps `base → player` and drops
`player_value → participation`.

### 2.5 Verdict on the accumulation

**The accumulation does not survive the overlap audit as a p = 0.0625 finding.**
It survives as a **directional lean somewhere in p ≈ 0.10–0.22**, on a boundary
that was never predeclared and was drawn after the signs were visible.

By this repository's own taxonomy (`docs/pool_edge_plan.md`, "Three kinds of
negative"), that is **category 3 — unresolved below detection power**. Which is
the correct place for it: keep it, stack it, confirm it prospectively, and stop
citing 0.0625 as though it were a result. The lean is still the strongest thing
in the project, because everything else is a coin flip; but "strongest available"
and "established" are different claims, and only the first is supported.

---

## 3. The mechanism screen — the one genuinely new result

The +1.75 points cannot be confirmed by size on 456 games; its interval crosses
zero and always will at that sample. It **can** be checked for mechanism, and a
mechanism statistic has far more power than an accuracy statistic, because it
uses all 456 games and a continuous covariate instead of a binary win/loss.

The two arms differ in exactly two ways, so there are exactly two axes to test,
declared by the arms' structure rather than chosen:

1. **`semantics_shift`** — how far the learned availability rates moved the
   inputs the baseline also reads (standardised L2 over the nine shared
   `diff_injury_*_unavailability` / `diff_qb_*` columns);
2. **`value_magnitude`** — how much injury value was lost, from the two columns
   the candidate feature set adds.

If the effect is the availability mechanism working, the advantage must
concentrate where those quantities are large, and games where the two tables
agree to rounding must contribute nothing.

### 3.1 Stratified accuracy (tercile, equal counts)

| axis | low third | middle | **high third** |
|---|---|---|---|
| `semantics_shift` | **+2.63 pts** (P+ 0.862) | +1.32 (P+ 0.651) | +1.32 (P+ 0.670) |
| `value_magnitude` | −0.66 (P+ 0.290) | +0.66 (P+ 0.573) | **+5.26 pts (P+ 0.977)** |

`value_magnitude` is monotone increasing. `semantics_shift` runs *backwards* —
the advantage is largest where the learned rates changed the inputs least.

**A placebo was run before either was believed**, because tercile machinery on
456 games swings on its own. Stratifying by axes with no availability content
gives |spread|: +3.29 / −0.66 / +2.63; total: −0.66 / +3.29 / +2.63; week:
+1.97 / +0.66 / +2.63. All non-monotone, all of comparable size, and all
recorded in `mechanism_screen.json` under `placebo_strata` so this table can
never be read without them. **So the tercile spread alone is not evidence** —
only the monotonicity distinguishes `value_magnitude`, and monotonicity over
three bins is one bit.

### 3.2 The disagreement statistic, which does resolve

Every point of the delta lives in the 34 games where the arms picked different
sides; an agreed pick contributes exactly zero. So the sharp question is whether
disagreement is predicted by input movement at all. Rank-biserial correlation
from Mann-Whitney U, with the exact null variance (N+1)/(3·n₁·n₂):

| axis | r | 95% interval | z | two-sided p |
|---|---|---|---|---|
| `semantics_shift` | **−0.024** | [−0.226, +0.178] | −0.23 | 0.816 |
| `value_magnitude` | **+0.248** | [+0.046, +0.450] | +2.40 | **0.016** |

Two axes were tested; Bonferroni gives 0.033, still resolved.

`value_magnitude` is nearly uncorrelated with the placebo axes (Pearson +0.134
with |spread|, −0.020 with total, +0.044 with week), so this is not a proxy for
game shape or season timing. It is also only +0.155 with `semantics_shift`,
which is why the two axes can disagree at all.

### 3.3 What that means

**The family splits, and the two halves have opposite mechanism evidence.**

- **The learned-availability-rate half fails its own mechanism check.** The arms
  split on games essentially at random with respect to how much the learned
  rates moved (r = −0.024, and the interval excludes the value axis's +0.248).
  Whatever produced M4's +0.10 points and part of M7's +1.75, it is not "we now
  know who will play". This is consistent with M4 being the weakest measurement
  in the whole inventory (+0.10 points, the smallest of the five).
- **The value-weighted-injury half passes.** The advantage concentrates
  resolvably in the games where a lot of player value is missing (r = +0.248,
  p = 0.016), and the accuracy gradient across that axis is monotone with the
  top third at +5.26 points.

This is a **hypothesis, not a promotion.** It was found by looking at a spent
window after the fact, on 456 games, and post-hoc stratification is exactly how
selection on noise manufactures stories. What makes it worth carrying is that
the disagreement statistic was not tuned, had only two candidate axes fixed by
the arms' structure, and resolved.

It does, however, sharpen what 2026 should watch: **the pool-relevant signal
looks like injury *value lost*, not availability *probability*.**

---

## 4. The 2026 confirmation

### 4.1 It is registered, and the registration works

Verified against the live tree, read-only (nothing was recorded):

| claim | status |
|---|---|
| `mod07_weak_signal_stack` registered `ACTIVE_PROSPECTIVE` | ✅ `artifacts/prospective/challengers.json` |
| registered fingerprint = fingerprint recomputed from its own `model` block | ✅ both `bc77638d47e2748c` |
| a real week-1 card exists at that fingerprint | ✅ `artifacts/margin_predictions/2026-week-01-20260817T215341Z`, profile `weak_stack` |
| no other 2026 card shares the fingerprint | ✅ the eight others are `base`/`player` |
| ledger reset after the August rehearsal | ✅ `challenger_decisions.parquet` absent, 0 rows |
| `prospective-score` runs clean | ✅ 0 decisions, both entrants, both grades |
| weekly pipeline produces and records it | ✅ `weekly-run` steps 8–11, all `optional` |

So the handoff's claim that confirmation "costs no window" is **correct**, and
the plumbing behind it is real rather than aspirational.

### 4.2 Three gaps, one of them serious

**(a) The registration is git-ignored.** `.gitignore:26` (`artifacts/**`)
matches `artifacts/prospective/challengers.json`. That file is the only record
of which challenger is frozen for 2026, of its `config_fingerprint` — the entire
anti-retune guarantee described in `docs/prospective_evidence.md` — and of its
frozen protocol. It exists on one machine and nowhere else.

This is worse than a backup problem. The value of a frozen challenger is the
**pre-commitment**, and in this repository pre-commitment is proved by git
history: `registry/rotation_registry.json` and `registry/weak_signals.json` are
both tracked precisely so the history of a declaration is the git history. The
challenger registry is the one declaration file left outside it, and it is the
one whose entire evidentiary weight rests on having been fixed before Week 1.
**Fix this before 2026-09-08.** Patch in §5.

**(b) The decision grade is a Tuesday-ish line, not the archived Tuesday
opener.** `prospective_scoring.record_challenger_decisions` stores
`decision_home_spread` from the card's `spread_line`, which comes from the
feature table (nflverse schedules) at Tuesday build time. That is the right
*kind* of grade — a pre-lock number, settled at the line the pick was made at —
but it is a different instrument from `clv.opener_pick_evaluation`'s archived
`tue_open` consensus, which is what the 52.50% headline and the whole MOD-07
window were measured against, and different again from Splash's own posted
number. The 2026 figure will therefore not be directly comparable to 52.50%
without saying so. Worth a sentence in the write-up when the first number lands;
not worth re-plumbing before Week 1.

**(c) A silently skipped week leaves no alarm.** Steps 8–11 are `optional` by
design, so a failed challenger build loses that week's evidence while the pool
card still ships — correct priority. But nothing reports "the challenger has
fewer recorded weeks than the active model". With ~18 weeks of evidence total,
losing three unnoticed is a third of the season. Suggested check in §5.

### 4.3 What the 2026 look actually settles, and what it does not

At 456 games the MOD-07 window could not resolve +1.75 points. A full 2026
regular season is **272 games** — *smaller*. One season of prospective evidence
will not confirm this family either, and nobody should expect it to.

What it does buy is the one thing history cannot: a look with **no multiplicity
discount at all**. The ~130–150-stream ledger discounts every 2018–2025 number;
2026 is clean. It also pairs per game against the active model, and the arms
already disagree on 3 of 16 Week-1 games, so the paired quantity accumulates
faster than the raw accuracy.

The honest framing for the season: **2026 is one clean bit, not a confirmation.**
Combined with the 456-game window it is two looks in the same direction on
independent football — which, given §2's finding that this family has had zero
independent football to date, is a genuine improvement in kind rather than in
size.

---

## 5. Exact next step, and the patches this document cannot apply

### Highest value, do first: track the challenger registry

`.gitignore` and the registry file are outside this document's edit scope. The
change is one negated pattern:

```gitignore
# after line 26 (`artifacts/**`)
!artifacts/prospective/
!artifacts/prospective/challengers.json
```

then `git add -f artifacts/prospective/challengers.json` once, and commit it
**before 2026-09-08** so the freeze has a timestamped, tamper-evident record.
Nothing else under `artifacts/**` should be un-ignored; this is one declaration
file, and it holds no data, no credentials, and no fitted model.

### Correct the +1.97 attribution

`docs/pool_edge_plan.md` line 182, currently:

> player-value/availability half carried the whole +1.97 (P+ 0.899).

replace with:

> player-value/availability half carried **+1.75 of the +1.97** (P+ 0.899;
> week-blocked [−0.69, +4.27]), the bias half the remaining +0.22
> (`artifacts/availability_experiments/mod07_ablation_2020_2021.json`).

### Downgrade the 0.0625 claim

`ROADMAP.md` RWB-16 row, currently:

> The one family where it accumulates is **player availability and value** --
> five measurements, all positive, p = 0.0625 within the family.

replace with:

> The one family where it accumulates is **player availability and value**, but
> the accumulation is weaker than first recorded (`docs/availability_confirmation.md`).
> The five measurements were never named; the set that reproduces p = 0.0625 is
> {M1, M2, M3, M4, M7}, and it excludes a same-kind negative (participation
> RAPM, −0.43 points) that any principled boundary admits — 5/6 gives p = 0.219,
> and the broadest injury-or-value boundary gives 7/9, p = 0.180. All five sit on
> the SAME 2,075 games (the opener window is 453 of them again), so the sign
> test's independence assumption is measured rather than assumed: a
> dependence-preserving centred bootstrap puts the honest figure at **p ≈ 0.098**,
> not 0.0625. Category 3, not a finding.

The same 0.899-with-no-artifact citation appears at `HANDOFF.md:80`,
`docs/pool_edge_plan.md:133` and `docs/play_level_audit.md:178`; each should
gain the artifact path
`artifacts/availability_experiments/mod07_ablation_2020_2021.json`.

### Add a challenger-coverage check

In `prospective-score`'s metadata, alongside the per-entrant blocks, report
`weeks_recorded` per entrant and flag any entrant whose recorded week count
trails the active model's. Cheap, and it converts gap (c) from silent to loud.

### Do NOT spend a window

Specifically: **do not spend [2022, 2023]** — the next eligible opener block,
and one of only two left in the project — on this family. `docs/mod07_stack.md`
already ruled that out as iterating-until-it-wins, and nothing here changes it:
the mechanism screen is suggestive, not confirmatory, and it was found post hoc
on the spent window, which is exactly the kind of evidence a second window
should not be spent chasing.

If a window is ever spent here, the candidate should be the **narrowed** one
this document's §3 points at — value-weighted injury magnitude, without the
learned-rate half that failed its mechanism check — and it should be a NEW
family declaration inheriting `mod07_weak_signal_stack`, predeclared with its
threshold, after 2026 has produced its own number. Not before.

### Free work that remains, and one path that is closed

**XLG-07 (cross-league availability semantics) is not feasible, and this is
already settled by a documented data audit, not by power.**
`docs/cfb_data.md` § "Availability semantics (fail closed)": the
`espn_cfb_injuries` release has zero assets, CFBD v5 has no injuries endpoint,
game rosters are scrape-time listings (zero of 27,471 players changed their
Active/Inactive flag across all of 2024, and those columns are quarantined out
of the canonical table), and play participants are credited actors — absence of
credit is never evidence of absence. There is no pregame CFB availability signal
to screen. The only feasible CFB proxy, realized participation continuity, was
already predeclared and scored: it did **not** clear the XLG-03 benchmark
(−0.67 points on 8,933 clean-core games, `docs/cfb_role_features.md`). So the
CFB side of this family has had its one look and lost it.

What is still free and unspent: everything in §3 can be extended on the same
window at no cost — in particular, whether the `value_magnitude` gradient
reproduces on the 2018–2025 close-graded set (M2/M3's own 2,075 games), which
would be a second, larger-sample mechanism check of the surviving half without
touching an opener window. That is the natural next run and it was scoped, not
executed, here.
