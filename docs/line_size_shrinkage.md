# Line-size shrinkage of the served residual (MOD-18 lane AK follow-on)

Acts on the diagnosis in `docs/big_spread_signal.md`, which named these arms and
did not run them. Sibling context: `docs/spread_hole_diagnosis.md` (lane F),
`docs/spread_hole_target.md` (lane N).

Closing-grounds taxonomy, verbatim, because this document reports intervals:
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

Within-week correlation is ZERO by owner mandate; every bootstrap here is
week-blocked, with the season-blocked reading alongside. Football margins are
discrete and multimodal, never Gaussian. **No arm here is a bucket rule or a
threshold flip**: every arm is a continuous, monotone, walk-forward-fitted
function of `|line|` that applies at every line, and at its null parameter
values it reproduces the served model bit-for-bit. Decide on expected value.

## Design, frozen 20260910T023340Z before any candidate was scored

Family `mod18_line_size_shrinkage_v1`.

### The defect being acted on

Two measured facts from `docs/big_spread_signal.md`, both read, not re-derived
here:

1. At `|line| >= 10.5` the assembled residual is **50.77%** right while its own
   `results` block alone is 55.38%, `elo` alone 56.15%, and the market's own
   opener-to-close move 62.96% (read: `docs/big_spread_signal.md:262-269`,
   `:431`). The assembled residual carries less about the cover, as the line
   grows, than the parts that feed it.
2. The served `gaussian_median` read puts `P(home cover) > 0.5` exactly when
   `predicted_residual + median(training residuals) > 0` (read:
   `src/nfl_ats/calibration.py:291-292`), so the served rule shifts every
   game's decision boundary by ONE global location constant fitted on all
   lines. That constant is worth **+1.00** accuracy points overall and
   **-6.15** at 10.5+ (read: `docs/big_spread_signal.md:476-496`). One number
   applied identically to a 1-point line and a 17-point line.

The mechanism both arms encode: **the information the assembled residual
carries about the cover, and the location correction it needs, are both
functions of the line's size, and the served model treats them as constants.**

### The served decision statistic, stated exactly

For one game with opener line `L_signed`, this week's fitted ridge residual
`r`, this week's served S3 home-side offset `o`
(`nfl_ats.home_side_location.fit_home_side_offsets`, fitted on the arm's OWN
prior out-of-time stream, zeroed outside the buckets `7`, `7.5-10`, `10.5+`),
and this week's out-of-time residual sample with median `m` and standard
deviation `s` (ddof=1):

```
P(home cover) = Phi( (r + o + m) / s )
```

which is `gaussian_median` written out. The served forced pick is home iff that
probability is >= 0.5, i.e. iff `r + o + m >= 0`. This identity is checked
against the harness's own `home_cover_probability_at_open` and the check is
reported as a number before any arm is scored.

### AK-1 (primary) — line-size shrinkage of the residual

```
k(L) = 2 / (1 + exp(a + b * L)),      L = |L_signed|
P(home cover) = Phi( (k(L) * r + o + m) / s )
```

Two parameters. At `(a, b) = (0, 0)`, `k(L) = 1` for every L and the arm
reproduces the served model exactly — that is the declared initialisation and
the identity check. `k` is continuous and, for `b > 0`, monotone
non-increasing in the line's size; it applies at every line; there is no
threshold anywhere in it. The served point becomes `L_signed + k(L) * r + o`,
so the arm's own S3 offset stream is refitted from its own point, exactly as
`scripts/spread_hole_arms.py` does for every candidate arm.

### AK-1b (secondary) — AK-1 plus a line-size location constant

The same shrinkage, with the `gaussian_median` location constant `m` also made
a function of the line's size, in the same functional form:

```
g(L) = 2 / (1 + exp(c + d * L))
P(home cover) = Phi( (k(L) * r + o + m * g(L)) / s )
```

Four parameters. At `(a, b, c, d) = (0, 0, 0, 0)` this is the served model
exactly. `m` is still this week's own out-of-time residual median; `g` only
scales how much of it a game of that line size receives.

### The fit, and the window it may see

`(a, b)` — and `(c, d)` for AK-1b — are fitted **once per scored week**, by
maximising the Bernoulli cover log-likelihood

```
sum_i [ y_i * log p_i + (1 - y_i) * log(1 - p_i) ]
```

over **only games completed strictly before that week's first kickoff**, where
`y_i = 1` if the home side covered the line that row was scored at and pushes
are excluded. Optimiser: `scipy.optimize.minimize`, method `L-BFGS-B`, started
at the null vector, box `[-20, 20]` on every parameter, `maxiter` 500. No grid
search, no restart from a second point, no re-fit after seeing any sign.

**The fit corpus is a strictly-prior out-of-time stream, declared here in
full.** Two segments, concatenated:

- **Seed, 2011-2019, close proxy.** A walk-forward on the schedule table's own
  `spread_line`: each week's ridge is fitted on completed regular-season games
  strictly before that week's first kickoff (minimum 500 training games, so the
  first scored week is the first that clears the floor), scored on that week's
  games at the schedule line, with the S3 offset accumulated from the same
  stream. This segment is never scored as a result; it exists only so the first
  scored opener week has a real corpus instead of running at the null.
- **Live, 2020-2025, opener.** The arm's own accumulated opener stream, one
  week at a time, as it is produced.

Minimum fit corpus: **500** completed rows. Below that the parameters are held
at the null, i.e. the arm IS the served model for those games. This is stated as
a handicap, not hidden: those games contribute an exact zero to every paired
delta.

The seed segment is a different line surface from the scored surface, and that
is a stated approximation, not a claim of equivalence. It is strictly prior
information at every scored week, so it cannot leak.

**The seed feeds the likelihood corpus ONLY.** The S3 home-side offset stream
stays opener-only, exactly as served — `prior_rows_before` looks back five
seasons, so letting 2015-2019 rows into the offset stream would silently change
the incumbent's own 2020 offsets and break the identity check. The seed
segment carries its own internal offset accumulation (2011-2019 close proxy,
same fitter) so its rows are constructed the same way as the rows they help
fit; that accumulation is never read by the scored window.

### Scoring

Population: the Tuesday opener archive
`artifacts/opener_evaluation/20260910T005854Z` (active model
`2e8c616b476dd0d2`, weak_stack market-residual ridge, alpha 10,
`gaussian_median`), **1,537 archived games 2020-2025, 1,503 non-push at the
opener**. The replay is `scripts/spread_hole_arms.py`'s own walk-forward, and
the reproduction gap against the archive is reported before any arm is read.

Two surfaces, both reported:

- **standalone** — the arm's own opener pick against the served model's own
  opener pick, no card.
- **card** — both arms put through the served nine-member card with
  `scripts/spread_hole_arms.py --card served`, so the composition (every
  overlay that conditions on the model's probability) is **recomputed on the
  candidate**, not inherited.

Paired accuracy-point deltas on the games both arms score, week-blocked AND
season-blocked bootstraps, 20,000 samples, seed 20260821, using
`nfl_ats.overlay_composition.blocked_bootstrap_matrix` — the project's own
resampler. Reported overall and by bucket `0-6.5`, `7`, `7.5-10`, `10.5+`
(`nfl_ats.spread_regime.spread_bucket`, coarsened as
`scripts/spread_hole_arms.py` coarsens it). Alongside every cell: Brier score,
log loss, picks changed, and the fitted `k` curve per season at
`|line| = 1, 3, 7, 10, 14` so the mechanism is visible rather than asserted.

**Positive control.** The perfect-foresight bound at 10.5+ from the diagnosis:
following the opener-to-close move is **62.96%** on the 108 big-spread games
that moved (read: `docs/big_spread_signal.md:431`), against the served model's
50.77% — a **+12.19**-point headroom that the instrument demonstrably resolves
(that cell's own week-blocked interval is `[+3.70, +22.17]`,
`probability_positive` 0.9967). A candidate at 10.5+ whose interval excludes
that magnitude is bounded by a control; a candidate whose interval merely
contains zero is not.

### AK-2 — the late-week follow's threshold as a function of the line

Runs **only if** `data/market/raw`'s 2023-2025 `intraday_hourly` archive
supports the replay that `docs/follow_threshold_live_card.md` used. The served
rule fires when `|leader_median_net_move| >= 0.5` regardless of line size; the
arm makes that threshold

```
t(L) = t0 * 2 / (1 + exp(e + f * L))
```

with `t0 = 0.5` the served constant, fitted the same way (walk-forward, cover
log-likelihood over strictly-prior rows, null start reproducing the served
rule). If the archive does not support it, this document says so and the arm is
skipped rather than approximated.

### Recording

Every cell is recorded with `nfl-ats weak-signals record`, units
`accuracy_points`, league `nfl`, family `mod18_line_size_shrinkage_v1`, named
`line_size_shrinkage_<arm>_<card|standalone>_<bucket>_2020_2025`, one command at
a time, argv saved to `record_commands.json` in the artifact directory.

**Terminal classification, declared before any cell was scored.** A cell is
`unresolved_below_power` unless exactly one of:

- `wrong_sign_resolved` — BOTH the week-blocked and the season-blocked 95%
  interval sit entirely below zero.
- `positive_control_bound` — the cell's 10.5+ interval excludes the
  +12.19-point control magnitude named above AND the control cell itself
  resolved (it did, at `probability_positive` 0.9967).

Nothing else closes a cell. An interval containing zero closes nothing.

### What this lane may and may not do

It may name the exact promotion path and the Week 1 picks that would change. It
may **not** promote, publish, or touch any ledger, manifest, forecast, board or
active-model file. Week 1 pick effects are written read-only to the scratchpad.

## Results

Every number below is **measured this session** by the commands at the bottom;
the tables live in `artifacts/line_size_shrinkage/20260910T023340Z/`. Nothing
above this line was edited after scoring, apart from the one predeclaration
correction named in "Two corrections" below, which is itself dated and stated.

### The replay is exact

The `served` arm is not an approximation of the archive, it is the archive.
Joined game for game against
`artifacts/opener_evaluation/20260910T005854Z/per_game.parquet` on all **1,537**
games: maximum residual gap **0.0**, maximum home-side-offset gap **0.0**,
maximum probability gap **1.1e-16**, pick agreement **1.000**, and the served
replay's own accuracy is **54.5576%**, identical to the archive's
`correct_at_open_probability_rule`. So `k = 1` really does reproduce the served
model, and every delta below is the shrinkage and nothing else.

### Two corrections to the design, both found by measurement, both stated

**1. The predeclared optimiser was wrong, and it was wrong in a way that
invented a threshold.** L-BFGS-B started at the null walked to the box corner
`(a, b) = (10.47, 20.0)` and reported convergence — the projected gradient is
zero on the boundary — which is `k = 0` at every line: the model stops picking
and the location constant alone decides every game. Profiling the SAME
predeclared likelihood on the same 1,503 opener rows shows the true optimum is
interior at `(0.20, 0.30)`, negative log-likelihood **1040.01** against the
corner's **1041.91** and the null's **1047.02** — a real maximum that L-BFGS-B
never saw. The fix is numerical, not a change of objective: a deterministic
multi-start (the null plus the four best points of a declared
9x7 grid over `a` and `b`, each polished by Nelder-Mead, lowest
negative log-likelihood wins). The corner run is kept in full at
`optimiser_corner_run/` and the null-start Nelder-Mead run at
`lbfgs_null_start_run/`; the results below are the multi-start run. **A
gradient optimiser that stops on a bound is how a continuous shrinkage turns
into an unexplained threshold flip by accident**, which is exactly what the
owner's ban names, so it is reported rather than quietly re-run.

**2. The predeclared seed corpus pulls the fit the wrong way, and the
measurement says why.** On the 2011-2019 close-proxy seed the served ridge's
residual has essentially NO sign accuracy — **50.31%** on 2,236 non-push games,
against **52.96%** at the opener — because the schedule table's line is close to
the closing line, where the model has almost no edge. Fitted on the seed alone
the likelihood optimum is `k(1) = 0.014` rising to `k(14) = 0.667`: the
**opposite** slope to the one the diagnosis predicted. Fitted on the opener
alone it is `k(1) = 0.755` falling to `k(14) = 0.024`, the predicted shape. The
seed is 2,236 of the 3,739 corpus rows, so it wins. Because the predeclaration
named the opener-only corpus as the alternative it was considering, both are
reported as declared arms — `ak1` / `ak1b` (seeded, primary as declared) and
`ak1_opener_corpus` / `ak1b_opener_corpus` (opener-only, secondary).

### The fitted k curve, which is the mechanism made visible

Mean fitted `k(|line|)` per season, walk-forward, the opener-only corpus (the
seeded arms' curves are in `k_curve_ak1.csv` / `k_curve_ak1b.csv`):

| season | fit rows | k(1) | k(3) | k(7) | k(10) | k(14) |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 208 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2021 | 443 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2022 | 690 | 1.538 | 0.785 | 0.190 | 0.169 | 0.167 |
| 2023 | 954 | 1.589 | 0.560 | 0.034 | 0.004 | 0.000 |
| 2024 | 1,221 | 1.283 | 0.613 | 0.087 | 0.019 | 0.003 |
| 2025 | 1,487 | 0.946 | 0.606 | 0.186 | 0.068 | 0.017 |

(2020 and the first 15 weeks of 2022 run at the null because the corpus is
below the 500-row floor; those games are the served model exactly and
contribute an exact zero to every paired delta.)

**The diagnosis' prediction is confirmed almost to the number.** AK-1 predicted
"`k` near 1 below 7, and near 0 above 10.5". Measured, on data the fit had not
seen: `k` is **0.95 to 1.59 at a 1-point line**, **0.56 to 0.79 at 3**, and
**0.000 to 0.19 above 7**. The model's own residual really does carry less about
the cover as the line grows, and a two-parameter logistic fitted only on prior
games finds that, every season, without being told where to look.

The seeded arms find the opposite slope (`k(1)` 0.03-0.14 rising to `k(14)`
0.15-0.72), for the reason in correction 2.

And the AK-1b location curve `g(|line|)` says the same thing about the second
defect: fitted on the opener corpus it is **1.29 to 1.45 at a 1-point line** and
**0.009 to 0.20 at 14**. The one global location constant the served rule adds
to every game is, by this fit, worth MORE than its face value on a short line
and almost nothing on a long one.

### The decision cell: through the played card, at the opener

All four arms, paired against the served card on the same 1,503 non-push games,
week-blocked and season-blocked bootstraps, 20,000 samples, seed 20260821. The
served card scores **56.89%**.

| arm | card accuracy | picks changed | delta | 95% week | week P+ | season P+ |
|---|---:|---:|---:|---|---:|---:|
| ak1b (seeded) | 56.55% | 117 | **-0.333** | [-1.694, +1.010] | **0.3214** | 0.2959 |
| ak1b_opener_corpus | 56.49% | 62 | **-0.399** | [-1.420, +0.661] | **0.2280** | 0.0423 |
| ak1_opener_corpus | 56.09% | 66 | -0.798 | [-1.754, +0.133] | 0.0470 | 0.0077 |
| ak1 (seeded) | 55.95% | 176 | -0.932 | [-2.512, +0.608] | 0.1229 | 0.1553 |

Standalone (no card), against the served model's own 54.56%:

| arm | accuracy | picks changed | delta | 95% week | week P+ | season P+ |
|---|---:|---:|---:|---|---:|---:|
| ak1_opener_corpus | 53.56% | 145 | -0.998 | [-2.791, +0.744] | 0.1361 | 0.0378 |
| ak1b_opener_corpus | 53.49% | 140 | -1.065 | [-2.817, +0.660] | 0.1150 | 0.0924 |
| ak1b (seeded) | 53.16% | 253 | -1.397 | [-3.409, +0.656] | 0.0908 | 0.0059 |
| ak1 (seeded) | 50.83% | 396 | **-3.726** | [-5.970, -1.472] | 0.0006 | 0.0000 |

### By bucket, on the card, where the lane was aimed

| arm | 0-6.5 (1,104) | 7 (74) | 7.5-10 (194) | 10.5+ (131) |
|---|---|---|---|---|
| ak1 | -0.815, P+ 0.218 | -6.757, P+ 0.056 | -1.031, P+ 0.301 | **+1.527, P+ 0.795** |
| ak1b | -0.272, P+ 0.382 | +2.703, P+ 0.721 | -2.577, P+ 0.096 | +0.763, P+ 0.649 |
| ak1_opener_corpus | -0.543, P+ 0.186 | -8.108, P+ 0.032 | -0.516, P+ 0.381 | **+0.763, P+ 0.715** |
| ak1b_opener_corpus | -0.453, P+ 0.212 | +1.351, P+ 0.625 | -1.031, P+ 0.326 | 0.000, P+ 0.501 |

Standalone at 10.5+ (n=131, served 47.33%): ak1 47.33% (+0.00, P+ 0.497), ak1b
50.38% (+3.05, P+ 0.792), ak1_opener_corpus 50.38% (+3.05, P+ 0.860),
ak1b_opener_corpus 51.91% (**+4.58**, P+ 0.925).

**A guard that belongs next to those four numbers rather than under them.** Home
teams covered **57.69%** of the 130 non-push 10.5+ games in this window. As `k`
falls toward zero at a big line the residual stops deciding and the S3 home-side
offset does, so the arms' home-pick share at 10.5+ rises from the served
**59.23%** to **60.77 / 66.92 / 70.00 / 73.08%**. The best of them reaches
51.91% while simply picking home every time would have reached 57.69%. The 10.5+
gain is therefore a partial, inefficient recovery of a home tilt this window
happens to carry, not evidence that shrinkage found the football. Recorded, open,
and not believed.

### What the shrinkage unambiguously does buy: the probability

Standalone, all lines, 1,503 games — Brier and log loss both improve for every
arm, and they improve most exactly where the diagnosis said the residual was
least informative:

| cell | served Brier | ak1 | ak1b | ak1_op | ak1b_op |
|---|---:|---:|---:|---:|---:|
| all lines | 0.25158 | **0.25013** | 0.25025 | 0.25114 | 0.25084 |
| 7.5-10 | 0.26030 | 0.25278 | **0.25163** | 0.25732 | 0.25667 |
| 10.5+ | 0.26130 | 0.25622 | 0.25452 | 0.25482 | **0.25347** |

Log loss moves the same way (all lines 0.69662 -> 0.69342 for ak1; 10.5+ 0.71625
-> 0.70028 for ak1b_opener_corpus). This is the honest reading of the whole
lane: **maximum likelihood asks for the shrinkage and gets a better-calibrated
number; the forced pick does not want it.** Profiled directly on the served
stream, a constant `k` maximises the log-likelihood at about **0.25**
(-0.69260 per game, against -0.69662 at `k = 1`) while forced-pick accuracy
rises monotonically over the same range — 51.50% at `k = 0`, 52.56% at 0.25,
53.56% at 0.5, 54.56% at 1. The two objectives point in opposite directions, and
the predeclaration chose the one that does not decide the card.

### The positive control

Following the market's own opener-to-close move, graded at the opener, measured
this session on the same replay:

| scope | moved games | zero-move excluded | accuracy | pts | 95% week | week P+ | served model on the same games |
|---|---:|---:|---:|---:|---|---:|---:|
| all lines | 1,133 | 370 | 55.08% | +5.08 | [+2.26, +7.89] | 0.9999 | 54.56% |
| **10.5+** | **108** | 22 | **62.96%** | **+12.96** | [+3.70, +22.17] | **0.9967** | **47.69%** |
| 7.5-10 | 157 | 37 | 53.50% | +3.50 | [-4.12, +11.07] | 0.8203 | 51.55% |

This reproduces `docs/big_spread_signal.md:431` exactly on the current archive,
including the split that lane found: at 10.5+ the served model is **59.21%**
right on the 76 games where it agrees with the move and **28.13%** on the 32
where it fights it. So the instrument resolves a +13-point effect on 108 games.
**That is a bound on the SIZE of what shrinkage did at 10.5+, and it is stated
as one — none of the 10.5+ cells is closed on it**, because they lean positive
and a magnitude bound is not evidence that a small effect is absent. Declining
to use a closing ground the predeclaration would have permitted is a deliberate
deviation in the keep-open direction, and it is named here so it can be audited.

### AK-2: the late-week follow's threshold as a function of the line

The 2023-2025 `intraday_hourly` archive supports it — **816** games, every one
with leader-book evidence, 522 firing at the served 0.5 — so the arm ran. Its
line-size threshold `t(|line|) = 0.5 * 2/(1+exp(e+f|line|))` is chosen each week
on a declared deterministic grid. **Stated deviation:** a hard switching rule has
no probability, so the predeclared cover log-likelihood is undefined for it; the
fit criterion is prior-window forced-pick accuracy, on the grid, ties to the
null. The 500-row floor means the arm is the served rule until late 2024.

Through the played card, at the opener, on 799 scoreable games:

| comparison | scope | baseline | candidate | delta | 95% week | week P+ | season P+ |
|---|---|---:|---:|---:|---|---:|---:|
| **AK-2 vs the served follow** | overall | 55.32% | **56.20%** | **+0.876** | [-0.882, +2.801] | **0.8256** | 0.8520 |
| AK-2 vs the served follow | 0-6.5 | 55.68% | 57.00% | +1.318 | [-0.663, +3.437] | 0.9010 | 0.8520 |
| AK-2 vs the served follow | 10.5+ | 56.90% | 55.17% | -1.724 | [-5.556, +0.000] | 0.1816 | 0.1480 |
| AK-2 vs no follow at all | overall | 56.07% | 56.20% | +0.125 | [-3.175, +3.549] | 0.5262 | 0.6314 |
| the served follow vs no follow | overall | 56.07% | 55.32% | -0.751 | [-4.172, +2.753] | 0.3315 | 0.2087 |

That last row reproduces `docs/follow_vs_tilts.md`'s **-0.751 at
probability_positive 0.332** to three decimals on an independent rebuild, which
is a useful check that this lane's card is the same card.

**AK-2 is the only arm in this lane that leans positive through the card**:
+0.876 accuracy points over the served constant-0.5 rule, `probability_positive`
0.8256 week-blocked and 0.8520 season-blocked. Two honest guards: the fitted
threshold is unstable across the three seasons it has (2024's fit follows almost
everything above a 7-point line, 2025's follows almost nothing below a 1-point
move), and 10.5+ is the one bucket where it loses — the opposite of what the
diagnosis expected. It is a lead, on 799 games with a live fit for about 420 of
them, not a promotion.

### Decision

**The served card has the higher expected opener accuracy, and it is not close
enough to be worth the churn.** Best candidate through the card is `ak1b` at
**-0.333** accuracy points, `probability_positive` **0.3214** — a 32% chance it
is better, which under "a promotion bar is not a decision bar" is still the
wrong side of the bet, because declining it here is taking the 68/32 side, not
the cautious side. `ak1b_opener_corpus` is -0.399 at 0.2280;
`ak1_opener_corpus` -0.798 at 0.0470; `ak1` -0.932 at 0.1229. **Play the served
card.**

**Two things this lane found that are worth more than its arms.**

1. **The mechanism is real and now measured.** Fitted walk-forward on prior
   games only, `k` is ~1 at a 1-point line and ~0.03-0.19 above 7, every season,
   on the opener corpus. The diagnosis' claim that the assembled residual's
   information about the cover falls with line size is confirmed out of sample.
   What does not follow is that acting on it helps the pick: shrinking the
   residual hands the decision to the location constant and the S3 offset, and
   those are worse deciders than the residual is at 0-6.5, where three quarters
   of the card lives.
2. **The right lever is the probability, not the side.** Every arm improves
   Brier and log loss at every big-spread bucket while losing accuracy. A
   follow-up lane that wants this shrinkage should predeclare it as a
   CALIBRATION change scored on Brier/log loss with the SIDE held at the served
   rule — that separates the two effects this lane confounded, and it is the
   change the board's push, flip-line and alternative-line answers would
   actually feel.

**If a candidate were promoted, this is exactly what it would take** (stated so
the follow-up lane does not have to rediscover it; nothing here was done):

- `src/nfl_ats/margin.py` — `MarginModel.predict` already takes `center_offset`,
  so AK-1 needs a sibling multiplicative hook on `predicted_residual` before the
  offset is added (the point, the fair spread, the market residual and every
  probability must move together, as `center_offset`'s docstring requires), plus
  the fitted `k` carried on the model.
- `src/nfl_ats/calibration.py` — AK-1b additionally needs `gaussian_median`'s
  location to accept a per-row scale rather than the single
  `median(self.residuals)` at line 292.
- A walk-forward fitter beside `nfl_ats.home_side_location`, which is the exact
  precedent: fitted on the prior out-of-time stream, served per week, with its
  own artifact so the curve is auditable.
- `nfl_ats.outcomes.score_outcome_week` (the sole production weekly-forecast
  entry point), then `weekly-run --record-decisions`, then the opener evaluation,
  the overlay composition re-selection, and `publish-board` /
  `publish-predictions` — because AGENTS.md requires every headline number to be
  recomputed against the active model before the site is regenerated.

**Which 2026 Week 1 picks would change** (read-only; computed from the published
Week 1 artifact's own residuals and offsets, with each arm's last fitted
parameters). This week's residual median is **+0.36906** and the residual scale
**12.6465**, recovered exactly from the artifact's own probabilities (max probit
residual 2.8e-16):

| arm | picks changed | games |
|---|---:|---|
| ak1 (seeded) | 6 | ATL at PIT, BAL at IND, MIA at LV, NE at SEA, NYJ at TEN, SF at LA |
| ak1b (seeded) | 1 | NE at SEA |
| ak1_opener_corpus | 2 | ARI at LAC, NE at SEA |
| ak1b_opener_corpus | 2 | ARI at LAC, NE at SEA |

**CLE at JAX does not change under any arm**, which confirms the diagnosis'
reading of it rather than its worry: its residual is -0.117, so the served pick
of JAX is carried by the +0.369 location constant, and shrinking the residual
makes that pick *more* JAX (probability 0.5327 served, 0.5356 / 0.5251 / 0.5358 /
0.5256 under the four arms). The fragile Week 1 game under shrinkage is **ARI at
LAC**: its residual is the largest on the card at -5.755, but the opener-corpus
arms shrink a 9.5-point line by `k = 0.11-0.13`, which is enough to let the
+0.784 home-side offset and the location constant carry it over to LAC (0.3580
served, 0.5160 / 0.5019 candidate).

### Registry

All **58** cells are recorded under family `mod18_line_size_shrinkage_v1`, named
`line_size_shrinkage_<arm>_<surface>_<bucket>_<window>`, run one at a time
against `registry/weak_signals.json` (now **5,016** signals, **58** carrying this
lane's prefix). The exact argv lists are `record_commands.json` in the artifact
directory and the outcomes are `registry_records.json`.

**Two** are terminal, both `refuted_mechanism` on `wrong_sign_resolved`, and
both are the seeded AK-1 arm scored standalone, where the week-blocked AND the
season-blocked interval sit entirely below zero:

- `line_size_shrinkage_ak1_standalone_overall_2020_2025` — -3.726 accuracy
  points, week [-5.970, -1.472], season [-5.939, -1.628], n=1,503.
- `line_size_shrinkage_ak1_standalone_0_6p5_2020_2025` — -4.167 accuracy points,
  week [-7.247, -1.122], season [-6.271, -2.138], n=1,104.

Those close the seeded two-parameter arm's stated direction on the raw surface.
They do not close AK-1's mechanism: the same arm on the opener corpus is
unresolved, the card surface is unresolved for every arm, and 10.5+ leans
positive throughout.

The other **56** are `unresolved_below_power` and report
`probability_positive`, including every favourable cell — AK-2 at +0.876
(P+ 0.8256), the four 10.5+ card cells, and the positive-control cells at
P+ 0.9999 and 0.9967. A favourable interval is not a closing ground either.

## Commands run

```
python stage1_streams.py            # 107 opener weeks + 153 seed weeks of served ridge refits
python stage2_arms.py               # 5 arm replays, served card recomposition, bootstraps
python stage3_control_ak2_week1.py  # positive control, AK-2, 2026 Week 1 read
python stage4_record.py             # 58 weak-signals record commands, one at a time

nfl-ats weak-signals record ...     (58 cells, mod18_line_size_shrinkage_v1;
the exact argv lists are saved as record_commands.json in the artifact directory
and were run one at a time with NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry)
```

All four scripts are copied into
`artifacts/line_size_shrinkage/20260910T023340Z/` beside their outputs.

