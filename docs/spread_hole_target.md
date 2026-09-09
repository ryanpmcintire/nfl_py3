# Constraining the line column and changing the target (MOD-18 lane N)

Follow-on to `docs/spread_hole_diagnosis.md` and `docs/spread_hole_group_penalty.md`.
Lane M ends by naming this arm: *"an intervention that leaves `spread_line` free
cannot remove the lean, because the lean is exactly what a free line column is
able to express — so the next arm should either constrain the line column's
coefficient too, or change the TARGET the ridge is trained on rather than the
parameterisation."* (read:
`docs/spread_hole_group_penalty.md:259-263`, in the lane-M worktree
`.claude/worktrees/agent-a179c21adb158c3cb`, which is where that document
currently lives — it is not yet in the main checkout).

Closing-grounds taxonomy, verbatim, because this document reports intervals:
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

## Design, frozen 2026-09-09T23:10:47Z before any candidate was scored

Family `mod18_spread_hole_target_v1`. Four arms. No tuning after scoring, no
subset search, no threshold rule, and no pick flip in T1, T2 or T3 — those
three are model-level changes applied to every game at every line. T4 is a
positive control and is explicitly NOT a candidate to play.

### What lanes F and M established, and what is left

- The served weak_stack ridge is 51.5% right on 7.5-10 and 47.3% on 10.5+ as
  served (read: `docs/spread_hole_diagnosis.md:23-28`).
- The favourite-ward lean survives orthogonalising all 47 team-quality columns
  against the line (arm G3), because the ridge rebuilds it out of `spread_line`
  itself: the `market` family's mean contribution toward the favourite at 10.5+
  flips from **-0.785 to +0.954** points while the net residual toward the
  favourite barely moves, +0.442 to +0.417 (read:
  `docs/spread_hole_group_penalty.md:218-243`).
- It also survives a 10x group penalty on the block (arm G2): the `results`
  lean at 10.5+ moves only +4.200 to +3.832 (read:
  `docs/spread_hole_group_penalty.md:254-257`).

So the remaining degrees of freedom are the line column itself and the target.
This lane takes both.

### The team-quality block, unchanged from lane M

Every column of the served `weak_stack` profile whose declared family
(`nfl_ats.margin.FEATURE_FAMILIES`) is `results`, `elo`, `offense` or `defense`
— 47 of the 90 columns. The list is enumerated in
`docs/spread_hole_group_penalty.md:34-54` and is recomputed, not retyped, by
this lane's script.

### Arm T1 — G3 plus the line column deleted

Two changes on top of the served configuration:

1. **Orthogonalise all 47 team-quality columns to the line** (this is exactly
   lane M's G3): each column is replaced by its residual after an ordinary
   least-squares regression of that column on the game's own `spread_line`,
   with slope and intercept fitted **only on the training rows of that week's
   walk-forward** and applied to the scoring rows at the opener line. No
   outcome enters the regression, so it is pregame-only.
2. **Remove `spread_line` from the feature matrix entirely.** The line then
   enters the model only as the offset the residual is measured against — the
   target is `market_residual` (`ats_margin` = final margin minus the line), and
   the served point is `line + predicted residual`. Implementation: the script
   registers a runtime feature set `full_weak_stack_no_line` = the served
   `full_weak_stack` set minus `spread_line`, and a matching profile
   `weak_stack_no_line`. Nothing in `src/` is edited; the registration is a
   dict insertion made by the lane's own script, and the resulting column list
   is written to the artifact JSON so it can be audited. 89 columns instead of
   90. `total_line` stays — the brief names `spread_line` only, and removing the
   total as well would be a second, unmeasured change.

Everything else is the served configuration: `weak_stack` columns otherwise
unchanged, `ridge_alpha` 10, target `market_residual`, mapping
`gaussian_median`, and the S3 home-side offset refitted from **this arm's own**
raw out-of-time opener stream exactly as the served policy does.

**Mechanism this tests.** If the lean is what a free line proxy expresses, then
removing every free line proxy — the 47 orthogonalised columns can no longer
express it, and the line column is gone — should remove it. The model would
then be answering "what does recent form say that the line does not already
say", with the line supplied only as the reference point. If the favourite-ward
residual at 10.5+ is still ~+0.44 points after that, the lean is coming from
the target, not from any column, and the next lane has to change the target.

### Arm T2 — T1 with the residual target winsorised at ±21 points

Identical to T1 with one addition: the training target `ats_margin` is clipped
to [-21, +21] before fitting. **Why 21, stated before scoring and not tuned:**
21 points is three touchdowns, the conventional football boundary between a
game that was competitive and a game that stopped being a game. The diagnosis'
own margin-minus-line histogram is the reason to expect this to matter on big
lines — the 7.5+ archive puts 5.67% of games at exactly +10 against the served
Gaussian's 1.75%, and 5.15% at -14 against 2.12% (read:
`docs/spread_hole_diagnosis.md:226-234`), i.e. the tails on big spreads are
fatter and lumpier than the fitted scale, and least-squares spends its budget
chasing them. No other clip value is scored, in this lane or any other. The
clip is applied to `ats_margin` in the training frame, so it is used both by
the coefficient fit and by the out-of-time residual sample that sets the
probability scale; that is one clip in one place and it is reported as such.

**Mechanism this tests.** Blowouts happen disproportionately on big spreads and
disproportionately favourite-ward, so a squared-error fit on the raw residual
buys accuracy on 35-point wins at the price of the sign on 3-point games. If
that is what drives the lean, capping the target at three touchdowns should
shrink the favourite-ward residual on 7.5+ without touching small lines.

### Arm T3 — a direct cover-side classifier

A logistic regression that predicts the cover side directly instead of a point,
on the same orthogonalised, no-line feature matrix as T1.

- Pipeline mirroring the served ridge's: `SimpleImputer(strategy="median",
  add_indicator=True)` then `StandardScaler` then
  `LogisticRegression(C=0.1, max_iter=2000)`. `C = 1/alpha = 0.1` mirrors ridge
  alpha 10.
- Target: `home_cover` at that training row's own line — 1 when `ats_margin` >
  0, 0 when `ats_margin` < 0. Exact pushes (`ats_margin` == 0) are dropped from
  training, because a push has no side; that is a predeclared choice, not a
  filter tuned on the answer.
- Walk-forward identical to the ridge's: one refit per archive week on
  completed regular-season games strictly before that week's first kickoff,
  minimum 500 training games.
- Served probability is the classifier's own output; side is `p >= 0.5`, the
  same probability rule production uses.
- **T3 carries no S3 home-side offset, and this is a stated handicap, not an
  oversight.** The S3 offset is defined in POINTS on the residual scale
  (`nfl_ats.home_side_location`) and a classifier has no point scale to shift.
  The offset is worth +3.1 accuracy points to the incumbent at 7.5-10 and +3.1
  at 10.5+ (read: `docs/spread_hole_diagnosis.md:34-36`), so comparing T3 to the
  served incumbent understates T3. **Predeclared secondary read, declared here
  before scoring:** T3 is also compared to the incumbent's RAW (pre-S3) picks,
  which the harness already carries as
  `correct_at_open_probability_rule_raw`, so the reader gets one like-for-like
  number. The PRIMARY T3 comparison remains against the served incumbent,
  because the served incumbent is the thing that would have to be replaced.

**Mechanism this tests.** A squared-error fit on a residual optimises points,
and points on big spreads are dominated by blowouts; a logistic fit on the side
optimises exactly the quantity the pool grades. If the lean is an artefact of
minimising squared points, the classifier should not have it.

### Arm T4 — positive control, NOT a candidate

The incumbent's served pick, FORCED to the underdog on the 7.5-10 bucket only,
every game, both directions. Nothing else changes; the other three buckets keep
the incumbent's served pick exactly. Where the forced side differs from the
incumbent's, the served home-cover probability is mirrored (`p -> 1-p`) so the
card's three overlays see a consistent card.

**This arm may never be played, and is not proposed for the card.** Owner's
ban on unexplained threshold flips is binding (AGENTS.md, "No unexplained
threshold flips on the played card"), and a rule that fires only inside a
spread bucket is exactly what that ban names. T4 exists for one purpose: the
market's underdog edge at 7.5-10 is 53.92% over 2009-2025, present in all three
eras at 51.93 / 54.36 / 55.24 percent (read:
`docs/spread_hole_diagnosis.md:189-192`), so an arm that takes the underdog
every time in that bucket measures the **ceiling** of what any repair confined
to that bucket could buy. If T4 itself gains little, then no fix to 7.5-10 —
however well motivated — can gain much, and that bound is the finding.

### Grading, identical for every arm and for the incumbent

Walk forward exactly as `nfl_ats.clv.opener_pick_evaluation` does, reusing
`scripts/spread_hole_arms.py`'s harness so the incumbent replays exactly: one
weekly market-residual fit per archive week, trained on completed regular-season
games strictly before that week's first kickoff, minimum 500 training games.
Scored at the **opener** on the same 1,537-game archive (1,503 non-push at the
opener), 2020-2025.

Then scored THROUGH the played three-member card — the OR union of
`coach_fade_overlay`, `division_revenge_tilt_overlay` and
`player_arrests_back_side_policy`
(`overlay_union_coach_division_revenge_player_arrests_v2`) — with every member
trigger recomputed against each incoming card, using
`nfl_ats.overlay_composition`'s own machinery.

**The six tilts served on today's card are NOT in this archive harness.** The
card scored here is the three-member union the archive harness implements, the
same one lanes F and M scored. Any statement about the played card below is a
statement about that three-member card, and I have not verified it against the
six tilts served today.

Paired week-blocked **and** season-blocked bootstrap, 20,000 samples, seed
20260821, within-week correlation zero by owner mandate. Reported overall and by
the four declared line buckets (0-6.5, exactly 7, 7.5-10, 10.5+).

The replay is only trusted if the incumbent reproduces
`artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` to a maximum
residual gap of 0.0, as it did in lanes F and M.

**Primary quantity**: the paired accuracy-point difference through the card at
the opener, overall, with its `probability_positive`. **Predeclared secondary
reads**: per-bucket paired deltas through the card; the standalone (pre-card)
paired delta overall and per bucket; the favourite share of picks per arm per
bucket; the net favourite-ward residual per arm per bucket; the mean `results`-
and team-quality-family contribution toward the favourite per arm per bucket;
Brier score and log loss of the served home-cover probability at the opener,
overall and per bucket, model and card; and for T3 only, the like-for-like
comparison against the incumbent's raw pre-S3 picks.

Every cell is recorded with `nfl-ats weak-signals record`, units
`accuracy_points`, league `nfl`, named
`spread_hole_<arm>_<card|standalone>_<scope>_2020_2025`. Only a whole-interval
wrong sign on BOTH the week-blocked and the season-blocked interval may be
recorded as `refuted_mechanism` / `wrong_sign_resolved`; everything else is
`unresolved_below_power`.

**Reuse discount, stated up front**: these arms are motivated by a descriptive
decomposition computed on the same 2020-2025 archive they are graded on. That is
the same stated discount `docs/spread_regime_program.md` declares, not
independent confirmation.

Thresholds govern only what this document may CLAIM. Which card is PLAYED is an
expected-value decision on the opener grade, full stop.

## Results

Every number in this section is **measured this session** by the command at the
bottom; the tables live in
`artifacts/spread_hole_target/20260909T231047Z/`.

The incumbent replay is exact: re-fitting all 107 archive weeks inside this
lane's own script reproduces
`artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` with a maximum
residual gap of **0.0** and a maximum probability gap of **0.0** across all
1,537 games, so the pairing below is the served model against itself plus one
change. The per-family decomposition closes against each ridge arm's own
prediction to at most **8.0e-15** points. The no-line matrix is **89 columns**
(`spread_line` absent, verified in `arm_results.json`'s `no_line_columns`); the
orthogonalised block is the same **47 columns** lane M used.

Incumbent card accuracy at the opener: **55.89%** on 1,503 non-push games; the
served model's own picks, before the card's three overlays, score **54.56%** on
the same games.

### The mechanism, which is the finding

Mean per-family ridge contribution **toward the favourite**, in points, and the
quantity that actually decides the pick, on the archive's lined games:

| bucket | arm | results | market | whole 47-col block | net residual toward favourite | favourite share of picks |
|---|---|---|---|---|---|---|
| overall | incumbent | +1.631 | -0.332 | +0.713 | **+0.164** | 53.18% |
| overall | T1 | +0.056 | +0.004 | +0.005 | **+0.003** | 49.97% |
| overall | T2 | +0.054 | +0.003 | +0.002 | **+0.008** | 50.23% |
| 7.5-10 | incumbent | +2.975 | -0.546 | +1.272 | **+0.486** | 57.95% |
| 7.5-10 | T1 | +0.395 | +0.010 | +0.115 | **+0.268** | 55.38% |
| 7.5-10 | T2 | +0.355 | +0.008 | +0.086 | **+0.221** | 52.82% |
| 10.5+ | incumbent | +4.200 | -0.785 | +1.522 | **+0.442** | 60.45% |
| 10.5+ | T1 | +0.437 | +0.026 | -0.191 | **+0.108** | 55.97% |
| 10.5+ | T2 | +0.386 | +0.026 | -0.165 | **+0.157** | 58.21% |

**T1 is the first arm in this program that actually removes the lean.** Lane
M's G3 dismantled the ±4-point cancellation and the net favourite-ward residual
did not move (+0.442 to +0.417 at 10.5+), because the ridge rebuilt the lean out
of `spread_line`, whose family contribution flipped from -0.785 to +0.954 (read:
`docs/spread_hole_group_penalty.md:226-243`). Delete that column as well and
there is nothing left to rebuild it from: at 10.5+ the net residual toward the
favourite falls from **+0.442 to +0.108** and the favourite share of picks from
**60.45% to 55.97%**; overall it falls from **+0.164 to +0.003** and the
favourite share from **53.18% to 49.97%**. Lane M's inference is confirmed by
construction — the lean is what a free line proxy expresses, and it dies when
every free line proxy is gone.

T3, which has no coefficient decomposition, shows the same thing through its
picks: it takes the favourite on **45.64%** of 7.5-10 games and **40.30%** of
10.5+ games, against the incumbent's 57.95% and 60.45%.

### And removing the lean does not buy accuracy

Paired delta through the played card, at the opener. Week-blocked and
season-blocked bootstrap, 20,000 samples, seed 20260821.

| arm | scope | n | incumbent card | candidate card | picks changed | delta (pts) | 95% week-blocked | `probability_positive` | 95% season-blocked | season P+ |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | overall | 1,503 | 55.89% | 55.22% | 44 | **-0.665** | [-1.545, +0.201] | **0.0709** | [-1.203, 0.000] | 0.0273 |
| T1 | 0-6.5 | 1,104 | 57.61% | 57.25% | 32 | -0.362 | [-1.435, +0.721] | 0.2605 | [-1.020, +0.591] | 0.2044 |
| T1 | 7 | 74 | 50.00% | 50.00% | 2 | 0.000 | [-4.054, +4.054] | 0.4984 | [-3.226, +4.348] | 0.5035 |
| T1 | 7.5-10 | 194 | 51.55% | 48.97% | 5 | -2.577 | [-5.000, -0.524] | 0.0029 | [-4.124, -1.064] | 0.0007 |
| T1 | 10.5+ | 131 | 51.15% | 50.38% | 5 | -0.763 | [-4.032, +2.500] | 0.3325 | [-3.279, +1.942] | 0.2822 |
| T2 | overall | 1,503 | 55.89% | 55.02% | 63 | **-0.865** | [-1.796, +0.067] | **0.0347** | [-1.793, +0.068] | 0.0353 |
| T2 | 0-6.5 | 1,104 | 57.61% | 57.34% | 45 | -0.272 | [-1.484, +0.914] | 0.3290 | [-0.952, +0.492] | 0.2442 |
| T2 | 7 | 74 | 50.00% | 48.65% | 1 | -1.351 | [-4.348, 0.000] | 0.1820 | [-3.409, 0.000] | 0.1698 |
| T2 | 7.5-10 | 194 | 51.55% | 47.94% | 11 | -3.608 | [-6.863, -0.510] | 0.0141 | [-5.319, -1.942] | 0.0000 |
| T2 | 10.5+ | 131 | 51.15% | 49.62% | 6 | -1.527 | [-5.147, +2.239] | 0.2078 | [-5.217, +1.869] | 0.2044 |
| T3 | overall | 1,503 | 55.89% | 54.56% | 240 | **-1.331** | [-3.209, +0.605] | **0.0884** | [-2.901, +0.513] | 0.0820 |
| T3 | 0-6.5 | 1,104 | 57.61% | 56.07% | 169 | -1.540 | [-3.792, +0.805] | 0.0947 | [-3.353, +0.260] | 0.0494 |
| T3 | 7 | 74 | 50.00% | 50.00% | 12 | 0.000 | [-9.333, +9.211] | 0.4978 | [-6.742, +8.929] | 0.5178 |
| T3 | 7.5-10 | 194 | 51.55% | 51.03% | 31 | -0.515 | [-5.914, +4.918] | 0.4253 | [-2.632, +2.451] | 0.3314 |
| T3 | 10.5+ | 131 | 51.15% | 49.62% | 28 | -1.527 | [-9.244, +6.015] | 0.3444 | [-7.463, +5.085] | 0.3157 |
| T4 | overall | 1,503 | 55.89% | 55.89% | 72 | **0.000** | [-1.061, +1.064] | **0.5050** | [-0.654, +0.798] | 0.4879 |
| T4 | 0-6.5 | 1,104 | 57.61% | 57.61% | 0 | 0.000 | [0, 0] dead heat | 0.5000 | [0, 0] | 0.5000 |
| T4 | 7 | 74 | 50.00% | 50.00% | 0 | 0.000 | [0, 0] dead heat | 0.5000 | [0, 0] | 0.5000 |
| T4 | 7.5-10 | 194 | 51.55% | 51.55% | 72 | 0.000 | [-8.205, +8.040] | 0.5070 | [-5.670, +5.769] | 0.4879 |
| T4 | 10.5+ | 131 | 51.15% | 51.15% | 0 | 0.000 | [0, 0] dead heat | 0.5000 | [0, 0] | 0.5000 |

Standalone, before the card:

| arm | scope | n | incumbent | candidate | picks changed | delta (pts) | 95% week-blocked | `probability_positive` | 95% season-blocked | season P+ |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | overall | 1,503 | 54.56% | 53.96% | 69 | -0.599 | [-1.721, +0.522] | 0.1454 | [-1.238, +0.137] | 0.0440 |
| T1 | 0-6.5 | 1,104 | 56.16% | 55.16% | 47 | -0.996 | [-2.246, +0.269] | 0.0558 | [-1.877, +0.277] | 0.0532 |
| T1 | 7 | 74 | 51.35% | 58.11% | 7 | **+6.757** | [0.000, +14.286] | **0.9785** | [0.000, +14.286] | 0.9666 |
| T1 | 7.5-10 | 194 | 51.55% | 48.97% | 7 | -2.577 | [-5.291, 0.000] | 0.0216 | [-4.124, -1.064] | 0.0007 |
| T1 | 10.5+ | 131 | 47.33% | 48.85% | 8 | **+1.527** | [-2.459, +5.970] | **0.7572** | [0.000, +4.000] | 0.9561 |
| T2 | overall | 1,503 | 54.56% | 54.09% | 95 | -0.466 | [-1.716, +0.841] | 0.2307 | [-1.354, +0.481] | 0.1742 |
| T2 | 0-6.5 | 1,104 | 56.16% | 55.62% | 62 | -0.543 | [-1.918, +0.827] | 0.2147 | [-1.307, +0.385] | 0.1150 |
| T2 | 7 | 74 | 51.35% | 59.46% | 8 | **+8.108** | [+1.333, +16.000] | **0.9893** | [+2.353, +14.706] | 0.9994 |
| T2 | 7.5-10 | 194 | 51.55% | 48.45% | 16 | -3.093 | [-7.065, +0.995] | 0.0644 | [-6.250, 0.000] | 0.0303 |
| T2 | 10.5+ | 131 | 47.33% | 46.56% | 9 | -0.763 | [-5.185, +3.817] | 0.3644 | [-3.200, +1.961] | 0.2805 |
| T3 | overall | 1,503 | 54.56% | 52.56% | 354 | -1.996 | [-4.444, +0.466] | 0.0550 | [-4.258, +0.598] | 0.0646 |
| T3 | 0-6.5 | 1,104 | 56.16% | 54.35% | 250 | -1.812 | [-4.681, +1.093] | 0.1103 | [-4.586, +1.228] | 0.1226 |
| T3 | 7 | 74 | 51.35% | 55.41% | 19 | +4.054 | [-7.353, +15.584] | 0.7548 | [-5.000, +18.182] | 0.7386 |
| T3 | 7.5-10 | 194 | 51.55% | 47.42% | 44 | -4.124 | [-11.735, +3.077] | 0.1394 | [-10.101, +1.500] | 0.0784 |
| T3 | 10.5+ | 131 | 47.33% | 43.51% | 41 | -3.817 | [-13.223, +5.469] | 0.2092 | [-11.111, +2.976] | 0.1440 |
| T4 | overall | 1,503 | 54.56% | 54.89% | 113 | **+0.333** | [-1.117, +1.757] | **0.6786** | [-0.986, +1.305] | 0.7252 |
| T4 | 7.5-10 | 194 | 51.55% | 54.12% | 113 | **+2.577** | [-8.411, +13.514] | **0.6785** | [-7.576, +9.794] | 0.7252 |

T3's predeclared like-for-like read, against the incumbent's RAW pre-S3 picks
(the comparison that removes T3's structural handicap): overall
**52.56% vs 53.96%**, -1.397, 95% week-blocked [-3.787, +0.937],
`probability_positive` 0.1235; at 7.5-10 **47.42% vs 48.45%**, -1.031,
[-7.937, +5.699], 0.3854; at 10.5+ **43.51% vs 44.27%**, -0.763,
[-9.160, +7.519], 0.4255. Removing the handicap does not turn T3 around, so its
shortfall is not the missing offset.

### What T4 bounds

T4 forced **113 of the 194** picks in the 7.5-10 bucket onto the underdog — the
side the market has favoured 53.92% of the time over 2009-2025 (read:
`docs/spread_hole_diagnosis.md:192`). Taking that side on every single game in
the bucket, with no model at all:

- **standalone**, the bucket goes 51.55% -> **54.12%**, +2.577 accuracy points,
  95% week-blocked [-8.411, +13.514], `probability_positive` **0.6785**;
- overall that is +0.333 accuracy points, `probability_positive` 0.6786;
- **through the played card the delta is exactly 0.000** on 72 changed picks —
  51.55% before, 51.55% after.

Read that as the ceiling it is. The best a bucket-confined side rule can reach
in 7.5-10 is roughly **54%**, because 54% is roughly what the market's own
underdog tilt is worth there, and 194 games is a cell in which +2.6 points
carries a ±11-point band. Every candidate in this lane and in lanes F and M is
being asked to find part of a **2.6-point** prize on 13% of the card — worth
about a third of an accuracy point overall if captured perfectly. And through
the card, which is what is actually played, the prize measured **0.000**: the
three overlays already move 72 of those picks, and the card's 51.55% at 7.5-10
does not change when the rest are forced to the market's better side.

The honest form of that is not "there is nothing there" — 194 games cannot
resolve 2.6 points. It is: **a repair confined to the 7.5-10 bucket cannot be
worth much, so the 7.5-10 dip is not where the card's remaining edge is.** The
next arm should stop treating that bucket as the target.

### Probability quality

Brier score and log loss at the opener (1,503 non-push games), model and card:

| arm | Brier (model) | log loss (model) | Brier (card) | log loss (card) |
|---|---|---|---|---|
| incumbent | 0.25158 | 0.69662 | 0.24877 | 0.69080 |
| T1 | 0.25156 | 0.69659 | 0.24886 | 0.69098 |
| T2 | 0.25214 | 0.69786 | 0.24960 | 0.69253 |
| T3 | 0.25429 | 0.70235 | 0.25159 | 0.69668 |
| T4 | 0.25088 | 0.69523 | 0.24836 | 0.69001 |

Every arm is within 0.003 Brier of the incumbent. Mean stated confidence is
55.76% (incumbent), 55.83% (T1), 56.13% (T2) and 56.07% (T3) overall, and
56.1-56.5% at 7.5-10 and 55.9-56.5% at 10.5+ for every arm. The flat,
bucket-blind confidence the diagnosis flagged survives all four arms — deleting
the line column changes which side the model takes without changing how sure it
says it is, which is a separate defect and still open.

### Decision

**Keep the incumbent card.** At the opener, on the forced-pick pool's own grade,
the cards score **55.89% (incumbent), 55.89% (T4, unplayable), 55.22% (T1),
55.02% (T2) and 54.56% (T3)** on the same 1,503 non-push games. Playing T1
instead of the incumbent is a bet at `probability_positive` **0.0709**
(season-blocked 0.0273), T2 at **0.0347** (0.0353) and T3 at **0.0884**
(0.0820). All three are the wrong side of an expected-value call, which is the
only bar that governs which card is played. T4 ties the incumbent exactly and is
barred from the card by the owner's ban on unexplained threshold flips
regardless.

Nothing here rests on an interval containing zero: the point estimates and the
`probability_positive` values themselves favour the incumbent, and T1 stays open
as a mechanism — it is the only arm that has ever removed the favourite-ward
lean, and it costs 0.665 accuracy points to do so.

Nothing in this lane was promoted, no ledger was touched, and the active
manifest was not edited. Had a candidate won through the card, promoting it
would require: (1) a named feature profile wired into `nfl_ats.margin` and into
the weekly refit path (`weekly-run`) so the served model fits with it — for T1
and T2 that is a real profile addition, since `weak_stack_no_line` currently
exists only as a runtime registration inside this lane's script; (2) editing
`artifacts/active_ats_model.json` (`feature_profile` / `model_id` /
`feature_table_sha256` / `historical_evaluation`); (3) regenerating the opener
evaluation under the new model; (4) re-running the overlay composition, because
the card members' triggers are recomputed against the incoming card; (5)
regenerating the weekly forecast and refreshing `CURRENT_PREDICTIONS.md` via
`nfl-ats publish-predictions`; and (6) `nfl-ats publish-board`, which fails
closed when a headline number's source names a different model. No Week 1 pick
comparison was computed, because no candidate won.

### What this says about where to look next

Three arms in three lanes have now been asked to fix the big-spread buckets by
changing what the model is made of, and the result is consistent: the
favourite-ward lean is real and removable (T1 removes it), and removing it does
not make the card better. The 7.5-10 dip is bounded at about +2.6 points in a
194-game bucket and **0.000 through the card** (T4). What is NOT bounded, and
what none of the seven arms has touched, is the flat 55.8-56.5% stated
confidence in every bucket — the model has one uncertainty and reuses it
everywhere (read: `docs/spread_hole_diagnosis.md:174-180`). That is a
per-bucket, mass-correct probability problem, not a side problem, and it is the
one place this program's own measurements keep pointing.

### Registry

All 45 cells are recorded under family `mod18_spread_hole_target_v1`, named
`spread_hole_<arm>_<card|standalone>_<scope>_2020_2025` (plus five
`_standalone_vs_raw_` cells for T3's like-for-like comparison). **Two** are
terminal: `spread_hole_t1_card_7p5_10_2020_2025` and
`spread_hole_t2_card_7p5_10_2020_2025` — for each, both the week-blocked and the
season-blocked interval sit **entirely below zero**, which is the one admissible
closing ground (`wrong_sign_resolved`): deleting the line column makes the
played card resolvedly worse in that bucket. The other **43** are
`unresolved_below_power` and report `probability_positive`, including every cell
whose interval merely touches or crosses zero and both of the favourable cells
(`spread_hole_t2_standalone_7_2020_2025`, week-blocked [+1.333, +16.000],
`probability_positive` 0.9893, whose interval sits entirely ABOVE zero — a
favourable interval is not a closing ground either, so it stays open).

The exact argv lists are saved as `record_commands.json` in the artifact
directory and were run one at a time against the main checkout's registry
(`registry/weak_signals.json`), which now holds 45 `spread_hole_t*` cells.

## Commands run

```
python scripts/spread_hole_target.py \
  --features data/processed/game_features_weak_stack.parquet \
  --data-root data --market-root data/market/raw \
  --archive artifacts/opener_evaluation/20260909T183120Z \
  --out artifacts/spread_hole_target/20260909T231047Z

nfl-ats weak-signals record ...   (45 cells, mod18_spread_hole_target_v1;
the exact argv lists are saved as record_commands.json in the artifact
directory)
```
