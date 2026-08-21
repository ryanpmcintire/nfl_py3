# Pool format levers (POL-04/05): what the format is worth

Written 2026-08-17. `docs/pool_edge_plan.md` names three places edge can live and
says the third — *"exploits the pool's format rather than the line"* — is largely
unexplored. This document explores it, and it is the only one of the three that
is **not** bounded by the team-quality ceiling, because it optimises a different
objective: not expected correct picks, but **probability of finishing first**.

Everything here is arithmetic conditional on accuracies measured elsewhere. It
spends no rotation window and reserves no data. The simulator lives in
`nfl_ats.pool`, the runs in `scripts/pool_levers.py`, the output in
`artifacts/pool_levers/levers.json`.

---

## 0. The headline, before any lever

**The format already converts our edge into far more than the edge looks like.**
285 forced picks is a very long series, so a 2.5-point accuracy edge compounds
into a large advantage in the standings while nothing about the picks changes.

Simulated: our card at 52.50% against a field of entrants who pick the public
side 65% of the time and are 50% accurate.

| field size | P(first), coin-flip entry | P(first), 52.5% entry | multiple | P(top 15%), coin flip | P(top 15%), 52.5% |
|---|---|---|---|---|---|
| 5 rivals | 17.1% | **39.4%** | 2.3× | 18.4% | 41.5% |
| 25 | 4.2% | **16.2%** | 3.9× | 17.4% | 43.5% |
| 100 | 1.19% | **6.56%** | 5.5× | 16.9% | 44.0% |
| 1,000 | 0.11% | **1.35%** | 12.7× | 17.3% | 44.7% |

Read the last row: in a 1,001-entrant pool, a fair share of first place is 0.1%;
our card is worth 1.35%. The multiple *grows* with field size, because winning a
big pool requires a tail result and a real edge shifts the whole distribution
rather than needing luck to do it.

That is the format's value. Every lever below is small next to it.

---

## 1. What already existed

- **POL-02/03 card builders** (`pool.py`): force a side per game, rank by
  confidence, and the same for straight-up. Long done.
- **POL-09 Best Pick ranker**: `sweep_robustness`, frozen in
  `nfl_ats/best_pick.py`, screened on [2013, 2015] and confirmed on [2020, 2021].
  Both windows permanently spent. It is genuinely live — `select_best_pick` is
  called by `public_board.py` and by the dashboard picks page, and `is_best_pick`
  persists into the MKT-04 ledger.
- **POL-10 prospective scoring**: settles 2026 picks at both grades and records
  the weekly Best Pick before kickoff.
- **POL-04 (pick popularity) and POL-05 (contest utility): nothing.** No field
  model, no ownership input, no simulator. That is what this document adds.

---

## 2. Verifying the Best Pick ranker

The claim under test: `sweep_robustness` is "confirmed + live". Recomputed from
`artifacts/best_pick_ranker/opener_2020_2021.picks.parquet`, independently of the
script that produced it:

| metric | recorded | recomputed |
|---|---|---|
| top-1 accuracy | 60.0% (21/35) | **60.0% (21/35)** |
| all-pick accuracy | 51.32% | **51.32%** |
| delta | +8.68 pts | **+8.68 pts** |
| Kendall tau | +0.067 (p = 0.099) | **+0.067 (p = 0.099)** |

The artifact is real, the arithmetic is right, the deployment is wired. So far the
claim holds. Two things it does not survive.

### 2.1 The tie-break is doing most of the work

`sweep_robustness` is a width on a half-point grid that runs from 0 to 8 points
and **censors at 8.0** — a limitation the write-up already flags. The consequence
was not traced: the signal is coarse, so weeks routinely end in a tie at the top,
and `select_best_pick` breaks ties **alphabetically by `game_id`**.

- Confirmation window: **24 of 35 weeks** had two or more games tied at the
  maximum. Screen window: **39 of 51**, with the maximum sitting on the censored
  ceiling in 45 of 51.
- Under a uniformly random tie-break — the same signal, with the arbitrary part
  acknowledged as arbitrary — expected top-1 accuracy is **52.24%**, not 60.0%.
  The delta over all picks falls from **+8.68 points to +0.92**.
- The recorded 60.0% sits at the 88th percentile of the tie-break distribution
  (Monte Carlo over tie-break orders: mean 52.3%, 5–95% [42.9%, 60.0%]).
- On the screen window the tie-break pushed the *other* way: recorded 54.90%,
  tie-break-agnostic **58.38%** (+9.05 rather than +5.57).

So the write-up's strongest argument — "two disjoint windows, two grades, same
direction, both clearing" — is partly a coincidence of alphabetical ordering.
Both windows are still positive tie-break-agnostic (+9.05 and +0.92), so the
*direction* survives; the *consistency* does not. The honest per-window effect is
+9.1 points on 51 screen weeks and +0.9 points on 35 opener weeks.

This is not a re-scoring of a spent window and changes no registry verdict. It is
a description of how the recorded number was produced.

### 2.2 A Best Pick effect of this size cannot be measured

All-pick accuracy 52.50%, so the standard error of a top-1 rate over `W` weeks is
`sqrt(0.25/W)`:

- 35 weeks (the confirmation) → **8.45 points**. The recorded +8.68 is 1.03 se.
- 107 weeks (the whole opener archive) → **4.83 points**. Any top-1 result between
  **43.0% and 62.0%** is indistinguishable from just picking one of our own picks.
- Resolving a +5-point Best Pick effect to 95% needs ~384 weeks ≈ **21 seasons**.
  At 18 Best Picks a year, prospective play will never settle this.

For context, every ordering computable from the opener archive, on all 107 weeks
(descriptive reporting on already-mined data, no window spent, no selection
implied — the same basis on which `pool_edge_plan.md` reported the 48.6%):

| weekly top-1 ordering | top-1 accuracy | vs 52.50% | z |
|---|---|---|---|
| away picks first | 59.8% | +7.3 pts | 1.52 |
| smallest \|residual\| | 57.0% | +4.5 | 0.94 |
| smallest opener spread | 54.2% | +1.7 | 0.35 |
| home picks first | 52.3% | −0.2 | −0.03 |
| biggest underdog we take | 49.5% | −3.0 | −0.61 |
| largest \|residual\| (the known-flat one) | 48.6% | −3.9 | −0.81 |
| key-number geometry | 48.6% | −3.9 | −0.81 |
| largest opener spread | 46.7% | −5.8 | −1.19 |

Nothing reaches 2σ. "Away picks first" leads, which is exactly the kind of result
this table exists to warn against: eight orderings, best z = 1.52, is what noise
looks like. **The right conclusion is that the Best Pick ordering problem is
under-powered by construction, not that a ranker was found.**

### 2.3 What the ranker is actually worth

Simulated, holding the card fixed and changing only *which* game is nominated
(one game per week genuinely covers at the higher rate; the card still averages
52.50%). Best Pick scores double, so a correct one is worth **+1** extra point.
That rule is **researched, not confirmed by the user** (Splash blog and contest
rules describe a double-points Best Pick); the simulator takes the bonus as a
parameter and every number below is linear in it, so a 3× rule roughly doubles
each gain and changes no conclusion.

| true ranker effect | field | P(first), arbitrary nomination | P(first), ranker | gain |
|---|---|---|---|---|
| +8.7 pts (as recorded) | 100 | 6.89% | **9.27%** | +2.38 pp (+35%) |
| +8.7 pts | 1,000 | 1.56% | **2.31%** | +0.76 pp (+49%) |
| +0.9 pts (tie-break-agnostic) | 100 | 7.29% | 7.36% | +0.07 pp (+0.9%) |
| +0.9 pts | 1,000 | 1.72% | 1.75% | +0.03 pp (+1.9%) |
| none (control) | 100 | 7.13% | 6.95% | −0.18 pp (noise) |

The lever is worth a lot **if** the +8.7 is real and roughly nothing if the
tie-break-agnostic +0.9 is. Section 2.1 says the second is the better estimate.

### 2.4 A defect found while verifying: the tie-break decides 2026 Week 1

On the live Week 1 card, `sweep_robustness` puts **two games at the censored
ceiling of 8.0** (ARI@LAC and WAS@PHI) and the alphabetical tie-break nominates
ARI@LAC. The Best Pick that goes into the pool on 8 September is chosen by
alphabetical order between two indistinguishable games.

The tie-break itself is correct — reproducibility matters more than a coin flip
in a nomination that must be stable across publishes. The defect is that nothing
*surfaces* the tie, so a 50/50 choice reads on the page as a signal-driven one.

---

## 3. The simulator (POL-05)

`nfl_ats.pool` gained `PoolFormat`, `FieldModel`, `Entry`, `simulate_pool_finish`,
`head_to_head_win_probability`, `deviate` and `strategy_comparison`.

**Model.** Games are independent and cannot push (Splash posts half-point
numbers). Our side covers game *i* with probability `p_i`. Every field entrant
independently takes the *public* side of each game with probability
`public_lean`; the public side is a property of the game, not of the outcome, so
a field entrant is a 50% picker by construction while entrants still resemble
each other. Conditional on the outcome vector, entrants are i.i.d. and each
entrant's per-game correctness takes exactly two values, so a field score is an
exact sum of two binomials rather than a simulated coin flip per pick. That is
what makes a sweep over field sizes affordable.

**Validated against five cases with known answers** (`tests/test_pool.py`):

| case | expected | simulator |
|---|---|---|
| no edge, uncorrelated field, 9 rivals | 1/10 | 0.100 |
| field copies our card exactly (`public_lean` = 1) | all tie: P(first) = 1/5, P(outright) = 0 | exactly that |
| perfect card | P(outright) = 1 | 1.000 |
| one deterministic rival, 9 disagreements | `head_to_head(9, 0.55)` closed form | matches to ±0.01 |
| expected score with a 2-point bonus | 64·0.525 + 4·2·0.525 | matches to ±0.15 |

**The independence assumption is measured, not assumed.** Over 107 opener-graded
weeks the observed variance of the weekly correct share is **1.036×** the
binomial expectation, and the season-level ratio is 0.72. There is no meaningful
positive correlation between our picks within a week, so the simulator's biggest
simplification is empirically harmless.

**Known limits.** Field Best Picks are drawn independently of the rest of the
entrant's card (one game in 285 — negligible). `public_lean` is a single number
where a real field has heterogeneous entrants; the sweeps report 0.65 and 0.85
so the conclusions can be checked against both. And **the field model has never
been fitted to a real field**, because no data on one exists (§4).

---

## 4. Pick popularity (POL-04): the data does not exist

This is a real finding, not a gap in effort.

- **Splash Sports does show a pick distribution** (Commissioner Handbook, "Stats
  page"). It **unlocks game by game as each game kicks off** — picks stay hidden
  until then, deliberately, as an integrity measure that even commissioners
  cannot override. It is therefore **structurally unavailable before the Tuesday
  lock**, which is the only moment it could be used. There is no public API.
- **The Odds API sells no betting percentages.** Its v4 endpoints are odds,
  historical odds snapshots and scores. Its "consensus" is consensus *of books*,
  not of bettors. Our $30 plan cannot buy this.
- **Free ticket/handle-percentage feeds**: none with an API and history.
  ScoresAndOdds and OddsShark publish current-week percentages by scrape with no
  archive and no documented methodology; Covers' "consensus" is its own contest
  users, not money; VegasInsider's is a panel of handicappers; Action Network's
  real splits are behind Action PRO. The rigorous option (Sports Insights / Bet
  Labs) is paid and not a clean REST API.

**What is obtainable is a proxy for the popular side, not its magnitude.** Two
survive, both measured here on the 1,491 non-pick'em opener-graded games:

- *Favourite*: the literature supports a favourite lean in real pool fields
  (Levitt 2004, on a real-money NFL contest: about three-quarters of entrants
  picked favourites more often than underdogs, and only ~2% took road underdogs
  on half their picks). **Our card is on the favourite 54.8% of the time and on
  the home team 46.7%** — we are only mildly chalky, so we are already about as
  differentiated from a favourite-loving field as a coin flip would be.
- *Line movement* is a money proxy but arrives **after** the Tuesday lock, so it
  can only ever be a prediction target. Quantified below, and it closes.

Base rates, for the record: at the Tuesday opener the favourite covered **49.30%**
and the home team **49.63%** (1,491 games, pushes already excluded). Both lean the
right way for a fade and neither is resolvable — the standard error is 1.3 points.

---

## 5. Contest utility: does differentiation pay?

The intuition is real and has a closed form. Games we and a rival both pick the
same way cancel; the margin is settled on the `d` games we disagree about, where
we win `Binomial(d, q)`. The mean of that margin grows like `d` and its spread
only like `sqrt(d)`, so **more disagreement is better — provided `q` stays above
0.5**:

| disagreements | q = 0.50 | q = 0.525 | q = 0.55 |
|---|---|---|---|
| 25 | 0.500 | 0.600 | 0.694 |
| 100 | 0.460 | 0.656 | 0.817 |
| 285 (everything) | 0.500 | **0.801** | 0.955 |

The trap is that the only way to differentiate in a forced-pick pool is to take a
side we think is worse, which pushes `q` **below** 0.5 on exactly the games doing
the differentiating. Simulated, flipping picks that cost the full 2.5-point edge:

| flips (of 168 shared with the public) | P(first), 100 rivals | 1,000 rivals |
|---|---|---|
| 0 | **6.95%** | 1.73% |
| 25 | 6.14% | 1.52% |
| 100 | 4.06% | 1.12% |
| 168 (fade everything) | 2.79% | 0.82% |

Monotonically worse everywhere, including the 1,000-entrant pool where
variance-seeking is supposed to pay. **Do not fade the public at the cost of
accuracy.** A 2.5-point edge over 285 picks is 0.84 standard deviations of the
score; giving it up costs more than the extra spread buys back.

The interesting question is the cheap version. Sweeping the accuracy cost of each
flip, 100 rivals, flipping 35 near-coin-flip picks that sit on the public side:

| accuracy cost per flip | P(first), 0 flips | P(first), 35 flips | change |
|---|---|---|---|
| 0.0 pts (free) | 7.21% | **8.57%** | +19% |
| 0.5 pts | 7.27% | 8.28% | +14% |
| 1.0 pts | 7.33% | 8.03% | +9.5% |
| 2.5 pts | 7.23% | 7.17% | ~0 (break-even) |
| 5.0 pts | 6.95% | 5.90% | −15% |

**Break-even is around 2.5 accuracy points per flip**, and the free version is
worth +19% relative at 100 rivals and +32% at 1,000. Is a cheap flip available?
Partly, and only in theory. **41% of the card carries a model residual under one
point**, which at the measured 13.1-point scatter is an edge of at most 1.5
accuracy points — well inside break-even. Restricting to games where our pick is
*also* the favourite leaves about **3.1 flippable games per week**.

### Why the answer is still no

Three reasons, in increasing order of severity.

1. **The measured cost is unresolved.** Flipping every sub-1.5-point-residual
   favourite pick would have cost **+1.75 accuracy points, 95% [−7.4, +11.0]**.
   That interval contains "free" and it contains "double the break-even". It is a
   category-3 result and cannot support an irreversible strategy change.
2. **It needs the ability the flat-confidence finding denies.** Free
   differentiation requires telling weak picks from strong ones — the same skill
   the Best Pick ranker needs and the same skill §2.2 says is unmeasurable. The
   two levers stand or fall together.
3. **It only works winner-take-all, and the pool is not.** Splash contests
   commonly pay roughly the top 15%. Under a top-15% prize the entire lever
   evaporates:

   | accuracy cost per flip | P(top 15%), 0 flips | P(top 15%), 35 flips |
   |---|---|---|
   | 0.0 (free) | 43.8% | 43.8% |
   | 1.0 pts | 43.1% | 42.4% |
   | 2.5 pts | 43.3% | 40.0% |
   | 5.0 pts | 42.7% | 36.4% |

   Free differentiation is exactly neutral and any positive cost strictly hurts.
   Differentiation buys the top tail by selling the middle, and a wide prize pays
   for the middle. **Confirm the prize structure before considering this again;
   if more than one place pays, it is closed.**

### The movement channel, closed with arithmetic

Line movement is the one money proxy we can measure, and it looks powerful: our
picks that agreed with the eventual movement scored **55.91%** against those that
disagreed at **45.96%** (1,133 moved games, baseline 51.46%). A perfect
movement-direction oracle would therefore lift a Best Pick to 55.91% — **+4.45
points, and that is the ceiling of the whole channel**. Feeding in a realistic
predictor:

| direction accuracy | Best Pick accuracy | lift |
|---|---|---|
| 0.572 (MKT-06's measured direction rate) | 52.15% | **+0.70 pts** |
| 0.65 | 52.89% | +1.43 |
| 0.90 | 55.09% | +3.63 |
| 1.00 (oracle) | 55.91% | +4.45 |

At the accuracy we actually have, predicting the money side is worth **0.7 points
on the Best Pick**, or about one extra correct nomination every eight seasons.
Closed, quantitatively, without an experiment.

---

## 6. Ranked recommendation

1. **Keep entering all 285 picks with the model's sign, and keep the edge.**
   Worth 2.3× to 12.7× a fair share of first place. It is the entire game; every
   lever below is a rounding error next to protecting it.
2. **Surface the Best Pick tie instead of hiding it.** Two games are tied at the
   ceiling for 2026 Week 1 and 24 of 35 confirmation weeks were ties. Report the
   tied set and the width on the card so a coin flip reads as a coin flip. Costs
   nothing, changes no confirmed definition, and stops a stability property from
   being mistaken for a signal. *(A code change to `public_board.py`, which this
   agent does not own — patch in the report.)*
3. **Keep using `sweep_robustness` for the Best Pick.** The alternative — an
   arbitrary nomination — has no evidence behind it and this has some. But budget
   its value at the tie-break-agnostic **+0.9 points**, not +8.7, and expect the
   2026 realised number to land near the all-pick rate.
4. **Record the pool's actual prize structure and field size in Week 1.** These
   are the two free parameters that decide every question in §5, and both are
   observable the first time the pool is entered. Until they are known, POL-05
   cannot be closed either way.
5. **Do not build POL-04 as specified.** Field ownership is structurally
   unavailable before lock and no proxy with history exists. The favourite flag
   is the only usable stand-in and it is already computable from the card.
6. **Do not fade the public**, at any cost per flip, unless the pool turns out to
   be winner-take-all *and* the flip cost is under ~2.5 accuracy points. Both
   conditions are currently unverified and the second is unresolvable at our
   sample size.
7. **Stop searching for a Best Pick ordering.** Eight orderings on 107 weeks all
   sit inside ±2σ of the all-pick rate, and resolving a 5-point effect needs 21
   seasons. This is a category-3 problem by construction: it belongs in the
   weak-signal registry, not in a rotation window.

## 7. Where the real remaining upside is, measured

The format converts accuracy into standings position at a favourable exchange
rate, and that rate is *already collected*. What it does not do is create
accuracy. Same simulator, same field, only the accuracy changes:

| forced-pick accuracy | P(first), 25 rivals | 100 | 1,000 | P(top 15%), 100 |
|---|---|---|---|---|
| 50.0% (coin flip) | 4.2% | 1.19% | 0.11% | 16.9% |
| **52.5% (today)** | 16.2% | **6.56%** | 1.35% | 44.0% |
| 53.5% | 24.3% | 11.4% | 2.79% | 56.5% |
| **54.5% (practical band)** | 34.7% | **18.3%** | 5.66% | 68.8% |
| 55.5% (ceiling-ish) | 46.2% | 27.3% | 9.94% | 79.0% |

**Two points of accuracy multiplies P(first) by 2.8× at 100 rivals and 4.2× at
1,000.** The best case for the Best Pick ranker was +2.4 percentage points at 100
rivals; two accuracy points are worth +11.8. One accuracy point — a tenth of what
the gap analysis calls reachable — is worth about twice the ranker's optimistic
case and twenty times its honest one.

That is the ranking, and it is why the format is a multiplier to protect rather
than a lever to pull: **every remaining unit of work belongs on the line, not on
the contest.** The one exception is §6 item 4 — writing down the field size and
prize structure in Week 1 — because it is free and it is the only thing that
could reopen §5.

---

## 8. Re-examining POL-04 under the corrected deadline model (2026-08-21)

**Read** (this doc §4; ROADMAP POL-04 owner corrections 2026-08-20): there is no
Tuesday *pick* lock — picks stay editable until min(kickoff, Sunday 16:00 ET),
and Splash's pick distribution unlocks **per game, at that game's kickoff**.
§4's "structurally unavailable before the Tuesday lock" premise is therefore
wrong in a specific direction: it is unavailable before *each game's own*
deadline, which means **our picks for later games are still live when earlier
games' distributions unlock** (inferred from those two documented mechanics).

**Inferred — can a later game's pick be informed by an earlier unlock?**
Mechanically yes: by Sunday ~13:00 ET the early window's distributions are
visible and every 16:00/SNF/MNF pick is still editable. But §5's arithmetic is
untouched by timing: using the distribution to *deviate* still means taking a
side we think is worse, so `q` drops below 0.5 exactly where it matters, and
under a top-15% payout even free deviations were exactly neutral. Knowing the
field's lean earlier does not change the cost of acting on it. What timing
*does* add: the unlocked distributions are a direct measurement of this pool's
actual `public_lean` — the one parameter §3 says has never been fitted to a
real field. That is an estimation input, not a strategy change.

**Inferred — obtainable proxy for Splash's distribution?** No pre-kickoff feed
of it exists (no API; §4). Two calibration paths, both free: (1) the favourite
flag, already measured here as near-neutral for us; (2) **read** (ROADMAP
MKT-12): the Action Network bet%/money% captures registered 2026-08-21
(Sat/Sun noon) land just before most kickoffs — correlating those captures
against each week's Splash distributions *after they unlock* would measure
whether AN bet% predicts pool-field lean, converting MKT-12 into the field
model input POL-04 lacked. That comparison needs in-season observation, not
research (same stance as §6 item 4).

**Actionable**: (1) nothing changes about the card today — §5's recommendation
stands; (2) during the season, record each game's Splash distribution when it
unlocks (free, manual) next to the MKT-12 capture log, and fit `public_lean`
from Week 1–2 data; (3) revisit deviation only if the pool proves
winner-take-all AND observed flip costs sit under the ~2.5-point break-even —
both currently unmeasured. POL-04 stays closed as a *data* question; the
corrected deadline converts it into a cheap *measurement* task, not a lever.

---

## Sources and reproduction

- Simulator: `nfl_ats.pool`; validation: `tests/test_pool.py`; runs:
  `scripts/pool_levers.py` (`--only <experiment>`), output
  `artifacts/pool_levers/levers.json` (gitignored; regenerate in ~8 minutes).
- Measured inputs: `artifacts/opener_evaluation/20260817T165135Z/per_game.parquet`
  (1,503 resolved opener-graded picks, 107 weeks, 2020–2025) and
  `artifacts/best_pick_ranker/*.picks.parquet`.
- Splash mechanics researched 2026-08-17: Commissioner Handbook (Stats page /
  pick distribution), NFL pick'em contest rules (picks hidden until kickoff),
  contest pages (prize structure, double-points Best Pick). None of it is
  user-confirmed; §6 item 4 exists to replace research with observation.
- Field behaviour: Levitt, *Why Are Gambling Markets Organised So Differently
  From Financial Markets?* (Economic Journal, 2004) — a real-money NFL contest
  field leaning heavily to favourites and winning ~49.5%.
- Betting-percentage vendors surveyed 2026-08-17: the-odds-api.com v4 docs
  (no percentages), Action Network (PRO paywall, no API), ScoresAndOdds and
  OddsShark (free, scrape-only, no archive), Covers (contest users, not money),
  VegasInsider (handicapper panel, not money), Sports Insights / Bet Labs (paid,
  no clean REST API).
