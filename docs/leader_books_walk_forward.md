# Is the leader set real, or is it hindsight? (lane AT)

Lane AT, 2026-09-10. Everything above the `## Results` line was written and
frozen **before** any accuracy, paired delta, interval or `probability_positive`
was computed. The counts quoted inside the predeclaration were measured first,
deliberately, from population membership, quote timestamps and market movement
only; no outcome column (`margin_vs_open`, `correct_*`) was read until the design
was fixed.

## Binding closing-grounds taxonomy (AGENTS.md), verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". Within-week
correlation is ZERO by owner mandate; the bootstrap is week-blocked and no ICC
term is estimated or padded. The pool is FORCED PICKS, so the decision is
expected value: the refresh rule with the higher expected accuracy at the
deadline is served.

## The question

The served late-week rule follows the **median** Wednesday-to-deadline move of
three "leader" books -- Bovada, William Hill (US), MyBookie
(`LEADER_BOOKS` in `src/nfl_ats/sharp_book_movement_features.py`) -- at a full
point (`LEADER_FOLLOW_THRESHOLD = 1.0`), with an injury-news veto
(`docs/late_week_refresh.md`, `docs/sharp_weighted_follow.md`,
`docs/follow_threshold_live_card.md`).

Those three books were identified in `docs/book_leadership.md` on the **same**
2023-2025 hourly archive the follow rule was then scored on. The leader choice
is therefore **in-sample**. This lane asks the one question that separates two
very different worlds:

- **the edge is a property of those three books** -- their feeds genuinely lead,
  stably, and naming them is a durable modelling choice; or
- **the edge is a property of "whichever books led in the past"** -- in which
  case a walk-forward set re-derived from prior data only should do at least as
  well, and the served triple is a hindsight artifact that happens to name the
  right books because it was allowed to look at the answer.

Those two worlds make an identical prediction on the in-sample window and
opposite predictions about what to serve in 2026, which is why the walk-forward
arms below are the whole experiment.

## Population (frozen)

- **Window.** 2023-2025, the only seasons with the hourly odds archive.
  **Measured**: 816 regular-season archive games, 54 `(season, week)` blocks,
  **799** opener-scored (17 opener pushes dropped -- not scoreable either way).
- **Frame.** Reused row for row from
  `artifacts/follow_news_gate/20260910T002019Z/frame.parquet`, the same frame
  `docs/served_refresh_card.md`'s served-card replay consumed, so the baseline
  is the **nine-member Tuesday card** (`tuesday_pick_home`, policy
  `overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`)
  and not a fresh composition.
- **Grading.** Forced picks at `margin_vs_open` (the frozen Tuesday opener),
  game-weighted, active model `c657058903f3232b`, `gaussian_median`, from
  `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`.
- **Market.** The `intraday_hourly` historical-backfill archive, cached at
  `artifacts/experiments/sharp_book_movement/quotes.parquet`, replayed through
  the frozen served filter chain of
  `nfl_ats.sharp_book_movement_features.sharp_book_movement_features` with each
  game's cutoff at its own pick deadline `min(kickoff, that week's Sunday 16:00
  ET)`.

### What the archive supports, stated before the design uses it

**Measured** (population and timestamps only, no outcome read): the market
archive holds spread quotes for **2023, 2024 and 2025 and no earlier season**.
Two direct consequences, both accepted rather than worked around:

1. **W1's 2023 season is a no-op.** There is no 2022 archive to derive a
   2023 leader set from, so on 2023 the W1 arm cannot fire at all and every
   pick is the Tuesday card's. 2024 derives its set from 2023; 2025 derives its
   set from 2023 and 2024 pooled. W1 therefore has **two** live seasons, not
   three, and its pooled three-season number is diluted by a third of the window
   in which it is deliberately inert. That dilution is reported, never hidden,
   and the per-season rows are the honest read.
2. **W2's earliest weeks are no-ops** for the same reason at week resolution:
   week 1 of 2023 has no prior week, and the first few weeks after it do not
   reach the participation floor below. The count of no-op weeks is reported.

**Measured**, per-book move participations available to a derivation window
(no outcome read): 2023 gives 11 of the 12 books a participation count between
607 and 2,241; 2024 gives 10 books between 464 and 1,832; 2025 gives 11 books
between 405 and 2,698. Week 1 of 2023 alone gives its busiest book 106
participations and its third-busiest 67. **Two of the twelve books do not span
the window**: `pointsbetus` and `fanatics` each carry a net move on 272 of the
816 games (one season each). A walk-forward set derived on prior data can
therefore name a book that has since left the feed; when that happens the arm's
median is taken over whichever of its books actually contributed an increment,
and if none did the arm cannot fire. That is a real property of walk-forward
serving, not a defect to patch, and it is measured rather than assumed.

## The leadership statistic (unchanged from `docs/book_leadership.md`)

For one game, consecutive snapshots by `observed_at_utc` where at least one
book's `home_spread_line` differs from its own value in the previous snapshot
form a **move event**. Among books that changed in that event, the book(s) with
the earliest `bookmaker_last_update_utc` split one credit evenly; every book
that changed earns a participation. **Leadership share = credits /
participations.** This is exactly `scripts/book_leadership.py`'s
`score_leadership`, reimplemented here only so it can be run on an arbitrary
prior-data slice instead of the whole archive.

**Candidate set.** The twelve books of `LEADERSHIP_WEIGHTS` -- the population
the served follow rule reads at all. Smaller books (barstool, superbook,
twinspires, unibet_us, wynnbet) are outside the served movement population and
are outside every arm here.

**Eligibility floor: 200 move participations** in the derivation window.
Predeclared, and its only purpose is to stop a three-game slice from crowning a
book on noise; **measured** above, it excludes no book in any full-season
derivation window and bites only in W2's earliest weeks.

**Selection.** Top three eligible books by leadership share; ties broken by
participations, then alphabetically. Fewer than three eligible books makes that
block a no-op.

## The arms (frozen)

Every arm is a pick side per game and every arm uses the **same** fire rule, so
the only thing that varies across W0-W4 is **which books the median is taken
over**:

> Let `net` be the median of that arm's book set over the books that contributed
> an increment for that game. If no arm book contributed, the arm cannot fire and
> the Tuesday pick stands. If `|net| >= 1.0` the pick becomes the side the market
> moved toward (`net > 0` picks HOME, else AWAY). Otherwise the Tuesday pick
> stands.

| Arm | Book set |
| --- | --- |
| **W0 served** | The fixed `LEADER_BOOKS` triple, exactly as served. |
| **W1 season walk-forward** | Top three by leadership share re-derived at the start of each season from **strictly prior seasons** in the archive. 2023 is a no-op. |
| **W2 week walk-forward** | Top three re-derived before each `(season, week)` from **all strictly prior weeks**, expanding across the season boundary. |
| **W3 all twelve** | All twelve `LEADERSHIP_WEIGHTS` books -- the median with no leadership selection at all. |
| **W4 single book** (twelve arms) | One book alone. **Attribution only**: these twelve are a mined set and are reported as a decomposition of where the triple's move comes from, never as candidates to serve. |

**The injury-news veto is OFF on every arm, including W0.** The served rule has
it on. It is switched off here so that W0-W4 differ in exactly one thing -- the
book set -- and the comparison answers the question asked. W0 as measured here
is therefore **not** the served pick chain, and no number in this lane may be
quoted as the served chain's accuracy; `docs/served_refresh_card.md` remains the
place that measures the served object.

## Statistics, declared before the numbers

- Paired deltas in accuracy points, game-weighted, forced picks, opener-graded,
  opener pushes dropped (799 games).
- `nfl_ats.overlay_composition.blocked_bootstrap_matrix`, paired, **20,000
  samples, seed 20260821**, **week-blocked** (blocks are `(season, week)`) and
  **season-blocked**. Within-week correlation is ZERO; no ICC is estimated. The
  season-blocked column on a three-season window has three blocks, so its
  `probability_positive` saturates and carries little information; it is
  reported for completeness, never as a second opinion.
- Every arm is compared **paired against W0** and **paired against the Tuesday
  card**, pooled over 2023-2025 and then per season.
- Reported for every arm: fires, picks changed against the Tuesday card, picks
  changed against W0, and `probability_positive`. The binary "contains zero" is
  never used as a verdict.
- **Set agreement**, the descriptive object this lane exists to produce: for
  every W1 and W2 block, whether the re-derived triple is exactly the served
  triple, and how many of its three members the served triple contains.

## Positive control

Perfect foresight, two ways, on the same blocks and the same bootstrap:

- **PC_W0changes** -- perfect foresight on exactly the games W0 changes against
  the Tuesday card. This is the ceiling any book-set choice could reach on W0's
  own decision set.
- **PC_reach** -- perfect foresight on every game where at least one of the
  twelve books contributed an increment.

If a walk-forward arm's shortfall against W0 sits far inside PC_W0changes' own
interval, the instrument is not proven able to resolve a difference that size,
and nothing here may be recorded `bounded_by_control`.

## Reproduction checks, fixed before scoring

**Measured** before the design was frozen, from the market replay alone:

1. The per-book replay's median over `LEADER_BOOKS` must equal the frame's own
   `leader_net` on all 816 rows. **Max absolute difference 0.0, mismatches 0.**
2. The per-book replay's mean over all twelve books must equal the frame's own
   `equal_net`. **Max absolute difference 0.0.**
3. The Tuesday card must score **56.070%** on the 799 scored games, the figure
   `docs/served_card_harness.md`, `docs/follow_news_gate.md` and
   `docs/served_refresh_card.md` all publish. A mismatch stops the lane.
4. W0 must reproduce **228 fires** at 1.0, the count
   `docs/follow_threshold_live_card.md` and `docs/served_refresh_card.md` both
   publish.

## The decision line, declared before the numbers

**Does the served fixed-leader rule beat the walk-forward-derived one?** The
answer is `probability_positive` on the paired deltas `W0 - W1` and `W0 - W2`,
reported as a number and never as a binary. Three readings are possible and all
three are stated in advance so none can be chosen after the fact:

- W0 clearly ahead of both walk-forward arms: the leader set is doing work a
  prior-data rule cannot recover, and the in-sample worry is that the *selection*
  is worth points, which on a forward card it will not be.
- The walk-forward arms level with or ahead of W0: the edge is "whichever books
  led recently", the served triple is one draw from that rule, and serving the
  rule instead of the triple is strictly more robust. In that case this document
  states **exactly** what wiring it would take -- the function, and where the set
  would be refreshed each Tuesday -- and **wires nothing**.
- W3 (no leadership at all) level with the leader arms: the median over any
  reasonably sized book set is the operative object and "leadership" is
  decoration. Reported plainly if it happens.

**Nothing in this lane touches a served rule, a ledger, a manifest, a forecast
or a published page.**

## Registry

Every cell is recorded through `nfl-ats weak-signals record` under names
`leader_books_walk_forward_<arm>_<window>`, units `accuracy_points`, league
`nfl`, family `leader_books_walk_forward`, one invocation at a time with
`NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry`, argv lists saved to
`record_commands.json`. The family is declared here, before the signs were seen.
Its members share one window, one grading and one market replay, so they are
**correlated readings of one question** rather than independent votes, and they
are a correlated decomposition of `sharp_weighted_follow_*`,
`follow_threshold_live_card_*`, `follow_news_gate_*` and `served_refresh_card_*`.
**Never pool any of these additively with each other or with those parents.**
The twelve W4 single-book arms are explicitly a mined battery; no multiplicity
correction is claimed and they are attribution only.

---

## Results

**Measured** this session,
`artifacts/leader_books_walk_forward/20260910T032517Z/` (`frame.parquet`,
`leadership_ledger.parquet`, `coverage.json`, `results.json`, `extras.json`,
`cells.csv`, `record_commands.json`, the four scripts that produced them, and a
copy of this predeclaration frozen at scoring time). The
`predeclaration_sha256`
`ff01b5ab91152637189114a798328ff65a34542cf62540424299942872083026` is the hash
of everything above this line, taken before any script ran.

### The decision, first

**The three named books are not what the edge is made of, and neither is
"leadership".** Two findings carry that, and they point the same way.

1. **The exact triple is a hindsight artifact of the window.** Run the same
   leadership statistic on prior data only and it returns a **different**
   triple in **52 of 54 weeks** and in **both** live walk-forward seasons.
   Bovada and MyBookie survive every re-derivation; **William Hill does not** --
   its leadership share is **0.462 in 2023** (5th of 12) and **0.720 / 0.754 in
   2024 / 2025** (2nd), so a rule that only knows the past puts PointsBet in
   its place. The served set is exactly what the statistic returns when it is
   allowed to see all three seasons (**measured**: bovada 0.6819, mybookieag
   0.6705, williamhill_us 0.6678 -- the top three of the pooled table, which
   is the check that this lane's reimplementation is the one that produced
   `LEADER_BOOKS`).
2. **The side never depends on which triple you use.** On every game where the
   served arm and a walk-forward arm BOTH fire -- 106, 186 and 214 games for
   W1, W2 and W3 -- they point at the **same side, 0 disagreements out of 506**.
   The entire measured difference between the arms is **how often they fire**,
   not which way.

**So the served rule's advantage is reach, not book identity.** W0 fires 228
times of 816; the season walk-forward fires 110, the week walk-forward 198, the
twelve-book median 221. Paired at the opener on the 799 scored games:

| Comparison | Delta (pts) | 95% week-blocked | `probability_positive` (arm ahead) | P+ (**W0** ahead) |
|---|---:|---|---:|---:|
| W1 - W0, season walk-forward | **-0.501** | [-1.963, +0.991] | 0.2467 | **0.7533** |
| W2 - W0, week walk-forward | **-0.626** | [-1.754, +0.497] | 0.1364 | **0.8636** |
| W3 - W0, all twelve books | **-0.250** | [-0.875, +0.371] | 0.2074 | **0.7926** |

**On a forced card the honest read is that this is close.** W0 is ahead of the
best walk-forward arm at about **86/14** and ahead of the no-leadership median
at about **79/21**, on paired deltas of a quarter to two-thirds of an accuracy
point. Nothing here is resolved, nothing is closed, and the served rule keeps
being the right thing to play -- but the reason to keep it is expected value on
a small margin, **not** the belief that Bovada, William Hill and MyBookie are
individually special. **Measured**: on 2025 alone the week walk-forward arm
**beats** W0 (+0.749 points, week 95% [0.000, +1.901], `probability_positive`
0.9402).

### Reproduction: all four checks pass

**Measured**, `coverage.json` and `results.json`:

| quantity | required | this run |
|---|---:|---:|
| archive games 2023-2025 | 816 | **816** |
| opener-graded (non-push) | 799 | **799** |
| `(season, week)` blocks | 54 | **54** |
| per-book replay's leader median vs the served `leader_net` | 0.0 max abs diff | **0.0** |
| per-book replay's twelve-book mean vs the served `equal_net` | 0.0 max abs diff | **0.0** |
| W0 fires at 1.0 | 228 | **228** |
| Tuesday card accuracy | 56.070% | **56.070%** |

W0 additionally reproduces lane AJ's own "follow at 1.0, no veto" reference row
exactly -- **100** picks changed, **57.322%**, **+1.252** points, week 95%
[-1.122, +3.713], `probability_positive` 0.8462 -- which is the check that this
lane's W0 and `docs/served_refresh_card.md`'s C2-noveto arm are the same object.

### What each walk-forward rule actually chose

**Measured**, `coverage.json`. Leadership share is credits / participations on
the prior slice only.

| Derivation window | Leader set chosen | Members shared with the served triple |
|---|---|---:|
| 2023 from prior seasons | **no-op** (archive has no 2022) | -- |
| 2024 from 2023 | pointsbetus, bovada, mybookieag | 2 of 3 |
| 2025 from 2023-2024 | mybookieag, bovada, pointsbetus | 2 of 3 |
| whole archive (in-sample) | bovada, mybookieag, williamhill_us | **3 of 3** |

Week-level (W2), over 54 weeks: **35** weeks choose {bovada, mybookieag,
pointsbetus}, **11** choose {fanatics, mybookieag, williamhill_us}, **3** choose
{bovada, fanatics, mybookieag}, **2** choose the served triple exactly, and
**3** are no-ops (weeks 1-3 of 2023, below the participation floor). Overlap
with the served triple is 2 members in **49** weeks, 3 in **2**, 0 in **3**.

### The hazard that costs the season arm most of its reach

**Measured**: `pointsbetus` carries a net move on **272 of 816** games -- all of
them in **2023** -- and `fanatics` on 272, all in **2025**. So the leader set
that 2023's data crowns for 2024 names a book that has **left the feed**, and
the arm's median silently becomes a two-book mean for both live seasons
(**measured**: W1 contributes exactly 2 books on all 544 games of 2024-2025,
never 3). A two-book mean lands on the quarter-point lattice and reaches a full
point far less often: W1's `|net| >= 1.0` share is **13.5%** against W0's
**27.9%**, W2's 24.3% and W3's 27.1%.

That is why the season arm loses ground, and it is a **property of walk-forward
serving**, not of walk-forward selection. Any served version needs a liveness
check; the arms here deliberately do not have one, so the cost is visible.

### Accuracy at the Tuesday opener, 799 games

**Measured**, `results.json` -> `accuracy_by_window`.

| Arm | 2023-2025 | 2023 | 2024 | 2025 | Fires | Picks changed vs card |
|---|---:|---:|---:|---:|---:|---:|
| Tuesday nine-member card | 56.070% | 59.398% | 54.135% | 54.682% | -- | -- |
| **W0** served triple | **57.322%** | 58.647% | 56.015% | 57.303% | 228 | 100 |
| **W1** season walk-forward | 56.821% | 59.398% | 54.887% | 56.180% | 110 | 52 |
| **W2** week walk-forward | 56.696% | 57.143% | 54.887% | **58.052%** | 198 | 88 |
| **W3** all twelve books | 57.071% | 58.647% | 56.015% | 56.554% | 221 | 98 |
| control: perfect foresight on W0's own 100 changes | 62.954% | 66.165% | 62.030% | 60.674% | -- | 55 |
| control: perfect foresight everywhere | 100% | | | | -- | 351 |

W1 is inert on 2023 by construction, so its 59.398% there is the card's own
number, not a measurement.

### Paired deltas, 2023-2025, and by season

Week-blocked interval quoted; 20,000 samples, seed 20260821. The season-blocked
column has three blocks, so its `probability_positive` saturates and carries
almost no information -- reported in `cells.csv`, never as a second opinion.

| Cell | Changed | Delta (pts) | 95% week | P+ |
|---|---:|---:|---|---:|
| **W0** vs the Tuesday card | 100 | **+1.252** | [-1.122, +3.713] | **0.8462** |
| W1 vs the Tuesday card | 52 | +0.751 | [-1.122, +2.736] | 0.7755 |
| W2 vs the Tuesday card | 87 | +0.626 | [-1.990, +3.270] | 0.6783 |
| W3 vs the Tuesday card | 98 | +1.001 | [-1.389, +3.491] | 0.7862 |
| **W1 vs W0** | 52 | **-0.501** | [-1.963, +0.991] | **0.2467** |
| **W2 vs W0** | 29 | **-0.626** | [-1.754, +0.497] | **0.1364** |
| **W3 vs W0** | 8 | **-0.250** | [-0.875, +0.371] | **0.2074** |
| control: perfect foresight on W0's changes, vs W0 | 45 | +5.632 | [+3.985, +7.412] | 1.0000 |
| control: perfect foresight everywhere, vs the card | 351 | +43.930 | [+40.511, +47.475] | 1.0000 |

By season, against W0: **W1** reads +0.752 (2023, a no-op comparison), -1.128
(2024), -1.124 (2025); **W2** reads -1.504, -1.128, **+0.749**; **W3** reads
0.000, 0.000, -0.749. Magnitudes per era, never absence.

**Post-hoc window, added after the predeclaration was frozen and labelled as
such** because W1 is inert on a third of the pooled window by construction. On
the two live seasons (2024-2025, 533 scored games): W0 is **56.660%** against
the card's 54.409% (**+2.251**, [-0.755, +5.333], P+ 0.9259); **W1 - W0 =
-1.126** ([-2.403, 0.000], P+ 0.0293, so P+ W0 ahead **0.9707**); **W2 - W0 =
-0.188** ([-1.318, +0.772], P+ 0.3868, so P+ W0 ahead **0.6132**); **W3 - W0 =
-0.375** ([-1.128, +0.371], P+ 0.1567). On the seasons where the season arm is
actually alive it loses more than the pooled number suggests, and the week arm
is close to a coin flip against the served triple.

### Where the fire gap sits, and what W0 does with it

**Measured**, `extras.json` -> `fire_gap_diagnostics`. On the games where the
served arm fires and the walk-forward arm does not, the median `|net|` is
exactly **1.0** -- the gap lives at the threshold, which is what a diluted or
shorter median does to a half-point lattice.

| Arm | Only W0 fires | ...W0 changes the pick | W0's record there | Only the arm fires | ...it changes the pick | Its record |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 121 | 50 | **27-23** | 4 | 2 | 1-1 |
| W2 | 42 | 21 | **13-8** | 11 | 8 | 4-4 |
| W3 | 14 | 5 | 2-3 | 7 | 3 | 0-3 |

On its own 100 changed picks W0 goes **55-45** and the Tuesday card goes 45-55.
The walk-forward arms go **29-23** (W1, 52 changes) and **46-41** (W2, 87
changes); the twelve-book median goes 53-45 on 98.

### Single books alone -- attribution only, a mined battery

**Measured**, twelve arms at the same full point against the Tuesday card,
2023-2025 pooled. These are not candidates to serve and no multiplicity
correction is claimed.

| Book alone | Accuracy | Delta vs card (pts) | 95% week | P+ | Fires |
|---|---:|---:|---|---:|---:|
| betrivers | 57.196% | **+1.126** | [-1.409, +3.755] | 0.7996 | 229 |
| mybookieag | 57.196% | **+1.126** | [-1.145, +3.524] | 0.8242 | 224 |
| betus | 56.696% | +0.626 | [-1.918, +3.287] | 0.6855 | 246 |
| betmgm | 56.696% | +0.626 | [-1.827, +3.110] | 0.6890 | 220 |
| lowvig | 56.571% | +0.501 | [-1.906, +2.993] | 0.6529 | 243 |
| williamhill_us | 56.446% | +0.376 | [-2.038, +2.857] | 0.6183 | 231 |
| bovada | 56.446% | +0.376 | [-2.253, +3.038] | 0.6056 | 239 |
| draftkings | 56.446% | +0.376 | [-2.163, +2.901] | 0.6139 | 243 |
| betonlineag | 56.320% | +0.250 | [-2.267, +2.919] | 0.5690 | 248 |
| fanatics | 56.320% | +0.250 | [-1.125, +1.663] | 0.6314 | 75 |
| fanduel | 55.820% | -0.250 | [-2.632, +2.044] | 0.4247 | 252 |
| pointsbetus | 55.319% | -0.751 | [-2.396, +0.886] | 0.1872 | 96 |

**Stated plainly because it bears directly on the question**: `betrivers` --
leadership share 0.532, sixth of twelve, not a leader book by this statistic --
scores exactly what `mybookieag` scores alone, and **more than Bovada or
William Hill alone**. The two highest single-book arms are one leader and one
non-leader. If the edge were a property of the three named feeds, that is not
what this table would look like.

### Positive controls

Perfect foresight on exactly the 100 picks W0 changes -- which differs from the
Tuesday card on **55** of them and from W0 itself on 45 -- is worth **+6.884
accuracy points over the Tuesday card, 95% [+5.263, +8.582]**, and **+5.632
against W0 itself, [+3.985, +7.412]**; perfect foresight on every reachable
game is **+43.930**. So the instrument is **proven** able to resolve effects of
roughly six to forty-four points, and is **not** proven able to resolve the
0.19 to 0.63 points that separate the book sets. **No cell in this battery is
`bounded_by_control`**: the control detected an effect and the arms are
present, not absent.

### Nothing here closes anything

All 75 recorded cells are `unresolved_below_power` with a null closing ground.
Not one interval sits wholly on the wrong side of zero, no split-half
reliability is zero, and the controls above are far too coarse to bound a
half-point difference. The two-line answer to the lane's question is a
*direction* on a forced-pick decision, never a refutation of anything.

### What serving a walk-forward leader set would take (NOT wired)

Stated because the week walk-forward arm is close to a coin flip against the
served triple on the live seasons (P+ 0.6132 for W0) and the twelve-book median
is within a quarter of a point, so a future session may want it. **This lane
changed none of it.**

1. **The constant becomes a parameter.**
   `sharp_book_movement_features.LEADER_BOOKS` is read at exactly two places on
   the served path -- `sharp_book_movement_features.py:99` (the
   `leader_move_observed` flag) and `:111` (the `leader_median_net_move` /
   `leader_books` aggregation). Serving a derived set means
   `sharp_book_movement_features(quotes, games, *, leader_books=LEADER_BOOKS)`,
   threaded through `late_week_follow_frame(..., leader_books=...)` to its one
   served caller, `nfl_ats.pick_refresh.plan_refresh`. Nothing else on the
   served path reads it (`scripts/sharp_weighted_follow.py` is a measurement
   script).
2. **The derivation function already exists in this artifact.**
   `lane_at_frame.py`'s `leadership_ledger` (the `book_leadership.py`
   statistic decomposed to `(season, week, book)` so any prior slice is a
   cumulative sum), `counts_from_ledger`, and `select_leaders` (participation
   floor 200, top three by share, ties by participations then name, fewer than
   three means no-op) are the whole rule. Promoting it means moving those three
   functions into `src/nfl_ats/` beside the constant they would replace.
3. **Where the set would be refreshed.** Tuesday, on the lock chain, between
   the `odds_tue_open` capture (12:05) and `weekly_lock` (12:20) in
   `scripts/capture_scheduler.py`'s `SCHEDULE` -- either a new job or a step
   inside `scripts/scheduled_weekly_lock.py` -- derived from quotes strictly
   before the current week and written to a dated artifact
   (`artifacts/book_leadership/<stamp>/served_leader_books.json`) that
   `plan_refresh` reads at each refresh pass, so the set the week is served
   under is frozen at the same instant as the grading lines.
4. **Two fail-closed guards, both named by measurements above.** A **liveness**
   check -- a book with no quote rows in the trailing window is dropped and the
   next eligible book takes its place, or the PointsBet failure repeats and a
   three-book median silently becomes a two-book mean -- and a **floor**: fewer
   than three eligible live books falls back to the current constant rather
   than narrowing the median.

### Caveats (label how you know it)

- **Historical-backfill replay, not the live path.** Everything is replayed
  from the `intraday_hourly` archive because the live store starts 2026-08-17.
  Evidence about the rule's SHAPE, not a re-grade of any promotion.
- **The injury-news veto is OFF on every arm, W0 included**, so W0 here is the
  follow rule's book set in isolation and is **not** the served pick chain. No
  number in this document may be quoted as the served chain's accuracy.
- **Three seasons is one walk-forward step for W1 and 51 for W2.** The season
  arm has two live seasons, and its 2023 cell is a construction no-op with an
  exactly-zero delta and `probability_positive` 0.5 -- recorded as such, and
  the registry refused it until the zero standard error was omitted rather
  than stored.
- **The participation floor of 200 is predeclared but arbitrary in the small.**
  It excludes no book in any full-season derivation and bites only in weeks 1-3
  of 2023.
- **Mined battery**: 75 cells over one window, one grading and one market
  replay, twelve of the arms single books. No multiplicity correction claimed,
  and never pooled additively with the parent families named in Registry.
- **The opener evaluator's inherited approximation applies**: only
  `spread_line` is swapped to the opener (`docs/opener_evaluation.md`).

## Commands run

```
python artifacts/leader_books_walk_forward/20260910T032517Z/lane_at_frame.py
python artifacts/leader_books_walk_forward/20260910T032517Z/lane_at_measure.py
python artifacts/leader_books_walk_forward/20260910T032517Z/lane_at_extra.py
python artifacts/leader_books_walk_forward/20260910T032517Z/lane_at_record.py
nfl-ats weak-signals record ...   (75 cells; argv lists in record_commands.json)
```

## Registry, measured after recording

**Measured**: **75** entries under the `leader_books_walk_forward_` prefix, all
`unresolved_below_power`, all with a null closing ground, in a registry then
holding 5,238 signals. Exact argument vectors in
`artifacts/leader_books_walk_forward/20260910T032517Z/record_commands.json`, run
one at a time with `NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry`. One cell
(`leader_books_walk_forward_w1_vs_card_2023`) was refused on its first
invocation for a non-positive standard error and re-run without that optional
field; the corrected argv and the reason are both in that file.

