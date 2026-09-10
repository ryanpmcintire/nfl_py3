# The late-week follow's threshold as a function of the line (AK-2, re-measured against the served 1.0)

Closing-grounds taxonomy, verbatim, because this document reports intervals: an
interval or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (b) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

Within-week correlation is ZERO by owner mandate; every bootstrap here is
week-blocked, with the season-blocked reading alongside. Decide on expected
value: the pool is forced picks, so the arm with the higher expected opener
accuracy through the played card is the one to serve, and a predeclared bar
governs only what this document may CLAIM.

## Why this lane exists

`docs/line_size_shrinkage.md`'s AK-2 arm measured a line-size-dependent follow
threshold and read **+0.876 accuracy points, `probability_positive` 0.8256**
through the played card — but it measured that against a served rule that fired
at a FIXED **0.5** points (read: `docs/line_size_shrinkage.md:391-422`). The
served constant moved to **1.0** the next day
(`sharp_book_movement_features.LEADER_FOLLOW_THRESHOLD = 1.0`, read:
`src/nfl_ats/sharp_book_movement_features.py:26`; promotion recorded in
`docs/follow_threshold_live_card.md:370-382`), and an injury-news veto was added
inside the follow branch (read: `docs/late_week_refresh.md:551-601`,
`src/nfl_ats/pick_refresh.py:1546-1566`). AK-2's baseline no longer exists.
**The +0.876 is therefore a measurement against a retired rule and cannot be
quoted as a live lead**; this lane re-measures the same question against what is
actually served.

The mechanism the line-size threshold would encode comes from
`docs/big_spread_signal.md`: on `|line| >= 10.5` the market's own opener-to-close
move is the strongest single signal on the board — **62.96%** right when
followed, against the served model's 47.69% on the same games (read:
`docs/line_size_shrinkage.md:373-384`, reproducing `docs/big_spread_signal.md:431`).
If big-spread market moves carry more information than small-line moves, the
gate should be LOWER on big spreads, not the same everywhere.

Against that sits `docs/follow_threshold_live_card.md`'s band decomposition,
which is why the served gate is at a full point at all: pooled over all lines,
the market side loses the reversals it makes inside the 0.5-to-0.75 band
(42-46%) and wins them at a full point or more (52-59%) (read:
`docs/follow_threshold_live_card.md:304-323`). **The two facts are only
compatible if the losing low band is concentrated at small lines**, which is
exactly the quantity this lane measures.

## Design, frozen before any candidate was scored

Family `follow_threshold_by_line`.

### Population and surface

- **Population.** The 2023-2025 `intraday_hourly` historical-backfill archive
  under `data/market/raw`, loaded with `nfl_ats.clv.load_decision_quotes`
  (`capture_kind=historical`, label `intraday_hourly`), joined to the opener
  archive of the ACTIVE model, `artifacts/opener_evaluation/20260910T005854Z`
  (`2e8c616b476dd0d2`, weak_stack market-residual ridge, alpha 10,
  `gaussian_median`). Expected: 816 archive games in window, **799 opener-scored**
  (17 opener pushes dropped), 54 week blocks, 3 season blocks.
- **Stated deviation, up front.** The served path loads `capture_kind="live"`
  and the market store holds no live capture before 2026-08-17, so 2023-2025 is
  reachable only through the historical backfill. Same substitution
  `docs/served_card_harness.md`, `docs/follow_vs_tilts.md` and
  `docs/follow_threshold_live_card.md` document; this is evidence about the
  rule's SHAPE, not a re-grade of a promotion.
- **Surface: the played card only.** The served nine-member joint OR
  `overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`,
  rebuilt with `nfl_ats.unserved_tilt_marginals.served_card_flip_set(card="served")`.
  There is no raw-surface column in this lane's decision table; the raw surface
  is reported once, as a reconciliation row, and never as the decision.
- **Follow replay.** `nfl_ats.sharp_book_movement_features.late_week_follow_frame`
  — the frozen function the served refresh itself calls — with each game's cutoff
  at its own pick deadline `min(kickoff, that week's Sunday 16:00 ET)`.
  `leader_median_net_move` is the read for every arm; the equal-book mean is not
  an arm here.
- **Grading.** Forced picks at the frozen Tuesday OPENER (`margin_vs_open`),
  opener pushes dropped, game-weighted accuracy points.

### The three reproduction checks, run before any arm is read

1. **The archive is the active model's.** `artifacts/opener_evaluation/20260910T005854Z`
   joined game for game against `20260909T183120Z` (the archive the two prior
   follow lanes used) on all 1,537 games: the maximum opener-probability gap and
   the maximum `margin_vs_open` gap are reported. If they are not zero, every
   prior-lane number quoted here is restated on this archive before use.
2. **The card.** The rebuilt nine-member card must score **56.070%** on the 799
   scored games (`docs/served_card_harness.md`,
   `docs/follow_threshold_live_card.md:204-207`).
3. **The follow frame.** The rebuilt `leader_median_net_move` must match, game
   for game, the frame `artifacts/follow_news_gate/20260910T002019Z/frame.parquet`
   already built for the news lane, and the served 1.0 gate must fire on
   **228** games (`docs/follow_threshold_live_card.md:243`). The news columns are
   JOINED from that frame rather than re-derived, which is admissible precisely
   because the news reading is defined from `sign(leader_median_net_move)` and
   the injury snapshot alone (read:
   `src/nfl_ats/injury_signal_refresh_tilt.py:502-566`) — it does not depend on
   the threshold, the model or the card, so it is identical for every arm.

### Line bands

`|tue_open_home_spread|`, four bands, fixed here: **0-3**, **3.5-6.5**,
**7-10**, **10.5+**. These are the prompt's declared bands and they are the same
cut points `docs/big_spread_signal.md` and `docs/line_size_shrinkage.md` used,
so the 10.5+ cell is commensurable with the diagnosis that motivated the lane.

### The news veto, applied identically to every arm

Exactly the served rule (`docs/late_week_refresh.md:551-601`): when an arm's gate
fires and post-Tuesday injury news points AGAINST the move, the market side is
discarded and the card's Tuesday pick stands. Reader:
`injury_signal_refresh_tilt.follow_news_for_game`, official rows at
`INJURY_NET_THRESHOLD = 2.0` where the season's rows carry a usable timestamp,
ProFootballTalk headlines at `PFT_NET_THRESHOLD = 1.0` where they do not,
fail-open everywhere. In this lane's frame that is
`contradict = (news_readable & news_toward_market <= -2.0) | (~news_readable &
pft_toward_market <= -1.0)`, the `f3p` construction of
`artifacts/follow_news_gate/20260910T002019Z/laneAC_measure.py:92-93`. **A vetoed
game counts as "the gate fired"**, exactly as served, so it never falls through
to anything below.

### The arms

Let `net` be the leader-median Wednesday-to-deadline move, `L = |line|`, and
`t` the arm's threshold for that game. An arm fires when `|net| >= t` and at
least one leading book contributed an increment; on a fire it takes the side
the market moved toward (`net > 0` picks HOME) unless the veto discards it;
otherwise the card's own pick stands.

| Arm | Threshold rule |
| --- | --- |
| **N0** | Reference. No follow at all — the card's own pick on every game. |
| **T0** | **The served rule.** `t = 1.0` at every line. Veto on. This is the baseline every candidate is paired against. |
| **T05** | Reference. `t = 0.5` at every line — the gate retired on 2026-09-10, so the table connects to `docs/line_size_shrinkage.md`'s AK-2 baseline. |
| **T1** | `t(L)` chosen per band from the grid **{0.5, 1.0, 1.5}**, fitted walk-forward: for each scored week and each band, the grid value with the highest forced-pick accuracy over that band's games in **strictly prior weeks** of the window (expanding window), **ties — including an empty prior window — resolve to 1.0**, which is the served value. |
| **T1F** | T1 with a declared floor: a band runs at the served 1.0 until it has **50** scored games in strictly prior weeks. Declared now as a sensitivity on T1's small-prior-window noise, not chosen afterwards. |
| **T2** | `t(L) = a + b·L`, clipped to `[0, 3]`, fitted walk-forward on prior-window accuracy over a deterministic grid `a ∈ {0.00, 0.25, …, 2.00}` (9 values) × `b ∈ {-0.15, -0.10, -0.05, 0.00, +0.05, +0.10, +0.15}` (7 values), all bands pooled, expanding window, **ties resolve to the null `(a, b) = (1.0, 0.0)`**, which reproduces the served rule exactly. This is the continuous, fitted, walk-forward function AGENTS.md's "no unexplained threshold flips" rule requires of anything that could be served. |
| **T3** | **The mechanism arm, not tuned.** `t = 1.0` everywhere EXCEPT `t = 0.5` at `10.5+`. One number, fixed here, from one named mechanism: on big spreads the market's own move is the strongest signal on the board (62.96% when followed, `docs/big_spread_signal.md:431`), so the gate should be looser there. No fit, no grid, no walk-forward — if it wins it wins on the mechanism and if it loses the mechanism is what lost. |

**A statement about servability, made before the numbers.** T1, T1F and T3 are
step functions of the line and therefore are NOT servable under AGENTS.md's "no
unexplained threshold flips" rule unless a mechanism is named for the step; T3
names one, T1/T1F do not and are diagnostic. T2 is the only arm whose functional
form is continuous and fitted, so it is the only arm this lane could recommend
serving on shape alone. This is stated now so that a favourable T1 cannot be
promoted after the fact on a shape the rule bans.

### Positive control

- **PC_REACH** — perfect foresight on the whole reachable set: every game where a
  leading book contributed and `net != 0`, the oracle takes the side that
  actually covered; elsewhere the card's own pick. Every arm's fire set is a
  strict subset of this set at every threshold in the grid, so this is the
  ceiling on the entire question.
- **PC_T0** — perfect foresight on exactly the games the served 1.0 gate fires
  on, which bounds what any change to the fire set of the served rule can be
  worth in the direction of firing LESS.
- **PC_BIG** — perfect foresight on the reachable set restricted to `10.5+`,
  because that is the band the mechanism points at and the band with the
  fewest games.

If an arm's effect sits well inside a control's own interval, the instrument is
not shown to be blind and nothing here is `bounded_by_control`.

### Statistics

- Paired accuracy-point deltas on the games both arms score, game-weighted.
- `nfl_ats.overlay_composition.blocked_bootstrap_matrix`, paired, **20,000
  samples, seed 20260821**, **week-blocked** (blocks `(season, week)`) and
  **season-blocked** (blocks seasons). No ICC is estimated or padded. The
  season-blocked column has three blocks, so its `probability_positive`
  saturates and carries almost no information — reported for completeness,
  never as a second opinion.
- Windows: 2023-2025 pooled, then 2023, 2024, 2025 separately; and the four
  line bands pooled over 2023-2025.
- Every arm's baseline column is **T0, the served rule**. N0 is additionally
  reported against T0 so "should we follow at all" stays visible.
- Fire and switch counts per band per arm; the fitted `t` curve per season for
  T1, T1F and T2 at `L = 1, 3, 7, 10, 14`.
- `probability_positive` for every cell. The binary "contains zero" is never a
  verdict.

### Decision rule, declared before the numbers

**The arm with the highest paired point estimate against T0 through the played
card over 2023-2025 is the arm with the higher expected opener accuracy, and
that is the decision**, with `probability_positive` reported as the strength of
the call. If that arm is not T0, this document states exactly what serving it
would take — the threshold function's home in
`sharp_book_movement_features` / `pick_refresh`, the policy id, the paired OFF
arm — and **wires nothing**. No code is edited in this lane, and no ledger,
manifest, forecast, board or active-model file is touched.

### Terminal classification, declared before any cell was scored

A cell is `unresolved_below_power` unless exactly one of:

- `wrong_sign_resolved` — BOTH the week-blocked and the season-blocked 95%
  interval sit entirely below zero.
- `positive_control_bound` — the cell's interval excludes the magnitude the
  matching control resolved at, AND that control cell itself resolved.

Nothing else closes a cell. An interval containing zero closes nothing.

### Registry

Every cell is recorded with `nfl-ats weak-signals record`, units
`accuracy_points`, league `nfl`, family `follow_threshold_by_line`, named
`follow_threshold_by_line_<arm>_<window>`, one invocation at a time with
`NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry`, argv lists saved to
`record_commands.json` in the artifact directory. The family is declared here,
before the signs were seen; its members share one population, one grading and
one follow computation, so they are correlated readings of one question rather
than independent votes.

## Results

Every number below is **measured this session** by the commands at the bottom;
the tables live in `artifacts/follow_threshold_by_line/20260910T032331Z/`.
Nothing above this line was edited after scoring; the three deviations found by
measurement are named in their own section below.

### The three reproduction checks pass

1. **The archive is the active model's, and it is the same object the two prior
   follow lanes read.** `artifacts/opener_evaluation/20260910T005854Z` (model
   `2e8c616b476dd0d2`) joined game for game against `20260909T183120Z` on all
   **1,537** games: maximum opener-probability gap **0.0**, maximum
   `margin_vs_open` gap **0.0**, maximum opener-spread gap **0.0**. So the
   active-model archive and the archive `docs/follow_threshold_live_card.md`
   used are identical on every quantity this lane touches, and every number
   quoted from that lane transfers without restatement.
2. **The card.** The rebuilt nine-member card scores **56.0700876%** on the 799
   scored games — `docs/served_card_harness.md`'s figure to the digit — with 260
   composition flips in the window, matching the news lane's own count.
3. **The follow frame.** The served 1.0 gate fires on exactly **228** games,
   reproducing `docs/follow_threshold_live_card.md:243`. Against the news lane's
   frame, **2 of 816** games differ in `leader_median_net_move` — the same two
   the earlier lane found (`2025_04_SEA_ARI`, 0.0 here against 0.5 there, and
   `2025_05_SF_LA`, 2.0 against 1.5) — and **neither changes any arm's
   decision**: SEA_ARI has no reachable move in this rebuild so no threshold can
   fire on it, and SF_LA clears every threshold in the grid under both readings
   with a news delta of exactly zero.

Population as declared: **816** archive games, **799 scored**, 17 opener pushes,
**0** refused quote rows, 54 week blocks, 3 season blocks. Bands, scored:
**0-3 (317), 3.5-6.5 (290), 7-10 (134), 10.5+ (58)**. The news reader is
official on **544** of 816 games and falls back to the headline path on the
rest; it vetoes **75** of the served rule's 228 fires.

### The decision cell: through the played card, at the opener, 799 games

The served rule T0 scores **57.572%**. Week-blocked interval quoted; 20,000
samples, seed 20260821.

| Arm | Fires | Vetoed | Card accuracy | Changed vs T0 | Delta (pts) | 95% week | P+ week | P+ season |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **T1F** per-band, floor 50 | 293 | 94 | **57.947%** | 35 | **+0.375** | [-1.252, +2.107] | **0.6640** | 0.7037 |
| **T3** mechanism, 0.5 at 10.5+ | 241 | 79 | **57.822%** | 6 | **+0.250** | [-0.373, +0.872] | **0.7905** | **0.9817** |
| T1 per-band, no floor | 309 | 99 | 57.697% | 45 | +0.125 | [-1.631, +1.985] | 0.5458 | 0.5948 |
| **T0 served, flat 1.0** | 228 | 75 | **57.572%** | 0 | — | — | — | — |
| T2 continuous a + b·L | 252 | 82 | 56.571% | 30 | -1.001 | [-2.331, +0.254] | 0.0658 | 0.0951 |
| T05 retired flat 0.5 | 522 | 171 | 56.446% | 97 | -1.126 | [-3.585, +1.266] | 0.1831 | 0.1514 |
| N0 no follow at all | 0 | 0 | 56.070% | 64 | -1.502 | [-3.457, +0.372] | 0.0571 | 0.0355 |
| *(post-hoc)* T1F + T3 | 306 | 98 | *58.198%* | 41 | *+0.626* | [-1.121, +2.399] | *0.7539* | *0.7215* |

By season against T0: **T3 is +0.376 (2023), +0.376 (2024), +0.000 (2025)** —
positive or level in all three, negative in none. T1F is -0.376 / +2.256 /
-0.749; T1 -1.504 / +2.632 / -0.749; T2 -1.504 / +0.752 / -2.247; T05 -1.128 /
+1.880 / -4.120; N0 +0.376 / -1.504 / -3.371.

**What the served move from 0.5 to 1.0 already banked.** `docs/line_size_shrinkage.md`'s
AK-2 read +0.876 for a line-size threshold *against the 0.5 gate*. On these same
799 games the flat 1.0 gate now served is **+1.126** over that same 0.5 gate, and
the best line-size arm reaches **57.947%** against the 0.5 gate's 56.446%, a
**+1.501**-point accuracy difference. So most of AK-2's headline was a
0.5-is-too-low finding, and the promotion of the flat 1.0 collected it. **What
is left for line size, on top of what is served, is +0.125 to +0.375 accuracy
points, and that is the honest size of this lane's question.**

### By line band: the mechanism, made visible

Paired against T0 on the same games; the band is `|tue_open_home_spread|` at
the opener.

| Arm | 0-3 (317) | 3.5-6.5 (290) | 7-10 (134) | 10.5+ (58) |
| --- | --- | --- | --- | --- |
| N0 no follow | +1.262, P+ 0.787 | -2.759, P+ 0.037 | **-5.224, P+ 0.017** | -1.724, P+ 0.370 |
| T05 flat 0.5 | **-3.470, P+ 0.015** | +1.034, P+ 0.661 | -2.239, P+ 0.238 | **+3.448, P+ 0.794** |
| T1 | +0.315, P+ 0.613 | +0.345, P+ 0.553 | -1.493, P+ 0.067 | +1.724, P+ 0.672 |
| T1F | +0.631, P+ 0.752 | +0.345, P+ 0.555 | 0.000 dead heat | 0.000 dead heat |
| T2 | 0.000, P+ 0.499 | -2.069, P+ 0.034 | -0.746, P+ 0.388 | -1.724, P+ 0.334 |
| T3 | 0.000 dead heat | 0.000 dead heat | 0.000 dead heat | **+3.448, P+ 0.794** |

**T05's row IS the mechanism experiment**, because T05 is T0 with one number
changed and nothing else: it lowers the gate to half a point in every band at
once. Lowering it costs **-3.470 accuracy points at 0-3** (week
[-6.854, -0.312], season [-8.421, -0.952], both entirely below zero) and buys
**+3.448 at 10.5+** (P+ 0.794). **A half-point drift in the median of three
books is noise on a pick'em and information on a blowout line**, which is the
one sentence this lane set out to test, and it is confirmed at both ends. It is
not monotone in between: 3.5-6.5 reads +1.034 for the lower gate and 7-10 reads
-2.239, so this is a big-spread effect with a small-line cost, not a clean slope.

The 10.5+ band carries **58 scored games and 6 changed picks**, so it is thin;
that is stated as its size, not as a verdict.

### The fitted t curves, which are the fit made visible

Mean fitted threshold per season, walk-forward, at `|line| = 1, 3, 7, 10, 14`:

| arm | season | t(1) | t(3) | t(7) | t(10) | t(14) |
|---|---|---:|---:|---:|---:|---:|
| T1 | 2023 | 1.194 | 1.194 | 0.972 | 0.972 | 0.972 |
| T1 | 2024 | **1.500** | **1.500** | 1.000 | 1.000 | **0.500** |
| T1 | 2025 | **1.500** | **1.500** | 1.000 | 1.000 | **0.500** |
| T1F | 2023 | 1.167 | 1.167 | 1.000 | 1.000 | 1.000 |
| T1F | 2024 | 1.500 | 1.500 | 1.000 | 1.000 | 1.000 |
| T1F | 2025 | 1.500 | 1.500 | 1.000 | 1.000 | 0.917 |
| T2 | 2023 | 1.672 | 1.683 | 1.706 | 1.722 | 1.744 |
| T2 | 2024 | 0.833 | 0.722 | 0.500 | 0.333 | 0.111 |
| T2 | 2025 | 0.958 | 0.819 | 0.542 | 0.333 | 0.072 |

**Given enough prior games the per-band fit finds the big-spread direction on
its own.** In every one of the 36 weeks of 2024 and 2025, T1's grid chose
**1.5 at 0-3, 0.5 at 3.5-6.5, 1.0 at 7-10 and 0.5 at 10.5+** — it lowers the
gate on blowout lines and raises it on pick'ems without being told where to
look. It is **not monotone**: the 3.5-6.5 band also chooses 0.5, so "the gate
should fall with the line" is the wrong summary; "the gate should be highest on
short lines and lowest on blowouts" is the measured one. T2's grid settles on
`(a, b) = (0.75, -0.05)` in 16 of 18 weeks in 2024 and 14 of 18 in 2025 — the
same negative slope — but with an intercept so low that it fires 252 times and
loses a point overall. T1F's floor of 50 prior in-band games is never cleared at
7-10 or 10.5+ until the last three weeks of 2025, so **T1F is the served rule at
every big spread** and its whole live effect is at 0-3 and 3.5-6.5; that is why
T1F and T3 change disjoint picks.

### The positive control sets the resolution

| Control | Changed | Accuracy | Delta vs T0 | 95% week | P+ |
| --- | ---: | ---: | ---: | --- | ---: |
| PC_REACH, all reachable games | 213 | 84.230% | **+26.658** | [+23.515, +29.834] | 1.0000 |
| PC_T0, exactly the served fire set | 84 | 68.085% | **+10.513** | [+8.281, +12.795] | 1.0000 |
| PC_BIG, reachable at 10.5+ | 76 | 58.323% | +0.751 | [-1.370, +2.813] | 0.7558 |
| PC_BIG **inside the 10.5+ band** | 17 | 81.034% | **+29.310** | [+17.544, +42.105] | 1.0000 |

So the instrument resolves a twenty-seven-point effect on the pooled window and
a twenty-nine-point effect inside the 58-game 10.5+ band. It is **not** proven
able to see the 0.1-to-3.5-point effects every threshold arm sits at, so nothing
here is bounded by a control. Note what PC_T0 also says: perfect foresight on
the served rule's own 228 fires is worth only **+10.5** points, so the entire
threshold question is a fight over a fraction of that.

### Three deviations from the predeclaration, all found by measurement

**1. The predeclared terminal rule would have closed ten cells on a one-block
bootstrap.** The predeclaration required BOTH the week- and the season-blocked
interval below zero. Measured: a single-season window has **one** season block,
so its season-blocked bootstrap resamples the same block every time and the
interval collapses onto the point estimate — `[-1.50376, -1.50376]` for
`n0_vs_t0_2024`, and nine more like it. That is a degenerate artifact, not
evidence, and taking it would have "resolved" ten cells whose week-blocked
intervals plainly cross zero. **The rule is therefore read as requiring a
non-degenerate season bootstrap (at least two blocks)**, and the ten
single-season cells stay `unresolved_below_power`. Deviation in the keep-open
direction, named so it can be audited.

**2. The predeclared control ground would have closed the lane's best band cell,
and it is declined.** T3's 10.5+ cell is +3.448 [-5.085, +11.667]; PC_BIG in the
same band resolved at +29.310 [+17.544, +42.105]. The cell's interval excludes
the control's magnitude, which the predeclaration's literal text makes a
`positive_control_bound`. **It is not taken.** A control proven able to detect a
twenty-nine-point effect is no evidence at all that a three-point effect is
absent, and using it that way is the crossing-zero rejection wearing a different
hat. Same deviation `docs/line_size_shrinkage.md:384-389` took, for the same
reason.

**3. T2's tie rule needed a spelled-out tiebreak.** The predeclaration said ties
resolve to the null `(1.0, 0.0)`, which settles a tie that INCLUDES the null but
not one among non-null maximizers. Completed mechanically as: nearest to the
null in `(a, 10·b)`, then lowest `a`, then lowest `b`. The null is still checked
first and still wins every tie it is part of.

### Decision

**Three arms beat the served flat 1.0 through the played card, so the served
card is not the highest-expected-accuracy option.** Highest point estimate is
**T1F at +0.375 accuracy points, `probability_positive` 0.664**; strongest call
is **T3 at +0.250, `probability_positive` 0.791 week-blocked and 0.982
season-blocked**; T1 is +0.125 at 0.546. Declining all three is taking the short
side of a 66/34, a 79/21 and a 55/45 bet in a pool where 285 forced cards get
submitted either way.

**The arm to serve is T3**, and the reason is not its point estimate — T1F's is
larger — but that T3 is the only one of the three whose change is named by a
mechanism that existed before this lane, is positive or level in every season
and negative in none, and moves **6 picks in 799** rather than 35. **Its
mechanism in one sentence: on spreads of 10.5 or more the market's own move is
the strongest single signal on the board (62.96% when followed against the
model's 47.69%, `docs/big_spread_signal.md:431`), so half a point of leader
movement is already information there, while at 3 points or less that same half
point is noise that costs 3.470 accuracy points — a difference this lane
resolved, with the whole interval below zero.**

**One thing that must be said next to that recommendation.** T3 is a step, not a
fitted continuous function, and AGENTS.md's threshold rule asks for the
continuous form. The continuous arm this lane actually fitted, T2, **loses** at
-1.001 (`probability_positive` 0.066): its grid drove the intercept to 0.75, so
it fires 252 times and gives away at short lines everything it gains at long
ones. The right follow-up is therefore not "fit a smoother curve" in general but
a **predeclared intercept-constrained fit** — `t(L) = 1.0` for `L < 7` by
construction, with only the big-spread tail free — which is the region T2's
declared grid never explored and the region the mechanism actually names.

**The post-hoc reading, labelled as such and not recommended.** T1F and T3
change disjoint picks (T1F never acts above a 7-point line because its floor is
never cleared there; T3 acts nowhere else), so their combination is exactly
additive: **+0.626 accuracy points, `probability_positive` 0.754**, 41 picks
changed. That arm was composed AFTER the signs were seen, it is recorded under a
separate family so it can never be mistaken for a predeclared member, and a
follow-up lane should predeclare it rather than inherit it.

### What serving T3 would take (NOT wired here)

Measured against the current source, not remembered:

1. **The gate stops being a scalar.**
   `sharp_book_movement_features.LEADER_FOLLOW_THRESHOLD = 1.0`
   (`src/nfl_ats/sharp_book_movement_features.py:26`) becomes a function of the
   game's own opener line. `refresh_pick`
   (`src/nfl_ats/sharp_book_movement_features.py:130-134`) already computes
   `net_move.abs().ge(threshold)`, which broadcasts elementwise against a
   per-row Series, so the arithmetic needs no change — the signature does, and
   so does the caller at line 224.
2. **The follow frame needs the line.** `LATE_WEEK_GAMES_COLUMNS`
   (`src/nfl_ats/sharp_book_movement_features.py:137-142`) is
   `game_id, commence_time_utc, week_first_commence_utc, cutoff_utc` — no
   spread — and the contract check at line 180 enforces it, so the opener line
   would be added there and passed through by every caller of
   `late_week_follow_frame`. The served path already HAS the number:
   `decision_home_spread` is read off the ledger row inside `plan_refresh`
   (`src/nfl_ats/pick_refresh.py:1515`), a few lines above the served gate at
   `:1541`, so `plan_refresh` itself needs no new data source.
3. **Three more scalar reads of the same constant.** `LATE_WEEK_FOLLOW_THRESHOLD`
   is also used by `_fires` (`src/nfl_ats/pick_refresh.py:956`), by the pass
   summary's `games_followed` count (`:904`) and by the metadata block that
   publishes `"threshold"` (`:1743`). Each needs the game's own `|line|`, and
   the published `"threshold"` becomes the function's description rather than a
   number.
4. **The policy id, which names the rule.**
   `LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY = "late_week_leader_median_follow_1_0"`
   (`src/nfl_ats/pick_refresh.py:294`) and
   `LATE_WEEK_FOLLOW_NEWS_VETO_POLICY = "late_week_leader_median_follow_1_0_news_veto"`
   (`:295`) both become the two-value id, and both are compared by NAME in
   `MOVEMENT_GOVERNED_POLICIES` (`:298-300`) and at `:1706`, `:1731` and
   `:1759`, so the ledger keeps the eras distinguishable without a migration
   only if every one of those sites moves together.
5. **The paired OFF arm, first.** The flat 1.0 arm's counterfactual side must be
   recorded on every eligible game as its own OFF challenger — the exact pattern
   `late_week_follow_frame` already runs for the retired half-point gate
   (`leader_median_half_would_be_pick_side`, `:228-232`) and for
   `late_week_follow_no_news_veto_off_incumbent` — so the incumbent accrues game
   for game instead of being reconstructed later.
6. **The reader-facing sentence.** `src/nfl_ats/pick_refresh.py:2074` publishes
   "three leading books moved the line at least a full point since Tuesday and
   the pick followed them", which is wrong the moment a blowout line follows at
   half a point. It becomes the two-value statement, in pool-player words, with
   no threshold slug in it.
7. **A live-surface caveat no code change fixes.** All of this is measured on
   the historical backfill; the served rule's first live fire was the Thursday
   2026-09-10 refresh, so nothing is reversed retroactively either way.

### Registry

All **61** cells are recorded, one invocation at a time against
`registry/weak_signals.json` (now **5,077** signals): **53** under the declared
family `follow_threshold_by_line` and **8** under
`follow_threshold_by_line_posthoc`, named
`follow_threshold_by_line_<arm>_<window>`. The exact argv lists are
`record_commands.json` in the artifact directory and the outcomes are
`registry_records.json`.

**Two** are terminal, both `refuted_mechanism` on `wrong_sign_resolved`, both
with the week-blocked AND a non-degenerate three-block season-blocked interval
entirely below zero:

- `follow_threshold_by_line_n0_band_7_10_2023_2025` — not following the
  late-week market at all costs **-5.224** accuracy points on 7-to-10-point
  spreads, week [-10.606, -0.685], season [-9.091, -2.500], n=134. What that
  refutes is "the follow is not worth running at 7-10".
- `follow_threshold_by_line_t05_band_0_3_2023_2025` — the retired half-point
  gate costs **-3.470** accuracy points on spreads of 3 or less, week
  [-6.854, -0.312], season [-8.421, -0.952], n=317. What that refutes is
  "lower the gate on short lines", which is the specific claim the line-size
  story had to survive at the small-spread end, and it does — by pointing the
  other way there.

The other **59** are `unresolved_below_power` and report
`probability_positive`, including every favourable cell (T3 at +0.250 with
P+ 0.982 season-blocked, T1F at +0.375, both 10.5+ cells at +3.448, and the
controls at P+ 1.0000). A favourable interval is not a closing ground either.

**Six** of those 59 are exact dead heats — the arm made identical picks to the
served rule on every game in the cell, so the effect is exactly zero with no
dispersion (`t1f` at 7-10 and 10.5+, `t3` at 0-3, 3.5-6.5 and 7-10, and the
post-hoc arm at 7-10). They are recorded with `probability_positive` **0.5**,
per AGENTS.md's `P(>0) + 0.5*P(==0)` convention, and with **no standard error**,
because there is none to record; the registry correctly refuses a zero standard
error, and a dead heat closes nothing.

## Commands run

```
python stage1_frame.py       # card, follow frame, news veto; three reproduction checks
python stage2_arms.py        # walk-forward fits, 7 arms + 3 controls, 53 bootstrapped cells
python stage2b_posthoc.py    # the POST-HOC T1F+T3 combination, 8 cells
python stage3_record.py      # 61 weak-signals record commands, one at a time

nfl-ats weak-signals record ...   (61 cells; argv lists in record_commands.json,
run one at a time with NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry)
```

All four scripts are copied into
`artifacts/follow_threshold_by_line/20260910T032331Z/` beside their outputs.
