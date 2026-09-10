# The late-week follow's threshold, read on the card that is played

Closing-grounds taxonomy, verbatim, because this document reports intervals: an
interval or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (b) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

## The conflict this lane exists to resolve

Two lanes measured the same served rule and disagreed about it, because they used
different baselines and different replays.

**The promotion lane** (`docs/sharp_weighted_follow.md`, artifact
`artifacts/sharp_weighted_follow/20260909T233606Z/`) promoted the leader-median
follow at 0.5 (`LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY`,
`src/nfl_ats/pick_refresh.py`). It scored every arm against the **raw production
opener card** — `pick_home_at_open_probability_rule` on the frozen
`artifacts/experiments/sharp_book_movement/20260905T205038Z/per_game.parquet`
baseline — and stated up front that the archive harness could not build the
nine-member composed card, so it did not try. On that surface S1 (leader median,
0.5) reads +3.004 accuracy points over the raw card and +1.252 over the
equal-book arm S4, which itself reads +1.752 over the raw card.

**The precedence lane** (`docs/follow_vs_tilts.md`, artifact
`artifacts/follow_vs_tilts/20260910T002107Z/`) scored the same leader-median arm
against the **nine-member composed card** that is actually played, on the current
opener archive. There the served rule reads **-0.751 accuracy points against not
following at all** (`probability_positive` 0.332), and a post-hoc arm that only
changes the threshold from 0.5 to 1.0 reads **+2.003 over the served rule**
(`probability_positive` 0.918). Its band table names a mechanism: on collisions
with member flips the card side wins 30-22 inside the 0.5-to-1.0 band and loses
17-24 at or above 1.0; on reversals of raw-model picks the market wins 40-48
inside the band and 31-28 above it.

The two readings are not contradictory measurements of one quantity — they are
measurements of two different quantities. This lane measures ONE quantity, on the
surface that decides the played card, with the threshold declared as a grid
before any sign is seen.

## Population, surfaces and the two reproduction checks

- **Population.** The 2023-2025 `intraday_hourly` historical-backfill archive
  under `data/market/raw`, loaded with `nfl_ats.clv.load_decision_quotes`
  (`capture_kind=historical`, label `intraday_hourly`), joined to the opener
  archive `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet` (active
  model `c657058903f3232b`, `gaussian_median`). Expected: 816 archive games in
  window, 799 opener-scored (17 opener pushes are dropped — they are not
  scoreable either way), 54 week blocks, 3 season blocks.
- **Stated deviation, up front.** The served path loads `capture_kind="live"` and
  the market store holds no live capture before 2026-08-17, so 2023-2025 is
  reachable only through the historical backfill. This is the same substitution
  `docs/served_card_harness.md` and `docs/follow_vs_tilts.md` document, and it is
  labelled one here too: this is evidence about the rule's SHAPE, not a re-grade
  of the promotion that was taken on the raw surface.
- **CARD surface (primary).** The served nine-member joint OR
  `overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`,
  rebuilt with `nfl_ats.unserved_tilt_marginals.served_card_flip_set(card="served")`.
  **Reproduction check 1: it must score 56.070% on the 799 scored games**, the
  figure `docs/served_card_harness.md` publishes. If it does not, this lane
  reports the mismatch and stops rather than reading a different card.
- **RAW surface (reconciliation).** The raw model's own opener pick,
  `home_cover_probability_at_open >= 0.5`, no card at all. This is the surface the
  promotion lane used, so running the identical arm grid on it puts both earlier
  lanes in one table. Note it is the same CONSTRUCT as the promotion lane's
  baseline but not the same NUMBERS: the promotion lane's frozen per-game archive
  is the 2026-09-05 baseline card, this one is the current active model, so its
  raw accuracy is expected near 55.069% rather than 52.816%.
- **Follow replay.** `nfl_ats.sharp_book_movement_features.late_week_follow_frame`
  — the frozen function the served refresh itself calls — with each game's cutoff
  at its own pick deadline `min(kickoff, that week's Sunday 16:00 ET)`, which is
  the maximum late-week information the rule could ever see. The served arm read
  is `leader_median_net_move` (median Wednesday-to-deadline net move across
  Bovada / William Hill (US) / MyBookie), never the equal-book mean, except in arm
  E10 which is declared as the equal-book arm on purpose.
- **Reproduction check 2: the replayed leader-median arm must reproduce the
  promotion lane's own S1 counts — 523 fires and 223 switches against the raw
  card** — when the same follow is applied to the promotion lane's frozen baseline
  card on the promotion lane's frozen 816-game quote cache. This is run as a
  standalone parity job so that a difference between the two quote sources
  (`data/market/raw` reload vs the frozen `quotes.parquet`) is reported as a
  number, not absorbed silently.

## The arms (a declared grid; nothing is searched after the signs are seen)

Every arm is the SAME rule with one number changed, applied to both surfaces. Let
`net` be the leader-median Wednesday-to-deadline move on a game and `T` the arm's
threshold. If `|net| >= T` and at least one leading book contributed an
increment, the pick becomes the side the market moved toward (`net > 0` picks
HOME); otherwise the surface's own pick stands.

| Arm | Rule |
| --- | --- |
| **L0** | No follow at all. The surface's own pick on every game. |
| **L05** | Leader-median follow at `T = 0.5`. **This is what is served today.** |
| **L075** | Leader-median follow at `T = 0.75`. |
| **L10** | Leader-median follow at `T = 1.0`. |
| **L15** | Leader-median follow at `T = 1.5`. |
| **E10** | The EQUAL-BOOK mean of all twelve books at `T = 1.0` — the old consensus rule's aggregate at the new threshold, so "is it the threshold or the aggregation?" is answered in the same table. |
| **PC** | Positive control, below. |

The grid 0.5 / 0.75 / 1.0 / 1.5 is fixed here, before any of it is computed. 0.5
is the served constant, 1.0 is the constant the independent
`MOVEMENT_POLICY_THRESHOLD` consensus rule already uses, and 0.75 and 1.5 bracket
it so the grid cannot be read as a two-point cherry-pick. No finer grid, no
per-season threshold, and no threshold chosen after the fact will be promoted out
of this lane.

**Positive control (PC).** Perfect-foresight switching on exactly the population
any of these thresholds could reach: on every game where the served 0.5 rule
fires, the oracle takes the side that actually covered; everywhere else it takes
the surface's own pick. Every higher threshold's fire set is a strict subset of
the 0.5 fire set, so this is the ceiling on the entire threshold question. A
second control, **PC_REACH**, does the same on every game with at least one
contributing leading book (whether or not the move cleared half a point), which
bounds what any threshold *lower* than 0.5 could add. If an arm's effect sits
well inside the control's own interval, the instrument is not shown to be blind
and nothing here is `bounded_by_control`.

## Band decomposition (the mechanism, made visible)

For each band of `|net|` — **0.5-0.75, 0.75-1.0, 1.0-1.5, 1.5+** — and for each
season and pooled:

- **Reversals of raw-model picks**: games where the follow fires, the card left
  the pick alone, and the market side differs from the card side. Reported as
  model side won / market side won, and the market's win rate.
- **Reversals of member flips**: games where the follow fires, one of the nine
  members flipped the pick, and the market side differs from the card side.
  Reported as card side won / market side won, and the market's win rate.
- Endorsement counts (follow fires and agrees) are reported alongside so the
  denominator of every band is legible.

This table is the DIAGNOSIS. Per AGENTS.md's "no unexplained threshold flips", a
threshold that changes the played card must name its mechanism; the mechanism
this lane will either confirm or fail to confirm is stated in advance: **a
sub-point drift in the median of three books is noise, and the market side only
carries information once the leaders have moved a full point.** If the band table
does not show the low band losing, the threshold change is unexplained and this
document says so and recommends nothing.

## Statistics, declared before the numbers

- Paired deltas in accuracy points, game-weighted, forced picks, graded at the
  frozen Tuesday OPENER (`margin_vs_open`); opener pushes dropped.
- `nfl_ats.overlay_composition.blocked_bootstrap_matrix`, paired, **20,000
  samples, seed 20260821**, **week-blocked** (blocks are `(season, week)`) and
  **season-blocked** (blocks are seasons). Within-week game correlation is ZERO
  by owner mandate; no ICC is estimated or padded. The season-blocked column has
  three blocks, so its `probability_positive` saturates and carries almost no
  information — it is reported for completeness, never as a second opinion.
- Two baseline columns for every arm: **vs L0** (does following at this threshold
  beat not following at all?) and **vs L05** (does this threshold beat the one
  being served?).
- Windows: 2023-2025 pooled, then 2023, 2024 and 2025 separately.
- Switch counts per arm on every surface: fires, picks changed against L0, picks
  changed against L05.
- `probability_positive` is reported for every cell. The binary "contains zero" is
  never used as a verdict.

## Decision rule, declared before the numbers

The pool is forced picks: 285 cards get submitted either way, so **the threshold
with the higher expected accuracy at the deadline on the played card is the one
to serve.** The decision line is the arm with the highest paired point estimate
against L05 on the CARD surface over 2023-2025, with its `probability_positive`
reported as the strength of the call. A predeclared bar governs what this
document may CLAIM; it never governs which card is played.

If L05 is not the best arm, this document states exactly what changing it would
take — the single constant, the policy id, the docs, and the paired challenger —
and **wires nothing**. No code is edited in this lane.

## Registry

Every cell is recorded through `nfl-ats weak-signals record` under names
`follow_threshold_live_card_<arm>_<window>`, units `accuracy_points`, league
`nfl`, family `follow_threshold_live_card`, one invocation at a time with
`NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry`, argv lists saved to
`record_commands.json`. The family is declared here, before the signs were seen;
its members share one population, one grading and one follow computation, so they
are correlated readings of one question rather than independent votes.

## What was measured

Runner `artifacts/follow_threshold_live_card/20260910T003442Z/lane_af.py`, artifact
`results.json` in the same directory. The `predeclaration_sha256` recorded there,
`fa4217150b6b7f3d38b50607f19ad0699bf9d077d60be40fed99d959d5e7bd49`, is the hash of
everything ABOVE this line, taken before the script ran; every section below was
written afterwards. Schedule snapshot `20260908T162105Z`.

Population: **816** archive games in 2023-2025, all 816 with replayable late-week
exposure and all 816 with at least one leading book contributing an increment,
**799 scored** (17 opener pushes dropped), **0 quote rows refused** by the frozen
function's own backdating guard. 54 week blocks, 3 season blocks.

### Both reproduction checks pass

**Check 1 — the card.** The rebuilt nine-member card scores **56.0700876%** on the
799 scored games, reproducing `docs/served_card_harness.md`'s 56.070% to the digit.
The served rule replayed on it scores **55.319%**, reproducing
`docs/follow_vs_tilts.md`'s P0 to the digit as well, so this lane is reading the
same two objects those lanes read.

**Check 2 — the follow arm.** `artifacts/follow_threshold_live_card/20260910T003442Z/parity.json`:
run on the promotion lane's own frozen quote cache
(`artifacts/experiments/sharp_book_movement/quotes.parquet`, 816 games), the
leader-median arm fires **523** times and switches **223** picks against the raw
card — exactly `scripts/sharp_weighted_follow.py`'s S1 counts. Reloading the same
rows from `data/market/raw` gives **522** fires and the identical **223** switches:
two games differ in leader-median net move between the two quote sources and
exactly one of them, `2025_04_SEA_ARI`, straddles the half-point gate. That single
game is the whole of the 522-vs-523 gap `docs/follow_vs_tilts.md` reported without
explaining, and it changes no switch count.

**A third check nobody asked for.** This lane's L10-vs-L05 cell reproduces
`docs/follow_vs_tilts.md`'s post-hoc P4-vs-P0 confound cell digit for digit
(+2.003, [-0.769, +4.851], `probability_positive` 0.9178) from an independently
written runner. The difference is that here it is a PREDECLARED member of a fixed
0.5 / 0.75 / 1.0 / 1.5 grid, not an arm added after a band table was read.

### A mechanical fact that collapses the grid

**The leader median never lands inside [0.75, 1.0) on this population — the band
holds zero games.** Three books quoting on the half-point grid produce a median on
the half-point grid whenever an odd number of them contributed, and the two-book
case did not produce a 0.75 anywhere in 816 games. So **L075 and L10 are the same
arm, game for game** (228 fires each), and the grid's real resolution here is
0.5 / 1.0 / 1.5. Both are reported so the empty band is on the record rather than
silently absorbed.

### Accuracy, forced picks at the opener, 799 games

| Arm | Fires | CARD (played) | 2023 | 2024 | 2025 | RAW (no card) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **L0** no follow | 0 | **56.070%** | 59.398% | 54.135% | 54.682% | 55.069% |
| **L05** served, 0.5 | 522 | **55.319%** | 58.271% | 55.263% | 52.434% | 56.320% |
| L075 (= L10) | 228 | 57.322% | 58.647% | 56.015% | 57.303% | 56.821% |
| **L10** 1.0 | 228 | **57.322%** | 58.647% | 56.015% | 57.303% | 56.821% |
| L15 1.5 | 116 | 56.446% | 59.023% | 54.887% | 55.431% | 55.569% |
| E10 equal-book 1.0 | 156 | 55.820% | 57.143% | 56.015% | 54.307% | 54.819% |
| PC control | 522 | 84.230% | 91.729% | 80.075% | 80.899% | 85.232% |
| PC_REACH control | 816 | 100.000% | | | | 100.000% |

### Paired results on the CARD that is played, 2023-2025

Week-blocked interval quoted; 20,000 samples, seed 20260821. The season-blocked
column has three blocks, so its `probability_positive` saturates at 0.000 or 1.000
and carries almost no information — reported for completeness, never as a second
opinion.

| Cell | vs | Changed | Delta (pts) | 95% week | P+ week | 95% season | P+ season |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| **L05 served** | L0 | 240 | **-0.751** | [-4.172, +2.753] | **0.3315** | [-2.247, +1.128] | 0.209 |
| **L10** | L0 | 100 | **+1.252** | [-1.122, +3.713] | **0.8462** | [-0.752, +2.622] | 0.965 |
| L15 | L0 | 49 | +0.375 | [-1.141, +1.877] | 0.6866 | [-1.128, +1.504] | 0.854 |
| E10 | L0 | 68 | -0.250 | [-2.020, +1.611] | 0.3892 | [-2.256, +1.880] | 0.377 |
| PC control | L0 | 225 | +28.160 | [+25.094, +31.297] | 1.0000 | [+25.940, +32.331] | 1.000 |
| **L10** | **L05** | 140 | **+2.003** | [-0.769, +4.851] | **0.9178** | [+0.376, +4.869] | 1.000 |
| L15 | **L05** | 191 | +1.126 | [-2.010, +4.245] | 0.7610 | [-0.376, +2.996] | 0.905 |
| E10 | **L05** | 172 | +0.501 | [-2.284, +3.283] | 0.6383 | [-1.128, +1.873] | 0.744 |
| PC control | **L05** | 231 | +28.911 | [+25.926, +31.915] | 1.0000 | [+24.812, +33.459] | 1.000 |

By season, L10 against L05 is **+0.376 (2023), +0.752 (2024), +4.869 (2025)** —
the same sign in all three, magnitudes differing, which is a magnitude statement
per era and never an absence. L05 against L0 is -1.128 / +1.128 / -2.247, and L10
against L0 is -0.752 / +1.880 / +2.622.

### The same arms on the RAW card, so the two earlier lanes reconcile

| Cell | vs | Changed | Delta (pts) | 95% week | P+ week | P+ season |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| L05 served | L0 | 224 | +1.252 | [-2.847, +5.416] | 0.7243 | 0.722 |
| **L10** | L0 | 92 | **+1.752** | [-0.511, +4.130] | **0.9305** | 0.965 |
| L15 | L0 | 44 | +0.501 | [-0.995, +1.988] | 0.7483 | 0.744 |
| E10 | L0 | 60 | -0.250 | [-2.112, +1.621] | 0.4007 | 0.301 |
| L10 | L05 | 132 | +0.501 | [-2.652, +3.639] | 0.6182 | 0.704 |
| L15 | L05 | 180 | -0.751 | [-4.489, +2.868] | 0.3444 | 0.372 |
| E10 | L05 | 164 | -1.502 | [-4.798, +1.750] | 0.1815 | 0.260 |
| PC control | L0 | 241 | +30.163 | [+26.692, +33.794] | 1.0000 | 1.000 |

**This is the whole reconciliation, in one comparison.** The promotion lane was
right about its own surface and the precedence lane was right about its own: on
the RAW card the served 0.5 follow does add (+1.252, P+ 0.724 here; +3.004,
P+ 0.930 on the frozen 2026-09-05 baseline card the promotion lane used), while on
the nine-member card that is actually played it subtracts (-0.751, P+ 0.332). The
two lanes never disagreed about a number; they measured a rule against two
different cards, and the played card already moves 260 of these 816 games, so the
half-point follow spends most of its firings overwriting picks the card had a
reason for. **The threshold at 1.0 is the only arm that is positive on BOTH
surfaces** (+1.252 on the card, +1.752 on the raw model), which is what a real
signal looks like and what a surface artifact does not.

### The band decomposition — the mechanism, made visible

Reversals only: the follow fires and points at the other side. "Card side won"
counts the pick the surface would have made; "market side won" counts the follow's.
Scored games only, so the bands sum to 514 of the 522 firings.

| Band | Games | Member-flip reversals (card-market) | Market % | Raw-pick reversals (model-market) | Market % | Raw-surface reversals | Market % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.5-0.75** | **288** | **30-22** | **42.3%** | **48-40** | **45.5%** | 68-64 | 48.5% |
| 0.75-1.0 | **0** | — | — | — | — | — | — |
| 1.0-1.5 | 111 | 9-11 | 55.0% | 13-18 | 58.1% | 19-29 | 60.4% |
| 1.5+ | 115 | 8-13 | 61.9% | 15-13 | 46.4% | 20-24 | 54.5% |
| **at least 1.0** | **226** | **17-24** | **58.5%** | **28-31** | **52.5%** | 39-53 | 57.6% |

By season, the low band's market win rate on member-flip reversals is 47.4% (2023),
27.3% (2024) and 45.5% (2025) — below a coin flip in all three, magnitudes
differing. At or above 1.0 it is 61.5% / 47.1% / 72.7%, above a coin flip in two of
three. On raw-model reversals the low band reads 50.0% / 57.1% / 33.3% and the high
band 40.0% / 65.0% / 57.1%: the per-season signs there do NOT line up, and that is
said plainly rather than smoothed — the mechanism is clean on the pooled read and on
member flips season by season, and noisy on raw-pick reversals season by season.

**The named mechanism, one sentence: a half-point drift in the median of three
books is not information — the market side loses 42-46% of the picks it reverses
inside the 0.5-to-0.75 band, wins 52-59% at a full point or more, and the served
rule spends 288 of its 514 scored firings in the losing band.**

### The positive control sets the resolution

Perfect-foresight switching on exactly the 522 games the served rule fires on is
worth **+28.911 accuracy points over L05, 95% [+25.926, +31.915]** (225-231 picks
changed of 799), and the second control — perfect foresight on every game with
leading-book evidence, which on this archive is all 816 — reaches 100% by
construction. So the instrument is PROVEN able to see an effect of roughly
twenty-nine points and is **not** proven able to see one of one to three points,
which is where every threshold arm sits. Nothing measured here is bounded by this
control, and nothing closes. Note also what the control says about the ceiling: the
served rule fires on 522 of the 816 games (it changes 240 of the 799 scored picks
and agrees with the card on the rest), and being right on every one of those
firings would be worth about 29 points — so the entire threshold question is a
fight over a few percent of a very large reachable pool.

## Decision

**On the card that is actually played, the threshold with the higher expected
accuracy at the deadline is 1.0, not the served 0.5.** L10 scores **57.322%**
against L05's **55.319%** — **+2.003 accuracy points paired, week-blocked 95%
[-0.769, +4.851], `probability_positive` 0.918** (season-blocked [+0.376, +4.869]
on three blocks), on 140 changed picks — and **+1.252 points against not following
at all**, `probability_positive` 0.846. The pool is forced picks: serving 0.5 over
1.0 on this evidence is taking the short side of a 92/8 bet.

**The mechanism is named and it is not a threshold flip bolted onto a dip.** The
0.5-to-0.75 band is where the served rule spends most of its firings (288 of 514
scored), and it is a band the market side loses in on both surfaces — 22 of 52
member-flip reversals and 40 of 88 raw-pick reversals. Raising the gate to a full
point deletes that band and keeps every firing where the market side wins. This is
a magnitude statement about a market's own information content at different move
sizes, not an accuracy hole located at a spread number, so it is the kind of
threshold change AGENTS.md's "no unexplained threshold flips" rule asks for rather
than the kind it bans.

**Three things that must be said plainly and none of them closes anything.** First,
the 1.5 arm is worse than the 1.0 arm on both surfaces (+1.126 vs +2.003 against
L05 on the card), so this is not "higher is always better" — it is one band that
loses. Second, the equal-book aggregate at the same 1.0 threshold (E10, +0.501
against L05, P+ 0.638) captures far less of the gain than the leader median does,
so the threshold and the leader-median aggregation are both carrying weight and
neither substitutes for the other. Third, this is the historical-backfill replay,
not the live path the rule actually runs on, so it is evidence about the rule's
shape and not a re-grade of the promotion taken on the raw surface.

### Served since 2026-09-10 (wired in a later lane, not this one)

L10 is now the served gate. The change is a SEPARATE constant --
`sharp_book_movement_features.LEADER_FOLLOW_THRESHOLD = 1.0` -- so the paired
equal-book challenger stays on `THRESHOLD = 0.5`, the constant it was measured
at, exactly as item 1 below requires; the policy id became
`late_week_leader_median_follow_1_0`; and the retired half-point leader-median
arm records as its own paired OFF challenger
(`late_week_leader_median_follow_0_5_off_incumbent`,
`leader_median_half_would_be_pick_side` on
`late_week_move_follow_refresh_decisions.parquet`), so both sides keep accruing
game for game. See `docs/late_week_refresh.md`'s promotion section. The list
below is the original, unedited statement of what it would take.

### What serving L10 would take (NOT wired here)

1. **The single constant.** `THRESHOLD = 0.5` at
   `src/nfl_ats/sharp_book_movement_features.py:25`, imported into the refresh path
   as `LATE_WEEK_FOLLOW_THRESHOLD` (`src/nfl_ats/pick_refresh.py:188`) and read at
   `src/nfl_ats/pick_refresh.py:1104` (the served gate) and
   `src/nfl_ats/pick_refresh.py:807` (the pass summary's `games_followed`).
   **Changing it in place would silently move the paired equal-book challenger too**,
   because the same constant gates `refresh_pick` and the `leader_flag` / `equal_flag`
   / `leader_median_flag` columns for both arms. The honest change is a separate
   served constant for the late-week gate, leaving the equal-book challenger on the
   threshold it was measured at, so the two arms stay comparable game for game.
2. **The policy id, which names the threshold.**
   `LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY = "late_week_leader_median_follow_0_5"`
   (`src/nfl_ats/pick_refresh.py:238`) becomes `..._1_0`. It is written into every
   pick-revision row's `movement_policy` column, so the ledger keeps both eras
   distinguishable without a migration — but the value is compared by name in
   `MOVEMENT_GOVERNED_POLICIES` (line 240) and at line 1266, and both would need the
   new id.
3. **The docs, the module docstrings and the reader-facing card text.** Measured
   this session: `docs/late_week_refresh.md` names the threshold in eight places
   (`0.5`) plus two more as "half a point"; `src/nfl_ats/pick_refresh.py` states it
   in its module docstring (line 110) and its challenger registration (line 739);
   and the `--publish-card` section text a reader actually sees is the literal at
   `src/nfl_ats/pick_refresh.py:1532` — "three leading books moved the line at least
   half a point since Tuesday and the pick followed them". All of those would be
   wrong the moment the constant moves.
4. **The paired challenger, first.** The follow already writes
   `late_week_move_follow_refresh_v1` with the equal-book arm as the OFF side and
   `served_challenger_id` `late_week_leader_median_follow_v1`. A threshold change
   should ride the same pattern: record the 0.5 arm's counterfactual side beside the
   served 1.0 side on every eligible game, so the incumbent accrues game for game
   instead of being reconstructed later. `late_week_net_move` is already on every
   pick-revision row, so the 0.5 counterfactual is recoverable from the ledger even
   before that wiring exists.
5. **A live-surface caveat that no code change fixes.** Everything above is measured
   on the historical backfill. The rule has never fired on a played card (its first
   live fire is the Thursday 2026-09-10 refresh), so nothing is being reversed
   retroactively and no continuity breaks either way.

## Registry

104 cells, family `follow_threshold_live_card`, named
`follow_threshold_live_card_<surface>_<arm>_vs_<baseline>_<window>` — both surfaces,
seven arms, two baseline columns, four windows. All 104 are
`unresolved_below_power`: no arm's week-blocked interval sits wholly on one side of
zero, no trait here has zero split-half reliability, and the positive control
resolves at about twenty-nine points while every arm sits at one to three, so the
control bounds nothing. The exact argv lists are saved as
`artifacts/follow_threshold_live_card/20260910T003442Z/record_commands.json` and were
run one at a time against `registry/weak_signals.json`.

## Commands run

```
python artifacts/follow_threshold_live_card/20260910T003442Z/parity.py
python artifacts/follow_threshold_live_card/20260910T003442Z/lane_af.py
python artifacts/follow_threshold_live_card/20260910T003442Z/record_af.py
nfl-ats weak-signals record ...   (104 cells; argv lists in record_commands.json)
```
