# Precedence between the late-week follow and the nine-member card

Closing-grounds taxonomy, verbatim, because this document reports intervals: an
interval or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (b) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

## The question

The Tuesday card is a nine-member joint OR
(`overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`,
`src/nfl_ats/four_overlay_composition.py:86` `COMPOSITION_ORDER`): coach fade,
division revenge, player arrests, bye-edge fade, cold visitor, protection
mismatch, interim head coach, tank zone, precipitation-on-a-high-total.

The served late-week rule is the leader-median follow
(`src/nfl_ats/pick_refresh.py:238` `LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY`): when
the median Wednesday-to-deadline net move across Bovada / William Hill / MyBookie
is at least 0.5 points, the pick becomes the side the market moved toward. Read
at `src/nfl_ats/pick_refresh.py:1106-1110`, that branch is taken FIRST and
unconditionally — `new_side = late_week_side` — with no reference to whether an
overlay member flipped the Tuesday pick. The composed union is already baked into
`overlaid` at `src/nfl_ats/pick_refresh.py:1016-1019`, so a follow that fires on a
member-flipped game silently undoes the mechanism flip.

`docs/served_card_harness.md` measured (equal-book rule, 2023-2025) that of the
118 games only the six new members flip, the follow reverses 20 and endorses 14.
That is the collision this lane is about: a mechanism said "the market is wrong
about this game", and then the rule follows the market anyway — possibly the very
market move the mechanism anticipates.

## Predeclared arms

All four are precedence orders over the SAME two ingredients: the frozen Tuesday
nine-member card side, and the leader-median follow side. None of them changes
either ingredient. `CARD` (the Tuesday nine-member card, no follow at all) is the
reference column; every arm is also scored paired against `P0`.

- **P0 — served precedence.** The follow overrides everything. If
  `|leader_median_net_move| >= 0.5` the pick is the market side; otherwise the
  Tuesday card side. This is what `plan_refresh` does today.
- **P1 — mechanism flips are protected.** The follow may not reverse a pick that
  any of the nine members flipped. On a game inside the composed union flip set,
  the card side stands no matter what the market did. Elsewhere (raw-model picks
  the card left alone) P1 is identical to P0.
- **P2 — only the three original members are protected.** As P1, but the
  protected set is the union of `coach_fade`, `division_revenge_tilt` and
  `player_arrests_back_side_policy` only. A game flipped solely by one of the six
  members added 2026-09-09 is still reversible by the follow. This arm is
  separated out because those three are the only members with their own column in
  the paper-decision ledger (`src/nfl_ats/pick_refresh.py:1357-1361`), so P2 is
  the only protective arm that is servable with the columns that exist today.
- **P3 — protected at 0.5, overridable at 1.0.** As P1 (all nine protected), plus
  an escape hatch: a leader-median move of at least 1.0 point against a member
  flip does reverse it. Below 1.0 the mechanism wins; at or above 1.0 the market
  does.

"Reverse" means the follow's side differs from the card side. When the follow
agrees with the card the follow is a no-op on the pick, so protection is
equivalent to suppressing the follow entirely on protected games.

## Positive control

**Perfect-foresight resolution of every follow-vs-tilt collision.** On every game
where a member flipped the pick AND the follow would reverse it, the oracle takes
the side that actually covered; everywhere else it takes P0's pick. This is the
ceiling of the whole precedence question — the most any re-ordering of these two
ingredients could ever be worth on this population. If an arm's measured effect is
well inside the oracle's own interval, the instrument is not shown to be blind;
if the oracle itself is indistinguishable from zero, the population is too small
to resolve ANY precedence and every arm is `unresolved_below_power` on that basis.

## Measurement plan (fixed before any sign is seen)

- **Card construction.** `nfl_ats.unserved_tilt_marginals.served_card_flip_set`
  with `card="served"` (nine members) and `card="three"` (the original three), on
  the opener archive `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`
  (active model `c657058903f3232b`, `gaussian_median`). Per-member flip sets come
  back from the same call.
- **Follow replay.** `nfl_ats.sharp_book_movement_features.late_week_follow_frame`
  — the frozen function the served path itself calls — fed the archive's
  `intraday_hourly` snapshots under `data/market/raw`, seasons 2023-2025, with
  each game's cutoff at its own pick deadline `min(kickoff, Sunday 16:00 ET)`.
  **The served arm read is `movement_would_be_pick_side` /
  `leader_median_net_move`** (the leader-median rule), not the equal-book
  `equal_would_be_pick_side` that `docs/served_card_harness.md` reported.
  Stated deviation, up front: the served path loads `capture_kind="live"`, and
  the archive holds no live capture before 2026-08-17, so 2023-2025 is reachable
  only through the historical backfill. This is the same substitution
  `docs/served_card_harness.md` documents and is labelled one here too.
- **Grade.** At the frozen Tuesday opener, `margin_vs_open`, forced picks; opener
  pushes are dropped (they are not scoreable either way).
- **Uncertainty.** `nfl_ats.overlay_composition.blocked_bootstrap_matrix`, paired,
  20,000 samples, seed 20260821, week-blocked and season-blocked. Within-week
  game correlation is ZERO by owner mandate; no ICC is estimated or padded.
- **Windows.** 2023-2025 overall, then 2023, 2024 and 2025 separately.
- **Columns per arm.** vs `CARD` (does adding this precedence to the Tuesday card
  help?) and vs `P0` (is this precedence better than the one being served?).
- **Counts reported.** Games where the follow fired on a raw-model pick; games
  where it fired on a member-flipped pick; within each, reversed and endorsed.
  Per member: how many of its flips the follow reverses, and how many of those
  reversals the market won versus the mechanism won.
- **Registry.** Every cell recorded as `follow_vs_tilts_<arm>_<window>`, family
  `follow_vs_tilts_precedence`, one `nfl-ats weak-signals record` invocation at a
  time, argv saved to `record_commands.json`.

## Decision rule (fixed before any sign is seen)

The pool is forced picks: a card is submitted either way, so the precedence with
the higher expected accuracy at the deadline is the one to serve. The arm with the
highest paired point estimate against P0 wins the decision line, and its
`probability_positive` is reported as the strength of that call. A predeclared
threshold governs what this document may CLAIM; it never governs which precedence
is served. If P0 does not win, this document states exactly what serving the
winner would take — the guard in `plan_refresh`, and which ledger column carries
the "protected by" reason — and wires nothing.

## What was measured

Runner `artifacts/follow_vs_tilts/20260910T002107Z/lane_ad.py`, artifact
`results.json` in the same directory. The `predeclaration_sha256` recorded there,
`08134fe4f64ea7bf2e49257a8141a82e2aa2ac9ffbf6a82de0a1e27da3b7c921`, is the hash of
everything ABOVE this line, taken before the script ran; the sections below were
written afterwards. Schedule snapshot `20260908T162105Z`.

Population: **816** archive games in 2023-2025, all 816 with replayable late-week
exposure, **799 scored** (17 opener pushes), **0 quote rows refused** by the
frozen function's own backdating guard. 54 week blocks, 3 season blocks.

Accuracy on those 799 games, forced picks at the opener:

| Arm | Accuracy | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: |
| raw model, no card, no follow | 55.069% | | | |
| CARD — nine-member Tuesday card, no follow | 56.070% | 59.398% | 54.135% | 54.682% |
| **P0 — served (follow overrides everything)** | **55.319%** | 58.271% | 55.263% | 52.434% |
| P1 — all nine members protected | 55.444% | 57.519% | 57.519% | 51.311% |
| P2 — only the three original members protected | 54.693% | 57.519% | 55.263% | 51.311% |
| **P3 — protected below 1.0, market wins at 1.0** | **56.320%** | 58.647% | 57.143% | 53.184% |
| positive control — perfect-foresight collisions | 61.202% | 63.910% | 61.654% | 58.052% |

The CARD row reproduces `docs/served_card_harness.md`'s 56.070% to the digit on
the same population, which is the check that the card was rebuilt correctly
(`artifacts/follow_vs_tilts/20260910T002107Z/verify.json`: the card equals the
archive's own pick on every non-flipped game and its exact complement on every
flipped one).

### Collision counts

| | Games |
| --- | ---: |
| Qualifying leader-median move (at least 0.5) | 522 |
| ...of at least 1.0 | 228 |
| Nine-member card flips, 2023-2025 | 260 |
| ...of which the original three account for | 142 |
| Follow fired on a **raw-model** pick (card left it alone) | 350 |
| — reversed it | 149 |
| — endorsed it | 201 |
| Follow fired on a **member-flipped** pick | 172 |
| — **reversed the mechanism flip** | **94** |
| — endorsed it | 78 |
| Of those 94 reversals: hit a three-member flip | 51 |
| Of those 94 reversals: hit a six-new-member-only flip | 43 |
| Of those 94 reversals: move was at least 1.0 | 41 |

### Paired results, 2023-2025

Week-blocked interval quoted; 20,000 samples, seed 20260821. The season-blocked
column has **three blocks**, so its `probability_positive` saturates at 0.000 or
1.000 and carries almost no information — it is reported for completeness, not as
a second opinion.

| Cell | vs | Changed | Delta (pts) | 95% week | P+ week | P+ season |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| P0 served | CARD | 240 | **-0.751** | [-4.172, +2.753] | 0.3315 | 0.209 |
| P1 all nine protected | CARD | 147 | -0.626 | [-3.865, +2.645] | 0.3548 | 0.266 |
| P2 three protected | CARD | 189 | -1.377 | [-4.478, +1.750] | 0.1908 | 0.151 |
| **P3 protected below 1.0** | CARD | 188 | **+0.250** | [-3.206, +3.797] | 0.5572 | 0.648 |
| positive control | CARD | 193 | +5.131 | [+1.619, +8.647] | 0.9978 | 1.000 |
| P1 all nine protected | **P0** | 93 | +0.125 | [-2.261, +2.519] | 0.5428 | 0.648 |
| P2 three protected | **P0** | 51 | -0.626 | [-2.256, +1.001] | 0.2324 | 0.019 |
| **P3 protected below 1.0** | **P0** | 52 | **+1.001** | [-0.639, +2.662] | **0.8801** | 1.000 |
| positive control | **P0** | 47 | +5.882 | [+4.035, +7.940] | 1.0000 | 1.000 |

By season, against P0: P3 is +0.376 (2023), +1.880 (2024), +0.749 (2025) — the
same sign in all three, magnitudes differing, which is a magnitude statement per
era and never an absence. P1 is -0.752 / +2.256 / -1.124 and P2 is
-0.752 / +0.000 / -1.124.

### The positive control sets the resolution

Perfect-foresight resolution of the 94 collisions is worth **+5.882 accuracy
points over P0, 95% [+4.035, +7.940]** — 47 picks changed out of 799. That is the
ceiling on this whole question, and the instrument's half-width on it is about
±2 points. So the instrument is PROVEN able to see an effect of about six points
and is **not** proven able to see one of one or two points, which is where every
arm sits. Nothing measured here is bounded by this control, and nothing closes.

### Per-member: whose flips does the market beat?

Scored reversals only — the games where that member flipped the pick and the
follow then flipped it back.

| Member | Flips | Follow fired | Reversed | Card side won | Market side won | Endorsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| division revenge | 80 | 62 | 32 | 12 | **20** | 30 |
| bye-edge fade | 41 | 25 | 18 | **13** | 5 | 7 |
| coach fade | 55 | 34 | 19 | 11 | 8 | 15 |
| protection mismatch | 67 | 40 | 19 | 11 | 8 | 21 |
| cold visitor | 19 | 13 | 8 | 4 | 4 | 5 |
| tank zone | 11 | 9 | 7 | 3 | 3 | 2 |
| precipitation | 4 | 4 | 1 | 0 | 1 | 3 |
| player arrests | 8 | 5 | 0 | — | — | 5 |
| interim head coach | 3 | 0 | 0 | — | — | 0 |

Members overlap on some games, so these rows do not sum to the 94. The market
beats **division revenge** (12-20) and loses to **bye-edge fade** (13-5); coach
fade and protection mismatch both lean to the mechanism (11-8 each). Across all
93 scored collisions the card side wins **47-46** — a dead heat, which is exactly
why P1 (protect everything) is +0.125 and says nothing.

### The band is what matters, not the precedence

Splitting the same collisions by how big the move was:

| Move against a member flip | Games | Card side won | Market side won | Card % |
| --- | ---: | ---: | ---: | ---: |
| 0.5 to 1.0 | 52 | 30 | 22 | 57.7% |
| at least 1.0 | 41 | 17 | 24 | 41.5% |
| all | 93 | 47 | 46 | 50.5% |

And the same split on the follow's reversals of **raw-model** picks, where no
mechanism is involved at all:

| Move against a raw-model pick | Games | Model side won | Market side won | Market % |
| --- | ---: | ---: | ---: | ---: |
| 0.5 to 1.0 | 88 | 48 | 40 | 45.5% |
| at least 1.0 | 59 | 28 | 31 | 52.5% |
| all | 147 | 76 | 71 | 48.3% |

**The 0.5-to-1.0 leader-median band loses on both surfaces.** That is the named
mechanism behind P3's +1.001: not "the tilt knows something", but "a half-point
drift across three books is not information, and the served rule spends 294 of its
522 firings inside that band."

### Post-hoc confound arm, added after the band table was seen

P4 raises the follow threshold from 0.5 to **1.0 for every game** and changes no
precedence at all — the card's own flips get no protection. Artifact
`artifacts/follow_vs_tilts/20260910T002107Z/confound.json`.

| Cell | vs | Changed | Delta (pts) | 95% week | P+ week |
| --- | --- | ---: | ---: | --- | ---: |
| **P4 threshold 1.0** | **P0** | 140 | **+2.003** | [-0.769, +4.851] | **0.9178** |
| P4 threshold 1.0 | CARD | 100 | +1.252 | [-1.122, +3.713] | 0.8462 |
| P3 protected below 1.0 | **P4** | 88 | **-1.001** | [-3.652, +1.757] | 0.2331 |

P4 scores **57.322%** on the 799 games, above every predeclared arm. Against P0 by
season it is +0.376 / +0.752 / +4.869. **P3 is worse than P4**, so protecting
mechanism flips ON TOP of a 1.0 threshold gives back a point. This arm was not
predeclared and its selection is stated out loud; it is recorded as a diagnostic,
not as a promotion candidate.

## Decision

**Among the four predeclared precedences, P3 has the higher expected accuracy and
P0 — what is served today — has the lowest but one.** P3 scores 56.320% against
P0's 55.319%, **+1.001 accuracy points paired, `probability_positive` 0.880**, on
52 changed picks. The pool is forced picks, so serving P0 over P3 is taking the
short side of an 88/12 bet on this evidence.

**But the precedence question itself answers NO.** The post-hoc P4 arm — the same
follow with its threshold moved to 1.0 and no protection for anything — beats both
(+2.003 over P0 at `probability_positive` 0.918, and P3 vs P4 is -1.001 at 0.233).
The thing that earns the gain is dropping the 0.5-to-1.0 band, which loses on
mechanism flips (22-30) and on raw-model picks (40-48) alike. A "protect the
tilts" guard would be a precedence rule bolted on to explain a threshold effect,
which is the pattern AGENTS.md bans on the played card.

Two things are worth saying plainly and neither closes anything. First, on this
population the served follow at 0.5 is **-0.751 accuracy points against simply not
following at all** (`probability_positive` 0.332) — it is not helping the
nine-member card here. Second, this is the historical-backfill replay, not the
live path the rule actually runs on, so it is evidence about the rule's shape and
not a re-grade of its promotion.

The strongest recommendation this lane supports is therefore **not a precedence
change but a threshold read**: re-run the leader-median follow's own promotion
comparison at 1.0 against 0.5, predeclared, on the live surface. Nothing here is
wired.

### What serving P3 would take (NOT wired here)

1. **The guard.** `src/nfl_ats/pick_refresh.py:1106` is the branch —
   `if late_week_fires:` — and it would become
   `if late_week_fires and (not protected or abs(late_week_net) >= 1.0):`, with
   `protected` read from the frozen Tuesday row rather than recomputed, so a
   refresh can never disagree with the card it is revising.
2. **The ledger column.** `composed_overlay_flip` already carries "the card
   flipped this pick" on every row
   (`src/nfl_ats/pick_refresh.py:1361`), and it is the right column for P1/P3,
   which protect the union rather than any single member. What does NOT exist is a
   per-member reason: only `coach_fade_flip`, `division_revenge_flip`,
   `player_arrests_flip` and `spread_gap_zone_flip` have their own columns
   (`src/nfl_ats/pick_refresh.py:1357-1360`), so **P2 is the only arm servable
   with today's columns** and it is the worst of the four. Serving P3 with a
   readable "protected by" reason means adding one string column — the member
   names behind the composed flip — to `PICK_REVISION_COLUMNS` and to the Tuesday
   card row that feeds it, since the six added members currently keep their
   provenance only on the composition result and in the card lineage
   (`docs/unserved_tilt_marginals.md`).
3. **A challenger ledger first.** The follow already writes a paired challenger
   (`late_week_move_follow_refresh_v1`); a precedence change should ride the same
   pattern — record P3's counterfactual side beside the served one for a season
   before it decides a pick.

## Registry

Twenty-one cells, family `follow_vs_tilts_precedence`, named
`follow_vs_tilts_<arm>_<window>`. All twenty-one are `unresolved_below_power`:
no arm has a resolved wrong sign (every arm's week-blocked interval reaches both
sides of zero), no trait here has zero split-half reliability, and the positive
control resolves at about six points while every arm sits at one to two, so the
control bounds nothing. The exact argv lists are saved as
`artifacts/follow_vs_tilts/20260910T002107Z/record_commands.json` and were run one
at a time against `registry/weak_signals.json`.

## Commands run

```
python artifacts/follow_vs_tilts/20260910T002107Z/lane_ad.py \
  --out artifacts/follow_vs_tilts/20260910T002107Z
python artifacts/follow_vs_tilts/20260910T002107Z/bands.py
python artifacts/follow_vs_tilts/20260910T002107Z/confound.py
python artifacts/follow_vs_tilts/20260910T002107Z/verify.py
nfl-ats weak-signals record ...   (21 cells; argv lists in record_commands.json)
```

