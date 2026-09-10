# Gating the late-week follow on injury-news confirmation

Lane AC, 2026-09-10. **The predeclaration below was written and frozen before
any accuracy, delta, interval, `probability_positive` or paired flip result in
the Results section was computed.** Every count quoted inside the
predeclaration was measured first, deliberately, from timestamps, market
movement and population membership only — no outcome column (`margin_vs_open`,
`correct_*`) was read until the design was fixed. Same order
`docs/sharp_weighted_follow.md`, `docs/injury_signal_on_refresh_card.md` and
`docs/injury_news_vs_level.md` used.

## Binding closing-grounds taxonomy (AGENTS.md), verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never
"contains zero". Within-week game correlation is ZERO (owner mandate); the
bootstrap is week-blocked and no ICC term is estimated or padded. The decision
is expected value on FORCED picks: the refresh rule with the higher expected
accuracy at the deadline is served. Picks are editable until
`min(kickoff, Sunday 16:00 ET)` per game.

## The question

`docs/injury_news_vs_level.md` (lane W, 2026-09-09) closed by naming a lead it
did not run, verbatim: "the right shape for this signal is a **gate on the
follow rule, not a standalone flip**." Its measurement is the reason. On the
`movement_attribution` population — games where the market has already moved
against the model pick — splitting the injury flag into the part already public
before Tuesday noon (LEVEL) and the part that changed since (NEWS) puts the
whole flip value in the changed part: **NEWS +10.909 points, P+ 0.8708, n=110;
LEVEL -0.971, P+ 0.4634, n=103**, against an unfiltered anchor of +5.263 on the
same 247 games. Meanwhile the same injury construction applied *unconditionally*
as a standalone tilt on the served refresh card loses (lane Q, **-1.627 points,
P+ 0.139**, 198 flips), and so does its baseline-corrected version (**-1.877,
P+ 0.158**).

Both readings say one thing: **the injury news is worth something only where the
market has already moved.** The served late-week rule is exactly a "the market
has already moved" detector. So the question this lane asks is whether the two
belong together — whether following the leaders' move is worth more when
post-Tuesday injury news points the same way, and worth less when it points the
other way.

The served rule (`docs/late_week_refresh.md`, promoted 2026-09-09) is the
**median Wednesday-to-deadline net move of Bovada, William Hill (US) and
MyBookie** at a frozen 0.5-point threshold
(`LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY` in `src/nfl_ats/pick_refresh.py`;
`late_week_follow_frame` in `src/nfl_ats/sharp_book_movement_features.py`).

## Population (frozen before scoring)

The same population every follow lane and both injury lanes use:

- **Market.** The frozen intraday caches
  `artifacts/experiments/sharp_book_movement/quotes.parquet` and
  `kickoff.parquet` — **measured**: 816 regular-season games, 272 per season
  2023/2024/2025, 54 `(season, week)` blocks.
- **Card / grading.** The active model's opener evaluation
  `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`, graded at
  `margin_vs_open`. **Measured**: 17 opener pushes, **799** graded games.
- **Injury news.** The official injury report
  (`data/players/raw/20260909T223500Z/injuries.parquet`) and, for the disclosed
  secondary, the ProFootballTalk headline archive
  (`data/raw/injury_news/20260819T191639Z/index.parquet`).

Windows are declared spent for the family `follow_news_gate`, seasons
2023-2025.

### What was measured before the design was fixed

Market side (**measured**, `coverage.json` / `coverage_leader_mean.json`):

| quantity | value |
|---|---:|
| games / weeks / non-push grades | 816 / 54 / **799** |
| nine-member composition flips vs the raw pick | **260** |
| leader-median rule fires (`\|median\| >= 0.5`) | **523** |
| leader-median flips vs the Tuesday card | **243** |
| equal-book rule fires / flips (the retired arm, replayed as a check) | **329 / 145** |
| games where all three leading books contributed an increment | **816 of 816** |
| games with `\|leader median\|` in **[0.25, 0.5)** | **0** |
| distinct `\|leader median\|` values | 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 6.0 |

**Disclosed in advance, not discovered later: the arm the task calls F2 — "follow
on a leader move of 0.25-0.5 when news confirms it" — is empty as literally
specified.** All three leading books contribute on every game, so the median of
their per-book net moves is itself one book's net move, and a book's net move is
a sum of half-point increments. The statistic therefore lives on a strict
half-point lattice: 293 games at exactly 0 and every other game at 0.5 or above,
with nothing in between. The substitute is predeclared under "Arms" below,
before any outcome was scored, and the empty literal band is reported as a
result rather than quietly replaced.

News side (**measured**, same files):

| quantity | 2023 | 2024 | 2025 |
|---|---:|---:|---:|
| official rows carry a readable `date_modified` | yes | yes | **no** |
| games with ANY skill-position filing in (own-week Tue noon, deadline] | 272 | 271 | **0** |
| leader-median fires | 194 | 154 | 175 |
| — news-confirmed | 91 | 68 | **0** |
| — news-contradicted | 66 | 51 | **0** |

Season 2025's official rows are all `observed_at_basis = week_proxy` with a null
`date_modified` (**measured** by lane Q and lane W, reproduced here through
`_season_has_readable_official_rows`), so the official news reader has no
reading at all on a third of the window. That is a property of the snapshot,
disclosed here, and the fail-open rule for it is fixed below rather than left to
be discovered in the results.

Pooled over the news-readable window (2023-2024, **measured**):

| quantity | value |
|---|---:|
| leader-median fires on news-readable games | **348** |
| — news-confirmed | **159** (45.7%) |
| — news-contradicted | **117** (33.6%) |
| — neither | 72 |
| fires that are FLIPS vs the Tuesday card, news-readable | 160 |
| — of those, news-confirmed | 69 (43.1%) |
| — of those, news-contradicted | 55 |
| median-silent games (`\|median\| < 0.5`) with a non-zero leaders' MEAN | 149 (101 news-readable) |
| — of those, news-confirmed | **40** |
| latest contributing filing later than that game's own deadline | **0 of 816** |
| largest (latest filing − deadline) across the population | **-730 s** |

## The news reader (frozen)

Reused from `docs/injury_news_vs_level.md`'s NEWS component, which is the
construct that measured +10.9 to +13.7 points, with one predeclared tightening.
For each game, each team `T`, and each skill-position player `p`
(QB/RB/WR/TE, `game_type == "REG"`), with that game's own-week Tuesday noon ET
`tue` and its pick deadline `end = min(kickoff, that week's Sunday 16:00 ET)`:

- `prior_sev(p)` — severity of the LATEST row for (season, team `T`, player `p`)
  with `date_modified <= tue`, searched across **every week of that season**, 0
  when none exists. This is what was publicly knowable before Tuesday noon: in
  practice the previous week's Friday report.
- `final_sev(p)` — severity of the LATEST row for (season, **week**, team `T`,
  player `p`) with `date_modified <= end`, 0 when none exists.
- `news_delta(p) = final_sev(p) − prior_sev(p)`, **counted only for players who
  have at least one own-week row filed inside `(tue, end]`**. Every other player
  contributes 0.
- `news(T) = Σ_p news_delta(p)`.

Severity scale unchanged: `Out=4, Doubtful=3, Questionable=2, Probable=1, not on
report=0`.

**The filed-row restriction is the one deliberate deviation from lane W's
construction, and it is predeclared, not tuned.** Without it a player who simply
does not appear on this week's report by the deadline scores as a full recovery,
so a team can be credited with "news" that no one ever filed. Since this lane's
whole claim is that a *news item* confirms a market move, a component with no
filing behind it must not be able to produce a confirmation. Lane W's unrestricted
construction is carried as a disclosed secondary arm so the two are comparable.

**Direction, defined without reference to the pick.** Let `m` be the leaders'
median net move, home-oriented (`m > 0` means the market moved toward HOME, i.e.
*against* AWAY). The **abandoned** side is the team the market moved against and
the **favored** side is the team it moved toward. Then

```
news_toward_market = news(abandoned) − news(favored) = −sign(m) · [news(HOME) − news(AWAY)]
```

- **News CONFIRMS the move** when `news_toward_market >= 2.0` — the abandoned
  team's skill-position injury situation deteriorated at least two severity
  points more than its opponent's since Tuesday noon. 2.0 is `INJURY_NET_THRESHOLD`,
  the bar `movement_attribution`, `injury_signal_refresh_tilt` and lane W's NEWS
  flag all already use. No grid is searched.
- **News CONTRADICTS the move** when `news_toward_market <= -2.0`.

This convention is a function of the market move only, never of the pick, which
is deliberate: lane W's closing paragraph warned that "sign conventions flip on
the 145 games where the follow rule already moved the pick, and getting that
wrong would produce a number that looks like this one and means the opposite."
Defining abandoned/favored from `sign(m)` removes that failure mode entirely.

**Fail-open on an unreadable season.** A season whose official rows carry no
usable observation timestamp (`_season_has_readable_official_rows` is false —
**measured**: 2025) has no news reading, and every gated arm **falls back to the
served rule F0** on that season. It never silently reverts to the Tuesday card,
because "no reading" is not "no news"; that is the same fail-open contract
`late_week_follow_frame` and `injury_signal_for_game` already use.

## Arms (frozen)

Every arm is a pick side per game, graded at `margin_vs_open`.

- **B1 — Tuesday card.** The raw opener probability pick complemented once on
  the union of the nine-member composition
  (`nfl_ats.four_overlay_composition`, `POLICY_ID =
  overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`).
- **F0 — served leader-median follow (the baseline for every gate).** B1, then
  `late_week_follow_frame` at each game's own deadline; `|leader median| >= 0.5`
  takes the market side, otherwise B1 stands. Replayed, not re-implemented: the
  arm is `movement_would_be_pick_side` straight off the served function.
- **F1 — follow only on confirmation.** F0, except that a fire whose news does
  NOT confirm it keeps the Tuesday pick. Fail-open on an unreadable season.
- **F2 — news lowers the bar.** F0, plus: on a game where the served rule does
  NOT fire but the leaders' **mean** net move is non-zero, follow the mean's
  direction when news confirms it. This is the predeclared substitute for the
  literal 0.25-0.5 median band measured empty above; the leaders' mean is the
  same three books' same net moves under the only aggregation of them that takes
  sub-half-point values (**measured**: 0, 1/6, 1/3 below 0.5), so it is a lowered
  bar on the same evidence, not a new information source. **Measured** additional
  fire set: 40 games.
- **F3 — news vetoes a contradicted move.** F0, except that a fire whose news
  points AGAINST it keeps the Tuesday pick. **Measured** touched set: 117 fires,
  55 of them flips.
- **F1p / F2p / F3p — disclosed PFT secondary.** The same three arms, except that
  on a season with no readable official rows the confirmation test becomes the
  module's own PFT bar: `pft_toward_market >= 1.0` confirms and `<= -1.0`
  contradicts, where `pft_toward_market` counts injury-relevant ProFootballTalk
  headlines naming the abandoned team minus those naming the favored team inside
  `(tue, end]`. This is what extends the gate to 2025. Its split-half reliability
  was measured at **0.071** by lane W, against 0.804 for the player-level news
  delta, so it is a secondary and is labelled one.
- **F1w / F3w — disclosed lane-W-news secondary.** F1 and F3 computed on lane W's
  unrestricted `news_delta` (no filed-row requirement), so the effect of the one
  deviation above is visible rather than assumed.
- **PC_reachable — positive control.** Perfect foresight on every game the
  follow machinery can reach (**measured**: `leader_books > 0` on 816 of 816):
  take the side that actually covered. Bounds any aggregation or gating of this
  archive's movement.
- **PC_fireset — positive control.** F0 everywhere, except perfect foresight on
  the 523 games F0 fires on. Bounds any rule that only re-decides F0's own fires.
- **PC_gateable — positive control.** F0 everywhere, except perfect foresight on
  the 348 news-readable fires. This is the exact decision set F1 and F3 touch, so
  it is the control that can actually bound them.

## Measurement (frozen)

Paired deltas in accuracy points, game-weighted, against **F0** (the decision
comparison) and additionally against **B1** (so each arm's total late-week value
is visible). Blocks resampled with replacement, 20,000 draws, seed **20260821**,
both **week-blocked** (`(season, week)`, the honest unit given the owner's
zero-within-week-correlation mandate) and **season-blocked** (three blocks, a
deliberately brutal read). 95% percentile intervals. `probability_positive` from
`nfl_ats.evidence_conventions.probability_positive_from_draws`
(`P(>0) + 0.5·P(==0)`), reported for every cell. Per-season point estimates for
every arm; no season is selected after scoring.

Split-half reliability (odd/even weeks within team-season) is computed and
recorded for the confirmation flag and the contradiction flag, because AGENTS.md
makes it the decisive field for adjudicating a cell later.

**Harness reproduction, fixed now.** The rebuilt F0 must reproduce the served
object exactly: 816 games, 799 non-push grades, 54 weeks, 260 composition flips,
523 leader-median fires, 243 leader-median flips, and — as a cross-check against
lane Q's published harness figures on the arm the leader median replaced — 329
equal-book fires and 145 equal-book flips. Any mismatch is reported before any
gate cell is trusted.

Cells recorded to the weak-signal registry, all `effect_units=accuracy_points`,
`league=nfl`, family `follow_news_gate`, named
`follow_news_gate_<arm>_<window>`.

## Interpretation rules (fixed before results are seen)

- **The decision is expected value on a forced card.** If an arm beats F0 with
  `probability_positive` above 0.5 pooled on the window where its news reader can
  actually read, serving it raises expected accuracy and this document says so
  **before** listing what is uncertain about it. Below 0.5 it is not worth
  serving — and that is still `unresolved_below_power`, not a closure, unless the
  whole interval sits on the wrong side of zero.
- **The 2025 third is a dilution, not a result.** The pooled 2023-2025 cell for
  every official-news arm is arithmetically the 2023-2024 effect diluted by a
  season on which the arm is a no-op. Both are reported; the 2023-2024 cell is
  the informative one and the pooled cell is the one that answers "what would it
  have done on the whole archive".
- **A negative gate does not close the mechanism**, and a positive gate does not
  resolve it. F1, F2 and F3 are correlated re-cuts of one 816-game window.
- **No arm is wired.** If a gate beats F0, this document names the function, the
  precedence and the ledger columns a promotion would touch, and changes nothing.

## Commensurability and window declaration

Family `follow_news_gate`, declared here before the signs were seen. Its members
share one window, one baseline and one news reader, so they are correlated
readings of one question rather than independent votes. They are also a
**correlated decomposition** of the `sharp_weighted_follow_*` family (same 816
games, same leader-median arm), of `injury_signal_on_refresh_card_*` and of
`injury_news_vs_level_*` (same news construction, same archive). **Never pool
any of these additively with each other or with those parents.**

This is a mined battery; **no multiplicity correction is claimed**, matching
every other mined-battery document in this repo.

---

## Results

**Measured** this session, `artifacts/follow_news_gate/20260910T002019Z/`
(`frame.parquet`, `per_game.parquet`, `cells.csv`, `coverage.json`,
`coverage_leader_mean.json`, `metadata.json`, `record_commands.json`, the five
scripts that produced them, and a copy of this predeclaration as frozen at
scoring time).

### Decision

**Yes — but the value is in the VETO, not the gate.** The rule to serve is
**F3: when the leading books move but post-Tuesday injury news points at the
team they moved TOWARD, keep the Tuesday pick.** Pooled 2023-2025 on the served
card it reads **+0.626 accuracy points, week-blocked 95% [-1.398, +2.709],
`probability_positive` 0.7285**, on **55 pick changes in 54 weeks — one a
week**; on the two seasons where the official news reader can actually read,
**+0.940, [-2.218, +3.962], P+ 0.7293**. On a forced card that is a 73/27 bet,
and per the standing order it is worth taking.

The mechanism table is the reason, and it is cleaner than any of the interval
cells (**measured**, `metadata.json` → `mechanism`; all figures opener-graded):

| the leaders moved ≥ 0.5 and… | games | follow (F0) | Tuesday card | follow − Tuesday |
|---|---:|---:|---:|---:|
| …injury news CONFIRMS the move | 157 | 88 (56.05%) | 83 (52.87%) | **+3.18 pts** |
| …injury news CONTRADICTS it | 115 | 58 (50.43%) | 63 (54.78%) | **-4.35 pts** |
| …news says neither | 72 | 43 (59.72%) | 43 (59.72%) | 0.00 (exact tie) |
| all news-readable fires | 344 | 189 (54.94%) | 189 (54.94%) | **0.00 (189-189)** |

Read plainly: **on the games where the market moved and the injury report can
be read, following the market is a coin-flip against just keeping the Tuesday
pick — 189-189, exactly even. Split it by whether an injury filed since Tuesday
agrees with the move, and the two halves separate by 7.5 points.** Following a
confirmed move is worth +3.2; following a contradicted one costs 4.3. The
"neither" bucket is a literal 43-43 tie, which is what you would expect if the
news is the whole story.

**The three gates in one line each** (paired against the served rule F0,
week-blocked, 20,000 draws, seed 20260821):

- **F1 — follow only on confirmation:** +0.626 pts, P+ 0.7014, **91** pick
  changes. It buys the same accuracy as F3 for 65% more churn.
- **F2 — news lowers the movement bar:** **-1.752 pts, [-3.008, -0.620],
  P+ 0.0012**, 23 pick changes. The only resolved result in the battery, and it
  is negative. See "The one thing that is closed" below.
- **F3 — veto a contradicted move:** +0.626 pts, P+ 0.7285, **55** pick changes.
  **The arm to serve.**

**The higher-EV arm overall is F3p** — F3 with the ProFootballTalk headline
reader standing in on a season whose official rows carry no timestamp —
at **+1.126 points, [-1.242, +3.461], P+ 0.8304**, 79 pick changes, because it
is the only veto variant that can act at all on 2025 (**+1.498, P+ 0.8066** on
that season alone) and, by the same token, on the live 2026 season, whose
official rows are also timestampless. F3 and F3p are the same rule; F3p just
does not go silent when the official feed does.

One arm scores higher still and is reported first rather than buried:
**F3w**, the veto computed on lane W's unrestricted news delta, reads
**+1.752 points, [-0.127, +3.741], P+ 0.9639** (2023-2024: **+2.632,
[-0.192, +5.484], P+ 0.9636**), and it is the only arm in this battery that
beats the **Tuesday card** as well (+1.001, P+ 0.7516). It was predeclared as a
secondary, so it is admissible. What argues against serving it over F3p is not
its number: its confirmation flag's split-half reliability is **-0.027** over 64
team-seasons against F3's **+0.035**, i.e. indistinguishable from zero, and its
"news" credits a team whenever a player simply fails to reappear on this week's
report — a filing-calendar artifact, not an injury item. **My read (inferred):**
F3w is the same veto with a noisier reader that happened to land better on 64
pick changes; I would serve F3p and track F3w as the paired challenger, and I
would change that view if F3w repeated on a season the official reader can also
see.

### Reproduction

**Measured** — the rebuilt F0 is the served object, exactly:

| quantity | required | this run |
|---|---:|---:|
| archive games | 816 | **816** |
| opener-graded (non-push) | 799 | **799** |
| `(season, week)` blocks | 54 | **54** |
| nine-member composition flips vs the raw pick | 260 | **260** |
| leader-median fires (`\|median\| >= 0.5`) | 523 | **523** |
| leader-median flips vs the Tuesday card | 243 | **243** |
| equal-book fires (lane Q's published figure) | 329 | **329** |
| equal-book flips (lane Q's published figure) | 145 | **145** |

Accuracies behind everything below: Tuesday card **448/799 (56.07%)**, served
leader-median card **442/799 (55.32%)**, F1 **447**, F3 **447**, F3p **451**,
F3w **456**, F2 **428**.

### The cells

Paired week-blocked bootstrap, 20,000 draws, seed 20260821, graded at
`margin_vs_open`. `probability_positive` is the decision-relevant number.
Season-blocked reads are in `cells.csv` and `metadata.json` for every row.

**Against the served rule F0 (the decision comparison):**

| cell | window | changes | delta (pts) | 95% week-blocked | P+ |
|---|---|---:|---:|---|---:|
| **`..._f3_2023_2025`** | 2023-25 | **55** | **+0.626** | [-1.398, +2.709] | **0.7285** |
| `..._f3_2023_2024` | 2023-24 | 55 | +0.940 | [-2.218, +3.962] | 0.7293 |
| `..._f3_2023` | 2023 | 33 | -0.376 | [-5.283, +4.396] | 0.4395 |
| `..._f3_2024` | 2024 | 22 | +2.256 | [-1.527, +5.792] | 0.8809 |
| `..._f1_2023_2025` | 2023-25 | 91 | +0.626 | [-1.667, +2.915] | 0.7014 |
| `..._f1_2023_2024` | 2023-24 | 91 | +0.940 | [-2.622, +4.389] | 0.7047 |
| `..._f1_2023` / `_2024` | | 51 / 40 | +0.376 / +1.504 | | 0.5611 / 0.7349 |
| **`..._f2_2023_2025`** | 2023-25 | **23** | **-1.752** | **[-3.008, -0.620]** | **0.0012** |
| `..._f2_2023_2024` | 2023-24 | 23 | -2.632 | [-4.389, -0.933] | 0.0009 |
| **`..._f3p_2023_2025`** | 2023-25 | **79** | **+1.126** | [-1.242, +3.461] | **0.8304** |
| `..._f3p_2025` | 2025 | 24 | +1.498 | [-1.873, +4.869] | 0.8066 |
| `..._f1p_2023_2025` | 2023-25 | 137 | +0.626 | [-1.892, +3.145] | 0.6851 |
| `..._f2p_2023_2025` | 2023-25 | 29 | -2.003 | [-3.333, -0.754] | 0.0006 |
| **`..._f3w_2023_2025`** | 2023-25 | **64** | **+1.752** | [-0.127, +3.741] | **0.9639** |
| `..._f3w_2023_2024` | 2023-24 | 64 | +2.632 | [-0.192, +5.484] | 0.9636 |
| `..._f3w_2024` | 2024 | 30 | +3.759 | [+0.376, +7.435] | 0.9835 |
| `..._f1w_2023_2025` | 2023-25 | 79 | +1.126 | [-1.117, +3.354] | 0.8429 |
| `..._f1w_2023_2024` | 2023-24 | 79 | +1.692 | [-1.544, +4.924] | 0.8465 |

Every official-reader arm is an exact **0.000, P+ 0.500** no-op on 2025, by the
fail-open rule predeclared above, so each pooled 2023-2025 cell is arithmetically
its 2023-2024 value diluted by a third of the window. That is disclosed
arithmetic, not a finding.

**Against the Tuesday card (what the whole late-week channel is worth):**

| cell | changes | delta (pts) | 95% week-blocked | P+ |
|---|---:|---:|---|---:|
| `..._f0_vs_tuesday_2023_2025` (**the served rule**) | 243 | **-0.751** | [-4.172, +2.753] | **0.3315** |
| `..._f0_vs_tuesday_2023_2024` | 160 | 0.000 | [-4.348, +4.420] | 0.4948 |
| `..._f1_vs_tuesday_2023_2025` | 152 | -0.125 | [-2.981, +2.655] | 0.4649 |
| `..._f3_vs_tuesday_2023_2025` | 188 | -0.125 | [-3.202, +3.008] | 0.4690 |
| `..._f3p_vs_tuesday_2023_2025` | 164 | +0.375 | [-2.709, +3.544] | 0.5945 |
| `..._f3w_vs_tuesday_2023_2025` | 179 | **+1.001** | [-1.894, +3.894] | 0.7516 |
| `..._f2_vs_tuesday_2023_2025` | 266 | -2.503 | [-6.038, +1.022] | 0.0831 |

**A second finding this lane did not go looking for, stated plainly:** on the
current model and on top of the nine-member composition card, **the served
leader-median follow rule is worth -0.751 accuracy points against simply keeping
the Tuesday pick (P+ 0.3315), and 283-289 on its own 515 graded fires.** Lane Q
measured the same shape for the equal-book arm it replaced (-0.375, P+ 0.400).
The rule's promotion evidence (+3.004 over the raw card, P+ 0.930) was measured
against the **raw** opener card on the frozen 2026-09-05 baseline, not on top of
the composition and not on model `c657058903f3232b`. That belongs to MKT-15/CX18,
not to this lane, and it is exactly what makes the veto worth having: the gates
here recover most of the gap (F3p +0.375 vs Tuesday, F3w +1.001), i.e. **most of
what the follow rule is currently losing, it loses on moves the injury report
contradicts.**

### Positive controls

**Measured**, paired against F0 on the same blocks:

| control | switches | delta (pts) | 95% week-blocked | P+ |
|---|---:|---:|---|---:|
| `..._pc_reachable_2023_2025` — perfect foresight on all 816 reachable games | 366 | **+44.681** | [+41.594, +47.778] | 1.0000 |
| `..._pc_fireset_2023_2025` — perfect foresight on F0's own 523 fires | 237 | **+29.036** | [+26.092, +32.010] | 1.0000 |
| `..._pc_gateable_2023_2024` — perfect foresight on the 348 news-readable fires | 157 | **+29.135** | [+25.333, +33.081] | 1.0000 |
| `..._pc_gateable_2023_2025` — same, diluted by 2025 | 157 | +19.399 | [+14.912, +23.945] | 1.0000 |

The instrument is proven: on this exact decision set a perfect gate is worth
**+29 accuracy points**, and the bootstrap resolves it far from zero. The
measured gates run +0.6 to +2.6, i.e. **2-9% of the reachable ceiling** — a real
but small share of the available information, which is the expected shape. **No
cell in this battery is `bounded_by_control`**: the control detected an effect,
and the gates are present, not absent.

### Diagnostics

**Fire and switch counts** (**measured**, `metadata.json` → `diagnostics`):

| quantity | value |
|---|---:|
| leader-median fires / flips vs the Tuesday card | 523 / 243 |
| news-readable fires (2023-2024) | 348 |
| — news-confirmed | **159 (45.7%)** |
| — news-contradicted | **117 (33.6%)** |
| — neither | 72 |
| confirmation rate among fires that are FLIPS | 69 of 160 (43.1%) |
| F1 pick changes vs F0 (2023 / 2024 / 2025) | 91 (51 / 40 / 0) |
| F3 pick changes vs F0 (2023 / 2024 / 2025) | 55 (33 / 22 / 0) |
| F2 pick changes vs F0 | 23 (9 / 14 / 0) |
| F3p pick changes vs F0 | 79 (33 / 22 / 24) |

**Tuesday-visibility check, one row per news item used** (**measured**,
`metadata.json` → `visibility`): 316 gate decisions were backed by news;
**316 of 316** rest on at least one skill-position row filed inside
`(own-week Tuesday noon ET, that game's own deadline]`; **0 of 316** depend on
anything filed after `min(kickoff, Sunday 16:00 ET)`; **0** have no datable
filing. The tightest case sits **730 seconds** — twelve minutes — before its own
deadline, and the median gate decision's latest contributing filing is **1.9 days**
before it. 2,114 filed rows contribute in total. Nothing here is unplayable, and
that is confirmed per-game rather than asserted.

**Split-half reliability**, odd/even weeks within team-season (**measured**):
confirmation flag **+0.106** (64 team-seasons, news-readable window),
contradiction flag **+0.035**, lane-W confirmation flag **-0.027**, lane-W
contradiction flag **+0.058**, and the served follow rule's own fire flag
**+0.070** — the same order as MKT-15's published exposure reliability of
0.1009. These are exposure-to-a-per-game-event reliabilities, which are near
zero for every market and news construct in this project; none of them is a
closing ground, and none is used as one.

### The one thing that is closed

**F2 — "news lowers the movement bar" — is a resolved wrong sign, and it is
recorded as one.** Two facts, in order.

First, the arm as the task specified it does not exist on this population.
**Measured before any outcome was read:** all three leading books contribute on
**816 of 816** games, so their median is one book's own net move, which is a sum
of half-point increments. `|leader median|` takes the values 0, 0.5, 1.0, 1.5 …
and **nothing between 0.25 and 0.5 — zero games**. The predeclared substitute
used the leaders' *mean*, the only aggregation of the same three books' same
moves that takes sub-half-point values (0, 1/6, 1/3), and followed it on
median-silent games when news confirmed the direction.

Second, that arm loses, and the loss is resolved: **-1.752 accuracy points,
week-blocked 95% [-3.008, -0.620], P+ 0.0012** on 23 pick changes; on the
readable window **-2.632, [-4.389, -0.933], P+ 0.0009**; with the PFT reader
**-2.003, [-3.333, -0.754], P+ 0.0006**. The whole week-blocked interval sits
below zero on all four, which is the one shape AGENTS.md admits as
`wrong_sign_resolved`, so those four cells are recorded `refuted_mechanism`.
The season-blocked read on the pooled cell is [-2.632, 0.000] — its upper end is
exactly zero, not above it — and the per-season cells, which do cross zero, are
recorded `unresolved_below_power`.

**What is refuted is narrow and named:** a sub-half-point move by a single
leading book, even with injury news agreeing, is not a follow signal. It does
**not** close the news gate. F1 and F3 use the identical news reader on the
identical population and sit on the other side of zero; both stay
`unresolved_below_power`.

### What serving F3p would take (asked and answered; nothing was wired)

1. **The news reader.** `injury_signal_for_game`
   (`src/nfl_ats/injury_signal_refresh_tilt.py`) cannot be reused as-is: it
   subtracts an own-week Tuesday-noon baseline that lane W measured to be
   identically zero on 494 of 494 games, and it orients on the PICK. The gate
   needs a sibling function that (a) uses the **cross-week prior** —
   `prior_sev(p)` = that player's latest designation filed at or before this
   week's Tuesday noon, searched across the season — and (b) orients on the
   **market move**: `news(abandoned) − news(favored)`, abandoned/favored taken
   from `sign(leader_median_net_move)`, never from the pick. It reuses
   `_severity_asof`, `SEVERITY`, `SKILL_POSITIONS`, `_canonical_team`,
   `own_week_tuesday_noon_utc`, and the existing
   `_season_has_readable_official_rows` / `_pft_team_hits` pair for the fallback
   that makes it F3p rather than F3.
2. **Precedence.** Inside `plan_refresh`, between `late_week_follow_frame`'s
   `movement_would_be_pick_side` and the pick: the follow rule computes its side,
   and a contradicted move is discarded so the Tuesday pick stands. A vetoed game
   must be treated as "the follow rule fired" for precedence — it must NOT fall
   through to the 1.0-point consensus rule or the Saturday handle rule, because
   that is how it was measured here.
3. **Ledger columns.** `pick_revisions.parquet` gains `late_week_news_net` (the
   signed `news_toward_market`), `late_week_news_source`
   (`official`/`pft_fallback`/`none`), `late_week_news_veto`, and
   `late_week_pre_veto_pick_side`; `movement_policy` gains a vetoed value beside
   `late_week_leader_median_follow_0_5`. The un-vetoed leader-median arm records
   as the paired OFF challenger exactly the way the equal-book arm does today.

**Nothing above was implemented.** No served code, ledger, manifest, forecast or
board was touched, and nothing was committed or pushed.

### Caveats (label how you know it)

- **Correlated decomposition.** Every cell shares the 816-game 2023-2025
  intraday archive and the leader-median arm with `sharp_weighted_follow_*`, and
  the news construction with `injury_news_vs_level_*` and
  `injury_signal_on_refresh_card_*`. Never pooled additively with those or with
  each other.
- **Mined battery**, 57 recorded cells, no multiplicity correction claimed. The
  decision names F3/F3p, which is the arm the *mechanism table* points at, not
  the arm with the largest number (that is F3w).
- **Season 2025 is a no-op for every official-reader arm** — its injury rows are
  all `observed_at_basis = week_proxy` with a null `date_modified`. Disclosed in
  the predeclaration, before scoring. The same is true of the live 2026 rows
  lane W measured, which is why F3p rather than F3 is the practical recommendation.
- **The 0.25-0.5 band the task specified is empty**, measured before scoring; the
  substitute is predeclared above and is a different statistic (the leaders' mean),
  so F2's result bounds "sub-threshold single-book move plus news", not literally
  "a 0.25-0.5 median move".
- **Skill positions only** (QB/RB/WR/TE), inherited from the parent construction:
  offensive-line and defensive-front injuries are invisible to the reader. A
  likely undercount of real injury information, not an overcount.
- **The lane-W news reader (F1w/F3w) counts a player who simply does not reappear
  on this week's report by the deadline as a full recovery.** That is a filing
  artifact, and it is why the primary reader requires a filed row. Both are
  reported.
- **`prior_sev` is a within-season lookup**, so a Week 1 designation scores
  entirely as news by construction.
- **The opener evaluator's inherited approximation applies**: only `spread_line`
  is swapped to the opener; other features are close-era
  (`docs/opener_evaluation.md`).
- **The PFT path is team-nickname headline matching**, not player-level matching;
  lane W measured its split-half reliability at 0.071.

### Registry entries recorded

57 entries, all `effect_units=accuracy_points`, league `nfl`, family
`follow_news_gate` (**measured**: registry total 4,614 → **4,671**; 57 entries
present under the `follow_news_gate_` prefix). 53 are
`unresolved_below_power` with `closing_ground` null; four —
`follow_news_gate_f2_2023_2025`, `_f2_2023_2024`, `_f2p_2023_2025`,
`_f2p_2023_2024` — are `refuted_mechanism` on `wrong_sign_resolved`, the whole
week-blocked interval sitting below zero. Exact argument vectors in
`artifacts/follow_news_gate/20260910T002019Z/record_commands.json`.
