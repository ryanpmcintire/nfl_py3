# Is the +17.07-point injury cell news, or is it a level?

Lane W, 2026-09-09. **The predeclaration below was written and frozen before
any accuracy, delta, interval or flip count in the Results section was
computed.** Every coverage count quoted inside the predeclaration was measured
first, deliberately, from timestamps and population membership only — no
outcome column (`margin_vs_open`, `correct_*`, `oracle_correct_*`) was read
until the design was fixed. Same order `docs/injury_signal_on_refresh_card.md`
and `docs/sharp_book_movement_lead.md` used.

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
is expected value on FORCED picks. Picks are editable until
`min(kickoff, Sunday 16:00 ET)` per game.

## The question

`docs/movement_attribution.md`'s `pop_threshold_injury` cell — **+17.07
accuracy points, week-blocked 95% [+0.79, +31.67], `probability_positive`
0.976, n=123** — is the largest registered standalone number in this project,
and it is described as "acting on injury news that moves the line". It is the
evidence chain the whole `injury_signal_refresh_tilt` challenger was built on
(that module's own docstring cites it first).

Lane Q (`docs/injury_signal_on_refresh_card.md`, 2026-09-09) measured the tilt
built on that cell at **-1.627 points on the served refresh card, P+ 0.139**,
and named a mechanism defect: the construction computes
`player_delta = severity(final) − severity(own-week Tuesday noon ET)`, but on
its own window only 7 of 816 games have ANY official skill-position row filed
by their own-week Tuesday noon, so the subtraction is against zero and
`net_injury_score` is the picked team's whole-week injury **level** minus the
opponent's, not a post-Tuesday **delta**. Lane Q then *inferred*, without
measuring it, that the +17.07 cell shares the construction and is therefore
also a level flag.

This lane measures that inference on the +17.07 cell's own population, and
asks the follow-on the tilt result leaves open: the one arm lane Q found
positive was the ProFootballTalk headline path on 2025 alone (reported by that
document as +1.498, P+ 0.716, 88 flips — re-measured here, not taken on
faith), the only arm whose input is a genuine post-Tuesday news window. On
2023 and 2024 the official path pre-empts it, because the module selects the
official path on the mere presence of rows for that season.

Three questions, in order:

1. **Reproduce** the +17.07 cell exactly on its own archive, or say precisely
   why not.
2. **Split** its INJURY flag into a NEWS half and a LEVEL half and re-score
   each against the same baseline. Is the cell news, or is it a level?
3. **Run the PFT construction as the tilt's only path on 2023-2024**, not just
   2025, on top of the served refresh card, exactly as lane Q scored it.

And one code question, answered by the measurement: should the module's path
selection require official rows that carry real observation timestamps, rather
than firing on their mere presence?

## Coverage, measured before the design was fixed

**Measured** this session (all from timestamps and population membership; no
outcome column read), scratchpad `coverage.py`:

| quantity | value |
|---|---:|
| POP_UNFILTERED (`open_move != 0`, market disagrees with the model pick, push-excluded) | **494** |
| POP_THRESHOLD (`\|open_move\| >= 1.0`) | **290** |
| POP_UNFILTERED games in seasons 2020-2024 (official-report path) | 412 |
| POP_THRESHOLD games in seasons 2020-2024 | 247 |
| POP_UNFILTERED games in season 2025 (PFT-headline path) | 82 |
| POP_THRESHOLD games in season 2025 | 43 |

Official injury snapshot `data/players/raw/20260817T184901Z/injuries.parquet`
— the file `scripts/movement_attribution.py` names, **measured**: seasons
2009-2024, `date_modified` non-null in every row of every season from 2010 on,
and the file carries **no `observed_at` / `observed_at_basis` column at all**.
So on the +17.07 cell's own official window every timestamp is real; the
`week_proxy` problem lane Q found belongs to the newer snapshot
(`20260909T223500Z`, **measured**: 6,068 of 6,068 season-2025 rows
`observed_at_basis = week_proxy` with a null `date_modified`, and 29 of 29 for
2026).

The filing calendar on the cell's own window, skill positions, REG,
seasons 2020-2024 (**measured**, 8,461 rows):

| filing day (ET) | rows |
|---|---:|
| Friday | **6,527** |
| Wednesday | 755 |
| Saturday | 623 |
| Thursday | 488 |
| Tuesday | 34 |
| Monday | 21 |
| Sunday | 13 |

And the consequence, on the population itself (**measured**):

| quantity | value |
|---|---:|
| own-week rows filed at or before own-week Tuesday noon ET, across all 494 games' two teams | **5** |
| POP_UNFILTERED games with ANY own-week pre-Tuesday-noon row | **4 of 494** |
| POP_THRESHOLD games with ANY own-week pre-Tuesday-noon row | **0 of 290** |
| rows landing inside (own-week Tuesday noon, kickoff] | 2,583 |

**Disclosed in advance, not discovered later:** on all 290 games of
POP_THRESHOLD — the cell that reads +17.07 — the own-week Tuesday-noon
severity baseline is empty for both teams, so `net_injury_score` reduces
*exactly* to the final skill-position severity sum of the abandoned side minus
that of the favored side. That is arithmetic, visible before any outcome is
scored, and it is why this lane exists.

A **usable pre-Tuesday baseline does exist**, just not inside the own week.
**Measured**: 760 of 988 population team-games have at least one
skill-position row filed at or before that week's Tuesday noon somewhere
earlier in the same season (in practice, the previous week's Friday report,
public well before this Tuesday); both teams have one on **376 of 494**
POP_UNFILTERED games and **230 of 290** POP_THRESHOLD games. That is the
baseline this lane's NEWS/LEVEL split uses, and it is the thing the module's
own-week-only baseline throws away.

PFT archive `data/raw/injury_news/20260819T191639Z/index.parquet`
(**measured**): 20,655 injury-relevant rows with a parseable `lastmod`,
including 1,676 in 2023, 1,775 in 2024 and 1,695 in 2025 — a genuine
post-Tuesday news window in every season of lane Q's card window, not just
2025.

## Part 1 — reproduction (frozen)

Re-run `scripts/movement_attribution.py`'s population and INJURY construction
verbatim (`load_population`, `compute_official_injury_scores`,
`compute_pft_fallback_scores`, `attach_injury_flag`) on the same archive, and
re-score `POP_THRESHOLD ∩ INJURY` with that document's own bootstrap spec:
`nfl_ats.clv.week_blocked_bootstrap`, `block="week"`, `samples=2000`,
`seed=20260820`. **Pass condition, fixed now:** n=123 and paired delta +17.07
points with `probability_positive` 0.976, matching
`docs/movement_attribution.md`'s table to its own reported precision. Anything
else is reported as a failure to reproduce, with the reason, and every
downstream number in this document is then labelled as resting on an
unreproduced parent.

The same cell is also re-reported at this lane's own spec (20,000 draws, seed
20260821) so the split halves below are commensurable with their parent.

## Part 2 — the NEWS / LEVEL split (frozen)

The split is an **exact additive decomposition** of the flag's own score, not
a re-tuning of it.

For each population game, each of the two teams T, and each skill-position
player p (QB/RB/WR/TE, `game_type == "REG"`), with that game's own-week
Tuesday noon ET `tue` and its kickoff `k`:

- `prior_sev(p)` — the severity of the LATEST row for (season, team T,
  player p) with `date_modified <= tue`, **searched across every week of that
  season**, not just the game's own week; 0 when none exists. This is what was
  publicly knowable about that player before Tuesday noon: in practice the
  previous week's final report.
- `final_sev(p)` — the severity of the LATEST row for (season, week, team T,
  player p) with `date_modified <= k`; 0 when none exists. Identical to
  `movement_attribution`'s `final_status`.
- `news_delta(p) = final_sev(p) − prior_sev(p)` — the genuine post-Tuesday
  change: a brand-new designation, a deterioration, or a recovery.

Severity scale unchanged from the parent document: `Out=4, Doubtful=3,
Questionable=2, Probable=1, not on report=0`.

With `A` the abandoned side (`moved_against_team`) and `F` the favored side:

- `LEVEL_net = Σ_p prior_sev(p, A) − Σ_p prior_sev(p, F)`
- `NEWS_net  = Σ_p news_delta(p, A) − Σ_p news_delta(p, F)`

**Identity:** `LEVEL_net + NEWS_net = Σ final_sev(A) − Σ final_sev(F)`, which
is the parent's `net_injury_score` on every game whose own-week Tuesday-noon
baseline is empty — measured above to be 490 of 494 and 290 of 290. The
identity is checked numerically per game and any residual is reported, not
hidden.

Flags, at the parent's own predeclared bar of 2.0, unchanged:

- **NEWS flag** = `NEWS_net >= 2` — the post-Tuesday news alone clears the bar.
- **LEVEL flag** = `LEVEL_net >= 2` — the already-knowable injury level alone
  clears the bar.

The halves are not disjoint, so five cuts are reported, never just two:
`NEWS`, `LEVEL`, `NEWS_ONLY` (NEWS and not LEVEL), `LEVEL_ONLY`,
`BOTH`, plus `SPLIT_NEITHER` (the parent INJURY flag fires but neither
component alone clears 2 — e.g. `NEWS_net = 1`, `LEVEL_net = 1`).

**Path handling, disclosed in advance.** The official path covers seasons
2020-2024. Season 2025 has no official coverage in this snapshot and the
parent uses the PFT-headline fallback, whose `lastmod ∈ (Tuesday noon,
kickoff]` window makes it a **news** construct by construction — it has no
level component to separate. The PRIMARY split cells are therefore computed on
the **official-path subpopulation** (2020-2024: 412 / 247 games), so NEWS and
LEVEL are the same kind of quantity measured the same way. A disclosed
secondary reports the whole population with 2025's PFT flags folded into NEWS.

**Scoring.** Identical to the parent in every respect but the bootstrap size:
candidate pick = `oracle_pick_home` (the market's side), baseline pick =
`pick_home_at_open_probability_rule` (the production pick), graded at
`margin_vs_open`, paired per-game accuracy delta × 100 (`accuracy_points`),
week-blocked bootstrap resampling whole `(season, week)` blocks with
replacement, **20,000 draws, seed 20260821**, 95% percentile interval,
`probability_positive` from
`nfl_ats.evidence_conventions.probability_positive_from_draws`
(`P(>0) + 0.5·P(==0)`).

**Split-half reliability** of each component score (`NEWS_net`, `LEVEL_net`)
is computed over odd/even team-seasons on the same window and recorded with
every cell, because AGENTS.md makes it the decisive field for adjudicating a
cell later.

## Part 3 — the PFT construction as the tilt's only path, 2023-2024 (frozen)

Lane Q's harness, reused unchanged
(`artifacts/injury_signal_on_refresh_card/20260909T231954Z/laneQ_measure.py`
and `nfl_ats.injury_signal_refresh_tilt`), on lane Q's frozen population:
`artifacts/experiments/sharp_book_movement/{kickoff,quotes}.parquet` (816
regular-season games, 272 per season 2023-2025) graded on the active model's
opener evaluation `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`
at `margin_vs_open`.

Arms, all graded at the frozen Tuesday opener:

- **B1 — Tuesday card**: the raw probability pick complemented on the union of
  the nine-member composition (`nfl_ats.four_overlay_composition`).
- **S — served refresh card** (the baseline for every Part 3 cell): B1, then
  `late_week_follow_frame` at each game's own `min(kickoff, Sunday 16:00 ET)`
  deadline; `|equal_net_move| >= 0.5` takes the market side.
- **S+PFT — the candidate**: S, plus `injury_signal_for_game` called with
  `injuries=None` on **every** season, which forces the module's own
  `net_pft_score >= 1` path over the `(own-week Tuesday noon, deadline]`
  headline window for 2023 and 2024 as well as 2025. `fires` flips S's side.
  Nothing else about the module changes; the same thresholds, the same
  nickname table, the same `now = deadline` cutoff.

Cells: per season 2023, 2024, 2025; pooled 2023-2024 (the two seasons the
official path currently pre-empts); pooled 2023-2025 (the decision cell). Same
bootstrap spec as Part 2 (20,000 draws, seed 20260821).

**Harness reproduction check, fixed now:** the rebuilt S arm must reproduce
lane Q's served card exactly — 816 games, 799 non-push grades, 54 weeks, 260
composition flips vs the raw pick, 329 follow-rule firings, 145 follow-rule
flips vs the Tuesday card, and 445/799 correct. Any mismatch is reported
before any Part 3 number is trusted.

Diagnostics reported alongside, never instead of, the cells: flip counts per
season and per week; the collision table against the follow rule (how many PFT
flips land on a game the follow rule already moved, and how many reverse an
actual follow-rule flip); and the **Tuesday-visibility check** — for every
firing, the latest contributing headline `lastmod` against that game's
`min(kickoff, Sunday 16:00 ET)` deadline, counting firings that are not
decidable inside the pool's own editing window.

## Part 4 — the path-selection question (decision rule fixed in advance)

`injury_signal_for_game` selects the official path whenever
`injuries["season"].eq(season).any()` — the mere presence of rows. When those
rows carry no usable observation timestamp, `_severity_asof`'s
`date_modified <= cutoff` filter admits nothing, both team deltas are 0.0, and
the module returns `net_score = 0.0` and `fires = False` for every game of
that season while reporting `source = "official"`. That is a silent no-op
presented as a reading.

**The fix is implemented in this worktree if and only if** the measurement
shows both of: (a) a season exists whose official rows cannot be read by the
official path (already **measured** above: 2025 is 6,068 of 6,068 `week_proxy`
with null `date_modified`, 2026 is 29 of 29), and (b) the PFT alternative on
that season is not a resolved wrong sign. The fix is a **selection** change
only — require the target season's official rows to carry a usable observation
timestamp before choosing the official path, and otherwise fall through to the
PFT path the module already implements. No threshold, no scoring, no new rule
is added, and nothing is wired into the served card by this lane.

## Interpretation rules (fixed before results are seen)

- **The split decides the description, not the fate.** If the flip-value
  concentrates in LEVEL and is absent from NEWS, the +17.07 cell is a level
  flag and every document that describes it as "acting on injury news" is
  overstating what was measured — including this project's own front-running
  sketch and `injury_signal_refresh_tilt`'s docstring. If it concentrates in
  NEWS, the news description stands and lane Q's inference is wrong. Either
  way the parent cell stays `unresolved_below_power`: a re-description is not
  a closing ground, and neither is an interval that contains zero.
- **Part 3 is decided on expected value.** If S+PFT vs S reads
  `probability_positive` above 0.5 pooled on 2023-2025, serving it raises
  expected accuracy on a forced card and this document says so **before**
  listing what is uncertain about it. Below 0.5, it is not worth serving — and
  that is still `unresolved_below_power` unless the whole interval sits on the
  wrong side of zero.
- **A negative half does not close the other half.** The NEWS and LEVEL cells
  are correlated subcuts of one population; neither bounds the other.
- No arm here is wired into production. If Part 3 comes out positive, this
  document names the refresh pass, the module and the precedence — it does not
  change any served code, ledger, manifest, forecast or board.

## Commensurability and window declaration

Windows are declared spent for the family `injury_news_vs_level`, seasons
2020-2025. Part 1/2 cells are a **correlated decomposition** of
`movement_attribution_pop_threshold_injury` /
`movement_attribution_pop_unfiltered_injury` and, through them, of the
`observed_movement_*` family — same archive, same population, a component
split rather than an independent sample. Part 3 cells are a correlated
decomposition of the `injury_signal_on_refresh_card_*` family (same 816-game
intraday archive, same served baseline). **Never pool any of these additively
with each other or with those parents.**

This is a mined battery; **no multiplicity correction is claimed**, matching
every other mined-battery document in this repo.

---

## Results

**Measured** this session, `artifacts/injury_news_vs_level/20260909T234630Z/`
(`per_game_split.parquet`, `pft_per_game.parquet`,
`corrected_baseline_per_game.parquet`, `cells_split.csv`, `cells_pft.csv`,
`cells_post_hoc.csv`, `metadata_split.json`, `metadata_pft.json`,
`metadata_extra.json`, `metadata_postfix.json`, `selection_fix_check.json`,
`record_commands.json`, and the five scripts that produced them alongside a
copy of this predeclaration as frozen at scoring time).

### Decision

**The +17.07 is a NEWS effect, not a level effect. Lane Q's inference is
refuted by measurement on the parent cell's own population.**

The construction defect lane Q named is real and reproduces here — **measured**:
on all 290 games of POP_THRESHOLD the own-week Tuesday-noon baseline is empty,
so `net_injury_score` is arithmetically the final skill-position severity
difference. But when that final difference is split into the part that was
already public before Tuesday noon (the previous week's report) and the part
that changed since, **the flip-value sits entirely in the part that changed**:

| POP_THRESHOLD, official path 2020-2024 | n | delta (pts) | P+ |
|---|---:|---:|---:|
| **NEWS** (`NEWS_net >= 2`) | 110 | **+10.909** | **0.8708** |
| **LEVEL** (`LEVEL_net >= 2`) | 103 | **-0.971** | **0.4634** |
| NEWS only | 95 | **+13.684** | 0.9063 |
| LEVEL only | 88 | **0.000** (44-44) | 0.4999 |

Against the right yardstick — the same 247 official-path games with no
attribution filter at all, **measured at +5.263, P+ 0.7878** — the NEWS half is
roughly **double** the anchor and the LEVEL half sits **below** it.

Read plainly: **following the market away from a team whose injury situation
got materially worse during the week is worth about +11 to +14 accuracy
points; following it away from a team that was already the more injured one is
worth nothing.** The description attached to this project's largest standalone
cell — "acting on injury news that moves the line" — survives. What does not
survive is the implementation: the module's own-week baseline collapses the
two components back into one score, so the served flag fires on a mixture of a
value-carrying news signal and a value-free level signal.

**On Part 3, the PFT-only tilt: no, do not serve it.** Pooled 2023-2025 on the
served refresh card it reads **-1.502 accuracy points, week-blocked 95%
[-5.65, +2.60], `probability_positive` 0.235**, on 282 flips — 5.2 a week.
On the two seasons the official path currently pre-empts it is **-3.008,
P+ 0.143**. Lane Q's one positive arm (PFT on 2025 alone) reproduces here
exactly at **+1.498, P+ 0.716, 88 flips**, and it does not survive contact
with 2023 (-0.376, P+ 0.468) or 2024 (-5.639, P+ 0.086). Serving it would be
taking a 23/77 bet.

And the follow-on the split points at, run post-hoc and reported anyway:
**fixing the baseline does not rescue the tilt on the played card either**
(cross-week prior baseline, 2023-2025: **-1.877, P+ 0.158**). The news
component's value is measured **on the subpopulation where the market has
already moved against the pick**; unconditionally, on the served card, no
version of this flag helps. That is the finding, and the lead it produces is
named below.

**Per the binding taxonomy this closes nothing.** Every interval above crosses
zero; both components have non-zero split-half reliability (**measured**:
LEVEL 0.967, NEWS 0.804 over 160 team-seasons); no positive control bounds any
of it. All 27 cells are recorded `unresolved_below_power`.

### Part 1 — the reproduction

**Measured**, `scripts/movement_attribution.py`'s own `load_population` /
`attach_injury_flag` / `score_cell` imported and run unchanged, at that
document's own spec (`week_blocked_bootstrap`, `block="week"`,
`samples=2000`, `seed=20260820`):

| cell | doc's n | this run | doc's delta | this run | doc's 95% | this run | doc's P+ | this run |
|---|---:|---:|---:|---:|---|---|---:|---:|
| POP_UNFILTERED anchor | 494 | **494** | +5.26 | **+5.263** | [-2.86, +14.12] | **[-2.86, +14.12]** | 0.8805 | 0.8852 |
| POP_THRESHOLD anchor | 290 | **290** | +9.66 | **+9.655** | [-2.68, +21.74] | **[-2.68, +21.74]** | 0.9350 | 0.9385 |
| POP_UNFILTERED INJURY | 185 | **185** | +10.27 | **+10.270** | [-4.24, +24.14] | **[-4.24, +24.14]** | 0.9135 | 0.9193 |
| **POP_THRESHOLD INJURY** | **123** | **123** | **+17.07** | **+17.073** | **[+0.79, +31.67]** | **[+0.79, +31.67]** | 0.9760 | **0.9785** |

**Exact reproduction** of every population size, point estimate and interval.
`probability_positive` reads a few thousandths higher on all four cells, in the
same direction, for a known reason: the 2026-09-08 fix replaced the strict
`draws > 0` with `P(>0) + 0.5·P(==0)`
(`nfl_ats.evidence_conventions.probability_positive_from_draws`), so every
resample that lands exactly on zero now scores half a vote instead of none.
The parent document's numbers were computed 2026-08-20, before that fix. This
is a convention change, not a disagreement, and it moves the cell slightly
*toward* the candidate.

### Part 2 — the split

The decomposition is **exact**. **Measured**: `LEVEL_net + NEWS_net` equals the
parent's own `net_injury_score` on **494 of 494** population games, maximum
absolute gap **0.0**. I expected four exceptions, from the four games measured
above to have a non-empty own-week Tuesday-noon baseline. There are none, and
the reason is worth stating because it makes the defect sharper: **all five of
those pre-Tuesday rows carry a null `report_status`** (2022 wk3 CLE TE, 2022
wk15 SEA TE and WR, 2023 wk17 CLE QB, 2024 wk9 HOU WR — **measured**), which
the severity scale maps to 0.0. So the own-week Tuesday-noon baseline is not
merely sparse on this population; it is **identically zero on all 494 games**,
including the four that have rows. Nothing below is a different flag score — it
is the same score, split.

**POP_THRESHOLD (`|open_move| >= 1.0`), official-report path 2020-2024, n=247.**
Candidate = the market's side, baseline = the production pick, graded at
`margin_vs_open`, 20,000 draws, seed 20260821.

| cut | n | delta (pts) | 95% week-blocked | P+ |
|---|---:|---:|---|---:|
| *anchor: all 247 official-path games* | 247 | *+5.263* | *[-7.94, +18.33]* | *0.7878* |
| parent INJURY (official half of the +17.07 cell) | 99 | +11.111 | [-7.37, +28.30] | 0.8838 |
| **NEWS** | **110** | **+10.909** | [-8.41, +29.29] | **0.8708** |
| **LEVEL** | **103** | **-0.971** | [-20.00, +19.30] | **0.4634** |
| NEWS only | 95 | **+13.684** | [-6.82, +33.33] | 0.9063 |
| LEVEL only | 88 | **0.000** | [-20.93, +21.84] | 0.4999 |
| BOTH | 15 | -6.667 | [-60.00, +46.67] | 0.3928 |
| SPLIT_NEITHER | **0** | — | — | — |

**POP_UNFILTERED, official-report path 2020-2024, n=412.**

| cut | n | delta (pts) | 95% week-blocked | P+ |
|---|---:|---:|---|---:|
| *anchor: all 412 official-path games* | 412 | *+2.427* | *[-6.80, +11.80]* | *0.6954* |
| parent INJURY (official) | 151 | +4.636 | [-11.80, +20.78] | 0.7097 |
| **NEWS** | **177** | **+8.475** | [-5.56, +22.77] | **0.8794** |
| **LEVEL** | **166** | **-3.614** | [-19.57, +13.04] | **0.3359** |
| NEWS only | 153 | +9.804 | [-5.45, +25.16] | 0.8947 |
| LEVEL only | 142 | -4.225 | [-21.68, +13.48] | 0.3146 |
| BOTH | 24 | 0.000 | [-41.67, +41.67] | 0.4907 |
| SPLIT_NEITHER | **0** | — | — | — |

Disclosed secondary, whole population with season 2025's news-by-construction
PFT flags folded into NEWS: POP_THRESHOLD **134 games, +16.418,
[0.00, +32.79], P+ 0.9729**; POP_UNFILTERED **211 games, +12.796,
[0.00, +25.36], P+ 0.9759**. The NEWS-only construction on the whole window
recovers essentially the parent's headline (+16.4 vs +17.07) on more games,
while the LEVEL half is unchanged from the official-path rows above (-0.971 /
-3.614), since 2025 contributes no level component.

**How the halves relate to each other and to the parent** (**measured**,
`metadata_extra.json`):

| quantity | POP_THRESHOLD | POP_UNFILTERED |
|---|---:|---:|
| parent INJURY fires | 99 | 151 |
| NEWS fires | 110 | 177 |
| LEVEL fires | 103 | 166 |
| NEWS **and** LEVEL | 15 | 24 |
| parent fires but neither half does | **0** | **0** |
| NEWS fires where the parent does not | 56 | 94 |
| LEVEL fires where the parent does not | 43 | 74 |
| correlation `LEVEL_net` vs `NEWS_net` | **-0.798** | **-0.784** |

**Stated as a limitation, not buried:** the two components are strongly
anti-correlated *by construction* (`NEWS_net = FINAL_net − LEVEL_net`, and each
component has a larger spread — sd 8.65 and 8.16 — than the total they sum
to). So the NEWS and LEVEL flags are near-disjoint re-cuts that each fire on
*more* games than the parent flag, not two halves of the parent's 99. The
score decomposition is exact; the flag partition is not a partition. What the
table supports is "the flip-value tracks the news component and not the level
component", which is the question asked. It does not support "56 of the
parent's flags were really level flags".

**Split-half reliability**, odd/even weeks within team-season, over **all**
2,651 REG team-weeks 2020-2024 and 160 team-seasons (**measured**,
`metadata_split.json`): prior severity (the LEVEL component) **r = 0.967**
(Spearman-Brown 0.983); the week-on-week news delta (the NEWS component)
**r = 0.804** (0.891); the final severity level 0.784. Both components are
stable team-season traits, so neither is refuted on reliability grounds. The
level is the *more* reliable of the two and carries none of the value — which
is the shape "already priced" always has.

### Why the value sits in the change, stated plainly

**Inferred, not measured:** I think this is the already-priced ceiling
(`docs/decision_rule.md`) doing exactly what it is supposed to. A standing
injury list is a season-long team-quality fact the Tuesday line has fully
absorbed, so a market move away from an already-hurt team carries no
information the opener did not already hold — hence LEVEL at 0.000 on 88
games. A *change* since last Friday's report is new information, and the
market prices it before this project's Tuesday-locked card does — hence NEWS
at +10.9 to +13.7. The +17.07 cell was measuring the right thing for the
right reason; it was just computing it with a baseline that does not exist.

### Part 3 — the PFT construction as the tilt's only path

Harness reproduction first (**measured**, `metadata_pft.json`): 816 games,
**799** non-push grades, **54** weeks, **260** composition flips vs the raw
pick, **329** follow-rule firings, **145** follow-rule flips vs the Tuesday
card, served card **445/799**. All seven figures match
`docs/injury_signal_on_refresh_card.md` exactly, so the S baseline is the same
object lane Q measured.

Candidate = S + the module's own `net_pft_score >= 1` path forced on every
season; baseline = S; 20,000 draws, seed 20260821.

| cell | n | flips | delta (pts) | 95% week-blocked | P+ |
|---|---:|---:|---:|---|---:|
| `..._pftonly_tilt_2023` | 266 | 91 | -0.376 | [-8.12, +7.01] | 0.4682 |
| `..._pftonly_tilt_2024` | 266 | 103 | **-5.639** | [-13.69, +2.55] | **0.0860** |
| `..._pftonly_tilt_2025` | 267 | 88 | **+1.498** | [-3.69, +6.69] | **0.7161** |
| `..._pftonly_tilt_2023_2024` | 532 | 194 | **-3.008** | [-8.52, +2.46] | **0.1431** |
| **`..._pftonly_tilt_2023_2025`** | **799** | **282** | **-1.502** | **[-5.65, +2.60]** | **0.2347** |
| `..._pftonly_tilt_where_follow_silent_2023_2025` | 474 | — | -2.743 | [-8.23, +2.68] | 0.1601 |

2025 reproduces lane Q's `..._tilt_pft2025_2025` cell exactly (+1.498,
P+ 0.716, 88 flips), which is the check that the forced-PFT path is the same
construction lane Q ran. The two seasons that had been hidden behind the
official path's pre-emption do not repeat it.

**Why the news mechanism can be right and this arm still negative:** the PFT
path is a *team-level headline count*, not a player-level designation change.
Its split-half reliability is **0.071** over 96 team-seasons (**measured**) —
against 0.804 for the player-level news delta. Counting how many times a team
was mentioned in an injury headline is a very noisy estimator of the quantity
Part 2 shows carries the value.

Collision and visibility (**measured**, `metadata_pft.json`):

| quantity | value |
|---|---:|
| PFT-tilt flips vs the served card | 282 (5.2 / week) |
| flips 2023 / 2024 / 2025 | 91 / 103 / 88 |
| flip where the follow rule also fired | 112 |
| — of which the flip reverses an actual follow-rule flip | 50 |
| flip where the follow rule was silent | 170 |
| firings backed by a headline inside (Tue noon, deadline] | **282 of 282** |
| firings depending on news after `min(kickoff, Sun 16:00 ET)` | **0 of 282** |

Nothing here is unplayable — every firing is decidable inside the pool's own
editing window, confirmed per-game rather than asserted — and 40% of the
firings would be undoing the promoted follow rule's own pick.

### Post-hoc: does fixing the baseline rescue the tilt on the played card?

**This arm was NOT predeclared.** It was added after the Part 2 split reported
where the flip-value sits, and is labelled post-hoc in the registry too. The
construction is the module's official path with one change: the Tuesday
baseline is the cross-week prior (that player's latest designation filed at or
before this week's Tuesday noon, searched across the season) instead of the
empty own-week one — i.e. the correction Part 2's decomposition implies.

| cell | n | flips | delta (pts) | 95% week-blocked | P+ |
|---|---:|---:|---:|---|---:|
| corrected baseline, 2023 | 266 | 119 | -6.015 | [-14.87, +3.33] | 0.0989 |
| corrected baseline, 2024 | 266 | 115 | +0.376 | [-5.68, +6.32] | 0.5433 |
| corrected baseline, 2023-2024 | 532 | 234 | -2.820 | [-8.27, +2.81] | 0.1607 |
| **corrected baseline, 2023-2025** | **799** | **234** | **-1.877** | [-5.62, +1.77] | **0.1577** |
| (uncorrected level arm, 2023-2025) | 799 | 198 | -1.627 | [-4.71, +1.25] | 0.1390 |

The last row reproduces lane Q's primary cell exactly (-1.627, P+ 0.139, 198
flips), as do its 2023 (-3.759) and 2024 (-1.128) seasons — a third
independent reproduction of that lane. Flag reliability of the corrected
firing is 0.184 over 64 team-seasons (**measured**). 2025 is a no-op for this
arm (its rows carry no timestamps), so the pooled cell is the 2023-2024 result
diluted by a third of the window.

**So the answer is no, and it is the informative kind of no.** The news
component is worth +11 to +14 points *conditional on the market having already
moved at least a point against the pick*, and roughly nothing when applied to
every game on the card. That is a statement about where the signal lives, not
about whether it exists.

### The lead this produces (named, not run)

**Inferred, not measured:** I think the right shape for this signal is a
**gate on the follow rule, not a standalone flip.** The served card already
follows the market when `|equal_net_move| >= 0.5`; Part 2 measures that
following is worth far more when a player-level injury deterioration on the
abandoned side accompanies the move, and roughly the anchor rate when it does
not. The study that would settle it: on the 816-game archive, compare the
served card against a variant that follows only when the corrected
`news_net >= 2` on the side the market is abandoning, and against one that
follows on a lower movement threshold when the news confirms. That needs its
own predeclaration — sign conventions flip on the 145 games where the follow
rule already moved the pick, and getting that wrong would produce a number
that looks like this one and means the opposite. This lane did not run it.

### Part 4 — the path-selection fix, implemented

Both predeclared conditions are met. (a) **Measured**
(`selection_fix_check.json`): in snapshot `20260909T223500Z`, season 2025 has
5,783 official rows and **0** with a readable `date_modified` (2026: 29 rows,
0 readable), so `_severity_asof`'s `date_modified <= cutoff` filter admits
nothing and the official path is a guaranteed `net_score = 0.0`,
`fires = False` no-op that nonetheless reports `source = "official"`. (b) The
PFT alternative on that season is +1.498, P+ 0.716 — not a resolved wrong
sign.

Implemented in this worktree, selection only
(`src/nfl_ats/injury_signal_refresh_tilt.py`): a new
`_season_has_readable_official_rows` requires the target season's rows to
carry at least one usable observation timestamp — a non-null `date_modified`
that is not flagged `observed_at_is_proxy` where the newer schema provides
that column — before `injury_signal_for_game` chooses the official path;
otherwise it falls through to the PFT path the module already implements, and
to `source = "none"` when neither is readable. No threshold, no scoring, no
new rule, and nothing served, no ledger, manifest, forecast or board touched.

**Measured** effect on the module's own reading, over lane Q's 816 games:

| season | selection before | selection after | firings after |
|---|---|---|---:|
| 2023 | official | official (unchanged) | 100 |
| 2024 | official | official (unchanged) | 98 |
| 2025 | official (0 firings, silent no-op) | **pft_fallback** | **88** |

Seasons 2009-2024 are unaffected; 2025 and 2026 flip to the fallback. The
resulting challenger arm on the served card measures **-1.126 accuracy points,
[-4.646, +2.261], P+ 0.2579**, 286 flips — which is an exact match to lane Q's
hand-built `..._tilt_pft2025_2023_2025` variant, i.e. the fix makes the module
compute by itself the arm lane Q had to construct outside it.

**This is a reporting-honesty fix, not a serving recommendation.** On the
evidence above the resulting arm should not be played (P+ 0.258); its value is
that the challenger ledger stops recording "official, no signal" for a season
where the official path cannot read anything.

**One disclosed edge:** season 2009 has 17 readable rows out of 4,596 and still
selects the official path under the minimal "at least one readable row" test.
A stricter bar (a majority, or a per-week test) would be an unpredeclared
threshold, so it was not added; 2009 is outside every window this project
scores.

Gates run on the worktree after the change (**measured**):
`ruff format --check src/nfl_ats/injury_signal_refresh_tilt.py` — 1 file
already formatted; `ruff check` — all checks passed; `mypy` — success, no
issues; `pytest tests/test_injury_signal_refresh_tilt.py` — 24 passed. The
full suite was not run (test moratorium; no test was added or changed).

### Caveats (label how you know it)

- **Correlated decomposition.** Part 1/2 cells share the parent's archive and
  population with `movement_attribution_pop_threshold_injury` and the
  `observed_movement_*` family, and `NEWS_net`/`LEVEL_net` are correlated
  -0.78 with each other. Part 3 and the post-hoc cells share lane Q's
  816-game intraday archive and served baseline with the
  `injury_signal_on_refresh_card_*` family. Never pooled additively with
  those or with each other.
- **Mined battery**, 27 recorded cells, no multiplicity correction claimed.
- **The flag partition is not a partition** (see the crosstab note above): the
  NEWS and LEVEL flags each fire on more games than the parent flag and
  overlap on only 15 of 198 at the threshold cut.
- **`prior_sev` is a within-season lookup.** A player's first appearance of
  the season has no prior, so a Week 1 designation scores entirely as NEWS by
  construction. **Measured**: 228 of 988 population team-games have no
  cross-week prior at all, concentrated in Week 1; that flatters the NEWS
  half rather than the LEVEL half.
- **The parent's grading line is inherited, including its `final = kickoff`
  cutoff.** `movement_attribution` is a backward-looking attribution, not a
  playable rule; the NEWS component reads the report through kickoff. Part 3
  and the post-hoc arm, which ARE playable constructions, use each game's own
  `min(kickoff, Sunday 16:00 ET)` deadline instead.
- **Skill positions only** (QB/RB/WR/TE), inherited from the parent:
  offensive-line and defensive-front injuries are invisible to both halves. A
  likely undercount of real injury information, not an overcount.
- **The PFT path is team-nickname headline matching**, not player-level
  matching; a headline naming both teams is not disambiguated. Its measured
  split-half reliability is 0.071.
- **The corrected-baseline arm is post-hoc**, suggested by the Part 2 result
  rather than predeclared, and is labelled so in its registry notes.
- **The opener evaluator's inherited approximation applies to Part 3**: only
  `spread_line` is swapped to the opener; other features are close-era
  (`docs/opener_evaluation.md`).
- Nothing in production was changed. No ledger, manifest, forecast or board
  was touched, and nothing was committed or pushed.

### Registry entries recorded

27 entries, all `effect_units=accuracy_points`,
`classification=unresolved_below_power`, `closing_ground=null`, league `nfl`,
family `injury_news_vs_level` (**measured**: registry total 4,591 after
recording; 27 entries present under the `injury_news_vs_level` prefix, none
carrying a closing ground). Exact argument vectors in
`artifacts/injury_news_vs_level/20260909T234630Z/record_commands.json`.
