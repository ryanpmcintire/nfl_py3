# The whole served chain, measured as one object (lane AJ)

Lane AJ, 2026-09-09. Everything above the `## Results` line was written and
frozen **before** any accuracy, delta, interval or `probability_positive` in the
results was computed. The counts quoted inside the predeclaration were measured
first, deliberately, from population membership, timestamps and market movement
only; no outcome column (`margin_vs_open`, `correct_*`) was read until the design
was fixed.

## Binding closing-grounds taxonomy (AGENTS.md), verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only a RESOLVED wrong sign (the whole
interval on the wrong side of zero), zero split-half reliability, or a positive
control proven able to detect an effect that size ever closes a line of work.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary
"contains zero". Within-week game correlation is ZERO by owner mandate; the
bootstrap is week-blocked and no ICC term is estimated or padded. The pool is
FORCED PICKS, so the decision is expected value: a promotion bar governs what
this document may claim, never which card is played.

## The gap this closes

Today the played card is a chain of six decisions, each promoted in its own lane
against its own baseline:

1. the Tuesday nine-member composition
   (`src/nfl_ats/four_overlay_composition.py`, `POLICY_ID =
   overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`);
2. the late-week leader-median follow
   (`LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY`), whose threshold a concurrent lane
   is moving from 0.5 to 1.0 (`docs/follow_threshold_live_card.md`) with an
   injury-news veto on a contradicted move (`docs/follow_news_gate.md`, arm F3p);
3. the 1.0-point consensus movement rule (`MOVEMENT_POLICY_THRESHOLD`);
4. the rookie-crew refit (`ROOKIE_CREW_POLICY`,
   `docs/rookie_crew_reconciliation.md`);
5. the heavy-handle follow on weekend passes (`HANDLE_FOLLOW_POLICY`,
   `docs/handle_follow_on_card.md`);
6. the small-spread Best Pick nominator
   (`docs/best_pick_bucket_confidence.md`, arm B2).

Every one of those was measured against a DIFFERENT baseline in a DIFFERENT
lane: step 2's promotion evidence is against the raw model, step 4's against the
nine-member card, step 5's against the nine-member card on a 260-game split
population, step 6 against the incumbent nominator on 107 weeks. **Nobody has
measured the whole chain as one object**, and the site's headline still reports
the Tuesday-card opener number alone. This lane measures the chain, in the
precedence the code actually serves, on the only window where every input
exists.

## The served precedence, read from the code

Read `src/nfl_ats/pick_refresh.py:1379-1414`, verbatim in structure:

```
if late_week_fires:      policy = late-week leader-median follow
elif consensus_fires:    policy = movement_ge_1.0
else:                    policy = rookie crew if it differs from the card, else model only
then: if policy == model_only and handle money >= 70 on the other side: policy = handle follow
```

Two consequences are load-bearing and are stated before any number is computed:

- the handle rule can only act where **all three** rules above it are silent
  (`policy == MOVEMENT_POLICY_MODEL_ONLY` is the literal guard at line 1408), so
  it never applies after the rookie-crew step either;
- a **vetoed** late-week fire counts as "the follow rule fired" for precedence
  and does NOT fall through to the consensus or handle rules. That is
  `docs/follow_news_gate.md`'s own promotion note ("A vetoed game must be treated
  as 'the follow rule fired' for precedence... because that is how it was
  measured here") and it is how this lane replays it.

## Population (frozen before scoring)

- **Window.** 2023-2025, the only seasons with the hourly odds archive, injury
  news and public splits together. **Measured**: 816 regular-season archive
  games (272 per season), 54 `(season, week)` blocks, **799** opener-scored
  (17 opener pushes dropped -- they are not scoreable either way).
- **Grading.** `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`,
  active model `c657058903f3232b`, `gaussian_median`, graded at `margin_vs_open`
  (the frozen Tuesday opener). Forced picks, game-weighted.
- **Market.** The `intraday_hourly` historical-backfill archive under
  `data/market/raw`, replayed through the frozen served function
  `nfl_ats.sharp_book_movement_features.late_week_follow_frame` with each game's
  cutoff at its own pick deadline `min(kickoff, that week's Sunday 16:00 ET)`.
- **Injury news.** The official injury report and the ProFootballTalk headline
  archive, through `docs/follow_news_gate.md`'s frozen reader (`news_toward_market`
  and its `pft_toward_market` fallback), reused row for row from
  `artifacts/follow_news_gate/20260910T002019Z/frame.parquet` rather than
  reimplemented.
- **Public splits.** `artifacts/handle_follow_on_card/20260909T232503Z/population.parquet`,
  lane S's frozen 260-game Wayback split population. **Measured**: 133 of those
  games carry a handle reading and all 133 are inside 2023-2025.
- **Officials.** The archive-extended officials table, head referee, with at most
  `ROOKIE_PRIOR_EXPERIENCE_MAX` prior seasons, season floor
  `ROOKIE_CREW_SEASON_FLOOR = 2010`, flag signed by the underdog side of the
  frozen Tuesday line -- the construction
  `nfl_ats.pick_refresh._rookie_crew_lookup` uses, and the refit archive
  `artifacts/rookie_crew_reconciliation/20260909T190000Z/opener_evaluation/rookie_crew_reconciled/per_game.parquet`.

### Stated deviations, up front, before any result

1. **Capture kind.** The served path loads `capture_kind="live"` and the market
   store holds no live capture before 2026-08-17, so 2023-2025 is reachable only
   through the historical backfill. Same substitution
   `docs/served_card_harness.md`, `docs/follow_vs_tilts.md` and
   `docs/follow_threshold_live_card.md` document. This is evidence about the
   chain's SHAPE, not a re-grade of any promotion taken on another surface.
2. **The late-week threshold is 1.0, not the 0.5 in the code today.** A
   concurrent lane is moving it; per instruction this lane treats **1.0 plus the
   injury-news veto** as the served rule and reports the 0.5 arm alongside so the
   difference is legible rather than assumed.
3. **The handle step can only reach 133 of 816 games**, because the Wayback split
   backfill covers only that much of the window. Every other game contributes an
   exact zero to that step's paired delta. Prospectively the live Saturday/Sunday
   captures give a reading on every game, so this step's measured reach is a
   floor, not its served reach.
4. **The rookie-crew step is replayed from the archive refit**, not from a
   refresh-time refit gated on the Wednesday crew snapshot's own arrival time.
   Crew-publication timing is not replayable on 2023-2025; the flag is taken from
   the officials archive, which is what `docs/served_card_harness.md` also did.
5. **2025's official injury rows carry no readable timestamp**
   (`observed_at_basis = week_proxy`, null `date_modified`), so the veto uses the
   PFT headline fallback on that season -- arm F3p, not F3. Disclosed by
   `docs/follow_news_gate.md` before its own scoring.
6. **2020-2022 has no intraday archive, no split backfill and no readable news**,
   so on those seasons the chain degrades to the Tuesday card plus the rookie-crew
   step (the only input that exists back there). The 2020-2025 headline is
   therefore "the refresh chain applied where its inputs exist", and it is
   labelled that way rather than presented as six seasons of the full chain.

## The chain, as cumulative arms (frozen)

Every arm is a pick side per game, graded at `margin_vs_open`.

| Arm | Definition |
| --- | --- |
| **C0 raw** | The raw model's opener pick, no card at all. |
| **C1 Tuesday** | C0 complemented once on the nine-member union. **This is what the site's headline reports today.** |
| **C2 +follow** | C1, then the late-week leader-median follow at 1.0 with the injury-news veto. |
| **C3 +consensus** | C2, then the 1.0-point consensus movement rule where C2's rule was silent (a veto is not silent). |
| **C4 +crew** | C3, then the rookie-crew refit's side where both market rules were silent and the refit differs. |
| **C5 served refresh card** | C4, then the heavy-handle follow where all three rules above were silent. **This is the full served chain.** |

Each step's **marginal** is `C_k - C_{k-1}`, paired on the same games. The
**chain total** is reported twice: `C5 - C1` (what the late-week rules are worth
on top of the Tuesday card, which is the number the headline needs) and
`C5 - C0` (what the whole card is worth over the bare model).

Two reference arms are computed for context and are not part of the chain:
**C2@0.5** (the follow at the threshold in the code today, with the same veto)
and **C2-noveto** (the follow at 1.0 with no veto).

**Positive control.** Perfect foresight on exactly the games the full chain
changes against C1 (`PC_chain`), and perfect foresight on every game any step
can reach (`PC_reach`). If a step's effect sits well inside the control's own
interval, the instrument is not shown blind and nothing here is
`bounded_by_control`.

## Best Pick (a ranking, never a side)

Separately, the served small-spread nominator (arm B2 of
`docs/best_pick_bucket_confidence.md`: v2's ranking restricted to games whose
frozen Tuesday spread is 6.5 or less) is scored on the same weeks, against the
incumbent bucket-blind nominator B0 and against the perfect-foresight control.
This changes which single game carries the star, never any pick side, so it is
reported as its own table and never folded into the chain arithmetic.

## Statistics, declared before the numbers

- Paired deltas in accuracy points, game-weighted, forced picks, opener-graded;
  opener pushes dropped.
- `nfl_ats.overlay_composition.blocked_bootstrap_matrix`, paired, **20,000
  samples, seed 20260821**, **week-blocked** (blocks are `(season, week)`) and
  **season-blocked**. Within-week correlation is ZERO; no ICC is estimated. The
  season-blocked column on a three-season window has three blocks, so its
  `probability_positive` saturates and carries little information; it is reported
  for completeness, never as a second opinion.
- Windows: 2023-2025 pooled, then 2023, 2024 and 2025 separately; plus 2020-2025
  and 2020-2022 for the wider view in section 4.
- Reported for every step: picks changed, picks changed **per week**, and
  `probability_positive`. The binary "contains zero" is never used as a verdict.
- **Collision matrix.** For every ordered pair of steps, the number of games
  where both would fire if each were applied alone, and the number where the
  later step is actually SUPPRESSED by the earlier one under the served
  precedence. This is the object that makes "each piece was measured in its own
  lane" quantitative.

## Reproduction checks, fixed now

1. The rebuilt Tuesday card must score **56.070%** on the 799 scored games, the
   figure `docs/served_card_harness.md` and `docs/follow_news_gate.md` both
   publish. A mismatch is reported and the lane stops rather than reading a
   different card.
2. The rebuilt leader-median arm at 0.5 must reproduce **523 fires** and **243
   flips** against the Tuesday card (`docs/follow_news_gate.md`'s reproduction
   table), and **228 fires** at 1.0 (`docs/follow_threshold_live_card.md`).
3. The nine-member composition must flip **260** of the 816 games.

## Registry

Every chain cell is recorded through `nfl-ats weak-signals record` under names
`served_refresh_card_<step|total>_<window>`, units `accuracy_points`, league
`nfl`, family `served_refresh_card`, one invocation at a time with
`NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry`, argv lists saved to
`record_commands.json`. The family is declared here, before the signs were seen.
Its members share one window, one grading and one market replay, so they are
**correlated readings of one question** rather than independent votes, and they
are a correlated decomposition of `sharp_weighted_follow_*`,
`follow_news_gate_*`, `follow_threshold_live_card_*`, `served_card_harness_*`
and `handle_follow_on_card_*`. **Never pool any of these additively with each
other or with those parents.** This is a mined battery; no multiplicity
correction is claimed.

## What this lane wires, and what it does not

The headline number on the site is wired: the board gains a second, pool-player
sentence -- "with the line-move and injury rules applied through the week: X%"
-- read from THIS lane's artifact, keyed to the active model id and to the served
policy ids, failing closed when the artifact's policy ids do not match what the
code serves. No served pick rule, ledger, manifest, forecast or published site
page is touched by this lane.

---

## Results

**Measured** this session, `artifacts/served_refresh_card/20260910T011328Z/`
(`frame.parquet`, `chain.parquet`, `archive_frame.parquet`, `coverage.json`,
`results.json`, `extras.json`, `headline.json`, `cells.csv`,
`record_commands.json`, the five scripts that produced them, and a copy of this
predeclaration frozen at scoring time). The `predeclaration_sha256`
`a8c6783a06df98f0a9bbc2b796a994bea7891fd9940346e30e08e71a7912a159` is the hash of
everything above this line, taken before any script ran.

### The decision, first

**The chain as a whole is not worth what its parts claim.** Measured on the 799
opener-graded games of 2023-2025, the Tuesday nine-member card scores
**56.070%** and the full served chain scores **55.945%** -- a paired
**-0.125 accuracy points**, week-blocked 95% [-2.726, +2.612],
`probability_positive` **0.4596**, on 157 changed picks. The late-week follow at
1.0 with the injury veto is worth **+1.502** points (P+ **0.9429**) and the
**1.0-point consensus movement rule immediately gives all of it back and more:
-1.627 points, P+ 0.0413, on 73 changed picks it goes 30-43.**

**On a forced card the actionable read is the pair, not the total.** Dropping
the consensus rule and keeping everything else scores **57.947%** --
**+2.003 accuracy points over the served chain, week-blocked 95%
[+0.126, +3.865], `probability_positive` 0.9816** (season-blocked [+0.752,
+4.135]), and **+1.877 over the Tuesday card alone**, P+ 0.9474. That is the
single largest lever this lane found, and it is a lever that already exists in
the code: it is one branch of `plan_refresh`'s precedence.

### Reproduction: all three checks pass

**Measured**, `coverage.json`:

| quantity | required | this run |
|---|---:|---:|
| archive games 2023-2025 | 816 | **816** |
| opener-graded (non-push) | 799 | **799** |
| `(season, week)` blocks | 54 | **54** |
| nine-member composition flips | 260 | **260** |
| Tuesday card accuracy | 56.070% | **56.070%** |
| leader-median fires at 0.5 | 523 | **523** |
| leader-median flips at 0.5 vs the card | 243 | **243** |
| leader-median fires at 1.0 | 228 | **228** |
| per-member flip counts | 107/155/24/73/57/111/6/17/15 | **identical** |

New coverage this lane measured: **342** of 816 games move at least a full point
in whole-market consensus between the frozen Tuesday opener and their own
deadline (mean signed move **+0.096**, median **0.000**, mean absolute move
**1.002** -- the two line sources are not systematically offset); **64** games
are worked by a first- or second-season head referee and the rookie refit
differs from the card on **3** of them; **133** games carry a public-money
reading and the 70% handle rule points away from the card on **37** of them,
reproducing lane S's own 37 flips.

### Accuracy at the Tuesday opener, cumulative chain, 799 games

| Arm | 2023-2025 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| **C0** raw model, no card | 55.069% | 56.391% | 53.008% | 55.805% |
| **C1** Tuesday nine-member card | **56.070%** | 59.398% | 54.135% | 54.682% |
| **C2** + late-week follow at 1.0 with the injury veto | **57.572%** | 59.023% | 55.639% | 58.052% |
| **C3** + the 1.0-point consensus rule | 55.945% | 55.639% | 55.263% | 56.929% |
| **C4** + the rookie-crew refit | 55.945% | 55.263% | 55.639% | 56.929% |
| **C5** the full served refresh card | **55.945%** | 54.887% | 56.015% | 56.929% |
| C5 without the consensus rule | **57.947%** | 59.023% | 56.767% | 58.052% |
| reference: follow at 0.5 with the veto | 56.446% | 57.895% | 57.519% | 53.933% |
| reference: follow at 1.0 with no veto | 57.322% | 58.647% | 56.015% | 57.303% |
| control: perfect foresight on the chain's own changes | 65.832% | 69.549% | 65.414% | 62.547% |
| control: perfect foresight everywhere reachable | 100% | | | |

### Each step's marginal, paired, 2023-2025

Week-blocked interval quoted; 20,000 samples, seed 20260821. The season-blocked
column has three blocks, so its `probability_positive` saturates and carries
almost no information -- reported for completeness, never as a second opinion.

| Step | vs | Changed | Per week | Delta (pts) | 95% week | P+ week | P+ season |
|---|---|---:|---:|---:|---|---:|---:|
| **1** Tuesday nine-member card | raw model | 254 | 4.81 | **+1.001** | [-3.186, +5.051] | 0.6862 | 0.853 |
| **2** late-week follow, 1.0 + veto | Tuesday card | 64 | 1.19 | **+1.502** | [-0.372, +3.457] | **0.9429** | 0.965 |
| **3** consensus movement, 1.0 | after step 2 | 73 | 1.37 | **-1.627** | [-3.409, +0.250] | **0.0413** | 0.000 |
| **4** rookie-crew refit | after step 3 | 2 | 0.04 | **0.000** | [-0.373, +0.376] | 0.4993 | 0.504 |
| **5** heavy-handle follow | after step 4 | 18 | 0.33 | **0.000** | [-1.023, +1.245] | 0.4773 | 0.504 |
| **chain total** | Tuesday card | 157 | 2.93 | **-0.125** | [-2.726, +2.612] | **0.4596** | 0.357 |
| **whole card** | raw model | 295 | 5.59 | +0.876 | [-3.690, +5.443] | 0.6510 | 0.798 |
| reference: follow at 0.5 + veto | Tuesday card | 161 | 2.98 | +0.375 | [-2.709, +3.544] | 0.5945 | 0.704 |
| reference: follow at 1.0, no veto | Tuesday card | 100 | 1.85 | +1.252 | [-1.122, +3.713] | 0.8462 | 0.965 |
| **chain minus the consensus rule** | served chain | 70 | 1.30 | **+2.003** | **[+0.126, +3.865]** | **0.9816** | 1.000 |
| chain minus the consensus rule | Tuesday card | 93 | 1.72 | +1.877 | [-0.373, +4.261] | 0.9474 | 0.964 |
| control: perfect foresight on the chain's changes | Tuesday card | 78 | 1.44 | +9.762 | [+7.990, +11.690] | 1.0000 | 1.000 |
| control: perfect foresight, all reachable | Tuesday card | 351 | 6.50 | +43.930 | [+40.511, +47.475] | 1.0000 | 1.000 |

By season, the chain total against the Tuesday card is **-4.511 (2023),
+1.880 (2024), +2.247 (2025)**; the late-week follow step is
**-0.376 / +1.504 / +3.371**; the consensus step is **-3.383 / -0.376 / -1.124**,
the same sign in all three seasons with magnitudes differing, which is a
magnitude statement per era and never an absence.

### The mechanism, made visible

Each step's own record on only the picks it changes (**measured**,
`results.json` -> `diagnostics`):

| Step | Picks it changes | Its record on them | The card's record on the same games |
|---|---:|---:|---:|
| late-week follow, 1.0 + veto | 64 | **38-26 (59.4%)** | 26-38 |
| 1.0-point consensus rule | 73 | **30-43 (41.1%)** | 43-30 |
| heavy-handle follow | 18 | 9-9 (50.0%) | 9-9 |

**The named mechanism, one sentence: the two market rules read the same money,
and the one that reads it through three leading books at a full point wins its
picks 59-41 while the one that reads the whole market's drift from Tuesday loses
its picks 41-59.** That is not an accuracy dip located at a threshold; it is a
difference in what the two aggregations measure, and it is the same distinction
`docs/follow_threshold_live_card.md` named at the threshold level.

### The collision matrix -- what the precedence actually spends

Fires and pick-changes are computed for each rule ALONE against the Tuesday
card, so they are comparable; "suppressed" is what the served precedence
actually removes.

| Rule | Fires alone | Would change the card pick |
|---|---:|---:|
| late-week follow (1.0 + veto) | 228 | 64 |
| consensus movement (1.0) | 342 | 150 |
| rookie-crew refit | 3 | 3 |
| heavy-handle follow | 37 | 37 |

| Earlier rule | Later rule | Both fire | Both would change the pick | Disagree with each other | Later suppressed | ...when it would have changed the pick |
|---|---|---:|---:|---:|---:|---:|
| late-week follow | consensus | **179** | 42 | **21** | 179 | **76** |
| late-week follow | rookie crew | 0 | 0 | 0 | 0 | 0 |
| late-week follow | handle | 11 | 3 | 6 | 11 | 11 |
| consensus | rookie crew | 1 | 1 | 0 | 1 | 1 |
| consensus | handle | 15 | 8 | 7 | 8 | 8 |
| rookie crew | handle | 0 | 0 | 0 | 0 | 0 |

**This is the answer to "each piece was measured in its own lane", in numbers.**
The two market rules collide on **179 of their fires** and point at opposite
sides on **21** of them; the served precedence hands all 179 to the follow rule
and deletes **76** consensus pick-changes, **29** of which are deleted by a
*vetoed* follow (a game where the follow rule fired, the injury report
contradicted it, and the pick stayed on Tuesday's side while the consensus rule
was blocked anyway). The handle rule is suppressed on 19 of its 37 fires by the
two market rules above it, which is exactly the double-counting
`docs/handle_follow_on_card.md` predicted and asked for.

Policy counts across the 816 games: the follow rule governs **153**, is
**vetoed on 75**, the consensus rule governs **163**, the rookie-crew rule **2**,
the handle rule **18**, and **405** games are left on the model's own side.

### Best Pick, on the same weeks (a ranking, never a side)

| Window | Weeks paired | Incumbent | Small-spread nominator | Weeks the nominee differs | Delta (pts) | 95% week | P+ |
|---|---:|---:|---:|---:|---:|---|---:|
| 2020-2025 | 102 | 56.86% (58) | **58.82% (60)** | 26 | **+1.96** | [-5.88, +9.80] | **0.6885** |
| **2023-2025** | 50 | **64.00% (32)** | 60.00% (30) | 14 | **-4.00** | [-16.00, +6.00] | **0.2393** |
| control 2020-2025 | 103 | 56.31% | 98.06% | 43 | +41.75 | [+32.04, +51.46] | 1.0000 |
| control 2023-2025 | 51 | 62.75% | 100% | 19 | +37.25 | [+23.53, +50.98] | 1.0000 |

Reported plainly because it was not asked as a leading question: **the
small-spread nominator's whole six-season gain sits in 2020-2022. On the
2023-2025 window this lane measures the chain over, it is 30-of-50 against the
incumbent's 32-of-50, -4.00 points, `probability_positive` 0.2393.** That is a
magnitude statement per era on a fourteen-week difference, not an absence and
not a closure; the instrument is a ~100-observation one and its own control
resolves at +37 to +42 points, so it can see an oracle and cannot see four
points. Nothing here closes and nothing is proposed.

### The 2020-2025 view, where the refresh chain's inputs only partly exist

2020-2022 has no hourly odds archive, no readable injury timestamps and no
public-split backfill, so on those 704 scored games only the Tuesday card and
the rookie-crew step have replayable inputs. **Measured**:

| Window | Games | Tuesday card | With the chain where inputs exist | Delta (pts) | 95% week | P+ |
|---|---:|---:|---:|---:|---|---:|
| **2020-2025** | 1,503 | **56.886%** | **56.953%** | **+0.067** | [-1.338, +1.500] | 0.5402 |
| 2020-2022 | 704 | 57.813% | 58.097% | +0.284 | [+0.000, +0.711] | 0.9356 |
| 2020-2025, Tuesday card vs raw model | 1,503 | 54.558% | 56.886% | +2.329 | [-0.464, +5.093] | 0.9498 |
| 2020-2022, Tuesday card vs raw model | 704 | 53.977% | 57.813% | +3.835 | [+0.143, +7.482] | 0.9803 |

**And the same six seasons under the chain the code serves TODAY** (the follow
at 0.5 with no injury veto, which is what `LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY`
still names): **56.221%** against the Tuesday card's 56.886%, **-0.665 points,
[-2.771, +1.471], `probability_positive` 0.2753**, 280 picks changed. On the
2023-2025 window alone that arm is **54.568%** against 56.070%. This is the
number now shown on the board, because it is the chain that is actually served;
it moves to the 1.0-plus-veto row the moment the concurrent lane lands that
change, because the board keys on the policy ids.

### Positive controls

Perfect foresight on exactly the 78 picks the full chain changes is worth
**+9.762 accuracy points over the Tuesday card, 95% [+7.990, +11.690]**, and
perfect foresight on every reachable game is worth **+43.930**. So the
instrument is PROVEN able to resolve effects of ten to forty-four points and is
**not** proven able to resolve one to two, which is where every chain step sits.
**No cell in this battery is `bounded_by_control`**: the control detected an
effect and the steps are present, not absent. Note also the ceiling the control
names -- being right on every one of the 157 picks the chain moves would be
worth about ten points, so the whole chain question is a fight over a small
share of a large reachable pool.

### Nothing here closes anything

All 60 recorded cells are `unresolved_below_power` with a null closing ground.
The consensus step's week-blocked interval is [-3.409, +0.250] -- its upper end
is above zero, so the sign is not RESOLVED and `wrong_sign_resolved` is
inadmissible, which is why this document reports it as a 96/4 expected-value
call to remove rather than as a refutation. The chain-minus-consensus cell's
week-blocked interval sits wholly above zero, which is a positive resolution for
a removal and still closes nothing.

### A staleness finding this lane did not go looking for

**Measured**: the site's own headline archive score reads
`played_union_subset_accuracy`, which matches the **three-member**
`PLAYED_UNION_MEMBER_IDS` subset (`src/nfl_ats/public_board.py`), so the number a
reader sees is **55.888%** while the card actually served is the nine-member
union at **56.886%** on the same 1,503 games. The caption beside it says "the
three-member overlay union that is actually on the board this week", which has
not been true since the nine-member promotion. That is a different lane's fix
(the accessor and its caption), and it is reported here rather than changed,
because changing the headline's own definition is not this lane's mandate.

### Caveats (label how you know it)

- **Historical-backfill replay, not the live path.** Everything on 2023-2025 is
  replayed from the `intraday_hourly` archive because the live store starts
  2026-08-17. Evidence about the chain's SHAPE, not a re-grade of any promotion.
- **The consensus replay uses each game's deadline consensus**, the maximum
  late-week information the rule could see, while production compares against
  whatever the morning's scheduled capture holds. So the 342 fires are an upper
  bound on how often that rule acts, and its measured cost is an upper bound too.
- **The handle step reaches 133 of 816 games**; every other game contributes an
  exact zero to its paired delta. Prospectively its reach is much larger.
- **The rookie-crew step is the archive refit**, gated on the officials-archive
  rookie flag rather than on the Wednesday crew snapshot's own arrival time,
  which is not replayable on this window.
- **2025's official injury rows carry no readable timestamp**, so the veto falls
  back to the headline reader on that season -- the F3p arm, whose reader
  `docs/follow_news_gate.md` measured at split-half reliability 0.071 against
  0.804 for the player-level construction.
- **Correlated decomposition**, and a mined battery: 60 cells over one window,
  one grading and one market replay. No multiplicity correction claimed, and
  never pooled additively with the parent families named in the Registry
  section.
- **The opener evaluator's inherited approximation applies**: only `spread_line`
  is swapped to the opener; other features are close-era
  (`docs/opener_evaluation.md`).
- **The active model changed mid-lane** (`c657058903f3232b` ->
  `2e8c616b476dd0d2`, a feature-table refresh). **Measured**: the two opener
  archives are identical to 0.0 in `home_cover_probability_at_open`,
  `margin_vs_open`, `tue_open_home_spread` and
  `correct_at_open_probability_rule` across all 1,537 rows; only the feature-table
  digest differs, so every number above carries over to the active model
  unchanged. The board artifact is keyed to the new id.

### What the board now shows, and what was NOT wired

Wired: `nfl_ats.public_board.load_refresh_chain_measurement` reads
`headline.json`, refuses a number recorded against a different active model,
refuses one whose policy ids name a chain the code does not serve, and returns
nothing at all when the lane has never run. `board_content` puts it on the
headline and `board_terminal` renders it on both the This Week and Model pages.

NOT wired: no served pick rule, no threshold, no precedence, no ledger, no
manifest, no forecast, and no published page. The consensus-rule finding above
names a branch of `plan_refresh` and changes none of it.

## Commands run

```
python artifacts/served_refresh_card/20260910T011328Z/lane_aj_frame.py
python artifacts/served_refresh_card/20260910T011328Z/lane_aj_measure.py
python artifacts/served_refresh_card/20260910T011328Z/lane_aj_extra.py
python artifacts/served_refresh_card/20260910T011328Z/lane_aj_headline.py
python artifacts/served_refresh_card/20260910T011328Z/lane_aj_record.py
nfl-ats weak-signals record ...   (60 cells; argv lists in record_commands.json)
nfl-ats publish-board --site-destination <scratch>
```

## Registry

60 cells, family `served_refresh_card`, named
`served_refresh_card_<step|total>_<window>`, units `accuracy_points`, league
`nfl`. **Measured** after recording: **60** entries present under the
`served_refresh_card_` prefix, all `unresolved_below_power`, all with a null
closing ground, in a registry then holding 4,895 signals. No before/after delta
is quoted, because concurrent lanes were writing the same file during this run
(three reads of the total during this session returned 4,887, 4,889 and 4,895),
so only the 60-cell count is this lane's own. Exact argument vectors in
`artifacts/served_refresh_card/20260910T011328Z/record_commands.json`, run one at
a time with `NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry`.
