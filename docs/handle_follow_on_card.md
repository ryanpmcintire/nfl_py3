# Following the heavy handle on the played card: frozen predeclaration + result

Lane S, 2026-09-09. Sections 1-4 were written and frozen **before**
`handle_follow_on_card.py` computed any effect; section 5 is filled in after.
Artifacts: `artifacts/handle_follow_on_card/20260909T232503Z/` (local,
gitignored).

## 1. Binding closing-grounds taxonomy (verbatim)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (a) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (b) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it, report
> `probability_positive`, never "contains zero". Within-week correlation is
> ZERO; week-blocked bootstrap. Decide on expected value: forced picks; P+
> above 0.5 on top of what is PLAYED is played.

Every cell below defaults to `unresolved_below_power` whatever its sign.

## 2. Why this lane exists (what lane O already measured)

Read this session, `docs/public_split_on_card.md` and
`artifacts/public_split_on_card/20260909T231541Z/result.json`. Lane O froze a
**fade-the-public** tilt and it lost on the card: B1 (flip to the side with
<=35% tickets when that side's handle is also <=45%) scored **-1.15** accuracy
points on top of the served nine-member card, week-blocked `probability_positive`
0.230; the ticket-only variant B2 scored -1.92, P+ 0.219.

The same run's arm A says the direction of the signal is the **opposite** one,
and it is carried by HANDLE, not tickets:

| Lane O cell | n | Opener-graded accuracy of that side | Week P+ |
|---|---:|---:|---:|
| A4 handle side, >=70% money | 73 | 56.16% (+6.16 pts) | 0.908 |
| A5 handle side, 60-70% money | 28 | 28.57% (-21.43 pts) | 0.016 |
| A6 handle side, <60% money | 32 | 37.50% (-12.50 pts) | 0.117 |
| A7 ticket/handle disagreement, follow handle | 31 | 35.48% (-14.52 pts) | 0.056 |

So the question this lane asks, and lane O did not: **does FOLLOWING the heavy
handle raise expected accuracy on top of what is played?** Arm A4 is a
standalone side rate, not a tilt on the card; a positive standalone rate can be
worth nothing once stacked on a card that already picks that side most of the
time (`docs/` composition rule: an overlay positive alone can be negative
stacked).

Handle exists only in the site's era2 schema, so this rule can touch at most
the era2 slice of the archive; section 3 states that fraction honestly before
any effect is computed.

## 3. Population (frozen, identical to lane O)

1. Splits: `data/raw/public_betting/20260820T111148Z/actionnetwork/
   index.parquet`, rows with `has_any_public_data`.
2. Matched to `data/processed/game_features.parquet` REG-season games on
   normalised `(away, home)` with `|kickoff - site start_time| <= 72h`.
3. Pick deadline per game = `min(kickoff, Sunday 16:00 ET of that week)`.
   Only captures **strictly before** the deadline are eligible; the single
   latest such capture is the reading.
4. Intersected with the active opener evaluation
   `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`
   (model `c657058903f3232b`, seasons 2020-2025).
5. Same 260 scored (non-push at the opener) games, 40 `(season, week)` blocks.
   The build is `build_population.build` from lane O's artifact directory,
   imported unchanged.

**Served card.** The nine-member joint-OR union of
`nfl_ats.four_overlay_composition.POLICY_ID`
(`overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`)
applied to the opener archive's raw probability-rule pick, complemented exactly
once per union member, exactly as `nfl_ats.unserved_tilt_marginals` builds it.
Replay gate: the union's accuracy over all 1,537 archive games must reproduce
`0.5688622754491018` to 1e-12 or the run stops. The served late-week
half-point follow arm is **not** reconstructed, for the reason lane O stated
(`late_week_follow_frame` consumes live intraday rows that begin with the 2026
prospective captures); that is a stated limitation of the baseline, not a
choice.

## 4. Arms (frozen before any effect was computed)

The heavy-handle side of a game is the side whose spread money% is the larger
of the two and is **>= 70%**. All three arms flip the served pick TO that side
when the served pick is on the other side, and are no-ops otherwise.

- **H1** (primary): flip the served pick to the side with handle >= 70% when
  the served pick is on the other side, else keep.
- **H2**: H1 but only when TICKETS on that same side are <= 60% -- money
  without the crowd, the sharp-money shape.
- **H3**: H1 restricted to lines within 3 points of pick'em
  (`|tue_open_home_spread| <= 3`), where half a point of information matters
  most.
- **Controls H1c/H2c/H3c**: perfect foresight on exactly the games each arm
  flips (those picks set to correct), same population, same bootstrap. This is
  the only thing that could make `bounded_by_control` admissible here.

Primary estimate for each arm: the **paired** delta in opener-graded
forced-pick accuracy, candidate minus served card, over the full 260-game
scored population (a game the rule does not touch contributes exactly 0).
Week-blocked bootstrap on `(season, week)`, 20,000 resamples, seed 20260821;
season-blocked reported alongside; overall and per season, with flip counts,
served/candidate accuracy on the flipped subset, and the fraction of the
archive the rule can touch.

Registry: every cell above is recorded through `nfl-ats weak-signals record`
with `--league nfl`, `--effect-units accuracy_points`, names
`handle_follow_on_card_<arm>_<window>`, family `handle_follow_on_card`, and
`--probability-positive` from the WEEK-blocked bootstrap.

Declared multiplicity: three arms plus three controls, one family, one
archive, and the family shares its population with lane O's ten cells. The
per-cell intervals therefore overstate precision; that is stated here, before
the signs are seen, and is not used as a verdict.

## 5. Result

Measured 2026-09-09,
`artifacts/handle_follow_on_card/20260909T232503Z/result.json`. Replay gate
passed exactly: the nine-member union reproduces 0.5688622754491018 over the
1,503 scored archive games (487 union flips) against the raw model's
0.5455755156353959. On the 260-game split population the served card scores
**57.31%** and the raw model 53.85%. Handle exists on **133** of the 260
(2023/30, 2024/71, 2025/32); 2020-2022 contribute exact zeros to every paired
delta.

Effects are accuracy points, paired, week-blocked bootstrap on `(season,
week)`, 40 blocks, 20,000 resamples, seed 20260821.

| Cell | Fires | Flips | Candidate vs served | Week 95% CI | Week P+ | Season P+ |
|---|---:|---:|---:|---|---:|---:|
| **H1 follow handle >=70%** | 73 | 37 | **+0.38** (57.69% vs 57.31%) | [-3.00, +4.28] | **0.5695** | 0.6031 |
| **H2 + tickets <=60%** | 27 | 14 | **0.00** (57.31% vs 57.31%) | [-1.87, +1.88] | **0.5028** | 0.4989 |
| **H3 + \|spread\| <= 3** | 28 | 12 | **-0.77** (56.54% vs 57.31%) | [-2.59, +1.14] | **0.2143** | 0.0449 |
| H1c control on 37 flips | - | 37 | +7.31 | [+3.36, +12.00] | 1.0000 | 0.9923 |
| H2c control on 14 flips | - | 14 | +2.69 | [+0.80, +4.86] | 0.9992 | 0.9561 |
| H3c control on 12 flips | - | 12 | +1.92 | [+0.40, +3.57] | 0.9971 | 0.9561 |

Records on the flipped picks only, which is the whole evidence base:

| Cell | Served record on flips | Candidate record on flips | Net games |
|---|---:|---:|---:|
| H1 | 18-19 (48.6%) | 19-18 (51.4%) | +1 |
| H2 | 7-7 (50.0%) | 7-7 (50.0%) | 0 |
| H3 | 7-5 (58.3%) | 5-7 (41.7%) | -2 |

Per season, H1: 2020-2022 zero flips (no handle in the era1 schema), 2023 12
flips 0.00 pts (6-6 both ways), 2024 21 flips **+4.23** pts (9-12 served
becomes 12-9), 2025 4 flips **-6.25** pts (3-1 served becomes 1-3). The whole
+0.38 is 2024's three-game gain minus 2025's two-game loss.

Reach, stated honestly. H1 fires on 73 of the 133 handle-carrying games
(54.9%) and changes 37 of them (27.8%). Against the population it changes
**0.93 picks per week block**; against the whole opener archive it touches
**2.46%** of games (37 of 1,503), because the Wayback split backfill covers
only 260 of them. Prospectively, with the Saturday/Sunday live captures giving
a reading on every game, the same 27.8% flip rate on handle-carrying games
would touch **about 4.4 picks in a 16-game week** -- a large share of the card
resting on a 19-18 record.

**Decision.** H1 is the only arm on the right side of a coin flip:
`probability_positive` **0.5695** week-blocked (0.6031 season-blocked) for
**+0.38** accuracy points on top of what is played. By the forced-pick rule
that is a play, not a pass -- declining a 57/43 bet is taking the 43. But the
magnitude deserves its own sentence: the entire effect is **one net game out
of 37 flipped picks**, and it would prospectively decide roughly 4.4 picks a
week. My read (inferred) is that this is worth serving only behind the two
market rules already promoted, never in front of them, and only once a
prospective handle reader exists; the historical evidence cannot distinguish
+0.38 from anything between -3.0 and +4.3.

H2 is an exact dead heat -- 7-7, effect 0.000000, P+ 0.5028. The sharp-money
shape (money without the crowd) does not carry H1's small positive; on the
27 games where it fires, following the money is a literal coin flip. H3 is
negative: -0.77 points, P+ 0.2143 week-blocked. Both narrowings removing the
effect is a **diagnosis**, not a verdict: whatever H1 has is not concentrated
in the sharp-money signature or in near-pick'em lines, which is the opposite
of the usual story told about handle, and it is what a session extending this
lane should chase.

Nothing here closes anything. All six cells are recorded
`unresolved_below_power`. H3's season-blocked interval is [-2.139, 0.000] --
its upper end is exactly zero, not below it, and its week-blocked interval
crosses zero, so the sign is not RESOLVED and `wrong_sign_resolved` is
inadmissible. The controls resolve at the perfect-foresight magnitude (+7.31 /
+2.69 / +1.92 points) at these flip counts, which proves the harness sees an
oracle and proves nothing about its ability to see a half-point handle effect,
so `bounded_by_control` is inadmissible too.

## 6. What serving H1 would take (not wired)

1. **Captures already exist.** `public_betting_sat` (Sat 12:00 ET, 240-minute
   window) and `public_betting_sun` (Sun 12:00 ET, 45-minute window), both
   writing `data/raw/public_betting_live/`
   (read, `scripts/capture_scheduler.py:506-529`). The store holds 14
   snapshots, all 2026, none on a graded game yet.
2. **Missing piece: a reader.** Nothing turns
   `data/raw/public_betting_live/` into a per-game
   `(spread money%, spread ticket%)` frame at each game's deadline. The
   historical path used for this lane reads the Wayback index parquet, not the
   live store.
3. **Module and pass.** The tilt belongs in
   `nfl_ats.pick_refresh.plan_refresh`, where both promoted market rules
   already live. The Saturday pass covers every Sunday game (its deadline is
   Sunday 16:00 ET or its own kickoff); the Sunday-noon pass is the last
   refresh before the 13:00 slate and the 16:00 lock. **Thursday games get no
   reading from either capture** -- both land after TNF kickoff -- so TNF stays
   untouched unless a Thursday-morning capture is added.
4. **Precedence: strictly below both existing market rules.** The promoted
   late-week move follow (`LATE_WEEK_MOVE_FOLLOW_POLICY`, >=0.5 points on the
   equal-book Wednesday-to-deadline net move) already takes precedence over the
   1.0-point observed-movement rule (read,
   `src/nfl_ats/pick_refresh.py:105-120`). Heavy handle is largely the cause of
   the line move those rules read, so a handle-follow tilt applied on top of
   them would count the same money twice; it may only apply where neither
   fired, and its ledger row must record the counterfactual like they do.
5. **Not wired in this lane, by instruction.**
