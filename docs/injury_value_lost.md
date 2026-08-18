# Injury value lost vs. injury availability rate: reliability, stress tests, and a cleaner isolation

Written 2026-08-18. Follows `docs/availability_confirmation.md`, which found that
the MOD-07 stack's advantage splits into two axes with opposite mechanism
evidence: `value_magnitude` (how much player value is missing) resolves
(rank-biserial +0.248, p=0.016, monotone tercile gradient), `semantics_shift`
(how much the learned availability RATE moved the inputs) does not (−0.024,
p=0.816, backwards). This document asks the harder question AGENTS.md
prescribes for a small effect: not "is it significant" but "is it reliable,"
then stress-tests the surviving half and screens it on CFB.

**One-line verdict:** the value-lost signal has very high split-half
reliability (0.87–0.93, not noise), the mechanism-screen result survives both
a market-line-movement control and dropping every QB-involved game, and a
methodologically cleaner isolation — free of the semantics-shift confound the
original ablation carried — reproduces it at +1.32 points, P+ 0.8875 on the
same 456 games. CFB cannot screen this family; the reason is a data-audit
fact, not a power problem. Still `unresolved_below_power` by the numbers —
recommend continuing to hold for the already-live 2026 prospective look, not
spending one of the last opener windows now.

---

## 1. What each half actually measures

Traced from `src/nfl_ats/players.py`, `src/nfl_ats/availability.py`,
`src/nfl_ats/participation.py`, `src/nfl_ats/constants.py`.

### 1.1 Value lost (`PLAYER_VALUE_STATE_METRICS`, `constants.py:250-253`)

Two columns: `injury_skill_epa_value_lost`, `injury_defense_disruption_value_lost`
(and their `diff_` regression form). Built by `_injury_value_features`
(`players.py:850-884`) as, per team per game:

```
sum over visible injured players of:
    severity(player)  ×  role_share(player)  ×  value_rate(player)
```

- `severity` — the same 0-1 unavailability probability the rate half uses
  (`_injury_unavailability`, `players.py:678-682`): fixed hand-authored prior
  or learned empirical rate, whichever table built the feature.
- `role_share` — the player's EWMA offense/defense snap share
  (`_compile_lineup`, role-span-8 exponential average).
- `value_rate` — `_player_value_rate` (`players.py:840-847`):
  `100 × EWMA(production)/EWMA(snaps) × reliability`, where reliability is
  `career_snaps / (career_snaps + 200)` — a new player's value is shrunk
  toward zero until it accumulates a track record. `production` is
  `rushing_epa + receiving_epa` for skill-position offense, a weighted sum of
  havoc-stat counts (TFL, forced fumbles, sacks, QB hits, INTs, PBUs) for
  defense.

**Structural finding, not previously documented in code:** the snap-value
update loop (`players.py:1260-1283`) sets `skill_epa = 0.0` unconditionally
for any player whose position is `QB` before accumulating career value — a
QB's rushing/receiving production is deliberately never counted into this
construct. A QB's own injury therefore contributes essentially **zero** to
`injury_skill_epa_value_lost`, by construction, independent of how severe the
injury is. QB value is captured elsewhere (`qb_expected_epa_per_dropback`,
`qb_start_probability`), never here. There is no comment in the code
explaining this; see §6 for the flag.

### 1.2 Availability rate (`availability.py`)

Does not add columns. It changes what `severity` means for the SAME shared
injury columns (`PLAYER_INJURY_STATE_METRICS` + the QB start-probability
columns) that both the baseline and candidate tables read:

- **Fixed** (`fixed_unavailability`, `availability.py:81-111`): a
  hand-authored lookup, e.g. `report_status == "out"` → 0.85-1.0. Frozen,
  bit-faithful to the original active model.
- **Learned** (`learned_unavailability` + `build_season_lagged_availability_rates`,
  `availability.py:300-434`): a season-lagged, hierarchically-shrunk empirical
  rate — `P(unavailable | report_status, practice_status, position_group)`
  — estimated from strictly prior seasons.

### 1.3 The split is a real construct difference, not two encodings

Confirmed three ways:
1. **Different inputs.** Value-lost needs weekly player production
   (`nflverse` player stats) and snap shares; availability rate needs only
   injury report/practice text. Neither is derivable from the other.
2. **Different manifest fingerprints.** `game_features_player_value.parquet`'s
   manifest records `player_feature_version: "v2"` — the version stamp
   `players.py:1292-1300` assigns only when **no** learned-availability table
   was passed at build time. So that table's severity is the fixed prior, and
   its only difference from the baseline `player` table is the two value-lost
   columns (§4 exploits this directly).
3. **Opposite mechanism evidence** (from `availability_confirmation.md`,
   reproduced exactly in §2 below): value_magnitude resolves, semantics_shift
   does not, and the two are only +0.155 correlated with each other on the
   456-game window — different enough to disagree on which games matter.

### 1.4 A third construct exists and already failed: don't conflate it

`participation.py` fits a **separate** value estimate — regularized
adjusted plus/minus (RAPM, ridge regression over play participation) —
producing `offense_rating`/`defense_rating` per player, consumed by
`_injury_participation_value_features` (`players.py:887-908`) into
`injury_offense_participation_value_lost` / `injury_defense_participation_value_lost`
(`PLAYER_PARTICIPATION_STATE_METRICS`, `constants.py:254-257`). This is M6 in
the inventory: **−0.43 points**, the family's one same-kind negative.
Unlike the EPA construct, `participation.py` has no QB carve-out anywhere in
its rating fit — RAPM prices QBs like any other participant. §3.4 tests
whether this is why M6 diverges.

---

## 2. Reproduction (exact, re-run this session, zero new looks)

`scripts/availability_ablation.py` and `scripts/availability_mechanism_screen.py`
were re-run against the live tree (both scripts are free re-reads of the
already-spent `mod07_weak_signal_stack` `[2020, 2021]` opener window per
their own docstrings; neither touches the registry). Every recorded number in
`docs/availability_confirmation.md` reproduced **exactly**:

| quantity | recorded | reproduced |
|---|---|---|
| B−A (availability+value) delta | +1.7544 pts | +1.7544 pts |
| B−A `probability_positive` | 0.899 | 0.899 |
| B−A week-blocked 95% | [−0.69, +4.27] | [−0.6877, +4.2709] |
| `semantics_shift` rank-biserial | −0.024, p=0.816 | −0.02398, p=0.8160 |
| `value_magnitude` rank-biserial | +0.248, p=0.016 | +0.24756, p=0.01628 |

Tercile gradient (from the reproduced `mechanism_screen.json`,
`value_magnitude`): low third **−0.66 pts**, mid **+0.66 pts**, high **+5.26
pts** — monotone. `semantics_shift`: **+2.63 / +1.32 / +1.32** — backwards.
The placebo strata (|spread|, total, week) remain non-monotone comparators.

---

## 3. Testing the mechanism, not the p-value

### 3.1 Split-half reliability of the value-lost signal (decisive test)

Per AGENTS.md, the question that actually settles category vs. category-3 is
reliability, not this window's p-value. Measured on the full leak-safe
history in `game_features_player_value.parquet` (4,431 completed REG games,
2009-2025 — free: no model fit, no forward-chained training, no rotation
window; this is a descriptive property of an existing feature column, the
same kind of "attribution on already-looked-at data" the repo's own ablation
scripts already treat as costless).

Reshaped to one row per team-game (`home_`/`away_` → long), summed
`skill_epa_value_lost + defense_disruption_value_lost` per team-game, split
each of the 384 team-seasons (32 teams × 12 seasons, 2009-2025, **all** of
them had ≥3 games in both halves) into odd- and even-numbered weeks, and
correlated the two halves' means:

| split | Pearson r | 95% CI | Spearman ρ | Spearman-Brown full-length reliability | 95% CI | P+ |
|---|---|---|---|---|---|---|
| **temporal (odd vs even weeks)** | **0.8736** | [0.843, 0.901] | 0.854 | **0.9325** | [0.915, 0.948] | **1.000** (4000/4000 bootstrap draws) |
| construct (skill component vs defense component, season means) | 0.1618 | [0.048, 0.269] | 0.151 | 0.2785 | [0.092, 0.424] | 0.9985 |

**This is not noise.** A team-season's injury-value-lost total in odd weeks
predicts its total in even weeks at r≈0.87 — the quantity is a highly stable
property of a team-season, exactly what AGENTS.md's "no split-half
reliability ⇒ refuted" bar is checking for, and it clears that bar by a wide
margin. (The construct split — offense-side value lost vs defense-side value
lost within the same season — is weaker, as expected: they are genuinely
different personnel groups, not two measurements of the same quantity, so a
smaller shared "bad injury luck this season" component is the right size to
find.)

**What this does and does not prove.** High reliability means the measured
quantity is a real, stably-estimated trait, not sampling noise — it rules
out "refuted: no split-half reliability." It does not by itself prove the
quantity is *predictive* of ATS outcomes; that is what §2's disagreement
statistic already resolved (p=0.016) and §3.2-3.3 stress below. The two
together — reliable measurement, resolvable disagreement statistic — is the
combination AGENTS.md asks for.

### 3.2 Market-move control

Does `value_magnitude` just proxy for how far the line itself moved
open-to-close (information the market already used)? Built the Tuesday-opener
→ close pairing (`clv.build_pairing_table` + `close_reference_table`, the
same machinery `opener_pick_evaluation` uses) for the 456-game window and
computed `market_move = |close_home_spread − tue_open_home_spread|`.

| check | result |
|---|---|
| `value_magnitude` vs `market_move` | Pearson **0.029**, Spearman **0.041** — essentially uncorrelated |
| disagreement rank-biserial, raw | r=0.2476, p=0.0163 (matches §2) |
| disagreement rank-biserial, **residualized** on `market_move` (OLS residual of value_magnitude regressed on market_move) | r=**0.2491**, 95% [0.047, 0.451], p=**0.0156** |

The residualized statistic is statistically indistinguishable from the raw
one. `value_magnitude` is not a stand-in for line movement; the effect
survives the control intact.

### 3.3 Drop-QB test

Split the 456 games by whether either team's presumed starter carried any
listed injury unavailability (`home_/away_qb_start_probability < 0.999` in
the candidate table):

| subset | games | share | disagreement rank-biserial | 95% CI | p (two-sided) | accuracy delta on subset |
|---|---|---|---|---|---|---|
| **QB-stable** (neither starter listed) | 397 | 87.1% | **+0.188** | [−0.027, 0.403] | 0.086 | +1.01 pts |
| QB-affected (either starter listed) | 59 | 12.9% | +0.700 | [0.109, 1.291]* | 0.020 | +6.78 pts (4 disagreements only) |

\*normal-approximation interval on a 59-game/4-disagreement subset; treat as
directional, not precise.

**The effect is not a QB story.** On the 87% of games where neither team had
any QB availability question, the value-lost signal still points the same
direction at comparable magnitude (r=0.188 vs the full-sample 0.248), just
below the resolution of this small a slice (p=0.086, one-sided ≈0.96 by
normal approximation). This is the expected, weaker-but-consistent shape for
a real effect measured on 87% of an already-small sample — and it is exactly
consistent with §1.1's structural finding that `injury_skill_epa_value_lost`
zeroes out QB production by construction. The QB-affected subset shows an
even larger point estimate, but on only 4 disagreement games it is noise-prone
and should not be read as "the effect concentrates in QB games" — if
anything it says QB-affected games tend to have simultaneous non-QB injuries
riding along, which the construct is already picking up.

### 3.4 Why M6 (RAPM-participation value lost) went negative, if reliability isn't the answer

Ran the identical temporal split-half check on
`game_features_player_participation.parquet` (M6's own construct):

| split | Pearson r | Spearman-Brown reliability | 95% CI | P+ | team-seasons |
|---|---|---|---|---|---|
| temporal | 0.8603 | 0.9249 | [0.900, 0.944] | 1.000 | 256 |
| construct | 0.1430 | 0.2502 | [0.031, 0.423] | 0.9855 | 256 |

**Reliability does not distinguish M6 from the positive EPA construct** —
both are equally stable, non-noise measurements. So "unreliable measurement"
cannot be the explanation for M6's −0.43. Two more likely explanations, from
reading the feature-set wiring (`constants.py:511-519`):
`full_player_participation = full_player_value + player_participation_values`
— M6's own contrast (`player_value → participation`) tests the **marginal**
value of adding the RAPM construct **on top of** a model that already
contains the EPA value-lost columns, not whether "value lost" as a concept
works. Two candidate mechanisms for a negative marginal addition, offered as
hypotheses, not proven: (a) the two value-lost constructs are correlated
enough that stacking both is closer to adding correlated noise than new
information; (b) unlike the EPA construct, `participation.py`'s RAPM fit has
no QB carve-out anywhere in its code, so it **does** price QB value into
`injury_offense_participation_value_lost` — reintroducing exactly the
QB-coupled noise §3.3 shows the EPA construct avoids by construction.

---

## 4. A cleaner isolation, free, on the same spent window

The original MOD-07 ablation's B−A contrast (+1.75 pts, P+ 0.899) is not a
pure value-lost measurement: table B (`weak_stack`) carries **learned**
severity for every shared injury column, so B−A mixes `value_magnitude` and
`semantics_shift` together, which is precisely why §2's stratification was
needed to separate them after the fact.

§1.3 established that `game_features_player_value.parquet` was built with
**fixed** severity (manifest `player_feature_version: "v2"`) — identical
injury semantics to the `player` baseline table. So `player_value` profile on
`game_features_player_value.parquet`, contrasted against `player` profile on
`game_features_player.parquet`, differs from the baseline by **only** the two
value-lost columns, with **zero** semantics-shift confound — a cleaner
version of arm B that was never run. Ran it (same free re-read precedent,
same [2020, 2021] spent window, no registry write):

| contrast | delta | week-blocked 95% | `probability_positive` | disagreements | disagreement split |
|---|---|---|---|---|---|
| **D−A: value-lost only, zero semantics confound** | **+1.316 pts** | [−0.460, +3.247] | **0.8875** | 26 | candidate 61.5% (16/26) vs baseline 38.5% (10/26) |
| (for comparison) B−A: value-lost + semantics-shift mixed | +1.754 pts | [−0.688, +4.271] | 0.899 | 34 | candidate 61.8% vs 38.2% |

Removing the semantics-shift confound costs about a third of a point and 8
disagreement games (exactly the games §2's stratification says the failed
`semantics_shift` axis was moving), but the value-lost effect itself survives
essentially intact: same sign, same disagreement win rate, comparable
`probability_positive`. **This is the number a narrowed predeclaration should
use**, not the original conflated +1.75.

---

## 5. CFB screen

**Not feasible for this construct, and it is a data-audit fact, not a power
problem.** `value_lost = severity(unavailability) × role_share ×
value_rate`; `severity` requires a pregame injury/availability signal.
`docs/cfb_data.md` § "Availability semantics (fail closed)" (lines 66-84)
already establishes, by direct audit, that **no such signal exists in any CFB
source**:

- `espn_cfb_injuries` has zero assets; CFBD v5 has no injuries endpoint.
- Game rosters are scrape-time listings, not pregame availability: **zero of
  27,471 players changed their Active/Inactive flag across all of 2024**.
  Those columns are quarantined out of the canonical CFB table by contract.
- Play participants are credited actors only (passer/rusher/tackler/...);
  absence of credit is never evidence of absence — there is no CFB analogue
  of `report_status`/`practice_status`.

Without `severity`, `value_lost` cannot be computed pregame on CFB at all —
not weakly, not proxy-able, structurally absent. The nearest available
substitute, a **postgame realized-participation-continuity** proxy (not a
severity-weighted value-lost measure, just "did the same players show up
again"), was already predeclared as its own family (`cfb_role_continuity`,
`registry/rotation_registry.json`, `status: closed_negative`) and **lost**:
**−0.67 points on 8,933 clean-core games** (`docs/cfb_role_features.md`).
That family closed with no NFL window ever assigned. The CFB side of this
family has had its one look and it did not clear. Nothing here reopens it.

---

## 6. Defect found in passing

`players.py:1260-1283`'s QB exclusion from `skill_epa` (§1.1) is undocumented
— no comment in the code explains it, and nothing in `docs/` states it as a
declared design choice. §3.3's empirical result depends on this exclusion
being intentional (it is exactly why the value-lost signal isn't a QB proxy),
so a future refactor that "fixes" the apparent asymmetry (why does defense
count QB havoc stats but offense zero out QB rushing/receiving?) without
reading this document would silently change the construct's meaning and
invalidate every number in this file. Recommend a one-line comment at
`players.py:1263` stating that QB production is deliberately excluded here
because it is captured separately via `qb_expected_epa_per_dropback`, and
that removing the exclusion is a model change requiring re-evaluation, not a
refactor (the same convention already used for `fixed_unavailability`,
`availability.py:92-93`). Not fixed in this session — touching frozen feature
semantics is out of scope for a research write-up and would itself require
re-evaluation.

---

## 7. Recommendation and predeclaration

**Classification: stays `unresolved_below_power`.** Every accuracy interval
in this document (§2, §3.2, §3.3, §4) still crosses zero at 456 games —
expected at this evaluator's ~2-point resolution, per AGENTS.md, and not a
reason to close the line. What changed this session is *evidence quality*,
not the interval: split-half reliability of 0.87-0.93 rules out "refuted: no
reliability," the market-move and drop-QB stress tests both survive intact,
and §4 supplies a methodologically cleaner number (+1.32 pts, P+ 0.8875, zero
semantics confound) to predeclare against instead of the original conflated
+1.75.

**Do not spend a new NFL window now.** Three independent reasons converge:
(1) `docs/mod07_stack.md` already ruled out a second opener window on this
exact family as iterating-until-it-wins; nothing here changes that reasoning.
(2) The cleanest available evidence is already in flight for free: the 2026
`mod07_weak_signal_stack` prospective challenger is registered
`ACTIVE_PROSPECTIVE` (`artifacts/prospective/challengers.json`, confirmed live
in `docs/availability_confirmation.md` §4.1) and is being scored weekly at no
window cost. (3) `registry/rotation_registry.json` (read-only this session,
confirmed unmodified) shows `mod07_weak_signal_stack` has spent its only
window, `[2020, 2021]`; a family inheriting that contamination would draw the
registry's earliest-eligible block next, **`[2022, 2023]`** — one of only two
remaining full 2-season opener blocks in the 2020-2025 opener-eligible pool
(`[2022, 2023]` and `[2024, 2025]`), consistent with the task brief's count of
very few opener windows left for the whole project. Do not touch it.

**Free next step, not executed here:** once 2026 accrues enough weeks, the
same ablation this document ran in §4 (D−A, `player_value` on a
fixed-severity table vs `player`) can be re-run on the live 2026 season for a
second, independent, discount-free look at the narrowed construct
specifically — the currently-registered prospective challenger tests the
whole bundle (value + rate + bias), not this isolated half.

**The predeclaration text, to freeze verbatim if a window is ever spent here:**

> **Family:** `injury_value_lost_narrowed`
> **Inherits:** `mod07_weak_signal_stack` (contaminates `[2020, 2021]`)
> **Grade:** `opener`
> **acknowledges_mined_2018_2025:** `true`
> **Hypothesis:** Value-weighted injury magnitude alone — `player_value`
> profile (fixed-prior severity, zero semantics-shift confound; see §4 of
> `docs/injury_value_lost.md`) contrasted against the `player` baseline —
> improves forced-pick accuracy against the Tuesday opener, without the
> learned-availability-RATE half (refuted its own mechanism check,
> `docs/availability_confirmation.md` §3) and without the three bias
> features (measured a coin flip, same document).
> **Predeclared thresholds:** `probability_positive` ≥ 0.90 → confirmed;
> ≤ 0.10 → closed_negative; otherwise unresolved. (Matches MOD-07's own
> convention for direct comparability.)
> **Prior evidence to cite, not re-litigate:** §4's free re-read gives
> +1.316 points, week-blocked 95% [−0.460, +3.247], `probability_positive`
> 0.8875 on the already-spent window — informative, not a look, and must not
> be treated as the confirmation result.
> **Window this would draw:** `[2022, 2023]`, the registry's earliest
> eligible block after inherited contamination — one of the project's last
> handful of opener windows. **Do not assign until the 2026 prospective
> number is in hand** (`docs/availability_confirmation.md` §4.3: "one clean
> bit, not a confirmation" — but the first non-mined evidence this family
> will ever have).
