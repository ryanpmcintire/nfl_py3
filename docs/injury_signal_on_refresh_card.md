# The injury-signal tilt as a marginal on the served refresh card

Lane Q, 2026-09-09. **The predeclaration below was written and frozen before
any accuracy, delta, interval or flip count in the Results section was
computed.** Coverage counts quoted inside the predeclaration were measured
first, deliberately, without touching outcomes — the same order
`docs/sharp_book_movement_lead.md` used.

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
is expected value on FORCED picks: a construct with `probability_positive`
above 0.5 on top of what is PLAYED is played; 0.90-style thresholds govern only
what the docs may claim. Picks are editable until `min(kickoff, Sunday 16:00
ET)` per game (`docs/late_week_refresh.md`).

## The question

`injury_signal_refresh_tilt` (`src/nfl_ats/injury_signal_refresh_tilt.py`,
challenger entry in `artifacts/prospective/challengers.json`) carries the
largest registered standalone number in this project: **+17.07 accuracy
points, `probability_positive` 0.976, n=123**
(`docs/movement_attribution.md`'s `pop_threshold_injury` cell). That number is
a **backward-looking attribution**: it grades "follow the market's
Tuesday-to-CLOSE move" on the subset of adverse moves that an official
post-Tuesday injury downgrade rationalises. It has never been measured as a
**marginal on top of what the refresh path actually serves**.

What is served today is:

1. the Tuesday card — the raw model probability pick at the frozen Tuesday
   opener, complemented by the nine-member joint-OR overlay composition
   (`src/nfl_ats/four_overlay_composition.py`, `POLICY_ID =
   overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`);
   then
2. the promoted late-week follow rule
   (`src/nfl_ats/pick_refresh.py` `LATE_WEEK_MOVE_FOLLOW_POLICY`,
   `src/nfl_ats/sharp_book_movement_features.py` `late_week_follow_frame`):
   at each refresh pass, the equal-book Wednesday-to-deadline net spread move
   over the frozen twelve-book universe; `|net| >= 0.5` takes the side the
   market moved toward.

The injury tilt's own mechanism (follow post-Tuesday injury news) and the
follow rule's mechanism (follow post-Wednesday line movement) are, per
`docs/movement_attribution.md`'s own attribution, **partly the same event seen
twice** — the line moves *because* the injury news landed. So the only
decision-relevant question is the marginal, and whether the two collide.

## Population (frozen before scoring)

Both sources must exist for a week to be scorable:

- **Hourly odds archive.** The frozen MKT-15/CX18 intraday population:
  `artifacts/experiments/sharp_book_movement/kickoff.parquet` (**measured**:
  816 regular-season games, 272 in each of 2023, 2024, 2025) with its paired
  `quotes.parquet` (**measured**: 2,387,404 spread/total quote rows, carrying
  `snapshot_timestamp_utc`, so the served `late_week_follow_frame` back-dating
  refusal runs unchanged). Reusing this cache rather than re-deriving the
  intraday load is the same cache `scripts/sharp_book_movement_on_production.py`
  accepts through `--quotes-cache` / `--kickoff-cache`.
- **Injury source.** Either the official injury report
  (`data/players/raw/20260909T223500Z/injuries.parquet`) or the
  ProFootballTalk headline archive
  (`data/raw/injury_news/20260819T191639Z/index.parquet`), exactly as
  `injury_signal_for_game` chooses between them. **Measured** coverage, taken
  before any outcome was read:

  | season | official rows | official rows with a real `date_modified` | `observed_at_basis` | PFT injury-relevant headlines in season |
  |---|---:|---:|---|---:|
  | 2023 | 5,451 | 5,451 | `date_modified` | present (archive spans 2009-09 to 2026-08) |
  | 2024 | 5,954 | 5,954 | `date_modified` | present |
  | 2025 | 5,783 | **0** | `week_proxy` | present |

  **Disclosed in advance, not discovered later:** the module selects the
  official path whenever the season has ANY official row
  (`injuries["season"].eq(season).any()`), and 2025's rows have a null
  `date_modified`, so `_severity_asof` admits nothing and the module returns
  `net_score = 0.0` for every 2025 game. That is a property of the module as
  written, so the PRIMARY arm keeps it; a disclosed secondary variant scores
  2025 through the PFT fallback the module would have used had the official
  snapshot carried no 2025 rows at all.

- **Model / grading.** The active model's own opener evaluation,
  `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`
  (**measured**: model `c657058903f3232b`, feature digest
  `5c5d1944...`, 1,537 games 2020-2025, of which the 816 above are the
  scorable window). Every arm is graded at `margin_vs_open` — the frozen
  Tuesday opener, the pool's own settlement line. Opener pushes
  (`margin_vs_open == 0`) are excluded from accuracy and kept in flip counts.

Windows are declared spent for the family `injury_signal_on_refresh_card`,
seasons 2023-2025. This window overlaps the MKT-15 screen and the
`observed_movement_*` family; the cells below are **correlated with**, and
must never be pooled additively with, `sharp_book_movement_equal_2023_2025`,
`movement_attribution_pop_threshold_injury`, or the `observed_movement_*`
entries.

## Arms (frozen)

Every arm is a pick side per game, graded at `margin_vs_open`.

- **B0 — raw probability pick.** `pick_home_at_open_probability_rule` from the
  opener evaluation (the served read, home-side offset included). Used only
  for the reproduction check.
- **B1 — Tuesday card.** B0's probability, complemented once on the union of
  the nine composition members
  (`nfl_ats.four_overlay_composition.apply_four_overlay_composition`, run on
  the opener archive with the point-in-time arrest incidents
  `data/raw/player_arrests/20260820T153000Z/incidents_point_in_time.parquet`,
  the Tuesday-noon and kickoff-nearest forecast archives, and the PBP-08 flag
  table). Any member reporting `disabled_input_unavailable` is disclosed in
  the results, not silently dropped.
- **S — served refresh card.** B1, then `late_week_follow_frame` at each
  game's own pick deadline. `now` is passed as the current instant so the
  function's back-dating refusal runs; the evidence window is bounded by the
  frame's own per-game `cutoff_utc = min(kickoff, that week's Sunday 16:00
  ET)` and by its Wednesday-open / Sunday-close bounds, which is exactly the
  deadline semantics `plan_refresh` serves. `|equal_net_move| >= 0.5` takes
  the market side; otherwise B1 stands.
- **S+I — PRIMARY candidate.** S, plus the injury tilt applied on top:
  `injury_signal_for_game` (imported unchanged from
  `nfl_ats.injury_signal_refresh_tilt`) with `picked_team` = the team **S**
  picks, `opponent_team` = the other, `now` = that game's pick deadline,
  `injuries` / `pft` = the snapshots named above. `fires` (official
  `net_injury_score >= 2`, PFT `net_pft_score >= 1`) flips S's side.
- **I+S — precedence variant.** The injury tilt applied to **B1** first, then
  the follow rule overriding wherever it fires. This is the same two rules in
  the opposite precedence order, and it is what decides the "precedence vs the
  follow rule" half of the decision line.
- **P25 — disclosed source variant.** S+I with 2025 forced through the PFT
  fallback (the module's own `net_pft_score >= 1` path), 2023-2024 unchanged.

## Measurement (frozen)

Paired, week-blocked bootstrap: **20,000 draws, seed 20260821**, whole
`(season, week)` blocks resampled with replacement, game-weighted paired
accuracy delta × 100 (`accuracy_points`), 95% percentile interval, and
`probability_positive` computed by
`nfl_ats.evidence_conventions.probability_positive_from_draws`
(`P(>0) + 0.5·P(==0)` — the 2026-09-08 fix; a no-op candidate reads 0.5, not
0.0). Season deltas are reported for every arm; no season is selected after
scoring.

Cells recorded to the weak-signal registry, all `effect_units=accuracy_points`,
`league=nfl`, named `injury_signal_on_refresh_card_<arm>_<window>`:

| name | candidate | baseline |
|---|---|---|
| `..._follow_on_raw_2023_2025` | follow rule on B0 | B0 |
| `..._follow_on_composed_2023_2025` | S | B1 |
| `..._tilt_on_served_2023_2025` | S+I | S |
| `..._tilt_on_served_2023` / `_2024` / `_2025` | S+I | S (per season) |
| `..._tilt_precedence_2023_2025` | I+S | S |
| `..._tilt_pft2025_2023_2025` | P25 | S |

Diagnostics reported alongside, never instead of, the cells:

1. **Follow-rule flips** vs B1, and the reproduction check against MKT-15's
   own measured `+1.75219` on B0.
2. **Injury flips** vs S: count, per season, picks per week.
3. **Collision table**: of the injury flips, how many land on a game where the
   follow rule also fired, split into *same side* (the injury tilt would flip
   the follow rule's own pick away from the market — a genuine conflict) and
   *opposite side* (the tilt and the market already agree, so the tilt is
   redundant there).
4. **Tuesday-visibility / playability check** (`docs/movement_attribution.md`
   demands it): for every injury flip, the timestamp of the latest piece of
   news that contributed to the firing score (official: the maximum
   `date_modified` among the two teams' skill-position rows in
   `(own-week Tuesday noon ET, deadline]`; PFT: the maximum `lastmod` in the
   same window) against that game's `min(kickoff, Sunday 16:00 ET)` deadline.
   Count flips **not** playable inside that window. By construction of the
   `now = deadline` cutoff this should be zero; the check exists to prove it
   rather than assert it, and to report how much of the firing evidence
   arrives after Tuesday at all (the whole premise of a refresh pass).

## Interpretation rules (fixed before results are seen)

- The decision is expected value on a forced card. If S+I vs S reads
  `probability_positive` above 0.5, serving it raises expected accuracy and
  the write-up says so **before** listing what is uncertain about it. If it
  reads below 0.5, the tilt is not worth serving on top of the follow rule —
  and that is still `unresolved_below_power`, not a closure, unless the whole
  interval sits on the wrong side of zero (`wrong_sign_resolved`).
- A collision table dominated by *opposite side* rows means the follow rule is
  already collecting the injury information, and the tilt's remaining value
  lives only where the market has not moved yet — the "injury-only" lane
  `classify_disagreement` was built to isolate. That lane is reported
  separately.
- No arm here is wired into production by this lane. If the decision line
  comes out positive, the write-up names the refresh pass, the module and the
  precedence — it does not change any served code.

---

## Results

**Measured** this session,
`artifacts/injury_signal_on_refresh_card/20260909T231954Z/`
(`per_game.parquet`, `injury_readings.parquet`, `cells.csv`,
`diagnostics.json`, `reliability.json`, `metadata.json`,
`record_commands.json`), active model `c657058903f3232b`, 816 archive games,
799 non-push grades, 54 weeks.

### Decision

**No. Do not serve the injury tilt on top of the refresh card.** On the 799
opener-graded games it reads **-1.627 accuracy points, week-blocked 95%
[-4.709, +1.255], `probability_positive` 0.139**, and it would touch **198 of
816 picks — 3.7 a week**, one pick in four. On a forced card the decision rule
is expected value, and 0.139 is the wrong side of the coin: playing it is
taking a 14/86 bet that a large, frequent pick change helps. Both seasons where
the signal fires at all point the same way (2023 **-3.759**, P+ 0.084; 2024
**-1.128**, P+ 0.383), and the lane the front-running story actually needs —
games where the market has NOT moved — is the worst of the three cuts
(**-2.954**, P+ 0.120, n=474).

Per the binding taxonomy this is **not** a closure. Every interval above
crosses zero, the flag has non-zero split-half reliability (0.0748 over 64
team-seasons, 2023-2024), and no positive control bounds it. All eleven cells
are recorded `unresolved_below_power`. What is settled is the **decision**, not
the mechanism.

### The reproduction check, and what it actually reproduced

| quantity | MKT-15 (`artifacts/experiments/sharp_book_movement/20260905T205038Z`) | this run |
|---|---:|---:|
| games flagged by the equal-book rule | 329 | **329** |
| exposure split-half reliability (odd/even team-season) | 0.1008829 | **0.1008828675834187** |
| follow-rule flips vs the raw pick | 143 | 140 |
| follow-rule delta on the raw pick | +1.75219 | **+0.876** [-1.880, +3.671], P+ 0.731 |

The **exposure** reproduces exactly — the same 329 flagged games and the same
reliability to seven decimals, which is the check that matters for trusting
every downstream cell. The **delta** does not, and the reason is not the rule:
the baseline pick stream changed. MKT-15 was scored 2026-09-05 against the
then-active model; this run uses `c657058903f3232b` with the served home-side
offset promoted 2026-09-07/08, which moves the raw pick on enough games to move
three flips and 0.88 points. Stated plainly rather than smoothed over: **the
promoted follow rule is worth about half of what its promotion measurement
said, on the current model** — and on top of the nine-member composition card
it is worth **-0.375** [-3.270, +2.532], P+ 0.400. That is a second, separate
finding this lane did not go looking for, and it belongs to MKT-15/CX18, not to
the injury question.

### The cells

Paired, week-blocked bootstrap, 20,000 draws, seed 20260821, graded at
`margin_vs_open`. `probability_positive` is the decision-relevant number.

| cell | candidate | baseline | n | delta (pts) | 95% week-blocked | P+ |
|---|---|---|---:|---:|---|---:|
| `..._composition_on_raw_2023_2025` | nine-member Tuesday card | raw pick | 799 | +1.001 | [-3.186, +5.051] | 0.686 |
| `..._follow_on_raw_2023_2025` | follow rule on raw | raw pick | 799 | +0.876 | [-1.880, +3.671] | 0.731 |
| `..._follow_on_composed_2023_2025` | **served refresh card** | Tuesday card | 799 | -0.375 | [-3.270, +2.532] | 0.400 |
| **`..._tilt_on_served_2023_2025`** | **served + injury tilt** | **served** | **799** | **-1.627** | **[-4.709, +1.255]** | **0.139** |
| `..._tilt_on_served_2023` | served + tilt | served | 266 | -3.759 | [-9.259, +1.487] | 0.084 |
| `..._tilt_on_served_2024` | served + tilt | served | 266 | -1.128 | [-8.487, +5.703] | 0.383 |
| `..._tilt_on_served_2025` | served + tilt | served | 267 | 0.000 | [0, 0] (no-op) | 0.500 |
| `..._tilt_precedence_2023_2025` | tilt first, follow overrides | served | 799 | -1.752 | [-4.738, +1.132] | 0.120 |
| `..._tilt_where_follow_silent_2023_2025` | served + tilt, market quiet | served | 474 | -2.954 | [-7.933, +1.949] | 0.120 |
| `..._tilt_pft2025_2023_2025` | 2025 via PFT fallback | served | 799 | -1.126 | [-4.646, +2.261] | 0.258 |
| `..._tilt_pft2025_2025` | PFT path alone, 2025 | served | 267 | **+1.498** | [-3.690, +6.691] | **0.716** |

Accuracies behind the primary row: Tuesday card 448/799 (56.07%), served
refresh card 445/799 (55.69%), served + tilt 432/799 (54.07%).

### Why it is negative: the tilt is measuring a level, not the news

**Measured** (`diagnostics.json`): the module's official path computes
`player_delta = severity(now) - severity(own-week Tuesday noon ET)`. Across the
816 games, **only 7 have ANY official skill-position injury row filed by their
own-week Tuesday noon** (4 in 2023, 3 in 2024, 0 in 2025) out of 3,402 rows
filed by the deadline. Of the 198 games where the tilt fires, **4** have a
Tuesday-noon baseline row.

The reason is the league's filing calendar, **measured** on the same snapshot
across 2020-2024: of 27,617 official filings, **21,249 land on Friday** and
only **187 land on a Monday or Tuesday** (Wed 2,510; Thu 1,602; Sat 2,036; Sun
33). The first filing of a week lands Wednesday or later in essentially every
week.

So the subtraction is against zero, and `net_injury_score` is not "how much
worse the picked team's news got since Tuesday" — it is **the picked team's
whole-week skill-position injury LEVEL minus the opponent's**. Firing on
`>= 2` therefore fades the team with the longer QB/RB/WR/TE injury list. That
is a team-quality quantity the Tuesday line has already priced (the
already-priced ceiling `docs/decision_rule.md` and
`docs/injury_news_sourcing.md` both describe), which is exactly where a
negative marginal is expected. The module's own docstring describes the
delta reading; the data cannot supply it. **This is a modelling defect in the
challenger, named, not a threshold to bolt on or off.**

The same construction — same Tuesday-noon baseline, same official archive —
is what produced `docs/movement_attribution.md`'s **+17.07-point**
`pop_threshold_injury` cell. **Inferred, not re-measured here:** I think that
cell's INJURY class is therefore also a level flag rather than the post-Tuesday
news flag it is described as, and its headline number should be re-read that
way. Re-running that document's own population is the way to settle it; this
lane did not do it.

### Flip and collision table

**Measured** (`metadata.json`, `collision`):

| quantity | count |
|---|---:|
| games in window | 816 |
| composition flips vs the raw pick | 260 |
| follow rule fired (`\|equal_net_move\| >= 0.5`) | 329 |
| follow-rule flips vs the Tuesday card | 145 |
| **injury-tilt flips vs the served card** | **198** (3.7 / week) |
| injury flip where the follow rule also fired | 79 |
| — of which the injury flip reverses an actual follow-rule flip | 38 |
| injury flip where the follow rule was silent | 119 |
| injury flips 2023 / 2024 / 2025 | 100 / 98 / 0 |
| injury flips under the PFT-2025 variant | 286 (88 of them in 2025) |

Every one of the 79 collisions is by construction *against* the market side —
on a follow-fired game the served pick already IS the market side, so flipping
it necessarily opposes the move. That is arithmetic, not evidence; the
decision-relevant number is that **40% of the tilt's firings would be undoing
the promoted follow rule's own pick.**

### Tuesday-visibility / playability check

**Measured** (`metadata.json`, `visibility` — one row per firing):

- 198 injury flips checked.
- **198 of 198** are backed by at least one piece of contributing news
  timestamped inside `(own-week Tuesday noon ET, that game's pick deadline]`.
- **0 of 198** depend on news dated after `min(kickoff, Sunday 16:00 ET)`.
- 0 flips have no datable contributing news.

So nothing here is unplayable: every firing is decidable inside the pool's own
editing window, by construction of the `now = deadline` cutoff and confirmed
per-game rather than asserted. The check also shows the flip side of the
mechanism defect above — the news IS all post-Tuesday, because the official
report simply does not exist before Wednesday, which is why there is no
Tuesday baseline to subtract.

### The one positive lane, reported honestly

`..._tilt_pft2025_2025` is **+1.498 points, [-3.690, +6.691], P+ 0.716** on 267
games, 88 flips. It is the only arm in this battery whose input is a genuine
post-Tuesday *news* window: ProFootballTalk headline timestamps inside
`(Tuesday noon, deadline]`, differenced between the two teams, with no empty
baseline to subtract against. Its split-half reliability is 0.333 over 32
team-seasons — four times the official flag's. It is also one season, mined out
of a battery of eleven cells, with no multiplicity correction claimed, and it
does not survive pooling with 2023-2024 (`..._tilt_pft2025_2023_2025`,
-1.126, P+ 0.258).

**Inferred:** I think the headline-count path, not the official-report path, is
where this challenger's remaining value lives, and the way to find out is to
run the PFT construction on 2023 and 2024 as well — where the official path
currently pre-empts it — rather than to re-tune the official one. That is a
new predeclaration, not a re-read of this one.

### What serving it would take (asked and answered; nothing was wired)

The decision is not to serve, so this is recorded only because the question was
asked. It would take: the **Thursday and Saturday `refresh-picks` passes**
(`nfl_ats.pick_refresh.plan_refresh`, the same passes the follow rule already
runs in); the tilt computed inside
`nfl_ats.injury_signal_refresh_tilt.injury_signal_for_game` against
`RefreshedGame.new_pick_side` instead of `model_only_pick_side` (today it reads
the model-only hold arm, so it is not even measuring the played side); and
precedence **after** the follow rule, since giving the follow rule the last
word measured worse (-1.752 vs -1.627). Both orderings are negative, so the
precedence question has no live answer to give.

### Caveats (label how you know it)

- **Correlated decomposition.** Every cell shares the 816-game 2023-2025
  intraday archive with `sharp_book_movement_equal_2023_2025` and the injury
  construction with `movement_attribution_pop_threshold_injury`. Never pooled
  additively with those or with each other.
- **Mined battery**, eleven cells, no multiplicity correction claimed —
  matching every other mined-battery document in this repo.
- **Skill positions only** (QB/RB/WR/TE), inherited from the module: offensive
  line and defensive front injuries are invisible to the flag. A likely
  undercount of real injury news, not an overcount.
- **The 2025 official rows are timestampless** (`observed_at_basis
  week_proxy`), so the primary arm is a literal no-op on that third of the
  window. Disclosed in the predeclaration, before scoring, not discovered
  afterwards.
- **The opener evaluator's inherited approximation applies**: only
  `spread_line` is swapped to the opener; other features are close-era
  (`docs/opener_evaluation.md`).
- Nothing in production was changed by this lane. No ledger, manifest,
  forecast or board was touched.

### Registry entries recorded

Eleven entries, all `effect_units=accuracy_points`,
`classification=unresolved_below_power`, `closing_ground=null`, league `nfl`,
family `injury_signal_on_refresh_card_2023_2025`. Registry total 4,509 ->
**4,520**. Exact argument vectors in
`artifacts/injury_signal_on_refresh_card/20260909T231954Z/record_commands.json`.
