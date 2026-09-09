# Public betting splits on the played card: frozen predeclaration + result

Lane O, 2026-09-09. Sections 1-4 were written and frozen **before**
`scripts/public_split_on_card.py` computed any effect; section 5 is filled in
after. Artifacts:
`artifacts/public_split_on_card/20260909T231541Z/` (local, gitignored).

## 1. Binding closing-grounds taxonomy (verbatim, AGENTS.md)

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

Every cell below defaults to `unresolved_below_power` whatever its sign. No
cell here has a positive control of its own except arm C, which IS the
control; `bounded_by_control` is therefore available only where arm C
resolves and the matching arm-B point estimate sits inside the control's
detectable range.

## 2. What was already measured (do not repeat)

Read this session, `docs/public_betting_sourcing.md` section 8 and
`docs/public_betting_battery_predeclaration.md`, plus the five registry rows
`public_betting_battery_*`:

| Prior cell | Population | Result (effect pts, week-blocked) | P+ |
|---|---:|---|---:|
| fade heavy public (>=70% tickets), close-graded, 2018-2025 | 91 | -3.85 [-12.26, +2.25] | 0.100 |
| fade heavy public, tue_open-graded, 2020-2025 | 80 | -2.50 [-10.00, +4.44] | 0.209 |
| follow sharp divergence (money-bet >= 15pts), era2, close | 62 | -3.23 [-18.29, +9.26] | 0.260 |
| production model accuracy when public heavy AGAINST its pick, close | 47 | -3.19 [-13.64, +6.76] | 0.214 |
| that minus accuracy when public heavy WITH the pick, close | 91 | -7.74 [-25.93, +5.32] | 0.123 |

All five are `unresolved_below_power` in the registry. What they did NOT do,
and what this lane does: (i) they graded the split as a **standalone** side
rule, never as a tilt **on top of the served card**; (ii) they used the
latest capture before **kickoff**, not before the **pick deadline**
`min(kickoff, Sunday 16:00 ET)`; (iii) they binned only at a single >=70%
threshold, with no 60-70 / <60 ladder and no ticket-vs-handle disagreement
cell; (iv) they ran no positive control, so the harness's resolution on this
population is unknown.

## 3. Population (frozen)

1. Splits: `data/raw/public_betting/20260820T111148Z/actionnetwork/
   index.parquet`, rows with `has_any_public_data`.
2. Matched to `data/processed/game_features.parquet` REG-season games on
   normalised `(away, home)` with `|kickoff - site start_time| <= 72h`.
3. Pick deadline per game = `min(kickoff, Sunday 16:00 ET of that week)`
   (`nfl_ats.pick_refresh.pick_deadline`). Only captures **strictly before**
   the deadline are eligible; the single latest such capture is the reading.
4. Intersected with the active opener evaluation
   `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`
   (model `c657058903f3232b`, seasons 2020-2025).
5. Measured before scoring: **268 games**, 260 non-push at the opener, 40
   distinct `(season, week)` blocks, seasons 2020/37, 2021/63, 2022/30,
   2023/31, 2024/74, 2025/33. 138 games carry a handle (money%) reading
   (era2 only; era1's schema has no money field at all). Deadline staleness
   median 30.9h, p25 11.3h, p75 75.3h.
6. Bin sizes before scoring: max ticket% >=70 on 77 games, 60-70 on 87, <60
   on 104; ticket-majority side differs from handle-majority side on 32.
   A side at <=35% tickets exists on 112 games; that side's handle is also
   <=45% on 51.

## 4. Arms (frozen before any effect was computed)

**Served card.** The nine-member joint-OR union of
`nfl_ats.four_overlay_composition.POLICY_ID`
(`overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`)
applied to the opener archive's raw probability-rule pick, complemented
exactly once per union member, exactly as
`nfl_ats.unserved_tilt_marginals` builds it. Replay gate: the union's
accuracy over all 1,537 archive games must reproduce
`artifacts/unserved_tilt_marginals/20260909T210338Z/result.json`'s
`all_probability_positive_above_half` candidate accuracy
(0.5688622754491018) to 1e-12, or the run stops.

**The served late-week half-point follow arm is NOT reconstructed**, and this
is a stated limitation, not an omission by choice: `late_week_follow_frame`
consumes "live intraday spread rows (never historical backfills)"
(read, `src/nfl_ats/sharp_book_movement_features.py:141`), and the live
intraday store begins with the 2026 prospective captures. The historical
`data/market/raw` snapshots could in principle be replayed into it; that is a
separate harness and is named in section 6 as follow-up.

**Arm A -- the split's own predictive value at the Tuesday opener line.**
Sign declared: each cell reports the **public side's** own forced-pick
accuracy at the opener minus 50, in accuracy points. Fading the public is the
negation of this number; no cell is reported only in its fade orientation.
- `A1/A2/A3` ticket side, bins max ticket% >=70, 60-70, <60.
- `A4/A5/A6` handle side, bins max money% >=70, 60-70, <60 (era2 only).
- `A7` disagreement cell (ticket-majority side != handle-majority side):
  accuracy of the **handle** side, minus 50.

**Arm B -- fade-the-public tilt on top of the served card**, applied at each
game's deadline reading.
- `B1` (primary, frozen rule, not tuned): **flip the played pick to the side
  with <= 35% tickets when that side's handle is also <= 45%; otherwise
  keep.** A game whose played pick is already that side is untouched.
- `B2` (secondary, frozen at the same moment): the ticket-only variant --
  flip to the side with <= 35% tickets, with no handle condition -- so that
  era1's 130 handle-less games are not silently dropped from the question.
- Primary estimate for both: the **paired** delta in forced-pick accuracy at
  the opener, candidate minus served card, over the full 260-game scored
  population (a game the rule does not touch contributes exactly 0).
  Week-blocked bootstrap on `(season, week)`, 20,000 resamples, seed
  20260821; season-blocked reported alongside; overall and per season, with
  flip counts and touched-games accuracy.

**Arm C -- positive control.** `C1`/`C2`: a perfect-foresight flip on exactly
the games `B1`/`B2` touch (those games are set to correct), same population,
same bootstrap. This bounds what the harness can resolve at that flip count;
it is the only thing that could ever make `bounded_by_control` admissible
here.

Registry: every cell above is recorded through `nfl-ats weak-signals record`
with `--league nfl`, `--effect-units accuracy_points`, names
`public_split_on_card_<arm>_<window>`, and
`--probability-positive` from the WEEK-blocked bootstrap.

## 5. Result

Measured 2026-09-09, `artifacts/public_split_on_card/20260909T231541Z/
result.json`. Replay gate passed exactly: the nine-member union reproduces
0.5688622754491018 over all 1,537 archive games (487 union flips) against the
raw model's 0.5455755156353959. On the 260-game split population the served
card scores **57.31%** and the raw model 53.85%.

Effects are accuracy points, week-blocked bootstrap on `(season, week)`,
40 blocks, 20,000 resamples, seed 20260821.

| Cell | n | Rate / delta | Week 95% CI | Week P+ | Season P+ |
|---|---:|---:|---|---:|---:|
| A1 ticket side, >=70% | 75 | 52.00% (+2.00) | [-7.65, +12.16] | 0.651 | 0.659 |
| A2 ticket side, 60-70% | 85 | 43.53% (-6.47) | [-16.67, +4.32] | 0.117 | 0.180 |
| A3 ticket side, <60% | 100 | 55.00% (+5.00) | [-5.00, +14.36] | 0.849 | 0.890 |
| A4 handle side, >=70% | 73 | 56.16% (+6.16) | [-3.03, +15.00] | 0.908 | 1.000 |
| A5 handle side, 60-70% | 28 | 28.57% (-21.43) | [-37.50, -2.00] | 0.016 | 0.036 |
| A6 handle side, <60% | 32 | 37.50% (-12.50) | [-30.95, +8.33] | 0.117 | 0.296 |
| A7 disagreement, follow handle | 31 | 35.48% (-14.52) | [-27.78, +3.85] | 0.056 | 0.148 |
| **B1 fade tilt on the card** | 260 | **-1.15** (56.15% vs 57.31%) | [-4.25, +1.99] | **0.230** | 0.175 |
| **B2 ticket-only variant** | 260 | **-1.92** (55.38% vs 57.31%) | [-6.84, +2.94] | **0.219** | 0.046 |
| C1 control on B1's 23 flips | 260 | +3.85 | [+1.55, +6.51] | 1.000 | 0.955 |
| C2 control on B2's 53 flips | 260 | +9.23 | [+5.86, +12.90] | 1.000 | 1.000 |

B1 fires on 51 of 260 games and changes 23 picks (8.8% of the population,
about 1.4 picks in a 16-game week if every game had a reading); B2 fires on
110 and changes 53 (20.4%, about 3.3 picks a week). B1 per season: 2020-2022
zero flips (no handle data before era2), 2023 3 flips -10.0 pts, 2024 13
flips -1.41 pts, 2025 7 flips +3.13 pts.

**Decision.** Do not serve either tilt. `probability_positive` is 0.230 (B1)
and 0.219 (B2) on top of what is played, so serving it is taking the short
side of a roughly 77/23 bet, and the forced-pick rule says the played card
stands. Nothing here closes the mechanism: both cells are recorded
`unresolved_below_power`. A5 is the one cell whose week-blocked interval sits
entirely below zero, but its season-blocked interval is [-40.0, +25.0] on 28
games, so the sign is not RESOLVED and `wrong_sign_resolved` is inadmissible;
it is recorded as category 3 like the rest. The arm-C controls resolve at the
perfect-foresight magnitude (+3.85 / +9.23 points), which proves the harness
can see an oracle at these flip counts and nothing about its ability to see a
half-point split effect, so `bounded_by_control` is inadmissible too.

Multiplicity, stated plainly rather than used as a verdict: the seven arm-A
cells are a mined ladder over one 268-game archive and overlap the five
`public_betting_battery_*` cells' population, so their per-cell intervals
overstate precision. They are recorded as a declared family
(`public_split_on_card`) for exactly that reason.

## 6. Follow-up named before scoring

1. Replay `data/market/raw` (8,795 snapshots, 2020-) through
   `sharp_book_movement_features` so the Wednesday-to-deadline half-point
   follow arm can sit in the historical served baseline.
2. The prospective live store `data/raw/public_betting_live/` holds 14
   snapshots, all 2026, all preseason/Week 1 -- no graded game yet. It is the
   only path to a per-game-guaranteed reading; the Wayback backfill tops out
   at 268 of 1,537 opener-archive games.
