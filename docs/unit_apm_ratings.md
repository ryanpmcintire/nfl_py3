# Unit-level APM ratings: reliability gate (predeclaration)

**Status:** predeclared 2026-09-03, BEFORE any unit-level rating or
reliability number was computed. Sections 0–8 are frozen; section 9
(Results) is appended after the run and nothing above it is edited
afterwards.

**Owning work package:** PER-09 (latent player ratings: hierarchy and units
remain after the special-teams slice). Files: this document,
`scripts/unit_apm_screen.py`, `tests/test_unit_apm_screen.py`,
`artifacts/unit_apm/`.

**Parents:** the season-lagged offense/defense APM
(`src/nfl_ats/participation.py`, failed its matched ATS screen) and the ST
reliability gate (`docs/st_player_ratings.md`, SB 0.436, recorded). This
slice reuses the frozen APM recipe (Ridge alpha 1000, team effects at scale
11, EPA clip 5.0, 500-play prior reported) and the ST screen's play-table
discipline, and modifies neither.

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

This slice measures trait reliability per unit, not an ATS effect. A unit
whose reliability is indistinguishable from zero meets the admissible
`no_split_half_reliability` ground FOR THAT UNIT ONLY (other units are
separate traits, never pooled into one verdict). Healthy units record
`unresolved_below_power`; no ATS read exists (rotation pools exhausted).

---

## 1. What this slice asks, and what it does not

Whether splitting the pooled offense/defense APM into units recovers
reliable per-player effects the pooled fit smears — the "units" half of
PER-09's remainder (hierarchy stays queued). It does NOT score ATS, build
season-lagged ratings, or wire anything.

## 2. Units (frozen)

Roster position per player-season (modal position in `weekly_rosters`,
20260817T184901Z snapshot) mapped once:

- `OFF_OL`: C, G, T, OT, OG, OC, OL
- `OFF_SKILL`: QB, RB, FB, HB, WR, TE
- `DEF_FRONT`: DE, DT, NT, LB, OLB, ILB, MLB, DL, EDGE, LDE, RDE, LDT, RDT
- `DEF_SECONDARY`: CB, S, SS, FS, DB, NB, LCB, RCB, SAF

Offense-side units draw from `offense_players`, defense-side from
`defense_players`. Unmapped/missing roster positions are excluded and
counted (never guessed). A player whose roster position changes across
seasons belongs to each season's unit for that season's plays (disclosed —
position switches are real, not errors).

## 3. Population and model (frozen)

Participation snapshot `20260813T131635Z`, PBP snapshot `20260817T184927Z`,
seasons 2019–2024. Plays: the valid competitive 11-on-11 scrimmage table
(the SAME `build_participation_play_table` the pooled screen uses —
identical population, so unit-vs-pooled comparisons are apples-to-apples).
Four separate Ridge fits (alpha 1000, `lsqr`, intercept, EPA clip ±5.0),
one per unit, each with its own team effects at scale 11.0: a unit fit sees
ONLY its unit's players (plus team effects), so an OL fit never borrows a
WR's coefficient.

## 4. Reliability gate (frozen, per unit)

Odd- vs even-week split fits per unit; correlate per-player coefficients
(Pearson + Spearman), Spearman-Brown corrected; eligibility ≥50 unit-plays
per half (same floor as the ST gate). Deterministic by construction
(`lsqr`, calendar halves); tests pin exact reproduction.

- Healthy unit (clearly above zero): `unresolved_below_power`, one registry
  entry per unit (units `correlation`).
- Dead unit (≈ zero): `no_split_half_reliability` admissible FOR THAT UNIT
  (record `refuted_mechanism` naming the unit; other units unaffected).

## 5. Positive control (frozen, diagnostic only)

Pooled offense/defense split-half reliability recomputed on the identical
population must come back positive (the pooled trait is known-real from the
prior screen's construction; if it does not, the harness — not the units —
is broken and the run is void). No closing eligibility.

## 6. Leakage (frozen)

No pregame application in this slice — descriptive season fits only, never
joined to a game table. Source seasons ≤ 2024 asserted fail-closed.
Tests pin the roster-position mapping and the per-unit player restriction
(a skill player can never enter the OL design) on synthetic rows.

## 7. Test contract (release-blocking)

`tests/test_unit_apm_screen.py` covers, without network access: unit mapping
(including position switches across seasons and unmapped exclusion),
per-unit design restriction, determinism, split-half helper, the 50-play
floor, and the pooled-control computation.

## 8. Decision rule (frozen)

Up to four `unresolved_below_power` entries (`unit_apm_<unit>_reliability`,
units `correlation`, `--league nfl`, 2019–2024) and/or per-unit
`refuted_mechanism` entries with `--closing-ground no_split_half_reliability`.
No card, model, profile, rotation window, or ATS comparison under any outcome.

## 9. Results (added after the run, 2026-09-03)

Measured by `scripts/unit_apm_screen.py` in per-unit runs against the
identical play population (unit artifacts `artifacts/unit_apm/20260903T202022Z/`
OFF_SKILL, `20260903T202044Z` OFF_OL, `DEF_FRONT/`, `DEF_SECONDARY/` —
chunked only because one process exceeds the runner's time budget; each run
executes frozen cells verbatim, and this section concatenates the four
blocks mechanically with no recomputation or selection):

| Unit | n (50+/half) | Split-half Pearson | Spearman | Spearman-Brown |
|---|---|---|---|---|
| OFF_SKILL | 840 | +0.245 | +0.209 | 0.394 |
| OFF_OL | 501 | +0.194 | +0.183 | 0.325 |
| DEF_FRONT | 814 | +0.134 | +0.107 | 0.236 |
| DEF_SECONDARY | 579 | +0.139 | +0.131 | 0.245 |

Unmapped roster positions (excluded, counted): K 11,971 / P 10,800 /
LS 10,635 (specialists, by design) plus KR 23 / PR 4.

What this implies for the decision, before what is wrong with it: every
unit carries a real, modestly reliable per-player trait — including the
offensive line (SB 0.325), which the pooled bundle never isolated — so no
unit meets `no_split_half_reliability` and none is refuted. What is wrong
with it: reliabilities are moderate (0.24–0.39, well below a scoutable
star-player signal), and no ATS read exists (rotation pools exhausted).
Per frozen §8: four `unresolved_below_power` records, season-lagged
unit-builder + ATS look queued, hierarchy still queued, nothing wired.

## Registry records

Recorded via `nfl-ats weak-signals record` (commands and outputs pasted in
the session report): `unit_apm_off_skill_reliability`,
`unit_apm_off_ol_reliability`, `unit_apm_def_front_reliability`,
`unit_apm_def_secondary_reliability` — units `correlation`,
`unresolved_below_power`, `--league nfl`, seasons 2019–2024.
