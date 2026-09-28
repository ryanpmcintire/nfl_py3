# Special-teams player ratings: reliability gate (predeclaration)

**Status:** predeclared 2026-09-03, BEFORE any special-teams rating or
reliability number was computed. Sections 0–8 are frozen; section 9
(Results) is appended after the run and nothing above it is edited
afterwards.

**Owning work package:** PER-09 (latent player ratings: hierarchy, units,
and special teams remain). Files: this document,
`scripts/st_player_ratings_screen.py`,
`tests/test_st_player_ratings_screen.py`,
`artifacts/st_player_ratings/`.

**Parent:** the season-lagged offense/defense APM (`src/nfl_ats/participation.py`,
failed its matched ATS screen; retained). This slice reuses its recipe
(Ridge alpha 1000, 500-play reliability prior, EPA clip 5.0, team effects)
and modifies none of its code — the ST fit lives in its own script and
imports the shared pieces.

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

This slice measures a trait's reliability, not an ATS effect. A reliability
indistinguishable from zero IS the admissible `no_split_half_reliability`
ground (no sample size rescues pure noise). A healthy reliability with no ATS
read is `unresolved_below_power`, never a promotion.

---

## 1. What this slice asks, and what it does not

Whether special-teams participants carry a stable, measurable per-player
effect — the minimum gate before any ST rating can be proposed as a feature.
It does NOT score ATS (rotation pools are exhausted; prospective 2026 is the
eventual judge), does NOT build season-lagged ratings, and wires nothing.

## 2. Population (frozen)

Participation snapshot `data/players/participation/raw/20260813T131635Z`,
seasons 2019–2024, plays whose EITHER personnel string carries a `1 K`,
`1 P`, or `1 LS` token (field-goal/XP, punt, and kickoff units by
construction; returners are whoever lines up opposite). Sub-units are NOT
split: one coefficient per player across all ST units (disclosed limitation
— a gunner who also returns blends both roles).

Validity (frozen): all 22 listed IDs present and unique per side, EPA
joinable from the PBP snapshot (`data/pbp/raw/20260817T184927Z`) by
game/play/season inner join, EPA clipped at ±5.0. No competitive filter
(punts and kicks are real plays in all game states — disclosed choice, not
an oversight). Design-time count (eligibility only, no outcome viewed):
7,709 ST plays in 2023 (~1/6 of the snapshot); six seasons ≈ 46k plays.

## 3. Model (frozen, borrowed without retuning)

Ridge alpha 1000 (`lsqr`, intercept), design = one `st_player::<gsis>`
one-hot per participant (+1 possession-team side, −1 other side) plus
`st_team::<abbr>` effects at scale 11.0, EPA from the kicking/possession
perspective as stored. Team effects absorb team strength exactly as the
offense/defense recipe's team features do. 500-play reliability prior is
REPORTED alongside raw coefficients but the reliability gate in §5 uses raw
split-half correlation, not the shrunk values.

## 4. Reliability gate (frozen)

Odd- vs even-week split: fit the identical model on odd-week and even-week
source plays separately, correlate per-player coefficients across halves
(Pearson + Spearman), Spearman-Brown corrected. Eligibility for the
correlation: players with ≥50 ST plays in EACH half (predeclared floor —
below it, single-half estimates are noise by construction and excluded,
counted). Seeds: the halves are calendar-defined (no RNG); the only
stochastic piece is none — Ridge `lsqr` is deterministic, so determinism
is by construction and the test pins exact reproduction instead.

- Gate outcome A (healthy, r clearly above zero): the ST trait is real;
  record `unresolved_below_power`, queue the season-lagged builder + ATS
  look for when windows open.
- Gate outcome B (≈ zero): `no_split_half_reliability` is admissible —
  record `refuted_mechanism` for the ST-APM trait as specified (single
  pooled coefficient per player). A unit-split reformulation would be a NEW
  predeclaration, not a rescue of this one.

## 5. Positive control (frozen, diagnostic only)

Position-group means must separate sanely (kickers/punters/long-snappers
are not asked to carry return value — report group means by roster position
from the weekly rosters as description). No closing eligibility.

## 6. Leakage (frozen)

No pregame application exists in this slice — coefficients are descriptive
fits on completed seasons, never joined to a game table. The script asserts
all source seasons ≤ 2024 (strictly before any 2025+ target the future
builder could score). Tests pin the personnel-token classifier and the
22-unique-ids validity gate on synthetic rows.

## 7. Test contract (release-blocking)

`tests/test_st_player_ratings_screen.py` covers, without network access:
token classification (FG vs punt vs kickoff vs scrimmage), validity gates
(duplicates, missing ids, EPA-join miss), determinism (two runs bit-identical),
split-half helper on a synthetic two-half frame, and the ≥50-play eligibility
floor.

## 8. Decision rule (frozen)

Record one entry (`st_player_rating_reliability`, units `correlation`,
`--league nfl`, seasons 2019–2024): the Spearman-Brown reliability with its
n, plus Pearson. Outcome A → `unresolved_below_power`. Outcome B →
`refuted_mechanism` with `--closing-ground no_split_half_reliability`.
No card, model, profile, rotation window, or ATS comparison under any outcome.

## 9. Results (added after the run, 2026-09-03)

Measured by `scripts/st_player_ratings_screen.py` in one run (artifact
`artifacts/st_player_ratings/20260903T200725Z/results.json`):

- **14,686 valid ST plays** (2019–2024): FG/XP 4,597, kickoff 4,704, punt
  4,560, other-ST 825. Personnel-token classification agrees with PBP
  `play_type` on **96.2%** of classifiable rows (FG 4,417/4,597, kickoff
  4,691/4,704, punt 4,228/4,560) — the token rule is validated against an
  independent field, diagnostic only.
- **Reliability (frozen gate):** odd/even-week split-half Pearson **+0.279**,
  Spearman +0.276, Spearman-Brown **0.436**, n = **996** players with 50+
  plays per half, from 3,020 rated participants.
- What this implies for the decision, before what is wrong with it: the ST
  trait is real but modest — clearly above zero, well below the
  offense/defense constructs' reliability, so a future season-lagged ST
  builder is worth attempting when rotation windows open, and nothing is
  refuted. What is wrong with it: one pooled coefficient blends kicking,
  punting, and coverage roles by declared design; the EPA target mixes unit
  types; no ATS read exists (rotation pools exhausted, prospective 2026 the
  eventual judge).

Per frozen §8 this is outcome A: record `unresolved_below_power`, queue the
season-lagged builder + ATS look, wire nothing.

## Registry record

Recorded via `nfl-ats weak-signals record` (command and output pasted in the
session report): `st_player_rating_reliability`, units `correlation`,
`unresolved_below_power`, `--league nfl`, seasons 2019–2024.

## 10. Future season-lagged builder design (added 2026-09-28)

The offline builder produces one pooled special-teams player rating for each
target season from exactly the prior three complete seasons. A target season
is rejected unless all three required source seasons are present. The fit never
uses target-season plays. For the frozen 2019–2024 source population, the
declared targets are 2022–2025, using 2019–2021 through 2022–2024 respectively.
The builder retains the historical screen's regular-season-only PBP scope;
postseason plays are excluded.

The population and model remain the frozen recipe above: either personnel
string carries a `1 K`, `1 P`, or `1 LS` token; each side supplies 11 unique
player IDs; one pooled player coefficient spans all special-teams units; Ridge
alpha is 1000; the possession-team feature scale is 11; EPA is clipped at 5;
and each coefficient is shrunk by its play count with a 500-play prior. Output
rows and metadata are canonical and sorted. The artifact retains the exact
declaration and source manifests and hashes them, and it records the source
partition hashes used by every fit.

The four target seasons are four fitted looks in the
`st_player_rating_season_lagged_builder` family. This construction performs
zero ATS outcome comparisons and makes zero promotion or serving decisions.
The ratings remain unserved. Any later ATS evaluation requires its own declared
chronological design, and one fitted calibrated probability must select every
served side.


## 11. Season-lagged construction result, 2026-09-28

**Measured:** `scripts/build_st_player_ratings.py` completed the four declared
fits with the unchanged recipe, producing **7,413 player-season rows** in
`artifacts/st_player_ratings/season_lagged_2019_2024/ratings.parquet`.

| Target season | Prior source seasons | Eligible ST plays | Rated players |
|---|---|---:|---:|
| 2022 | 2019–2021 | 141 | 1,377 |
| 2023 | 2020–2022 | 131 | 1,389 |
| 2024 | 2021–2023 | 7,328 | 2,159 |
| 2025 | 2022–2024 | 14,545 | 2,488 |

**Measured:** the two disjoint 2019–2021 and 2022–2024 windows together contain
14,686 eligible plays. The early windows contain only 141 and 131 plays: source
year presence does not establish dense special-teams coverage. Every declared
fold is retained. These construction counts establish no ATS effect, stability,
or promotion decision; the existing research line remains unresolved.

**Measured:** independent artifact verification checked all 12 source partition
hashes and row counts, retained declaration/source manifests, implementation
hashes, output schema/hash, unique target/player keys, strict source chronology,
and 22 player exposures per eligible play. The run records four fitted looks in
`st_player_rating_season_lagged_builder`, zero ATS comparisons, and zero promotion
decisions. The artifact's `declaration.md` preserves the exact pre-fit sections
0–10; `manifest.json` binds it to the inputs, code, and ratings.

Reproduction command (choose a new output directory; existing output is refused):

```powershell
.tools/uv.exe --cache-dir .tmp/uv-cache run --no-sync python scripts/build_st_player_ratings.py --participation-snapshot data/players/participation/raw/20260813T131635Z --pbp-snapshot data/pbp/raw/20260817T184927Z --declaration artifacts/st_player_ratings/season_lagged_2019_2024/declaration.md --target-seasons 2022 2023 2024 2025 --output artifacts/st_player_ratings/season_lagged_reproduction
```

The builder implementation is complete. A later ATS evaluation still needs its
own declared chronological design and coverage accounting. Ratings remain
unserved; no weak-signal verdict, rotation verdict, card, or model was changed.
