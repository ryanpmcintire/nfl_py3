# Era-weighted refit (half-life 8) scored THROUGH the played card

Predeclared 2026-09-09, **before** `scripts/era_weighted_on_card.py` was run
and before any number in the Results section existed. Provenance tags:
**measured** (run this session, command/path given), **read** (file opened
this session), **reported** (another doc's claim, unverified here),
**inferred** (reasoning, not evidence).

## Binding closing-grounds taxonomy (pasted verbatim, per AGENTS.md)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (a) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (b) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it, report
> `probability_positive`, never "contains zero".

Within-week correlation is ZERO (owner mandate); the week-blocked bootstrap
is the primary uncertainty and no ICC is estimated or padded. Era readings
differ in MAGNITUDE; a weaker era is never absence. The decision is expected
value: forced picks, the card with the higher expected opener accuracy is
played. A 0.90-style threshold governs only what these docs may CLAIM.

## 1. The gap this fills

`era_weighted_half_life_8` (**read**,
`src/nfl_ats/era_weighted_half_life_8_overlay.py:1-108`) is a refit of the
active recipe with per-training-row sample weights decaying at an 8-season
half-life. It is registered `ACTIVE_PROSPECTIVE` in
`artifacts/prospective/challengers.json` (**read**), and it has four prior
looks, all `unresolved_below_power` (**reported**, from
`docs/era_weighting_screen.md` and `docs/era_weighting_promotion.md`, not
re-verified here):

| Look | Instrument | Effect (accuracy points) | P+ |
| --- | --- | ---: | ---: |
| 1 | CFB clean core, week-blocked, n=8,933 | +0.3470 | 0.8987 |
| 2 | NFL CLOSE grade, week-blocked, n=2,047 | +0.6839 | 0.8505 |
| 3 | NFL OPENER information read, week-blocked, n=1,503 | -0.3992 | 0.2990 |
| 4 | NFL OPENER confirmation, [2020, 2021], n=456 | -0.2193 | 0.4246 |

**Every one of those four looks scores the arm STANDALONE, on the bare
model.** None of them has ever been scored through the card that is actually
played. That matters here for the same reason `docs/unserved_tilt_marginals.md`
(**read**) gave for the tilt overlays: the served three-member joint-OR card
already moves 279 of 1,503 picks, so a change to the underlying model can
either add to or be absorbed by flips the card already makes. A standalone
number is not the decision number.

Looks 3 and 4 also predate the current served instrument: they used the
model-default ECDF probability read on the 1,537-game archive, whereas the
active model `c657058903f3232b` (**read**, `artifacts/active_ats_model.json`)
serves `gaussian_median` and, since 2026-09-07, a fitted home-side offset. So
this look is not a re-run of look 3 with a different aggregation; it is the
first read of this arm on the instrument that is currently served.

## 2. Population, instrument, and what is NOT in it

- **Archive**: `artifacts/opener_evaluation/20260909T183120Z` -- the 2020-2025
  Tuesday-opener archive of the ACTIVE model (**read**,
  `artifacts/active_ats_model.json` names model `c657058903f3232b`;
  `artifacts/overlay_subset_composition/20260909T222214375455Z/result.json`
  names the same model on 1,537 games / 34 pushes / **1,503 scored games**).
- **Harness**: `scripts/spread_hole_arms.py`'s `walk_forward` (**read**),
  extended with a sample-weight path. That harness replays the served
  incumbent on this archive with a residual gap of **0.0** on all 1,537 rows
  (**read**,
  `artifacts/spread_hole_diagnosis/20260909T212843Z/arms/arm_results.json`),
  which is why it is the harness used rather than a fresh reimplementation.
- **Grade**: Tuesday opener (`tue_open`), production probability rule
  (`home_cover_probability >= 0.5`), `gaussian_median`, with the served
  per-arm home-side offset fitted forward-only from that arm's own prior
  weeks. Pushes do not score, so n = 1,503.
- **Card**: the served three-member joint-OR overlay card -- `coach_fade_overlay`
  OR `division_revenge_tilt_overlay` OR `player_arrests_back_side_policy`
  (**read**, `scripts/spread_hole_arms.py:46-50`; the same three members the
  latest composition selected, **read**,
  `artifacts/overlay_subset_composition/20260909T222214375455Z/result.json`).
- **Stated limitation, not smoothed over**: any tilt promoted onto the LIVE
  card during today's sessions beyond those three members is **not** in this
  archive harness. The archive card here is exactly the three-member opener
  composition plus the served home-side offset; the six tilts served on
  today's live card are not represented, so this measurement answers "does
  the era-weighted refit help through the three-member archive card", not
  "through every rule served this afternoon".

## 3. Arms

Every arm fits the frozen production recipe -- `weak_stack` features,
`ridge`, `ridge_alpha=10.0`, `market_residual` target, `min_train_games=500`,
forward-chained weekly refits on training rows strictly earlier than the
scored week's first gameday. **Only the per-row sample weight changes.**

- `incumbent` -- `nfl_ats.margin.fit_margin_model`, unweighted. This is the
  served model.
- `uniform` -- the challenger's own
  `era_weighted_half_life_8_overlay.fit_weighted_ridge_margin` with
  `sample_weight = 1` for every row. **A gate, not a result** (see 4).
- `hl4` / `hl8` / `hl16` -- the same weighted fitter with
  `era_weighted_half_life_8_overlay.half_life_weights`, half-lives 4, 8 and
  16 seasons: weight `0.5 ** ((predict_season - row_season) / h)`, season
  granularity, elapsed clamped at zero. `hl8` is the registered challenger's
  exact definition; `hl4` and `hl16` are **sensitivity, declared as
  sensitivity and not selection** -- the promoted-or-not decision is read off
  `hl8` alone, and the two flanking half-lives exist only to say whether the
  answer is a knife-edge in the tuning parameter. No half-life will be
  chosen by looking at these three numbers.

## 4. The reproduction gate, run before any weighted number is read

The `uniform` arm must reproduce the `incumbent` arm's `residual_at_open` and
`home_cover_probability_at_open` to `atol=1e-9` on every one of the 1,537
archive rows, and the `incumbent` arm must reproduce the archive itself
(`max_residual_gap`, `max_probability_gap`). This is the challenger module's
own verify-reproduction-then-swap-one-thing discipline (**read**,
`src/nfl_ats/era_weighted_half_life_8_overlay.py:74-83`) applied to the
harness. A miss is a bug in this script's adaptation, not a finding, and is
fixed before any half-life arm's numbers are interpreted.

## 5. Endpoints, declared before the run

Names are `era_weighted_hl<h>_<card|standalone>_<window>`.

- **Primary**: paired forced-pick opener accuracy improvement, candidate
  minus incumbent, on the **card** surface, overall (`window` = `overall`).
- **Secondary, same footing, reported in full whatever the sign**: the
  `standalone` surface (the served raw model, no overlays); the per-season
  windows 2020..2025; the spread-bucket windows `0-6.5`, `7`, `7.5-10`,
  `10.5+` (the coarse map already in `scripts/spread_hole_arms.py:51`);
  Brier and log loss on the served opener cover probability, both surfaces;
  and picks changed on every cut.
- **Uncertainty**: paired week-blocked and season-blocked bootstrap,
  20,000 samples, seed 20260821 -- the same machinery and the same seed the
  composition study and `spread_hole_arms.py` use. Season-blocked on a
  per-season window is one block and therefore degenerate; it is reported as
  degenerate, never as an interval.
- **Family declared before signs are seen**: the commensurable family for any
  pooling is `{hl8 card overall, hl8 standalone overall}` in
  `accuracy_points` on this archive. The per-season and per-bucket cells are
  decompositions of those same games and are recorded with the overlap
  stated; they are not independent inputs and must not be pooled as such.

## 6. Classification rule, mechanical, decided by reading the artifact

- Whole week-blocked primary interval below zero -> `refuted_mechanism` with
  `--closing-ground wrong_sign_resolved`.
- No positive control is run anywhere in this document, so
  `bounded_by_control` is unavailable by construction.
- Everything else, including a wholly-positive interval ->
  `unresolved_below_power`, reported with `probability_positive`.
- If a `weak-signals record` call errors, the verdict is wrong, not the
  validator: reclassify as `unresolved_below_power`.

## 7. Decision rule, stated before the numbers

The pool is forced picks. The decision is expected value at the opener,
through the card: if the `hl8` card-surface point estimate is positive, the
honest statement is "playing the era-weighted card is the better side of a
P+ bet", and the write-up says exactly what promoting it would take and which
Week 1 picks move. If it is negative, the same logic runs the other way and
the arm is not proposed for the card -- and either way the result is recorded
as `unresolved_below_power` unless the mechanical rule in Section 6 fires.
**No promotion is performed by this document under any outcome**; the active
manifest, the ledgers, the forecast and the board are not touched.

## 8. Commands

```
uv run --no-sync python scripts/era_weighted_on_card.py \
  --features data/processed/game_features_weak_stack.parquet \
  --data-root data --market-root data/market/raw \
  --archive artifacts/opener_evaluation/20260909T183120Z \
  --out artifacts/era_weighted_on_card/20260909T231403Z
```

## Results

**Measured** 2026-09-09, `scripts/era_weighted_on_card.py`, artifact
`artifacts/era_weighted_on_card/20260909T231403Z/`. Command in Section 8.

### 9.1 Both gates pass exactly, before any weighted arm was read

- Incumbent vs. the archive: 1,537 rows, `max_residual_gap` **0.0**,
  `max_probability_gap` **0.0**.
- Uniform-weight refit through the challenger's own
  `fit_weighted_ridge_margin` vs. incumbent: 1,537 rows, both gaps **0.0**,
  `passes_atol_1e_9: true`. Their served three-member flip sets are both
  **279** games, the same number `docs/unserved_tilt_marginals.md` records for
  the served card.

So every difference below is the sample weight and nothing else.

### 9.2 The decision number

Paired forced-pick accuracy, candidate minus incumbent, at the Tuesday
opener, 1,503 non-push games, 107 week blocks, 6 season blocks, 20,000
bootstrap samples, seed 20260821.

| Name | Card % (base -> cand) | Picks changed | Effect (pts) | Week 95% | P+ (week) | Season 95% | P+ (season) |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| `era_weighted_hl8_card_overall` | 55.888 -> 54.158 | 104 | **-1.730** | **[-3.112, -0.330]** | **0.0078** | [-2.550, -1.105] | 0.0000 |
| `era_weighted_hl8_standalone_overall` | 54.558 -> 52.894 | 157 | -1.663 | [-3.363, +0.066] | 0.0277 | [-3.204, -0.543] | 0.0000 |
| `era_weighted_hl4_card_overall` | 55.888 -> 54.225 | 163 | -1.663 | [-3.371, 0.000] | 0.0278 | [-2.623, -0.554] | 0.0028 |
| `era_weighted_hl4_standalone_overall` | 54.558 -> 52.828 | 246 | -1.730 | [-3.844, +0.397] | 0.0543 | [-3.692, +0.133] | 0.0336 |
| `era_weighted_hl16_card_overall` | 55.888 -> 54.291 | 50 | -1.597 | [-2.449, -0.736] | 0.0002 | [-2.219, -0.950] | 0.0000 |
| `era_weighted_hl16_standalone_overall` | 54.558 -> 52.961 | 88 | -1.597 | [-2.867, -0.337] | 0.0060 | [-2.628, -0.479] | 0.0047 |

`hl4` and `hl16` are the declared sensitivity flanks, not selections. Nothing
was chosen by looking at them; they say the answer is not a knife-edge in the
half-life -- every half-life tested, at every surface, lands in the same
place, and the gentlest one (50 changed picks in six seasons) is the most
resolved of all.

### 9.3 Where it goes wrong (`hl8`, by spread bucket)

| Name | n | Card % (base -> cand) | Picks changed | Effect (pts) | Week 95% | P+ |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `era_weighted_hl8_card_0-6.5` | 1104 | 57.609 -> 56.431 | 77 | -1.178 | [-2.770, +0.450] | 0.0775 |
| `era_weighted_hl8_card_7` | 74 | 50.000 -> 50.000 | 4 | 0.000 | [-5.479, +5.405] | 0.4959 |
| `era_weighted_hl8_card_7.5-10` | 194 | 51.546 -> 46.907 | 17 | **-4.639** | **[-9.360, -0.481]** | 0.0182 |
| `era_weighted_hl8_card_10.5+` | 131 | 51.145 -> 48.092 | 6 | -3.053 | [-6.667, 0.000] | 0.0410 |

Standalone, the same shape is sharper: `7.5-10` reads -6.701 pts,
[-12.042, -1.932], P+ 0.0028 on 194 games, while the exactly-7 bucket reads
**+8.108** pts, [0.000, +16.667], P+ 0.9779 on 74 games with 10 picks changed
-- the one cut anywhere in this study that leans toward the candidate, and it
disappears entirely once the card is applied (0.000, P+ 0.4959). It is a
74-game decomposition of a losing whole and is recorded as such, not as a
finding to build on.

This is a diagnosis to publish, not a threshold to bolt on: the buckets say
the recency-weighted refit's damage is concentrated where the incumbent is
already weakest (7.5-10 and 10.5+, the same zone
`docs/spread_hole_diagnosis.md` is about), which is the opposite of what a
useful fix would do.

### 9.4 By season (`hl8`, card surface)

Season-blocked uncertainty is one block per row and therefore degenerate; the
week-blocked interval is the readable one. Magnitudes differ by era; no season
is an absence.

| Season | n | Card % (base -> cand) | Picks changed | Effect (pts) | Week 95% | P+ |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| 2020 | 220 | 55.455 -> 53.636 | 10 | -1.818 | [-5.217, +1.818] | 0.1491 |
| 2021 | 236 | 54.237 -> 52.966 | 11 | -1.271 | [-5.021, +2.165] | 0.2427 |
| 2022 | 248 | 58.065 -> 54.435 | 13 | -3.629 | [-6.800, -0.787] | 0.0084 |
| 2023 | 266 | 57.895 -> 56.767 | 27 | -1.128 | [-3.817, +1.859] | 0.2172 |
| 2024 | 266 | 53.008 -> 51.128 | 19 | -1.880 | [-4.851, +1.099] | 0.1054 |
| 2025 | 267 | 56.554 -> 55.805 | 24 | -0.749 | [-5.243, +3.690] | 0.3740 |

Six of six seasons lean negative; the smallest lean is 2025 (-0.749,
P+ 0.3740), the largest 2022 (-3.629, P+ 0.0084).

### 9.5 Brier and log loss

| Metric | Baseline | Candidate | Improvement | Week 95% | P+ |
| --- | ---: | ---: | ---: | --- | ---: |
| Brier, `hl8` card | 0.248772 | 0.249545 | -0.000773 | [-0.001909, +0.000329] | 0.0895 |
| Log loss, `hl8` card | 0.690802 | 0.692373 | -0.001571 | [-0.003900, +0.000684] | 0.0914 |
| Brier, `hl8` standalone | 0.251581 | 0.251749 | -0.000168 | [-0.001239, +0.000902] | 0.3826 |
| Log loss, `hl8` standalone | 0.696618 | 0.696979 | -0.000362 | [-0.002551, +0.001838] | 0.3779 |

**The accuracy-versus-calibration divergence that softened the four earlier
looks is not here.** In `docs/era_weighting_screen.md` Section 8 and
`docs/era_weighting_promotion.md` the continuous metrics leaned toward the
candidate while accuracy leaned against it; on this instrument both lean the
same way, against it.

### 9.6 Classification, read off the artifact

Applying Section 6's mechanical rule, unchanged:

- `era_weighted_hl8_card_overall` (PRIMARY): week-blocked interval
  [-3.112, -0.330] sits **entirely below zero** -> **`refuted_mechanism`**,
  `--closing-ground wrong_sign_resolved`. This is admissible ground (a), a
  RESOLVED wrong sign, and it is the one thing that closes a line of work
  under the taxonomy. It closes the half-life-8 refit **as a replacement for
  the served model on the played card at the opener** -- not era weighting on
  the CFB instrument, not at the close grade, not as a research direction.
- `era_weighted_hl16_card_overall`, `era_weighted_hl16_standalone_overall`,
  `era_weighted_hl8_card_7.5-10`: same ground, same reading.
- Everything else, including `era_weighted_hl8_standalone_overall` (upper
  bound +0.066) and `era_weighted_hl4_card_overall` (upper bound exactly
  0.000): **`unresolved_below_power`**, reported with
  `probability_positive`. An upper bound that touches zero is not below it.

Twelve cells recorded with `nfl-ats weak-signals record`, one call at a time,
family `era_weighting_half_life_on_card`; argv and return codes in
`artifacts/era_weighted_on_card/20260909T231403Z/record_commands.json`, all
`rc=0`, all read back from `registry/weak_signals.json`.

### 9.7 The decision

**Do not promote, and do not keep spending sessions on this arm as a card
candidate.** Forced picks, expected value at the opener, through the card:
the era-weighted card is **1.73 accuracy points worse** than the card that is
played, and the probability it is better is **0.0078**. Declining it is not
caution -- taking it would be the 1-in-128 side of the bet.

What promoting it WOULD have taken, for the record, since the question was
asked: refit weights inside `weekly-run`'s model fit (the challenger's
`half_life_weights` applied to `fit_margin_model`'s Ridge step), a fresh
`opener-evaluation` and `overlay-composition` under the new model id, a new
`artifacts/active_ats_model.json` (model id, feature-table digest,
`historical_evaluation`), then `publish-predictions` and `publish-board`.
**None of that was done. Nothing was promoted, no ledger, manifest, forecast
or board was touched.**

Which Week 1 picks it would have moved (**measured**, read-only, active model
`c657058903f3232b`, forecast `2026-week-01-20260909T220047Z`, uniform-weight
reproduction gate passed inside
`apply_era_weighted_half_life_8_overlay`): `hl8` moves **3 of 16** model
picks, all of them by a hair --

| Game | Model p(home), served -> era-weighted | Model pick | Published pick today |
| --- | --- | --- | --- |
| CHI at CAR | 0.5164 -> 0.4995 | CAR -> CHI | CAR +2.5 |
| NO at DET | 0.5225 -> 0.4990 | DET -> NO | NO +6.5 (rule-flipped) |
| NYJ at TEN | 0.4843 -> 0.5022 | NYJ -> TEN | NYJ +1.5 |

`hl4` moves the same three games; `hl16` moves **none**. All three land within
0.006 of a coin flip after the reweighting, which is what a 1.73-point loss
bought. **Inferred, not measured:** if the served rules' flip set were
unchanged on those three games, the published card would read CHI, DET and
TEN instead of CAR, NO and NYJ -- but the archive shows the flip set is not
strictly arm-independent (279 flips under the incumbent, 287 under `hl8`), so
that last step is reasoning, not a measurement.

### 9.8 Stated limitation

The card measured here is the served **three-member** opener composition. The
live Week 1 card serves **nine** rules -- those three plus bye edge fade,
forecast cold visitor tilt, pbp08 protection mismatch tilt, interim HC first
game tilt, tank zone fade tilt and precip high total tilt (**read**,
`CURRENT_PREDICTIONS.md`). Those six tilts are not in this archive harness, so
this answers "through the three-member archive card", not "through every rule
served today". Given the standalone surface, the card surface and all three
half-lives agree on sign and magnitude, I do not expect the six extra tilts to
reverse it -- but that is **inferred**, and it is the one thing a follow-up
would measure.
