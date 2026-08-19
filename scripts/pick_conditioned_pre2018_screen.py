"""Pick-conditioned replication screen: do four error-analysis-mined,
pick-conditioned bucket leads on the ACTIVE ``weak_stack``/``market_residual``
model replicate on 2011-2017 walk-forward picks -- seasons that error
analysis never touched?

**Lead-gen replication, not a rotation-registry confirmation.** No
rotation-registry window is declared or spent by this screen; findings
record straight to ``registry/weak_signals.json``, the same convention
``scripts/nfl_bias_battery_screen.py`` and
``scripts/odds_microstructure_battery.py`` already use for their
predeclared batteries (both propose, rather than execute, their
``nfl-ats weak-signals record`` commands; this screen does the same).

CLOSING-GROUNDS TAXONOMY (binding, pasted verbatim per AGENTS.md/CLAUDE.md
-- any process that runs, scores, or adjudicates an experiment must repeat
this, not just cite it):

    An interval or CI that contains zero is NEVER grounds to reject, fail,
    or close an experiment. At this evaluator's ~2-point resolution,
    "contains zero" is the EXPECTED outcome for a real small signal. Only
    two grounds ever close a line of work: (1) refuted mechanism -- a
    RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
    split-half reliability; (2) bounded by a positive control proven able
    to detect an effect that size. Everything else is
    unresolved_below_power: record it with ``nfl-ats weak-signals record``,
    report probability_positive, never the binary "contains zero". The
    registry code hard-rejects inadmissible closures; if a record command
    errors, the verdict is wrong, not the validator.

Mined leads (**reported**, source: error analysis of the ACTIVE model
written this session -- 2020-2025 opener / 2018-2025 close grade, both
MINED, lead-gen only, never a predeclared look):

    1. pick-is-road-favorite: 55.39% opener (n=482) / 55.75% close (n=687).
    2. rest-mismatch: 50.68% (unequal) vs 54.77% (equal) opener; 49.46% vs
       52.73% close.
    3. picked-team-off-own-bye: 40.74% opener (n=54) / 43.96% close (n=91).
    4. abs(spread) 7.5-10 gap zone: 45.96% opener (n=198) / 47.97% close
       (n=271).
    (A fifth mined bucket, opener/close pick-flip games, is not testable
    here at all: no Tuesday-opener archive exists before 2020, so it is
    skipped rather than silently mis-scored.)

This script fits the production ``weak_stack``/``market_residual`` config
(**read** 2026-08-19 from the active model manifest: ``feature_profile``
weak_stack, ``method`` market_residual, ``regressor`` ridge,
``ridge_alpha`` 10.0) walk-forward on 2011-2017 with
``nfl_ats.outcomes.walk_forward_outcomes`` -- the identical invocation
pattern ``nfl_ats.experiment_runner.run_feature_arm_experiment`` uses for
its single ``_arm_predictions`` call, read there for the pattern but not
imported (this screen has no baseline/candidate pair, only one arm) -- and
re-buckets the resulting picks by the same four pick-conditioned
definitions, close grade only (``feature_arm`` itself supports no other
grade for the same reason: no opener archive covers 2011-2017). Production
pick rule: ``home_cover_probability >= 0.5`` (``pool.py``'s actual rule,
not the retired opener artifact's sign rule). Same-direction leans on this
never-mined window would be real measurement; coin flips would say the
mined buckets were noise.

Warm-up floor: scoring starts at season 2011
(``nfl_ats.rotation.MIN_ELIGIBLE_START_SEASON``, **read** = 2009 feature
table start + 2 warm-up seasons), with training strictly prior to each
scored week enforced by ``walk_forward_outcomes`` itself (cutoff = the
week's earliest kickoff; only completed games before that cutoff train).

Overlap disclosure (rule-4-overlap precedent): seasons 2011-2017 overlap
rotation-registry windows already spent by OTHER families -- [2013, 2015],
[2013, 2017], [2014, 2017]. Windows retire PER-FAMILY, not globally
(AGENTS.md / "Opener windows are not scarce"), and this screen spends no
window of its own, so the overlap does not block it; it is disclosed here
and in every recorded description for the next reader.

Bootstrap: a joint week-blocked (primary) + season-blocked (secondary)
block bootstrap of ``100 * (bucket_mean - complement_mean)`` on the
``correct`` (0/1) column, both arms resampled from the SAME drawn block
set each draw -- a local reimplementation of
``scripts/nfl_bias_battery_screen.py``'s ``block_bootstrap_two_group`` /
``nfl_ats.experiment_runner``'s ``_block_bootstrap_subset_gap`` (same
``np.unique`` block-id derivation, same single ``rng.multinomial`` draw
shared by both arms, same bincount sums/counts trick). Reimplemented
locally rather than imported so this screen carries no runtime dependency
on ``experiment_runner``'s private internals, which other sessions are
actively editing this pass; the one PUBLIC function this screen does
import from there, ``classify_subset_bias_result``, is the project's own
canonical mechanical classifier -- reusing it (read-only) is safer than a
second hand-rolled copy of the exact widening-factor arithmetic AGENTS.md
already warns is a sign-bug hazard.

Effect convention (**NOT full-slate scaled -- these are conditional-
accuracy reads**, not whole-slate promotion comparisons): each construct
reports both (a) the literal RAW gap, bucket accuracy minus complement
accuracy in accuracy points, and (b) a HYPOTHESIS-SIGNED value (raw gap *
+1 if the mined lead predicted the bucket would score HIGHER, * -1 if it
predicted LOWER) so that positive always means "replicates the mined
direction" -- the sign convention ``classify_subset_bias_result`` and
``weak_signals.WeakSignal.favours_candidate`` already assume. The
hypothesis-signed primary interval is what gets classified and recorded;
the raw gap is reported alongside for transparency.

Writes a predeclaration file to the scratch directory BEFORE any scoring
runs, and a results file after. Writes nothing under this repository's
generated-artifact directory and never touches ``registry/`` itself --
findings are proposed as ``nfl-ats weak-signals record`` commands on
stdout for the operator (or a wrapping session honoring the registry
write-lock protocol) to run.

Run:  .\\.tools\\uv.exe run --no-sync python scripts/pick_conditioned_pre2018_screen.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES
from nfl_ats.estimation_variance import MIN_BLOCKS_FOR_INTERVAL, guard_block_count
from nfl_ats.experiment_runner import classify_subset_bias_result
from nfl_ats.outcomes import walk_forward_outcomes

REPO = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
SCRATCH_DIR = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\26042060-ffd8-45a7-b2e7-a9b30b87bd34\scratchpad\agent_pre2018"
)

# Active model config, read 2026-08-19 from the active model manifest.
FEATURE_PROFILE = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
METHOD = "market_residual"

START_SEASON = 2011  # nfl_ats.rotation.MIN_ELIGIBLE_START_SEASON
END_SEASON = 2017

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260819  # fixed, this screen's own date-stamped seed

# Exact port of agent_errors/analyze_buckets.py's picked_team_off_bye flag:
# our own team is off its bye when rest_diff (home_rest - away_rest) favours
# the side we picked by at least this many days.
OFF_BYE_REST_DIFF = 6

# Exact port of close_grade_corroborate.py's spread_bin "7.5-10" bin:
# strictly above 7.0, at or below 10.0.
SPREAD_GAP_LOW = 7.0
SPREAD_GAP_HIGH = 10.0

_CLOSING_GROUNDS_TAXONOMY = (
    "An interval or CI that contains zero is NEVER grounds to reject, fail, or close an "
    'experiment. At this evaluator\'s ~2-point resolution, "contains zero" is the EXPECTED '
    "outcome for a real small signal. Only two grounds ever close a line of work: (1) refuted "
    "mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero "
    "split-half reliability; (2) bounded by a positive control proven able to detect an effect "
    "that size. Everything else is unresolved_below_power: record it with `nfl-ats weak-signals "
    'record`, report probability_positive, never the binary "contains zero". The registry code '
    "hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not "
    "the validator."
)


def _predeclaration() -> dict[str, Any]:
    return {
        "predeclared_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Replicate four pick-conditioned, mined error-analysis leads on 2011-2017 "
            "walk-forward picks from the ACTIVE weak_stack/market_residual model -- seasons "
            "the leads have never seen -- to see whether the pattern is a real (if small) "
            "signal or an artifact of the 2018-2025/2020-2025 mining window."
        ),
        "population": {
            "league": "nfl",
            "seasons": [START_SEASON, END_SEASON],
            "grade": "close",
            "model_config": {
                "feature_profile": FEATURE_PROFILE,
                "method": METHOD,
                "regressor": REGRESSOR,
                "ridge_alpha": RIDGE_ALPHA,
                "min_train_games": DEFAULT_MIN_TRAIN_GAMES,
                "pick_rule": "home_cover_probability >= 0.5 (production rule, pool.py)",
            },
        },
        "constructs": [
            {
                "key": "road_favorite",
                "registry_name": "pick_conditioned_road_favorite_pre2018",
                "bucket": (
                    "our_pick_side == 'AWAY' and our_pick_is_favorite -- the picked team is "
                    "the road favorite"
                ),
                "complement": "all other picks (home favorite, home dog, road dog)",
                "mined_direction": "higher",
                "mined_reference": (
                    "55.39% opener (n=482), 55.75% close (n=687) -- error analysis Q1, "
                    "pick-side-by-favorite interaction"
                ),
            },
            {
                "key": "rest_mismatch",
                "registry_name": "pick_conditioned_rest_mismatch_pre2018",
                "bucket": "rest_diff != 0 -- unequal rest between the two teams",
                "complement": "rest_diff == 0 -- equal rest",
                "mined_direction": "lower",
                "mined_reference": (
                    "50.68% (mismatch) vs 54.77% (equal) opener; 49.46% vs 52.73% close -- "
                    "error analysis Q1, rest_equal"
                ),
            },
            {
                "key": "off_bye_fade",
                "registry_name": "pick_conditioned_off_bye_fade_pre2018",
                "bucket": (
                    "(our_pick_home & rest_diff >= 6) | (~our_pick_home & rest_diff <= -6) -- "
                    "the picked team is off its own bye; exact port of "
                    "agent_errors/analyze_buckets.py's picked_team_off_bye"
                ),
                "complement": "everyone else (not picked-team-off-bye)",
                "mined_direction": "lower",
                "mined_reference": (
                    "40.74% opener (n=54), 43.96% close (n=91) -- error analysis Q5, post-bye"
                ),
            },
            {
                "key": "spread_gap_zone",
                "registry_name": "pick_conditioned_spread_gap_zone_pre2018",
                "bucket": (
                    "7.0 < abs(spread_line) <= 10.0 -- the '7.5-10' bin, exact port of "
                    "close_grade_corroborate.py's spread_bin"
                ),
                "complement": "abs(spread_line) <= 7.0 or abs(spread_line) > 10.0",
                "mined_direction": "lower",
                "mined_reference": (
                    "45.96% opener (n=198), 47.97% close (n=271) -- error analysis Q1, spread bins"
                ),
            },
        ],
        "method": {
            "bootstrap": (
                "joint week-blocked (primary) + season-blocked (secondary) block bootstrap of "
                "100*(bucket_mean - complement_mean) on the per-game 'correct' (0/1) column; "
                "both arms resampled from the SAME drawn block set each draw"
            ),
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "effect_convention": (
                "raw = bucket accuracy minus complement accuracy, accuracy points, NOT "
                "full-slate scaled (a conditional-accuracy read, not a whole-slate promotion "
                "comparison). signed = raw * hypothesis_sign, where hypothesis_sign is +1 if "
                "the mined lead predicted the bucket would be HIGHER, -1 if LOWER, so positive "
                "always means 'replicates the mined direction'."
            ),
            "classification": (
                "nfl_ats.experiment_runner.classify_subset_bias_result on the hypothesis-signed "
                "primary (week-blocked) interval -- the project's own mechanical taxonomy: only "
                "a resolved wrong sign (interval entirely on the anti-mined side of zero, "
                "widening factor to re-cross zero above the documented honest refit-correction "
                "bound) is ever a refutation; everything else is unresolved_below_power."
            ),
        },
        "closing_grounds_taxonomy_pasted_verbatim": _CLOSING_GROUNDS_TAXONOMY,
        "overlap_disclosure": (
            "seasons 2011-2017 overlap rotation-registry windows already spent by other "
            "families: [2013, 2015], [2013, 2017], [2014, 2017]. Windows retire PER-FAMILY, not "
            "globally, and this screen declares/spends no rotation-registry window of its own "
            "(lead-gen replication only, recorded straight to weak_signals -- precedent: "
            "odds-microstructure/nfl-bias-battery batteries)."
        ),
        "reliability_check": {
            "method": "not_applicable",
            "reason": (
                "each construct compares model-pick accuracy across a situational bucket vs. "
                "its complement, not a persistent per-team/per-entity trait -- there is nothing "
                "to split-half, matching every other situational subset_bias flag builder in "
                "this project (extra-rest edge, short week, sandwich spot, etc.)."
            ),
        },
    }


def _joint_block_bootstrap(
    df: pd.DataFrame,
    *,
    flag: npt.NDArray[np.bool_],
    value_col: str,
    block_col: str,
    samples: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Vectorized joint block bootstrap of ``100 * (bucket_mean - complement_mean)``.

    Both arms' means for a given draw come from the SAME resampled set of
    blocks (one multinomial draw over the shared block ids) -- the correct
    way to jointly bootstrap a two-group comparison sharing a blocking
    structure. Mirrors ``scripts/nfl_bias_battery_screen.py``'s
    ``block_bootstrap_two_group`` / ``nfl_ats.experiment_runner``'s
    ``_block_bootstrap_subset_gap`` (same ``np.unique`` block-id derivation,
    same single ``rng.multinomial`` call shape, same bincount sums/counts
    trick), reimplemented locally rather than imported.
    """

    blocks, block_index = np.unique(df[block_col].to_numpy(), return_inverse=True)
    block_index = np.asarray(block_index).reshape(-1)
    block_count = len(blocks)
    values = df[value_col].to_numpy(dtype=np.float64)
    flag_arr = np.asarray(flag, dtype=bool)

    sums: dict[bool, npt.NDArray[np.float64]] = {}
    counts: dict[bool, npt.NDArray[np.float64]] = {}
    for group in (True, False):
        mask = flag_arr == group
        sums[group] = np.bincount(
            block_index[mask], weights=values[mask], minlength=block_count
        ).astype(np.float64)
        counts[group] = np.bincount(block_index[mask], minlength=block_count).astype(np.float64)

    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(block_count, np.full(block_count, 1.0 / block_count), size=samples)
    bucket_count = drawn @ counts[True]
    complement_count = drawn @ counts[False]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bucket = (drawn @ sums[True]) / bucket_count
        mean_complement = (drawn @ sums[False]) / complement_count
    gap = (mean_bucket - mean_complement) * 100.0
    valid = (bucket_count > 0) & (complement_count > 0)
    return np.asarray(gap[valid], dtype=np.float64)


@dataclass(frozen=True)
class ConstructResult:
    key: str
    registry_name: str
    mined_direction: str
    n_bucket: int
    n_complement: int
    acc_bucket: float
    acc_complement: float
    raw_effect_pts: float
    hypothesis_sign: int
    week_block_count: int
    week_degenerate: bool
    signed_estimate: float
    signed_lower: float
    signed_upper: float
    probability_positive: float
    season_block_count: int
    season_degenerate: bool
    season_signed_estimate: float
    season_signed_lower: float
    season_signed_upper: float
    classification: str
    closing_ground: str | None
    classification_note: str
    same_direction_as_mined: bool


def _score_construct(
    df: pd.DataFrame,
    *,
    key: str,
    registry_name: str,
    bucket_flag: pd.Series,
    mined_direction: str,
    samples: int,
    seed: int,
) -> ConstructResult:
    flag = bucket_flag.to_numpy(dtype=bool)
    n_bucket = int(flag.sum())
    n_complement = int((~flag).sum())
    if n_bucket == 0 or n_complement == 0:
        raise ValueError(f"{key}: bucket or complement is empty after restricting to 2011-2017")

    acc_bucket = float(df.loc[flag, "correct"].mean())
    acc_complement = float(df.loc[~flag, "correct"].mean())
    raw_effect_pts = (acc_bucket - acc_complement) * 100.0
    hypothesis_sign = 1 if mined_direction == "higher" else -1

    week_block_count = int(df["week_block"].nunique())
    week_verdict = guard_block_count(
        week_block_count,
        min_blocks=MIN_BLOCKS_FOR_INTERVAL,
        on_degenerate="warn",
        context=f"pick_conditioned_pre2018_screen {key} (week-blocked)",
    )
    week_draws = _joint_block_bootstrap(
        df, flag=flag, value_col="correct", block_col="week_block", samples=samples, seed=seed
    )
    signed_week = hypothesis_sign * week_draws
    signed_estimate = float(np.mean(signed_week))
    signed_lower = float(np.quantile(signed_week, 0.025))
    signed_upper = float(np.quantile(signed_week, 0.975))
    probability_positive = float(np.mean(signed_week > 0.0))

    season_block_count = int(df["season"].nunique())
    season_verdict = guard_block_count(
        season_block_count,
        min_blocks=MIN_BLOCKS_FOR_INTERVAL,
        on_degenerate="warn",
        context=f"pick_conditioned_pre2018_screen {key} (season-blocked)",
    )
    season_draws = _joint_block_bootstrap(
        df, flag=flag, value_col="correct", block_col="season", samples=samples, seed=seed
    )
    signed_season = hypothesis_sign * season_draws
    season_signed_estimate = float(np.mean(signed_season))
    season_signed_lower = float(np.quantile(signed_season, 0.025))
    season_signed_upper = float(np.quantile(signed_season, 0.975))

    classification = classify_subset_bias_result(
        estimate=signed_estimate, lower=signed_lower, upper=signed_upper
    )

    same_direction_as_mined = (raw_effect_pts > 0.0) == (mined_direction == "higher")

    return ConstructResult(
        key=key,
        registry_name=registry_name,
        mined_direction=mined_direction,
        n_bucket=n_bucket,
        n_complement=n_complement,
        acc_bucket=acc_bucket,
        acc_complement=acc_complement,
        raw_effect_pts=raw_effect_pts,
        hypothesis_sign=hypothesis_sign,
        week_block_count=week_block_count,
        week_degenerate=week_verdict.degenerate,
        signed_estimate=signed_estimate,
        signed_lower=signed_lower,
        signed_upper=signed_upper,
        probability_positive=probability_positive,
        season_block_count=season_block_count,
        season_degenerate=season_verdict.degenerate,
        season_signed_estimate=season_signed_estimate,
        season_signed_lower=season_signed_lower,
        season_signed_upper=season_signed_upper,
        classification=classification.classification,
        closing_ground=classification.closing_ground,
        classification_note=classification.note,
        same_direction_as_mined=same_direction_as_mined,
    )


def _record_command(result: ConstructResult, *, spec: dict[str, Any]) -> str:
    description = (
        f"Pick-conditioned replication (lead-gen, 2011-2017 close-grade walk-forward, ACTIVE "
        f"weak_stack/market_residual/ridge_alpha=10 model, production rule "
        f"home_cover_probability>=0.5): bucket={spec['bucket']!r}; mined on 2018-2025/2020-2025 "
        f"and reported here as {spec['mined_reference']}; "
        f"mined direction={result.mined_direction!r}. "
        f"Raw (bucket-complement) gap={result.raw_effect_pts:+.4f} pts on n_bucket="
        f"{result.n_bucket}/n_complement={result.n_complement}. Effect below is HYPOTHESIS-SIGNED "
        f"(positive = replicates the mined direction). Seasons 2011-2017 overlap rotation-registry "
        f"windows spent by other families ([2013,2015], [2013,2017], [2014,2017]); windows retire "
        f"per-family, not globally, and this screen spends no window of its own."
    )
    evidence = (
        f"n_bucket={result.n_bucket}, n_complement={result.n_complement}; "
        f"acc_bucket={result.acc_bucket:.4f}, acc_complement={result.acc_complement:.4f}; "
        f"week-blocked ({result.week_block_count} blocks"
        f"{' [DEGENERATE]' if result.week_degenerate else ''}, "
        f"{BOOTSTRAP_SAMPLES} draws, seed={BOOTSTRAP_SEED}): "
        f"signed estimate={result.signed_estimate:+.4f} "
        f"pts, 95% [{result.signed_lower:+.4f}, {result.signed_upper:+.4f}] pts, "
        f"P+={result.probability_positive:.4f}; season-blocked ({result.season_block_count} blocks"
        f"{' [DEGENERATE, below MIN_BLOCKS_FOR_INTERVAL]' if result.season_degenerate else ''}): "
        f"signed estimate={result.season_signed_estimate:+.4f} pts, 95% "
        f"[{result.season_signed_lower:+.4f}, {result.season_signed_upper:+.4f}] pts. "
        f"same_direction_as_mined={result.same_direction_as_mined}. {result.classification_note}"
    )
    parts = [
        "nfl-ats weak-signals record",
        f'--name "{result.registry_name}"',
        f'--description "{description}"',
        f'--source "scripts/pick_conditioned_pre2018_screen.py, seed={BOOTSTRAP_SEED}, '
        f'samples={BOOTSTRAP_SAMPLES}"',
        f"--effect {result.signed_estimate:.6f}",
        "--effect-units accuracy_points",
        f"--classification {result.classification}",
        "--league nfl",
        f"--season-start {START_SEASON}",
        f"--season-end {END_SEASON}",
        f"--interval-low {result.signed_lower:.6f}",
        f"--interval-high {result.signed_upper:.6f}",
        f"--probability-positive {result.probability_positive:.6f}",
        f"--sample-games {result.n_bucket + result.n_complement}",
        f"--sample-blocks {result.week_block_count}",
        f'--classification-evidence "{evidence}"',
    ]
    if result.closing_ground is not None:
        parts.append(f"--closing-ground {result.closing_ground}")
    return " ".join(parts)


def main() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    predeclaration = _predeclaration()
    predeclaration_path = SCRATCH_DIR / "predeclaration.json"
    predeclaration_path.write_text(json.dumps(predeclaration, indent=2), encoding="utf-8")
    print(f"Predeclared {len(predeclaration['constructs'])} constructs -> {predeclaration_path}")

    if not FEATURES_PATH.is_file():
        raise SystemExit(f"Feature table not found: {FEATURES_PATH}")
    features = pd.read_parquet(FEATURES_PATH)

    outcome_result = walk_forward_outcomes(
        features,
        start_season=START_SEASON,
        end_season=END_SEASON,
        regressor=REGRESSOR,
        feature_profile=FEATURE_PROFILE,  # type: ignore[arg-type]
        methods=(METHOD,),
        ridge_alpha=RIDGE_ALPHA,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
    )
    predictions = outcome_result.predictions
    predictions = predictions.loc[predictions["method"] == METHOD].copy()
    n_scored_total = len(predictions)

    graded = predictions.loc[
        predictions["home_cover"].notna() & predictions["home_cover_probability"].notna()
    ].copy()
    n_pushes_dropped = n_scored_total - len(graded)

    graded["our_pick_home"] = graded["home_cover_probability"] >= 0.5
    graded["correct"] = (graded["our_pick_home"] == (graded["home_cover"] == 1)).astype(float)
    graded["our_pick_side"] = np.where(graded["our_pick_home"], "HOME", "AWAY")
    graded["our_pick_is_favorite"] = np.where(
        graded["our_pick_home"], graded["spread_line"] < 0, graded["spread_line"] > 0
    )
    graded["abs_spread"] = graded["spread_line"].abs()
    graded["rest_equal"] = graded["rest_diff"] == 0
    graded["picked_team_off_bye"] = (
        graded["our_pick_home"] & (graded["rest_diff"] >= OFF_BYE_REST_DIFF)
    ) | ((~graded["our_pick_home"]) & (graded["rest_diff"] <= -OFF_BYE_REST_DIFF))
    graded["week_block"] = graded["season"] * 100 + graded["week"]

    season_lo, season_hi = int(graded["season"].min()), int(graded["season"].max())
    print(
        f"Walk-forward {START_SEASON}-{END_SEASON} close-grade picks (weak_stack/market_residual/"
        f"ridge_alpha={RIDGE_ALPHA}): {len(graded)} scored non-push games out of {n_scored_total} "
        f"total ({n_pushes_dropped} pushes dropped); observed seasons {season_lo}-{season_hi}."
    )

    construct_specs = [
        {
            "key": "road_favorite",
            "registry_name": "pick_conditioned_road_favorite_pre2018",
            "bucket": (
                "our_pick_side=='AWAY' and our_pick_is_favorite (picked team is road favorite)"
            ),
            "mined_reference": "55.39% opener (n=482) / 55.75% close (n=687)",
            "flag": (graded["our_pick_side"] == "AWAY") & graded["our_pick_is_favorite"],
            "mined_direction": "higher",
        },
        {
            "key": "rest_mismatch",
            "registry_name": "pick_conditioned_rest_mismatch_pre2018",
            "bucket": "rest_diff != 0 (unequal rest) vs rest_diff == 0 (equal rest)",
            "mined_reference": "50.68% (mismatch) vs 54.77% (equal) opener; 49.46% vs 52.73% close",
            "flag": ~graded["rest_equal"],
            "mined_direction": "lower",
        },
        {
            "key": "off_bye_fade",
            "registry_name": "pick_conditioned_off_bye_fade_pre2018",
            "bucket": "picked_team_off_bye (rest_diff>=6 in the picked team's favor)",
            "mined_reference": "40.74% opener (n=54) / 43.96% close (n=91)",
            "flag": graded["picked_team_off_bye"],
            "mined_direction": "lower",
        },
        {
            "key": "spread_gap_zone",
            "registry_name": "pick_conditioned_spread_gap_zone_pre2018",
            "bucket": f"{SPREAD_GAP_LOW} < abs(spread_line) <= {SPREAD_GAP_HIGH}",
            "mined_reference": "45.96% opener (n=198) / 47.97% close (n=271)",
            "flag": (graded["abs_spread"] > SPREAD_GAP_LOW)
            & (graded["abs_spread"] <= SPREAD_GAP_HIGH),
            "mined_direction": "lower",
        },
    ]

    results: list[ConstructResult] = []
    record_commands: list[str] = []
    for spec in construct_specs:
        result = _score_construct(
            graded,
            key=spec["key"],
            registry_name=spec["registry_name"],
            bucket_flag=spec["flag"],
            mined_direction=spec["mined_direction"],
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        results.append(result)
        record_commands.append(_record_command(result, spec=spec))

        print(f"\n=== {result.registry_name} ===")
        print(f"  mined direction: {result.mined_direction} ({spec['mined_reference']})")
        print(
            f"  n_bucket={result.n_bucket} acc_bucket={result.acc_bucket:.4f}  "
            f"n_complement={result.n_complement} acc_complement={result.acc_complement:.4f}"
        )
        print(f"  raw effect (bucket-complement) = {result.raw_effect_pts:+.4f} pts")
        print(
            f"  hypothesis-signed week-blocked ({result.week_block_count} blocks"
            f"{' DEGENERATE' if result.week_degenerate else ''}): "
            f"estimate={result.signed_estimate:+.4f} pts, 95% "
            f"[{result.signed_lower:+.4f}, {result.signed_upper:+.4f}] pts, "
            f"P+={result.probability_positive:.4f}"
        )
        print(
            f"  hypothesis-signed season-blocked ({result.season_block_count} blocks"
            f"{' DEGENERATE' if result.season_degenerate else ''}): "
            f"estimate={result.season_signed_estimate:+.4f} pts, 95% "
            f"[{result.season_signed_lower:+.4f}, {result.season_signed_upper:+.4f}] pts"
        )
        print(f"  same_direction_as_mined={result.same_direction_as_mined}")
        print(f"  classification={result.classification} closing_ground={result.closing_ground}")

    output = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "population": {
            "league": "nfl",
            "seasons": [START_SEASON, END_SEASON],
            "grade": "close",
            "n_games_scored": len(graded),
            "n_pushes_dropped": n_pushes_dropped,
            "observed_season_range": [season_lo, season_hi],
        },
        "results": [asdict(r) for r in results],
        "proposed_record_commands": record_commands,
    }
    results_path = SCRATCH_DIR / "results.json"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults -> {results_path}")

    print("\nProposed (NOT executed) nfl-ats weak-signals record commands:")
    for command in record_commands:
        print(f"\n{command}")


if __name__ == "__main__":
    main()
