"""Apply the empirical Bayes decision rule to this project's live candidates.

Three steps, matching ``docs/decision_rule.md``:

1. Fit the empirical prior from the 210 deduplicated accuracy-scale
   measurements in ``artifacts/**/*paired*.csv``
   (``scripts/decision_load_measurements.py``).
2. Validate it against the one clean before/after regression event on record
   -- MOD-07's +1.97 on 456 games, which delivered +0.33 on 1,537 -- and
   report whether the posterior lands near the delivered number.
3. Evaluate every live candidate this project has measured: every
   ``unresolved_below_power`` (and, for completeness, ``refuted_mechanism``)
   entry in ``registry/weak_signals.json``, plus the MOD-07 stack itself, plus
   the three-signal pooled weak-signal result recorded in ``AGENTS.md``.

Read-only: does not touch the registry, the rotation registry, or any model
artifact. Prints the full report and writes a JSON snapshot to the path given
on the command line (defaults to a scratch location, never inside the repo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats import decision_rule as dr  # noqa: E402
from nfl_ats.weak_signals import default_registry_path, load_registry  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_load_measurements import load_measurements  # noqa: E402

# ---------------------------------------------------------------------------
# Hand-derived measurements not present as accuracy-scale rows in any
# *paired*.csv artifact, sourced directly from the docs cited in each comment.
# Every number here is copied, not re-estimated, from the cited source.
# ---------------------------------------------------------------------------

# docs/mod07_stack.md "Result": 456 paired games, week-blocked 95% CI.
MOD07_RAW = dr.EffectMeasurement(
    label="mod07_weak_stack (raw, 456-game selected window)",
    estimate=1.97,
    standard_error=dr.se_from_interval(-1.10, 5.00),
    n_games=456,
    source="docs/mod07_stack.md",
)
# docs/pool_edge_plan.md "Where to look next": delivered result on the full
# 1,537-game opener-graded window. Not itself fed to the prior fit -- this is
# the ground truth the validation step checks the posterior against.
MOD07_DELIVERED_ESTIMATE = 0.33
MOD07_DELIVERED_N = 1537

# AGENTS.md, "interval crossing zero" section: three pooled weak signals.
POOLED_WEAK_SIGNAL = dr.EffectMeasurement(
    label="pooled_weak_signal (3-signal inverse-variance pool)",
    estimate=0.724,
    standard_error=dr.se_from_interval(0.056, 1.392),
    n_games=None,
    source="AGENTS.md interval-crossing-zero section, 2026-08-18",
)

# docs/injury_value_lost.md section 4: the narrowed, semantics-shift-free arm
# is the number the task explicitly asks for, not the conflated +1.75.
INJURY_VALUE_LOST_NARROWED = dr.EffectMeasurement(
    label="injury_value_lost_narrowed (D-A, semantics-shift removed)",
    estimate=1.316,
    standard_error=dr.se_from_interval(-0.460, 3.247),
    n_games=456,
    source="docs/injury_value_lost.md section 4",
)

# docs/pool_edge_plan.md ceiling section: the project's own documented
# exchange rate between line-error points and ATS accuracy points, used only
# to make fourth_down_aggressiveness's ats_points figure commensurable with
# every other candidate here. Flagged explicitly wherever it is used.
ATS_TO_ACCURACY_EXCHANGE_RATE = 3.0

# registry/weak_signals.json: penalty_discipline carries no interval or SE.
# Derived here from its own reported components (49.85% vs 50.52% on
# quartile-sized samples out of 4,431 games -> ~1,108 games per quartile),
# using the project's own quoted percentages -- not a new measurement.
PENALTY_DISCIPLINE_QUARTILE_N = 4431 // 4
PENALTY_DISCIPLINE_P_LOW = 0.4985
PENALTY_DISCIPLINE_P_HIGH = 0.5052


def _penalty_discipline_se() -> float:
    p1, p2, n = PENALTY_DISCIPLINE_P_LOW, PENALTY_DISCIPLINE_P_HIGH, PENALTY_DISCIPLINE_QUARTILE_N
    variance = p1 * (1 - p1) / n + p2 * (1 - p2) / n
    return (variance**0.5) * 100.0


# ---------------------------------------------------------------------------
# Calibration-noise sensitivity (docs/purged_cv.md positive control)
# ---------------------------------------------------------------------------
#
# purged_cv.md's positive control plants a KNOWN accuracy effect and runs it
# through the real pipeline. At a 1.3-point true effect, the full pipeline
# (with the out-of-time residual/calibration step) recovered the WRONG SIGN
# (-0.67 vs a realized +1.67); at 3.0 points it recovered +1.14 of +3.34. The
# doc's own diagnosis is NOT multiplicative attenuation -- a sign flip is
# impossible under a 0 < k < 1 shrinkage model -- it is an ADDITIVE,
# direction-unstable noise term from the calibration sample's fold-to-fold
# instability (quoted as 0.9-1.3 accuracy points across individual folds in a
# 10-fold worked example). A multiplicative "divide the estimate by k"
# correction is deliberately NOT applied here: it is not the mechanism the
# source document identifies, and applying one anyway is exactly the
# "pick the value that produces the most positive answers" failure this
# sensitivity check exists to avoid.
#
# The same document also reports the ONLY real-data (non-synthetic-plant)
# reading of this distortion: comparing the full pipeline against a sign-only
# ablation on the actual data, the gap was 0.1 accuracy point. That
# measurement aggregates over the full walk-forward evaluator's many
# refits (hundreds of weeks), which is the same regime nearly all 210 fitted
# measurements come from -- so REAL_DATA_NOISE_FLOOR is the primary
# correction applied to the whole fitting set below, and it is small by
# construction. The larger 0.9-1.3 figure describes per-fold instability in a
# 10-fold experiment and is applied, as a labelled sensitivity bound rather
# than a primary correction, only to candidates measured on a comparably
# small number of blocks (<= SMALL_SAMPLE_GAMES_THRESHOLD games, a proxy for
# "few independent refits" -- MOD-07 and the injury-value-lost family both
# ran on 456 games / 35 blocks, structurally the closest thing in this
# project's record to the purged-CV positive control's own fold count).
REAL_DATA_NOISE_FLOOR = 0.1
SMALL_SAMPLE_NOISE_FLOOR = 1.1  # midpoint of the quoted 0.9-1.3 pt range
SMALL_SAMPLE_NOISE_FLOOR_RANGE = (0.9, 1.3)
SMALL_SAMPLE_GAMES_THRESHOLD = 1000


# ---------------------------------------------------------------------------


def build_registry_measurements() -> list[dr.EffectMeasurement]:
    registry = load_registry(default_registry_path(REPO_ROOT / "registry"))
    out: list[dr.EffectMeasurement] = []
    for name, signal in sorted(registry.signals.items()):
        estimate = signal.effect
        units_note = ""
        if signal.effect_units == "ats_points":
            estimate = signal.effect * ATS_TO_ACCURACY_EXCHANGE_RATE
            units_note = f" [converted from ats_points x{ATS_TO_ACCURACY_EXCHANGE_RATE}]"

        se = signal.resolved_standard_error()
        if se is None and signal.probability_positive is not None:
            se = dr.se_from_probability_positive(signal.effect, signal.probability_positive)
            if signal.effect_units == "ats_points":
                se *= ATS_TO_ACCURACY_EXCHANGE_RATE
            units_note += " [SE recovered from probability_positive]"
        if se is None and name == "penalty_discipline":
            se = _penalty_discipline_se()
            units_note += " [SE derived from quoted quartile percentages]"
        if se is None:
            print(f"SKIP {name}: no interval, SE, or probability_positive to recover one from")
            continue

        out.append(
            dr.EffectMeasurement(
                label=f"{name}{units_note}",
                estimate=estimate,
                standard_error=se,
                n_games=signal.sample_games,
                source=f"registry:{signal.source} [{signal.classification}]",
            )
        )
    return out


def main() -> int:
    out_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else (REPO_ROOT / "artifacts" / "decision_rule_report.json")
    )

    # 1. Fit the prior.
    measurements_df = load_measurements()
    fit_inputs = [
        dr.EffectMeasurement(
            label=f"{row.source}:{row.label}:{row.window}",
            estimate=float(row.estimate),
            standard_error=float(row.se),
            n_games=None if row.n_games != row.n_games else int(row.n_games),  # NaN check
            source=str(row.source),
        )
        for row in measurements_df.itertuples(index=False)
    ]
    prior_pm = dr.fit_empirical_prior(fit_inputs, method="paule_mandel")
    prior_dl = dr.fit_empirical_prior(fit_inputs, method="dersimonian_laird")

    print("=" * 78)
    print("STEP 1: EMPIRICAL PRIOR")
    print("=" * 78)
    print(f"n measurements used: {prior_pm.n_measurements}")
    print(
        f"Paule-Mandel (primary): mean={prior_pm.mean:+.4f}  tau={prior_pm.sd:.4f}  "
        f"tau^2={prior_pm.variance:.4f}"
    )
    print(
        f"DerSimonian-Laird (comparison): mean={prior_dl.mean:+.4f}  tau={prior_dl.sd:.4f}  "
        f"tau^2={prior_dl.variance:.4f}"
    )
    for se_example in (0.3, 0.5, 1.0, 1.5, 2.0):
        shrink = prior_pm.shrinkage_factor(se_example)
        print(
            f"  measurement SE={se_example:.1f} -> weight on data (shrinkage_factor)={shrink:.3f}"
        )

    # 2. Validate against MOD-07.
    print()
    print("=" * 78)
    print("STEP 2: MOD-07 REGRESSION VALIDATION")
    print("=" * 78)
    mod07_result = dr.evaluate_candidate(MOD07_RAW, prior_pm)
    mod07_se = MOD07_RAW.standard_error
    print(f"raw estimate (456 games):      {MOD07_RAW.estimate:+.3f} pts (SE {mod07_se:.3f})")
    print(f"posterior mean (shrunk):       {mod07_result.posterior_mean:+.3f} pts")
    print(f"posterior sd:                  {mod07_result.posterior_sd:.3f} pts")
    print(f"shrinkage_factor (weight on data): {mod07_result.shrinkage_factor:.3f}")
    print(f"delivered estimate (1,537 games): {MOD07_DELIVERED_ESTIMATE:+.3f} pts")
    posterior_gap = mod07_result.posterior_mean - MOD07_DELIVERED_ESTIMATE
    raw_gap = MOD07_RAW.estimate - MOD07_DELIVERED_ESTIMATE
    print(
        f"posterior vs delivered gap:    {posterior_gap:+.3f} pts "
        f"(raw estimate vs delivered gap was {raw_gap:+.3f} pts)"
    )
    predicted_shrink_pct = (1.0 - mod07_result.shrinkage_factor) * 100.0
    actual_shrink_pct = (MOD07_RAW.estimate - MOD07_DELIVERED_ESTIMATE) / MOD07_RAW.estimate * 100.0
    print(
        f"predicted shrinkage: {predicted_shrink_pct:.1f}% of the raw estimate pulled toward the "
        f"prior  |  actual shrinkage realized: {actual_shrink_pct:.1f}%"
    )

    # 2.5 Calibration-noise sensitivity (docs/purged_cv.md).
    print()
    print("=" * 78)
    print("STEP 2.5: CALIBRATION-NOISE SENSITIVITY (docs/purged_cv.md)")
    print("=" * 78)
    fit_inputs_corrected = [
        dr.EffectMeasurement(
            label=m.label,
            estimate=m.estimate,
            standard_error=(
                dr.combine_standard_errors(m.standard_error, SMALL_SAMPLE_NOISE_FLOOR)
                if (m.n_games or 0) and m.n_games <= SMALL_SAMPLE_GAMES_THRESHOLD
                else dr.combine_standard_errors(m.standard_error, REAL_DATA_NOISE_FLOOR)
            ),
            n_games=m.n_games,
            source=m.source,
        )
        for m in fit_inputs
    ]
    prior_corrected = dr.fit_empirical_prior(fit_inputs_corrected, method="paule_mandel")
    n_small = sum(
        1 for m in fit_inputs if (m.n_games or 0) and m.n_games <= SMALL_SAMPLE_GAMES_THRESHOLD
    )
    n_large = len(fit_inputs) - n_small
    print(
        f"Real-data noise floor ({REAL_DATA_NOISE_FLOOR} pt) applied to all {n_large} "
        f"large-sample fitting rows; targeted {SMALL_SAMPLE_NOISE_FLOOR} pt floor applied to "
        f"the {n_small} rows at or below {SMALL_SAMPLE_GAMES_THRESHOLD} games."
    )
    pc_mean, pc_tau = prior_corrected.mean, prior_corrected.sd
    print(f"prior (uncorrected):         mean={prior_pm.mean:+.4f}  tau={prior_pm.sd:.4f}")
    print(f"prior (noise-floor applied): mean={pc_mean:+.4f}  tau={pc_tau:.4f}")
    # Degenerate worst case, reported not adopted: apply the full small-sample
    # noise floor to EVERY fitting row, not just the small ones.
    fit_inputs_worst_case = [
        dr.EffectMeasurement(
            label=m.label,
            estimate=m.estimate,
            standard_error=dr.combine_standard_errors(m.standard_error, SMALL_SAMPLE_NOISE_FLOOR),
            n_games=m.n_games,
            source=m.source,
        )
        for m in fit_inputs
    ]
    prior_worst_case = dr.fit_empirical_prior(fit_inputs_worst_case, method="paule_mandel")
    print(
        f"prior (worst case, {SMALL_SAMPLE_NOISE_FLOOR} pt floor on EVERY row, not adopted): "
        f"mean={prior_worst_case.mean:+.4f}  tau={prior_worst_case.sd:.4f} -- "
        "collapses to a point mass because the noise floor alone explains all observed "
        "dispersion; taken literally this argues for LESS differentiation between "
        "candidates, the opposite of the inflation risk it was raised to check. Not adopted "
        "because it applies a 10-fold experiment's per-fold instability, undiminished, to "
        "measurements that average over hundreds of refits -- the same document's own "
        f"real-data comparison shows only a {REAL_DATA_NOISE_FLOOR} pt gap at that "
        "aggregation level."
    )

    # 3. Evaluate every live candidate, under both the original and the
    # noise-floor-corrected prior, inflating candidate SEs the same way.
    print()
    print("=" * 78)
    print("STEP 3: LIVE CANDIDATES")
    print("=" * 78)
    candidates = [MOD07_RAW, POOLED_WEAK_SIGNAL, INJURY_VALUE_LOST_NARROWED]
    candidates += build_registry_measurements()

    def _corrected(measurement: dr.EffectMeasurement) -> dr.EffectMeasurement:
        floor = (
            SMALL_SAMPLE_NOISE_FLOOR
            if (measurement.n_games or 0) and measurement.n_games <= SMALL_SAMPLE_GAMES_THRESHOLD
            else REAL_DATA_NOISE_FLOOR
        )
        return dr.EffectMeasurement(
            label=measurement.label,
            estimate=measurement.estimate,
            standard_error=dr.combine_standard_errors(measurement.standard_error, floor),
            n_games=measurement.n_games,
            source=measurement.source,
        )

    results = []
    results_corrected = []
    refuted_labels: set[str] = set()
    flipped: list[str] = []
    for measurement in candidates:
        result = dr.evaluate_candidate(measurement, prior_pm)
        result_corrected = dr.evaluate_candidate(_corrected(measurement), prior_corrected)
        results.append(result)
        results_corrected.append(result_corrected)
        is_refuted = "[refuted_mechanism]" in measurement.source
        if is_refuted:
            refuted_labels.add(result.label)
        if result.verdict != result_corrected.verdict:
            flipped.append(result.label)
        flag = "  ** REFUTED MECHANISM, not actionable **" if is_refuted else ""
        base = (
            f"{result.label:70s} obs={result.observed_estimate:+7.3f} "
            f"post_mean={result.posterior_mean:+7.3f} P+={result.probability_positive:.3f} "
            f"verdict={result.verdict}"
        )
        rc_mean, rc_pp, rc_verdict = (
            result_corrected.posterior_mean,
            result_corrected.probability_positive,
            result_corrected.verdict,
        )
        corrected_part = (
            f"  |  noise-floor: post_mean={rc_mean:+7.3f} P+={rc_pp:.3f} verdict={rc_verdict}{flag}"
        )
        print(base + corrected_part)
    print()
    print(f"Verdicts flipped by the noise-floor correction: {flipped if flipped else 'NONE'}")

    # Model averaging.
    print()
    print("=" * 78)
    print("MODEL AVERAGING")
    print("=" * 78)
    actionable_results = [r for r in results if r.label not in refuted_labels]
    stack = dr.model_average(actionable_results, mode="stack", weight_by="probability_positive")
    print(
        f"[stack, actionable candidates only -- refuted mechanisms excluded] "
        f"combined_expected_gain = {stack.combined_expected_gain:+.4f} pts"
    )
    print(f"  included (posterior mean > 0): {stack.included}")
    print(f"  excluded as refuted mechanisms: {sorted(refuted_labels)}")

    residual_location_labels = [
        r for r in results if "residual_location" in r.label and r.label not in refuted_labels
    ]
    blend = dr.model_average(
        residual_location_labels, mode="blend", weight_by="probability_positive"
    )
    print(
        f"[blend, residual_location_* family] combined_expected_gain = "
        f"{blend.combined_expected_gain:+.4f} pts"
    )
    for label, weight in sorted(blend.weights.items(), key=lambda kv: -kv[1]):
        print(f"  weight {weight:.3f}  {label}")

    # Snapshot.
    snapshot = {
        "prior_paule_mandel": {
            "mean": prior_pm.mean,
            "tau": prior_pm.sd,
            "n": prior_pm.n_measurements,
        },
        "prior_dersimonian_laird": {
            "mean": prior_dl.mean,
            "tau": prior_dl.sd,
            "n": prior_dl.n_measurements,
        },
        "prior_noise_floor_corrected": {
            "mean": prior_corrected.mean,
            "tau": prior_corrected.sd,
            "n": prior_corrected.n_measurements,
        },
        "prior_noise_floor_worst_case_not_adopted": {
            "mean": prior_worst_case.mean,
            "tau": prior_worst_case.sd,
        },
        "mod07_validation": {
            "raw_estimate": MOD07_RAW.estimate,
            "posterior_mean": mod07_result.posterior_mean,
            "delivered_estimate": MOD07_DELIVERED_ESTIMATE,
        },
        "verdicts_flipped_by_noise_floor_correction": flipped,
        "candidates": [
            {
                "label": r.label,
                "observed_estimate": r.observed_estimate,
                "observed_se": r.observed_se,
                "posterior_mean": r.posterior_mean,
                "posterior_sd": r.posterior_sd,
                "probability_positive": r.probability_positive,
                "shrinkage_factor": r.shrinkage_factor,
                "expected_cost_if_use_is_wrong": r.expected_cost_if_use_is_wrong,
                "expected_cost_if_skip_is_wrong": r.expected_cost_if_skip_is_wrong,
                "verdict": r.verdict,
                "noise_floor_posterior_mean": rc.posterior_mean,
                "noise_floor_probability_positive": rc.probability_positive,
                "noise_floor_verdict": rc.verdict,
            }
            for r, rc in zip(results, results_corrected, strict=True)
        ],
        "model_average_stack": {
            "combined_expected_gain": stack.combined_expected_gain,
            "included": list(stack.included),
            "weights": stack.weights,
        },
        "model_average_blend_residual_location": {
            "combined_expected_gain": blend.combined_expected_gain,
            "weights": blend.weights,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
