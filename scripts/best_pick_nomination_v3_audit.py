"""AUDIT + HEAD-TO-HEAD, measure-only: do either of the two open Best-Pick
opener-ranker registry leads
(``registry/weak_signals.json:best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered``,
``registry/weak_signals.json:best_pick_opener_ranker_candidate_prob_distance_vs_status_quo``)
survive a tie-break audit, and if so, do they beat the LIVE production v2
nomination rule (``nfl_ats.best_pick_nomination.nominate_v2`` /
``select_nominee``) rather than the older status-quo/unfiltered baselines the
registry entries were actually measured against?

Context this script exists to check: the project's only prior ``confirmed``
Best-Pick result (``sweep_robustness``, docs/best_pick_ranker.md) collapsed
from a recorded +8.68 points to a tie-break-agnostic +0.92 once its majority
-tie alphabetical tie-break was audited. Both open leads here use the SAME
"ascending game_id" tie-break convention
(``scripts/best_pick_opener_ranker_eval.py::nominate``), so the same failure
mode is checked here before either lead is treated as live evidence.

Reuses ``scripts/best_pick_opener_ranker_eval.py`` (predeclared script,
unmodified, imported by file path -- same pattern
``scripts/surface_familiarity_screen.py`` uses to reuse
``scripts/nfl_weather_battery_screen.py``) for the working frame, the
dispersion pool, the nomination/tie-break machinery, and the bootstrap
helpers. No number here is hand-typed from a doc; everything is recomputed
from the same two stored source artifacts
(``artifacts/ridge_alpha_promotion/20260818T221459Z``,
``artifacts/odds_microstructure/20260818T225430Z``) the two registered leads
were computed from, with a reproduction check against the already-recorded
artifact (``artifacts/best_pick_opener_ranker/20260818T230550Z/summary.json``)
before anything new is computed.

Adds exactly ONE new chooser beyond the eight already scored: the live
production rule itself, ``candidate_dist`` (primary) restricted to the
below-median-dispersion pool, ties broken by lower ``spread_std`` then
ascending ``game_id`` -- i.e. ``nfl_ats.best_pick_nomination.select_nominee``
reproduced on this historical population. The module docstring of
``best_pick_nomination.py`` states explicitly that this exact composition
("chooser 6's filter, PLUS a dispersion tie-break applied *within* that
filtered pool") was never itself scored as one chooser in the original
screen; this script is that missing chooser.

BINDING (owner mandates, restated here because this script's output feeds a
registry write): an interval or CI that contains zero is NEVER grounds to
reject, fail, or close an experiment -- "contains zero" is the EXPECTED
outcome for a real small signal at this evaluator's resolution. Only two
closing grounds exist: (1) refuted mechanism -- a RESOLVED wrong sign (the
WHOLE interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``, reported with
``probability_positive``, never collapsed to "contains zero". Decide on EV:
the pool is forced picks, a nomination happens every week regardless, so
``probability_positive`` above 0.5 favours the switch; a promotion bar
governs what may be CLAIMED, never what gets PLAYED. Within-week correlation
is zero (owner-mandated, hardcoded).

Run:  .\\.tools\\uv.exe run --no-sync python scripts/best_pick_nomination_v3_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_EVAL_PATH = REPO / "scripts" / "best_pick_opener_ranker_eval.py"
_spec = importlib.util.spec_from_file_location("best_pick_opener_ranker_eval", _EVAL_PATH)
assert _spec is not None and _spec.loader is not None
_ranker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ranker)

ChooserSpec = _ranker.ChooserSpec
load_working_frame = _ranker.load_working_frame
build_dispersion_pool = _ranker.build_dispersion_pool
nominate = _ranker.nominate
evaluate_chooser = _ranker.evaluate_chooser
_paired_delta = _ranker._paired_delta
DEFAULT_SOURCE = _ranker.DEFAULT_SOURCE
DEFAULT_MICROSTRUCTURE_SOURCE = _ranker.DEFAULT_MICROSTRUCTURE_SOURCE

# Same seed/sample count as the artifact being audited (20260818T230550Z) --
# reused deliberately for comparability, not re-chosen after seeing results.
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260818

STORED_ARTIFACT = (
    REPO / "artifacts" / "best_pick_opener_ranker" / "20260818T230550Z" / "summary.json"
)

# The live production tie-break, exactly matching
# nfl_ats.best_pick_nomination.select_nominee: primary candidate_dist desc,
# secondary neg_spread_std desc (== spread_std asc, NaN last under pandas'
# default na_position="last" regardless of sort direction, matching
# select_nominee's explicit na_position="last"), tertiary game_id asc.
# This composition (chooser 6's filter + chooser 8's tie-break, WITHIN the
# filtered pool) is the one best_pick_nomination.py's own docstring says was
# "never itself scored as one chooser."
LIVE_V2_SPEC = ChooserSpec(
    "live_v2_production",
    "candidate_dist",
    "neg_spread_std",
    "candidate_correct_open",
    False,
    pool_column="dispersion_pool_pass",
)

AUDITED_SPECS = {spec.name: spec for spec in _ranker.PRIMARY_CHOOSERS}
AUDITED_SPECS.update({spec.name: spec for spec in _ranker.DISPERSION_CHOOSERS})


def main() -> None:
    work = load_working_frame(DEFAULT_SOURCE)
    work, dispersion_summary = build_dispersion_pool(work, DEFAULT_MICROSTRUCTURE_SOURCE)

    # -----------------------------------------------------------------
    # 0. Reproduction check against the already-recorded artifact, before
    #    trusting this frame for anything new.
    # -----------------------------------------------------------------
    stored = json.loads(STORED_ARTIFACT.read_text(encoding="utf-8"))
    repro: dict[str, Any] = {}
    for name in ("candidate_prob_distance", "dispersion_filtered_candidate", "status_quo_residual"):
        spec = AUDITED_SPECS[name]
        result = evaluate_chooser(spec, work, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
        stored_row = stored["choosers"][name]
        matches = (
            abs(result["top1_accuracy"] - stored_row["top1_accuracy"]) < 1e-9
            and result["n_tie_weeks"] == stored_row["n_tie_weeks"]
            and abs(
                result["mean_weekly_lift"]["estimate"] - stored_row["mean_weekly_lift"]["estimate"]
            )
            < 1e-9
        )
        repro[name] = {
            "recomputed_top1_accuracy": result["top1_accuracy"],
            "stored_top1_accuracy": stored_row["top1_accuracy"],
            "recomputed_n_tie_weeks": result["n_tie_weeks"],
            "stored_n_tie_weeks": stored_row["n_tie_weeks"],
            "recomputed_mean_lift": result["mean_weekly_lift"]["estimate"],
            "stored_mean_lift": stored_row["mean_weekly_lift"]["estimate"],
            "exact_match": matches,
        }
    if not all(r["exact_match"] for r in repro.values()):
        raise SystemExit(
            "Reproduction check FAILED -- recomputed frame diverges from the "
            "stored artifact; refusing to compute new numbers off a "
            f"population that does not match. Detail: {json.dumps(repro, indent=2)}"
        )

    # -----------------------------------------------------------------
    # 1. Tie audit: same-kind check as the sweep_robustness collapse
    #    (docs/best_pick_ranker.md Tier-2 re-read) -- tie-agnostic paired
    #    deltas for both registered leads.
    # -----------------------------------------------------------------
    candidate_a = evaluate_chooser(
        AUDITED_SPECS["dispersion_filtered_candidate"],
        work,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    candidate_b = evaluate_chooser(
        AUDITED_SPECS["candidate_prob_distance"],
        work,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    status_quo = evaluate_chooser(
        AUDITED_SPECS["status_quo_residual"], work, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )

    tie_audit = {
        "dispersion_filtered_candidate_vs_candidate_prob_distance": {
            "recorded_lift_based": _paired_delta(
                candidate_a["weekly_frame"],
                candidate_b["weekly_frame"],
                "lift",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "tie_agnostic": _paired_delta(
                candidate_a["weekly_frame"],
                candidate_b["weekly_frame"],
                "tie_agnostic_lift",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "n_tie_weeks_candidate_a": candidate_a["n_tie_weeks"],
            "n_tie_weeks_candidate_b": candidate_b["n_tie_weeks"],
        },
        "candidate_prob_distance_vs_status_quo_residual": {
            "recorded_lift_based": _paired_delta(
                candidate_b["weekly_frame"],
                status_quo["weekly_frame"],
                "lift",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "tie_agnostic": _paired_delta(
                candidate_b["weekly_frame"],
                status_quo["weekly_frame"],
                "tie_agnostic_lift",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "n_tie_weeks_candidate_b": candidate_b["n_tie_weeks"],
            "n_tie_weeks_status_quo": status_quo["n_tie_weeks"],
        },
        "dispersion_filter_threshold_provenance": (
            "Below-median spread_std, a PER-WEEK RELATIVE split (pandas .median() "
            "on that week's own games), not a scalar constant chosen by scanning "
            "candidate thresholds. Fallback to the full week is triggered by "
            "missing data or an empty strict filter (both counted separately in "
            "dispersion_pool_summary below). reported (best_pick_opener_ranker_eval.py "
            "module docstring, unverified independently since the session scratchpad "
            "predeclaration file no longer exists on disk): predeclared in "
            "scratchpad/bestpick_opener/predeclaration.md's evening addendum before "
            "any accuracy number was computed. No numeric threshold (e.g. a specific "
            "percentile or point value) was ever tuned against outcomes -- the median "
            "split is defined structurally, so there is nothing to have tuned post hoc."
        ),
    }

    # -----------------------------------------------------------------
    # 2. The missing 9th chooser: the LIVE production rule, reproduced on
    #    this historical population.
    # -----------------------------------------------------------------
    live_v2 = evaluate_chooser(LIVE_V2_SPEC, work, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)

    # Where do candidate A / candidate B actually disagree with the live
    # rule's nominee? (diagnostic, not part of the bootstrap)
    def _diverging_weeks(a: pd.DataFrame, b: pd.DataFrame) -> list[dict[str, Any]]:
        merged = a[["season", "week", "nominee_game_id"]].merge(
            b[["season", "week", "nominee_game_id"]],
            on=["season", "week"],
            suffixes=("_a", "_b"),
        )
        diff = merged.loc[merged["nominee_game_id_a"] != merged["nominee_game_id_b"]]
        return diff.to_dict(orient="records")

    divergence_a = _diverging_weeks(candidate_a["weekly_frame"], live_v2["weekly_frame"])
    divergence_b = _diverging_weeks(candidate_b["weekly_frame"], live_v2["weekly_frame"])

    head_to_head = {
        "dispersion_filtered_candidate_vs_live_v2_production": {
            **_paired_delta(
                candidate_a["weekly_frame"],
                live_v2["weekly_frame"],
                "lift",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "n_weeks_nominee_diverges": len(divergence_a),
            "diverging_weeks": divergence_a,
        },
        "candidate_prob_distance_vs_live_v2_production": {
            **_paired_delta(
                candidate_b["weekly_frame"],
                live_v2["weekly_frame"],
                "lift",
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "n_weeks_nominee_diverges": len(divergence_b),
            "diverging_weeks": divergence_b,
        },
    }

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "population": {
            "games": len(work),
            "weeks": int(work.groupby(["season", "week"]).ngroups),
            "seasons": sorted(int(s) for s in work["season"].unique()),
        },
        "dispersion_pool_summary": dispersion_summary,
        "reproduction_check": repro,
        "tie_audit": tie_audit,
        "live_v2_production_chooser": {k: v for k, v in live_v2.items() if k != "weekly_frame"},
        "head_to_head_vs_live_v2": head_to_head,
        "multiplicity_note": (
            "Fourth reuse of the same 107 opener weeks this session-family "
            "(ridge_alpha promotion look, odds-microstructure battery, the "
            "original ranker screen, now this audit) -- compounding "
            "look-reuse discount, stated explicitly. Not a rotation-registry "
            "window; no window is spent or implied by this script."
        ),
        "binding_note": (
            "An interval containing zero is never treated as a rejection here; "
            "probability_positive is the continuous evidence to read, not a "
            "significant/not-significant collapse. Closing grounds are limited "
            "to refuted_mechanism (resolved wrong sign or zero split-half "
            "reliability) and bounded_by_control; nothing here claims either."
        ),
    }

    print(json.dumps(summary, indent=2, default=str))

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "artifacts" / "best_pick_nomination_v3_audit" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    candidate_a["weekly_frame"].to_parquet(out_dir / "dispersion_filtered_candidate.weekly.parquet")
    candidate_b["weekly_frame"].to_parquet(out_dir / "candidate_prob_distance.weekly.parquet")
    status_quo["weekly_frame"].to_parquet(out_dir / "status_quo_residual.weekly.parquet")
    live_v2["weekly_frame"].to_parquet(out_dir / "live_v2_production.weekly.parquet")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
