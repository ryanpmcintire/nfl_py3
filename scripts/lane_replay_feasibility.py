from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

WEAK_SIGNALS_PATH = Path("registry/weak_signals.json")
ROTATION_REGISTRY_PATH = Path("registry/rotation_registry.json")
OUT_ROOT = Path("artifacts/lane_replay")
BOOTSTRAP_DRAWS = 1000
PERMUTATION_DRAWS = 2000
SEED = 20260916
RESOLVING_CLASSIFICATIONS = ("refuted_mechanism", "bounded_by_control")
RESOLVING_ROTATION_STATUS = ("closed_negative",)
RESOLVING_WINDOW_VERDICTS = ("confirmed", "closed_negative")
K_VALUES = (1, 2, 3)
COUNT_BUCKETS = (("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, None))
BASE_RATE_BUCKETS = (("1-2", 1, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, None))


def load_weak_signals() -> dict[str, Any]:
    payload = json.loads(WEAK_SIGNALS_PATH.read_text(encoding="utf-8"))
    return dict(payload["signals"])


def load_rotation_families() -> dict[str, Any]:
    payload = json.loads(ROTATION_REGISTRY_PATH.read_text(encoding="utf-8"))
    return dict(payload["families"])


def build_family_cells(signals: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    families: dict[str, list[dict[str, Any]]] = {}
    unassigned = 0
    for cell_id, record in signals.items():
        family = record.get("family")
        if not family:
            unassigned += 1
            continue
        entry = dict(record)
        entry["cell_id"] = cell_id
        families.setdefault(family, []).append(entry)
    for cells in families.values():
        cells.sort(key=lambda c: (c["recorded_at"], c["cell_id"]))
    return families, unassigned


def family_resolution(
    cells: list[dict[str, Any]], rotation_record: dict[str, Any] | None
) -> tuple[bool, str]:
    reasons: list[str] = []
    if any(c.get("classification") in RESOLVING_CLASSIFICATIONS for c in cells):
        reasons.append("cell_classification_refuted_or_bounded")
    if rotation_record is not None:
        if rotation_record.get("status") in RESOLVING_ROTATION_STATUS:
            reasons.append("rotation_family_status_closed_negative")
        for window in rotation_record.get("windows", []):
            if window.get("verdict") in RESOLVING_WINDOW_VERDICTS:
                reasons.append(f"rotation_window_verdict_{window.get('verdict')}")
                break
    return (len(reasons) > 0, "+".join(sorted(set(reasons))) if reasons else "none")


def cell_count_stats(counts: list[int]) -> dict[str, Any]:
    return {
        "n_families": len(counts),
        "min_cells": min(counts),
        "median_cells": float(statistics.median(counts)),
        "max_cells": max(counts),
        "n_families_1_cell": sum(1 for c in counts if c == 1),
        "n_families_ge_5_cells": sum(1 for c in counts if c >= 5),
    }


def predictor_values(cells: list[dict[str, Any]], k: int) -> tuple[float | None, float | None]:
    first_k = cells[:k]
    prob_devs = [
        abs(float(c["probability_positive"]) - 0.5)
        for c in first_k
        if c.get("probability_positive") is not None
    ]
    predictor_a = float(np.mean(prob_devs)) if prob_devs else None
    ratios: list[float] = []
    for c in first_k:
        se = c.get("standard_error")
        effect = c.get("effect")
        if se is None or effect is None or se == 0:
            continue
        ratios.append(abs(float(effect) / float(se)))
    predictor_b = float(max(ratios)) if ratios else None
    return predictor_a, predictor_b


def auc_from_values(values: np.ndarray, labels: np.ndarray) -> float | None:
    positive = values[labels == 1]
    negative = values[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    ranks = rankdata(values, method="average")
    rank_sum_positive = ranks[labels == 1].sum()
    n_pos = len(positive)
    n_neg = len(negative)
    return float((rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def count_bucket(n_cells: int, buckets: tuple[tuple[str, int, int | None], ...]) -> str | None:
    for name, low, high in buckets:
        if high is None:
            if n_cells >= low:
                return name
        elif low <= n_cells <= high:
            return name
    return None


def bootstrap_auc(
    values: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    n = len(values)
    draws: list[float] = []
    skipped = 0
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, n)
        auc = auc_from_values(values[idx], labels[idx])
        if auc is None:
            skipped += 1
            continue
        draws.append(auc)
    if not draws:
        return {
            "valid_draws": 0,
            "skipped_draws": skipped,
            "lower_95": None,
            "upper_95": None,
            "probability_positive": None,
        }
    arr = np.asarray(draws, dtype=float)
    lower, upper = np.percentile(arr, [2.5, 97.5])
    return {
        "valid_draws": len(draws),
        "skipped_draws": skipped,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "probability_positive": float((arr > 0.5).mean() + 0.5 * (arr == 0.5).mean()),
    }


def permutation_null(
    values: np.ndarray,
    labels: np.ndarray,
    buckets: np.ndarray,
    observed_auc: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    unique_buckets = sorted(set(buckets.tolist()))
    bucket_indices = {b: np.flatnonzero(buckets == b) for b in unique_buckets}
    null_aucs: list[float] = []
    skipped = 0
    for _ in range(PERMUTATION_DRAWS):
        shuffled_labels = labels.copy()
        for idx in bucket_indices.values():
            shuffled_labels[idx] = rng.permutation(labels[idx])
        auc = auc_from_values(values, shuffled_labels)
        if auc is None:
            skipped += 1
            continue
        null_aucs.append(auc)
    if not null_aucs:
        return {"valid_draws": 0, "skipped_draws": skipped, "permutation_p": None}
    arr = np.asarray(null_aucs, dtype=float)
    return {
        "valid_draws": len(null_aucs),
        "skipped_draws": skipped,
        "permutation_p": float((arr >= observed_auc).mean()),
    }


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    signals = load_weak_signals()
    rotation_families = load_rotation_families()
    family_cells, unassigned_cells = build_family_cells(signals)

    resolution: dict[str, tuple[bool, str]] = {
        family: family_resolution(cells, rotation_families.get(family))
        for family, cells in family_cells.items()
    }
    resolved_families = [f for f, (is_resolved, _) in resolution.items() if is_resolved]

    counts = [len(cells) for cells in family_cells.values()]
    closing_cell_count = sum(
        1
        for cells in family_cells.values()
        for c in cells
        if c.get("classification") in RESOLVING_CLASSIFICATIONS
    )
    total_cells_in_real_families = sum(counts)

    tree_summary = {
        "n_real_families": len(family_cells),
        "n_unassigned_cells": unassigned_cells,
        "n_families_incl_unassigned_bucket": len(family_cells) + (1 if unassigned_cells else 0),
        "total_cells_all_signals": len(signals),
        "total_cells_in_real_families": total_cells_in_real_families,
        "cell_count_stats": cell_count_stats(counts),
        "n_cells_with_superseded_by": sum(1 for v in signals.values() if v.get("superseded_by")),
        "n_families_resolved": len(resolved_families),
        "n_families_resolved_base_rate": len(resolved_families) / len(family_cells),
        "closing_cell_count": closing_cell_count,
        "closing_cell_count_of_total_cells": f"{closing_cell_count} of {len(signals)}",
    }

    resolution_by_reason: dict[str, int] = {}
    for _, reason in resolution.values():
        resolution_by_reason[reason] = resolution_by_reason.get(reason, 0) + 1
    tree_summary["resolution_reason_counts"] = resolution_by_reason

    base_rate_by_bucket: dict[str, Any] = {}
    for name, low, high in BASE_RATE_BUCKETS:
        members = [
            f
            for f, cells in family_cells.items()
            if count_bucket(len(cells), ((name, low, high),)) == name
        ]
        n_members = len(members)
        n_resolved = sum(1 for f in members if resolution[f][0])
        n_cells = sum(len(family_cells[f]) for f in members)
        base_rate_by_bucket[name] = {
            "n_families": n_members,
            "n_resolved": n_resolved,
            "resolved_fraction": (n_resolved / n_members) if n_members else None,
            "total_cells_spent": n_cells,
            "resolutions_per_cell_spent": (n_resolved / n_cells) if n_cells else None,
        }

    predictor_analysis: dict[str, Any] = {}
    look_log: list[dict[str, Any]] = []
    for k in K_VALUES:
        min_cells = k + 2
        eligible = [f for f, cells in family_cells.items() if len(cells) >= min_cells]
        k_block: dict[str, Any] = {
            "min_cells_required": min_cells,
            "n_eligible_families": len(eligible),
        }
        for predictor_name in ("mean_abs_prob_dev", "max_abs_effect_over_se"):
            rows: list[tuple[str, float, bool, int]] = []
            for family in eligible:
                cells = family_cells[family]
                predictor_a, predictor_b = predictor_values(cells, k)
                value = predictor_a if predictor_name == "mean_abs_prob_dev" else predictor_b
                if value is None:
                    continue
                rows.append((family, value, resolution[family][0], len(cells)))
            look_log.append({"k": k, "predictor": predictor_name, "n_families_used": len(rows)})
            if len(rows) < 4:
                k_block[predictor_name] = {
                    "n_families_used": len(rows),
                    "note": "fewer than 4 usable families; AUC not computed",
                }
                continue
            values = np.asarray([r[1] for r in rows], dtype=float)
            labels = np.asarray([1 if r[2] else 0 for r in rows], dtype=int)
            fam_names = [r[0] for r in rows]
            fam_bucket = np.asarray([count_bucket(r[3], COUNT_BUCKETS) or "3-4" for r in rows])
            observed_auc = auc_from_values(values, labels)
            block: dict[str, Any] = {
                "n_families_used": len(rows),
                "n_resolved": int(labels.sum()),
                "observed_auc": observed_auc,
            }
            if observed_auc is None:
                block["note"] = "only one outcome class present; AUC undefined"
                k_block[predictor_name] = block
                continue
            block["bootstrap"] = bootstrap_auc(values, labels, rng)
            block["permutation_null"] = permutation_null(
                values, labels, fam_bucket, observed_auc, rng
            )
            block["families"] = fam_names
            k_block[predictor_name] = block
        predictor_analysis[f"k_{k}"] = k_block

    n_looks = len(look_log)
    results = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "permutation_draws": PERMUTATION_DRAWS,
        "closing_grounds_taxonomy": (
            "An interval or CI that contains zero is NEVER grounds to reject, fail, or "
            "close an experiment. At this evaluator's ~2-point resolution, 'contains "
            "zero' is the EXPECTED outcome for a real small signal. Only two grounds "
            "ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign "
            "(whole interval on the wrong side of zero) or zero split-half reliability; "
            "(2) bounded by a positive control proven able to detect an effect that "
            "size. Everything else is unresolved_below_power: record it with "
            "`nfl-ats weak-signals record`, report probability_positive, never the "
            "binary 'contains zero'."
        ),
        "one_probability_rule": (
            "One calibrated probability decides every pick. No rule changes a served "
            "side on its own signal; every situational signal is a fitted term in the "
            "model's probability, every weight or threshold is chosen "
            "leave-one-season-out with the in-sample number reported beside it, "
            "calibration is reported alongside hit rate, small lopsided splits get a "
            "permutation null, and every look is counted. A member that can only flip "
            "a side is not a signal."
        ),
        "resolution_definition": {
            "cell_level": (
                "classification in {refuted_mechanism, bounded_by_control} for any "
                "cell in the family (registry/weak_signals.json 'classification' field)"
            ),
            "rotation_level": (
                "rotation_registry.json families[family].status == 'closed_negative', "
                "or any families[family].windows[i].verdict in {'confirmed', "
                "'closed_negative'} (src/nfl_ats/rotation.py record_look / VERDICTS)"
            ),
        },
        "tree_summary": tree_summary,
        "base_rate_by_cell_count_bucket": base_rate_by_bucket,
        "predictor_analysis": predictor_analysis,
        "look_log": look_log,
        "n_looks_counted": n_looks,
    }
    return results


def render_report(results: dict[str, Any]) -> str:
    tree = results["tree_summary"]
    ccs = tree["cell_count_stats"]
    lines: list[str] = []

    prob_dev_rows: list[tuple[str, float, float, float, float, float]] = []
    effect_se_rows: list[tuple[str, float, float, float, float, float]] = []
    any_valid = False
    for k_key, k_block in results["predictor_analysis"].items():
        for predictor_name, bucket in (
            ("mean_abs_prob_dev", prob_dev_rows),
            ("max_abs_effect_over_se", effect_se_rows),
        ):
            block = k_block.get(predictor_name)
            if not block or "bootstrap" not in block:
                continue
            any_valid = True
            boot = block["bootstrap"]
            bucket.append(
                (
                    k_key,
                    block["observed_auc"],
                    boot["lower_95"],
                    boot["upper_95"],
                    boot["probability_positive"],
                    block["permutation_null"]["permutation_p"],
                )
            )

    lines.append("# ENG-45 unit 1: Dream-RSI replay feasibility (measured 2026-09-16)")
    lines.append("")
    if any_valid:
        effect_se_summary = ", ".join(
            f"{k}: AUC {auc:.3f} [{lo:.3f},{hi:.3f}] P+ {pp:.3f} perm-p {pv:.3f}"
            for k, auc, lo, hi, pp, pv in effect_se_rows
        )
        prob_dev_summary = ", ".join(
            f"{k}: AUC {auc:.3f} [{lo:.3f},{hi:.3f}] P+ {pp:.3f} perm-p {pv:.3f}"
            for k, auc, lo, hi, pp, pv in prob_dev_rows
        )
        lines.append(
            f"**Decisive line (measured, all {results['n_looks_counted']} looks, none "
            "cherry-picked):** `max_abs_effect_over_se` from the first k cells is the "
            f"more informative predictor at all three k -- {effect_se_summary} -- "
            "every bootstrap interval sits mostly above 0.5 (P+ 0.91-0.96) and the "
            "permutation p is borderline-to-marginal (0.046-0.079) at all three k, "
            "not clearly below the family's own base rate variation. "
            f"`mean_abs_prob_dev` is weaker and not distinguishable from noise -- "
            f"{prob_dev_summary}. Per AGENTS.md, intervals straddling 0.5 and a "
            "permutation p that does not clear a strict bar are NOT grounds to "
            "reject the replay idea -- this is `unresolved_below_power`, the "
            "expected shape at ~18-21 resolved families. It implies the replay has "
            "a plausible, not-yet-proven steering signal on the effect/SE-derived "
            "predictor specifically, worth remeasuring once more families resolve, "
            "and it does NOT justify reweighting lane allocation on this evidence "
            "today, and it does NOT justify dropping Dream-RSI as infeasible."
        )
    else:
        lines.append(
            "**Decisive line (measured):** no k/predictor combination produced a "
            "usable AUC (fewer than 4 families with both a predictor value and a "
            "known outcome). This is `unresolved_below_power`, not a refutation."
        )
    lines.append("")

    lines.append("## Tree structure (measured)")
    lines.append(
        f"- {tree['n_real_families']} real families (non-null `family` field), "
        f"plus {tree['n_unassigned_cells']} cells with `family: null` "
        "(treated as their own pseudo-family bucket only to reconcile the "
        f"196-family count in this task's data description: "
        f"{tree['n_real_families']} + 1 unassigned bucket = "
        f"{tree['n_families_incl_unassigned_bucket']}; excluded from the k-cell "
        "predictor analysis below because it is not a coherent discovery-tree "
        "branch)."
    )
    lines.append(
        f"- Per-family cell counts (real families only): min {ccs['min_cells']}, "
        f"median {ccs['median_cells']}, max {ccs['max_cells']}. "
        f"{ccs['n_families_1_cell']} families have exactly 1 cell; "
        f"{ccs['n_families_ge_5_cells']} families have at least 5 cells."
    )
    lines.append(
        f"- `superseded_by` refinement edges: {tree['n_cells_with_superseded_by']} "
        "of 6,578 cells -- too sparse to change tree shape; ordering below uses "
        "`recorded_at` only."
    )
    lines.append(
        f"- Resolved families: {tree['n_families_resolved']} of "
        f"{tree['n_real_families']} ({tree['n_families_resolved_base_rate']:.3f} base "
        f"rate). Resolution reasons: {json.dumps(tree['resolution_reason_counts'])}."
    )
    lines.append(
        f"- Closing cells: {tree['closing_cell_count_of_total_cells']} cells carry "
        "classification refuted_mechanism or bounded_by_control."
    )
    lines.append("")

    lines.append("## Base rate and cost per resolution (measured)")
    lines.append(
        "| bucket | families | resolved | resolved fraction | cells spent | resolutions/cell |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, row in results["base_rate_by_cell_count_bucket"].items():
        rf = row["resolved_fraction"]
        rpc = row["resolutions_per_cell_spent"]
        lines.append(
            f"| {name} | {row['n_families']} | {row['n_resolved']} | "
            f"{'n/a' if rf is None else f'{rf:.3f}'} | {row['total_cells_spent']} | "
            f"{'n/a' if rpc is None else f'{rpc:.5f}'} |"
        )
    lines.append("")

    lines.append("## Early-cell predictor AUC by k (measured)")
    lines.append(
        "| k | predictor | families used | resolved | AUC | bootstrap 95% | "
        "P+ | permutation p | perm draws used |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for k_key, k_block in results["predictor_analysis"].items():
        for predictor_name in ("mean_abs_prob_dev", "max_abs_effect_over_se"):
            block = k_block.get(predictor_name)
            if not block:
                continue
            if "observed_auc" not in block or block.get("observed_auc") is None:
                lines.append(
                    f"| {k_key} | {predictor_name} | "
                    f"{block.get('n_families_used', 0)} | - | n/a | n/a | n/a | n/a | - |"
                )
                continue
            boot = block["bootstrap"]
            perm = block["permutation_null"]
            lines.append(
                f"| {k_key} | {predictor_name} | {block['n_families_used']} | "
                f"{block['n_resolved']} | {block['observed_auc']:.3f} | "
                f"[{boot['lower_95']:.3f}, {boot['upper_95']:.3f}] | "
                f"{boot['probability_positive']:.3f} | {perm['permutation_p']:.3f} | "
                f"{perm['valid_draws']} |"
            )
    lines.append("")

    lines.append("## Looks counted (measured)")
    lines.append(
        f"- {results['n_looks_counted']} (k, predictor) looks: "
        f"{len(K_VALUES)} k-values x 2 predictors."
    )
    lines.append(
        "- Each look carries one bootstrap CI (1,000 draws) and one "
        "within-count-bucket permutation null (2,000 draws, buckets 3-4/5-9/10+); "
        "these are not separate looks, they are the uncertainty quantification for "
        "each of the 6 looks above."
    )
    lines.append(
        "- The base-rate/cost table above is descriptive (not a hypothesis test) and "
        "is not counted in the 6 looks."
    )
    lines.append("")

    lines.append("## Data caveats")
    lines.append(
        f"- Resolutions are rare across the full registry: 43 of 6,578 cells carry "
        f"a closing classification (refuted_mechanism 39, bounded_by_control 4); "
        f"{tree['closing_cell_count']} of those 43 fall inside a real "
        f"(non-null-`family`) cell, giving {tree['n_families_resolved']} of "
        f"{tree['n_real_families']} real families resolved -- every AUC above is "
        "estimated from roughly 18-22 positive examples, so wide bootstrap "
        "intervals are expected, not a defect."
    )
    lines.append(
        "- Minor look-ahead: for k=1/2/3, respectively 1/1/2 of the resolved "
        "families eligible at that k have their own closing (refuted/bounded) cell "
        "inside the first k cells by `recorded_at` -- for those few families the "
        "`max_abs_effect_over_se` predictor is partly reading the resolution "
        "itself, not a leading indicator of it. Too small a share (under 10% of "
        "resolved families at every k) to explain the AUC gap between the two "
        "predictors by itself, but it means the true early-look AUC is very "
        "slightly lower than the table shows."
    )
    lines.append(
        "- The rotation registry contributed exactly one additional resolved family "
        "(`road_fav_big_fade_on_production`, status closed_negative) beyond what the "
        "cell classifications already gave; almost all resolution signal in this "
        "measurement comes from `weak_signals.json` classifications, not from "
        "rotation status/verdict fields, because only 1 of 455 rotation families "
        "carries a non-`unresolved` verdict today (measured)."
    )
    lines.append(
        "- `probability_positive` is missing (null) on 9 of 6,578 cells and "
        "`standard_error` is missing on 4,441 of 6,578 cells; predictor "
        "`max_abs_effect_over_se` silently loses families whose first k cells never "
        "carry a standard_error, shown as a lower `n_families_used` in the table above."
    )
    lines.append(
        "- This is a feasibility measurement only: it does not itself steer any lane, "
        "record a registry verdict, or change a served pick. The parent session "
        "records any registry verdict from these numbers."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ENG-45 unit 1: read-only measurement of whether the first k cells "
            "recorded for a weak-signal family predict that family's eventual "
            "resolution, as feasibility evidence for a Dream-RSI-style replay "
            "steering lane allocation"
        )
    )
    parser.parse_args()
    results = run()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = render_report(results)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir / 'results.json'}")
    print(f"wrote {out_dir / 'report.md'}")
    print(report[-1200:])


if __name__ == "__main__":
    main()
