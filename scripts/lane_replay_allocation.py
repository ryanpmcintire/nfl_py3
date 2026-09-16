from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lane_replay_feasibility as unit1

OUT_ROOT = Path("artifacts/lane_replay")
SPEC_CUT = "2026-08-01"
OPERATIONAL_CUT = "2026-09-05"
EARLY_K = 2
CANDIDATE_CUTS = (0.5, 1.0, 1.5, 2.0, 3.0)
BOOTSTRAP_DRAWS = 1000
PERMUTATION_DRAWS = 2000
SEED = 20260917
SIZE_BUCKETS = (("1-2", 1, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, None))


def size_bucket(n_cells):
    for name, low, high in SIZE_BUCKETS:
        if high is None:
            if n_cells >= low:
                return name
        elif low <= n_cells <= high:
            return name
    return "10+"


def early_score(cells, k):
    pool = [c for c in cells[:k] if c.get("classification") not in unit1.RESOLVING_CLASSIFICATIONS]
    ratios = []
    for c in pool:
        se = c.get("standard_error")
        effect = c.get("effect")
        if se is None or effect is None or se == 0:
            continue
        ratios.append(abs(float(effect) / float(se)))
    if not ratios:
        return None
    return float(max(ratios))


def split_families(family_cells, cut):
    pre = {}
    post = {}
    for family, cells in family_cells.items():
        first_seen = min(c["recorded_at"] for c in cells)
        if first_seen < cut:
            pre[family] = cells
        else:
            post[family] = cells
    return pre, post


def family_outcomes(families, rotation_families):
    info = {}
    for family, cells in families.items():
        resolved, _ = unit1.family_resolution(cells, rotation_families.get(family))
        closing_ordinals = [
            i
            for i, c in enumerate(cells)
            if c.get("classification") in unit1.RESOLVING_CLASSIFICATIONS
        ]
        if closing_ordinals:
            closing_ordinal = min(closing_ordinals)
        elif resolved:
            closing_ordinal = len(cells) - 1
        else:
            closing_ordinal = None
        info[family] = {
            "n_cells": len(cells),
            "resolved": resolved,
            "closing_ordinal": closing_ordinal,
            "score": early_score(cells, EARLY_K),
            "first_seen": min(c["recorded_at"] for c in cells),
        }
    return info


def order_cells(families, info, policy, cut_value=None):
    if policy == "recorded" or policy == "breadth":
        ranked_families = sorted(families, key=lambda f: (info[f]["first_seen"], f))
    else:
        top = sorted(
            [f for f in families if info[f]["score"] is not None and info[f]["score"] >= cut_value],
            key=lambda f: (-info[f]["score"], f),
        )
        rest = sorted(
            [
                f
                for f in families
                if not (info[f]["score"] is not None and info[f]["score"] >= cut_value)
            ],
            key=lambda f: (info[f]["first_seen"], f),
        )
        ranked_families = top + rest
    order = []
    if policy == "breadth":
        max_len = max(len(families[f]) for f in families)
        for ordinal in range(max_len):
            round_cells = []
            for family in ranked_families:
                cells = families[family]
                if ordinal < len(cells):
                    round_cells.append(cells[ordinal]["cell_id"])
            round_cells.sort()
            order.extend(round_cells)
        return order
    for family in ranked_families:
        order.extend(c["cell_id"] for c in families[family])
    return order


def reached_at_budget(order, families, info, budget):
    position = {cell_id: i for i, cell_id in enumerate(order)}
    reached = 0
    for family, cells in families.items():
        if not info[family]["resolved"]:
            continue
        closing_cell = cells[info[family]["closing_ordinal"]]["cell_id"]
        if position[closing_cell] < budget:
            reached += 1
    return reached


def rpc_for_cut(families, info, cut_value):
    members = [
        f for f in families if info[f]["score"] is not None and info[f]["score"] >= cut_value
    ]
    cells_spent = sum(info[f]["n_cells"] for f in members)
    resolved = sum(1 for f in members if info[f]["resolved"])
    if not cells_spent:
        return None, 0, 0, 0
    return resolved / cells_spent, cells_spent, resolved, len(members)


def run_split(families, info, label):
    scored = [f for f in families if info[f]["score"] is not None]
    look_log = []
    best = None
    for cut_value in CANDIDATE_CUTS:
        rpc, spent, resolved, n_members = rpc_for_cut(families, info, cut_value)
        look_log.append(
            {
                "cut": cut_value,
                "rpc": rpc,
                "cells_spent": spent,
                "resolved": resolved,
                "n_families": n_members,
            }
        )
        if rpc is None:
            continue
        key = (rpc, -spent, -cut_value)
        if best is None or key > best[0]:
            best = (key, cut_value, rpc, spent, resolved, n_members)
    total_resolved = sum(1 for f in families if info[f]["resolved"])
    total_cells = sum(info[f]["n_cells"] for f in families)
    full_rpc = (total_resolved / total_cells) if total_cells else None
    return {
        "label": label,
        "n_families": len(families),
        "n_scored": len(scored),
        "n_resolved": total_resolved,
        "total_cells": total_cells,
        "full_spend_rpc": full_rpc,
        "look_log": look_log,
        "best": (
            {
                "cut": best[1],
                "rpc": best[2],
                "cells_spent": best[3],
                "resolved": best[4],
                "n_families": best[5],
            }
            if best is not None
            else None
        ),
    }


def score_post(families, info, frozen_cut, rng):
    order_ref = order_cells(families, info, "refine", frozen_cut)
    order_bre = order_cells(families, info, "breadth")
    order_rec = order_cells(families, info, "recorded")
    _, budget, _, _ = rpc_for_cut(families, info, frozen_cut)
    budget = int(budget)
    out = {"frozen_cut": frozen_cut, "matched_budget_cells": budget}
    if not budget:
        out["note"] = "frozen cut selects no post families so no matched budget exists"
        return out
    reached = {}
    rpc = {}
    for name, order in (("refine", order_ref), ("breadth", order_bre), ("recorded", order_rec)):
        reached[name] = reached_at_budget(order, families, info, budget)
        rpc[name] = reached[name] / budget
    out["reached"] = reached
    out["rpc_at_matched_budget"] = rpc
    out["diff_refine_minus_recorded"] = rpc["refine"] - rpc["recorded"]
    out["diff_refine_minus_breadth"] = rpc["refine"] - rpc["breadth"]
    fam_list = list(families)
    rec_list = np.asarray(
        [reached_at_budget(order_rec, {f: families[f]}, info, budget) for f in fam_list],
        dtype=float,
    )
    ref_list = np.asarray(
        [reached_at_budget(order_ref, {f: families[f]}, info, budget) for f in fam_list],
        dtype=float,
    )
    single_orders_note = "per-family reached evaluated against frozen post orders"
    out["bootstrap_note"] = single_orders_note
    diffs = []
    n = len(fam_list)
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, n)
        diffs.append((ref_list[idx].sum() - rec_list[idx].sum()) / budget)
    arr = np.asarray(diffs, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    out["bootstrap_diff_refine_minus_recorded"] = {
        "lower_95": float(lo),
        "upper_95": float(hi),
        "probability_positive": float((arr > 0).mean() + 0.5 * (arr == 0).mean()),
    }
    buckets = np.asarray([size_bucket(info[f]["n_cells"]) for f in fam_list])
    scores = np.asarray(
        [info[f]["score"] if info[f]["score"] is not None else np.nan for f in fam_list]
    )
    observed = out["diff_refine_minus_recorded"]
    null_diffs = []
    for _ in range(PERMUTATION_DRAWS):
        perm_scores = scores.copy()
        for bucket in sorted(set(buckets.tolist())):
            idx = np.flatnonzero(buckets == bucket)
            perm_scores[idx] = rng.permutation(scores[idx])
        perm_info = dict(info)
        for f, s in zip(fam_list, perm_scores, strict=True):
            entry = dict(info[f])
            entry["score"] = None if bool(np.isnan(s)) else float(s)
            perm_info[f] = entry
        perm_order = order_cells(families, perm_info, "refine", frozen_cut)
        perm_reached = reached_at_budget(perm_order, families, info, budget)
        null_diffs.append(perm_reached / budget - rpc["recorded"])
    null_arr = np.asarray(null_diffs, dtype=float)
    out["permutation_diff_refine_minus_recorded"] = {
        "permutation_p": float((null_arr >= observed).mean()),
        "null_mean": float(null_arr.mean()),
    }
    return out


def run():
    rng = np.random.default_rng(SEED)
    signals = unit1.load_weak_signals()
    rotation_families = unit1.load_rotation_families()
    family_cells, unassigned = unit1.build_family_cells(signals)
    info = family_outcomes(family_cells, rotation_families)
    spec_pre, spec_post = split_families(family_cells, SPEC_CUT)
    spec_pre_info = {f: info[f] for f in spec_pre}
    spec_fit = run_split(spec_pre, spec_pre_info, "spec_pre_2026_08_01")
    op_pre, op_post = split_families(family_cells, OPERATIONAL_CUT)
    op_pre_info = {f: info[f] for f in op_pre}
    op_post_info = {f: info[f] for f in op_post}
    op_fit = run_split(op_pre, op_pre_info, "operational_pre_2026_09_05")
    results = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "spec_cut": SPEC_CUT,
        "operational_cut": OPERATIONAL_CUT,
        "early_window_k": EARLY_K,
        "candidate_cuts": list(CANDIDATE_CUTS),
        "n_unassigned_cells_excluded": unassigned,
        "spec_split": {
            "n_pre_families": len(spec_pre),
            "n_post_families": len(spec_post),
            "fit": spec_fit,
        },
        "operational_fit": op_fit,
        "operational_post": {
            "n_families": len(op_post),
            "n_resolved": sum(1 for f in op_post if info[f]["resolved"]),
            "total_cells": sum(info[f]["n_cells"] for f in op_post),
        },
    }
    if op_fit["best"] is not None:
        results["operational_score"] = score_post(op_post, op_post_info, op_fit["best"]["cut"], rng)
    else:
        results["operational_score"] = {"note": "no fittable cut in operational pre set"}
    results["tie_break_rule"] = "top rpc wins ties break to fewer cells then lower cut"
    results["n_fitting_looks"] = len(CANDIDATE_CUTS)
    results["n_fixed_policies"] = 2
    return results


def render_report(results):
    lines = []
    lines.append("ENG-45 unit 2 allocation replay measurement")
    lines.append("===========================================")
    lines.append("")
    spec = results["spec_split"]
    lines.append("Spec cut 2026-08-01 (measured)")
    lines.append("-----------------------------")
    lines.append(
        "Pre-cut families "
        + str(spec["n_pre_families"])
        + " so no cut is fittable on the spec split "
        + "and nothing is scored from it. Raw numbers only, no verdict."
    )
    lines.append("")
    fit = results["operational_fit"]
    lines.append("Operational split 2026-09-05 (measured)")
    lines.append("--------------------------------------")
    lines.append(
        "Pre families "
        + str(fit["n_families"])
        + " scored "
        + str(fit["n_scored"])
        + " resolved "
        + str(fit["n_resolved"])
        + " cells "
        + str(fit["total_cells"])
        + " full-spend rpc "
        + ("n/a" if fit["full_spend_rpc"] is None else f"{fit['full_spend_rpc']:.5f}")
    )
    for row in fit["look_log"]:
        lines.append(
            "cut "
            + str(row["cut"])
            + " pre rpc "
            + ("n/a" if row["rpc"] is None else f"{row['rpc']:.5f}")
            + " cells "
            + str(row["cells_spent"])
            + " resolved "
            + str(row["resolved"])
            + " families "
            + str(row["n_families"])
        )
    best = fit["best"]
    if best is not None:
        lines.append(
            "Frozen cut "
            + str(best["cut"])
            + " with pre rpc "
            + f"{best['rpc']:.5f}"
            + " on "
            + str(best["cells_spent"])
            + " cells."
        )
    lines.append("")
    post = results["operational_post"]
    lines.append("Post-cut scoring (measured)")
    lines.append("---------------------------")
    lines.append(
        "Post families "
        + str(post["n_families"])
        + " resolved "
        + str(post["n_resolved"])
        + " cells "
        + str(post["total_cells"])
    )
    score = results["operational_score"]
    if "rpc_at_matched_budget" in score:
        lines.append("Matched budget cells " + str(score["matched_budget_cells"]))
        for name in ("recorded", "breadth", "refine"):
            lines.append(
                name
                + " rpc "
                + f"{score['rpc_at_matched_budget'][name]:.5f}"
                + " reached "
                + str(score["reached"][name])
            )
        lines.append("Diff refine minus recorded " + f"{score['diff_refine_minus_recorded']:.5f}")
        lines.append("Diff refine minus breadth " + f"{score['diff_refine_minus_breadth']:.5f}")
        boot = score["bootstrap_diff_refine_minus_recorded"]
        lines.append(
            "Bootstrap diff 95pct ["
            + f"{boot['lower_95']:.5f}"
            + ", "
            + f"{boot['upper_95']:.5f}"
            + "] Pplus "
            + f"{boot['probability_positive']:.3f}"
        )
        perm = score["permutation_diff_refine_minus_recorded"]
        lines.append(
            "Permutation p "
            + f"{perm['permutation_p']:.3f}"
            + " null mean "
            + f"{perm['null_mean']:.5f}"
        )
    else:
        lines.append(score.get("note", "no post score"))
    lines.append("")
    lines.append("Looks counted (measured)")
    lines.append("------------------------")
    lines.append(
        "Fitting looks "
        + str(results["n_fitting_looks"])
        + " candidate cuts plus "
        + str(results["n_fixed_policies"])
        + " fixed policies with no cut."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ENG-45 unit 2 replay of allocation policies by resolutions per cell"
    )
    parser.parse_args()
    results = run()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = render_report(results)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(str(out_dir / "results.json"))
    print(str(out_dir / "report.md"))
    print(report)


if __name__ == "__main__":
    main()
