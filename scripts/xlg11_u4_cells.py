# ruff: noqa: E501
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "xlg11"
FAMILY = "xlg11_cfb_centered_ridge"
ERAS = {"nfl_opener_2020_25": ("20_25", 2020, 2025), "nfl_proxy_2011_19": ("11_19", 2011, 2019)}
EVIDENCE = (
    "Week-block paired bootstrap interval reported; no positive control of this size; "
    "no closing ground claimed by this unit."
)
ONE_SIDED = " Interval lies wholly on one side of zero; adjudication left to the orchestrator."
H14_NOTE = (
    "Unit 4, pooled 14-input opener-input harness, weekly walk-forward; strength chosen by inner "
    "leave-one-season-out; 20 declared looks in the unit."
)
H88_NOTE = (
    "Unit 4, production opener evaluator on the 88-input base with an in-memory estimator swap; "
    "college centre on the 14 analogue inputs and zero elsewhere; 20 declared looks in the unit."
)
PLAIN = {
    "real": "Starting the pro model's weights from what the same stats say in college football changed its picks against the opening line by an amount too small to tell from chance.",
    "zero": "Shrinking the pro model's weights harder toward nothing, with no college information, changed its picks against the opening line by an amount too small to tell from chance.",
    "gap": "The part of the college-start change that comes from the college information itself, rather than from the heavier shrinking, is too small to tell from chance.",
    "placebo": "College weights assigned to the right stats did about as well against the opening line as the same weights shuffled onto the wrong stats, within chance.",
    "control": "A no-information comparison model, used to show what the test can and cannot see.",
}


def cell(name, desc, units, effect, low, high, pp, n, blocks, seasons, source, note, plain, cat):
    evidence = EVIDENCE + (ONE_SIDED if (low > 0 or high < 0) else "")
    return {
        "name": name,
        "description": desc,
        "source": source,
        "effect_units": units,
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": seasons[0],
        "season_end": seasons[1],
        "family": FAMILY,
        "category": cat,
        "classification_evidence": evidence,
        "notes": note,
        "effect": effect,
        "interval_low": low,
        "interval_high": high,
        "probability_positive": pp,
        "sample_games": n,
        "sample_blocks": blocks,
        "plain_summary": plain,
    }


def boot_cell(name, c, metric, units, seasons, source, note, plain, cat="modeling"):
    e = c[metric]
    desc = (
        f"{c.get('arm', 'arm')} {c['arm_wins']}-{c['n'] - c['arm_wins']} versus "
        f"{c.get('ref', 'served base')} {c['ref_wins']}-{c['n'] - c['ref_wins']}, changed picks "
        f"{c['flips_win']}-{c['flips_loss']}; {metric} difference, paired week-block bootstrap."
    )
    return cell(
        name,
        desc,
        units,
        e["effect"],
        e["low"],
        e["high"],
        e["p_pos"],
        c["n"],
        e["blocks"],
        seasons,
        source,
        note,
        plain,
        cat,
    )


def placebo_cell(name, p, n, blocks, seasons, source, note):
    desc = (
        f"College centre {p['real_wins']} wins versus {p['count']} permuted centres mean "
        f"{p['placebo_wins_mean']:.1f} (sd {p['placebo_wins_sd']:.1f}); {p['placebos_at_or_above_real']} "
        "placebos at or above the real centre; interval is the placebo distribution."
    )
    return cell(
        name,
        desc,
        "accuracy_points",
        p["effect"],
        p["low"],
        p["high"],
        p["share_beaten"],
        n,
        blocks,
        seasons,
        source,
        note,
        PLAIN["placebo"],
        "modeling",
    )


def line(tag, c):
    e = c["accuracy"]
    return (
        f"{tag}: {c['arm_wins']} vs {c['ref_wins']} of {c['n']} flips {c['flips_win']}-{c['flips_loss']} "
        f"acc {e['effect']:+.2f} [{e['low']:.2f},{e['high']:.2f}] P+ {e['p_pos']:.3f} | "
        f"ll {c['arm_ll']:.5f}/{c['ref_ll']:.5f} P+ {c['log_loss']['p_pos']:.3f} | "
        f"br {c['arm_br']:.5f}/{c['ref_br']:.5f} P+ {c['brier']['p_pos']:.3f}"
    )


def main():
    cells = []
    p14 = OUT / "report_h14.json"
    if p14.exists():
        r = json.loads(p14.read_text())
        src = "artifacts/xlg11/report_h14.json"
        print("H14 identity", r["identity"], "mismatch", r["gram_mismatch"], r["lam_mismatch"])
        print("H14 columns", r["columns"])
        for pop, (tag, lo, hi) in ERAS.items():
            res = r["populations"][pop]
            for key in res:
                if key not in ("placebo", "in_out"):
                    print(pop, line(key, res[key]))
            print(pop, "placebo", res["placebo"])
            print(pop, "in_out", res["in_out"])
            s = (lo, hi)
            blocks = res["R_vs_base"]["accuracy"]["blocks"]
            n = res["R_vs_base"]["n"]
            cells += [
                boot_cell(
                    f"xlg11_h14_zero_same_minus_base_{tag}_u4",
                    res["Z_vs_base"],
                    "accuracy",
                    "accuracy_points",
                    s,
                    src,
                    H14_NOTE,
                    PLAIN["zero"],
                ),
                boot_cell(
                    f"xlg11_h14_real_minus_zero_same_{tag}_u4",
                    res["R_vs_Z"],
                    "accuracy",
                    "accuracy_points",
                    s,
                    src,
                    H14_NOTE,
                    PLAIN["gap"],
                ),
                placebo_cell(
                    f"xlg11_h14_real_minus_placebo_mean_{tag}_u4",
                    res["placebo"],
                    n,
                    blocks,
                    s,
                    src,
                    H14_NOTE,
                ),
                boot_cell(
                    f"xlg11_h14_int_only_minus_base_{tag}_u4",
                    res["int_only_vs_base"],
                    "accuracy",
                    "accuracy_points",
                    s,
                    src,
                    H14_NOTE,
                    PLAIN["control"],
                    "control",
                ),
                boot_cell(
                    f"xlg11_h14_line_only_minus_base_{tag}_u4",
                    res["line_only_vs_base"],
                    "accuracy",
                    "accuracy_points",
                    s,
                    src,
                    H14_NOTE,
                    PLAIN["control"],
                    "control",
                ),
            ]
        print("H14 strength", json.dumps(r["strength_by_season"]))
    p88 = OUT / "report_h88.json"
    if p88.exists():
        r = json.loads(p88.read_text())
        src = "artifacts/xlg11/report_h88.json"
        s = (2020, 2025)
        print("H88 identity", r.get("identity"), "shared", len(r["shared"]), r["market_columns"])
        arms = r["arms"]
        for key, c in arms.items():
            print("H88", line(key, c), "| market ll/br", c["market_ll"], c["market_br"])
            print("H88", key, "season boot", c["accuracy_season_boot"], c["per_season_d_acc"])
        for key, f in r["fits"].items():
            print("H88 fits", key, json.dumps(f))
        print("H88 placebo", json.dumps(r.get("placebo")))
        print("H88 market-only error", r.get("market_only_error"))
        spec = (
            ("R", "real_minus_served", PLAIN["real"], "modeling", True),
            ("Z", "zero_same_minus_served", PLAIN["zero"], "modeling", False),
            ("C10", "c10_minus_served", PLAIN["real"], "modeling", True),
            ("R_vs_Z", "real_minus_zero_same", PLAIN["gap"], "modeling", False),
            ("L", "market_inputs_only_minus_served", PLAIN["control"], "control", False),
        )
        for key, stem, plain, cat, extra in spec:
            if key not in arms:
                continue
            c = dict(arms[key])
            c["arm"] = key
            c["ref"] = "Z" if key == "R_vs_Z" else "served base"
            cells.append(
                boot_cell(
                    f"xlg11_h88_{stem}_20_25_u4",
                    c,
                    "accuracy",
                    "accuracy_points",
                    s,
                    src,
                    H88_NOTE,
                    plain,
                    cat,
                )
            )
            if extra:
                cells.append(
                    boot_cell(
                        f"xlg11_h88_{stem}_20_25_ll_u4",
                        c,
                        "log_loss",
                        "log_loss_improvement",
                        s,
                        src,
                        H88_NOTE,
                        plain,
                        cat,
                    )
                )
                cells.append(
                    boot_cell(
                        f"xlg11_h88_{stem}_20_25_brier_u4",
                        c,
                        "brier",
                        "brier_improvement",
                        s,
                        src,
                        H88_NOTE,
                        plain,
                        cat,
                    )
                )
        pl = r.get("placebo") or {}
        if pl.get("available") and "R" in arms:
            note = (
                H88_NOTE
                + f" Placebo graded by the {pl['rule']} sign rule, which matches the production pick "
                + f"on {100 * pl['agreement'][pl['rule']][0]:.1f}% of games."
                + " The sign rule failed its check, so this cell is not graded on the production"
                + " mapping and is indicative only."
            )
            cells.append(
                placebo_cell(
                    "xlg11_h88_real_minus_placebo_mean_signrule_20_25_u4",
                    pl["placebo"],
                    pl["covered"],
                    arms["R"]["accuracy"]["blocks"],
                    s,
                    src,
                    note,
                )
            )
    (OUT / "weak_signals_batch_u4.json").write_text(json.dumps({"cells": cells}, indent=1))
    print("cells", len(cells))
    for c in cells:
        print(
            " ",
            c["name"],
            c["effect_units"],
            round(c["effect"], 5),
            round(c["interval_low"], 5),
            round(c["interval_high"], 5),
            round(c["probability_positive"], 3),
        )


if __name__ == "__main__":
    main()
