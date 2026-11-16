from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import xlg_early_overreaction_cfb as eo  # noqa: E402

XP, M36, u1, x14, p2q = eo.XP, eo.M36, eo.u1, eo.x14, eo.p2q
OUT = ROOT / "artifacts" / "t_signrule"
RESULTS = OUT / "results.json"
FAMILY = "t_signrule_cfb"
COL = "F"
INDEX = len(XP.TEMPLATES) + len(eo.WINDOWS)
FIRST_HALF_SEASONS = 6


def build():
    reg = eo.load_reg()
    d2 = p2q.surprise_values(reg)
    reg["D2"] = d2.reindex(reg["game_id"]).to_numpy()
    reg["count"] = eo.prior_counts(reg).reindex(reg["game_id"]).to_numpy()
    reg["window"] = eo.window_labels(reg)
    d = x14.cfb_frame()
    t = reg.set_index("game_id")
    win = t["window"].reindex(d["game_id"]).fillna("none").to_numpy()
    raw = t["D2"].reindex(d["game_id"]).to_numpy(float)
    d[COL] = np.where(win == "early", -np.sign(np.nan_to_num(raw)), 0.0)
    return d


def halves(d):
    seasons = sorted(int(s) for s in d["season"].unique())
    split = {"first": seasons[:FIRST_HALF_SEASONS], "second": seasons[FIRST_HALF_SEASONS:]}
    base_p, _ = u1.loso(d, ["logit_p"])
    term_p, _ = u1.loso(d, ["logit_p", COL])
    sc = d.assign(p_base=base_p, p_term=term_p)

    def gain(f):
        return 100.0 * (
            ((f["p_term"] >= 0.5).astype(int) == f["y"]).mean()
            - ((f["p_base"] >= 0.5).astype(int) == f["y"]).mean()
        )

    def llg(f):
        return u1.metrics(f, "p_base")["log_loss"] - u1.metrics(f, "p_term")["log_loss"]

    out = {}
    for name, ss in split.items():
        f = sc.loc[sc["season"].isin(ss)]
        rng = np.random.default_rng([XP.SEED, INDEX, len(out)])
        dec = f.loc[(f["p_term"] >= 0.5) != (f["p_base"] >= 0.5)]
        out[name] = {
            "seasons": ss,
            "games": len(f),
            "changed": len(dec),
            "term_right": int(((dec["p_term"] >= 0.5).astype(int) == dec["y"]).sum()),
            "accuracy_points": u1.summarize(u1.block_bootstrap(f, gain, rng), float(gain(f))),
            "log_loss": u1.summarize(u1.block_bootstrap(f, llg, rng), float(llg(f))),
            "coefficient": float(u1.fit(d.loc[d["season"].isin(ss)], ["logit_p", COL]).coef_[0][-1]),
        }
    return out


def ladder(st, tpl, null_sd):
    cells, reached = [], {}
    for k in range(1, M36.LADDER_RUNGS + 1):
        size = k * null_sd
        rng = np.random.default_rng([XP.SEED, XP.STREAM_ADDONE, INDEX, k])
        tabs = XP.plant_tables(st, size)
        rows = []
        for i in range(XP.DRAWS):
            M36.draw_rows(st, tpl, "planted", size, i, XP.planted_from(st, tabs, rng), rows)
        g = pd.DataFrame(rows)
        g = g.loc[g["reference"] == "served"]
        cell = {
            "size_logit_per_sd": float(size),
            "true_accuracy_points": float(g["oracle_acc"].mean()),
            "share_seen_ll": float((g["pp_ll"] >= M36.NOTABLE[0]).mean()),
            "share_seen_acc": float((g["pp_acc"] >= M36.NOTABLE[0]).mean()),
        }
        cells.append(cell)
        for m in eo.SEEN_BY:
            if m not in reached and cell[f"share_seen_{m}"] >= M36.POWER_TARGET:
                reached[m] = cell
        if len(reached) == len(eo.SEEN_BY):
            break
    return {"grid": cells, "smallest": reached}


def compute():
    started = time.time()
    d = build()
    res = {"family": FAMILY, "harness_games": len(d), "nonzero": int((d[COL] != 0).sum())}
    res["added_term"] = eo.addone(d, COL)
    print("added term", f"{time.time() - started:.0f}s", flush=True)
    res["halves"] = halves(d)
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            M36.SERVED_COLUMN: expit(d["logit_p"].to_numpy(float)),
            COL: d[COL].to_numpy(float),
        }
    )
    tpl = {"name": "t_signrule_fade", "column": COL, "intercept": False}
    st, null = eo.matched_null(frame, tpl, INDEX)
    res["matched_null"] = null
    RESULTS.write_text(json.dumps(res, indent=1, default=eo.plain), encoding="utf-8")
    print("matched null", f"{time.time() - started:.0f}s", flush=True)
    res["positive_control"] = ladder(st, tpl, null["null_coefficient_sd"])
    a, h = res["added_term"], res["halves"]
    res["survivor"] = {
        "coefficient_positive": a["coef_full"][-1] > 0,
        "ll_percentile_at_least": null["percentile"]["ll"] >= 100 * M36.NOTABLE[0],
        "both_halves_positive": all(v["coefficient"] > 0 for v in h.values()),
    }
    res["survives"] = all(res["survivor"].values())
    res["elapsed_seconds"] = time.time() - started
    RESULTS.write_text(json.dumps(res, indent=1, default=eo.plain), encoding="utf-8")


def cell(name, desc, g, games, seasons):
    return {
        "name": name,
        "description": desc,
        "effect": g["point"],
        "interval_low": g["ci95"][0],
        "interval_high": g["ci95"][1],
        "probability_positive": g["probability_positive"],
        "season_start": min(seasons),
        "season_end": max(seasons),
        "sample_games": games,
    }


def report():
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    a, h, m = res["added_term"], res["halves"], res["matched_null"]
    allseasons = [s for v in h.values() for s in v["seasons"]]
    cells = []
    for metric, units in eo.ADDED_METRICS:
        c = cell(
            f"t_signrule_cfb_fade_{units}",
            (
                "College-first harness: fade sign of surprise to date (minus sign of T) in the "
                "early window as one added fitted term on the opener-input cover probability, "
                f"leave-one-season-out, {units} against the base; changed picks "
                f"{a['decisive_term_correct']}-{a['decisive_base_correct']}; matched-null "
                f"percentile {m['percentile']['acc' if units == 'accuracy_points' else 'll']:.1f}"
            ),
            a[metric],
            res["harness_games"],
            allseasons,
        )
        c["effect_units"] = units
        cells.append(c)
        for name, v in h.items():
            key = "accuracy_points" if units == "accuracy_points" else "log_loss"
            c = cell(
                f"t_signrule_cfb_fade_{name}_half_{units}",
                (
                    f"Same term, {name} chronological half of seasons, out-of-season {units}; "
                    f"changed picks {v['term_right']}-{v['changed'] - v['term_right']}; "
                    f"coefficient fit inside the half {v['coefficient']:+.4f}"
                ),
                v[key],
                v["games"],
                v["seasons"],
            )
            c["effect_units"] = units
            cells.append(c)
    batch = {
        "league": "cfb",
        "classification": "unresolved_below_power",
        "source": "artifacts/t_signrule/results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "One declared sign-rule form of the early-surprise term (MOD-57 T), declared in "
            "docs/lanes/t-signrule-2026-11-16.md before outcomes were joined; 6 looks."
        ),
        "classification_evidence": (
            "Season-week block bootstrap with probability_positive; matched within-season "
            "shuffle null; no closing ground claimed."
        ),
        "plain_summary": (
            "Whether fading a college team that has beaten the line early in a season, "
            "entered as one fitted term, improves the opener cover probability."
        ),
        "cells": cells,
    }
    (OUT / "batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    acc, ll, fc = a["oos_accuracy_points_gain"], a["oos_log_loss_gain"], a["fold_coefficient"]
    lines = [
        f"games {res['harness_games']} nonzero {res['nonzero']} elapsed {res['elapsed_seconds']:.0f}s",
        f"ADDED changed {a['decisive_pick_changed_oos']} term {a['decisive_term_correct']} base "
        f"{a['decisive_base_correct']} | acc {acc['point']:+.3f} [{acc['ci95'][0]:+.3f}, "
        f"{acc['ci95'][1]:+.3f}] P+ {acc['probability_positive']:.3f} | ll {ll['point']:+.6f} "
        f"[{ll['ci95'][0]:+.6f}, {ll['ci95'][1]:+.6f}] P+ {ll['probability_positive']:.3f}",
        f" in-sample ll {a['in_sample_log_loss_gain']:+.6f} gap {a['in_minus_out_log_loss_gain']:+.6f}"
        f" in-sample acc {a['in_sample_accuracy_points_gain']:+.3f} gap "
        f"{a['in_minus_out_accuracy_points_gain']:+.3f}",
        f" coef {a['coef_full'][-1]:+.5f} folds mean {fc['mean']:+.5f} [{fc['min']:+.5f}, "
        f"{fc['max']:+.5f}] negative {fc['negative_folds']}/{fc['folds']}",
        " folds " + " ".join(f"{s}:{v[-1]:+.4f}" for s, v in a["coef_per_heldout_season"].items()),
        f"CONTROL intercept-only {a['intercept_only_oos']}",
        f"CONTROL base {a['baseline_oos']}",
        f"TERM {a['term_oos']}",
    ]
    for name, v in h.items():
        lines.append(
            f"HALF {name} {v['seasons'][0]}-{v['seasons'][-1]} games {v['games']} changed "
            f"{v['term_right']}-{v['changed'] - v['term_right']} acc {v['accuracy_points']['point']:+.3f} "
            f"P+ {v['accuracy_points']['probability_positive']:.3f} ll {v['log_loss']['point']:+.6f} "
            f"P+ {v['log_loss']['probability_positive']:.3f} coef {v['coefficient']:+.4f}"
        )
    r, n, p = m["real"], m["null"], m["percentile"]
    lines.append(
        f"NULL real coef/SD {r['full_coefficient_per_sd']:+.4f} acc {r['served']['acc']['effect']:+.3f} "
        f"ll {r['served']['ll']['effect']:+.6f} | percentile acc {p['acc']:.1f} ll {p['ll']:.1f} "
        f"coef {p['coef']:.1f} | null ll {n['ll']['mean']:+.6f} [{n['ll']['q05']:+.6f}, "
        f"{n['ll']['q95']:+.6f}] coef sd {m['null_coefficient_sd']:.4f} draws {m['draws']}"
    )
    for k, c in res["positive_control"]["smallest"].items():
        lines.append(
            f"DETECT by {k}: {c['size_logit_per_sd']:.4f} per SD = {c['true_accuracy_points']:+.2f} pts"
        )
    for c in res["positive_control"]["grid"]:
        lines.append(
            f"  grid {c['size_logit_per_sd']:.4f} true {c['true_accuracy_points']:+.2f} "
            f"ll {c['share_seen_ll']:.3f} acc {c['share_seen_acc']:.3f}"
        )
    lines.append(f"SURVIVOR {res['survivor']} survives {res['survives']}")
    lines.append(f"cells {len(cells)}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] != ["report"]:
        compute()
    report()


if __name__ == "__main__":
    main()
