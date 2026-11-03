# ruff: noqa: E501
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U2 = Path(__file__).resolve().parent / "mod54_unit2_penalty.py"
NS2 = {"__file__": str(U2), "__name__": "mod54_unit2_import"}
exec(compile(U2.read_text().rsplit("\nmain()", 1)[0], str(U2), "exec"), NS2)

from nfl_ats import clv  # noqa: E402
from nfl_ats import margin as margin_mod  # noqa: E402

u3 = NS2["u3"]
HARNESS = NS2["HARNESS"]
FULL = NS2["FULL"]
BASE_ALPHA = NS2["BASE_ALPHA"]
BASE5 = NS2["BASE5"]
compare = NS2["compare"]
ORIG = NS2["ORIG"]
MirrorPipe = NS2["MirrorPipe"]
STATE = NS2["STATE"]
LOG = NS2["LOG"]
install = NS2["install"]
SUPP = HARNESS["SUPP"]
regular_season_rows = HARNESS["regular_season_rows"]
OUT = u3.ROOT / "artifacts" / "mod54_unit3"
U2OUT = u3.ROOT / "artifacts" / "mod54_unit2"
MATCHED = 2 * BASE_ALPHA
ROW_FACTOR = 2.0
GRID = tuple(MATCHED * ROW_FACTOR**k for k in range(-2, 3))
FIRST_SEASON = 2011
LAST_SEASON = 2025
MIN_INNER_SEASONS = 2
BOOT = 20000
SEED = 20261103
ERAS = {"2011_2019": (2011, 2019), "2020_2025": (2020, 2025)}
ARMS = {"A": "mirrored_alpha_matched", "L": "mirrored_alpha_loso"}


def inner_mse(frame, alpha, hold_seasons):
    tot, n = 0.0, 0
    for t in hold_seasons:
        train = frame.loc[frame["season"].isin(hold_seasons) & frame["season"].ne(t)]
        test = frame.loc[frame["season"].eq(t)]
        est = MirrorPipe(
            steps=list(
                ORIG("ridge", 42, ridge_alpha=alpha, suppressed_indicator_columns=SUPP).steps
            )
        )
        est.fit(train.loc[:, list(FULL)], pd.to_numeric(train["ats_margin"]))
        pred = np.asarray(est.predict(test.loc[:, list(FULL)]), dtype=float)
        err = pd.to_numeric(test["ats_margin"]).to_numpy(dtype=float) - pred
        tot += float(np.sum(err**2))
        n += len(err)
    return tot / n


def choose_alphas(features):
    reg = regular_season_rows(features)
    reg = reg.loc[pd.to_numeric(reg["ats_margin"], errors="coerce").notna()].copy()
    chosen, tables = {}, {}
    STATE["arm"] = "M"
    for s in range(FIRST_SEASON, LAST_SEASON + 1):
        seasons = sorted(int(x) for x in reg["season"].unique() if x < s)
        if len(seasons) < MIN_INNER_SEASONS:
            chosen[s] = MATCHED
            tables[s] = {}
            continue
        frame = reg.loc[reg["season"].isin(seasons)]
        sc = {a: inner_mse(frame, a, seasons) for a in GRID}
        chosen[s] = min(GRID, key=lambda a: (sc[a], abs(a - MATCHED)))
        tables[s] = {str(a): v for a, v in sc.items()}
        print("alpha", s, chosen[s], flush=True)
    return chosen, tables


def boot_stats(d):
    g = d.groupby("season").agg(sum_=("d_acc", "sum"), n=("d_acc", "count"))
    s, n = g["sum_"].to_numpy(), g["n"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(s), (BOOT, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return {
        "effect": float(s.sum() / n.sum()),
        "low": float(np.percentile(b, 2.5)),
        "high": float(np.percentile(b, 97.5)),
        "pp": float((b > 0).mean()),
        "blocks": len(s),
    }


def homogeneity(d):
    seasons = d["season"].to_numpy()
    y = d["d_acc"].to_numpy(dtype=float)
    codes = pd.factorize(seasons)[0]
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k).astype(float)

    def q(lab):
        m = np.bincount(lab, weights=y, minlength=k) / cnt
        return float(np.sum(cnt * (m - y.mean()) ** 2))

    obs = q(codes)
    rng = np.random.default_rng(SEED)
    ge = 0
    for _ in range(BOOT):
        if q(rng.permutation(codes)) >= obs:
            ge += 1
    return {"Q": obs, "p_perm": (ge + 1) / (BOOT + 1), "seasons": int(k)}


def slope_stats(per):
    x = np.array(sorted(per), dtype=float)
    v = np.array([per[int(s)] for s in x])

    def sl(xx, vv):
        return float(np.polyfit(xx, vv, 1)[0])

    obs = sl(x, v)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1:
            bs.append(sl(x[i], v[i]))
    perm = [abs(sl(x, rng.permutation(v))) >= abs(obs) for _ in range(BOOT)]
    bs = np.array(bs)
    return {
        "slope": obs,
        "low": float(np.percentile(bs, 2.5)),
        "high": float(np.percentile(bs, 97.5)),
        "pp": float((bs > 0).mean()),
        "p_perm": (sum(perm) + 1) / (BOOT + 1),
        "seasons": len(x),
    }


def split_half(d):
    out = []
    for _s, t in d.groupby("season"):
        t = t.sort_values(["week"])
        a, b = t.iloc[::2]["d_acc"].mean(), t.iloc[1::2]["d_acc"].mean()
        out.append((a, b))
    a, b = np.array(out).T
    return float(np.corrcoef(a, b)[0, 1])


def era_diff(d, early, late):
    g1 = d.loc[d["season"].between(*early)].groupby("season")["d_acc"].agg(["sum", "count"])
    g2 = d.loc[d["season"].between(*late)].groupby("season")["d_acc"].agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    i1 = rng.integers(0, len(g1), (BOOT, len(g1)))
    i2 = rng.integers(0, len(g2), (BOOT, len(g2)))
    s1, n1 = g1["sum"].to_numpy(), g1["count"].to_numpy()
    s2, n2 = g2["sum"].to_numpy(), g2["count"].to_numpy()
    e1 = s1[i1].sum(1) / n1[i1].sum(1)
    e2 = s2[i2].sum(1) / n2[i2].sum(1)
    diff = e2 - e1
    obs = s2.sum() / n2.sum() - s1.sum() / n1.sum()
    return {
        "late_minus_early": float(obs),
        "low": float(np.percentile(diff, 2.5)),
        "high": float(np.percentile(diff, 97.5)),
        "pp": float((diff > 0).mean()),
        "early_ge_late_effect": float((e1 >= s2.sum() / n2.sum()).mean()),
    }


def joined(run, base):
    j = run.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    j["flip"] = j["pick_home"] != j["pick_home_b"]
    return j


def season_profile(j):
    rows = {}
    for s, t in j.groupby("season"):
        fw = int((t["flip"] & t["correct"].eq(1.0)).sum())
        fl = int((t["flip"] & t["correct"].eq(0.0)).sum())
        rows[int(s)] = {
            "n": len(t),
            "d_acc": float(t["d_acc"].mean()),
            "flips_W_L": f"{fw}-{fl}",
            "record": f"{int(t['correct'].sum())}-{len(t) - int(t['correct'].sum())}",
            "base": f"{int(t['correct_b'].sum())}-{len(t) - int(t['correct_b'].sum())}",
        }
    return rows


def run_loso(features, pop, chosen):
    install("M")
    LOG.clear()
    cache = {}
    try:
        early = HARNESS["proxy_eval"](
            features, pop.loc[pop["season"].le(2019)], lambda s: (FULL, chosen[int(s)])
        )
        parts = []
        for s in range(2020, 2026):
            a = chosen[s]
            if a not in cache:
                cache[a] = HARNESS["true_eval"](features, tuple(FULL), a)
            parts.append(cache[a].loc[cache[a]["season"].eq(s)])
    finally:
        margin_mod.make_margin_estimator = ORIG
    lg = pd.DataFrame(LOG)
    return pd.concat([early, *parts]), float(lg["train_acc"].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    chosen, tables = choose_alphas(features)
    pop = HARNESS["proxy_population"](features)
    runs = {"A": pd.read_parquet(U2OUT / "A_run.parquet")}
    runs["L"], train_acc_l = run_loso(features, pop, chosen)
    runs["L"].to_parquet(OUT / "L_run.parquet")
    report = {
        "grid": list(GRID),
        "chosen": chosen,
        "inner_tables": tables,
        "train_acc_L": train_acc_l,
        "arms": {},
    }
    for tag, run in runs.items():
        run = run.loc[run.index.isin(base.index)]
        cmp_ = compare(run, base)
        j = joined(run, base)
        arm = {"compare": cmp_, "eras": {}}
        for era, (lo, hi) in ERAS.items():
            d = j.loc[j["season"].between(lo, hi)]
            prof = season_profile(d)
            arm["eras"][era] = {
                "profile": prof,
                "boot": boot_stats(d),
                "homogeneity": homogeneity(d),
                "slope": slope_stats({s: v["d_acc"] for s, v in prof.items()}),
                "split_half_r": split_half(d),
                "n_positive_seasons": sum(v["d_acc"] > 0 for v in prof.values()),
                "n_negative_seasons": sum(v["d_acc"] < 0 for v in prof.values()),
            }
        pooled = season_profile(j)
        arm["pooled"] = {
            "boot": boot_stats(j),
            "homogeneity": homogeneity(j),
            "slope": slope_stats({s: v["d_acc"] for s, v in pooled.items()}),
        }
        arm["era_diff"] = era_diff(j, ERAS["2011_2019"], ERAS["2020_2025"])
        report["arms"][tag] = arm
        for era in ERAS:
            e = arm["eras"][era]
            c = cmp_["sbr_2011_2019" if era == "2011_2019" else "tuesday_2020_2025"]
            print(
                tag,
                era,
                c["record"],
                c["record_base"],
                c["flips_W_L"],
                round(c["d_acc"], 2),
                round(e["boot"]["low"], 2),
                round(e["boot"]["high"], 2),
                round(e["boot"]["pp"], 2),
                "ll",
                round(c["ll_arm"], 4),
                round(c["ll_base"], 4),
                round(c["ll_market"], 4),
                "br",
                round(c["brier_arm"], 4),
                round(c["brier_base"], 4),
                round(c["brier_market"], 4),
                flush=True,
            )
            print(
                tag,
                era,
                "hom",
                e["homogeneity"],
                "slope",
                e["slope"],
                "sh",
                e["split_half_r"],
                flush=True,
            )
            print(
                tag,
                era,
                "per_season",
                {s: (round(v["d_acc"], 2), v["flips_W_L"]) for s, v in e["profile"].items()},
                flush=True,
            )
        print(tag, "pooled", arm["pooled"], flush=True)
        print(tag, "era_diff", arm["era_diff"], flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    cells = []
    for tag in ("L",):
        for era, (lo, hi) in ERAS.items():
            e = report["arms"][tag]["eras"][era]
            b = e["boot"]
            c = report["arms"][tag]["compare"][
                "sbr_2011_2019" if era == "2011_2019" else "tuesday_2020_2025"
            ]
            cells.append(
                {
                    "name": f"mod54_unit3_{ARMS[tag]}_{era}_accuracy_diff",
                    "description": f"Mirrored rows with ridge alpha chosen leave-one-season-out inside the training seasons, paired vs base ridge alpha 10, {lo}-{hi}; {c['record']} vs {c['record_base']}, decisive flips {c['flips_W_L']}.",
                    "effect": b["effect"],
                    "interval_low": b["low"],
                    "interval_high": b["high"],
                    "probability_positive": b["pp"],
                    "sample_games": c["n"],
                    "sample_blocks": b["blocks"],
                    "season_start": lo,
                    "season_end": hi,
                    "plain_summary": "Letting the training seasons pick the penalty for mirrored home and away rows moved picks against the opener by an unresolved amount.",
                }
            )
    for tag in ARMS:
        for era, (lo, hi) in ERAS.items():
            e = report["arms"][tag]["eras"][era]
            s = e["slope"]
            cells.append(
                {
                    "name": f"mod54_unit3_{ARMS[tag]}_{era}_season_slope",
                    "description": f"Slope over seasons of the per-season accuracy difference of {ARMS[tag]} versus base, {lo}-{hi}, accuracy points per season.",
                    "effect": s["slope"],
                    "interval_low": s["low"],
                    "interval_high": s["high"],
                    "probability_positive": s["pp"],
                    "sample_games": report["arms"][tag]["compare"][
                        "sbr_2011_2019" if era == "2011_2019" else "tuesday_2020_2025"
                    ]["n"],
                    "sample_blocks": s["seasons"],
                    "season_start": lo,
                    "season_end": hi,
                    "plain_summary": "How the mirroring gain moves from season to season is unresolved at this number of seasons.",
                }
            )
    shared = {
        "source": "artifacts/mod54_unit3/report.json",
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "family": "mod54_mirrored_rows",
        "category": "modeling",
        "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
        "notes": "Base model, same inputs, past-only weekly refits, opener input; mirrored rows with a home-view column; alpha grid 5 to 80 in factors of two around the row-doubling match; ten looks in unit 3 (two LOSO-alpha accuracy, four slopes, four homogeneity tests reported in the lane).",
    }
    for c in cells:
        for k, v in shared.items():
            c.setdefault(k, v)
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
