import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod23_unit3 as u3

from nfl_ats import clv

HARNESS_PATH = Path(__file__).resolve().parent / "mod24_u1.py"
HARNESS = {"__file__": str(HARNESS_PATH), "__name__": "mod24_u1_harness"}
exec(compile(HARNESS_PATH.read_text().rsplit("\nmain()", 1)[0], str(HARNESS_PATH), "exec"), HARNESS)

ROOT = u3.ROOT
OUT = ROOT / "artifacts" / "mod23_unit10b"
BASE5 = ROOT / "artifacts" / "mod24_u5" / "base_per_game.parquet"
FULL = HARNESS["FULL"]
BASE_ALPHA = HARNESS["BASE_ALPHA"]
BOOT = 20000
SEED = 20261019
STARTERS = 11
NEW = ("diff_dc_off_ovr", "diff_dc_def_ovr", "diff_dc_off_yrs", "diff_dc_def_yrs")
ARMS = {"A": NEW, "B": NEW[:2]}
COEF_LOG = []
SNAP_FILE = sorted((ROOT / "data" / "players" / "raw").glob("2*/snap_counts.parquet"))[-1]
COMBINE = sorted((ROOT / "data" / "raw" / "combine").glob("*/combine.parquet"))[-1]
PICKS_DIR = sorted((ROOT / "data" / "cfb" / "draft_picks" / "raw").glob("*"))[-1]
SUFFIX = re.compile(r"(jr|sr|ii|iii|iv|v)")


def name_key(s):
    return s.astype(str).str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False).map(
        lambda x: re.sub(r"[^a-z0-9]", "", SUFFIX.sub("", x))
    )


def load_picks():
    frames = [pd.read_parquet(f, columns=["year", "overall", "name"]) for f in sorted(PICKS_DIR.glob("season=*/draft_picks.parquet"))]
    picks = pd.concat(frames, ignore_index=True)
    picks["key"] = name_key(picks["name"])
    picks = picks.loc[~picks["key"].duplicated(keep=False)]
    return picks[["key", "year", "overall"]].rename(columns={"year": "pick_year", "overall": "pick_ovr"})


def build_inputs():
    snaps = pd.read_parquet(SNAP_FILE)
    comb = pd.read_parquet(COMBINE)
    comb = comb.loc[comb["pfr_id"].notna()].sort_values("draft_ovr").drop_duplicates("pfr_id")
    undrafted = float(comb["draft_ovr"].max()) + 1.0
    snaps = snaps.merge(
        comb[["pfr_id", "draft_ovr", "draft_year"]], left_on="pfr_player_id", right_on="pfr_id", how="left"
    )
    first_seen = snaps.groupby("pfr_player_id")["season"].transform("min")
    snaps["key"] = name_key(snaps["player"])
    snaps = snaps.merge(load_picks(), on="key", how="left")
    fill = snaps["draft_ovr"].isna() & snaps["pick_ovr"].notna() & (snaps["pick_year"] <= snaps["season"])
    print("combine match rate", float((snaps["draft_ovr"].notna()).mean()), "name-key fallback rows", float(fill.mean()), flush=True)
    snaps["draft_ovr"] = np.where(fill, snaps["pick_ovr"], snaps["draft_ovr"])
    snaps["draft_year"] = np.where(fill, snaps["pick_year"], snaps["draft_year"])
    undrafted = max(undrafted, float(snaps["draft_ovr"].max()) + 1.0)
    known = snaps["draft_year"].notna() & snaps["draft_ovr"].notna()
    snaps["ovr"] = np.where(known, snaps["draft_ovr"], undrafted)
    snaps["rookie_year"] = np.where(known, snaps["draft_year"], first_seen)
    if not (snaps["rookie_year"] <= snaps["season"]).all():
        raise SystemExit("draft year after season")
    snaps["yrs"] = snaps["season"] - snaps["rookie_year"]
    print("combine match rate", float(known.mean()), "undrafted value", undrafted, flush=True)
    rows = []
    for unit, col in (("off", "offense_snaps"), ("def", "defense_snaps")):
        d = snaps.loc[snaps[col] > 0].sort_values(["game_id", "team", col], ascending=[True, True, False])
        d = d.groupby(["game_id", "team"]).head(STARTERS)
        d = d.assign(w=d[col], wo=d[col] * d["ovr"], wy=d[col] * d["yrs"])
        g = d.groupby(["game_id", "team", "season", "week"]).agg(w=("w", "sum"), wo=("wo", "sum"), wy=("wy", "sum")).reset_index()
        g[f"{unit}_ovr"] = g["wo"] / g["w"]
        g[f"{unit}_yrs"] = g["wy"] / g["w"]
        rows.append(g[["game_id", "team", "season", "week", f"{unit}_ovr", f"{unit}_yrs"]])
    tg = rows[0].merge(rows[1], on=["game_id", "team", "season", "week"], how="outer")
    tg["prior_order"] = tg["season"].astype(int) * 100 + tg["week"].astype(int)
    return tg.sort_values("prior_order").reset_index(drop=True)


def add_features(features):
    tg = build_inputs()
    f = features.copy()
    f["order"] = f["season"].astype(int) * 100 + f["week"].astype(int)
    stat_cols = ["off_ovr", "def_ovr", "off_yrs", "def_yrs"]
    out = {}
    for side in ("home", "away"):
        left = f[["game_id", "order", f"{side}_team"]].rename(columns={f"{side}_team": "team"}).astype({"team": object}).sort_values("order")
        right = tg[["team", "prior_order", *stat_cols]].astype({"team": object})
        m = pd.merge_asof(left, right, left_on="order", right_on="prior_order", by="team", allow_exact_matches=False)
        used = m.loc[m["prior_order"].notna()]
        if not (used["prior_order"] < used["order"]).all():
            raise SystemExit("availability assertion failed")
        out[side] = m.set_index("game_id")[stat_cols].reindex(f["game_id"]).to_numpy()
    for i, c in enumerate(stat_cols):
        f[f"diff_dc_{c}"] = out["home"][:, i] - out["away"][:, i]
    cov = f.groupby("season")[list(NEW)].apply(lambda d: float(d.notna().all(axis=1).mean()))
    print("coverage by season", cov.round(3).to_dict(), flush=True)
    return f.drop(columns=["order"])


def install_capture():
    orig = clv.fit_margin_model

    def wrapped(frame, **kw):
        model = orig(frame, **kw)
        cols = list(model.feature_columns)
        stats = np.asarray(model.estimator.named_steps["imputer"].statistics_, dtype=float)
        kept = [c for c, v in zip(cols, stats) if np.isfinite(v)]
        coef = np.asarray(model.estimator.steps[-1][1].coef_, dtype=float)
        if len(coef) < len(kept):
            raise SystemExit("coefficient mapping failed")
        got = {c: float(coef[kept.index(c)]) for c in NEW if c in kept}
        COEF_LOG.append({"season": int(pd.Timestamp(model.training_max_gameday).year), "coef": got})
        return model

    clv.fit_margin_model = wrapped
    HARNESS["fit_margin_model"] = wrapped


def coef_summary(arm):
    df = pd.DataFrame([{"season": r["season"], **r["coef"]} for r in COEF_LOG])
    out = {}
    for c in ARMS[arm]:
        per = df.groupby("season")[c].mean()
        out[c] = {
            "mean": float(per.mean()),
            "sd_across_seasons": float(per.std()),
            "share_positive_seasons": float((per > 0).mean()),
            "per_season": {int(k): float(v) for k, v in per.items()},
        }
    return out


def boot_season(df, col):
    g = df.groupby("season")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(s), (BOOT, len(s)))
    d = s[idx].sum(1) / n[idx].sum(1)
    return {"low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)), "pp": float((d > 0).mean()), "blocks": len(s)}


def market_only(j):
    y = np.where(j["pick_home_b"], j["correct_b"], 1.0 - j["correct_b"])
    j = j.assign(y=y)
    ll, br = [], []
    for s, t in j.groupby("season"):
        prior = j.loc[j["season"].lt(s) & (j["season"].le(2019) == (s <= 2019)), "y"]
        p = float(prior.mean()) if len(prior) else 0.5
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll.append(-(t["y"] * np.log(p) + (1 - t["y"]) * np.log(1 - p)))
        br.append((p - t["y"]) ** 2)
    return pd.concat(ll), pd.concat(br)


def compare(arm, base):
    j = arm.join(base, rsuffix="_b", how="inner")
    j["d_acc"] = 100.0 * (j["correct"] - j["correct_b"])
    mll, mbr = market_only(j)
    j["ll_m"] = mll
    j["brier_m"] = mbr
    out = {}
    for name, mask in (("sbr_2013_2019", j["season"].between(2013, 2019)), ("tuesday_2020_2025", j["season"].ge(2020))):
        s = j.loc[mask]
        w = int(s["correct"].sum())
        flips = s["pick_home"] != s["pick_home_b"]
        per = {}
        for season, t in s.groupby("season"):
            per[int(season)] = {
                "record": f"{int(t['correct'].sum())}-{len(t) - int(t['correct'].sum())}",
                "base": f"{int(t['correct_b'].sum())}-{len(t) - int(t['correct_b'].sum())}",
                "d_acc": float(t["d_acc"].mean()),
            }
        out[name] = {
            "n": len(s),
            "record": f"{w}-{len(s) - w}",
            "record_base": f"{int(s['correct_b'].sum())}-{len(s) - int(s['correct_b'].sum())}",
            "flips": int(flips.sum()),
            "flips_W_L": f"{int((flips & s['correct'].eq(1.0)).sum())}-{int((flips & s['correct'].eq(0.0)).sum())}",
            "d_acc": float(s["d_acc"].mean()),
            "d_acc_season_boot": boot_season(s, "d_acc"),
            "ll_arm": float(s["ll"].mean()),
            "ll_base": float(s["ll_b"].mean()),
            "ll_market": float(s["ll_m"].mean()),
            "brier_arm": float(s["brier"].mean()),
            "brier_base": float(s["brier_b"].mean()),
            "brier_market": float(s["brier_m"].mean()),
            "per_season": per,
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(u3.FEATURES)
    pop = HARNESS["proxy_population"](features)
    base = pd.read_parquet(BASE5)
    tr = base.loc[base["season"].ge(2020)]
    print("base true", int(tr["correct"].sum()), len(tr), flush=True)
    if (int(tr["correct"].sum()), len(tr)) != (802, 1503):
        raise SystemExit("base reproduction failed")
    sb = base.loc[base["season"].le(2019)]
    if (int(sb["correct"].sum()), len(sb)) != (1142, 2231):
        raise SystemExit("proxy base reproduction failed")
    sb13 = base.loc[base["season"].between(2013, 2019)]
    print("base 2013-19", int(sb13["correct"].sum()), len(sb13), flush=True)
    feats = add_features(features)
    install_capture()
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    results = {}
    for name, extra in ARMS.items():
        COEF_LOG.clear()
        cols = tuple(FULL) + tuple(extra)
        run = HARNESS["run_arm"](feats, pop, lambda s: (cols, BASE_ALPHA))
        run.to_parquet(OUT / f"{name}_run.parquet")
        run = run.loc[run.index.isin(base.index)]
        results[name] = compare(run, base)
        results[name]["coefficients"] = coef_summary(name)
        for era, r in results[name].items():
            if era == "coefficients":
                continue
            print(name, era, r["record"], "base", r["record_base"], "d_acc", round(r["d_acc"], 3), r["d_acc_season_boot"], "flips", r["flips"], r["flips_W_L"], "ll", round(r["ll_arm"], 4), round(r["ll_base"], 4), round(r["ll_market"], 4), "brier", round(r["brier_arm"], 4), round(r["brier_base"], 4), round(r["brier_market"], 4), flush=True)
        print(name, "coef", {c: (round(v["mean"], 4), round(v["sd_across_seasons"], 4), v["share_positive_seasons"]) for c, v in results[name]["coefficients"].items()}, flush=True)
    (OUT / "report_all.json").write_text(json.dumps(results, indent=1, default=str))
    cells = []
    for name in ARMS:
        for era, r in results[name].items():
            if era == "coefficients":
                continue
            lo, hi = (2013, 2019) if era.startswith("sbr") else (2020, 2025)
            what = "offence/defence draft pick and years in league" if name == "A" else "offence/defence draft pick only"
            cells.append(
                {
                    "source": "artifacts/mod23_unit10b/report_all.json",
                    "effect_units": "accuracy_points",
                    "classification": "unresolved_below_power",
                    "league": "nfl",
                    "season_start": lo,
                    "season_end": hi,
                    "family": "mod23_unit10_draft_capital",
                    "category": "modeling",
                    "classification_evidence": "No positive control and no admissible closing ground; season-blocked bootstrap interval reported.",
                    "notes": f"Base fit plus starter draft-capital inputs, arm {name} ({what}); four looks (two arms, two eras).",
                    "name": f"mod23_unit10_draft_capital_{name}_{era}_accuracy_diff",
                    "description": f"Starter draft capital arm {name} paired vs base, {lo}-{hi}; {r['record']} vs {r['record_base']}.",
                    "effect": r["d_acc"],
                    "interval_low": r["d_acc_season_boot"]["low"],
                    "interval_high": r["d_acc_season_boot"]["high"],
                    "probability_positive": r["d_acc_season_boot"]["pp"],
                    "sample_games": r["n"],
                    "sample_blocks": r["d_acc_season_boot"]["blocks"],
                    "plain_summary": "Adding the draft pedigree and experience of last week's starters to the base inputs moved picks against the opener by an unresolved amount.",
                }
            )
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": cells}, indent=1))


main()
