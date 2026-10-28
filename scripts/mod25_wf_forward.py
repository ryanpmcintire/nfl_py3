import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod25_forward_q as mfq
import mod25_pipeline as mp

REPO = mp.REPO
WF = REPO / "artifacts" / "sim10_wf"
OUT_DIR = WF / "forward_2026"
SPLASH = REPO / "data" / "splash"
ARMS = ("ridge_nomkt", "hgb_nomkt")
WEEKS = tuple(range(1, 8))
SEASON = 2026
BOOT = 20000
SEED = 20261028


def owner_map(ts):
    base = min(ts)
    return lambda s: max([t for t in ts if t <= s] or [base])


def distilled_frame():
    ts = list(range(2020, SEASON + 1))
    pick = owner_map(ts)
    frames = {t: pd.read_parquet(WF / "pipeline" / f"T{t}" / "real_distilled.parquet") for t in ts}
    parts = []
    for s in sorted(frames[min(ts)]["season"].unique()):
        f = frames[pick(int(s))]
        parts.append(f.loc[f["season"].eq(s)])
    return pd.concat(parts, ignore_index=True)[["game_id", "season", *ARMS]]


def splash_lines():
    rows = {}
    for path in sorted(SPLASH.glob(f"{SEASON}_week*.json")):
        d = json.loads(path.read_text())
        for g in d["games"]:
            rows[g["game_id"]] = float(g["home_spread"])
    return rows


def tuesday_open(features):
    market = mfq.market_frame()
    k = pd.to_datetime(features["kickoff"], utc=True)
    out = {}
    for gid, kick in zip(features["game_id"], k):
        et = kick.tz_convert("America/New_York").normalize()
        tue = et - pd.Timedelta(days=(et.weekday() - 1) % 7)
        g = market.loc[market["game_id"].eq(gid) & market["observed_at_utc"].ge(tue.tz_convert("UTC"))]
        if not g.empty:
            first = g.sort_values("observed_at_utc").iloc[0]
            out[gid] = (float(first["market_line"]), first["observed_at_utc"].isoformat())
    return out


def set_lines(features, lines):
    f = features.copy()
    m = f["season"].eq(SEASON) & f["game_id"].isin(lines.keys())
    new = f.loc[m, "game_id"].map(lines).astype(float)
    f.loc[m, "spread_line"] = new
    f.loc[m, "ats_margin"] = f.loc[m, "result"] - new
    return f


def score_all(merged, arm):
    comb_cols = mfq.FULL_COLS + ["distilled_margin", "season"]
    mfq.CapturingTwoStage.captured.clear()
    mfq.score_week(merged, 1, comb_cols, mfq.CapturingTwoStage)
    first = mfq.CapturingTwoStage.captured[0]
    last = mfq.CapturingTwoStage.captured[-1]
    assert max(c["max_train_season"] for c in mfq.CapturingTwoStage.captured) == SEASON - 1
    frozen = {k: first[k] for k in ("w", "qbar", "bbar")}

    def factory(base_cols, orig, kw):
        return mfq.FrozenTwoStage(base_cols, orig, kw, frozen["w"], frozen["qbar"], frozen["bbar"])

    rows = []
    for week in WEEKS:
        base, _ = mfq.score_week(merged, week, list(mfq.BASE_COLS), None)
        comb, _ = mfq.score_week(merged, week, comb_cols, factory)
        idx = base.index.intersection(comb.index)
        rows.append(
            pd.DataFrame(
                {
                    "game_id": idx,
                    "week": week,
                    "p_base": base.loc[idx, "home_cover_probability"].to_numpy(float),
                    "p_arm": comb.loc[idx, "home_cover_probability"].to_numpy(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), {**frozen, "n_train": last["n_train"]}


def boot_pos(delta):
    rng = np.random.default_rng(SEED)
    d = np.asarray(delta, float)
    means = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_pos": float((means > 0).mean())}


def metrics(df):
    y = df["y_home"].to_numpy(float)
    out = {"n": int(len(df))}
    ph = {"base": df["p_base"].to_numpy(float), "arm": df["p_arm"].to_numpy(float)}
    pick = {k: v > 0.5 for k, v in ph.items()}
    corr = {k: (pick[k] == (y == 1)).astype(float) for k in pick}
    out["base_record"] = f"{int(corr['base'].sum())}-{int(len(y) - corr['base'].sum())}"
    out["arm_record"] = f"{int(corr['arm'].sum())}-{int(len(y) - corr['arm'].sum())}"
    diff = pick["base"] != pick["arm"]
    out["flips"] = int(diff.sum())
    out["flip_arm_W_L"] = f"{int((corr['arm'][diff] > corr['base'][diff]).sum())}-{int((corr['arm'][diff] < corr['base'][diff]).sum())}"
    ll = {k: -(y * np.log(np.clip(v, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - v, 1e-6, 1))) for k, v in ph.items()}
    br = {k: (v - y) ** 2 for k, v in ph.items()}
    out["logloss_base"] = float(ll["base"].mean())
    out["logloss_arm"] = float(ll["arm"].mean())
    out["brier_base"] = float(br["base"].mean())
    out["brier_arm"] = float(br["arm"].mean())
    out["acc_diff_points"] = boot_pos(100 * (corr["arm"] - corr["base"]))
    out["logloss_gain"] = boot_pos(ll["base"] - ll["arm"])
    out["brier_gain"] = boot_pos(br["base"] - br["arm"])
    out["by_week_acc_diff_points"] = {
        int(w): float(100 * (corr["arm"][df["week"].to_numpy() == w] - corr["base"][df["week"].to_numpy() == w]).mean())
        for w in sorted(df["week"].unique())
    }
    out["by_week_record_arm_base"] = {
        int(w): f"{int(corr['arm'][df['week'].to_numpy() == w].sum())}-{int((df['week'] == w).sum() - corr['arm'][df['week'].to_numpy() == w].sum())}"
        f" / {int(corr['base'][df['week'].to_numpy() == w].sum())}-{int((df['week'] == w).sum() - corr['base'][df['week'].to_numpy() == w].sum())}"
        for w in sorted(df["week"].unique())
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pool,open")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(mfq.FEATURES)
    features["gameday"] = pd.to_datetime(features["gameday"])
    dist = distilled_frame()
    now26 = features.loc[features["season"].eq(SEASON) & features["week"].isin(WEEKS)]
    splash = splash_lines()
    check = now26["game_id"].map(splash)
    diff = (check - now26["spread_line"]).abs()
    report = {
        "games": int(len(now26)),
        "splash_games": int(check.notna().sum()),
        "table_vs_splash_max_abs_diff": float(diff.max()),
        "table_vs_splash_n_diff": int((diff > 0).sum()),
    }
    openers = tuesday_open(now26)
    report["tuesday_open_available"] = len(openers)
    variants = {"pool": {g: splash.get(g, l) for g, l in zip(now26["game_id"], now26["spread_line"])}, "open": {g: v[0] for g, v in openers.items()}}
    for arm in ARMS:
        d = dist[["game_id", arm]].rename(columns={arm: "distilled_margin"})
        for name in a.variants.split(","):
            feats = set_lines(features, variants[name])
            merged = feats.merge(d, on="game_id", how="left")
            reg = merged.loc[merged["game_type"].eq("REG")]
            assert reg["distilled_margin"].notna().all()
            pred, frozen = score_all(merged, arm)
            tab = pred.merge(feats[["game_id", "spread_line", "result"]], on="game_id")
            tab = tab.loc[tab["game_id"].isin(variants[name].keys())]
            tab = tab.loc[tab["result"].notna() & (tab["result"] != tab["spread_line"])].copy()
            tab["y_home"] = (tab["result"] - tab["spread_line"] > 0).astype(float)
            tab.to_parquet(OUT_DIR / f"scored_{arm}_{name}.parquet")
            report[f"{arm}_{name}"] = {"frozen": frozen, **metrics(tab)}
            print(arm, name, json.dumps(report[f"{arm}_{name}"], default=float), flush=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
