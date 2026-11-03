import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import live_kickoff
import mod25_forward_q as mfq
import mod25_wf_forward as mwf

from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

REPO = mwf.REPO
PIPE = REPO / "artifacts" / "sim10_wf" / "pipeline_joint"
OUT_DIR = REPO / "artifacts" / "sim10_wf" / "forward_2026_joint"
RUNS_DIR = REPO / "artifacts" / "mod25_wfj_forward" / "runs"
FORECASTS = REPO / "artifacts" / "margin_predictions"
ARM = "hgb_nomkt"
STUDENT_T = 2025
WEEKS = tuple(range(1, 9))
SEASON = 2026
STUDENT_COLS = list(mfq.NOMKT_COLS)


def stamp_of(folder):
    return pd.Timestamp(
        datetime.strptime(folder.name.split("-")[-1], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    )


def distilled_frame():
    ts = list(range(2020, STUDENT_T + 1))
    pick = mwf.owner_map(ts)
    frames = {t: pd.read_parquet(PIPE / f"T{t}" / "real_distilled.parquet") for t in ts}
    parts = []
    for s in sorted(frames[min(ts)]["season"].unique()):
        f = frames[pick(int(s))]
        parts.append(f.loc[f["season"].eq(s)])
    return pd.concat(parts, ignore_index=True)[["game_id", "season", *mwf.ARMS]]


def fit_student():
    import mod25_pipeline as mp
    from sklearn.ensemble import HistGradientBoostingRegressor

    syn_dir = REPO / "data" / "processed" / "synthetic" / "wf_joint" / f"T{STUDENT_T}"
    f = pd.read_parquet(syn_dir / "features.parquet")
    t = pd.read_parquet(syn_dir / "truth.parquet")
    syn = f.merge(t[["game_id", "true_em"]], on="game_id", how="inner")
    syn = syn.loc[syn["sidx"].ge(1)].reset_index(drop=True)
    mu, sd = mp.z_stats(syn, STUDENT_COLS)
    Z = mp.z_apply(syn, STUDENT_COLS, mu, sd)
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=0,
    ).fit(Z, syn["true_em"].to_numpy())
    return mp, model


def locked_student(features, locked, saved):
    mp, model = fit_student()
    real = mp.real_frame().sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    prior = real.loc[real["season"].lt(SEASON)]
    rmu, rsd = mp.z_stats(prior, STUDENT_COLS)
    repro = real.loc[real["season"].eq(SEASON)]
    check = model.predict(mp.z_apply(repro, STUDENT_COLS, rmu, rsd))
    ref = repro["game_id"].map(saved.set_index("game_id")["distilled_margin"]).to_numpy(float)
    reproduction = float(np.nanmax(np.abs(check - ref)))
    rows = real.loc[real["game_id"].isin(locked.keys())].copy()
    for gid, r in locked.items():
        for c, v in r["locked_features"].items():
            rows.loc[rows["game_id"].eq(gid), c] = pd.to_numeric(
                pd.Series([v]), errors="coerce"
            ).iloc[0]
    pred = model.predict(mp.z_apply(rows, STUDENT_COLS, rmu, rsd))
    return dict(zip(rows["game_id"], pred, strict=True)), reproduction


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locked_served(game_ids, deadlines):
    rows = {}
    folders = sorted(FORECASTS.glob(f"{SEASON}-week-*"))
    by_week = {}
    for f in folders:
        by_week.setdefault(int(f.name.split("-")[2]), []).append(f)
    for gid in game_ids:
        week = int(gid.split("_")[1])
        dl = deadlines[gid]
        ok = [
            f for f in by_week.get(week, []) if stamp_of(f) < dl and (f / "pool_card.csv").exists()
        ]
        for f in reversed(ok):
            pool = pd.read_csv(f / "pool_card.csv")
            rec = pd.read_csv(f / "recommendations.csv")
            if gid in set(pool["game_id"]) and gid in set(rec["game_id"]):
                p = pool.set_index("game_id").loc[gid]
                r = rec.set_index("game_id").loc[gid]
                rows[gid] = {
                    "folder": f.name,
                    "folder_stamp": stamp_of(f).isoformat(),
                    "deadline": dl.isoformat(),
                    "served_side": p["pool_side"],
                    "served_home_p": float(r["home_cover_probability"]),
                    "locked_line": float(p["pick_line"]),
                    "locked_features": {c: r[c] for c in STUDENT_COLS if c in r.index},
                }
                break
    return rows


def deadlines_for(features):
    out = {}
    for week, g in features.groupby("week"):  # noqa: B007
        k = live_kickoff.earliest_by_game_id(g)
        lock = sunday_pick_lock(k)
        for gid, kk in zip(g["game_id"], k, strict=True):
            out[gid] = pick_deadline(kk, lock)
    return out


def feature_check(features, locked):
    cur = features.set_index("game_id")
    worst = 0.0
    n_cmp = 0
    bad = []
    for gid, row in locked.items():
        for c, v in row["locked_features"].items():
            a = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            b = cur.loc[gid, c]
            if pd.isna(a) and pd.isna(b):
                continue
            d = abs(float(a) - float(b)) if not (pd.isna(a) or pd.isna(b)) else np.inf
            n_cmp += 1
            if d > 1e-9:
                bad.append((gid, c, float(d) if np.isfinite(d) else None))
            worst = max(worst, d)
    return {
        "values_compared": n_cmp,
        "values_differing": len(bad),
        "max_abs_diff": None if not np.isfinite(worst) else worst,
        "examples": bad[:8],
    }


def boot(delta):
    return mwf.boot_pos(delta)


def vs_served(tab, locked, graded_line_col):
    t = tab.loc[tab["game_id"].isin(locked.keys())].copy()
    t["served_side"] = t["game_id"].map(lambda g: locked[g]["served_side"])
    t["served_home_p"] = t["game_id"].map(lambda g: locked[g]["served_home_p"])
    t = t.loc[t["served_side"].isin(["HOME", "AWAY"])].copy()
    y = t["y_home"].to_numpy(float)
    arm_home = t["p_arm"].to_numpy(float) > 0.5
    base_home = t["p_base"].to_numpy(float) > 0.5
    srv_home = (t["served_side"] == "HOME").to_numpy()
    ca = (arm_home == (y == 1)).astype(float)
    cb = (base_home == (y == 1)).astype(float)
    cs = (srv_home == (y == 1)).astype(float)
    dec = arm_home != srv_home
    out = {
        "n": len(t),
        "arm_record": f"{int(ca.sum())}-{int(len(t) - ca.sum())}",
        "rebuilt_base_record": f"{int(cb.sum())}-{int(len(t) - cb.sum())}",
        "served_record": f"{int(cs.sum())}-{int(len(t) - cs.sum())}",
        "decisive_games_arm_v_served": int(dec.sum()),
        "decisive_arm_W_L": f"{int(ca[dec].sum())}-{int(dec.sum() - ca[dec].sum())}",
        "decisive_served_W_L": f"{int(cs[dec].sum())}-{int(dec.sum() - cs[dec].sum())}",
        "acc_diff_points_arm_minus_served": boot(100 * (ca - cs)),
    }
    pa = np.clip(t["p_arm"].to_numpy(float), 1e-6, 1 - 1e-6)
    pb = np.clip(t["p_base"].to_numpy(float), 1e-6, 1 - 1e-6)
    ps = np.clip(t["served_home_p"].to_numpy(float), 1e-6, 1 - 1e-6)

    def ll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def br(p):
        return (p - y) ** 2

    half = np.full(len(y), 0.5)
    out["logloss"] = {
        "market_half": float(ll(half).mean()),
        "served": float(ll(ps).mean()),
        "rebuilt_base": float(ll(pb).mean()),
        "arm": float(ll(pa).mean()),
    }
    out["brier"] = {
        "market_half": float(br(half).mean()),
        "served": float(br(ps).mean()),
        "rebuilt_base": float(br(pb).mean()),
        "arm": float(br(pa).mean()),
    }
    out["logloss_gain_arm_v_served"] = boot(ll(ps) - ll(pa))
    out["logloss_gain_arm_v_rebuilt_base"] = boot(ll(pb) - ll(pa))
    out["logloss_gain_arm_v_market_half"] = boot(ll(half) - ll(pa))
    out["brier_gain_arm_v_served"] = boot(br(ps) - br(pa))
    out["brier_gain_arm_v_market_half"] = boot(br(half) - br(pa))
    return out


def run_read():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mwf.WEEKS = WEEKS
    mwf.SEASON = SEASON
    features = pd.read_parquet(mfq.FEATURES)
    features["gameday"] = pd.to_datetime(features["gameday"])
    dist = distilled_frame()
    now26 = features.loc[features["season"].eq(SEASON) & features["week"].isin(WEEKS)]
    dl = deadlines_for(features.loc[features["season"].eq(SEASON)])
    splash = mwf.splash_lines()
    openers = mwf.tuesday_open(now26)
    variants = {
        "pool": {
            g: splash.get(g, line)
            for g, line in zip(now26["game_id"], now26["spread_line"], strict=False)
        },
        "open": {g: v[0] for g, v in openers.items()},
    }
    report = {
        "student": {
            "arm": ARM,
            "engine": "joint",
            "T": STUDENT_T,
            "file": str(PIPE / f"T{STUDENT_T}" / "real_distilled.parquet"),
            "sha256": sha(PIPE / f"T{STUDENT_T}" / "real_distilled.parquet"),
            "file_mtime_utc": datetime.fromtimestamp(
                (PIPE / f"T{STUDENT_T}" / "real_distilled.parquet").stat().st_mtime, UTC
            ).isoformat(),
        },
        "prospective": False,
    }
    d = dist[["game_id", ARM]].rename(columns={ARM: "distilled_margin"})
    all_locked = locked_served(list(now26["game_id"]), dl)
    lock_pred, reproduction = locked_student(features, all_locked, d)
    report["refit_student_reproduces_saved_2026_max_abs_diff"] = reproduction
    report["locked_games_with_student_from_locked_features"] = len(lock_pred)
    d = d.copy()
    d["distilled_margin"] = [
        lock_pred.get(g, v) for g, v in zip(d["game_id"], d["distilled_margin"], strict=True)
    ]
    for name in ("pool", "open"):
        feats = mwf.set_lines(features, variants[name])
        merged = feats.merge(d, on="game_id", how="left")
        reg = merged.loc[merged["game_type"].eq("REG")]
        assert reg["distilled_margin"].notna().all()
        pred, frozen = mwf.score_all(merged, ARM)
        tab = pred.merge(feats[["game_id", "spread_line", "result"]], on="game_id")
        tab = tab.loc[tab["game_id"].isin(variants[name].keys())]
        tab = tab.loc[tab["result"].notna() & (tab["result"] != tab["spread_line"])].copy()
        tab["y_home"] = (tab["result"] - tab["spread_line"] > 0).astype(float)
        locked = locked_served(list(tab["game_id"]), dl)
        tab.to_parquet(OUT_DIR / f"scored_{ARM}_{name}.parquet")
        entry = {
            "frozen": frozen,
            "arm_v_rebuilt_base": mwf.metrics(tab),
            "locked_served_found": len(locked),
            "games": len(tab),
        }
        entry["vs_locked_served"] = vs_served(tab, locked, name)
        entry["locked_folder_after_deadline"] = int(
            sum(
                1
                for g, r in locked.items()
                if pd.Timestamp(r["folder_stamp"]) >= pd.Timestamp(r["deadline"])
            )
        )
        entry["student_feature_check_vs_locked_folder"] = feature_check(features, locked)
        entry["student_inputs_lock_gap_hours_min_max"] = [
            float(
                min(
                    (pd.Timestamp(r["deadline"]) - pd.Timestamp(r["folder_stamp"])).total_seconds()
                    / 3600
                    for r in locked.values()
                )
            ),
            float(
                max(
                    (pd.Timestamp(r["deadline"]) - pd.Timestamp(r["folder_stamp"])).total_seconds()
                    / 3600
                    for r in locked.values()
                )
            ),
        ]
        report[name] = entry
        print(name, json.dumps(entry, default=float), flush=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))


def record_week(week):
    mwf.SEASON = SEASON
    now = pd.Timestamp(datetime.now(UTC))
    features = pd.read_parquet(mfq.FEATURES)
    features["gameday"] = pd.to_datetime(features["gameday"])
    dist = distilled_frame()
    d = dist[["game_id", ARM]].rename(columns={ARM: "distilled_margin"})
    merged = features.merge(d, on="game_id", how="left")
    target = merged.loc[merged["season"].eq(SEASON) & merged["week"].eq(week)].copy()
    dl = deadlines_for(features.loc[features["season"].eq(SEASON)])
    eligible = [
        g
        for g in target["game_id"]
        if pd.isna(features.set_index("game_id").loc[g, "result"]) and now < dl[g]
    ]
    assert target["distilled_margin"].notna().all()
    mwf.WEEKS = (1,)
    comb_cols = [*mfq.FULL_COLS, "distilled_margin", "season"]
    mfq.CapturingTwoStage.captured.clear()
    mfq.score_week(merged, 1, comb_cols, mfq.CapturingTwoStage)
    first = mfq.CapturingTwoStage.captured[0]
    assert max(c["max_train_season"] for c in mfq.CapturingTwoStage.captured) == SEASON - 1
    frozen = {k: first[k] for k in ("w", "qbar", "bbar")}

    def factory(base_cols, orig, kw):
        return mfq.FrozenTwoStage(base_cols, orig, kw, frozen["w"], frozen["qbar"], frozen["bbar"])

    base, _ = mfq.score_week(merged, week, list(mfq.BASE_COLS), None)
    comb, _ = mfq.score_week(merged, week, comb_cols, factory)
    idx = merged.set_index("game_id")
    rows = []
    recorded = pd.Timestamp(datetime.now(UTC))
    for g in eligible:
        if recorded >= dl[g]:
            continue
        bp = float(base.loc[g, "home_cover_probability"])
        cp = float(comb.loc[g, "home_cover_probability"])
        rows.append(
            {
                "game_id": g,
                "season": SEASON,
                "week": week,
                "deadline": dl[g].isoformat(),
                "recorded_at_utc": recorded.isoformat(),
                "pool_line": float(idx.loc[g, "spread_line"]),
                "distilled_margin": float(idx.loc[g, "distilled_margin"]),
                "base_home_cover_probability": bp,
                "base_side": "HOME" if bp > 0.5 else "AWAY",
                "student_home_cover_probability": cp,
                "student_side": "HOME" if cp > 0.5 else "AWAY",
            }
        )
    payload = {
        "schema": "mod25_wfj_forward/1",
        "arm": ARM,
        "season": SEASON,
        "week": week,
        "recorded_at_utc": recorded.isoformat(),
        "frozen": frozen,
        "student_file_sha256": sha(PIPE / f"T{STUDENT_T}" / "real_distilled.parquet"),
        "feature_table_sha256": sha(mfq.FEATURES),
        "script_sha256": sha(__file__),
        "rows": rows,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{SEASON}-week-{week:02d}-{recorded.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(path, "x") as handle:
        handle.write(json.dumps(payload, indent=1, default=float))
    print(json.dumps({"path": str(path), "games": len(rows), "frozen": frozen}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None)
    a = ap.parse_args()
    if a.week is None:
        run_read()
    else:
        record_week(a.week)


if __name__ == "__main__":
    main()
