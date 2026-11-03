import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod10_graph_blitz as bl
import mod10_graph_term as g
import mod10_graph_term_warm as w
import mod23_unit12_ftn_scheme as u
import mod24_ftn_turnover_luck as m
import mod25_wf_forward as mwf
import mod25_wfj_forward as wj
import numpy as np
import pandas as pd

OUT = m.ROOT / "artifacts" / "forward_2026_ftn_terms"
SEASON = 2026
WEEKS = tuple(range(1, 8))
SUFFIX = "_forward2026_w01_07"


def frozen_term(table, columns, diff_columns, tag):
    model, _ = g.fit_frame(table, columns, SEASON)
    names = list(model[:-1].get_feature_names_out())
    coef = np.ravel(model[-1].coef_)
    scaler = model.named_steps["scaler"]
    idx = {n: i for i, n in enumerate(names)}
    keep = table["game_type"].eq("REG") & table["ats_margin"].notna()
    train = table.loc[keep & table["season"].lt(SEASON)]
    payload = {
        "term": tag,
        "created_at": datetime.now(UTC).isoformat(),
        "train_seasons_lt": SEASON,
        "train_rows": len(train),
        "ridge_alpha": m.BASE_ALPHA,
        "target": "market_residual",
        "columns": {
            c: {
                "coef": float(coef[idx[c]]),
                "scaler_mean": float(scaler.mean_[idx[c]]),
                "scaler_scale": float(scaler.scale_[idx[c]]),
            }
            for c in diff_columns
        },
    }
    path = OUT / f"frozen_{tag}.json"
    path.write_text(json.dumps(payload, indent=1))
    return path


def apply_frozen(path, frame):
    spec = json.loads(Path(path).read_text())["columns"]
    delta = np.zeros(len(frame))
    for c, s in spec.items():
        delta += s["coef"] * (frame[c].to_numpy(float) - s["scaler_mean"]) / s["scaler_scale"]
    return delta


def locked_inputs(served26):
    dl = wj.deadlines_for(served26)
    locked = wj.locked_served(list(served26["game_id"]), dl)
    rows = {}
    for gid, info in locked.items():
        folder = wj.FORECASTS / info["folder"]
        rec = pd.read_csv(folder / "recommendations.csv").set_index("game_id")
        sweep = pd.read_parquet(folder / "line_sweep.parquet")
        sweep = sweep.loc[sweep["method"].eq("market_residual") & sweep["game_id"].eq(gid)]
        rows[gid] = {
            "info": info,
            "rec": rec.loc[gid],
            "sweep": sweep.sort_values("alternative_line"),
        }
    return rows


def score(tag, deltas, rows, served26, openers):
    out_rows = []
    for gid, (open_line, _) in openers.items():
        if gid not in rows or gid not in deltas:
            continue
        r = served26.loc[served26["game_id"].eq(gid)].iloc[0]
        if pd.isna(r["result"]) or r["result"] == open_line:
            continue
        sw = rows[gid]["sweep"]
        x = sw["alternative_line"].to_numpy(float)
        p = sw["home_cover_probability_excluding_push"].to_numpy(float)
        d = float(deltas[gid])
        if open_line - d < x.min() or open_line - d > x.max():
            continue
        out_rows.append(
            {
                "game_id": gid,
                "week": int(r["week"]),
                "open_line": open_line,
                "delta": d,
                "y": float(r["result"] - open_line > 0),
                "p_served": float(np.interp(open_line, x, p)),
                "p_arm": float(np.interp(open_line - d, x, p)),
                "locked_line": float(rows[gid]["rec"]["spread_line"]),
            }
        )
    t = pd.DataFrame(out_rows)
    y = t["y"].to_numpy()
    ps = np.clip(t["p_served"].to_numpy(), 1e-6, 1 - 1e-6)
    pa = np.clip(t["p_arm"].to_numpy(), 1e-6, 1 - 1e-6)
    cs = ((ps > 0.5) == (y == 1)).astype(float)
    ca = ((pa > 0.5) == (y == 1)).astype(float)
    dec = (ps > 0.5) != (pa > 0.5)

    def ll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def br(p):
        return (p - y) ** 2

    res = {
        "term": tag,
        "n": len(t),
        "weeks": sorted(set(t["week"])),
        "opener_equals_locked_line": int((t["open_line"] == t["locked_line"]).sum()),
        "served_record": f"{int(cs.sum())}-{int(len(t) - cs.sum())}",
        "arm_record": f"{int(ca.sum())}-{int(len(t) - ca.sum())}",
        "flips": int(dec.sum()),
        "decisive_arm": f"{int(ca[dec].sum())}-{int(dec.sum() - ca[dec].sum())}",
        "decisive_served": f"{int(cs[dec].sum())}-{int(dec.sum() - cs[dec].sum())}",
        "delta_mean_abs": float(np.abs(t["delta"]).mean()),
        "delta_sd": float(t["delta"].std()),
        "accuracy_points": mwf.boot_pos(100 * (ca - cs)),
        "log_loss_gain": mwf.boot_pos(ll(ps) - ll(pa)),
        "brier_gain": mwf.boot_pos(br(ps) - br(pa)),
        "mean_log_loss_served": float(ll(ps).mean()),
        "mean_log_loss_arm": float(ll(pa).mean()),
        "mean_brier_served": float(br(ps).mean()),
        "mean_brier_arm": float(br(pa).mean()),
    }
    t.to_csv(OUT / f"scored_{tag}.csv", index=False)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    served = pd.read_parquet(m.SERVED_FEATURES)
    served["gameday"] = pd.to_datetime(served["gameday"])
    served26 = served.loc[served["season"].eq(SEASON) & served["week"].isin(WEEKS)]
    openers = mwf.tuesday_open(served26)
    plays = u.load_plays()
    _, per_team, n0 = u.build_scheme(plays, served)
    cutoff = served26.loc[served26["week"].eq(7), "gameday"].max()
    u.CUTOFF = cutoff
    u.assert_pregame(plays, served, per_team)
    table_u = u.attach(served, per_team)
    cols_u = (*m.BASE_COLUMNS, *u.DIFF_COLUMNS)
    path_u = frozen_term(table_u, cols_u, u.DIFF_COLUMNS, "unit12_v2")

    t = bl.load()
    rel = bl.reliability(t)
    k_def, k_qb = bl.kfun(rel["def_blitz_rate"]), bl.kfun(rel["qb_catchable"])
    w.WARM = 2022
    tv_train = bl.team_values(t, served.loc[served["season"].le(2025)], k_def, k_qb)
    table_b = bl.attach(served.loc[served["season"].le(2025)], tv_train)
    cols_b = (*m.BASE_COLUMNS, bl.COLUMN)
    path_b = frozen_term(table_b, cols_b, (bl.COLUMN,), "blitz")

    rows = locked_inputs(served26)
    games26 = served.copy()
    for gid, r in rows.items():
        for c in ("home_projected_qb_id", "away_projected_qb_id"):
            games26.loc[games26["game_id"].eq(gid), c] = r["rec"][c]
    w.WARM = SEASON
    tv26 = bl.team_values(t, games26, k_def, k_qb)
    home = tv26.rename(columns={"team": "home_team", "val": "hv"})
    away = tv26.rename(columns={"team": "away_team", "val": "av"})
    f26 = games26.loc[games26["season"].eq(SEASON)]
    f26 = f26.merge(home, on=["game_id", "home_team"], how="left").merge(
        away, on=["game_id", "away_team"], how="left"
    )
    f26[bl.COLUMN] = f26["hv"].fillna(0.0) - f26["av"].fillna(0.0)
    f26 = f26.loc[f26["game_id"].isin(rows.keys())]

    tu = table_u.loc[table_u["season"].eq(SEASON) & table_u["game_id"].isin(rows.keys())]
    d_u = dict(zip(tu["game_id"], apply_frozen(path_u, tu), strict=True))
    d_b = dict(zip(f26["game_id"], apply_frozen(path_b, f26), strict=True))
    report = {
        "frozen_files": [str(path_u), str(path_b)],
        "locked_games": len(rows),
        "openers": len(openers),
        "n0_unit12": n0,
        "k_def": k_def,
        "k_qb": k_qb,
        "scored_at": datetime.now(UTC).isoformat(),
        "unit12_v2": score("unit12_v2", d_u, rows, served26, openers),
        "blitz": score("blitz", d_b, rows, served26, openers),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    shown = {k: v for k, v in report.items() if k != "n0_unit12"}
    print(json.dumps(shown, indent=1, default=float))


if __name__ == "__main__":
    main()
