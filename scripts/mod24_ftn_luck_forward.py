import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod24_ftn_turnover_luck as m24
import mod25_forward_q as mfq

from nfl_ats.margin import (
    _PROFILE_SUPPRESSED_MISSING_INDICATORS,
    _target_values,
    make_margin_estimator,
)
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

OUT = m24.OUT_DIR
FROZEN = OUT / "frozen_luck_terms.json"
FORWARD = OUT / "forward_2026.json"
FORECASTS = m24.ROOT / "artifacts" / "margin_predictions"
SEASON = 2026
WEEKS = tuple(range(1, 8))
DRAWS = 20000
SEED = 20261103


def table_and_plays():
    served = pd.read_parquet(m24.SERVED_FEATURES)
    plays = m24.load_plays()
    per_team = m24.build_luck(plays, served)
    return served, plays, m24.attach(served, per_team), per_team


def stage_freeze():
    _, _, table, _ = table_and_plays()
    columns = (*m24.BASE_COLUMNS, *m24.LUCK_COLUMNS)
    suppressed = tuple(
        c for c in _PROFILE_SUPPRESSED_MISSING_INDICATORS.get(m24.PROFILE, ()) if c in columns
    )
    frame = table.loc[
        table["game_type"].eq("REG") & table["ats_margin"].notna() & table["season"].lt(SEASON)
    ]
    model = make_margin_estimator(
        "ridge", ridge_alpha=m24.BASE_ALPHA, suppressed_indicator_columns=suppressed
    )
    y = _target_values(frame, "market_residual")
    model.fit(frame.loc[:, list(columns)], y)
    names = list(model[:-1].get_feature_names_out())
    coef = np.ravel(model[-1].coef_)
    luck = {n: float(c) for n, c in zip(names, coef, strict=True) if n in m24.LUCK_COLUMNS}
    resid = np.asarray(y, dtype=float) - model.predict(frame.loc[:, list(columns)])
    payload = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "fit_seasons": [int(frame["season"].min()), int(frame["season"].max())],
        "fit_rows": len(frame),
        "ftn_rows_seasons": [2022, int(frame["season"].max())],
        "alpha": m24.BASE_ALPHA,
        "coefficients_per_scaled_unit": luck,
        "residual_sd_margin_points": float(np.std(resid, ddof=1)),
        "scaler": {},
        "script": "scripts/mod24_ftn_luck_forward.py freeze",
    }
    transformed = model[:-1].transform(frame.loc[:, list(columns)])
    for n in m24.LUCK_COLUMNS:
        i = names.index(n)
        raw = frame[n].to_numpy(dtype=float)
        payload["scaler"][n] = {
            "mean": float(np.mean(raw)),
            "sd": float(np.std(raw)),
            "check_max_abs_vs_pipeline": float(
                np.max(np.abs((raw - np.mean(raw)) / np.std(raw) - transformed[:, i]))
            ),
        }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["sha256_of_body"] = hashlib.sha256(body.encode()).hexdigest()
    FROZEN.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1))


def load_frozen():
    payload = json.loads(FROZEN.read_text())
    digest = payload.pop("sha256_of_body")
    body = json.dumps(payload, indent=1, sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == digest
    return payload, digest


def locked_rows(table):
    t26 = table.loc[table["season"].eq(SEASON) & table["week"].isin(WEEKS)]
    k = pd.to_datetime(t26["kickoff"], utc=True)
    deadlines = {}
    for _, g in k.groupby(t26["week"]):
        lock = sunday_pick_lock(g)
        for gid, kk in zip(t26.loc[g.index, "game_id"], g, strict=True):
            deadlines[gid] = pick_deadline(kk, lock)
    by_week = {}
    for f in sorted(FORECASTS.glob(f"{SEASON}-week-*")):
        by_week.setdefault(int(f.name.split("-")[2]), []).append(f)
    out = {}
    for gid, dl in deadlines.items():
        week = int(gid.split("_")[1])
        for f in reversed(by_week.get(week, [])):
            stamp = pd.Timestamp(
                datetime.strptime(f.name.split("-")[-1], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            )
            if stamp >= dl or not (f / "recommendations.csv").exists():
                continue
            rec = pd.read_csv(f / "recommendations.csv")
            hit = rec.loc[rec["game_id"].eq(gid)]
            if len(hit):
                r = hit.iloc[0]
                out[gid] = {
                    "folder": f.name,
                    "stamp": stamp,
                    "deadline": dl,
                    "p": float(r["home_cover_probability"]),
                    "line": float(r["spread_line"]),
                }
                break
    return out


def assert_week_pregame(plays, served, per_team):
    dates = dict(zip(served["game_id"], pd.to_datetime(served["gameday"]), strict=True))
    day = plays["game_id"].map(dates)
    for week in WEEKS:
        games = served.loc[served["season"].eq(SEASON) & served["week"].eq(week)]
        cutoff = pd.to_datetime(games["gameday"]).min()
        blanked = plays.copy()
        future = day.ge(cutoff).to_numpy()
        for c in ("interception", "is_interception_worthy", "is_catchable_ball", "is_drop"):
            blanked.loc[future, c] = 0.0
        trunc = m24.build_luck(blanked, served)
        ids = set(games["game_id"])
        a = per_team.loc[per_team["game_id"].isin(ids)].set_index(["game_id", "team"])
        b = trunc.loc[trunc["game_id"].isin(ids)].set_index(["game_id", "team"])
        both = a.index.intersection(b.index)
        cols = list(m24.TEAM_FEATURES)
        same = np.isclose(
            a.loc[both, cols].to_numpy(float),
            b.loc[both, cols].to_numpy(float),
            equal_nan=True,
            rtol=0,
            atol=1e-9,
        )
        assert len(both) > 0 and same.all(), f"week {week} leaks"
        print("week", week, "pregame truncation passed", len(both), flush=True)


def boot(delta, groups):
    keys = pd.Series(groups)
    blocks = [delta[(keys == k).to_numpy()] for k in keys.unique()]
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(SEED)
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        idx = rng.integers(0, len(blocks), len(blocks))
        out[i] = sums[idx].sum() / sizes[idx].sum()
    return {
        "mean": float(delta.mean()),
        "low": float(np.percentile(out, 2.5)),
        "high": float(np.percentile(out, 97.5)),
        "probability_positive": float((out > 0).mean()),
    }


def stage_score():
    frozen, digest = load_frozen()
    served, plays, table, per_team = table_and_plays()
    assert_week_pregame(plays, served, per_team)
    locked = locked_rows(table)
    market = mfq.market_frame()
    t26 = table.loc[table["game_id"].isin(locked.keys())].copy()
    k = pd.to_datetime(t26["kickoff"], utc=True)
    opener = {}
    for gid, kick in zip(t26["game_id"], k, strict=True):
        et = kick.tz_convert("America/New_York").normalize()
        tue = et - pd.Timedelta(days=(et.weekday() - 1) % 7)
        g = market.loc[
            market["game_id"].eq(gid) & market["observed_at_utc"].ge(tue.tz_convert("UTC"))
        ]
        if not g.empty:
            opener[gid] = float(g.sort_values("observed_at_utc").iloc[0]["market_line"])
    delta = np.zeros(len(t26))
    for n in m24.LUCK_COLUMNS:
        s = frozen["scaler"][n]
        z = (t26[n].to_numpy(float) - s["mean"]) / s["sd"]
        delta += frozen["coefficients_per_scaled_unit"][n] * z
    t26["delta_margin"] = delta
    t26["p_served"] = t26["game_id"].map(lambda g: locked[g]["p"])
    t26["locked_line"] = t26["game_id"].map(lambda g: locked[g]["line"])
    t26["opener"] = t26["game_id"].map(opener)
    sigma = frozen["residual_sd_margin_points"]
    p = np.clip(t26["p_served"].to_numpy(float), 1e-6, 1 - 1e-6)
    t26["p_luck"] = norm.cdf(norm.ppf(p) + delta / sigma)
    assert all(locked[g]["stamp"] < locked[g]["deadline"] for g in t26["game_id"])
    settled = t26.loc[t26["result"].notna()].copy()
    report = {
        "frozen_sha256": digest,
        "frozen_at_utc": frozen["frozen_at_utc"],
        "run_at_utc": datetime.now(UTC).isoformat(),
        "games_locked": len(t26),
        "games_settled": len(settled),
        "games_opener_equals_locked_line": int((settled["opener"] == settled["locked_line"]).sum()),
        "mean_abs_delta_margin": float(np.abs(delta).mean()),
    }
    for label in ("opener", "locked_line"):
        s = settled.loc[settled[label].notna()].copy()
        if label == "opener":
            s = s.loc[s["opener"].eq(s["locked_line"])]
        s = s.loc[s["result"].ne(s[label])]
        y = (s["result"] - s[label] > 0).to_numpy(float)
        ps = np.clip(s["p_served"].to_numpy(float), 1e-6, 1 - 1e-6)
        pl = np.clip(s["p_luck"].to_numpy(float), 1e-6, 1 - 1e-6)
        cs = ((ps > 0.5) == (y == 1)).astype(float)
        cl = ((pl > 0.5) == (y == 1)).astype(float)
        dec = (ps > 0.5) != (pl > 0.5)

        def ll(q, y=y):
            return -(y * np.log(q) + (1 - y) * np.log(1 - q))

        groups = s["week"].to_numpy()
        report[label] = {
            "n": len(s),
            "served_record": f"{int(cs.sum())}-{int(len(s) - cs.sum())}",
            "luck_record": f"{int(cl.sum())}-{int(len(s) - cl.sum())}",
            "decisive_flips": int(dec.sum()),
            "decisive_luck_W_L": f"{int(cl[dec].sum())}-{int(dec.sum() - cl[dec].sum())}",
            "decisive_served_W_L": f"{int(cs[dec].sum())}-{int(dec.sum() - cs[dec].sum())}",
            "acc_points": boot(100 * (cl - cs), groups),
            "ll_gain": boot(ll(ps) - ll(pl), groups),
            "brier_gain": boot((ps - y) ** 2 - (pl - y) ** 2, groups),
            "ll_served": float(ll(ps).mean()),
            "ll_luck": float(ll(pl).mean()),
        }
    FORWARD.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    if sys.argv[1:] == ["freeze"]:
        stage_freeze()
    elif sys.argv[1:] == ["score"]:
        stage_score()
    else:
        raise SystemExit("usage: mod24_ftn_luck_forward.py freeze|score")


if __name__ == "__main__":
    main()
