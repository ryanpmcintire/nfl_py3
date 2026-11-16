from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod22_qb_input_repair as m22  # noqa: E402
from mod27_fenced_repairs_replay import FOUR, Fit, log_loss, paired, record  # noqa: E402

import nfl_ats.margin as margin  # noqa: E402
from nfl_ats import clv  # noqa: E402
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.outcomes import score_outcome_week  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v4_candidate"
TABLE = ROOT / "data/processed/game_features_weak_stack.parquet"
ACTIVE = json.loads((ROOT / "artifacts/active_ats_model.json").read_text(encoding="utf-8"))
PICK = ROOT / "artifacts/pick_probability/20261116T171555Z"
ARCHIVE = ROOT / "data/raw/forecast_archive/pool_decision_2009_2025_aware/forecasts.parquet"
BULLETINS = ROOT / "data/raw/forecast_archive/mos_bulletins"
STATIONS = ROOT / "registry/reference/stadium_station_map.csv"
PREDICTIONS = ROOT / "artifacts/margin_predictions"
SET_KEY = margin.margin_feature_set("market_residual", ACTIVE["feature_profile"])
FULL = tuple(margin.FEATURE_SETS[SET_KEY])
WEATHER = ("temp", "wind")
DROPPED = tuple(c for c in FULL if c not in WEATHER)
QB = tuple(c for c in FULL if "_qb_" in c)
ALPHA = float(ACTIVE["ridge_alpha"])
OPEN_AIR = ("outdoors", "open")
KNOTS_TO_MPH = 1.15078
MOS_MATCH = pd.Timedelta(hours=3)
POOL_CUTOFF_HOUR_ET = 16
LIVE_SEASON = 2026
LIVE_WEEKS = range(1, 11)
ARMS = {
    "S": (False, None),
    "Q": (True, None),
    "W": (False, "forecast"),
    "D": (False, "drop"),
    "QW": (True, "forecast"),
    "QD": (True, "drop"),
}
FAMILY = "v4_pinned_batch"
CACHED = False
EPS = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def kickoff_and_cutoff(frame):
    et = pd.to_datetime(
        frame["gameday"].astype(str) + " " + frame["gametime"].fillna("13:00").astype(str)
    ).dt.tz_localize("America/New_York")
    day = pd.to_datetime(frame["gameday"])
    shift = np.where(day.dt.dayofweek.eq(0), -1, (6 - day.dt.dayofweek) % 7)
    sunday = (day + pd.to_timedelta(shift, unit="D")).dt.tz_localize("America/New_York")
    deadline = sunday + pd.Timedelta(hours=POOL_CUTOFF_HOUR_ET)
    kick = et.dt.tz_convert("UTC")
    return kick, np.minimum(kick, deadline.dt.tz_convert("UTC"))


def live_forecasts(table):
    rows = table.loc[table["season"].eq(LIVE_SEASON)].copy()
    schedule = pd.read_parquet(
        sorted(glob.glob(str(ROOT / "data/raw/2026*/schedules.parquet")))[-1],
        columns=["game_id", "stadium", "roof"],
    ).drop_duplicates("game_id")
    schedule["game_id"] = schedule["game_id"].astype(str)
    rows = rows.drop(columns=[c for c in ("stadium", "roof") if c in rows]).merge(
        schedule, on="game_id", how="left"
    )
    stations = pd.read_csv(STATIONS)[["stadium", "icao_station"]].drop_duplicates("stadium")
    rows = rows.merge(stations, on="stadium", how="left")
    rows["kick"], rows["cutoff"] = kickoff_and_cutoff(rows)
    cache = {}
    out = {}
    for _, r in rows.iterrows():
        if str(r["roof"]).lower() not in OPEN_AIR or not isinstance(r["icao_station"], str):
            continue
        best = None
        for model in ("GFS", "MEX"):
            key = (model, r["icao_station"])
            if key not in cache:
                loaded = []
                for f in glob.glob(str(BULLETINS / model / r["icao_station"] / "*.json")):
                    j = json.loads(Path(f).read_text(encoding="utf-8"))
                    t = pd.DataFrame(j.get("data", []))
                    if t.empty:
                        continue
                    loaded.append((pd.Timestamp(j["cached_at_utc"]), t))
                cache[key] = loaded
            for cached, t in cache[key]:
                if cached > r["cutoff"]:
                    continue
                ft = pd.to_datetime(t["ftime"], utc=True)
                rt = pd.to_datetime(t["runtime"], utc=True)
                gap = (ft - r["kick"]).abs()
                i = int(np.lexsort((-rt.astype("int64").to_numpy(), gap.to_numpy()))[0])
                candidate = (gap.iloc[i], -rt.iloc[i].value)
                if gap.iloc[i] <= MOS_MATCH and (best is None or candidate < best[0]):
                    best = (
                        candidate,
                        float(t["tmp"].iloc[i]),
                        float(t["wsp"].iloc[i]) * KNOTS_TO_MPH,
                    )
        if best:
            out[r["game_id"]] = best[1:]
    return out, int(rows["roof"].astype(str).str.lower().isin(OPEN_AIR).sum())


def forecast_weather(table):
    fc = pd.read_parquet(ARCHIVE)
    fc = fc.loc[fc["fetch_status"].eq("ok")].drop_duplicates("game_id").set_index("game_id")
    out = table.copy()
    hist = out["season"].lt(LIVE_SEASON)
    roof = fc["roof"].astype(str).str.lower().reindex(out["game_id"]).to_numpy()
    open_air = np.isin(roof, OPEN_AIR)
    temp = fc["forecast_temp_f"].reindex(out["game_id"]).to_numpy(float)
    wind = fc["forecast_wind_mph"].reindex(out["game_id"]).to_numpy(float)
    out.loc[hist, "temp"] = np.where(open_air, temp, np.nan)[hist.to_numpy()]
    out.loc[hist, "wind"] = np.where(open_air, wind, np.nan)[hist.to_numpy()]
    live, live_open_air = live_forecasts(table)
    live_rows = ~hist
    out.loc[live_rows, ["temp", "wind"]] = np.nan
    for gid, (t, w) in live.items():
        k = out["game_id"].eq(gid)
        out.loc[k, "temp"] = t
        out.loc[k, "wind"] = w
    coverage = {
        "history_open_air_with_forecast": int((open_air & ~np.isnan(temp))[hist.to_numpy()].sum()),
        "history_observed_temp_present": int(table.loc[hist, "temp"].notna().sum()),
        "live_with_forecast": len(live),
        "live_open_air": live_open_air,
    }
    return out, live, coverage


def qb_repaired(table):
    inputs = m22.load_inputs()
    orig = m22.enrich(m22.load_variant("orig"), inputs, list(QB)).set_index("game_id")
    ab = m22.enrich(m22.load_variant("repair_ab"), inputs, list(QB)).set_index("game_id")
    out = table.copy()
    ids = out["game_id"]
    identity = {}
    for c in QB:
        a = orig[c].reindex(ids).to_numpy(float)
        b = out[c].to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        identity[c] = {
            "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum()),
            "max_abs": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0,
        }
        out[c] = ab[c].reindex(ids).to_numpy(float)
    return out, identity


def with_columns(columns, fn):
    saved = margin.FEATURE_SETS[SET_KEY]
    margin.FEATURE_SETS[SET_KEY] = tuple(columns)
    try:
        return fn()
    finally:
        margin.FEATURE_SETS[SET_KEY] = saved


def opener(features, name, columns):
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(features[list(columns)], index=False).to_numpy().tobytes()
    ).hexdigest()[:12]
    config = {
        "feature_profile": ACTIVE["feature_profile"],
        "regressor": ACTIVE["regressor"],
        "target": "market_residual",
        "probability_method": ACTIVE["probability_method"],
        "ridge_alpha": ALPHA,
        "v4_candidate_arm": f"{name}_{digest}",
    }
    scored = with_columns(
        columns,
        lambda: clv.opener_pick_evaluation(
            m22.MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        ),
    )
    return scored.drop_duplicates("game_id").set_index("game_id")


def stage_two(pop, model_prob):
    frame = pop.copy()
    frame["model_logit"] = logit(model_prob)
    p = pd.Series(np.nan, index=frame.index)
    folds = {}
    for season in sorted(frame["season"].unique()):
        fit = Fit(FOUR, frame[frame["season"] != season])
        held = frame["season"] == season
        p.loc[held] = fit.predict(frame[held])
        folds[str(int(season))] = fit.natural()
    full = Fit(FOUR, frame)
    insample = full.predict(frame)
    return p.to_numpy(float), folds, full, insample


def season_table(pop, p):
    y = pop["home_covered"].to_numpy(float)
    rows = {}
    for season in sorted(pop["season"].unique()):
        m = (pop["season"] == season).to_numpy()
        hit = ((p[m] >= 0.5) == (y[m] == 1)).mean()
        rows[str(int(season))] = {
            "games": int(m.sum()),
            "record": record(p[m], y[m]),
            "accuracy": float(hit),
            "log_loss": float(log_loss(p[m], y[m]).mean()),
        }
    return rows


def grade(tables, pop):
    raw = {}
    for name, (features, columns) in tables.items():
        cached = OUT / f"opener_{name}.parquet"
        if CACHED and cached.exists():
            raw[name] = pd.read_parquet(cached)
            continue
        scored = opener(features, name, columns)
        scored.to_parquet(cached)
        raw[name] = scored
    ids = pop["game_id"].to_numpy()
    y = pop["home_covered"].to_numpy(float)
    served_stored = pop["out_of_season_home_probability"].to_numpy(float)
    stored_model = pop["model_probability"].to_numpy(float)
    result = {"games": len(pop)}
    probs = {}
    for name in tables:
        mp = raw[name]["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
        p, folds, full, insample = stage_two(pop, mp)
        probs[name] = p
        result[name] = {
            "missing_model": int(np.isnan(mp).sum()),
            "record": record(p, y),
            "accuracy": float(((p >= 0.5) == (y == 1)).mean()),
            "log_loss": float(log_loss(p, y).mean()),
            "brier": float(((p - y) ** 2).mean()),
            "in_sample_log_loss": float(log_loss(insample, y).mean()),
            "model_only_record": record(mp, y),
            "fold_coefficients": folds,
            "full_coefficients": full.natural(),
            "per_season": season_table(pop, p),
        }
    s_model = raw["S"]["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
    result["identity"] = {
        "S_model_vs_stored_max_abs": float(np.nanmax(np.abs(s_model - stored_model))),
        "S_served_vs_stored_max_abs": float(np.nanmax(np.abs(probs["S"] - served_stored))),
        "stored_served_record": record(served_stored, y),
    }
    frame = pop.assign(week_block=pop["season"] * 100 + pop["week"])
    pushes = {}
    for name in tables:
        if name == "S":
            continue
        a, b = probs[name], probs["S"]
        changed = (a >= 0.5) != (b >= 0.5)
        result[name]["vs_S"] = {
            "season_block": paired(frame, b, a, "season"),
            "week_block": paired(frame, b, a, "week_block"),
            "changed_picks": int(changed.sum()),
            "changed_record_arm": record(a[changed], y[changed]) if changed.any() else "0-0",
            "changed_record_served": record(b[changed], y[changed]) if changed.any() else "0-0",
        }
        allp = raw[name]["home_cover_probability_at_open"]
        base = raw["S"]["home_cover_probability_at_open"].reindex(allp.index)
        push = raw[name]["margin_vs_open"].eq(0.0)
        pushes[name] = int(((allp >= 0.5) != (base >= 0.5))[push].sum())
    result["push_games"] = int(raw["S"]["margin_vs_open"].eq(0.0).sum())
    result["push_model_side_changes"] = pushes
    return result


def served_dir(table, week):
    dirs = sorted(PREDICTIONS.glob(f"{LIVE_SEASON}-week-{week:02d}-*"))
    rows = table[(table.season == LIVE_SEASON) & (table.week == week)]
    days = pd.to_datetime(rows["gameday"])
    sunday = days[days.dt.dayofweek == 6].min()
    cutoff = (sunday + pd.Timedelta(hours=17)).strftime("%Y%m%dT%H%M%SZ")
    eligible = [d for d in dirs if d.name.split("-")[-1] < cutoff]
    return (eligible or dirs)[-1]


def replay(tables, live_weather, coef):
    base_table = tables["S"][0]
    rows = []
    for week in LIVE_WEEKS:
        d = served_dir(base_table, week)
        served = pd.read_csv(d / "predictions.csv", low_memory=False)
        served["game_id"] = served["game_id"].astype(str)
        served = served.drop_duplicates("game_id").set_index("game_id")
        card = pd.read_csv(d / "pool_card.csv").drop_duplicates("game_id").set_index("game_id")
        probs = {}
        for name, (features, columns) in tables.items():
            frame = features.copy()
            k = frame["season"].eq(LIVE_SEASON) & frame["week"].eq(week)
            ids = frame.loc[k, "game_id"]
            for c in FULL:
                if c in served.columns and c not in QB and c not in WEATHER:
                    frame.loc[k, c] = pd.to_numeric(
                        served[c].reindex(ids), errors="coerce"
                    ).to_numpy()
            if name in ("S", "Q", "D", "QD"):
                frame.loc[k, list(WEATHER)] = np.nan
            if name in ("S", "W", "D") and all(c in served.columns for c in QB):
                for c in QB:
                    frame.loc[k, c] = pd.to_numeric(served[c].reindex(ids), errors="coerce").to_numpy()
            scored = with_columns(
                columns,
                lambda frame=frame: score_outcome_week(
                    frame,
                    season=LIVE_SEASON,
                    week=week,
                    feature_profile=ACTIVE["feature_profile"],
                    ridge_alpha=ALPHA,
                ),
            )
            scored = scored[scored["method"] == "market_residual"]
            probs[name] = scored.drop_duplicates("game_id").set_index("game_id")[
                "home_cover_probability"
            ].astype(float)
        result = base_table.set_index("game_id")["result"]
        for gid in card.index:
            if gid not in probs["S"].index:
                continue
            c = card.loc[gid]
            home = str(c["pool_side"]).upper() == "HOME"
            p_card_home = float(c["pick_probability"]) if home else 1.0 - float(c["pick_probability"])
            home_line = float(c["pick_line"]) if home else -float(c["pick_line"])
            margin_home = result.get(gid, np.nan)
            graded = pd.notna(margin_home) and margin_home + home_line != 0.0
            entry = {
                "week": week,
                "game_id": gid,
                "card_home": round(p_card_home, 4),
                "home_line": home_line,
                "home_covered": (float(margin_home + home_line > 0.0) if graded else None),
                "artifact": d.name,
            }
            for name in tables:
                if name == "S":
                    continue
                delta = float(logit(probs[name][gid]) - logit(probs["S"][gid]))
                p_arm = float(sigmoid(logit(p_card_home) + coef[name] * delta))
                entry[f"{name}_model_logit_delta"] = round(delta, 4)
                entry[f"{name}_home"] = round(p_arm, 4)
                entry[f"{name}_side_change"] = bool((p_arm >= 0.5) != (p_card_home >= 0.5))
            entry["live_weather"] = live_weather.get(gid)
            rows.append(entry)
    out = pd.DataFrame(rows)
    summary = {}
    for name in tables:
        if name == "S":
            continue
        ch = out.loc[out[f"{name}_side_change"]]
        g = ch.loc[ch["home_covered"].notna()]
        arm_hit = ((g[f"{name}_home"] >= 0.5) == (g["home_covered"] == 1)).sum()
        summary[name] = {
            "games": len(out),
            "side_changes": len(ch),
            "changed_games": ch["game_id"].tolist(),
            "graded_changed": len(g),
            "arm_record_on_changed": f"{int(arm_hit)}-{int(len(g) - arm_hit)}",
            "card_record_on_changed": f"{int(len(g) - arm_hit)}-{int(arm_hit)}",
            "mean_abs_model_logit_delta": float(out[f"{name}_model_logit_delta"].abs().mean()),
        }
    return out, summary


def main():
    parser = argparse.ArgumentParser(description="Grade the post-season v4 base-model candidate.")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    global CACHED
    CACHED = args.cached
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(TABLE)
    table["game_id"] = table["game_id"].astype(str)
    q_table, qb_identity = qb_repaired(table)
    w_table, live_weather, coverage = forecast_weather(table)
    qw_table, _, _ = forecast_weather(q_table)
    tables = {
        "S": (table, FULL),
        "Q": (q_table, FULL),
        "W": (w_table, FULL),
        "D": (table, DROPPED),
        "QW": (qw_table, FULL),
        "QD": (q_table, DROPPED),
    }
    pop = pd.read_parquet(PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    graded = grade(tables, pop)
    graded["qb_identity_orig_vs_table"] = qb_identity
    graded["weather_coverage"] = coverage
    (OUT / "results.json").write_text(json.dumps(graded, indent=2, default=str), encoding="utf-8")
    if not args.skip_replay:
        coef = {n: graded[n]["full_coefficients"]["model_logit"] for n in tables if n != "S"}
        rows, summary = replay(tables, live_weather, coef)
        rows.to_csv(OUT / "replay_2026.csv", index=False)
        graded["replay_2026"] = summary
        (OUT / "results.json").write_text(
            json.dumps(graded, indent=2, default=str), encoding="utf-8"
        )
    print(json.dumps({k: v for k, v in graded.items() if k != "qb_identity_orig_vs_table"}, default=str)[:6000])


if __name__ == "__main__":
    main()
