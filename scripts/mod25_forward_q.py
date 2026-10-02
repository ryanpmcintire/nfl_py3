import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod24_forward_log as fl
import mod25_pipeline as mp

import nfl_ats.margin as mg
from nfl_ats.cli_commands.prediction import (
    MarginPredictRequest,
    _key_line_policy,
    _served_discrete_push_read,
    _served_home_side_offsets,
    _with_discrete_fallback,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.outcomes import score_outcome_week

ROOT = mp.REPO
OUT_DIR = ROOT / "artifacts" / "mod25_forward"
FROZEN_DIR = OUT_DIR / "frozen"
RUNS_DIR = OUT_DIR / "runs"
FROZEN_JSON = FROZEN_DIR / "frozen.json"
STUDENT_NPZ = FROZEN_DIR / "student_ridge_nomkt.npz"
FEATURES = fl.u3.FEATURES
SEASON = fl.SEASON
SET_NAME = fl.u3.SET_NAME
PROFILE = fl.u3.PROFILE
MARKET_ROOT = ROOT / "data" / "market" / "raw"
MARKET_FIRST_STAMP = "20260901"
STUDENT_ALPHA = 1.0
STUDENT_ARM = "ridge_nomkt"
LAST_TRAIN_SEASON = 2025
FULL_COLS, NOMKT_COLS = mp.v2_cols()
BASE_COLS = tuple(FULL_COLS)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_student():
    from sklearn.linear_model import Ridge

    nomkt = NOMKT_COLS
    f = pd.read_parquet(mp.SYN / "features.parquet")
    t = pd.read_parquet(mp.SYN / "truth.parquet")
    syn = f.merge(t[["game_id", "true_em"]], on="game_id", how="inner")
    syn = syn.loc[syn["sidx"].ge(1)].reset_index(drop=True)
    mu, sd = mp.z_stats(syn, nomkt)
    model = Ridge(alpha=STUDENT_ALPHA).fit(mp.z_apply(syn, nomkt, mu, sd), syn["true_em"].to_numpy())
    return nomkt, mu, sd, model, len(syn)


def real_rows():
    return mp.real_frame().sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def load_student():
    z = np.load(STUDENT_NPZ, allow_pickle=False)
    return list(z["cols"]), z["coef"], float(z["intercept"]), z["syn_mu"], z["syn_sd"], z["real_mu_2026"], z["real_sd_2026"]


def distilled_for(frame, cols, coef, intercept, real_mu, real_sd):
    out = np.zeros(len(frame))
    seasons = frame["season"].to_numpy()
    for s in sorted(set(seasons.tolist())):
        m = seasons == s
        if s >= SEASON:
            mu = pd.Series(real_mu, index=cols)
            sd = pd.Series(real_sd, index=cols)
        else:
            prior = frame.loc[frame["season"].lt(s)]
            if len(prior) < 200:
                prior = frame.loc[frame["season"].eq(s)]
            mu, sd = mp.z_stats(prior, cols)
        out[m] = mp.z_apply(frame.loc[m], cols, mu, sd) @ coef + intercept
    return out


class FrozenTwoStage:
    def __init__(self, base_cols, orig, kw, w, qbar, bbar):
        self.base_cols = list(base_cols)
        self.orig = orig
        self.kw = kw
        self.w = w
        self.qbar = qbar
        self.bbar = bbar

    def get_params(self, deep=True):
        return {"base_cols": self.base_cols, "orig": self.orig, "kw": self.kw, "w": self.w, "qbar": self.qbar, "bbar": self.bbar}

    def fit(self, X, y):
        self.plain = None
        if "distilled_margin" not in X.columns:
            self.plain = self.orig("ridge", 42, ridge_alpha=mp.BASE_ALPHA, **self.kw).fit(X, np.asarray(y, dtype=float))
            return self
        self.base = self.orig("ridge", 42, ridge_alpha=mp.BASE_ALPHA, **self.kw).fit(X[self.base_cols], np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        if self.plain is not None:
            return self.plain.predict(X)
        b = np.asarray(self.base.predict(X[self.base_cols]), dtype=float)
        q = (X["distilled_margin"] - X["spread_line"]).to_numpy(float)
        return b + self.w * ((q - self.qbar) - (b - self.bbar))


class CapturingTwoStage(mp.TwoStage):
    captured = []

    def predict(self, X):
        if getattr(self, "plain", None) is not None:
            return self.plain.predict(X)
        return super().predict(X)

    def fit(self, X, y):
        if "distilled_margin" not in X.columns:
            self.plain = self.orig("ridge", 42, ridge_alpha=mp.BASE_ALPHA, **self.kw).fit(X, np.asarray(y, dtype=float))
            return self
        self.plain = None
        super().fit(X, y)
        CapturingTwoStage.captured.append({"w": self.w, "qbar": self.qbar, "bbar": self.bbar, "max_train_season": int(X["season"].max()), "n_train": int(len(X))})
        return self


def patch_estimator(factory):
    full = FULL_COLS
    orig = mg.make_margin_estimator

    def patched(model_name, random_state=42, **kw):
        if model_name == "ridge":
            return factory(full, orig, {k: v for k, v in kw.items() if k == "suppressed_indicator_columns"})
        return orig(model_name, random_state, **kw)

    return orig, patched


def request_for(week):
    return MarginPredictRequest(
        features=FEATURES,
        season=SEASON,
        week=week,
        regressor="ridge",
        min_edge=fl.MIN_EDGE,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        feature_profile=PROFILE,
        ridge_alpha=mp.BASE_ALPHA,
        probability_method=fl.PROBABILITY_METHOD,
        line_sweep=False,
    )


def score_week(merged, week, columns, factory):
    request = request_for(week)
    original_set = FEATURE_SETS[SET_NAME]
    original_est = mg.make_margin_estimator
    FEATURE_SETS[SET_NAME] = tuple(columns)
    if factory is not None:
        _, patched = patch_estimator(factory)
        mg.make_margin_estimator = patched
    try:
        home_side = _served_home_side_offsets(merged, request)
        center = home_side["center_offsets"] if home_side is not None else None
        discrete = _served_discrete_push_read(merged, request)
        log = {}

        def score(reader):
            return score_outcome_week(
                merged,
                season=SEASON,
                week=week,
                regressor="ridge",
                min_edge=fl.MIN_EDGE,
                min_train_games=DEFAULT_MIN_TRAIN_GAMES,
                feature_profile=PROFILE,
                ridge_alpha=mp.BASE_ALPHA,
                probability_method=fl.PROBABILITY_METHOD,
                center_offsets=center,
                discrete_read=reader,
                discrete_read_log=log,
                key_line_pick_read=_key_line_policy(reader),
            )

        predictions, discrete = _with_discrete_fallback(score, discrete, log)
    finally:
        FEATURE_SETS[SET_NAME] = original_set
        mg.make_margin_estimator = original_est
    return predictions.loc[predictions["method"].eq("market_residual")].set_index("game_id"), discrete is not None


def with_distilled(features, cols, coef, intercept, real_mu, real_sd):
    reg = features["game_type"].eq("REG")
    rows = features.loc[reg].copy()
    rows["gameday"] = pd.to_datetime(rows["gameday"])
    rows = rows.sort_values(["season", "week", "game_id"])
    rows["distilled_margin"] = distilled_for(rows, cols, coef, intercept, real_mu, real_sd)
    return features.merge(rows[["game_id", "distilled_margin"]], on="game_id", how="left")


def cmd_freeze(args):
    if FROZEN_JSON.exists():
        raise SystemExit("already frozen")
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    nomkt, mu, sd, model, n_syn = fit_student()
    real = real_rows()
    prior = real.loc[real["season"].le(LAST_TRAIN_SEASON)]
    rmu, rsd = mp.z_stats(prior, nomkt)
    with open(STUDENT_NPZ, "xb") as handle:
        np.savez(
            handle,
            cols=np.array(nomkt),
            coef=model.coef_,
            intercept=np.array(model.intercept_),
            syn_mu=mu.to_numpy(float),
            syn_sd=sd.to_numpy(float),
            real_mu_2026=rmu.to_numpy(float),
            real_sd_2026=rsd.to_numpy(float),
        )
    cols, coef, intercept, _, _, rm, rs = load_student()
    saved = pd.read_parquet(mp.OUT / "real_distilled.parquet")[["game_id", f"{STUDENT_ARM}"]]
    check = real[["game_id", "season"]].copy()
    check["mine"] = distilled_for(real, cols, coef, intercept, rm, rs)
    check = check.merge(saved, on="game_id")
    reproduction = float((check["mine"] - check[STUDENT_ARM]).abs().max())
    features = pd.read_parquet(FEATURES)
    merged = with_distilled(features, cols, coef, intercept, rm, rs)
    full = FULL_COLS
    CapturingTwoStage.captured.clear()
    week1 = 1
    score_week(merged, week1, full + ["distilled_margin", "season"], CapturingTwoStage)
    fits = CapturingTwoStage.captured
    last = fits[-1]
    spread = {k: float(max(f[k] for f in fits) - min(f[k] for f in fits)) for k in ("w", "qbar", "bbar")}
    frozen = {
        "schema": "mod25_forward_q/1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "student": {
            "arm": STUDENT_ARM,
            "model": "Ridge",
            "alpha": STUDENT_ALPHA,
            "target": "synthetic true expected margin",
            "synthetic_world": f"{mp.VARIANT} scale {mp.SCALE}",
            "synthetic_rows": n_syn,
            "columns": len(cols),
            "file": STUDENT_NPZ.name,
            "sha256": sha256_file(STUDENT_NPZ),
            "reproduces_real_distilled_max_abs_diff": reproduction,
            "standardisation": "synthetic z-score inside the student; real inputs z-scored with the mean and sd of all regular-season real games from strictly earlier seasons, fixed at the 2025-final values for every 2026 game",
        },
        "combination": {
            "rule": "predicted_ats_margin = b + w * ((q - qbar) - (b - bbar)), b = base ridge on the served weak_stack columns refit on games before the target week, q = distilled_margin - pool_line",
            "w": last["w"],
            "qbar": last["qbar"],
            "bbar": last["bbar"],
            "fitted_through_season": last["max_train_season"],
            "fit_games": last["n_train"],
            "fit_method": "leave-one-season-out base predictions over seasons through 2025, one-term least squares of the ats margin residual",
            "capture_calls": len(fits),
            "capture_spread": spread,
            "weight_history_note": "out-of-season weight fell from .69 in 2010 to .19 in 2025 in the v2 grade",
        },
        "base_alpha": mp.BASE_ALPHA,
        "probability_method": fl.PROBABILITY_METHOD,
        "feature_table_sha256_at_freeze": sha256_file(FEATURES),
        "script_sha256_at_freeze": sha256_file(__file__),
        "pipeline_script_sha256": sha256_file(mp.__file__),
    }
    with open(FROZEN_JSON, "x") as handle:
        handle.write(json.dumps(frozen, indent=1))
    print(json.dumps(frozen, indent=1))


def load_frozen():
    frozen = json.loads(FROZEN_JSON.read_text())
    if sha256_file(STUDENT_NPZ) != frozen["student"]["sha256"]:
        raise SystemExit("student file hash differs from the frozen record")
    return frozen


def market_frame(since_stamp=MARKET_FIRST_STAMP):
    parts = []
    for folder in sorted(MARKET_ROOT.iterdir()):
        if not folder.is_dir() or folder.name < since_stamp or folder.name.endswith("private"):
            continue
        quotes = folder / "quotes.parquet"
        if not quotes.exists():
            continue
        q = pd.read_parquet(quotes, columns=["observed_at_utc", "nflverse_game_id", "market", "outcome_side", "line", "bookmaker_key"])
        q = q.loc[q["market"].eq("spreads") & q["outcome_side"].eq("HOME") & q["nflverse_game_id"].notna()]
        if not q.empty:
            parts.append(q)
    if not parts:
        return pd.DataFrame(columns=["game_id", "observed_at_utc", "market_line"])
    q = pd.concat(parts, ignore_index=True)
    q["observed_at_utc"] = pd.to_datetime(q["observed_at_utc"], utc=True)
    q["market_line"] = -q["line"].astype(float)
    g = q.groupby(["nflverse_game_id", "observed_at_utc"])["market_line"].median().reset_index()
    return g.rename(columns={"nflverse_game_id": "game_id"})


def market_as_of(market, game_id, before):
    g = market.loc[market["game_id"].eq(game_id) & market["observed_at_utc"].lt(before)]
    if g.empty:
        return None, None
    last = g.sort_values("observed_at_utc").iloc[-1]
    return float(last["market_line"]), last["observed_at_utc"].isoformat()


def side_of(p):
    return fl.side_of(p)


def record(week, now):
    frozen = load_frozen()
    cols, coef, intercept, _, _, rm, rs = load_student()
    features = pd.read_parquet(FEATURES)
    features["gameday"] = pd.to_datetime(features["gameday"])
    target = fl.week_frame(features, week)
    if target.empty:
        raise SystemExit(f"no {SEASON} week {week} games in the feature table")
    refused = []
    eligible = []
    for row in target.itertuples():
        if pd.notna(row.result):
            refused.append({"game_id": row.game_id, "reason": "outcome_present"})
        elif now >= row.deadline:
            refused.append({"game_id": row.game_id, "reason": "past_deadline"})
        else:
            eligible.append(row.game_id)
    if not eligible:
        print(json.dumps({"week": week, "logged": [], "refused": refused}, indent=1))
        return
    merged = with_distilled(features, cols, coef, intercept, rm, rs)
    full = FULL_COLS
    comb_cols = full + ["distilled_margin", "season"]
    c = frozen["combination"]

    def factory(base_cols, orig, kw):
        return FrozenTwoStage(base_cols, orig, kw, c["w"], c["qbar"], c["bbar"])

    base_pred, base_discrete = score_week(merged, week, list(BASE_COLS), None)
    comb_pred, comb_discrete = score_week(merged, week, comb_cols, factory)
    folder, served = fl.served_forecast(week)
    served = served.set_index("game_id") if served is not None else None
    market = market_frame()
    recorded = pd.Timestamp(datetime.now(UTC))
    tgt = target.set_index("game_id")
    feat_idx = merged.set_index("game_id")
    rows = []
    for game_id in eligible:
        meta = tgt.loc[game_id]
        if recorded >= meta["deadline"]:
            refused.append({"game_id": game_id, "reason": "past_deadline_at_write"})
            continue
        pool_line = float(meta["spread_line"])
        dm = float(feat_idx.loc[game_id, "distilled_margin"])
        bp = float(base_pred.loc[game_id, "home_cover_probability"])
        cp = float(comb_pred.loc[game_id, "home_cover_probability"])
        mline, mseen = market_as_of(market, game_id, recorded)
        row = {
            "game_id": game_id,
            "season": SEASON,
            "week": week,
            "kickoff": meta["kickoff"].isoformat(),
            "deadline": meta["deadline"].isoformat(),
            "recorded_at_utc": recorded.isoformat(),
            "pool_line": pool_line,
            "market_line": mline,
            "market_line_observed_at_utc": mseen,
            "distilled_margin": dm,
            "q": dm - pool_line,
            "base_home_cover_probability": bp,
            "base_side": side_of(bp),
            "combined_home_cover_probability": cp,
            "combined_side": side_of(cp),
            "q_side": "HOME" if dm - pool_line > 0 else ("AWAY" if dm - pool_line < 0 else "PASS"),
        }
        if served is not None and game_id in served.index:
            sp = float(served.loc[game_id, "home_cover_probability"])
            row["served_home_cover_probability"] = sp
            row["served_side"] = side_of(sp)
        rows.append(row)
    if not rows:
        print(json.dumps({"week": week, "logged": [], "refused": refused}, indent=1))
        return
    payload = {
        "schema": "mod25_forward_q/1",
        "season": SEASON,
        "week": week,
        "recorded_at_utc": recorded.isoformat(),
        "hashes": {
            "script_sha256": sha256_file(__file__),
            "frozen_json_sha256": sha256_file(FROZEN_JSON),
            "student_sha256": frozen["student"]["sha256"],
            "feature_table_sha256": sha256_file(FEATURES),
        },
        "frozen_w": c["w"],
        "discrete_lattice_read": {"base": base_discrete, "combined": comb_discrete},
        "served_forecast_dir": None if folder is None else folder.name,
        "refused": refused,
        "rows": rows,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = recorded.strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{SEASON}-week-{week:02d}-{stamp}.json"
    with open(path, "x") as handle:
        handle.write(json.dumps(payload, indent=1, default=str))
    print(json.dumps({"path": str(path), "logged": [r["game_id"] for r in rows], "refused": refused}, indent=1))


def canonical_rows():
    log = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            log.append({**row, "run_file": path.name})
    if not log:
        return pd.DataFrame(), 0
    df = pd.DataFrame(log)
    df["recorded_at_utc"] = pd.to_datetime(df["recorded_at_utc"], utc=True)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    df["deadline"] = pd.to_datetime(df["deadline"], utc=True)
    df = df.loc[df["recorded_at_utc"].lt(df["deadline"])]
    canon = df.sort_values("recorded_at_utc").groupby("game_id").tail(1)
    return canon, len(df)


def mean_ci(values):
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "share_toward": float((v > 0).mean()), "share_away": float((v < 0).mean())}


def status():
    features = pd.read_parquet(FEATURES, columns=["game_id", "season", "week", "kickoff"])
    canon, n_rows = canonical_rows()
    now = pd.Timestamp(datetime.now(UTC))
    coverage = []
    for week in range(fl.FIRST_WEEK, fl.LAST_WEEK + 1):
        sub = features.loc[features["season"].eq(SEASON) & features["week"].eq(week)]
        if sub.empty:
            coverage.append({"week": week, "games": 0})
            continue
        done = canon.loc[canon["week"].eq(week)] if not canon.empty else canon
        coverage.append(
            {
                "week": week,
                "games": len(sub),
                "logged_games": int(len(done)),
                "already_kicked_off": int(pd.to_datetime(sub["kickoff"], utc=True).le(now).sum()),
            }
        )
    out = {"coverage": coverage, "logged_rows_before_deadline": n_rows, "canonical_games": 0 if canon.empty else len(canon)}
    if not canon.empty:
        market = market_frame()
        values = {"q_side": [], "base_side": [], "combined_side": []}
        for row in canon.itertuples():
            if row.kickoff > now:
                continue
            close, _ = market_as_of(market, row.game_id, row.kickoff)
            if close is None:
                continue
            move = close - row.pool_line
            for key, side in (("q_side", row.q_side), ("base_side", row.base_side), ("combined_side", row.combined_side)):
                if side in ("HOME", "AWAY"):
                    values[key].append(move if side == "HOME" else -move)
        out["line_value_pool_to_close_toward_side_points"] = {k: mean_ci(v) for k, v in values.items()}
    print(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description="Forward log of the MOD-25 distilled-student q term")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        cmd_freeze(args)
        return
    if args.status:
        status()
        return
    now = pd.Timestamp(datetime.now(UTC))
    week = args.week
    if week is None:
        features = pd.read_parquet(FEATURES, columns=["season", "week", "kickoff"])
        week = fl.current_week(features, now)
        if week is None:
            print("no upcoming 2026 week in range")
            return
    if not fl.FIRST_WEEK <= week <= fl.LAST_WEEK:
        raise SystemExit("week outside the logged range")
    record(week, now)


if __name__ == "__main__":
    main()
