import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U8 = Path(__file__).resolve().parent / "mod23_unit8_span.py"
NS = {"__file__": str(U8), "__name__": "mod23_unit8_span_import"}
exec(compile(U8.read_text().rsplit("\nmain()", 1)[0], str(U8), "exec"), NS)

import mod24_more_games as base  # noqa: E402
import pooled_screen as ps  # noqa: E402
import xlg11_u4_h14 as h14  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from nfl_ats import clv  # noqa: E402
from nfl_ats import margin as margin_mod  # noqa: E402

u3 = NS["u3"]
HARNESS = NS["HARNESS"]
FULL = NS["FULL"]
BASE_ALPHA = NS["BASE_ALPHA"]
BASE5 = NS["BASE5"]
market_only = NS["market_only"]
boot_season = NS["boot_season"]
COLFAM = HARNESS["COLFAM"]
PINNED = HARNESS["PINNED"]
FIRST = HARNESS["TRUE_SEASONS"][0]
OUT = u3.ROOT / "artifacts" / "xlg11"
ORIG = margin_mod.make_margin_estimator
GRID_DECADES = 8
GRID = np.array([10.0**k for k in range(GRID_DECADES)])
PERMS = 100
AFFINE_CORR = 0.999999
IDENTITY = (802, 1503)
ARM_INDEX = {"R": 0, "Z": 1, "ID": 2, "C10": 3}
PLACEBO_START = 5
NONMARKET = [c for c in FULL if COLFAM[c] != PINNED]
MARKET = tuple(c for c in FULL if COLFAM[c] == PINNED)
STATE = {"arm": None, "ctx": None, "perms": None, "cfb": None, "shared": None}
MAPS = {}
LOG = []
STASH = []


def shared_positions(X, Z):
    cols = list(X.columns)
    found = []
    for c in STATE["shared"]:
        if c not in cols:
            continue
        x = X[c].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 3 or np.std(x[ok]) == 0.0:
            continue
        i = cols.index(c)
        for j in [i] + [k for k in range(Z.shape[1]) if k != i]:
            z = Z[ok, j]
            if np.std(z) == 0.0:
                continue
            if np.corrcoef(x[ok], z)[0, 1] > AFFINE_CORR:
                found.append((c, j))
                break
    return found


def college_prior(Z, y, ctx):
    X = ctx["X"]
    cols = list(X.columns)
    pos = shared_positions(X, Z)
    cfb = STATE["cfb"]
    past = cfb.loc[cfb["gameday"].lt(ctx["cut"])]
    shared = {c for c, _j in pos}
    med = X[cols].astype(float).median()
    data = {}
    for c in cols:
        if c in shared:
            v = past[c].to_numpy(dtype=float).copy()
            v[np.isnan(v)] = med[c]
        else:
            v = np.full(len(past), med[c])
        data[c] = v
    t = pd.DataFrame(data, columns=cols)
    for _name, step in ctx["pipe"].steps[:-1]:
        t = step.transform(t)
    t = np.asarray(t, dtype=float)
    j = [p for _c, p in pos]
    cy = past["ats_margin"].to_numpy(dtype=float)
    ratio = float(np.std(y, ddof=1) / np.std(cy, ddof=1))
    prior = np.zeros(Z.shape[1])
    prior[j] = ratio * Ridge(alpha=BASE_ALPHA).fit(t[:, j], cy).coef_
    return prior, j, len(past), ratio


class ArmRidge(BaseEstimator, RegressorMixin):
    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)
        ctx = STATE["ctx"]
        seasons = ctx["seasons"]
        n, d = Z.shape
        prior, j, cfb_n, ratio = college_prior(Z, y, ctx)
        zero = np.zeros(d)
        cols = [prior, zero]
        if STATE["arm"] == "R":
            for p in STATE["perms"]:
                q = np.argsort(p[: len(j)])
                c = np.zeros(d)
                c[j] = prior[j][q]
                cols.append(c)
        centres = np.column_stack(cols)
        total, held = h14.loso(Z, y, seasons, centres, GRID)
        best = held.argmin(axis=0)
        lam_k = GRID[best]
        dec = h14.decomp(total)
        beta, b0 = h14.betas(dec, centres, lam_k)
        fc = np.column_stack([zero, zero, prior])
        fb, f0 = h14.betas(dec, fc, [lam_k[0], BASE_ALPHA, BASE_ALPHA])
        self.multi_coef_ = np.column_stack([beta[:, :1], fb, beta[:, 1:]])
        self.multi_int_ = np.concatenate([b0[:1], f0, b0[1:]])
        k = ARM_INDEX[STATE["arm"]]
        self.coef_ = self.multi_coef_[:, k].copy()
        self.intercept_ = float(self.multi_int_[k])
        fitted = Z @ self.multi_coef_ + self.multi_int_
        self.multi_med_ = np.median(y[:, None] - fitted, axis=0)
        i10 = int(np.argmin(np.abs(GRID - BASE_ALPHA)))
        rec = {
            "smax": int(seasons.max()),
            "cut": str(ctx["cut"]),
            "n": n,
            "d": d,
            "shared_used": len(j),
            "cfb_n": cfb_n,
            "ratio": ratio,
            "lam_R": float(lam_k[0]),
            "lam_zero_opt": float(lam_k[1]),
            "train_mse_R": float(np.mean((y - fitted[:, 0]) ** 2)),
            "loso_mse_R": float(held[best[0], 0] / n),
            "train_mse_base": float(np.mean((y - fitted[:, 2]) ** 2)),
            "loso_mse_base": float(held[i10, 1] / n),
            "loso_mse_c10": float(held[i10, 0] / n),
            "loso_mse_zero_opt": float(held[best[1], 1] / n),
            "acc_in_R": h14.sign_acc(fitted[:, 0], y),
            "acc_in_base": h14.sign_acc(fitted[:, 2], y),
            "prior_norm": float(np.linalg.norm(prior)),
            "base_shared_norm": float(np.linalg.norm(self.multi_coef_[j, 2])),
            "prior_base_corr": float(np.corrcoef(prior[j], self.multi_coef_[j, 2])[0, 1])
            if len(j) > 2
            else float("nan"),
        }
        if STATE["arm"] == "R":
            rec["placebo_lam_median"] = float(np.median(lam_k[2:]))
            rec["placebo_top_share"] = float(np.mean(lam_k[2:] == GRID[-1]))
            rec["placebo_bottom_share"] = float(np.mean(lam_k[2:] == GRID[0]))
        LOG.append(rec)
        return self

    def predict(self, Z):
        return np.asarray(Z, dtype=float) @ self.coef_ + self.intercept_


def stash(pipe, X):
    reg = pipe.steps[-1][1]
    if not all(c in X.columns for c in NONMARKET):
        return
    keys = pd.util.hash_pandas_object(X[NONMARKET], index=False).to_numpy()
    info = MAPS["pred"].reindex(keys)
    keep = (info["gameday"] > pipe.cut_).to_numpy()
    if not keep.any():
        return
    t = X.loc[keep]
    for _name, step in pipe.steps[:-1]:
        t = step.transform(t)
    p = np.asarray(t, dtype=float) @ reg.multi_coef_ + reg.multi_int_
    STASH.append(
        (
            info["game_id"].to_numpy()[keep],
            np.full(int(keep.sum()), pipe.cut_.value),
            p,
            p + reg.multi_med_,
        )
    )


class ArmPipe(Pipeline):
    def fit(self, X, y=None, **kw):
        keys = pd.util.hash_pandas_object(X[list(X.columns)], index=False).to_numpy()
        looked = MAPS["fit"].reindex(keys)
        if looked["season"].isna().any():
            raise RuntimeError("season lookup failed")
        self.cut_ = looked["gameday"].max()
        STATE["ctx"] = {
            "seasons": looked["season"].to_numpy().astype(int),
            "cut": self.cut_,
            "X": X,
            "pipe": self,
        }
        return super().fit(X, y, **kw)

    def predict(self, X, **kw):
        out = super().predict(X, **kw)
        if STATE["arm"] == "R" and isinstance(X, pd.DataFrame):
            stash(self, X)
        return out


def make(model_name, random_state=42, **kw):
    est = ORIG(model_name, random_state, **kw)
    if model_name != "ridge":
        return est
    return ArmPipe(steps=[*est.steps[:-1], ("regressor", ArmRidge())])


def versus(run, ref, rng):
    j = run.join(ref, rsuffix="_b", how="inner")
    p = pd.DataFrame(
        {
            "block": j["season"].astype(int).astype(str) + "-" + j["week"].astype(int).astype(str),
            "season": j["season"].astype(int),
            "d_acc": 100.0 * (j["correct"].astype(float) - j["correct_b"].astype(float)),
            "d_ll": j["ll_b"] - j["ll"],
            "d_br": j["brier_b"] - j["brier"],
        }
    )
    flips = j["pick_home"].to_numpy(dtype=bool) != j["pick_home_b"].to_numpy(dtype=bool)
    won = j["correct"].astype(float).to_numpy() == 1.0
    mll, mbr = market_only(j)
    per = p.groupby("season")["d_acc"].mean()
    return {
        "n": len(j),
        "arm_wins": int(j["correct"].astype(float).sum()),
        "ref_wins": int(j["correct_b"].astype(float).sum()),
        "flips_win": int((flips & won).sum()),
        "flips_loss": int((flips & ~won).sum()),
        "arm_ll": float(j["ll"].mean()),
        "ref_ll": float(j["ll_b"].mean()),
        "market_ll": float(mll.mean()),
        "arm_br": float(j["brier"].mean()),
        "ref_br": float(j["brier_b"].mean()),
        "market_br": float(mbr.mean()),
        "accuracy": base.block_boot(p, "d_acc", rng),
        "log_loss": base.block_boot(p, "d_ll", rng),
        "brier": base.block_boot(p, "d_br", rng),
        "accuracy_season_boot": boot_season(p, "d_acc"),
        "per_season_d_acc": json.loads(per.to_json()),
    }


def summarize():
    lg = pd.DataFrame(LOG)
    by = lg.groupby("smax")[["lam_R", "lam_zero_opt", "shared_used"]].median()
    by["lam_R_min"] = lg.groupby("smax")["lam_R"].min()
    by["lam_R_max"] = lg.groupby("smax")["lam_R"].max()
    by["fits"] = lg.groupby("smax").size()
    num = lg.select_dtypes("number").mean()
    return {
        "fits": len(lg),
        "by_train_max_season": json.loads(by.to_json(orient="index")),
        "means": json.loads(num.to_json()),
        "lam_R_top_fits": int((lg["lam_R"] == GRID[-1]).sum()),
        "lam_R_bottom_fits": int((lg["lam_R"] == GRID[0]).sum()),
        "lam_zero_top_fits": int((lg["lam_zero_opt"] == GRID[-1]).sum()),
    }


def placebo_read(run, ref):
    if not STASH:
        return {"available": False}
    ids = np.concatenate([s[0] for s in STASH])
    cuts = np.concatenate([s[1] for s in STASH])
    raw = np.vstack([s[2] for s in STASH])
    shifted = np.vstack([s[3] for s in STASH])
    df = pd.DataFrame({"game_id": ids, "cut": cuts, "row": np.arange(len(ids))})
    df = df.sort_values("cut").drop_duplicates("game_id", keep="last").set_index("game_id")
    rows = df["row"].reindex(run.index)
    have = rows.notna().to_numpy()
    if not have.any():
        return {"available": False, "stashed": len(ids)}
    r = run.loc[have]
    idx = rows[have].astype(int).to_numpy()
    picked = r["pick_home"].to_numpy(dtype=bool)
    right = r["correct"].astype(float).to_numpy() == 1.0
    home = np.where(picked, right, ~right)
    ref_pick = ref["pick_home"].reindex(r.index).to_numpy(dtype=bool)
    rules = {}
    for name, p in (("raw", raw), ("median_shift", shifted)):
        picks = p[idx] > 0
        rules[name] = {
            "agree_R": float(np.mean(picks[:, 0] == picked)),
            "agree_base": float(np.mean(picks[:, 2] == ref_pick)),
            "wins": (picks == home[:, None]).sum(axis=0),
        }
    best = max(rules, key=lambda k: rules[k]["agree_R"])
    w = rules[best]["wins"]
    n = int(have.sum())
    out = {
        "available": True,
        "covered": n,
        "games": len(run),
        "rule": best,
        "agreement": {k: [v["agree_R"], v["agree_base"]] for k, v in rules.items()},
        "sign_wins": {
            "R": int(w[0]),
            "Z": int(w[1]),
            "base": int(w[2]),
            "C10": int(w[3]),
            "zero_opt": int(w[4]),
        },
        "production_R_wins_covered": int(right.sum()),
        "production_base_wins_covered": int(ref["correct"].reindex(r.index).astype(float).sum()),
        "placebo": h14.placebo_read(int(w[0]), w[PLACEBO_START:], n),
    }
    out["placebo"]["placebos_above_sign_base"] = int(np.sum(w[PLACEBO_START:] > w[2]))
    return out


def run_arm(tag, features):
    LOG.clear()
    STATE["arm"] = tag
    margin_mod.make_margin_estimator = make
    t0 = time.time()
    try:
        late = HARNESS["true_eval"](features, tuple(FULL), BASE_ALPHA)
    finally:
        margin_mod.make_margin_estimator = ORIG
    late = late.loc[late["season"].ge(FIRST)]
    late.to_parquet(OUT / f"h88_{tag}_run.parquet")
    pd.DataFrame(LOG).to_csv(OUT / f"h88_{tag}_fits.csv", index=False)
    print(
        tag,
        "wins",
        int(late["correct"].astype(float).sum()),
        len(late),
        "fits",
        len(LOG),
        "seconds",
        round(time.time() - t0, 1),
        flush=True,
    )
    return late


def show(tag, c):
    e = c["accuracy"]
    print(
        tag,
        c["arm_wins"],
        c["ref_wins"],
        c["n"],
        "flips",
        c["flips_win"],
        c["flips_loss"],
        "acc",
        round(e["effect"], 2),
        round(e["low"], 2),
        round(e["high"], 2),
        "P+",
        round(e["p_pos"], 3),
        "ll",
        round(c["arm_ll"], 5),
        round(c["ref_ll"], 5),
        round(c["market_ll"], 5),
        "llP+",
        round(c["log_loss"]["p_pos"], 3),
        "br",
        round(c["arm_br"], 5),
        round(c["ref_br"], 5),
        round(c["market_br"], 5),
        "brP+",
        round(c["brier"]["p_pos"], 3),
        flush=True,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ[clv.CACHE_DISABLED_ENV] = "1"
    features = pd.read_parquet(u3.FEATURES)
    gd = pd.to_datetime(features["gameday"]).to_numpy()
    kf = pd.util.hash_pandas_object(features[list(FULL)], index=False).to_numpy()
    fm = pd.DataFrame(
        {"season": features["season"].astype(int).to_numpy(), "gameday": gd}, index=kf
    )
    dup = fm.index.duplicated(keep=False)
    if int(fm.loc[dup, ["season"]].groupby(level=0).nunique().gt(1).any(axis=1).sum()):
        raise SystemExit("season key conflict")
    MAPS["fit"] = fm[~fm.index.duplicated(keep="first")]
    kp = pd.util.hash_pandas_object(features[NONMARKET], index=False).to_numpy()
    pm = pd.DataFrame(
        {"game_id": features["game_id"].astype(str).to_numpy(), "gameday": gd}, index=kp
    )
    MAPS["pred"] = pm[~pm.index.duplicated(keep=False)]
    cfb = pd.read_parquet(base.CFB_FEATURES)
    cfb["gameday"] = pd.to_datetime(cfb["gameday"])
    STATE["cfb"] = base.cfb_prior_rows(cfb)
    STATE["shared"] = [c for c in ps.shared_columns() if c in FULL]
    rng = np.random.default_rng(base.SEED)
    ident = np.arange(len(STATE["shared"]))
    perms = []
    while len(perms) < PERMS:
        p = rng.permutation(len(ident))
        if not np.array_equal(p, ident):
            perms.append(p)
    STATE["perms"] = perms
    stored = pd.read_parquet(BASE5)
    b = stored.loc[stored["season"].ge(FIRST)]
    have = (int(b["correct"].astype(float).sum()), len(b))
    print("stored base", have, "shared", len(STATE["shared"]), "market", list(MARKET), flush=True)
    if have != IDENTITY:
        raise SystemExit("stored base is not the identity record")
    report = {
        "shared": STATE["shared"],
        "market_columns": list(MARKET),
        "grid": [float(x) for x in GRID],
        "stored_base": list(have),
        "arms": {},
        "fits": {},
    }
    runs = {}
    for tag in ("ID", "R", "C10", "Z"):
        run = run_arm(tag, features)
        run = run.loc[run.index.isin(b.index)]
        runs[tag] = run
        if tag == "ID":
            got = (int(run["correct"].astype(float).sum()), len(run))
            ref = b.reindex(run.index)
            report["identity"] = {
                "rerun": list(got),
                "pick_agreement": float(
                    np.mean(
                        run["pick_home"].to_numpy(dtype=bool)
                        == ref["pick_home"].to_numpy(dtype=bool)
                    )
                ),
                "max_abs_p_diff": float(
                    np.max(np.abs(run["p"].to_numpy(float) - ref["p"].to_numpy(float)))
                ),
            }
            report["fits"]["ID"] = summarize()
            (OUT / "report_h88.json").write_text(json.dumps(report, indent=1, default=str))
            print("identity", json.dumps(report["identity"]), flush=True)
            if got != IDENTITY:
                raise SystemExit("identity rerun failed")
            continue
        report["arms"][tag] = versus(run, b, rng)
        report["fits"][tag] = summarize()
        show(tag, report["arms"][tag])
        print(tag, "fits", json.dumps(report["fits"][tag]["by_train_max_season"]), flush=True)
        print(tag, "means", json.dumps(report["fits"][tag]["means"]), flush=True)
        if tag == "R":
            report["placebo"] = placebo_read(run, b)
            print("placebo", json.dumps(report["placebo"], default=str), flush=True)
        (OUT / "report_h88.json").write_text(json.dumps(report, indent=1, default=str))
    report["arms"]["R_vs_Z"] = versus(runs["R"], runs["Z"], rng)
    show("R_vs_Z", report["arms"]["R_vs_Z"])
    (OUT / "report_h88.json").write_text(json.dumps(report, indent=1, default=str))
    try:
        t0 = time.time()
        late = HARNESS["true_eval"](features, MARKET, BASE_ALPHA)
        late = late.loc[late["season"].ge(FIRST)]
        late = late.loc[late.index.isin(b.index)]
        late.to_parquet(OUT / "h88_L_run.parquet")
        report["arms"]["L"] = versus(late, b, rng)
        show("L", report["arms"]["L"])
        print("L seconds", round(time.time() - t0, 1), flush=True)
    except Exception as exc:
        report["market_only_error"] = repr(exc)
        print("L failed", repr(exc), flush=True)
    (OUT / "report_h88.json").write_text(json.dumps(report, indent=1, default=str))
    print("H88_DONE", flush=True)


if __name__ == "__main__":
    main()
