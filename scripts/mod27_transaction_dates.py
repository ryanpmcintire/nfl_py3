import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_ats.pick_probability_fit as pf
from nfl_ats import transaction_flag_features as T

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "mod27_transaction_dates"
PICK_TABLE = ROOT / "artifacts" / "pick_probability" / "20261101T171318Z" / "per_game.parquet"
COLUMN = T.DEADLINE_INTEGRATION_DRAG_COLUMN
EPS = 1e-9
SAMPLES = 4000
SEED = 20261102
PREDICTION_TZ = "America/New_York"
PREDICTION_HOUR = 12


def article_dates():
    files = glob.glob(str(ROOT / "data/raw/pfr_transactions/*/sample_articles/*.json"))
    rows = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        pub = d.get("json_ld_date_published") or []
        rows.append({"url": d["url"], "published": pd.to_datetime(pub[0], utc=True) if pub else pd.NaT})
    return pd.DataFrame(rows).drop_duplicates("url")


def dated_index():
    tx = T.default_transactions_index()
    tx = tx.merge(article_dates(), on="url", how="left")
    tx["lastmod_utc"] = pd.to_datetime(tx["lastmod_contaminated"], utc=True)
    both = tx.dropna(subset=["published", "lastmod_utc"])
    stats = {
        "articles_with_published": int(tx["published"].notna().sum()),
        "lastmod_before_published": int((both["lastmod_utc"] < both["published"]).sum()),
        "lastmod_minus_published_hours_median": float(
            ((both["lastmod_utc"] - both["published"]).dt.total_seconds() / 3600).median()
        ),
    }
    tx["tx_ts"] = tx["published"].fillna(tx["lastmod_utc"])
    tx["date_source"] = np.where(tx["published"].notna(), "published", "lastmod")
    return tx, stats


def prediction_ts(gameday):
    gd = pd.to_datetime(gameday)
    tue = gd - pd.to_timedelta((gd.dt.weekday - 1) % 7, unit="D")
    return (tue + pd.Timedelta(hours=PREDICTION_HOUR)).dt.tz_localize(PREDICTION_TZ).dt.tz_convert("UTC")


def events(tx, sn, sch, dated):
    rows = T.confirmed_acquisition_transactions(tx)
    slugs = T.distinct_player_slugs(sn)
    reg = sch.loc[sch["game_type"].eq("REG")].copy()
    reg["season"] = reg["season"].astype(int)
    reg["week"] = reg["week"].astype(int)
    reg["pred_ts"] = prediction_ts(reg["gameday"])
    out = []
    for _, r in rows.iterrows():
        if pd.isna(r["url_year"]) or pd.isna(r["url_month"]):
            continue
        parsed = T._parse_acquisition(str(r["slug"]))
        if parsed is None:
            continue
        team, part = parsed
        player = T.find_player_in_segment(part, slugs)
        if player is None:
            continue
        season = int(r["url_year"])
        ts = r["tx_ts"]
        if dated:
            if pd.isna(ts):
                continue

            def ok(g, ts=ts):
                return g["pred_ts"] > ts
        else:
            end = T._month_end_timestamp(season, int(r["url_month"]))

            def ok(g, end=end):
                return pd.to_datetime(g["gameday"]) > end

        tg = reg.loc[
            reg["season"].eq(season) & (reg["home_team"].eq(team) | reg["away_team"].eq(team))
        ].sort_values("week")
        decision = tg.loc[ok(tg)]
        if decision.empty:
            continue
        before_week = int(decision["week"].min())
        sr = sn.loc[(sn["player"] == player) & (sn["season"] == season) & (sn["week"] < before_week)]
        prior = sr.loc[sr["team"] != team]
        if prior.empty:
            continue
        lw = int(prior["week"].max())
        giving = str(prior.loc[prior["week"] == lw, "team"].iloc[0])
        share = float(prior.loc[prior["team"] == giving, "snap_share"].mean())
        if share < T.HIGH_SNAP_SHARE_THRESHOLD:
            continue
        games = tg.loc[(tg["week"] > lw) & ok(tg)].head(T.DEADLINE_INTEGRATION_GAMES)
        for _, g in games.iterrows():
            out.append(
                {
                    "slug": r["slug"],
                    "player": player,
                    "team": team,
                    "season": season,
                    "week": int(g["week"]),
                    "game_id": g["game_id"],
                    "tx_ts": ts,
                    "date_source": r["date_source"],
                    "pred_ts": g["pred_ts"],
                }
            )
    return pd.DataFrame(out)


def to_flag(sch, ev):
    q = ev[["season", "week", "team"]].drop_duplicates()
    return T._attach_qualifying_sides(sch, q, COLUMN)[["game_id", COLUMN]]


def met(p, yy):
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "acc": float(np.mean((p >= 0.5) == (yy == 1))),
        "ll": float(-np.mean(yy * np.log(p) + (1 - yy) * np.log1p(-p))),
        "brier": float(np.mean((p - yy) ** 2)),
    }


def rec(p, yy):
    c = (p >= 0.5) == (yy == 1)
    return f"{int(c.sum())}-{int((~c).sum())}"


def strs(df):
    return df.assign(tx_ts=df["tx_ts"].astype(str), pred_ts=df["pred_ts"].astype(str))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sch = T.default_schedule()
    sn = T.default_snap_counts()
    tx, date_stats = dated_index()
    leaky_ev = events(tx, sn, sch, dated=False)
    fix_ev = events(tx, sn, sch, dated=True)
    if len(fix_ev) and not (fix_ev["tx_ts"] < fix_ev["pred_ts"]).all():
        raise AssertionError("repaired flag uses a transaction not strictly before the prediction timestamp")
    leaky_ev["violates"] = ~(leaky_ev["tx_ts"] < leaky_ev["pred_ts"])
    v = leaky_ev.loc[leaky_ev["violates"]]
    leak = {
        "date_stats": date_stats,
        "leaky_team_game_flags": int(len(leaky_ev)),
        "leaky_transactions": int(leaky_ev["slug"].nunique()),
        "leaky_flags_violating_day_level": int(len(v)),
        "leaky_games_violating": int(v["game_id"].nunique()),
        "leaky_transactions_violating": int(v["slug"].nunique()),
        "violators_by_source": v["date_source"].value_counts().to_dict(),
        "violators_by_season": v.groupby("season").size().to_dict(),
        "leaky_by_source": leaky_ev["date_source"].value_counts().to_dict(),
        "repaired_team_game_flags": int(len(fix_ev)),
        "repaired_transactions": int(fix_ev["slug"].nunique()),
        "repaired_by_source": fix_ev["date_source"].value_counts().to_dict(),
    }
    strs(fix_ev).to_csv(OUT / "repaired_events.csv", index=False)
    strs(leaky_ev).to_csv(OUT / "leaky_events.csv", index=False)

    H = pd.read_parquet(PICK_TABLE).reset_index(drop=True)
    for tag, ev in (("leaky", leaky_ev), ("fixed", fix_ev)):
        f = to_flag(sch, ev).rename(columns={COLUMN: "f_" + tag})
        H = H.merge(f, on="game_id", how="left")
        H["f_" + tag] = H["f_" + tag].fillna(0.0)
    y = H["home_covered"].to_numpy(float)
    H["season_week"] = H["season"] * 100 + H["week"]
    base = pf.FIT_FEATURES

    def loso(feats):
        pf.FIT_FEATURES = feats
        p = pd.Series(np.nan, index=H.index)
        co = {}
        for h in sorted(H["season"].unique()):
            tr, te = H[H["season"] != h], H[H["season"] == h]
            m, s = pf._standardisers(tr)
            b = pf._fit_logit(pf._design(tr, m, s), tr["home_covered"].astype(float).to_numpy(), pf.FIT_RIDGE)
            p.loc[te.index] = pf._predict(te, b, m, s)
            co[int(h)] = pf._natural_coefficients(b, m, s)
        m, s = pf._standardisers(H)
        b = pf._fit_logit(pf._design(H, m, s), y, pf.FIT_RIDGE)
        return p.to_numpy(), co, pf._predict(H, b, m, s)

    def boot(mask, pa, pb):
        yy = y[mask]
        ca = ((pa[mask] >= 0.5) == (yy == 1)).astype(float)
        cb = ((pb[mask] >= 0.5) == (yy == 1)).astype(float)
        keys = H.loc[mask, "season_week"].to_numpy()
        u = np.unique(keys)
        idx = [np.where(keys == k)[0] for k in u]
        rng = np.random.default_rng(SEED)
        res = []
        for _ in range(SAMPLES):
            ii = np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])
            res.append((cb[ii] - ca[ii]).mean())
        res = np.array(res)
        return {
            "diff_pts": float((cb - ca).mean() * 100),
            "lo": float(np.percentile(res, 2.5) * 100),
            "hi": float(np.percentile(res, 97.5) * 100),
            "prob_pos": float((res > 0).mean()),
            "n": int(mask.sum()),
        }

    p4, c4, i4 = loso(base)
    allm = np.ones(len(H), bool)
    R = {"leak": leak, "base": {"rec": rec(p4, y), **met(p4, y)}, "in_sample_base": met(i4, y)}
    P = {}
    for tag in ("leaky", "fixed"):
        H["_f"] = H["f_" + tag]
        p5, c5, i5 = loso(base + ("_f",))
        P[tag] = p5
        fl = (H["_f"] != 0).to_numpy()
        dis = (p4 >= 0.5) != (p5 >= 0.5)
        R[tag] = {
            "flagged_games": int(fl.sum()),
            "flagged_by_season": {int(k): int(n) for k, n in H.loc[fl].groupby("season").size().items()},
            "flagged_only_rec_base": rec(p4[fl], y[fl]),
            "flagged_only_rec_five": rec(p5[fl], y[fl]),
            "all_rec": rec(p5, y),
            "oos": met(p5, y),
            "in_sample": met(i5, y),
            "gap_acc_in_minus_oos": met(i5, y)["acc"] - met(p5, y)["acc"],
            "coef_per_fold": {k: v["_f"] for k, v in c5.items()},
            "changed_sides": int(dis.sum()),
            "changed_base_rec": rec(p4[dis], y[dis]),
            "changed_five_rec": rec(p5[dis], y[dis]),
            "boot_all_vs_base": boot(allm, p4, p5),
            "boot_2022_25_vs_base": boot((H["season"] >= 2022).to_numpy(), p4, p5),
        }
    R["fixed_vs_leaky_all"] = boot(allm, P["leaky"], P["fixed"])
    dl = (P["leaky"] >= 0.5) != (P["fixed"] >= 0.5)
    R["fixed_vs_leaky_changed"] = {
        "n": int(dl.sum()),
        "leaky_rec": rec(P["leaky"][dl], y[dl]),
        "fixed_rec": rec(P["fixed"][dl], y[dl]),
    }
    R["looks"] = 5
    json.dump(R, open(OUT / "results.json", "w"), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main()
