import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sky01_news_availability as base

ROOT = base.ROOT
OUT = base.OUT
SNAPS = ROOT / "data" / "players" / "raw" / "20261101T141534Z" / "snap_counts.parquet"
ROSTERS = ROOT / "data" / "players" / "raw" / "20261101T141534Z" / "weekly_rosters.parquet"
BOOT = base.BOOT
NS_H = 3.6e12


def series_arrays(series):
    d = {}
    for gid, g in series.sort_values("observed_at_utc").groupby("nflverse_game_id"):
        d[gid] = (
            g.observed_at_utc.dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .values.astype("datetime64[ns]")
            .astype("int64"),
            g.home_spread_line.values.astype(float),
        )
    return d


def evaluate(arr, t0, kick, home):
    if arr is None:
        return None
    ts, v = arr
    i = np.searchsorted(ts, t0, side="right") - 1
    if i < 0:
        return None
    ref = v[i]
    j = np.searchsorted(ts, kick, side="right")
    if j - 1 <= i:
        return (0.0, np.nan, 0.0)
    seg_t = ts[i + 1 : j]
    seg_v = v[i + 1 : j]
    adv = seg_v > ref if home else seg_v < ref
    lead = (seg_t[adv.argmax()] - t0) / NS_H if adv.any() else np.nan
    move = (seg_v[-1] - ref) if home else (ref - seg_v[-1])
    return (float(adv.any()), lead, float(move))


def ns(ts):
    return (
        pd.Timestamp(ts)
        .tz_convert("UTC")
        .tz_localize(None)
        .to_datetime64()
        .astype("datetime64[ns]")
        .astype("int64")
    )


def drift_control(lt, sched, arrs):
    ev = (
        lt.groupby(["season", "week", "game_id", "team"])
        .agg(t0=("beat_first_out", "min"), kickoff=("kickoff", "first"))
        .reset_index()
    )
    ev["home"] = ev.team == ev.game_id.str.split("_").str[3]
    flagged = set(ev.game_id)
    sk = sched[["season", "week", "game_id", "kickoff"]]
    ctrl = sk[~sk.game_id.isin(flagged)]
    by_week = {k: g for k, g in ctrl.groupby(["season", "week"])}
    rows = []
    empty = pd.DataFrame(columns=["game_id", "kickoff"])
    for r in ev.itertuples():
        if r.t0 >= r.kickoff:
            continue
        tt = evaluate(arrs.get(r.game_id), ns(r.t0), ns(r.kickoff), r.home)
        if tt is None:
            continue
        tto = ns(r.kickoff) - ns(r.t0)
        cs = []
        for c in by_week.get((r.season, r.week), empty).itertuples():
            ck = ns(c.kickoff)
            e = evaluate(arrs.get(c.game_id), ck - tto, ck, r.home)
            if e is not None:
                cs.append(e)
        if not cs:
            continue
        c = np.array(cs)
        rows.append(
            {
                "season": r.season,
                "week": r.week,
                "game_id": r.game_id,
                "team": r.team,
                "tto_h": tto / NS_H,
                "t_adv": tt[0],
                "t_lead": tt[1],
                "t_move": tt[2],
                "c_adv": c[:, 0].mean(),
                "c_lead": np.nanmedian(c[:, 1]) if np.isfinite(c[:, 1]).any() else np.nan,
                "c_move": c[:, 2].mean(),
                "n_ctrl": len(c),
            }
        )
    return pd.DataFrame(rows)


def stat(x):
    return {
        "share_adv_treated": x.t_adv.mean(),
        "share_adv_control": x.c_adv.mean(),
        "net_share": (x.t_adv - x.c_adv).mean(),
        "median_lead_treated_h": np.nanmedian(x.t_lead) if x.t_lead.notna().any() else np.nan,
        "median_lead_control_h": np.nanmedian(x.c_lead) if x.c_lead.notna().any() else np.nan,
        "net_move_to_close": (x.t_move - x.c_move).mean(),
    }


def drift_summary(d, rng):
    pt = stat(d)
    pt["net_median_lead_h"] = pt["median_lead_treated_h"] - pt["median_lead_control_h"]
    keys = ["net_share", "net_median_lead_h", "net_move_to_close"]
    bs = {k: np.empty(BOOT) for k in keys}
    n = len(d)
    for b in range(BOOT):
        s = stat(d.iloc[rng.integers(0, n, n)])
        bs["net_share"][b] = s["net_share"]
        bs["net_median_lead_h"][b] = s["median_lead_treated_h"] - s["median_lead_control_h"]
        bs["net_move_to_close"][b] = s["net_move_to_close"]
    res = {"n_events": int(n), **{k: float(v) for k, v in pt.items()}}
    for k in keys:
        a = bs[k][np.isfinite(bs[k])]
        res[k + "_ci"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        res[k + "_prob_positive"] = float((a > 0).mean())
    return res


def prior_share():
    s = pd.read_parquet(
        SNAPS,
        columns=["season", "week", "game_type", "pfr_player_id", "offense_pct", "defense_pct"],
    )
    s = s[s.game_type == "REG"].copy()
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    r = pd.read_parquet(ROSTERS, columns=["gsis_id", "pfr_id"]).dropna().drop_duplicates("pfr_id")
    s = s.merge(r, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    s["season"] = s.season.astype("int64")
    s["key"] = (s.season * 100 + s.week).astype("int64")
    s = s.sort_values(["gsis_id", "season", "key"])
    g = s.groupby(["gsis_id", "season"])
    s["cs"] = g.share.cumsum() - s.share
    s["cn"] = g.cumcount()
    s["in_season_prior"] = np.where(s.cn > 0, s.cs / s.cn.replace(0, np.nan), np.nan)
    season_mean = s.groupby(["gsis_id", "season"]).share.mean().rename("sm").reset_index()
    season_mean["season"] += 1
    return s[["gsis_id", "season", "key", "in_season_prior"]], season_mean


def weight_for(q, ps, sm):
    q = q.copy()
    q["gsis_id"] = q.gsis_id.astype(object)
    sm = sm.assign(gsis_id=sm.gsis_id.astype(object))
    q["season"] = q.season.astype("int64")
    q["key"] = (q.season * 100 + q.week).astype("int64")
    q = q.sort_values("key")
    ps2 = (
        ps.assign(gsis_id=ps.gsis_id.astype(object))
        .dropna(subset=["in_season_prior"])
        .sort_values("key")
        .rename(columns={"in_season_prior": "w_in"})
    )
    m = pd.merge_asof(q, ps2, on="key", by=["gsis_id", "season"], allow_exact_matches=False)
    m = m.merge(sm, on=["gsis_id", "season"], how="left")
    m["w"] = m.w_in.fillna(m.sm).fillna(0.0)
    m["w_source"] = np.where(
        m.w_in.notna(), "in_season", np.where(m.sm.notna(), "prior_season", "none")
    )
    return m


def weighted_feature(p, g, ps, sm):
    q = p[p.cls.isin(base.OUT_CLASSES) & (p.created_at < p.opener_ts)].drop_duplicates(
        ["game_id", "team", "gsis_id"]
    )
    q = weight_for(q[["game_id", "team", "gsis_id", "season", "week"]], ps, sm)
    n = q.groupby(["game_id", "team"]).w.sum().rename("n").reset_index()
    h = (
        g[["game_id", "home_team"]]
        .merge(n, left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left")["n"]
        .fillna(0)
        .values
    )
    a = (
        g[["game_id", "away_team"]]
        .merge(n, left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left")["n"]
        .fillna(0)
        .values
    )
    return h - a, q.w_source.value_counts().to_dict(), float((q.w > 0).mean())


def score(sub, x, seed):
    y = sub.home_covered.astype(float).values
    pr = sub.out_of_season_home_probability.clip(1e-6, 1 - 1e-6).values
    off = base.logit(pr)
    elig = [2024, 2025]
    pooled = np.zeros(len(sub))
    ins = np.zeros(len(sub))
    folds = {}
    for s in elig:
        te = (sub.season == s).values
        tr = np.isin(sub.season.values, [e for e in elig if e != s])
        b = base.fit_b(off[tr], x[tr], y[tr])
        folds[int(s)] = float(b)
        pooled[te] = 1 / (1 + np.exp(-(off[te] + b * x[te])))
    mask = sub.season.isin(elig).values
    b_all = base.fit_b(off[mask], x[mask], y[mask])
    ins[mask] = 1 / (1 + np.exp(-(off[mask] + b_all * x[mask])))
    b0 = pr[mask]
    yy = y[mask]
    rng = np.random.default_rng(seed)
    res = {
        "fold_coefficients": folds,
        "pooled_coefficient": float(b_all),
        "x_nonzero_games": int((x[mask] != 0).sum()),
        "n": int(mask.sum()),
    }
    for label, arr in (("oos", pooled[mask]), ("ins", ins[mask])):
        d_brier = (b0 - yy) ** 2 - (arr - yy) ** 2
        d_ll = -(yy * np.log(b0) + (1 - yy) * np.log(1 - b0)) + (
            yy * np.log(arr) + (1 - yy) * np.log(1 - arr)
        )
        d_acc = (
            ((arr > 0.5) == (yy == 1)).astype(float) - ((b0 > 0.5) == (yy == 1)).astype(float)
        ) * 100
        dec = (b0 > 0.5) != (arr > 0.5)
        res[label] = {
            "brier_gain": base.boot(d_brier, rng),
            "logloss_gain": base.boot(d_ll, rng),
            "accuracy_gain_points": base.boot(d_acc, rng),
            "decisive_games": int(dec.sum()),
            "decisive_new_correct": int(((arr > 0.5) == (yy == 1))[dec].sum()),
            "decisive_base_correct": int(((b0 > 0.5) == (yy == 1))[dec].sum()),
        }
    res["gap_brier_gain_ins_minus_oos"] = res["ins"]["brier_gain"][0] - res["oos"]["brier_gain"][0]
    return res


def inactive_join(lt):
    ins = pd.read_parquet(OUT / "inactive_first_seen.parquet")
    r = pd.read_parquet(ROSTERS, columns=["season", "team", "gsis_id", "full_name"])
    r = r[r.season == 2026].copy()
    r["team"] = r.team.map(base.alias)
    r["name"] = r.full_name.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    r = r.drop_duplicates(["team", "gsis_id"])
    m = lt[lt.season == 2026].merge(
        r[["team", "name", "gsis_id"]], on=["team", "gsis_id"], how="left"
    )
    m = m.merge(ins, on=["season", "week", "team", "name"], how="inner")
    m["inactive_minus_beat_h"] = (m.inactive_first - m.beat_first_out).dt.total_seconds() / 3600
    m["inactive_minus_official_h"] = (
        m.inactive_first - m.official_ts_eff
    ).dt.total_seconds() / 3600
    m["inactive_minus_market_after_beat_h"] = (
        m.inactive_first - m.market_adverse_after_beat
    ).dt.total_seconds() / 3600
    out = {"n_joined": len(m), "n_beat_before_inactive": int((m.inactive_minus_beat_h > 0).sum())}
    for c in (
        "inactive_minus_beat_h",
        "inactive_minus_official_h",
        "inactive_minus_market_after_beat_h",
    ):
        x = m[c].dropna()
        out[c] = {
            "n": len(x),
            "median": float(x.median()) if len(x) else None,
            "q25": float(x.quantile(0.25)) if len(x) else None,
            "q75": float(x.quantile(0.75)) if len(x) else None,
        }
    m.to_parquet(OUT / "inactive_joined_2026.parquet")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base.SEED)
    t, p, s = base.build_table()
    lt = pd.read_parquet(OUT / "lead_times.parquet")
    arrs = series_arrays(base.spread_series())
    d = drift_control(lt, s, arrs)
    d.to_parquet(OUT / "drift_control_events.parquet")
    res = {"drift_all": drift_summary(d, rng)}
    for k, g in d.groupby("season"):
        res["drift_" + str(int(k))] = drift_summary(g.reset_index(drop=True), rng)
    res["inactive_join_2026"] = inactive_join(lt)
    g = pd.read_parquet(base.PERGAME)
    sub = g[g.margin_vs_open != 0].copy().reset_index(drop=True)
    ps, sm = prior_share()
    xw, src, cov = weighted_feature(p, sub, ps, sm)
    res["weight_source_counts"] = {k: int(v) for k, v in src.items()}
    res["weight_positive_share"] = cov
    res["weighted_term"] = score(sub, xw, base.SEED + 1)
    res["count_term_reproduced"] = score(sub, base.feature(p, sub), base.SEED + 2)
    json.dump(res, open(OUT / "drift_weighted.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
