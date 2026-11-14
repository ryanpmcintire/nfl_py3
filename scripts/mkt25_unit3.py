import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mkt25_opener_revision import BOOT, SEASONS, build, ols, week_groups
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

REPLICATION_SEASONS = range(2013, 2020)
BRIDGE_SEASONS = range(2020, 2022)
EXTENDED = "artifacts/extended_fit_population/20260923T205910Z/population.parquet"
SCORED = "artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet"
SBR = "data/processed/sbr_odds.parquet"
FEATURES = "data/processed/game_features.parquet"
OUT = Path("tests/scratch/mkt25_unit3_out.txt")
BATCH = Path("artifacts/mkt25/record_batch_unit3.json")
PERMUTATIONS = 5000
MOD36_GAMES = 1503
MOD36_LOGIT_MATCHED_NULL = 0.17
MOD36_LOGIT_P90 = 0.25
STRUCTURE = ["cm", "cm_pos", "cm_neg", "move", "inj_news", "inj_level"]
NEW_2020_LOOKS = ("outcome_coef", "follow_all", "follow_vs_served", "alone_coef", "alone_dLL")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    with OUT.open("a") as f:
        f.write(s + "\n")


def true_opener():
    return pd.read_parquet(sorted(glob.glob("artifacts/opener_evaluation/*/per_game.parquet"))[-1])


def proxy_lines():
    sc = pd.read_parquet(SCORED)
    sbr = pd.read_parquet(SBR).dropna(subset=["game_id"]).drop_duplicates("game_id")
    sbr = sbr.set_index("game_id")
    sc["sbr_open"] = sc.game_id.map(sbr.open_home_spread)
    sc["sbr_close"] = sc.game_id.map(sbr.close_home_spread)
    return sc


def replication_frame():
    sc = proxy_lines()
    ext = pd.read_parquet(EXTENDED)
    ext = ext[ext.opener_source != "tue_open"].set_index("game_id")
    sc = sc[sc.game_id.isin(ext.index)]
    logit_at_open = sc.game_id.map(ext.model_logit).to_numpy(float)
    return pd.DataFrame(
        {
            "game_id": sc.game_id.to_numpy(),
            "season": sc.season.to_numpy(),
            "week": sc.week.to_numpy(),
            "tue_open_home_spread": sc.proxy_open_home_spread.to_numpy(float),
            "close_home_spread": sc.sbr_close.to_numpy(float),
            "home_cover_probability_at_open": expit(logit_at_open),
            "margin_vs_open": sc.margin_vs_open_proxy.to_numpy(float),
            "home_covered": sc.game_id.map(ext.home_covered).to_numpy(float),
        }
    )


def bridge_frame(o):
    sc = proxy_lines().set_index("game_id")
    b = o[o.season.isin(BRIDGE_SEASONS)].copy()
    b["tue_open_home_spread"] = b.game_id.map(sc.proxy_open_home_spread)
    return b.dropna(subset=["tue_open_home_spread"])


def gap(a, b):
    both = pd.concat([a, b], axis=1).dropna()
    d = both.iloc[:, 0] - both.iloc[:, 1]
    r = np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]
    return (
        f"n {len(d)} mean|diff| {d.abs().mean():.3f} median|diff| {d.abs().median():.2f} "
        f"exact {100 * (d == 0).mean():.1f}% within1 {100 * (d.abs() <= 1).mean():.1f}% "
        f"mean diff {d.mean():+.3f} r {r:.3f}"
    )


def stage_proxy():
    OUT.write_text("")
    o = true_opener()
    sc = proxy_lines()
    g = pd.read_parquet(FEATURES, columns=["game_id", "spread_line"]).set_index("game_id")
    sc["nfl_close"] = sc.game_id.map(g.spread_line)
    say("== stage proxy (lines only, no result joined) ==")
    say("scored proxy_open vs sbr_odds open:", gap(sc.proxy_open_home_spread, sc.sbr_open))
    ov = sc.merge(o[["game_id", "tue_open_home_spread", "close_home_spread"]], on="game_id")
    ov = ov[ov.season.isin(BRIDGE_SEASONS)]
    pairs = {
        "proxy vs true opener": ("proxy_open_home_spread", "tue_open_home_spread"),
        "capture close vs true opener": ("close_home_spread", "tue_open_home_spread"),
        "proxy vs capture close": ("proxy_open_home_spread", "close_home_spread"),
        "proxy vs SBR close": ("proxy_open_home_spread", "sbr_close"),
        "proxy vs nflverse close": ("proxy_open_home_spread", "nfl_close"),
        "SBR close vs capture close": ("sbr_close", "close_home_spread"),
    }
    say(f"overlap {min(BRIDGE_SEASONS)}-{max(BRIDGE_SEASONS)} (games with both lines)")
    for label, (a, b) in pairs.items():
        say(f"  {label}: {gap(ov[a], ov[b])}")
    for s, x in ov.groupby("season"):
        say(f"  {s} proxy vs true opener: {gap(x.proxy_open_home_spread, x.tue_open_home_spread)}")
    to_open = (ov.proxy_open_home_spread - ov.tue_open_home_spread).abs()
    to_close = (ov.proxy_open_home_spread - ov.close_home_spread).abs()
    say(
        f"  proxy nearer true opener {100 * (to_open < to_close).mean():.1f}% "
        f"nearer close {100 * (to_open > to_close).mean():.1f}% "
        f"tie {100 * (to_open == to_close).mean():.1f}%"
    )
    path = (ov.close_home_spread - ov.tue_open_home_spread).to_numpy(float)
    off = (ov.proxy_open_home_spread - ov.tue_open_home_spread).to_numpy(float)
    b = ols(path, off)
    say(
        f"  position on the open-to-close path (0 opener, 1 close): slope {b[1]:.3f} "
        f"intercept {b[0]:+.3f}; residual sd {np.std(off - b[0] - b[1] * path):.3f}; "
        f"sd of true move {path.std():.3f}"
    )
    rep = sc[sc.season.isin(REPLICATION_SEASONS)]
    say(f"replication era {min(REPLICATION_SEASONS)}-{max(REPLICATION_SEASONS)}")
    say(f"  proxy vs SBR close: {gap(rep.proxy_open_home_spread, rep.sbr_close)}")
    say(f"  proxy vs nflverse close: {gap(rep.proxy_open_home_spread, rep.nfl_close)}")
    say(f"  SBR close vs nflverse close: {gap(rep.sbr_close, rep.nfl_close)}")
    say(f"  open_ambiguous rows {int(rep.open_ambiguous.sum())} of {len(rep)}")
    for s, x in rep.groupby("season"):
        say(f"  {s} proxy vs SBR close: {gap(x.proxy_open_home_spread, x.sbr_close)}")
    ext = pd.read_parquet(EXTENDED)
    pop = ext[ext.season.isin(REPLICATION_SEASONS)]
    say("  population rows", pop.groupby(["season", "opener_source"]).size().to_dict())
    both = ext.merge(o[["game_id", "home_cover_probability_at_open"]], on="game_id")
    diff = expit(both.model_logit) - both.home_cover_probability_at_open
    say(
        f"archived model_logit vs current probability at the true opener, 2020-25: n {len(both)} "
        f"mean|diff| {diff.abs().mean():.4f} max {diff.abs().max():.4f}"
    )


def interval(bs):
    return (
        float(np.percentile(bs, 2.5)),
        float(np.percentile(bs, 97.5)),
        float((bs > 0).mean()),
    )


def season_groups(d):
    s = d.season.to_numpy()
    return [np.flatnonzero(s == k) for k in np.unique(s)]


def block_boot(d, fn, rng, groups):
    out = np.empty(BOOT)
    for k in range(BOOT):
        pick = rng.choice(len(groups), len(groups))
        out[k] = fn(d.iloc[np.concatenate([groups[i] for i in pick])])
    return out


def mean_boot(v, rng, groups):
    out = np.empty(BOOT)
    for k in range(BOOT):
        pick = rng.choice(len(groups), len(groups))
        out[k] = np.concatenate([v[groups[i]] for i in pick]).mean()
    return out


def strat_boot(dp, fn, rng):
    era = dp.era.to_numpy()
    season = dp.season.to_numpy()
    strata = [[np.flatnonzero(season == s) for s in np.unique(season[era == e])] for e in (0, 1)]
    out = np.empty(BOOT)
    for k in range(BOOT):
        idx = []
        for gs in strata:
            idx += [gs[i] for i in rng.choice(len(gs), len(gs))]
        out[k] = fn(dp.iloc[np.concatenate(idx)])
    return out


def slope(c, target="R"):
    def fn(x):
        return ols(x[[c]].to_numpy(float), x[target].to_numpy(float))[1]

    return fn


def fit(x, cols):
    return LogisticRegression(C=1e6, max_iter=1000).fit(x[cols].to_numpy(float), x.y.to_numpy())


def coef_fn(cols, target):
    k = cols.index(target)

    def fn(x):
        return float(fit(x, cols).coef_[0][k])

    return fn


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def loso(d, cols, seasons):
    p = np.full(len(d), 0.5)
    coef = {}
    season = d.season.to_numpy()
    for s in seasons:
        tr, te = d[season != s], season == s
        if not cols:
            p[te] = tr.y.mean()
            continue
        m = fit(tr, cols)
        p[te] = m.predict_proba(d.loc[te, cols].to_numpy(float))[:, 1]
        coef[int(s)] = np.round(m.coef_[0], 3).tolist()
    return np.clip(p, 1e-6, 1 - 1e-6), coef


def cell(cells, name, effect, units, ci, evidence, d):
    cells.append(
        {
            "name": f"mkt25u3_{name}",
            "effect": round(float(effect), 5),
            "effect_units": units,
            "interval_low": round(ci[0], 5),
            "interval_high": round(ci[1], 5),
            "probability_positive": round(ci[2], 3),
            "classification_evidence": evidence,
            "season_start": int(d.season.min()),
            "season_end": int(d.season.max()),
            "sample_games": len(d),
        }
    )


def permutation_p(obs, draws):
    one = (1 + int((draws >= obs).sum())) / (1 + len(draws))
    centre = draws.mean()
    two = (1 + int((np.abs(draws - centre) >= abs(obs - centre)).sum())) / (1 + len(draws))
    return one, two


def record_text(win, n):
    return f"{int(win)}-{int(n - win)}"


def era_report(d, seasons, tag, line, rng, cells):
    d = d.reset_index(drop=True).copy()
    y = d.y.to_numpy()
    n_seasons = d.season.nunique()
    say(f"== {tag} ({line}): n {len(d)} seasons {d.season.value_counts().sort_index().to_dict()}")
    say(
        f"R sd {d.R.std():.3f} mean {d.R.mean():+.3f}; hfa mean {d.hfa.mean():.3f}; "
        f"fit resid sd {d.fit_sd.mean():.3f}; corr(R,R_close) "
        f"{np.corrcoef(d.R, d.R_close)[0, 1]:.3f}; home cover rate {y.mean():.4f}"
    )
    wg, sg = week_groups(d), season_groups(d)
    res = {"n": len(d), "sd": float(d.R.std())}
    for c in STRUCTURE:
        fn = slope(c)
        b = fn(d)
        wk = interval(block_boot(d, fn, rng, wg))
        sn = interval(block_boot(d, fn, rng, sg))
        per = [fn(d[d.season == s]) for s in sorted(d.season.unique())]
        pos = int(sum(v > 0 for v in per))
        say(
            f"slope R~{c}: {b:.4f} week-block CI[{wk[0]:.4f},{wk[1]:.4f}] P+ {wk[2]:.3f}; "
            f"season-block CI[{sn[0]:.4f},{sn[1]:.4f}] P+ {sn[2]:.3f}; "
            f"seasons positive {pos}/{n_seasons}; by season {np.round(per, 3).tolist()}"
        )
        res["slope_" + c] = (b, sn)
        cell(
            cells,
            f"slope_{c}_{tag}",
            b,
            "slope_difference",
            sn,
            f"{line}; n {len(d)}; slope of opener revision on {c}; season-block bootstrap; "
            f"week-block CI [{wk[0]:.4f},{wk[1]:.4f}]; seasons positive {pos}/{n_seasons}",
            d,
        )
    multi = ["cm", "move", "inj_news"]
    d["Rhat_cm"] = 0.0
    d["Rhat_full"] = 0.0
    for s in seasons:
        tr, te = d[d.season != s], d.season == s
        b = ols(tr[["cm"]].to_numpy(float), tr.R.to_numpy(float))
        d.loc[te, "Rhat_cm"] = b[0] + b[1] * d.loc[te, "cm"]
        b = ols(tr[multi].to_numpy(float), tr.R.to_numpy(float))
        d.loc[te, "Rhat_full"] = b[0] + d.loc[te, multi].to_numpy(float) @ b[1:]
    d["e_cm"] = d.R - d.Rhat_cm
    d["e_full"] = d.R - d.Rhat_full
    terms = {"R_raw": "R", "e_cm": "e_cm", "e_full": "e_full"}
    for k, c in terms.items():
        d["t_" + k] = d[c] / d[c].std()
    p_int, _ = loso(d, [], seasons)
    p_srv, _ = loso(d, ["z"], seasons)
    for k in terms:
        cols = ["z", "t_" + k]
        p_t, coef = loso(d, cols, seasons)
        flip = (p_t > 0.5) != (p_srv > 0.5)
        fw = int(((p_t > 0.5) == (y == 1))[flip].sum())
        dv = ll(y, p_srv) - ll(y, p_t)
        wk = interval(mean_boot(dv, rng, wg))
        sn = interval(mean_boot(dv, rng, sg))
        sb = pd.Series(dv).groupby(d.season.to_numpy()).mean()
        ins = float(ll(y, fit(d, cols).predict_proba(d[cols].to_numpy(float))[:, 1]).mean())
        say(
            f"TERM served+{k}: DECISIVE flips {int(flip.sum())} new side "
            f"{record_text(fw, flip.sum())}; dLL vs served {dv.mean():.5f} "
            f"week-block CI[{wk[0]:.5f},{wk[1]:.5f}] P+ {wk[2]:.3f}; season-block "
            f"CI[{sn[0]:.5f},{sn[1]:.5f}] P+ {sn[2]:.3f}; seasons+ {(sb > 0).sum()}/{n_seasons}; "
            f"LOSO LL {ll(y, p_t).mean():.5f} in-sample {ins:.5f} "
            f"gap {ll(y, p_t).mean() - ins:.5f}"
        )
        say("   per-fold coef [z, term]:", coef)
        cell(
            cells,
            f"term_{k}_dLL_{tag}",
            dv.mean(),
            "log_loss_improvement",
            wk,
            f"{line}; LOSO n {len(d)}; flips {int(flip.sum())} new side "
            f"{record_text(fw, flip.sum())}; seasons positive {(sb > 0).sum()}/{n_seasons}; "
            f"week-block bootstrap; season-block CI [{sn[0]:.5f},{sn[1]:.5f}] P+ {sn[2]:.3f}",
            d,
        )
    fn = coef_fn(["z", "R"], "R")
    b = fn(d)
    sn = interval(block_boot(d, fn, rng, sg))
    per = [fn(d[d.season == s]) for s in sorted(d.season.unique())]
    pos = int(sum(v > 0 for v in per))
    p_full = fit(d, ["z"]).predict_proba(d[["z"]].to_numpy(float))[:, 1]
    resid = y - p_full
    r_val = d.R.to_numpy(float)
    follow_home = r_val > 0
    win = follow_home == (y == 1)
    draws_score = np.empty(PERMUTATIONS)
    draws_win = np.empty(PERMUTATIONS)
    for k in range(PERMUTATIONS):
        idx = np.arange(len(d))
        for grp in sg:
            idx[grp] = rng.permutation(grp)
        rp = r_val[idx]
        draws_score[k] = float(resid @ rp)
        draws_win[k] = float(((rp > 0) == (y == 1)).sum())
    ps = permutation_p(float(resid @ r_val), draws_score)
    say(
        f"OUTCOME logit coef of R per point beside z (full fit): {b:.5f} season-block "
        f"CI[{sn[0]:.5f},{sn[1]:.5f}] P+ {sn[2]:.3f}; per SD {b * d.R.std():.4f}; "
        f"seasons positive {pos}/{n_seasons}; by season {np.round(per, 4).tolist()}; "
        f"within-season permutation p one-sided {ps[0]:.4f} two-sided {ps[1]:.4f}"
    )
    res["coef"] = (b, sn)
    cell(
        cells,
        f"outcome_coef_{tag}",
        b,
        "logit_coefficient",
        sn,
        f"{line}; n {len(d)}; logit coefficient per point of opener revision beside the "
        f"at-open model logit; per SD {b * d.R.std():.4f}; seasons positive {pos}/{n_seasons}; "
        f"within-season permutation p one-sided {ps[0]:.4f}; season-block bootstrap",
        d,
    )
    by = pd.DataFrame({"s": d.season.to_numpy(), "w": win}).groupby("s").w.agg(["sum", "size"])
    above = int((by["sum"] * 2 > by["size"]).sum())
    rate = 100 * win.mean() - 50
    sn = interval(100 * mean_boot(win.astype(float), rng, sg) - 50)
    pw = permutation_p(float(win.sum()), draws_win)
    say(
        f"FOLLOW rule (side the opener revised toward) vs the opener: "
        f"{record_text(win.sum(), len(win))} ({rate:+.2f} pts vs 50) season-block "
        f"CI[{sn[0]:.2f},{sn[1]:.2f}] P+ {sn[2]:.3f}; seasons above half {above}/{n_seasons}; "
        f"by season {[record_text(a, b_) for a, b_ in zip(by['sum'], by['size'], strict=True)]}; "
        f"within-season permutation p one-sided {pw[0]:.4f} two-sided {pw[1]:.4f}"
    )
    res["follow"] = (win.sum(), len(win), rate, sn, above, pw)
    cell(
        cells,
        f"follow_all_{tag}",
        rate,
        "rate_pp",
        sn,
        f"{line}; follow-the-revision side wins {record_text(win.sum(), len(win))} vs the "
        f"opener; win rate minus 50; seasons above half {above}/{n_seasons}; within-season "
        f"permutation p one-sided {pw[0]:.4f}; season-block bootstrap",
        d,
    )
    dis = follow_home != (d.z.to_numpy() > 0)
    wd = win[dis]
    byd = pd.DataFrame({"s": d.season.to_numpy()[dis], "w": wd}).groupby("s").w.agg(["sum", "size"])
    above_d = int((byd["sum"] * 2 > byd["size"]).sum())
    rate_d = 100 * wd.mean() - 50

    def dis_rate(x):
        m = (x.R.to_numpy() > 0) != (x.z.to_numpy() > 0)
        return 100 * ((x.R.to_numpy() > 0) == (x.y.to_numpy() == 1))[m].mean() - 50

    sn = interval(block_boot(d, dis_rate, rng, sg))
    say(
        f"FOLLOW rule where it disagrees with the model side: {record_text(wd.sum(), len(wd))} "
        f"({rate_d:+.2f} pts vs 50) season-block CI[{sn[0]:.2f},{sn[1]:.2f}] P+ {sn[2]:.3f}; "
        f"seasons above half {above_d}/{len(byd)}"
    )
    cell(
        cells,
        f"follow_vs_served_{tag}",
        rate_d,
        "rate_pp",
        sn,
        f"{line}; games where the follow side differs from the model side at the opener; "
        f"follow wins {record_text(wd.sum(), len(wd))}; seasons above half {above_d}/{len(byd)}; "
        f"season-block bootstrap",
        d,
    )
    dv0 = ll(y, p_int) - ll(y, p_srv)
    sn = interval(mean_boot(dv0, rng, sg))
    say(
        f"CONTROL model logit vs intercept LOSO dLL {dv0.mean():.5f} season-block "
        f"CI[{sn[0]:.5f},{sn[1]:.5f}] P+ {sn[2]:.3f}"
    )
    cell(
        cells,
        f"control_served_vs_intercept_{tag}",
        dv0.mean(),
        "log_loss_improvement",
        sn,
        f"{line}; LOSO n {len(d)}; at-open model logit against an intercept-only pick; "
        f"season-block bootstrap",
        d,
    )
    fn = coef_fn(["R"], "R")
    b0 = fn(d)
    sn = interval(block_boot(d, fn, rng, sg))
    p_alone, _ = loso(d, ["t_R_raw"], seasons)
    dva = ll(y, p_int) - ll(y, p_alone)
    sna = interval(mean_boot(dva, rng, sg))
    say(
        f"CONTROL intercept-only base: logit coef of R per point {b0:.5f} season-block "
        f"CI[{sn[0]:.5f},{sn[1]:.5f}] P+ {sn[2]:.3f}; R alone vs intercept LOSO dLL "
        f"{dva.mean():.5f} CI[{sna[0]:.5f},{sna[1]:.5f}] P+ {sna[2]:.3f}"
    )
    cell(
        cells,
        f"alone_coef_{tag}",
        b0,
        "logit_coefficient",
        sn,
        f"{line}; n {len(d)}; logit coefficient per point of opener revision on an "
        f"intercept-only base (no model logit); season-block bootstrap",
        d,
    )
    cell(
        cells,
        f"alone_dLL_{tag}",
        dva.mean(),
        "log_loss_improvement",
        sna,
        f"{line}; LOSO n {len(d)}; opener revision alone against an intercept-only pick; "
        f"season-block bootstrap",
        d,
    )
    for name, size in (("matched null", MOD36_LOGIT_MATCHED_NULL), ("P+ .90", MOD36_LOGIT_P90)):
        det = size * np.sqrt(MOD36_GAMES / len(d))
        say(
            f"detectable logit per SD at n {len(d)} ({name}, MOD-36 {size} at {MOD36_GAMES} "
            f"games scaled by sqrt({MOD36_GAMES}/n)): {det:.3f}; measured per SD "
            f"{b * d.R.std():.4f} = {abs(b * d.R.std()) / det:.2f} of it"
        )
    return d, res


def frozen(d20, d13, rng, cells, line):
    sd20 = d20.R.std()
    a, b = d20.assign(t=d20.R / sd20), d13.assign(t=d13.R / sd20)
    m_s, m_t = fit(a, ["z"]), fit(a, ["z", "t"])
    y = b.y.to_numpy()
    p_s = m_s.predict_proba(b[["z"]].to_numpy(float))[:, 1]
    p_t = m_t.predict_proba(b[["z", "t"]].to_numpy(float))[:, 1]
    dv = ll(y, p_s) - ll(y, p_t)
    sn = interval(mean_boot(dv, rng, season_groups(b)))
    wk = interval(mean_boot(dv, rng, week_groups(b)))
    flip = (p_t > 0.5) != (p_s > 0.5)
    fw = int(((p_t > 0.5) == (y == 1))[flip].sum())
    sb = pd.Series(dv).groupby(b.season.to_numpy()).mean()
    say(
        f"FROZEN 2020-25 fit applied unchanged (coef z {m_t.coef_[0][0]:.3f}, R per 2020-25 SD "
        f"{m_t.coef_[0][1]:.4f}): dLL vs frozen model-only {dv.mean():.5f} season-block "
        f"CI[{sn[0]:.5f},{sn[1]:.5f}] P+ {sn[2]:.3f}; week-block CI[{wk[0]:.5f},{wk[1]:.5f}] "
        f"P+ {wk[2]:.3f}; DECISIVE flips {int(flip.sum())} new side "
        f"{record_text(fw, flip.sum())}; seasons+ {(sb > 0).sum()}/{len(sb)}"
    )
    cell(
        cells,
        "frozen_2020_25_term_dLL_2013_19",
        dv.mean(),
        "log_loss_improvement",
        sn,
        f"{line}; coefficients fitted on 2020-25 true openers and applied unchanged; flips "
        f"{int(flip.sum())} new side {record_text(fw, flip.sum())}; seasons positive "
        f"{(sb > 0).sum()}/{len(sb)}; season-block bootstrap",
        b,
    )


def bridge(o, rng, cells):
    dt = build(o, BRIDGE_SEASONS)
    dp = build(bridge_frame(o), BRIDGE_SEASONS)
    m = dt[["game_id", "season", "week", "cm", "R"]].merge(
        dp[["game_id", "R"]], on="game_id", suffixes=("_true", "_proxy")
    )
    wg = week_groups(m)
    ft, fp = slope("cm", "R_true"), slope("cm", "R_proxy")
    ct = interval(block_boot(m, ft, rng, wg))
    cp = interval(block_boot(m, fp, rng, wg))
    cd = interval(block_boot(m, lambda x: fp(x) - ft(x), rng, wg))
    say(
        f"== BRIDGE 2020-21 same {len(m)} games: slope R~cm true opener {ft(m):.4f} "
        f"CI[{ct[0]:.4f},{ct[1]:.4f}]; SBR proxy {fp(m):.4f} CI[{cp[0]:.4f},{cp[1]:.4f}]; "
        f"proxy minus true {fp(m) - ft(m):+.4f} CI[{cd[0]:.4f},{cd[1]:.4f}] P+ {cd[2]:.3f}; "
        f"R sd true {m.R_true.std():.3f} proxy {m.R_proxy.std():.3f} "
        f"corr {np.corrcoef(m.R_true, m.R_proxy)[0, 1]:.3f} (week-block)"
    )
    cell(
        cells,
        "bridge_slope_cm_proxy_minus_true_2020_21",
        fp(m) - ft(m),
        "slope_difference",
        cd,
        f"same {len(m)} games 2020-21; slope of opener revision on prior cover margin with the "
        f"SBR proxy open ({fp(m):.4f}) minus with the true Tuesday opener ({ft(m):.4f}); "
        f"week-block bootstrap",
        m,
    )


def pooled(d13, d20, r13, r20, rng, cells):
    dp = pd.concat([d13.assign(era=0), d20.assign(era=1)], ignore_index=True)
    line = "2013-19 SBR proxy open pooled with 2020-25 true Tuesday opener"
    sc, cf = slope("cm"), coef_fn(["z", "R"], "R")
    looks = {
        "pooled_slope_cm": (sc, "slope_difference", "pooled slope of revision on prior cover"),
        "era_diff_slope_cm": (
            lambda x: sc(x[x.era == 0]) - sc(x[x.era == 1]),
            "slope_difference",
            "2013-19 minus 2020-25 slope of revision on prior cover",
        ),
        "pooled_outcome_coef": (cf, "logit_coefficient", "pooled logit coef per point beside z"),
        "era_diff_outcome_coef": (
            lambda x: cf(x[x.era == 0]) - cf(x[x.era == 1]),
            "slope_difference",
            "2013-19 minus 2020-25 logit coef per point",
        ),
    }
    say(f"== POOLED 2013-25: n {len(dp)} (season blocks resampled inside each era)")
    for name, (fn, units, text) in looks.items():
        b = fn(dp)
        ci = interval(strat_boot(dp, fn, rng))
        say(f"{name}: {b:.5f} CI[{ci[0]:.5f},{ci[1]:.5f}] P+ {ci[2]:.3f}")
        cell(cells, f"{name}_2013_25", b, units, ci, f"{line}; {text}; n {len(dp)}", dp)
    sd = dp.R.std()
    det = MOD36_LOGIT_MATCHED_NULL * np.sqrt(MOD36_GAMES / len(dp))
    say(
        f"pooled per SD {cf(dp) * sd:.4f} (R sd {sd:.3f}); detectable at n {len(dp)} "
        f"(matched null) {det:.3f}; eras: slope cm {r13['slope_cm'][0]:.4f} vs "
        f"{r20['slope_cm'][0]:.4f}; coef {r13['coef'][0]:.5f} vs {r20['coef'][0]:.5f}"
    )


def stage_replicate():
    rng = np.random.default_rng(2503)
    cells = []
    o = true_opener()
    rf = replication_frame()
    agree = ((rf.margin_vs_open > 0) == (rf.home_covered == 1)).mean()
    say("== stage replicate ==")
    say(f"proxy-era frame rows {len(rf)}; cover flag agrees with margin vs proxy open {agree:.4f}")
    d13 = build(rf.drop(columns="home_covered"), REPLICATION_SEASONS)
    d20 = build(o, SEASONS)
    line13 = "2013-19 SBR proxy open (not a timestamped opener)"
    d13, r13 = era_report(d13, REPLICATION_SEASONS, "2013_19", line13, rng, cells)
    d20, r20 = era_report(d20, SEASONS, "2020_25", "2020-25 true Tuesday opener", rng, cells)
    frozen(d20, d13, rng, cells, line13)
    bridge(o, rng, cells)
    pooled(d13, d20, r13, r20, rng, cells)
    keep = [
        c
        for c in cells
        if not c["name"].endswith("_2020_25")
        or any(c["name"].startswith(f"mkt25u3_{k}") for k in NEW_2020_LOOKS)
    ]
    say("total looks recorded as cells", len(keep), "of", len(cells), "computed")
    batch = {
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": min(REPLICATION_SEASONS),
        "season_end": max(REPLICATION_SEASONS),
        "family": "mkt25_opener_revision_unit3",
        "category": "market",
        "source": "scripts/mkt25_unit3.py; tests/scratch/mkt25_unit3_out.txt",
        "sample_games": len(d13),
        "plain_summary": (
            "Whether the opening line's habit of chasing a team's last result against the "
            "spread, and any edge from siding with that revision, also shows in 2013-19 on an "
            "older and rougher record of opening lines."
        ),
        "description": (
            "Unit 3 replication of the opener-revision slopes and terms on 2013-19 with the "
            "SBR Open column as a proxy opener, direction fixed from 2020-25, plus a 2020-21 "
            "bridge on the same games and the pooled 2013-25 figures"
        ),
        "cells": keep,
    }
    BATCH.write_text(json.dumps(batch, indent=1))


def artifact_frame(o):
    sc = proxy_lines().set_index("game_id")
    b = o[o.season.isin(BRIDGE_SEASONS)].copy()
    proxy = b.game_id.map(sc.proxy_open_home_spread)
    b["margin_vs_open"] = b.margin_vs_open + b.tue_open_home_spread - proxy
    b["tue_open_home_spread"] = proxy
    return b.dropna(subset=["tue_open_home_spread"])


def logit_coef(cols, target, ycol):
    k = cols.index(target)

    def fn(x):
        m = LogisticRegression(C=1e6, max_iter=1000)
        return float(m.fit(x[cols].to_numpy(float), x[ycol].to_numpy()).coef_[0][k])

    return fn


def stage_artifact():
    rng = np.random.default_rng(2504)
    o = true_opener()
    dt = build(o, BRIDGE_SEASONS)
    dp = build(artifact_frame(o), BRIDGE_SEASONS)
    m = dt[["game_id", "season", "week", "R", "y"]].merge(
        dp[["game_id", "R", "y"]], on="game_id", suffixes=("_true", "_proxy")
    )
    m["gap"] = m.R_proxy - m.R_true
    wg = week_groups(m)
    looks = {
        "true_graded_coef": (["R_true"], "R_true", "y_true", "true opener, graded at it"),
        "proxy_graded_coef": (["R_proxy"], "R_proxy", "y_proxy", "SBR proxy, graded at it"),
        "gap_coef": (["R_true", "gap"], "gap", "y_proxy", "proxy minus true opener part"),
        "true_part_coef": (["R_true", "gap"], "R_true", "y_proxy", "true revision part"),
    }
    say(
        f"== POST HOC line-error check, 2020-21 same {len(m)} games (declared after the "
        f"2013-19 sign was seen; intercept-only base; week-block); gap sd {m.gap.std():.3f}"
    )
    out = {}
    for name, (cols, target, ycol, text) in looks.items():
        fn = logit_coef(cols, target, ycol)
        out[name] = (fn(m), interval(block_boot(m, fn, rng, wg)), text)
        b, ci = out[name][0], out[name][1]
        say(f"{name} ({text}): {b:.5f} CI[{ci[0]:.5f},{ci[1]:.5f}] P+ {ci[2]:.3f}")
    d13 = build(replication_frame().drop(columns="home_covered"), REPLICATION_SEASONS)
    c13 = coef_fn(["R"], "R")(d13)
    need = float(np.sqrt(max(d13.R.var() * c13 / out["gap_coef"][0], 0.0)))
    say(
        f"2013-19 coef {c13:.5f} at R sd {d13.R.std():.3f}: a pure line error with the measured "
        f"gap coef would need sd {need:.2f} pts to produce all of it"
    )
    batch = json.loads(BATCH.read_text())
    names = {c["name"] for c in batch["cells"]}
    for name in ("proxy_graded_coef", "gap_coef"):
        b, ci, text = out[name]
        cells = []
        cell(
            cells,
            f"posthoc_{name}_2020_21",
            b,
            "logit_coefficient",
            ci,
            f"post hoc, declared after the 2013-19 sign was seen; same {len(m)} games 2020-21; "
            f"logit coefficient per point, {text}; intercept-only base; week-block bootstrap",
            m,
        )
        if cells[0]["name"] not in names:
            batch["cells"] += cells
    BATCH.write_text(json.dumps(batch, indent=1))
    say("total looks recorded as cells", len(batch["cells"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["proxy", "replicate", "artifact"], required=True)
    args = parser.parse_args()
    {"proxy": stage_proxy, "replicate": stage_replicate, "artifact": stage_artifact}[args.stage]()
