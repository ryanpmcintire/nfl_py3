import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_year as cy
import forward_ledger as FL
import mod25_wf_forward as mwf
import outside_model_gap as omg
import pooled_stack as ps
import xlg14_u6_nfl_confirmation as u6
from mod27_fenced_repairs_replay import FOUR, Fit

from nfl_ats.features import _kickoff_utc
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    STATION_MAP_RELATIVE_PATH,
    fetch_tuesday_noon_forecast_temps_fail_open,
)
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import games_for_forecast_fetch
from nfl_ats.pick_probability import FLAG_SUM_COLUMN, signed_composition_flags
from nfl_ats.pick_probability_fit import _arrest_incidents, _protection_back_side
from nfl_ats.published_picks import game_deadlines
from nfl_ats.snapshots import latest_snapshot, load_snapshot
from nfl_ats.splash_lines import load_splash_capture, splash_decision_lines

ROOT = ps.ROOT
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "pooled_stack_forward"
FROZEN = OUT / "frozen.json"
SEASON = 2026
REPLAY_FIRST, REPLAY_LAST = 1, 10
FIRST_WEEK, LAST_WEEK = 11, 18
STAMP = "%Y%m%dT%H%M%SZ"
FOUR_SIGNALS = ("flag_coach", "flag_cold_visitor", "last_opponent_strength", "contract_year_gap")
SCORE_SLOT = ("tue", "09:15")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def label_of(names):
    return lambda k: names[k] if k >= 0 else f"{names[-1 - k]}_missing"


def setup(full):
    pop = ps.population()
    sig, _meta = ps.signals(pop)
    season = pop["season"].to_numpy(int)
    decided = ~pop["push"].to_numpy(bool)
    y = np.where(decided, pop["home_covered"].astype(float).to_numpy(), 0.0)
    ylab = pop["y"].to_numpy(float)
    seasons = sorted(set(season.tolist()))
    cache, off = {}, {}
    for s in seasons:
        off[(s,)] = ps.offsets(pop, season, decided, {s}, seasons, cache)
        if full:
            for t in seasons:
                if t != s:
                    off[(s, t)] = ps.offsets(
                        pop, season, decided, {s, t}, [u for u in seasons if u != s], cache
                    )
    qw = np.full(len(pop), np.nan)
    for s in seasons:
        qw[season == s] = expit(off[(s,)][season == s])
    raw = sig.to_numpy(float)
    gaps = [bool(np.isnan(raw[:, j]).any()) for j in range(raw.shape[1])]
    ctx = {
        "season": season,
        "decided": decided,
        "y": y,
        "ylab": ylab,
        "gaps": gaps,
        "gaps_all": gaps,
        "seasons": seasons,
        "off": off,
        "raw": raw,
        "raw_all": raw,
        "qw": qw,
    }
    ps.CTX.update(ctx)
    return pop, sig, ctx


def moments(raw, gaps):
    out = []
    for j in range(raw.shape[1]):
        x = raw[:, j]
        obs = ~np.isnan(x)
        m = {"mean": float(x[obs].mean()), "sd": float(x[obs].std())}
        if gaps[j]:
            ind = (~obs).astype(float)
            m["missing_mean"] = float(ind.mean())
            m["missing_sd"] = float(ind.std())
        out.append(m)
    return out


def design(raw, mom, neutral=None):
    cols = []
    for j, m in enumerate(mom):
        x = raw[:, j]
        obs = ~np.isnan(x)
        cols.append(np.where(obs, (x - m["mean"]) / m["sd"], 0.0) if m["sd"] > 0 else 0.0 * x)
        if "missing_mean" in m:
            ind = (~obs).astype(float)
            z = (ind - m["missing_mean"]) / m["missing_sd"] if m["missing_sd"] > 0 else 0.0 * ind
            cols.append(0.0 * z if neutral is not None and neutral[j] else z)
    return np.nan_to_num(np.column_stack(cols))


def asof_totals(sch, games):
    import hindsight_audit_team as hat

    table = pd.read_parquet(hat.v4.TABLE)
    table["game_id"] = table["game_id"].astype(str)
    cols = [c for c in table.columns if c in sch.columns]
    new = sch[sch["game_id"].isin(games["game_id"])][cols]
    totals, _ = hat.tuesday_quotes(pd.concat([table, new], ignore_index=True))
    return totals.reindex(games["game_id"]).to_numpy(float)


def cmd_freeze(a):
    if FROZEN.exists():
        raise SystemExit(f"already frozen: {FROZEN}")
    pop, sig, c = setup(False)
    season, decided, y, ylab = c["season"], c["decided"], c["y"], c["ylab"]
    raw, gaps, seasons = c["raw"], c["gaps"], c["seasons"]
    X, names = ps.standardise(raw, gaps, np.ones(len(y), bool))
    mom = moments(raw, gaps)
    if not np.allclose(design(raw, mom), X):
        raise SystemExit("frozen standardisation does not reproduce the fit design")
    off = ps.logit(c["qw"])
    lams = ps.lam_path(X[decided], y[decided], off[decided])
    scores = np.zeros((len(ps.L1), len(lams) + 1))
    for t in seasons:
        inner = season != t
        Xi, _ = ps.standardise(raw, gaps, inner)
        oi = c["off"][(t,)]
        scores += ps.path_scores(Xi, inner & decided, season == t, y, ylab, oi, oi, lams)
    ai, li = ps.choose(scores, lams)
    b = ps.fit_at(X[decided], y[decided], off[decided], ps.L1[ai], lams, li)
    p_in = expit(off + X @ b)
    lab = label_of(list(sig.columns))
    coef = {lab(k): float(v) for k, v in zip(names, b, strict=True)}
    res = json.loads((ps.OUT / "results.json").read_text())
    want = res["in_sample"]
    ll_in = float(ps.ll(p_in[decided], y[decided]).mean())
    served = Fit(FOUR, pop[decided])
    w, *_ = np.linalg.lstsq(
        pop[list(ps.FLAGS)].to_numpy(float), pop[FLAG_SUM_COLUMN].to_numpy(float), rcond=None
    )
    body = {
        "family": ps.FAMILY,
        "lane": ps.LANE,
        "rule": (
            "elastic-net logistic on standardised as-of signals with offset logit(QW); l1 ratio "
            "and lambda chosen by single-level LOSO over 2020-25; one fit on all 2020-25 decided"
        ),
        "signals": list(sig.columns),
        "columns": [lab(k) for k in names],
        "moments": mom,
        "coefficients": coef,
        "signs": {k: int(np.sign(v)) for k, v in coef.items()},
        "nonzero": [k for k, v in coef.items() if v != 0.0],
        "l1_ratio": ps.L1[ai],
        "lambda_index": int(li),
        "lambda": float(lams[li]) if li < len(lams) else None,
        "lambda_max": float(lams[0]),
        "games": len(y),
        "decided": int(decided.sum()),
        "check": {
            "results_in_sample_choice": want["choice"],
            "matches_choice": want["choice"] == {"l1_ratio": ps.L1[ai], "lambda_index": int(li)},
            "log_loss_in_sample": ll_in,
            "results_log_loss_in_sample": want["log_loss_decided_in_sample"],
        },
        "served_stage": {
            "flag_sum_coef_per_unit": served.natural()[FLAG_SUM_COLUMN],
            "flag_sum_weights_by_flag": {
                f: round(float(v), 6) for f, v in zip(ps.FLAGS, w, strict=False)
            },
            "convention": "each flag is +1 when it favours home, -1 when it favours away",
        },
        "offset_2026": (
            "logit of the last published card home probability at or before the deadline"
        ),
        "signals_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(sig, index=False).to_numpy().tobytes()
        ).hexdigest(),
        "created_at_utc": datetime.now(UTC).strftime(STAMP),
    }
    body["body_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(body, indent=1))
    print(json.dumps({k: body[k] for k in body if k != "moments"}, indent=1))


def arm(cols):
    c = ps.CTX
    c["gaps"] = [c["gaps_all"][j] for j in cols]
    p, _folds = ps.nested(c["raw_all"][:, cols])
    return p.tolist()


def cmd_ablate(a):
    _pop, sig, c = setup(True)
    names = list(sig.columns)
    arms = {"full": list(range(len(names)))}
    for j, n in enumerate(names):
        arms[f"without_{n}"] = [k for k in range(len(names)) if k != j]
    arms["four_only"] = [names.index(n) for n in FOUR_SIGNALS]
    with ProcessPoolExecutor(
        max_workers=a.workers, initializer=ps.init_worker, initargs=(c,)
    ) as ex:
        ps_ = dict(zip(arms, ex.map(arm, arms.values()), strict=True))
    y, ylab, d, qw = c["y"], c["ylab"], c["decided"], c["qw"]
    full = np.array(ps_["full"])
    stored = pd.read_parquet(ps.OUT / "per_game.parquet")["stack"].to_numpy(float)
    out = {"full_matches_stored": bool(np.allclose(full, stored)), "arms": {}}
    for k, p in ps_.items():
        p = np.array(p)
        acc_q, ll_q = ps.gains(p, qw, y, ylab, d)
        acc_f, ll_f = ps.gains(p, full, y, ylab, d)
        out["arms"][k] = {
            "record": ps.record(p[d], y[d]),
            "accuracy_points_v_qw": acc_q,
            "log_loss_gain_v_qw": ll_q,
            "accuracy_points_v_full": acc_f,
            "log_loss_gain_v_full": ll_f,
        }
    out["qw_record"] = ps.record(qw[d], y[d])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps(out, indent=1))
    rows = sorted(out["arms"].items(), key=lambda kv: kv[1]["accuracy_points_v_full"])
    for k, v in rows:
        print(
            f"{k:<34} {v['record']:>8} acc_v_qw {v['accuracy_points_v_qw']:+.2f} "
            f"ll_v_qw {v['log_loss_gain_v_qw']:+.5f} acc_v_full {v['accuracy_points_v_full']:+.2f}"
        )
    print("full_matches_stored", out["full_matches_stored"])


def schedules():
    sch, _ = load_snapshot(latest_snapshot(DATA / "raw"))
    sch["game_id"] = sch["game_id"].astype(str)
    return sch


def season_games(sch, weeks):
    g = sch[(sch["season"] == SEASON) & (sch["game_type"] == "REG") & sch["week"].isin(weeks)]
    g = g.copy().reset_index(drop=True)
    g["kickoff"] = _kickoff_utc(g)
    dl = game_deadlines(g[["game_id"]].assign(kickoff=g["kickoff"]))
    g["pick_deadline_utc"] = g["game_id"].map(dl)
    g["cutoff_utc"] = g[["kickoff", "pick_deadline_utc"]].min(axis=1)
    return g


def team_rows(sch, season, upto):
    s = sch[(sch["season"] == season) & (sch["game_type"] == "REG") & (sch["week"] <= upto)]
    m = (s["home_score"] - s["away_score"]).fillna(0.0).to_numpy(float)
    base = {"game_id": s["game_id"].to_numpy(), "season": season, "week": s["week"].to_numpy()}
    home = pd.DataFrame(
        {
            **base,
            "team": s["home_team"].to_numpy(),
            "opponent": s["away_team"].to_numpy(),
            "margin": m,
        }
    )
    away = pd.DataFrame(
        {
            **base,
            "team": s["away_team"].to_numpy(),
            "opponent": s["home_team"].to_numpy(),
            "margin": -m,
        }
    )
    return pd.concat([home, away], ignore_index=True)


def u6_values(sch, season, upto):
    return u6.to_date(team_rows(sch, season, upto))


def cy_values(games, season):
    snaps = cy.load_snaps()
    snaps = snaps[snaps["season"] == season]
    if snaps.empty:
        return None, 0
    tv = cy.team_values(snaps, cy.flags(snaps, cy.walk_years(), cy.player_map()))
    rows = []
    for t in ("home", "away"):
        for gid, team, wk in zip(games["game_id"], games[f"{t}_team"], games["week"], strict=True):
            past = tv[(tv["team"] == team) & (tv["week"] < wk)]
            n = float(past["snaps"].sum())
            rows.append(
                {
                    "game_id": gid,
                    "team": team,
                    "prior_snaps": n,
                    cy.NAME: float(past["walk_snaps"].sum()) / n if n > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows), int(snaps["week"].max())


def live_signals(games, sch, names):
    s = pd.DataFrame(np.nan, index=games.index, columns=names)
    info = {}
    fc = fetch_tuesday_noon_forecast_temps_fail_open(
        games_for_forecast_fetch(games, sch), ROOT / "registry" / STATION_MAP_RELATIVE_PATH
    )
    info["forecast_fetch_status"] = (
        fc["fetch_status"].value_counts().to_dict() if "fetch_status" in fc else {}
    )
    f = signed_composition_flags(
        games,
        sch,
        incidents=_arrest_incidents(DATA),
        forecasts_tuesday_noon=fc,
        protection_back_side=_protection_back_side(sch, DATA),
    )
    f = f.drop_duplicates("game_id").set_index("game_id").reindex(games["game_id"])
    for c in ps.FLAGS:
        if c in s:
            s[c] = f[c].to_numpy(float)
    v = u6_values(sch, SEASON, int(games["week"].max()))
    for n in ("margin_sos_gap", "last_opponent_strength"):
        if n in s:
            s[n] = ps.two_sided(v, games, n, "prior_games")
    cv, snap_week = cy_values(games, SEASON)
    info["snap_counts_last_week"] = snap_week
    if cv is not None and "contract_year_gap" in s:
        s["contract_year_gap"] = ps.two_sided(cv, games, cy.NAME, "prior_snaps")
    wt = pd.read_parquet(ps.WT).drop_duplicates("game_id").set_index("game_id")
    for n in ("wt_drift", "sb_drift"):
        if n in s:
            s[n] = wt[n].reindex(games["game_id"]).to_numpy(float)
    if "nfelo_gap" in s:
        _, nf, _ = omg.terms()
        op = mwf.tuesday_open(games)
        line = games["game_id"].map(lambda i: op.get(i, (np.nan,))[0]).to_numpy(float)
        s["nfelo_gap"] = (
            nf.set_index("game_id")["nfelo_margin"].reindex(games["game_id"]).to_numpy(float) - line
        )
    if "model_w1" in s:
        s["model_w1"] = np.where(games["week"].eq(1), np.nan, 0.0)
    try:
        s["asof_total"] = asof_totals(sch, games)
        info["asof_total"] = "tuesday capture"
    except Exception as e:
        info["asof_total"] = f"failed: {type(e).__name__}: {e}"[:300]
    built = {
        "asof_total",
        *ps.FLAGS,
        "margin_sos_gap",
        "last_opponent_strength",
        "contract_year_gap",
        "wt_drift",
        "sb_drift",
        "nfelo_gap",
        "model_w1",
    }
    info["no_live_builder"] = [n for n in names if n not in built]
    info["missing_counts"] = {n: int(s[n].isna().sum()) for n in names if s[n].isna().any()}
    return s, info


def stack_prob(fz, s, base, unbuilt=()):
    neutral = [n in unbuilt for n in fz["signals"]]
    X = design(s[fz["signals"]].to_numpy(float), fz["moments"], neutral)
    b = np.array([fz["coefficients"][c] for c in fz["columns"]])
    return expit(ps.logit(base) + X @ b), X @ b, X * b


def pool_lines(weeks):
    out = {}
    for w in weeks:
        cap = load_splash_capture(DATA, SEASON, int(w))
        if cap is not None:
            out.update(splash_decision_lines(cap))
    return out


def grade(t, pc, pst):
    push = t["result"].eq(t["pool_line"]).to_numpy()
    y = (t["result"] > t["pool_line"]).astype(float).to_numpy()
    d = ~push
    ylab = np.where(push, ps.PUSH_LABEL, y)
    ch = ((pst >= 0.5) != (pc >= 0.5)) & d
    acc_game = 100 * (((pst >= 0.5) == (y == 1)) & d).astype(float) - 100 * (
        ((pc >= 0.5) == (y == 1)) & d
    ).astype(float)
    ll_game = ps.ll(pc, ylab) - ps.ll(pst, ylab)
    rng = np.random.default_rng(ps.SEED)
    wk = t["week"].to_numpy(int)
    return {
        "games_graded": int(d.sum()),
        "pushes": int(push.sum()),
        "card_record": ps.record(pc[d], y[d]),
        "stack_record": ps.record(pst[d], y[d]),
        "accuracy_points": float(acc_game[d].mean()),
        "log_loss_gain": float(ll_game.mean()),
        "log_loss_decided": {
            "card": float(ps.ll(pc[d], y[d]).mean()),
            "stack": float(ps.ll(pst[d], y[d]).mean()),
        },
        "brier_decided": {
            "card": float(((pc[d] - y[d]) ** 2).mean()),
            "stack": float(((pst[d] - y[d]) ** 2).mean()),
        },
        "accuracy_week_block": ps.block_boot(wk, acc_game, d.astype(float), rng),
        "log_loss_week_block": ps.block_boot(wk, ll_game, np.ones(len(t)), rng),
        "changed_picks": int(ch.sum()),
        "changed_record_stack": ps.record(pst[ch], y[ch]) if ch.any() else "0-0",
        "changed_record_card": ps.record(pc[ch], y[ch]) if ch.any() else "0-0",
        "changed_home_side": int(((pst >= 0.5) & ch).sum()),
        "per_week_stack_card": {
            int(w): [
                ps.record(pst[d & (wk == w)], y[d & (wk == w)]),
                ps.record(pc[d & (wk == w)], y[d & (wk == w)]),
            ]
            for w in np.unique(wk)
        },
    }


def cmd_replay(a):
    fz = json.loads(FROZEN.read_text())
    sch = schedules()
    weeks = list(range(REPLAY_FIRST, REPLAY_LAST + 1))
    g = season_games(sch, weeks)
    s, info = live_signals(g, sch, fz["signals"])
    card = FL._played_card().set_index("game_id")
    g["card_p"] = g["game_id"].map(card["served_p"])
    g["pool_line"] = g["game_id"].map(pool_lines(weeks))
    g["result"] = pd.to_numeric(g["result"], errors="coerce")
    base = g["card_p"].fillna(0.5).to_numpy(float)
    keep = (g["card_p"].notna() & g["pool_line"].notna() & g["result"].notna()).to_numpy()
    arms = {}
    for name, unbuilt in (("amended", info["no_live_builder"]), ("declared", ())):
        p, dl, contrib = stack_prob(fz, s, base, unbuilt)
        g[f"stack_p_{name}"] = p
        arms[name] = grade(g[keep], base[keep], p[keep])
        arms[name]["delta_logit_mean_sd"] = [float(dl[keep].mean()), float(dl[keep].std())]
        arms[name]["mean_contribution_by_column"] = {
            c: round(float(v), 4)
            for c, v in zip(fz["columns"], contrib[keep].mean(0), strict=False)
            if v != 0
        }
    g26 = pd.read_parquet(ROOT / "artifacts" / "mod27_replay_inputs" / "g26.parquet")
    both = g26.set_index("game_id")[list(ps.FLAGS)].join(
        s.set_index(g["game_id"])[list(ps.FLAGS)], rsuffix="_live", how="inner"
    )
    agree = {
        f: int((both[f].astype(float) == both[f + "_live"].astype(float)).sum()) for f in ps.FLAGS
    }
    v25 = u6_values(sch, 2025, 18).set_index(["game_id", "team"])
    a25 = pd.read_parquet(ps.U6).query("season == 2025").set_index(["game_id", "team"])
    j25 = a25.join(v25, rsuffix="_live", how="inner")
    rep = {
        "label": "PARTLY SEEN: 2026 weeks 1-10 were looked at before the freeze",
        "offset": "served card (QW has no 2026 rows)",
        "games_scheduled": len(g),
        "missing_card": int(g["card_p"].isna().sum()),
        "missing_pool_line": int(g["pool_line"].isna().sum()),
        "amended": arms["amended"],
        "declared": arms["declared"],
        "signal_info": info,
        "flag_parity_with_g26": {"games": len(both), "agree": agree},
        "u6_parity_2025": {
            n: float((j25[n] - j25[n + "_live"]).abs().max())
            for n in ("margin_sos_gap", "last_opponent_strength")
        },
        "nonzero_signals_2026": {c: int((s[c].fillna(0) != 0).sum()) for c in fz["signals"]},
        "frozen_sha256": sha_file(FROZEN),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    g.assign(**{c: s[c] for c in fz["signals"]}).to_parquet(OUT / "replay_2026_w1_10.parquet")
    (OUT / "replay_2026_w1_10.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


def resolve(arg, sch, now):
    if arg != "auto":
        return int(arg)
    g = sch[(sch["season"] == SEASON) & (sch["game_type"] == "REG")]
    k = _kickoff_utc(g)
    ahead = g.loc[k > pd.Timestamp(now), "week"]
    if ahead.empty:
        raise SystemExit("no unplayed regular-season week")
    return int(ahead.min())


def cmd_pregame(a):
    now = datetime.now(UTC)
    fz = json.loads(FROZEN.read_text())
    sch = schedules()
    week = resolve(a.week, sch, now)
    if not a.dry and not FIRST_WEEK <= week <= LAST_WEEK:
        raise SystemExit(
            f"week {week} outside the declared forward window {FIRST_WEEK}-{LAST_WEEK}"
        )
    g = season_games(sch, [week])
    s, info = live_signals(g, sch, fz["signals"])
    base, base_at = FL_served_now(week, now)
    games, skipped = [], {}
    for i, r in g.iterrows():
        if not a.dry and r["cutoff_utc"] <= pd.Timestamp(now):
            skipped[r["game_id"]] = "past_cutoff"
            continue
        if r["game_id"] not in base:
            skipped[r["game_id"]] = "no_published_card"
            continue
        p0 = float(base[r["game_id"]])
        p, dl, _ = stack_prob(fz, s.loc[[i]], np.array([p0]), info["no_live_builder"])
        pdec, _, _ = stack_prob(fz, s.loc[[i]], np.array([p0]))
        games.append(
            {
                "game_id": r["game_id"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "kickoff_utc": pd.Timestamp(r["kickoff"]).isoformat(),
                "pick_deadline_utc": pd.Timestamp(r["pick_deadline_utc"]).isoformat(),
                "signals": {
                    c: (None if pd.isna(s.at[i, c]) else float(s.at[i, c])) for c in fz["signals"]
                },
                "base_home_probability": p0,
                "base_published_at_utc": base_at[r["game_id"]],
                "delta_logit": float(dl[0]),
                "stack_home_probability": float(p[0]),
                "declared_rule_home_probability": float(pdec[0]),
                "side_changed": bool((p0 >= 0.5) != (p[0] >= 0.5)),
            }
        )
    doc = {
        "season": SEASON,
        "week": week,
        "created_at_utc": now.strftime(STAMP),
        "frozen_sha256": sha_file(FROZEN),
        "signal_info": info,
        "skipped": skipped,
        "games": games,
    }
    if a.dry:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        for gm in games:
            print(
                f"{gm['game_id']:<18} base {gm['base_home_probability']:.4f} "
                f"dlogit {gm['delta_logit']:+.4f} stack {gm['stack_home_probability']:.4f} "
                f"changed {gm['side_changed']}"
            )
        print("DRY: nothing written")
        return
    if not games:
        print(json.dumps({k: v for k, v in doc.items() if k != "games"}, indent=1, default=str))
        print("no game before its cutoff with a published card; nothing written")
        return
    out = OUT / f"{SEASON}_week{week:02d}_{now.strftime(STAMP)}.json"
    out.write_text(json.dumps(doc, indent=1, default=str))
    print(out, len(games), "games")


def FL_served_now(week, now):
    pub = pd.read_parquet(ROOT / "artifacts" / "clv_ledger" / "published_picks.parquet")
    pub = pub[(pub["season"] == SEASON) & (pub["week"] == week)]
    pub = pub[pd.to_datetime(pub["published_at_utc"], utc=True) <= pd.Timestamp(now)]
    last = pub.sort_values("published_at_utc").groupby("game_id").tail(1)
    home = np.where(
        last["pick_team"] == last["home_team"], last["displayed_score"], 1 - last["displayed_score"]
    )
    return dict(zip(last["game_id"], home, strict=True)), dict(
        zip(last["game_id"], last["published_at_utc"].astype(str), strict=True)
    )


def cmd_score(a):
    sha = sha_file(FROZEN)
    rows = []
    for f in sorted(OUT.glob(f"{SEASON}_week*_*.json")):
        d = json.loads(f.read_text())
        if d["frozen_sha256"] != sha or not FIRST_WEEK <= d["week"] <= LAST_WEEK:
            continue
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC")
        made = pd.Timestamp(datetime.strptime(d["created_at_utc"], STAMP).replace(tzinfo=UTC))
        for gm in d["games"]:
            cut = min(pd.Timestamp(gm["kickoff_utc"]), pd.Timestamp(gm["pick_deadline_utc"]))
            if made < cut and mtime < cut:
                rows.append({**gm, "week": d["week"], "created": d["created_at_utc"]})
    if not rows:
        print("no admissible forward rows yet; nothing graded")
        return
    t = pd.DataFrame(rows).sort_values("created").drop_duplicates("game_id", keep="last")
    sch = schedules().set_index("game_id")
    t["result"] = pd.to_numeric(t["game_id"].map(sch["result"]), errors="coerce")
    t["pool_line"] = t["game_id"].map(pool_lines(sorted(t["week"].unique())))
    t = t.dropna(subset=["result", "pool_line"])
    t = t[t["result"] != t["pool_line"]]
    if t.empty:
        print("no settled forward game at the pool line yet; nothing graded")
        return
    y = (t["result"] > t["pool_line"]).astype(float).to_numpy()
    pc, pst = (
        t["base_home_probability"].to_numpy(float),
        t["stack_home_probability"].to_numpy(float),
    )
    ch = (pst >= 0.5) != (pc >= 0.5)
    rep = {
        "games": len(t),
        "weeks": sorted(int(w) for w in t["week"].unique()),
        "card_record": ps.record(pc, y),
        "stack_record": ps.record(pst, y),
        "changed_picks": int(ch.sum()),
        "changed_record_stack": ps.record(pst[ch], y[ch]) if ch.any() else "0-0",
        "log_loss": {"card": float(ps.ll(pc, y).mean()), "stack": float(ps.ll(pst, y).mean())},
        "frozen_sha256": sha,
    }
    if len(np.unique(t["week"])) > 1:
        rng = np.random.default_rng(ps.SEED)
        wk = t["week"].to_numpy(int)
        acc = 100 * (
            ((pst >= 0.5) == (y == 1)).astype(float) - ((pc >= 0.5) == (y == 1)).astype(float)
        )
        rep["accuracy_week_block"] = ps.block_boot(wk, acc, np.ones(len(t)), rng)
        rep["log_loss_week_block"] = ps.block_boot(
            wk, ps.ll(pc, y) - ps.ll(pst, y), np.ones(len(t)), rng
        )
    (OUT / "score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_slots(a):
    import capture_scheduler as cs

    jobs = [
        j
        for v in vars(cs).values()
        if isinstance(v, (list, tuple)) and v and type(v[0]).__name__ == "Job"
        for j in v
    ]
    fid = sorted(
        {(j.day, j.at) for j in jobs if j.name.startswith("card_fidelity_")},
        key=lambda x: (["mon", "tue", "wed", "thu", "fri", "sat", "sun"].index(x[0]), x[1]),
    )
    script = str(ROOT / "scripts" / "pooled_stack_forward.py")
    argv = [str(cs.UV), "run", "--no-sync", "python", script]
    print(json.dumps({"pregame_argv": [*argv, "pregame"], "pregame_slots": fid}, indent=1))
    print(json.dumps({"score_argv": [*argv, "score"], "score_slot": SCORE_SLOT}, indent=1))


def main():
    p = argparse.ArgumentParser(
        description=(
            "Pooled as-of signal stack: frozen fit on 2020-25, 2026 weeks 1-10 replay on the "
            "served card, ablation, and the declared weeks 11-18 forward log. Research only."
        )
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("freeze").set_defaults(fn=cmd_freeze)
    ab = sp.add_parser("ablate")
    ab.add_argument("--workers", type=int, default=4)
    ab.set_defaults(fn=cmd_ablate)
    sp.add_parser("replay").set_defaults(fn=cmd_replay)
    pg = sp.add_parser("pregame")
    pg.add_argument("--week", default="auto")
    pg.add_argument("--dry", action="store_true")
    pg.set_defaults(fn=cmd_pregame)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    sp.add_parser("slots").set_defaults(fn=cmd_slots)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
