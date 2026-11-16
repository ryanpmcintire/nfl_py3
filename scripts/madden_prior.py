from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/madden"
OUT = ROOT / "artifacts/madden_prior"
SNAPS = ROOT / "data/players/raw/20261111T141547Z/snap_counts.parquet"
SPINE = ROOT / "data/processed/game_features.parquet"
SOURCE = (
    "https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/main/"
    "data/madden/{kind}/{season}.csv"
)
KIND_PREFIX = {"processed": "", "stage": "stage_"}
USER_AGENT = "Mozilla/5.0 (research-client)"
UNITS = {"off": "offense_pct", "def": "defense_pct"}
MADDEN_COLUMNS = ["season", "fullname", "team", "position", "pfr_id", "overallrating"]
LINE = "tue_open_home_spread"
EARLY = (1, 6)
REFERENCE_RECORD = "836-667"
SHUFFLES = 200
SEASON_2026 = 2026
FAMILY = "madden_talent_prior_gap_frame"


def fetch(seasons, pause):
    RAW.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        for kind, prefix in KIND_PREFIX.items():
            path = RAW / f"{prefix}{season}.csv"
            if path.exists():
                continue
            req = urllib.request.Request(
                SOURCE.format(kind=kind, season=season), headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
            time.sleep(pause)


def load_madden(seasons):
    frames, joins = [], {}
    for season in seasons:
        stage = pd.read_csv(
            RAW / f"stage_{season}.csv", usecols=[*MADDEN_COLUMNS[:-2], "overallrating"]
        )
        assert (stage.season == season).all()
        stage = stage.drop_duplicates()
        reg = pd.read_csv(RAW / f"{season}.csv", usecols=["fullname", "team", "pfr_id"]).dropna()
        by_team = reg.drop_duplicates(["fullname", "team"], keep=False)
        unique_name = reg.drop_duplicates("fullname", keep=False)[["fullname", "pfr_id"]]
        d = stage.merge(by_team, on=["fullname", "team"], how="left")
        fill = d.pfr_id.isna()
        d.loc[fill, "pfr_id"] = d.loc[fill, "fullname"].map(
            unique_name.set_index("fullname").pfr_id
        )
        joins[season] = {
            "madden_players": len(d),
            "pfr_by_name_team": int((~fill).sum()),
            "pfr_by_unique_name": int((fill & d.pfr_id.notna()).sum()),
            "unjoined": int(d.pfr_id.isna().sum()),
        }
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["overallrating"] = pd.to_numeric(m.overallrating, errors="coerce")
    m = m.dropna(subset=["overallrating", "team", "position"])
    return m, joins


def load_snaps():
    s = pd.read_parquet(
        SNAPS,
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "pfr_player_id",
            "team",
            *UNITS.values(),
        ],
    )
    s = s[s.game_type == "REG"].drop(columns="game_type")
    s["season"] = s.season.astype(int)
    s["week"] = s.week.astype(int)
    for c in UNITS.values():
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)
    return s


def team_code_map(madden, snaps):
    mapping = {}
    for season, roster in madden.groupby("season"):
        wk = snaps[snaps.season == season]
        known = set(wk.team)
        for team in sorted(set(roster.team) - known):
            ids = set(roster.loc[roster.team == team, "pfr_id"].dropna())
            overlap = wk[wk.pfr_player_id.isin(ids)].groupby("team").pfr_player_id.nunique()
            if len(overlap):
                mapping[(season, team)] = overlap.idxmax()
    return mapping


def unit_index(frame, weight_col):
    w = frame[weight_col].to_numpy(dtype=float)
    r = frame.overallrating.to_numpy(dtype=float)
    rated = np.isfinite(r) & (w > 0)
    total = w[w > 0].sum()
    if not rated.any() or total <= 0:
        return np.nan, np.nan
    return float((w[rated] * r[rated]).sum() / w[rated].sum()), float(w[rated].sum() / total)


def prev_game_indexes(snaps, ratings):
    s = snaps.merge(
        ratings, left_on=["season", "pfr_player_id"], right_on=["season", "pfr_id"], how="left"
    )
    rows = []
    for (season, team, week), g in s.groupby(["season", "team", "week"]):
        rec = {"season": season, "team": team, "snap_week": week}
        for unit, col in UNITS.items():
            rec[f"{unit}_idx"], rec[f"{unit}_cov"] = unit_index(g, col)
        rows.append(rec)
    return pd.DataFrame(rows)


def ranked_roster(madden):
    r = madden.copy()
    r["slot_rank"] = r.groupby(["season", "team", "position"]).overallrating.rank(
        ascending=False, method="first"
    )
    return r


def week1_slot_pct(roster, snaps):
    w1 = snaps[snaps.week == snaps.groupby("season").week.transform("min")]
    w1 = w1.rename(columns={"pfr_player_id": "pfr_id"})[
        ["season", "team", "pfr_id", *UNITS.values()]
    ]
    played_teams = w1[["season", "team"]].drop_duplicates()
    r = roster.merge(played_teams, on=["season", "team"], how="inner")
    r = r.merge(w1, on=["season", "team", "pfr_id"], how="left")
    for c in UNITS.values():
        r[c] = r[c].fillna(0.0)
    return r.groupby(["season", "position", "slot_rank"])[list(UNITS.values())].mean().reset_index()


def week1_indexes(roster, slot_pct, seasons):
    rows = []
    for season in seasons:
        prior = slot_pct[slot_pct.season < season]
        if prior.empty:
            continue
        weights = (
            prior.groupby(["position", "slot_rank"])[list(UNITS.values())].mean().reset_index()
        )
        cur = roster[roster.season == season].merge(
            weights, on=["position", "slot_rank"], how="left"
        )
        for c in UNITS.values():
            cur[c] = cur[c].fillna(0.0)
        for team, g in cur.groupby("team"):
            rec = {"season": season, "team": team}
            for unit, col in UNITS.items():
                rec[f"{unit}_idx"], rec[f"{unit}_cov"] = unit_index(g, col)
            rows.append(rec)
    return pd.DataFrame(rows)


def loso_vs_line(games, col, line_col):
    out = pd.Series(np.nan, index=games.index)
    ok = games[col].notna() & games[line_col].notna()
    for season in sorted(games.season.unique()):
        train = games[ok & (games.season != season)]
        test = ok & (games.season == season)
        if train.empty or not test.any():
            continue
        b, a = np.polyfit(train[line_col].to_numpy(), train[col].to_numpy(), 1)
        out[test] = games.loc[test, col] - (a + b * games.loc[test, line_col])
    return out


def build(args):
    seasons = list(range(args.first_season, args.last_season + 1))
    fetch(seasons, args.pause)
    madden, joins = load_madden(seasons)
    snaps = load_snaps()
    snaps = snaps[snaps.season.isin(seasons)]
    mapping = team_code_map(madden, snaps)
    madden["team"] = [
        mapping.get((s, t), t) for s, t in zip(madden.season, madden.team, strict=True)
    ]
    ratings = (
        madden.dropna(subset=["pfr_id"])
        .sort_values("overallrating", ascending=False)
        .drop_duplicates(["season", "pfr_id"])[["season", "pfr_id", "overallrating"]]
    )
    prev = prev_game_indexes(snaps, ratings)
    roster = ranked_roster(madden)
    slot_pct = week1_slot_pct(roster, snaps)
    wk1 = week1_indexes(roster, slot_pct, seasons)

    spine = pd.read_parquet(
        SPINE,
        columns=["game_id", "season", "week", "game_type", "home_team", "away_team", "spread_line"],
    )
    spine = spine[(spine.game_type == "REG") & spine.season.isin(seasons)].copy()
    spine["season"] = spine.season.astype(int)
    spine["week"] = spine.week.astype(int)
    first_week = spine.groupby("season").week.transform("min")
    team_rows = []
    for side in ("home", "away"):
        t = spine[["game_id", "season", "week"]].copy()
        t["team"] = spine[f"{side}_team"]
        t["side"] = side
        t["is_first"] = spine.week.eq(first_week)
        team_rows.append(t)
    teams = pd.concat(team_rows, ignore_index=True)
    first = teams[teams.is_first].merge(wk1, on=["season", "team"], how="left")
    first["source"] = "week1_template"
    later = teams[~teams.is_first].sort_values("week")
    prev_sorted = prev.sort_values("snap_week")
    later = pd.merge_asof(
        later,
        prev_sorted,
        left_on="week",
        right_on="snap_week",
        by=["season", "team"],
        allow_exact_matches=False,
        direction="backward",
    )
    later["source"] = "previous_game"
    teams = pd.concat([first, later], ignore_index=True)
    teams["idx"] = teams[["off_idx", "def_idx"]].mean(axis=1, skipna=False)
    wide = teams.pivot(
        index="game_id", columns="side", values=["idx", "off_cov", "def_cov", "source"]
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    games = spine.merge(wide.reset_index(), on="game_id", how="left")
    games["madden_talent_diff"] = pd.to_numeric(games.idx_home) - pd.to_numeric(games.idx_away)
    games["madden_talent_vs_line"] = loso_vs_line(games, "madden_talent_diff", "spread_line")
    OUT.mkdir(parents=True, exist_ok=True)
    keep = [
        "game_id",
        "season",
        "week",
        "spread_line",
        "madden_talent_diff",
        "madden_talent_vs_line",
        "source_home",
        "off_cov_home",
        "def_cov_home",
        "off_cov_away",
        "def_cov_away",
    ]
    games[keep].to_parquet(OUT / "madden_prior_features.parquet", index=False)
    diag = {
        "pfr_joins": joins,
        "team_code_map": {f"{s}:{t}": v for (s, t), v in mapping.items()},
        "rated_players_by_season": madden.groupby("season").size().to_dict(),
        "rated_with_pfr_share": float(madden.pfr_id.notna().mean()),
        "games": len(games),
        "games_with_term": int(games.madden_talent_vs_line.notna().sum()),
        "missing_by_season": games.groupby("season")
        .madden_talent_vs_line.apply(lambda x: int(x.isna().sum()))
        .to_dict(),
        "mean_snap_coverage": {
            c: float(pd.to_numeric(games[c]).mean())
            for c in ["off_cov_home", "def_cov_home", "off_cov_away", "def_cov_away"]
        },
        "corr_diff_line_by_season": games.groupby("season")
        .apply(lambda g: float(g.madden_talent_diff.corr(g.spread_line)))
        .round(3)
        .to_dict(),
    }
    (OUT / "build_diagnostics.json").write_text(
        json.dumps(diag, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps(diag, default=str))


def served(args):
    import pooled_stack as PS
    from expert_rankings_forward import evaluate, offset_fit, standardise
    from mod27_fenced_repairs_replay import FOUR, Fit
    from scipy.special import expit

    pop = PS.population()
    pop["game_id"] = pop["game_id"].astype(str)
    feats = pd.read_parquet(OUT / "madden_prior_features.parquet")
    feats["game_id"] = feats["game_id"].astype(str)
    pop["madden_talent_diff"] = pop["game_id"].map(feats.set_index("game_id")["madden_talent_diff"])
    term = loso_vs_line(pop, "madden_talent_diff", LINE).to_numpy(float)
    season = pop["season"].to_numpy(int)
    week = pop["week"].to_numpy(int)
    decided = ~pop["push"].to_numpy(bool)
    ylab = pop["y"].to_numpy(float)
    early = (week >= EARLY[0]) & (week <= EARLY[1])
    raw = np.where(early, term, np.nan)
    seasons = sorted(set(season.tolist()))
    cache = {}

    def off_for(excl, rows):
        key = frozenset(excl)
        if key not in cache:
            cache[key] = Fit(FOUR, pop[decided & ~np.isin(season, list(key))])
        return PS.logit(cache[key].predict(pop[rows]))

    off1 = np.full(len(pop), np.nan)
    for s in seasons:
        off1[season == s] = off_for({s}, season == s)
    qw = expit(off1)
    qw_record = PS.record(qw[decided], ylab[decided])
    if qw_record != REFERENCE_RECORD:
        raise SystemExit(f"QW reference not reproduced: {qw_record}")
    off_tr = {}
    for s in seasons:
        o = np.full(len(pop), np.nan)
        for t in seasons:
            if t != s:
                o[season == t] = off_for({s, t}, season == t)
        off_tr[s] = o

    def run(values, control=False):
        p = np.full(len(pop), np.nan)
        folds = {}
        for s in seasons:
            tr = season != s
            if control:
                xs, mu, sd = early.astype(float), 0.0, 1.0
            else:
                xs, mu, sd = standardise(values, tr)
            b = offset_fit(xs[tr], ylab[tr], off_tr[s][tr])
            held = season == s
            p[held] = expit(off1[held] + b * xs[held])
            folds[str(s)] = {"coefficient_per_sd": b, "mean": mu, "sd": sd}
        return p, folds

    def gains(p, m):
        d = m & decided
        hit_p = (p >= 0.5) == (ylab == 1)
        hit_q = (qw >= 0.5) == (ylab == 1)
        return (
            float(100 * (hit_p[d].astype(float) - hit_q[d].astype(float)).mean()),
            float((PS.ll(qw, ylab) - PS.ll(p, ylab))[m].mean()),
        )

    p, folds = run(raw)
    p_ctrl, ctrl_folds = run(raw, control=True)
    xs_all, mu_all, sd_all = standardise(raw, np.ones(len(pop), bool))
    b_full = offset_fit(xs_all, ylab, off1)
    p_in = expit(off1 + b_full * xs_all)
    split = seasons[len(seasons) // 2]
    halves = {
        name: offset_fit(xs_all[m], ylab[m], off1[m])
        for name, m in (
            (f"{seasons[0]}-{split - 1}", season < split),
            (f"{split}-{seasons[-1]}", season >= split),
        )
    }
    rng = np.random.default_rng(PS.SEED)
    wk = season * 100 + week
    ev = {
        "weeks_1_6_week_block": evaluate(p[early], qw[early], ylab[early], wk[early], rng),
        "weeks_1_6_season_block": evaluate(p[early], qw[early], ylab[early], season[early], rng),
        "all_games_week_block": evaluate(p, qw, ylab, wk, rng),
        "control_weeks_1_6_indicator": evaluate(
            p_ctrl[early], qw[early], ylab[early], wk[early], rng
        ),
    }
    real_acc, real_ll = gains(p, early)
    in_acc, in_ll = gains(p_in, early)
    finite = early & np.isfinite(term)
    null_acc, null_ll = [], []
    for _ in range(SHUFFLES):
        shuffled = raw.copy()
        for s in seasons:
            idx = np.flatnonzero(finite & (season == s))
            shuffled[idx] = rng.permutation(raw[idx])
        ps, _ = run(shuffled)
        a, ll = gains(ps, early)
        null_acc.append(a)
        null_ll.append(ll)
    null_acc, null_ll = np.array(null_acc), np.array(null_ll)
    replay = {
        "status": "not_computable",
        "stage_2026_file_present": (RAW / "stage_2026.csv").exists(),
        "snap_counts_2026_rows": int((load_snaps().season == SEASON_2026).sum()),
    }
    res = {
        "games": len(pop),
        "decided": int(decided.sum()),
        "pushes": int((~decided).sum()),
        "weeks_1_6_games": int(early.sum()),
        "weeks_1_6_term_games": int(finite.sum()),
        "line": LINE,
        "qw_record": qw_record,
        "qw_weeks_1_6_record": PS.record(qw[early & decided], ylab[early & decided]),
        "corr_diff_line": float(
            np.corrcoef(
                pop.loc[finite, "madden_talent_diff"].to_numpy(float),
                pop.loc[finite, LINE].to_numpy(float),
            )[0, 1]
        ),
        "fold": folds,
        "folds_positive": int(sum(f["coefficient_per_sd"] > 0 for f in folds.values())),
        "control_fold": ctrl_folds,
        "coefficient_per_sd_full": b_full,
        "term_mean": mu_all,
        "term_sd": sd_all,
        "halves_coefficient_per_sd": halves,
        "evaluation": ev,
        "held_out_weeks_1_6": {"accuracy_points": real_acc, "log_loss_gain": real_ll},
        "in_sample_weeks_1_6": {"accuracy_points": in_acc, "log_loss_gain": in_ll},
        "in_sample_minus_held_out": {
            "accuracy_points": in_acc - real_acc,
            "log_loss_gain": in_ll - real_ll,
        },
        "matched_null": {
            "shuffles": SHUFFLES,
            "accuracy_percentile": float((null_acc < real_acc).mean()),
            "log_loss_percentile": float((null_ll < real_ll).mean()),
            "accuracy_quantiles": {
                str(q): float(np.quantile(null_acc, q)) for q in (0.05, 0.5, 0.95)
            },
            "log_loss_quantiles": {
                str(q): float(np.quantile(null_ll, q)) for q in (0.05, 0.5, 0.95)
            },
        },
        "frozen": {
            "coefficient_logit_per_sd": b_full,
            "term_mean": mu_all,
            "term_sd": sd_all,
            "weeks": list(EARLY),
            "line": LINE,
        },
        "replay_2026_weeks_1_6": replay,
    }
    (OUT / "served_look_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    w = ev["weeks_1_6_week_block"]
    sb = ev["weeks_1_6_season_block"]
    text = (
        "Madden launch-rating talent diff minus a leave-one-season-out OLS map on the Tuesday "
        "opener, weeks 1-6 only, one offset-logistic coefficient per SD on the QW served stage "
        "at the opener 2020-25, pushes kept at 0.5, nested LOSO offsets. Full "
        f"{b_full:+.4f}/SD, folds positive {res['folds_positive']}/{len(seasons)}; weeks 1-6 "
        f"{w['record_term']} v QW {w['record_base']}; changed {w['changed']} "
        f"({w['changed_record_term']} v {w['changed_record_base']}); season-block P+ acc "
        f"{sb['accuracy_points']['probability_positive']:.3f} ll "
        f"{sb['log_loss_gain']['probability_positive']:.3f}; matched-null percentile acc "
        f"{res['matched_null']['accuracy_percentile']:.3f} ll "
        f"{res['matched_null']['log_loss_percentile']:.3f}."
    )
    cells = []
    for metric, units in (
        ("accuracy_points", "accuracy_points"),
        ("log_loss_gain", "log_loss_improvement"),
    ):
        b = w[metric]
        c = {
            "name": f"madden_talent_served_opener_weeks_1_6_{units}",
            "description": text,
            "effect": b["effect"],
            "effect_units": units,
            "interval_low": b["interval_low"],
            "interval_high": b["interval_high"],
            "probability_positive": b["probability_positive"],
            "season_start": seasons[0],
            "season_end": seasons[-1],
            "sample_games": int(w["decided"] if units == "accuracy_points" else w["games"]),
        }
        if b["interval_high"] < 0 and sb[metric]["interval_high"] < 0:
            c["classification"] = "refuted_mechanism"
            c["closing_ground"] = "wrong_sign_resolved"
            c["classification_evidence"] = (
                "Week-block and season-block intervals lie wholly below zero on the served stage."
            )
        cells.append(c)
    batch = {
        "league": "nfl",
        "classification": "unresolved_below_power",
        "source": "artifacts/madden_prior/served_look_results.json",
        "category": "market",
        "family": FAMILY,
        "notes": (
            "The one served look declared after the margin-label gate; matched null, halves, "
            "folds, weeks 1-6 indicator control and in-sample gap in the source. The 2026 replay "
            "was not computable: the archive has no 2026 launch file."
        ),
        "classification_evidence": (
            "Week-block bootstrap with probability_positive; no refuted mechanism and no "
            "positive control bounds an effect this size."
        ),
        "plain_summary": (
            "Adds the Madden launch ratings of each team's expected starters to the pick "
            "probability for Weeks 1-6 and checks the picks it would have changed."
        ),
        "cells": cells,
    }
    (OUT / "served_look_batch.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("fold", "control_fold")}, indent=1))
    print(json.dumps({s: round(f["coefficient_per_sd"], 4) for s, f in folds.items()}))


def main():
    ap = argparse.ArgumentParser(description="Madden launch-rating team talent prior per game.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the per-game Madden talent features")
    b.add_argument("--first-season", type=int, required=True, help="first season to build")
    b.add_argument("--last-season", type=int, required=True, help="last season to build")
    b.add_argument("--pause", type=float, required=True, help="seconds between source requests")
    b.set_defaults(func=build)
    s = sub.add_parser("served", help="the one declared served look at the opener 2020-25")
    s.set_defaults(func=served)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
