import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import xlg14_family_u7 as u7  # noqa: E402
import xlg14_luck_family as u3  # noqa: E402

from nfl_ats.cfb_features import cfb_season_partitions  # noqa: E402

x14 = u3.x14
OUT = ROOT / "artifacts" / "xlg14_family_u10"
LANE = "docs/lanes/xlg-14-u10-2026-11-21.md"
POWER_LANE = u3.POWER_LANE
FAMILY = "xlg14_run_stuff_family_u10"
STREAM = 10
RUN_TYPES = ["Rush", "Rushing Touchdown"]
LINE_OF_SCRIMMAGE = 0.0
MIN_FIT_ROWS = u7.MIN_FIT_ROWS
PLAY_COLUMNS = [
    "game_id",
    "week",
    "seasonType",
    "pos_team_id",
    "def_pos_team_id",
    "type.text",
    "rush",
    "kneel_down",
    "statYardage",
]
SUMS = ["or_n", "or_d", "dr_n", "dr_d"]
GAME_COLUMNS = [*SUMS, "ats_v", "ats_n"]
RAW = {
    "off_stuff_avoid_oe": "off_raw",
    "def_stuff_rate_oe": "def_raw",
    "run_stuff_net_oe": "net_raw",
}
CANDIDATES = {
    "off_stuff_avoid_oe": {
        "hypothesis": "positive",
        "what": (
            "minus the offence's designed runs stopped at or behind the line of scrimmage, in "
            "excess of the league rate of each game's own week, per own designed run over earlier "
            "weeks, residualised on the team's own ATS margin to date."
        ),
        "plain": "get stuffed at the line less often when they run",
    },
    "def_stuff_rate_oe": {
        "hypothesis": "positive",
        "what": (
            "the defence's stops of opponent designed runs at or behind the line of scrimmage, in "
            "excess of the league rate of each game's own week, per opponent designed run over "
            "earlier weeks, residualised on the team's own ATS margin to date."
        ),
        "plain": "stuff opposing runs at the line more often",
    },
    "run_stuff_net_oe": {
        "hypothesis": "positive",
        "what": (
            "the sum of the offensive and defensive run-stuff reads, residualised on the team's "
            "own ATS margin to date."
        ),
        "plain": "win the line of scrimmage on running plays on both sides of the ball",
    },
}
NAMES = list(CANDIDATES)


def season_tables(season, path, ats):
    f = pd.read_parquet(path, columns=PLAY_COLUMNS)
    f = f[pd.to_numeric(f["seasonType"], errors="coerce").eq(u3.REGULAR_SEASON)].copy()
    f["week"] = pd.to_numeric(f["week"], errors="coerce")
    f = f[f["pos_team_id"].notna() & f["def_pos_team_id"].notna() & f["week"].notna()]
    f = f.reset_index(drop=True)
    f["game_id"] = pd.to_numeric(f["game_id"]).astype("int64")
    f["pos"] = pd.to_numeric(f["pos_team_id"]).astype("int64")
    f["dfn"] = pd.to_numeric(f["def_pos_team_id"]).astype("int64")
    f = f[f["pos"].ne(f["dfn"])].copy()
    pairs = f[["game_id", "pos", "dfn"]].drop_duplicates()
    teams = pd.concat(
        [
            pairs.rename(columns={"pos": "team", "dfn": "opp"}),
            pairs.rename(columns={"dfn": "team", "pos": "opp"}),
        ]
    ).drop_duplicates()
    count = teams.groupby("game_id")["team"].transform("nunique")
    teams = teams[count.eq(2)].drop_duplicates(["game_id", "team"]).reset_index(drop=True)
    week = f.groupby("game_id")["week"].first()
    rush = f["rush"].fillna(False).astype(bool)
    kneel = f["kneel_down"].fillna(False).astype(bool)
    kneel_share = float(kneel[rush].mean()) if rush.any() else 0.0
    if kneel_share == 0.0:
        return None, {"run_rows": int(rush.sum()), "kneel_share": kneel_share}
    run = rush & ~kneel & f["type.text"].astype(str).isin(RUN_TYPES)
    yards = pd.to_numeric(f["statYardage"], errors="coerce")
    run = run & yards.notna()
    r = f[run].copy()
    r["stuff"] = yards[run].le(LINE_OF_SCRIMMAGE).astype(float)
    r["one"] = 1.0
    league = r.groupby("week")[["stuff", "one"]].sum()
    rate = league["stuff"] / league["one"]
    r["dev"] = r["stuff"] - r["week"].map(rate)
    o = r.groupby(["game_id", "pos"])[["one", "dev"]].sum()
    o.columns = ["or_n", "or_d"]
    o.index = o.index.set_names(["game_id", "team"])
    d = r.groupby(["game_id", "dfn"])[["one", "dev"]].sum()
    d.columns = ["dr_n", "dr_d"]
    d.index = d.index.set_names(["game_id", "team"])
    t = pd.concat([o, d], axis=1).reset_index()
    m = teams.merge(t, on=["game_id", "team"], how="left").merge(
        ats, on=["game_id", "team"], how="left"
    )
    raw = pd.DataFrame(
        {
            "game_id": m["game_id"].astype("int64"),
            "team": m["team"].astype("int64"),
            "opp": m["opp"].astype("int64"),
            "season": season,
        }
    )
    raw["week"] = m["game_id"].map(week).to_numpy(float)
    for c in SUMS:
        raw[c] = m[c].fillna(0.0).to_numpy(float)
    raw["ats_n"] = m["ats"].notna().astype(float).to_numpy()
    raw["ats_v"] = m["ats"].fillna(0.0).to_numpy(float)
    diag = {
        "run_rows": int(run.sum()),
        "kneel_share": kneel_share,
        "league_stuff_share": float(r["stuff"].mean()),
        "week_rate_min_max": [float(rate.min()), float(rate.max())],
        "team_games": len(raw),
        "share_with_opener": float(raw["ats_n"].mean()),
        "mean_runs_per_team_game": float(raw["or_n"].mean()),
    }
    return raw, diag


def to_date(raw):
    r = raw.reset_index(drop=True).copy()
    r["games"] = 1.0
    cols = [*GAME_COLUMNS, "games"]
    tw = r.groupby(["team", "season", "week"])[cols].sum().reset_index()
    weeks = r[["season", "week"]].drop_duplicates()
    grid = tw[["team", "season"]].drop_duplicates().merge(weeks, on="season")
    grid = grid.merge(tw, on=["team", "season", "week"], how="left")
    grid[cols] = grid[cols].fillna(0.0)
    grid = grid.sort_values(["team", "season", "week"]).reset_index(drop=True)
    pr = grid.groupby(["team", "season"])[cols].cumsum() - grid[cols]
    st = grid[["team", "season", "week"]].copy()
    st["off_raw"] = (-pr["or_d"] / pr["or_n"].where(pr["or_n"] > 0)).fillna(0.0)
    st["def_raw"] = (pr["dr_d"] / pr["dr_n"].where(pr["dr_n"] > 0)).fillna(0.0)
    st["net_raw"] = st["off_raw"] + st["def_raw"]
    st["ats_td"] = (pr["ats_v"] / pr["ats_n"].where(pr["ats_n"] > 0)).fillna(0.0)
    st["has_ats"] = pr["ats_n"] > 0
    st["prior_games"] = pr["games"]
    out = r[["game_id", "team", "season", "week"]].merge(
        st, on=["team", "season", "week"], how="left"
    )
    out = out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    fit = out["has_ats"] & out["prior_games"].gt(0)
    x = out["ats_td"].where(fit, 0.0)
    one = fit.astype(float)
    key = ["season", "week"]
    for name, rc in RAW.items():
        y = out[rc].where(fit, 0.0)
        s = pd.DataFrame({"n": one, "x": x, "y": y, "xx": x * x, "xy": x * y})
        s[key] = out[key]
        gs = s.groupby(key)[["n", "x", "y", "xx", "xy"]].sum().sort_index()
        before = gs.cumsum() - gs
        b = (before["xy"] - before["x"] * before["y"] / before["n"]) / (
            before["xx"] - before["x"] ** 2 / before["n"]
        )
        a = (before["y"] - b * before["x"]) / before["n"]
        ok = before["n"] >= MIN_FIT_ROWS
        coef = pd.DataFrame({"a": a.where(ok), "b": b.where(ok)}).reset_index()
        c = out[key].merge(coef, on=key, how="left")
        res = out[rc] - (c["a"].to_numpy() + c["b"].to_numpy() * out["ats_td"].to_numpy())
        res = res.where(c["a"].notna().to_numpy(), out[rc])
        out[name] = res.where(out["prior_games"].gt(0), 0.0).to_numpy(float)
    return out


def asof_check(raw, values, seasons=None, seed=None):
    seasons = x14.u1.SEASONS if seasons is None else seasons
    seed = [u3.ASOF_SEED, STREAM, u3.ASOF_STREAM] if seed is None else seed
    rng = np.random.default_rng(seed)
    pool = values[values["season"].isin(seasons)][["season", "week"]].drop_duplicates()
    picks = pool.sample(u3.ASOF_CUTOFFS, random_state=rng)
    worst, rows, cutoffs = 0.0, 0, []
    for s, w in picks.itertuples(index=False):
        r2 = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["week"] <= w))].copy()
        r2.loc[(r2["season"] == s) & (r2["week"] == w), GAME_COLUMNS] = 0.0
        v2 = to_date(r2)
        key = ["game_id", "team"]
        a = values[(values["season"] == s) & (values["week"] == w)].set_index(key)[NAMES]
        b = v2[(v2["season"] == s) & (v2["week"] == w)].set_index(key)[NAMES].reindex(a.index)
        gap = (a - b).abs().to_numpy()
        worst = max(worst, float(np.nanmax(gap)) if gap.size else 0.0)
        worst = float("inf") if np.isnan(gap).any() else worst
        rows += len(a)
        cutoffs.append([int(s), int(w), len(a), int((a != 0).any(axis=1).sum())])
    return {
        "cutoffs_season_week_teamgames_nonzero": cutoffs,
        "team_games": rows,
        "max_abs_difference": worst,
        "tolerance": u3.TOLERANCE,
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = cfb_season_partitions(ROOT / "data" / "cfb", "pbp")
    population = set(x14.u1.SEASONS)
    ats = u7.ats_rows()
    raws, diag, skipped = [], {}, []
    for season in sorted(s for s in parts if s in population):
        have = set(pq.ParquetFile(parts[season]).schema_arrow.names)
        if not set(PLAY_COLUMNS) <= have:
            skipped.append([int(season), "missing columns"])
            continue
        r, d = season_tables(int(season), parts[season], ats)
        diag[str(season)] = d
        if r is None:
            skipped.append([int(season), "no kneel-down flag"])
            continue
        raws.append(r)
    raw = pd.concat(raws, ignore_index=True)
    values = to_date(raw)
    check = asof_check(raw, values)
    pop = values[values["season"].isin(x14.u1.SEASONS)]
    cols = [*NAMES, *RAW.values(), "ats_td"]
    corr = pop[cols].corr().round(3)
    report = {
        "family": FAMILY,
        "skipped_seasons": skipped,
        "population_team_games": len(pop),
        "population_nonzero_share": {n: float(pop[n].ne(0).mean()) for n in NAMES},
        "population_sd": {n: float(pop[n].std(ddof=1)) for n in cols},
        "population_correlation": {n: corr[n].to_dict() for n in cols},
        "asof_check": check,
        "per_season": diag,
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("skipped", skipped)
    print("per_season", json.dumps(diag))
    print("nonzero", report["population_nonzero_share"])
    print("sd", {k: round(v, 4) for k, v in report["population_sd"].items()})
    print("corr", corr.to_string())
    print("asof", check, flush=True)
    if not check["max_abs_difference"] <= u3.TOLERANCE:
        raise SystemExit("as-of check failed; screen not run")
    values.to_parquet(OUT / "team_values.parquet", index=False)


def cells(m36, name, res, seasons, draws):
    spec = CANDIDATES[name]
    held = res["held_out"]
    early = res["halves"]["earlier"]["coefficient_per_sd"]
    late = res["halves"]["later"]["coefficient_per_sd"]
    survived = "yes" if res["survivor"] else "no"
    verdict = (
        "It stood out from chance and earns one declared pro-football check."
        if res["survivor"]
        else "No effect large enough for this test to see showed up."
    )
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"xlg14_run_stuff_cfb_{name}_{units}",
                "description": (
                    f"College-first run-stuff family, {name}: {spec['what']} "
                    "Built from strictly earlier weeks of seasons with a kneel-down flag "
                    "(2014 on; 2012-13 games carry 0); the residualising slope from team-games "
                    "strictly before the target week; home minus away; one offset-logistic "
                    "coefficient on a leave-one-season-out "
                    f"ridge cover probability with the opener as input and grade, {len(seasons)} "
                    f"seasons; declared sign {spec['hypothesis']}. Changed picks "
                    f"{held['changed']}, record {held['arm_right']}-{held['ref_right']}; "
                    f"coefficient per SD {res['coefficient_per_sd']:+.4f}; earlier and later "
                    f"season-half coefficients {early:+.4f} and {late:+.4f}; this metric sits at "
                    f"percentile {res['null_percentile'][k]:.1f} of {draws} within-season "
                    "shuffles of the same column; in-sample minus held-out "
                    f"{res['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(held[k]["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "cfb",
                "season_start": int(seasons[0]),
                "season_end": int(seasons[-1]),
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "No resolved wrong sign for the declared line-of-scrimmage mechanism, "
                    "split-half reliability of the trait not measured, and no positive "
                    "control bounds an effect this small: the college add-one harness shows "
                    f"about .08 to .11 logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: 3 candidates x 3 metrics = 9 recorded cells; 3 gating "
                    "reads (log-loss percentile in the matched null), 6 season-half "
                    "coefficients and 2 control reads; survivor rule declared in the lane "
                    f"before any outcome was joined; survivor: {survived}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    f"Checks whether college teams that {spec['plain']}, beyond what their "
                    f"cover record already says, go on to beat the opening spread more or less "
                    f"often. {verdict}"
                ),
            }
        )
    return out


def screen():
    started = time.time()
    u3.STREAM_LUCK = STREAM
    u7.NAMES = NAMES
    hp = u3.load_power()
    m36 = hp.M36
    d = hp.x14.cfb_frame()
    values = pd.read_parquet(OUT / "team_values.parquet")
    terms, both = u7.joined(d, values)
    frame = pd.DataFrame(
        {
            "game_id": d["game_id"].to_numpy(),
            "season": d["season"].to_numpy(int),
            "home_covered": d["y"].to_numpy(float),
            m36.SERVED_COLUMN: expit(d["logit_p"].to_numpy(float)),
        }
    )
    for n in NAMES:
        frame[n] = terms[n].to_numpy(float)
    seasons = sorted(int(s) for s in frame["season"].unique())
    diag = {n: float(np.corrcoef(terms[n], terms["ats_td"])[0, 1]) for n in NAMES}
    diag["served_logit"] = {n: float(np.corrcoef(terms[n], d["logit_p"])[0, 1]) for n in NAMES}
    out = {
        "family": FAMILY,
        "games": len(frame),
        "games_with_both_teams_in_play_by_play": both,
        "seasons": seasons,
        "draws": hp.DRAWS,
        "seed": [hp.SEED, STREAM],
        "correlation_with_own_ats_gap_and_served_logit": diag,
        "candidates": {},
    }
    print("corr", diag, flush=True)
    batch, lines = [], []
    for index, n in enumerate(NAMES):
        res, st = u3.read_candidate(hp, frame, n, index)
        if index == 0:
            out["controls"] = u3.controls(m36, st)
        del st
        out["candidates"][n] = res
        batch.extend(cells(m36, n, res, seasons, hp.DRAWS))
        h = res["held_out"]
        lines.append(
            f"{n}: games {res['games']} nonzero {res['nonzero_games']}; changed {h['changed']} "
            f"record {h['arm_right']}-{h['ref_right']}; acc {h['acc']['effect']:+.3f} "
            f"[{h['acc']['low']:+.3f},{h['acc']['high']:+.3f}] pp {h['acc']['pp']:.3f} "
            f"pct {res['null_percentile']['acc']:.1f}; ll {h['ll']['effect']:+.6f} "
            f"[{h['ll']['low']:+.6f},{h['ll']['high']:+.6f}] pp {h['ll']['pp']:.3f} "
            f"pct {res['null_percentile']['ll']:.1f}; brier {h['brier']['effect']:+.6f} "
            f"pct {res['null_percentile']['brier']:.1f}; null ll mean "
            f"{res['null']['ll']['mean']:+.6f} [{res['null']['ll']['q05']:+.6f},"
            f"{res['null']['ll']['q95']:+.6f}]; in-sample ll {res['in_sample']['ll']:+.6f} gap "
            f"{res['in_sample_minus_held_out']['ll']:+.6f}; coef {res['coefficient_per_sd']:+.4f} "
            f"folds {res['folds_with_full_sign']}/{len(seasons)}; halves "
            f"{res['halves']['earlier']['coefficient_per_sd']:+.4f} / "
            f"{res['halves']['later']['coefficient_per_sd']:+.4f}; survivor {res['survivor']}"
        )
        print(lines[-1], f"{time.time() - started:.0f}s", flush=True)
    out["survivors"] = [n for n in NAMES if out["candidates"][n]["survivor"]]
    c = out["controls"]
    lines.append(
        "controls: "
        + "; ".join(f"{k} acc {v['accuracy']:.3f} ll {v['log_loss']:.6f}" for k, v in c.items())
    )
    lines.append(f"survivors: {out['survivors']}; both-team games {both} of {len(frame)}")
    print(lines[-2], lines[-1], sep="\n", flush=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": batch}, indent=1), encoding="utf-8"
    )


NFL_OUT = OUT / "nfl"
NFL_COLUMNS = [
    "game_id",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "play_type",
    "qb_dropback",
    "qb_kneel",
    "yards_gained",
]
NFL_RUN = "run"
NFL_REGULAR = "REG"
NFL_EXPECTED_SIGN = 1.0
NFL_EARLIER_CONFIRMATIONS = 4
NFL_PLAIN = {
    "off_stuff_avoid_oe": "pro offences that get stuffed at the line less often on designed runs",
    "def_stuff_rate_oe": "pro defences that stuff designed runs at the line more often",
    "run_stuff_net_oe": "pro teams that win the line of scrimmage on designed runs on both sides",
}


def nfl_modules():
    import xlg14_fumble_luck_nfl_confirmation as fl
    import xlg14_u6_nfl_confirmation as u6n

    u6n.OUT = NFL_OUT
    u6n.FAMILY = FAMILY
    u6n.EXPECTED_SIGN = NFL_EXPECTED_SIGN
    return fl, u6n


def nfl_ats():
    f = pd.read_parquet(
        ROOT / "data" / "processed" / "game_features.parquet",
        columns=["game_id", "home_team", "away_team", "ats_margin"],
    )
    f = f[f["ats_margin"].notna()]
    rows = [
        pd.DataFrame(
            {"game_id": f["game_id"], "team": f[k], "ats": s * f["ats_margin"].to_numpy(float)}
        )
        for k, s in (("home_team", 1.0), ("away_team", -1.0))
    ]
    return pd.concat(rows, ignore_index=True).drop_duplicates(["game_id", "team"])


def nfl_table(season, path, ats):
    f = pd.read_parquet(path, columns=NFL_COLUMNS)
    f = f[f["season_type"].eq(NFL_REGULAR)].reset_index(drop=True)
    sides = f.groupby("game_id")[["week", "home_team", "away_team"]].first().reset_index()
    teams = pd.concat(
        [
            sides.rename(columns={"home_team": "team", "away_team": "opp"}),
            sides.rename(columns={"away_team": "team", "home_team": "opp"}),
        ],
        ignore_index=True,
    )[["game_id", "team", "opp", "week"]]
    run = (
        f["play_type"].eq(NFL_RUN)
        & f["qb_dropback"].fillna(0).ne(1)
        & f["qb_kneel"].fillna(0).ne(1)
        & f["yards_gained"].notna()
        & f["posteam"].notna()
        & f["defteam"].notna()
    )
    r = f[run].copy()
    r["stuff"] = r["yards_gained"].le(LINE_OF_SCRIMMAGE).astype(float)
    r["one"] = 1.0
    league = r.groupby("week")[["stuff", "one"]].sum()
    rate = league["stuff"] / league["one"]
    r["dev"] = r["stuff"] - r["week"].map(rate)
    o = r.groupby(["game_id", "posteam"])[["one", "dev"]].sum()
    o.columns = ["or_n", "or_d"]
    o.index = o.index.set_names(["game_id", "team"])
    d = r.groupby(["game_id", "defteam"])[["one", "dev"]].sum()
    d.columns = ["dr_n", "dr_d"]
    d.index = d.index.set_names(["game_id", "team"])
    t = pd.concat([o, d], axis=1).reset_index()
    m = teams.merge(t, on=["game_id", "team"], how="left").merge(
        ats, on=["game_id", "team"], how="left"
    )
    raw = pd.DataFrame(
        {"game_id": m["game_id"], "team": m["team"], "opp": m["opp"], "season": season}
    )
    raw["week"] = m["week"].to_numpy(float)
    for c in SUMS:
        raw[c] = m[c].fillna(0.0).to_numpy(float)
    raw["ats_n"] = m["ats"].notna().astype(float).to_numpy()
    raw["ats_v"] = m["ats"].fillna(0.0).to_numpy(float)
    diag = {
        "designed_runs": int(run.sum()),
        "league_stuff_share": float(r["stuff"].mean()),
        "week_rate_min_max": [float(rate.min()), float(rate.max())],
        "team_games": len(raw),
        "share_with_ats": float(raw["ats_n"].mean()),
        "mean_runs_per_team_game": float(raw["or_n"].mean()),
    }
    return raw, sides[["game_id", "home_team", "away_team"]], diag


def nfl_build():
    NFL_OUT.mkdir(parents=True, exist_ok=True)
    fl, u6n = nfl_modules()
    m36 = fl.load_mod36()
    files = fl.latest_plays()
    ats = nfl_ats()
    raws, sides, diag = [], [], {}
    for season in sorted(s for s in files if s <= m36.LAST_SEASON):
        r, s, dg = nfl_table(season, files[season], ats)
        raws.append(r)
        sides.append(s)
        diag[str(season)] = dg
    raw = pd.concat(raws, ignore_index=True)
    values = to_date(raw)
    span = list(range(m36.FIRST_SEASON, m36.LAST_SEASON + 1))
    seed = [m36.SEED, len(m36.TEMPLATES) + NFL_EARLIER_CONFIRMATIONS, u6n.ASOF_STREAM]
    check = asof_check(raw, values, span, seed)
    pop = values[values["season"].between(m36.FIRST_SEASON, m36.LAST_SEASON)]
    cols = [*NAMES, *RAW.values(), "ats_td"]
    report = {
        "seasons": diag,
        "as_of_check": check,
        "span_team_games": len(pop),
        "span_nonzero_share": {n: float(pop[n].ne(0).mean()) for n in NAMES},
        "span_sd": {n: float(pop[n].std(ddof=1)) for n in cols},
        "span_correlation": {n: pop[cols].corr().round(3)[n].to_dict() for n in cols},
    }
    (NFL_OUT / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("seasons", json.dumps({k: diag[k] for k in list(diag)[-6:]}))
    print("sd", {k: round(v, 4) for k, v in report["span_sd"].items()})
    print("corr", pop[cols].corr().round(3).to_string())
    print("asof", check, flush=True)
    if not check["max_abs_difference"] <= u3.TOLERANCE:
        raise SystemExit("as-of check failed; confirmation not run")
    values.to_parquet(NFL_OUT / "team_values.parquet", index=False)
    pd.concat(sides, ignore_index=True).to_parquet(NFL_OUT / "game_teams.parquet", index=False)


def nfl_cells(m36, name, res, power_lane):
    held = res["held_out"]
    lo, hi = res["coefficient_interval_per_sd"]
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"xlg14_run_stuff_nfl_{name}_{units}",
                "description": (
                    f"NFL confirmation of the college-first survivor {name}: designed runs "
                    "(play_type run, not a dropback or kneel) gaining zero or fewer yards, centred "
                    "on the league rate of each game's own week, per run over earlier weeks of "
                    "the season, residualised on own ATS margin to date; home minus away; one "
                    "offset-logistic coefficient per SD on the served out-of-season probability, "
                    f"opener grade {m36.FIRST_SEASON}-{m36.LAST_SEASON}, leave-one-season-out. "
                    f"Nonzero games {res['nonzero_games']}; changed picks {held['changed']}, "
                    f"record {held['arm_right']}-{held['ref_right']}; coefficient per SD "
                    f"{res['coefficient_per_sd']:+.4f} [{lo:+.4f}, {hi:+.4f}], positive in "
                    f"{res['folds_with_expected_sign']} of {res['folds']} folds (expected "
                    "positive, the college sign); this metric sits at percentile "
                    f"{res['null_percentile'][k]:.1f} of {res['draws']} within-season shuffles "
                    "of the same column; in-sample minus held-out "
                    f"{res['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(held[k]["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": m36.FIRST_SEASON,
                "season_end": m36.LAST_SEASON,
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "No resolved wrong sign (the coefficient's interval is not wholly negative), "
                    "split-half reliability of the trait not measured, and no positive control "
                    "bounds an effect this small: the NFL add-one harness shows about .26 logit "
                    f"per SD four times in five ({power_lane})."
                ),
                "notes": (
                    f"Family {FAMILY}: one of the three pre-declared NFL confirmations of the "
                    "college survivors; 3 gating reads (log-loss percentile in the matched null "
                    "with the full-sample coefficient sign), 3 recorded cells each; terms "
                    "declared in the lane before any outcome was joined; outcome: "
                    f"{res['outcome']}. Not served."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": int(res["folds"]),
                "plain_summary": (
                    f"Checks whether {NFL_PLAIN[name]} go on to beat the opening spread more "
                    f"often, as college teams did. {u6n_verdict(res['outcome'])}"
                ),
            }
        )
    return out


def u6n_verdict(outcome):
    import xlg14_u6_nfl_confirmation as u6n

    return u6n.VERDICTS[outcome]


def nfl_confirm():
    started = time.time()
    fl, u6n = nfl_modules()
    m36 = fl.load_mod36()
    pick, d = m36.games()
    values = pd.read_parquet(NFL_OUT / "team_values.parquet")
    teams = pd.read_parquet(NFL_OUT / "game_teams.parquet").drop_duplicates("game_id")
    cols = [*NAMES, "ats_td", "prior_games"]
    terms, both = u6n.joined(d, values, teams, cols)
    frame = d[["game_id", "season", "week", "home_covered", m36.SERVED_COLUMN]].copy()
    for n in NAMES:
        frame[n] = terms[n]
    results, batch = {}, []
    lines = [f"base {pick.relative_to(ROOT)} column {m36.SERVED_COLUMN}; games with both {both}"]
    for i, n in enumerate(NAMES):
        index = len(m36.TEMPLATES) + NFL_EARLIER_CONFIRMATIONS + i
        others = {
            "own_ats_to_date_diff": terms["ats_td"],
            **{f"other_{o}": terms[o] for o in NAMES if o != n},
        }
        res, ln = u6n.read_one(m36, frame, n, index, others)
        res["base_artifact"] = str(pick.relative_to(ROOT))
        results[n] = res
        lines += ln
        batch += nfl_cells(m36, n, res, u6n.POWER_LANE)
        print("\n".join(ln), f"{time.time() - started:.0f}s", flush=True)
    (NFL_OUT / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (NFL_OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (NFL_OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": batch}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    {"build": build, "screen": screen, "nfl_build": nfl_build, "nfl_confirm": nfl_confirm}[
        sys.argv[1]
    ]()
