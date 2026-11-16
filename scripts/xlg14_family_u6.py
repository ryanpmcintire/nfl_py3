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

import xlg14_luck_family as u3  # noqa: E402

from nfl_ats.cfb_features import cfb_season_partitions  # noqa: E402

x14 = u3.x14
OUT = ROOT / "artifacts" / "xlg14_family_u6"
LANE = "docs/lanes/xlg-14-u6-2026-11-16.md"
POWER_LANE = u3.POWER_LANE
FAMILY = "xlg14_schedule_strength_family_u6"
STREAM = 6
PLAY_COLUMNS = [
    "game_id",
    "week",
    "seasonType",
    "pos_team_id",
    "def_pos_team_id",
    "rush",
    "pass",
    "kneel_down",
    "EPA",
    "pos_team_score",
    "def_pos_team_score",
]
GAME_COLUMNS = ["o_n", "o_epa", "d_n", "d_epa", "margin"]
CANDIDATES = {
    "off_sos_gap": {
        "hypothesis": "positive",
        "what": (
            "minus the mean, over the team's earlier games, of each opponent's allowed EPA per "
            "play to date minus the league rate to date (offence that faced stingy defences)."
        ),
        "plain": "have put up their offense against tougher defenses than most",
    },
    "def_sos_gap": {
        "hypothesis": "positive",
        "what": (
            "the mean, over the team's earlier games, of each opponent's offensive EPA per play "
            "to date minus the league rate to date (defence that faced strong offences)."
        ),
        "plain": "have played their defense against stronger offenses than most",
    },
    "margin_sos_gap": {
        "hypothesis": "positive",
        "what": (
            "the mean, over the team's earlier games, of each opponent's points margin per game "
            "to date (a strength-of-schedule margin adjustment)."
        ),
        "plain": "have built their record against a harder schedule",
    },
    "last_opponent_strength": {
        "hypothesis": "positive",
        "what": "the most recent earlier opponent's points margin per game to date.",
        "plain": "just played a strong opponent",
    },
}
NAMES = list(CANDIDATES)


def season_tables(season, path):
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
    f["ps"] = pd.to_numeric(f["pos_team_score"], errors="coerce")
    f["ds"] = pd.to_numeric(f["def_pos_team_score"], errors="coerce")
    a = f.groupby(["game_id", "pos"])["ps"].max()
    b = f.groupby(["game_id", "dfn"])["ds"].max()
    b.index = b.index.set_names(["game_id", "pos"])
    pts = pd.concat([a, b], axis=1).max(axis=1)
    pts.index = pts.index.set_names(["game_id", "team"])
    scrim = pd.to_numeric(f["rush"], errors="coerce").eq(1) | pd.to_numeric(
        f["pass"], errors="coerce"
    ).eq(1)
    kneel = pd.to_numeric(f["kneel_down"], errors="coerce").eq(1)
    f["epa"] = pd.to_numeric(f["EPA"], errors="coerce")
    p = f[scrim & ~kneel & f["epa"].notna()].copy()
    p["n"] = 1.0
    o = p.groupby(["game_id", "pos"])[["n", "epa"]].sum()
    o.index = o.index.set_names(["game_id", "team"])
    d = p.groupby(["game_id", "dfn"])[["n", "epa"]].sum()
    d.index = d.index.set_names(["game_id", "team"])
    t = pd.concat([o.add_prefix("o_"), d.add_prefix("d_"), pts.rename("pts")], axis=1)
    t = t.reset_index()
    m = teams.merge(t, on=["game_id", "team"], how="left")
    own = m.set_index(["game_id", "team"])["pts"]
    key = pd.MultiIndex.from_arrays([m["game_id"], m["opp"]])
    m["opp_pts"] = own.reindex(key).to_numpy()
    raw = pd.DataFrame(
        {
            "game_id": m["game_id"].astype("int64"),
            "team": m["team"].astype("int64"),
            "opp": m["opp"].astype("int64"),
            "season": season,
        }
    )
    raw["week"] = m["game_id"].map(week).to_numpy(float)
    for c in ["o_n", "o_epa", "d_n", "d_epa"]:
        raw[c] = m[c].fillna(0.0).to_numpy(float)
    raw["margin"] = (m["pts"] - m["opp_pts"]).fillna(0.0).to_numpy(float)
    diag = {
        "plays": len(f),
        "scrimmage_plays": len(p),
        "team_games": len(raw),
        "share_score_present": float((m["pts"].notna() & m["opp_pts"].notna()).mean()),
        "team_games_without_offense": int(raw["o_n"].eq(0).sum()),
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
    prior = grid.groupby(["team", "season"])[cols].cumsum() - grid[cols]
    st = grid[["team", "season", "week"]].copy()
    st["off_rate"] = prior["o_epa"] / prior["o_n"].where(prior["o_n"] > 0)
    st["def_rate"] = prior["d_epa"] / prior["d_n"].where(prior["d_n"] > 0)
    st["mpg"] = prior["margin"] / prior["games"].where(prior["games"] > 0)
    lw = r.groupby(["season", "week"])[["o_n", "o_epa"]].sum().sort_index()
    lc = lw.groupby(level="season").cumsum() - lw
    league = (lc["o_epa"] / lc["o_n"].where(lc["o_n"] > 0)).rename("lrate").reset_index()
    st = st.merge(league, on=["season", "week"], how="left")
    tgt = r[["game_id", "team", "season", "week"]]
    e = tgt.merge(
        r[["team", "season", "opp", "week"]].rename(columns={"week": "wk"}),
        on=["team", "season"],
    )
    e = e[e["wk"] < e["week"]]
    e = e.merge(st.rename(columns={"team": "opp"}), on=["opp", "season", "week"], how="left")
    e["off_sos_gap"] = -(e["def_rate"] - e["lrate"])
    e["def_sos_gap"] = e["off_rate"] - e["lrate"]
    e["margin_sos_gap"] = e["mpg"]
    g = e.groupby(["game_id", "team"])
    agg = g[["off_sos_gap", "def_sos_gap", "margin_sos_gap"]].mean()
    agg["prior_games"] = g.size()
    last = e.sort_values("wk").groupby(["game_id", "team"]).tail(1)
    agg["last_opponent_strength"] = last.set_index(["game_id", "team"])["mpg"]
    out = tgt.merge(agg.reset_index(), on=["game_id", "team"], how="left")
    out[NAMES] = out[NAMES].fillna(0.0)
    out["prior_games"] = out["prior_games"].fillna(0.0)
    return out


def asof_check(raw, values):
    rng = np.random.default_rng([u3.ASOF_SEED, STREAM, u3.ASOF_STREAM])
    pool = values[values["season"].isin(x14.u1.SEASONS)][["season", "week"]].drop_duplicates()
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
    last = max(x14.u1.SEASONS)
    raws, diag, skipped = [], {}, []
    for season in sorted(s for s in parts if s <= last):
        have = set(pq.ParquetFile(parts[season]).schema_arrow.names)
        if not set(PLAY_COLUMNS) <= have:
            skipped.append(int(season))
            continue
        r, d = season_tables(int(season), parts[season])
        raws.append(r)
        diag[str(season)] = d
    raw = pd.concat(raws, ignore_index=True)
    values = to_date(raw)
    check = asof_check(raw, values)
    pop = values[values["season"].isin(x14.u1.SEASONS)]
    corr = pop[NAMES].corr().round(3)
    report = {
        "family": FAMILY,
        "skipped_seasons_missing_columns": skipped,
        "population_team_games": len(pop),
        "population_nonzero_share": {n: float(pop[n].ne(0).mean()) for n in NAMES},
        "population_sd": {n: float(pop[n].std(ddof=1)) for n in NAMES},
        "population_correlation": {n: corr[n].to_dict() for n in NAMES},
        "asof_check": check,
        "per_season": diag,
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("skipped", skipped)
    print("score share", {s: round(d["share_score_present"], 3) for s, d in diag.items()})
    print("nonzero", report["population_nonzero_share"])
    print("sd", {k: round(v, 4) for k, v in report["population_sd"].items()})
    print("corr", corr.to_string())
    print("asof", check, flush=True)
    if not check["max_abs_difference"] <= u3.TOLERANCE:
        raise SystemExit("as-of check failed; screen not run")
    values.to_parquet(OUT / "team_values.parquet", index=False)


def joined(d, values):
    look = values.set_index(["game_id", "team"])
    gid = pd.to_numeric(d["game_id"]).astype("int64")
    side = {}
    for s in ("home", "away"):
        key = pd.MultiIndex.from_arrays([gid, pd.to_numeric(d[f"{s}_id"]).astype("int64")])
        side[s] = look[[*NAMES, "prior_games"]].reindex(key).reset_index(drop=True)
    both = side["home"]["prior_games"].notna() & side["away"]["prior_games"].notna()
    out = pd.DataFrame(index=d.index)
    for n in NAMES:
        diff = side["home"][n].to_numpy() - side["away"][n].to_numpy()
        out[n] = np.where(both.to_numpy(), diff, 0.0)
    return out, int(both.sum())


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
                "name": f"xlg14_schedule_strength_cfb_{name}_{units}",
                "description": (
                    f"College-first schedule-strength family, {name}: {spec['what']} "
                    "Opponent strength from each opponent's games in strictly earlier weeks "
                    "than the target game, home minus away; one offset-logistic coefficient "
                    "on a leave-one-season-out "
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
                    "No resolved wrong sign for the declared schedule-strength mechanism, "
                    "split-half reliability of the trait not measured, and no positive "
                    "control bounds an "
                    "effect this small: the college add-one harness shows about .08 to .11 "
                    f"logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: 4 candidates x 3 metrics = 12 recorded cells; 4 gating "
                    "reads (log-loss percentile in the matched null), 8 season-half "
                    "coefficients and 2 control reads; survivor rule declared in the lane "
                    f"before any outcome was joined; survivor: {survived}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    f"Checks whether college teams that {spec['plain']} go on to beat the "
                    f"opening spread more or less often. {verdict}"
                ),
            }
        )
    return out


def screen():
    started = time.time()
    u3.STREAM_LUCK = STREAM
    hp = u3.load_power()
    m36 = hp.M36
    d = hp.x14.cfb_frame()
    values = pd.read_parquet(OUT / "team_values.parquet")
    terms, both = joined(d, values)
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
    out = {
        "family": FAMILY,
        "games": len(frame),
        "games_with_both_teams_in_play_by_play": both,
        "seasons": seasons,
        "draws": hp.DRAWS,
        "seed": [hp.SEED, STREAM],
        "candidates": {},
    }
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


if __name__ == "__main__":
    {"build": build, "screen": screen}[sys.argv[1]]()
