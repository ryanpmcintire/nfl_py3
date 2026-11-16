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
OUT = ROOT / "artifacts" / "xlg14_family_u4"
LANE = "docs/lanes/xlg-14-xlg14-cfb-first-discovery.md"
POWER_LANE = u3.POWER_LANE
FAMILY = "xlg14_scoreboard_family_u4"
STREAM = 4
LAST_REGULATION_PERIOD_BEFORE_Q4 = 3
PLAY_COLUMNS = [
    "game_id",
    "week",
    "seasonType",
    "pos_team_id",
    "def_pos_team_id",
    "period",
    "pos_team_score",
    "def_pos_team_score",
    "EPA",
    "EPA_success",
]
GAME_COLUMNS = ["margin", "margin3", "epa_m", "sr_m"]
CANDIDATES = {
    "last_game_epa_gap": {
        "what": (
            "the previous game's points margin minus the margin its EPA margin predicts, "
            "slope and intercept fitted on team-games in strictly earlier seasons."
        ),
        "plain": "won their last game by more than their play-by-play quality said they should",
    },
    "last_game_success_gap": {
        "what": (
            "the previous game's points margin minus the margin its success-rate margin "
            "predicts, slope and intercept fitted on team-games in strictly earlier seasons."
        ),
        "plain": "won their last game by more than their share of successful plays said they should",
    },
    "season_epa_gap": {
        "what": (
            "the per-game mean, over earlier weeks of the season, of points margin minus the "
            "margin the EPA margin predicts (earlier-season fit)."
        ),
        "plain": "have outscored their play-by-play quality so far this season",
    },
    "last_game_q4_margin": {
        "what": "the previous game's points margin from the fourth period on.",
        "plain": "won the fourth quarter of their last game by a lot",
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
    f["ps"] = pd.to_numeric(f["pos_team_score"], errors="coerce")
    f["ds"] = pd.to_numeric(f["def_pos_team_score"], errors="coerce")
    f["epa"] = pd.to_numeric(f["EPA"], errors="coerce")
    f["sr"] = pd.to_numeric(f["EPA_success"], errors="coerce")
    f["early"] = pd.to_numeric(f["period"], errors="coerce").le(LAST_REGULATION_PERIOD_BEFORE_Q4)
    pairs = f[["game_id", "pos", "dfn"]].drop_duplicates()
    pairs = pairs[pairs["pos"].ne(pairs["dfn"])]
    teams = pd.concat(
        [pairs.rename(columns={"pos": "team", "dfn": "opp"}), pairs.rename(columns={"dfn": "team", "pos": "opp"})]
    ).drop_duplicates()
    count = teams.groupby("game_id")["team"].transform("nunique")
    teams = teams[count.eq(2)].drop_duplicates(["game_id", "team"]).reset_index(drop=True)

    def score(rows):
        a = rows.groupby(["game_id", "pos"])["ps"].max()
        b = rows.groupby(["game_id", "dfn"])["ds"].max()
        b.index = b.index.set_names(["game_id", "pos"])
        return pd.concat([a, b], axis=1).max(axis=1)

    pts = score(f).rename("pts")
    pts3 = score(f[f["early"]]).rename("pts3")
    off = f.groupby(["game_id", "pos"]).agg(epa=("epa", "sum"), epa_n=("epa", "count"), sr=("sr", "mean"))
    t = pd.concat([pts, pts3, off], axis=1)
    t.index = t.index.set_names(["game_id", "team"])
    t = t.reset_index()
    t["pts3"] = t["pts3"].fillna(0.0)
    t.loc[t["epa_n"].fillna(0).eq(0), "epa"] = np.nan
    m = teams.merge(t, on=["game_id", "team"], how="left").merge(
        t.rename(columns={"team": "opp"}), on=["game_id", "opp"], how="left", suffixes=("", "_o")
    )
    week = f.groupby("game_id")["week"].first()
    raw = pd.DataFrame(
        {
            "game_id": m["game_id"].astype("int64"),
            "team": m["team"].astype("int64"),
            "season": season,
            "week": m["game_id"].map(week).to_numpy(float),
            "margin": m["pts"] - m["pts_o"],
            "margin3": m["pts3"] - m["pts3_o"],
            "epa_m": m["epa"] - m["epa_o"],
            "sr_m": m["sr"] - m["sr_o"],
        }
    )
    diag = {
        "plays": len(f),
        "team_games": len(raw),
        "missing_margin": int(raw["margin"].isna().sum()),
        "missing_epa_margin": int(raw["epa_m"].isna().sum()),
        "missing_success_margin": int(raw["sr_m"].isna().sum()),
        "share_epa_play_nonnull": float(f["epa"].notna().mean()),
    }
    return raw, diag


def season_fits(raw):
    fits = {}
    seasons = sorted(raw["season"].unique())
    for x in ("epa_m", "sr_m"):
        rows = {}
        for s in seasons:
            e = raw[(raw["season"] < s)][["margin", x]].dropna()
            if len(e) < 2 or e[x].var() == 0:
                continue
            b = float(np.cov(e[x], e["margin"], ddof=1)[0, 1] / e[x].var(ddof=1))
            rows[int(s)] = (float(e["margin"].mean() - b * e[x].mean()), b)
        fits[x] = rows
    return fits


def to_date(raw, fits=None):
    fits = season_fits(raw) if fits is None else fits
    r = raw.reset_index(drop=True).copy()
    for x, name in (("epa_m", "epa_gap"), ("sr_m", "sr_gap")):
        a = r["season"].map({s: v[0] for s, v in fits[x].items()})
        b = r["season"].map({s: v[1] for s, v in fits[x].items()})
        r[name] = r["margin"] - (a + b * r[x])
    r["q4"] = r["margin"] - r["margin3"]
    r["epa_n"] = r["epa_gap"].notna().astype(float)
    r["games"] = 1.0
    cols = ["epa_gap", "sr_gap", "q4"]
    tw = r.groupby(["team", "season", "week"]).agg(
        epa_gap=("epa_gap", "mean"),
        sr_gap=("sr_gap", "mean"),
        q4=("q4", "mean"),
        epa_sum=("epa_gap", "sum"),
        epa_n=("epa_n", "sum"),
        games=("games", "sum"),
    ).reset_index()
    g = tw.groupby(["team", "season"])
    last = g[cols].shift(1)
    prior = g[["epa_sum", "epa_n", "games"]].cumsum() - tw[["epa_sum", "epa_n", "games"]]
    out = tw[["team", "season", "week"]].copy()
    out["last_game_epa_gap"] = last["epa_gap"].fillna(0.0)
    out["last_game_success_gap"] = last["sr_gap"].fillna(0.0)
    out["season_epa_gap"] = (prior["epa_sum"] / prior["epa_n"].where(prior["epa_n"] > 0)).fillna(0.0)
    out["last_game_q4_margin"] = last["q4"].fillna(0.0)
    out["prior_games"] = prior["games"]
    return r[["game_id", "team", "season", "week"]].merge(out, on=["team", "season", "week"], how="left")


def asof_check(raw, fits, values):
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
    fits = season_fits(raw)
    values = to_date(raw, fits)
    check = asof_check(raw, fits, values)
    pop = values[values["season"].isin(x14.u1.SEASONS)]
    corr = pop[NAMES].corr().round(3)
    report = {
        "family": FAMILY,
        "skipped_seasons_missing_columns": skipped,
        "fits_intercept_slope_by_season": {
            k: {str(s): [round(a, 4), round(b, 4)] for s, (a, b) in v.items()} for k, v in fits.items()
        },
        "population_team_games": len(pop),
        "population_nonzero_share": {n: float(pop[n].ne(0).mean()) for n in NAMES},
        "population_sd": {n: float(pop[n].std(ddof=1)) for n in NAMES},
        "population_correlation": {n: corr[n].to_dict() for n in NAMES},
        "asof_check": check,
        "per_season": diag,
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("skipped", skipped)
    print("fits", report["fits_intercept_slope_by_season"])
    print("missing", {s: [d["missing_margin"], d["missing_epa_margin"], d["missing_success_margin"]] for s, d in diag.items()})
    print("nonzero", report["population_nonzero_share"])
    print("sd", {k: round(v, 3) for k, v in report["population_sd"].items()})
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
                "name": f"xlg14_scoreboard_cfb_{name}_{units}",
                "description": (
                    f"College-first scoreboard-over-play-quality family, {name}: {spec['what']} "
                    "Team value from games in strictly earlier weeks of the same season, home "
                    "minus away; one offset-logistic coefficient on a leave-one-season-out ridge "
                    f"cover probability with the opener as input and grade, {len(seasons)} "
                    f"seasons. Changed picks {held['changed']}, record {held['arm_right']}-"
                    f"{held['ref_right']}; coefficient per SD {res['coefficient_per_sd']:+.4f}; "
                    f"earlier and later season-half coefficients {early:+.4f} and {late:+.4f}; "
                    f"this metric sits at percentile {res['null_percentile'][k]:.1f} of {draws} "
                    "within-season shuffles of the same column; in-sample minus held-out "
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
                    "No resolved wrong sign for the declared over-reaction mechanism, split-half "
                    "reliability of the trait not measured, and no positive control bounds an "
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
    (OUT / "weak_signals_batch.json").write_text(json.dumps({"cells": batch}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"build": build, "screen": screen}[sys.argv[1]]()
