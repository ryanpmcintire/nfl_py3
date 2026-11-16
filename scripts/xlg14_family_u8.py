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
OUT = ROOT / "artifacts" / "xlg14_family_u8"
LANE = "docs/lanes/xlg-14-u8-2026-11-16.md"
POWER_LANE = u3.POWER_LANE
FAMILY = "xlg14_special_teams_family_u8"
STREAM = 8
PUNT_TYPE = "Punt"
MIN_FIT_ROWS = u7.MIN_FIT_ROWS
PLAY_COLUMNS = ["game_id", "week", "seasonType", "pos_team_id", "def_pos_team_id", "type.text", "EPA"]
SUMS = ["pk_n", "pk_e", "pr_n", "pr_e"]
GAME_COLUMNS = [*SUMS, "ats_v", "ats_n"]
RAW = {"punt_cover_oe": "cover_raw", "punt_return_oe": "return_raw", "punt_unit_net_oe": "net_raw"}
CANDIDATES = {
    "punt_cover_oe": {
        "hypothesis": "positive",
        "what": (
            "mean EPA per own punt over earlier weeks (punting team's view; blocks, muffs and "
            "return touchdowns excluded), residualised on the team's own ATS margin to date."
        ),
        "plain": "gain more field position than expected when they punt",
    },
    "punt_return_oe": {
        "hypothesis": "positive",
        "what": (
            "minus the mean EPA per opponent punt received over earlier weeks, residualised on "
            "the team's own ATS margin to date."
        ),
        "plain": "give up less field position than expected when opponents punt to them",
    },
    "punt_unit_net_oe": {
        "hypothesis": "positive",
        "what": (
            "the sum of the punt-coverage and punt-return field-position reads, residualised on "
            "the team's own ATS margin to date."
        ),
        "plain": "win the punting exchange on field position",
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
    f["epa"] = pd.to_numeric(f["EPA"], errors="coerce")
    p = f[f["type.text"].eq(PUNT_TYPE) & f["epa"].notna()].copy()
    p["one"] = 1.0
    k = p.groupby(["game_id", "pos"])[["one", "epa"]].sum()
    k.columns = ["pk_n", "pk_e"]
    k.index = k.index.set_names(["game_id", "team"])
    r = p.groupby(["game_id", "dfn"])[["one", "epa"]].sum()
    r.columns = ["pr_n", "pr_e"]
    r.index = r.index.set_names(["game_id", "team"])
    t = pd.concat([k, r], axis=1).reset_index()
    m = teams.merge(t, on=["game_id", "team"], how="left").merge(ats, on=["game_id", "team"], how="left")
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
        "punts": len(p),
        "punt_epa_mean": float(p["epa"].mean()),
        "team_games": len(raw),
        "share_with_opener": float(raw["ats_n"].mean()),
        "mean_punts_per_team_game": float(raw["pk_n"].mean()),
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
    st["cover_raw"] = (pr["pk_e"] / pr["pk_n"].where(pr["pk_n"] > 0)).fillna(0.0)
    st["return_raw"] = (-pr["pr_e"] / pr["pr_n"].where(pr["pr_n"] > 0)).fillna(0.0)
    st["net_raw"] = st["cover_raw"] + st["return_raw"]
    st["ats_td"] = (pr["ats_v"] / pr["ats_n"].where(pr["ats_n"] > 0)).fillna(0.0)
    st["has_ats"] = pr["ats_n"] > 0
    st["prior_games"] = pr["games"]
    out = r[["game_id", "team", "season", "week"]].merge(st, on=["team", "season", "week"], how="left")
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
    ats = u7.ats_rows()
    raws, diag, skipped = [], {}, []
    for season in sorted(s for s in parts if s <= last):
        have = set(pq.ParquetFile(parts[season]).schema_arrow.names)
        if not set(PLAY_COLUMNS) <= have:
            skipped.append(int(season))
            continue
        r, d = season_tables(int(season), parts[season], ats)
        raws.append(r)
        diag[str(season)] = d
    raw = pd.concat(raws, ignore_index=True)
    values = to_date(raw)
    check = asof_check(raw, values)
    pop = values[values["season"].isin(x14.u1.SEASONS)]
    cols = [*NAMES, *RAW.values(), "ats_td"]
    corr = pop[cols].corr().round(3)
    report = {
        "family": FAMILY,
        "skipped_seasons_missing_columns": skipped,
        "population_team_games": len(pop),
        "population_nonzero_share": {n: float(pop[n].ne(0).mean()) for n in NAMES},
        "population_sd": {n: float(pop[n].std(ddof=1)) for n in cols},
        "population_correlation": {n: corr[n].to_dict() for n in cols},
        "asof_check": check,
        "per_season": diag,
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("skipped", skipped)
    print(
        "punts",
        {s: (d["punts"], round(d["punt_epa_mean"], 3), round(d["mean_punts_per_team_game"], 2)) for s, d in diag.items()},
    )
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
                "name": f"xlg14_special_teams_cfb_{name}_{units}",
                "description": (
                    f"College-first special-teams punt-unit quality family, {name}: {spec['what']} "
                    "Built from strictly earlier weeks; the residualising slope from team-games "
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
                    "No resolved wrong sign for the declared punt-unit quality mechanism, "
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


if __name__ == "__main__":
    {"build": build, "screen": screen}[sys.argv[1]]()
