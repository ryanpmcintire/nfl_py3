import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xlg14_fumble_luck_nfl_confirmation import (
    coefficient_draws,
    controls,
    latest_plays,
    load_mod36,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
OUT = ROOT / "artifacts" / "xlg14_family_u4" / "nfl_confirmation"
LANE = "docs/lanes/xlg-14-xlg14-cfb-first-discovery.md"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "xlg14_scoreboard_family_u4"
NAME = "season_epa_gap"
EXPECTED_SIGN = 1.0
EARLIER_CONFIRMATIONS = 1
REGULAR_SEASON = "REG"
PLAY_COLUMNS = ["game_id", "season_type", "week", "home_team", "away_team", "posteam", "epa"]
SCORE_COLUMNS = ["game_id", "home_score", "away_score"]
EPA_INPUTS = [
    "diff_off_epa_per_play",
    "diff_def_epa_per_play",
    "diff_off_pass_epa_per_play",
    "diff_off_rush_epa_per_play",
    "diff_def_pass_epa_per_play",
    "diff_def_rush_epa_per_play",
]
KEYS = ["game_id", "team", "season", "week"]
OUTCOMES = ["margin", "epa_margin"]
SIDES = ("home", "away")
TEAMS_PER_GAME = len(SIDES)
NULL_STREAM = 0
ASOF_STREAM = 1
ASOF_CUTOFFS = 5
TOLERANCE = float(np.sqrt(np.finfo(float).eps))
WHAT = (
    "points margin minus the margin predicted from play-by-play EPA margin (least squares on "
    "team-games in strictly earlier seasons), per game over the team's games in strictly "
    "earlier weeks of the same season, regular season"
)
VERDICTS = {
    "confirmed": (
        "The pro games showed the same pattern clearly enough to keep watching it; it does not "
        "change any pick."
    ),
    "not confirmed": (
        "The pro games did not show the pattern clearly enough to tell it from chance; nothing "
        "is ruled out and no pick changes."
    ),
    "opposite sign": (
        "The pro games leaned the other way, though not clearly enough to rule the idea out; no "
        "pick changes."
    ),
}


def season_table(season, path, scores):
    f = pd.read_parquet(path, columns=PLAY_COLUMNS)
    f = f[f["season_type"].eq(REGULAR_SEASON) & f["posteam"].notna() & f["epa"].notna()]
    sides = f.groupby("game_id")[["week", "home_team", "away_team"]].first().reset_index()
    own = f.groupby(["game_id", "posteam"])["epa"].sum()
    teams_seen = f.groupby("game_id")["posteam"].nunique()
    if not teams_seen.eq(TEAMS_PER_GAME).all():
        raise SystemExit(f"season {season}: a game without exactly two teams in possession")
    g = sides.merge(scores, on="game_id", how="left", validate="one_to_one")
    if g[["home_score", "away_score"]].isna().any().any():
        raise SystemExit(f"season {season}: a play-by-play game without a final score")
    home_epa = own.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["home_team"]])).to_numpy()
    away_epa = own.reindex(pd.MultiIndex.from_arrays([g["game_id"], g["away_team"]])).to_numpy()
    if np.isnan(home_epa).any() or np.isnan(away_epa).any():
        raise SystemExit(f"season {season}: home or away team missing from possession plays")
    margin = (g["home_score"] - g["away_score"]).to_numpy(float)
    rows = []
    for team, sign in (("home_team", 1.0), ("away_team", -1.0)):
        rows.append(
            pd.DataFrame(
                {
                    "game_id": g["game_id"],
                    "team": g[team],
                    "season": season,
                    "week": g["week"].to_numpy(float),
                    "margin": sign * margin,
                    "epa_margin": sign * (home_epa - away_epa),
                }
            )
        )
    raw = pd.concat(rows, ignore_index=True)
    diag = {
        "source": str(path.relative_to(ROOT)),
        "plays": len(f),
        "games": len(g),
        "team_games": len(raw),
        "weeks": [int(raw["week"].min()), int(raw["week"].max())],
    }
    return raw, g[["game_id", "home_team", "away_team"]], diag


def fits(raw):
    out = {}
    for s in sorted(raw["season"].unique()):
        e = raw[raw["season"] < s]
        if len(e):
            slope, intercept = np.polyfit(e["epa_margin"], e["margin"], 1)
            out[int(s)] = (float(slope), float(intercept))
    return out


def to_date(raw):
    fit = fits(raw)
    r = raw[raw["season"].isin(list(fit))].reset_index(drop=True)
    slope = r["season"].map({s: v[0] for s, v in fit.items()})
    intercept = r["season"].map({s: v[1] for s, v in fit.items()})
    v = r[KEYS].copy()
    v["gap"] = r["margin"] - (intercept + slope * r["epa_margin"])
    v["games"] = 1.0
    tw = v.groupby(["team", "season", "week"])[["gap", "games"]].sum().reset_index()
    prior = tw.groupby(["team", "season"])[["gap", "games"]].cumsum() - tw[["gap", "games"]]
    out = tw[["team", "season", "week"]].copy()
    played = prior["games"].where(prior["games"] > 0)
    out[NAME] = (prior["gap"] / played).fillna(0.0)
    out["prior_games"] = prior["games"]
    return v[KEYS].merge(out, on=["team", "season", "week"], how="left"), fit


def asof_check(seed, raw, values, first, last):
    rng = np.random.default_rng(seed)
    pool = values.loc[values["season"].between(first, last), ["season", "week"]]
    picks = pool.drop_duplicates().sample(ASOF_CUTOFFS, random_state=rng)
    worst, rows, cutoffs = 0.0, 0, []
    key = ["game_id", "team"]
    for s, w in picks.itertuples(index=False):
        r2 = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["week"] <= w))].copy()
        at = (r2["season"] == s) & (r2["week"] == w)
        r2.loc[at, OUTCOMES] = 0.0
        v2, _ = to_date(r2)
        a = values[(values["season"] == s) & (values["week"] == w)].set_index(key)[NAME]
        b = v2[(v2["season"] == s) & (v2["week"] == w)].set_index(key)[NAME].reindex(a.index)
        gap = (a - b).abs().to_numpy()
        worst = max(worst, float(np.nanmax(gap)) if gap.size else 0.0)
        worst = float("inf") if np.isnan(gap).any() else worst
        rows += len(a)
        cutoffs.append([int(s), int(w), len(a), int((a != 0).sum())])
    return {
        "cutoffs_season_week_teamgames_nonzero": cutoffs,
        "team_games": rows,
        "max_abs_difference": worst,
        "tolerance": TOLERANCE,
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    m36 = load_mod36()
    files = latest_plays()
    scores = pd.read_parquet(FEATURES, columns=SCORE_COLUMNS)
    raws, sides, diag = [], [], {}
    for season in sorted(s for s in files if s <= m36.LAST_SEASON):
        raw, side, d = season_table(season, files[season], scores)
        raws.append(raw)
        sides.append(side)
        diag[str(season)] = d
        print(season, d, flush=True)
    raw = pd.concat(raws, ignore_index=True)
    values, fit = to_date(raw)
    index = len(m36.TEMPLATES) + EARLIER_CONFIRMATIONS
    check = asof_check(
        [m36.SEED, index, ASOF_STREAM], raw, values, m36.FIRST_SEASON, m36.LAST_SEASON
    )
    print("as-of", check, flush=True)
    if not check["max_abs_difference"] <= TOLERANCE:
        (OUT / "build.json").write_text(
            json.dumps({"as_of_check": check, "failed": True}, indent=1), encoding="utf-8"
        )
        raise SystemExit("as-of check failed")
    print("fits", {s: fit[s] for s in fit if s >= m36.FIRST_SEASON}, flush=True)
    values.to_parquet(OUT / "team_values.parquet", index=False)
    raw.to_parquet(OUT / "team_game_margins.parquet", index=False)
    pd.concat(sides, ignore_index=True).to_parquet(OUT / "game_teams.parquet", index=False)
    span = values[values["season"].between(m36.FIRST_SEASON, m36.LAST_SEASON)]
    shape = {
        "team_games": len(span),
        "nonzero_team_games": int((span[NAME] != 0).sum()),
        "sd": float(span[NAME].std(ddof=1)),
        "mean": float(span[NAME].mean()),
    }
    print("values", shape, flush=True)
    out = {
        "candidate": NAME,
        "what": WHAT,
        "seasons": diag,
        "fits_slope_intercept": {str(s): v for s, v in fit.items()},
        "as_of_check": check,
        "values": shape,
    }
    (OUT / "build.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def joined(d, values, teams):
    look = values.set_index(["game_id", "team"])[[NAME, "prior_games"]]
    g = d[["game_id"]].merge(teams, on="game_id", how="left", validate="one_to_one")
    side = {}
    for s in SIDES:
        key = pd.MultiIndex.from_arrays([g["game_id"], g[f"{s}_team"]])
        side[s] = look.reindex(key).reset_index(drop=True)
    both = side["home"]["prior_games"].notna() & side["away"]["prior_games"].notna()
    diff = side["home"][NAME].to_numpy() - side["away"][NAME].to_numpy()
    return np.where(both.to_numpy(), diff, 0.0), int(both.sum())


def correlations(frame, z):
    f = frame.merge(
        pd.read_parquet(FEATURES, columns=["game_id", *EPA_INPUTS]),
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    x = f[NAME].to_numpy(float)
    nz = x != 0
    out = {}
    for name, col in [("served_logit", z), *((c, f[c].to_numpy(float)) for c in EPA_INPUTS)]:
        ok = ~np.isnan(col)
        sub = ok & nz
        out[name] = {
            "all": float(np.corrcoef(x[ok], col[ok])[0, 1]),
            "nonzero": float(np.corrcoef(x[sub], col[sub])[0, 1]),
            "games": int(ok.sum()),
        }
    return out


def outcome_of(coef, pct, bar):
    if coef * EXPECTED_SIGN > 0:
        return "confirmed" if pct > bar else "not confirmed"
    return "opposite sign"


def cells(m36, res):
    held = res["held_out"]
    lo, hi = res["coefficient_interval_per_sd"]
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"xlg14_scoreboard_nfl_{NAME}_{units}",
                "description": (
                    f"NFL confirmation of the college-first survivor {NAME}: {WHAT}. Home minus "
                    "away; one offset-logistic coefficient per SD on the served out-of-season "
                    f"probability, opener grade {m36.FIRST_SEASON}-{m36.LAST_SEASON}, "
                    f"leave-one-season-out. Nonzero games {res['nonzero_games']}; changed picks "
                    f"{held['changed']}, record {held['arm_right']}-{held['ref_right']}; "
                    f"coefficient per SD {res['coefficient_per_sd']:+.4f} [{lo:+.4f}, "
                    f"{hi:+.4f}], positive in {res['folds_with_expected_sign']} of "
                    f"{res['folds']} folds (expected positive); this metric sits at percentile "
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
                    "No resolved wrong sign (the coefficient's interval is not wholly on the "
                    "negative side), split-half reliability of the trait not measured, and no "
                    "positive control bounds an effect this small: the NFL add-one harness "
                    f"shows about .26 logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: the one pre-declared NFL confirmation of the college "
                    "survivor; 1 gating read (log-loss percentile in the matched null with the "
                    "full-sample coefficient sign), 3 recorded cells with accuracy and Brier "
                    "secondary, 3 control reads; terms declared in the lane before any outcome "
                    f"was joined; outcome: {res['outcome']}. Not served."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": int(res["folds"]),
                "plain_summary": (
                    "Checks whether pro teams that have outscored what their play-by-play "
                    "efficiency would predict so far this season go on to beat the opening "
                    f"spread more often, as college teams appeared to. {VERDICTS[res['outcome']]}"
                ),
            }
        )
    return out


def confirm():
    started = time.time()
    m36 = load_mod36()
    pick, d = m36.games()
    values = pd.read_parquet(OUT / "team_values.parquet")
    teams = pd.read_parquet(OUT / "game_teams.parquet")
    term, both = joined(d, values, teams)
    frame = d[["game_id", "season", "week", "home_covered", m36.SERVED_COLUMN]].copy()
    frame[NAME] = term
    index = len(m36.TEMPLATES) + EARLIER_CONFIRMATIONS
    tpl = {
        "name": NAME,
        "column": NAME,
        "intercept": False,
        "matched": "served",
        "reported": False,
        "what": "how far each team's scoring so far has run ahead of its play efficiency",
    }
    st = m36.setup(frame, tpl, index)
    real = m36.real_read(st, tpl)
    held = real["served"]
    y, z = st["y"], st["z"]
    xd = m36.design(st["xs"], False)
    base = st["refs"]["served"]["m"]
    full = m36.fit_offset(xd, y, z)
    inside = ((m36.metric_matrix(expit(z + xd @ full), y) - base) * m36.DIRECTION).mean(axis=0)
    rng = np.random.default_rng([m36.SEED, index, NULL_STREAM])
    rows = []
    for i in range(m36.DRAWS):
        m36.draw_rows(st, tpl, "null", 0.0, i, m36.permuted(st, rng), rows)
    null = pd.DataFrame(rows)
    null.to_parquet(OUT / "null_draws.parquet", index=False)
    coef = real["full_coefficient_per_sd"]
    folds = real["fold_coefficients_per_sd"]
    draws = coefficient_draws(m36, st)
    lo, hi = (float(v) for v in np.percentile(draws, m36.INTERVAL))
    pct = {k: m36.percentile_of(null[k], held[k]["effect"]) for k in m36.SHORT}
    bar = 100 * m36.NOTABLE[1]
    seasons = sorted(int(s) for s in frame["season"].unique())
    res = {
        "family": FAMILY,
        "candidate": NAME,
        "base_artifact": str(pick.relative_to(ROOT)),
        "base_column": m36.SERVED_COLUMN,
        "games": st["n"],
        "games_with_both_teams_in_regular_season_play_by_play": both,
        "nonzero_games": int((st["xs"] != 0).sum()),
        "seasons": seasons,
        "folds": len(seasons),
        "draws": m36.DRAWS,
        "seed": [m36.SEED, index, NULL_STREAM],
        "shape": st["shape"],
        "coefficient_per_sd": coef,
        "coefficient_interval_per_sd": [lo, hi],
        "coefficient_probability_positive": float((draws > 0).mean()),
        "coefficient_null_percentile": m36.percentile_of(null["coef"], coef),
        "null_coefficient_sd": float(null["coef"].std(ddof=1)),
        "fold_coefficients_per_sd": dict(zip((str(s) for s in seasons), folds, strict=True)),
        "folds_with_expected_sign": int(sum(c * EXPECTED_SIGN > 0 for c in folds)),
        "held_out": held,
        "in_sample": dict(zip(m36.SHORT, (float(v) for v in inside), strict=True)),
        "in_sample_minus_held_out": {
            k: float(inside[i] - held[k]["effect"]) for i, k in enumerate(m36.SHORT)
        },
        "null": {k: m36.band(null[k]) for k in m36.SHORT},
        "null_changed_median": float(null["changed"].median()),
        "null_percentile": pct,
        "declared_bar_percentile": bar,
        "controls": controls(m36, st),
        "correlations": correlations(frame, z),
    }
    res["outcome"] = outcome_of(coef, pct["ll"], bar)
    res["wrong_sign_resolved"] = bool(min(lo * -EXPECTED_SIGN, hi * -EXPECTED_SIGN) > 0)
    batch = cells(m36, res)
    c = res["controls"]
    lines = [
        f"base {res['base_artifact']} column {m36.SERVED_COLUMN}; games {res['games']} both "
        f"{both} nonzero {res['nonzero_games']}; sd {st['sd']:.4f}",
        f"decisive: changed {held['changed']} record {held['arm_right']}-{held['ref_right']}",
        f"DECLARED READ log-loss gain {held['ll']['effect']:+.6f} [{held['ll']['low']:+.6f},"
        f"{held['ll']['high']:+.6f}] probability_positive {held['ll']['pp']:.3f} null percentile "
        f"{pct['ll']:.1f} (bar {bar:.0f}); null mean {res['null']['ll']['mean']:+.6f} "
        f"[{res['null']['ll']['q05']:+.6f},{res['null']['ll']['q95']:+.6f}]",
        f"coefficient per SD {coef:+.4f} [{lo:+.4f},{hi:+.4f}] P+ "
        f"{res['coefficient_probability_positive']:.3f} null percentile "
        f"{res['coefficient_null_percentile']:.1f} null sd {res['null_coefficient_sd']:.4f}; "
        f"folds {' '.join(f'{s}:{v:+.4f}' for s, v in res['fold_coefficients_per_sd'].items())}"
        f"; expected sign in {res['folds_with_expected_sign']}/{res['folds']}",
        f"OUTCOME {res['outcome']}; wrong sign resolved {res['wrong_sign_resolved']}",
        f"context accuracy {held['acc']['effect']:+.3f} [{held['acc']['low']:+.3f},"
        f"{held['acc']['high']:+.3f}] pp {held['acc']['pp']:.3f} pct {pct['acc']:.1f}; brier "
        f"{held['brier']['effect']:+.6f} [{held['brier']['low']:+.6f},"
        f"{held['brier']['high']:+.6f}] pp {held['brier']['pp']:.3f} pct {pct['brier']:.1f}",
        f"context in-sample ll {res['in_sample']['ll']:+.6f} gap "
        f"{res['in_sample_minus_held_out']['ll']:+.6f}; null changed median "
        f"{res['null_changed_median']:.0f}",
        "context controls: "
        + "; ".join(
            f"{k} acc {v['accuracy']:.3f} ll {v['log_loss']:.6f}"
            for k, v in c.items()
            if k != "fold_intercepts"
        ),
        "correlations: "
        + "; ".join(
            f"{k} {v['all']:+.3f} (nonzero {v['nonzero']:+.3f}, n {v['games']})"
            for k, v in res["correlations"].items()
        ),
        f"elapsed {time.time() - started:.0f}s",
    ]
    print("\n".join(lines), flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": batch}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    {"build": build, "confirm": confirm}[sys.argv[1]]()
