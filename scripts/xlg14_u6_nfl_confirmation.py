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
from xlg14_scoreboard_nfl_confirmation import season_table

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
OUT = ROOT / "artifacts" / "xlg14_u6_nfl"
LANE = "docs/lanes/xlg-14-u6-2026-11-16.md"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "xlg14_schedule_strength_family_u6"
NAMES = ["margin_sos_gap", "last_opponent_strength"]
EXPECTED_SIGN = -1.0
EARLIER_CONFIRMATIONS = 2
SCORE_COLUMNS = ["game_id", "home_score", "away_score"]
DIAG_COLUMNS = ["game_id", "home_team", "ats_margin", "home_score", "away_score"]
KEYS = ["game_id", "team", "season", "week"]
SIDES = ("home", "away")
NULL_STREAM = 0
ASOF_STREAM = 1
ASOF_CUTOFFS = 5
TOLERANCE = float(np.sqrt(np.finfo(float).eps))
WHAT = {
    "margin_sos_gap": (
        "mean over the team's earlier games this season of each opponent's points margin per "
        "game over that opponent's games in weeks before this one (including the game against "
        "this team), regular season"
    ),
    "last_opponent_strength": (
        "the most recent earlier opponent's points margin per game over that opponent's games "
        "in weeks before this one (including the game against this team), regular season"
    ),
}
PLAIN = {
    "margin_sos_gap": "pro teams whose schedule so far has been against stronger opponents",
    "last_opponent_strength": "pro teams coming off a game against a stronger opponent",
}
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


def with_opponents(raw, sides):
    opp = pd.concat(
        [
            sides.rename(columns={"home_team": "team", "away_team": "opponent"}),
            sides.rename(columns={"away_team": "team", "home_team": "opponent"}),
        ],
        ignore_index=True,
    )
    return raw.merge(opp, on=["game_id", "team"], how="left", validate="one_to_one")


def to_date(raw):
    r = raw[KEYS + ["opponent", "margin"]]
    tgt = r[KEYS]
    early = r[["team", "season", "week", "opponent"]].rename(columns={"week": "week_e"})
    pairs = tgt.merge(early, on=["team", "season"])
    pairs = pairs[pairs["week_e"] < pairs["week"]]
    og = r[["team", "season", "week", "margin"]].rename(
        columns={"team": "opponent", "week": "week_o"}
    )
    q = pairs.merge(og, on=["opponent", "season"])
    q = q[q["week_o"] < q["week"]]
    strength = (
        q.groupby(["game_id", "team", "week_e"])["margin"].mean().rename("strength").reset_index()
    )
    sos = strength.groupby(["game_id", "team"])["strength"].mean().rename("margin_sos_gap")
    last = (
        strength.sort_values("week_e")
        .groupby(["game_id", "team"])["strength"]
        .last()
        .rename("last_opponent_strength")
    )
    prior = pairs.groupby(["game_id", "team"]).size().rename("prior_games")
    out = tgt.set_index(["game_id", "team"]).join([sos, last, prior]).reset_index()
    out[NAMES + ["prior_games"]] = out[NAMES + ["prior_games"]].fillna(0.0)
    return out


def asof_check(seed, raw, values, first, last):
    rng = np.random.default_rng(seed)
    pool = values.loc[values["season"].between(first, last), ["season", "week"]]
    picks = pool.drop_duplicates().sample(ASOF_CUTOFFS, random_state=rng)
    worst, rows, cutoffs = 0.0, 0, []
    key = ["game_id", "team"]
    for s, w in picks.itertuples(index=False):
        r2 = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["week"] <= w))].copy()
        r2.loc[(r2["season"] == s) & (r2["week"] == w), "margin"] = 0.0
        v2 = to_date(r2)
        for n in NAMES:
            a = values[(values["season"] == s) & (values["week"] == w)].set_index(key)[n]
            b = v2[(v2["season"] == s) & (v2["week"] == w)].set_index(key)[n].reindex(a.index)
            gap = (a - b).abs().to_numpy()
            worst = max(worst, float(np.nanmax(gap)) if gap.size else 0.0)
            worst = float("inf") if np.isnan(gap).any() else worst
        a = values[(values["season"] == s) & (values["week"] == w)]
        rows += len(a)
        cutoffs.append([int(s), int(w), len(a), int((a[NAMES[0]] != 0).sum())])
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
    teams = pd.concat(sides, ignore_index=True)
    raw = with_opponents(pd.concat(raws, ignore_index=True), teams)
    if raw["opponent"].isna().any():
        raise SystemExit("a team-game without an opponent")
    values = to_date(raw)
    seed = [m36.SEED, len(m36.TEMPLATES) + EARLIER_CONFIRMATIONS, ASOF_STREAM]
    check = asof_check(seed, raw, values, m36.FIRST_SEASON, m36.LAST_SEASON)
    print("as-of", check, flush=True)
    if not check["max_abs_difference"] <= TOLERANCE:
        (OUT / "build.json").write_text(
            json.dumps({"as_of_check": check, "failed": True}, indent=1), encoding="utf-8"
        )
        raise SystemExit("as-of check failed")
    values.to_parquet(OUT / "team_values.parquet", index=False)
    raw.to_parquet(OUT / "team_game_margins.parquet", index=False)
    teams.to_parquet(OUT / "game_teams.parquet", index=False)
    span = values[values["season"].between(m36.FIRST_SEASON, m36.LAST_SEASON)]
    shape = {
        n: {
            "team_games": len(span),
            "nonzero_team_games": int((span[n] != 0).sum()),
            "sd": float(span[n].std(ddof=1)),
            "mean": float(span[n].mean()),
        }
        for n in NAMES
    }
    shape["correlation_between_terms"] = float(span[NAMES[0]].corr(span[NAMES[1]]))
    print("values", shape, flush=True)
    out = {"candidates": NAMES, "what": WHAT, "seasons": diag, "as_of_check": check, "values": shape}
    (OUT / "build.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def own_to_date(raw):
    f = pd.read_parquet(FEATURES, columns=DIAG_COLUMNS)
    r = raw[KEYS + ["margin"]].merge(f, on="game_id", how="left", validate="many_to_one")
    home = (r["home_score"] - r["away_score"]).to_numpy(float)
    sign = np.where(r["team"].eq(r["home_team"]), 1.0, -1.0)
    if not np.allclose(sign * home, r["margin"].to_numpy(float)):
        raise SystemExit("team-game margin disagrees with game_features scores")
    r["ats"] = sign * r["ats_margin"]
    sanity = float(np.corrcoef(r["ats_margin"].to_numpy(float), home)[0, 1])
    r["games"] = 1.0
    r = r.sort_values(["team", "season", "week"])
    g = r.groupby(["team", "season"])
    prior = g[["margin", "ats", "games"]].cumsum() - r[["margin", "ats", "games"]]
    played = prior["games"].where(prior["games"] > 0)
    out = r[["game_id", "team"]].copy()
    out["own_margin_to_date"] = (prior["margin"] / played).fillna(0.0)
    out["own_ats_to_date"] = (prior["ats"] / played).fillna(0.0)
    return out, sanity


def joined(d, values, teams, cols):
    look = values.set_index(["game_id", "team"])[cols]
    g = d[["game_id"]].merge(teams, on="game_id", how="left", validate="one_to_one")
    side = {}
    for s in SIDES:
        key = pd.MultiIndex.from_arrays([g["game_id"], g[f"{s}_team"]])
        side[s] = look.reindex(key).reset_index(drop=True)
    both = side["home"][cols[0]].notna() & side["away"][cols[0]].notna()
    out = {}
    for c in cols:
        diff = side["home"][c].to_numpy() - side["away"][c].to_numpy()
        out[c] = np.where(both.to_numpy(), diff, 0.0)
    return out, int(both.sum())


def correlations(x, others):
    nz = x != 0
    out = {}
    for name, col in others.items():
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


def cells(m36, name, res):
    held = res["held_out"]
    lo, hi = res["coefficient_interval_per_sd"]
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"xlg14_schedule_strength_nfl_{name}_{units}",
                "description": (
                    f"NFL confirmation of the college-first survivor {name}: {WHAT[name]}. Home "
                    "minus away; one offset-logistic coefficient per SD on the served "
                    f"out-of-season probability, opener grade {m36.FIRST_SEASON}-"
                    f"{m36.LAST_SEASON}, leave-one-season-out. Nonzero games "
                    f"{res['nonzero_games']}; changed picks {held['changed']}, record "
                    f"{held['arm_right']}-{held['ref_right']}; coefficient per SD "
                    f"{res['coefficient_per_sd']:+.4f} [{lo:+.4f}, {hi:+.4f}], negative in "
                    f"{res['folds_with_expected_sign']} of {res['folds']} folds (expected "
                    f"negative, the college sign); this metric sits at percentile "
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
                    "positive side), split-half reliability of the trait not measured, and no "
                    "positive control bounds an effect this small: the NFL add-one harness "
                    f"shows about .26 logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: one of the two pre-declared NFL confirmations of the "
                    "college survivors; 2 gating reads in total (log-loss percentile in the "
                    "matched null with the full-sample coefficient sign), 3 recorded cells each "
                    "with accuracy and Brier secondary, 3 control reads; terms declared in the "
                    f"lane before any outcome was joined; outcome: {res['outcome']}. Not served."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": int(res["folds"]),
                "plain_summary": (
                    f"Checks whether {PLAIN[name]} go on to beat the opening spread less often, "
                    f"as college teams appeared to. {VERDICTS[res['outcome']]}"
                ),
            }
        )
    return out


def read_one(m36, frame, name, index, others):
    tpl = {
        "name": name,
        "column": name,
        "intercept": False,
        "matched": "served",
        "reported": False,
        "what": "how strong the opponents a team has already played were",
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
    null.to_parquet(OUT / f"null_draws_{name}.parquet", index=False)
    coef = real["full_coefficient_per_sd"]
    folds = real["fold_coefficients_per_sd"]
    draws = coefficient_draws(m36, st)
    lo, hi = (float(v) for v in np.percentile(draws, m36.INTERVAL))
    pct = {k: m36.percentile_of(null[k], held[k]["effect"]) for k in m36.SHORT}
    bar = 100 * m36.NOTABLE[1]
    seasons = sorted(int(s) for s in frame["season"].unique())
    res = {
        "family": FAMILY,
        "candidate": name,
        "base_column": m36.SERVED_COLUMN,
        "games": st["n"],
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
        "correlations": correlations(
            frame[name].to_numpy(float), {"served_logit": z, **others}
        ),
    }
    res["outcome"] = outcome_of(coef, pct["ll"], bar)
    res["wrong_sign_resolved"] = bool(min(lo * -EXPECTED_SIGN, hi * -EXPECTED_SIGN) > 0)
    c = res["controls"]
    lines = [
        f"== {name} (null stream index {index}); games {res['games']} nonzero "
        f"{res['nonzero_games']}; sd {st['sd']:.4f}",
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
    ]
    return res, lines


def confirm():
    started = time.time()
    m36 = load_mod36()
    pick, d = m36.games()
    values = pd.read_parquet(OUT / "team_values.parquet")
    teams = pd.read_parquet(OUT / "game_teams.parquet")
    raw = pd.read_parquet(OUT / "team_game_margins.parquet")
    own, sanity = own_to_date(raw)
    values = values.merge(own, on=["game_id", "team"], how="left", validate="one_to_one")
    cols = NAMES + ["own_margin_to_date", "own_ats_to_date", "prior_games"]
    terms, both = joined(d, values, teams, cols)
    frame = d[["game_id", "season", "week", "home_covered", m36.SERVED_COLUMN]].copy()
    for n in NAMES:
        frame[n] = terms[n]
    results, lines, batch = {}, [], []
    lines.append(
        f"base {pick.relative_to(ROOT)} column {m36.SERVED_COLUMN}; games with both teams "
        f"{both}; ats_margin vs home points margin corr {sanity:+.3f}"
    )
    for i, n in enumerate(NAMES):
        index = len(m36.TEMPLATES) + EARLIER_CONFIRMATIONS + i
        others = {
            "own_ats_to_date_diff": terms["own_ats_to_date"],
            "own_margin_to_date_diff": terms["own_margin_to_date"],
            **{f"other_{o}": terms[o] for o in NAMES if o != n},
        }
        res, ln = read_one(m36, frame, n, index, others)
        res["base_artifact"] = str(pick.relative_to(ROOT))
        res["games_with_both_teams_in_regular_season_play_by_play"] = both
        results[n] = res
        lines += ln
        batch += cells(m36, n, res)
        print("\n".join(ln), flush=True)
    lines.append(f"elapsed {time.time() - started:.0f}s")
    print(lines[0], lines[-1], flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": batch}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    {"build": build, "confirm": confirm}[sys.argv[1]]()
