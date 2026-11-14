import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
MOD36 = ROOT / "scripts" / "mod36_addone_noise_floor.py"
PBP_ROOT = ROOT / "data" / "pbp" / "raw"
OUT = ROOT / "artifacts" / "xlg14_luck_family" / "nfl_confirmation"
LANE = "docs/lanes/xlg-14-xlg14-cfb-first-discovery.md"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "xlg14_luck_family_u3"
NAME = "fumble_recovery_luck"
EXPECTED_SIGN = -1.0
REGULAR_SEASON = "REG"
SCRIMMAGE_TYPES = ("pass", "run")
PLAY_COLUMNS = [
    "game_id",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "play_type",
    "fumbled_1_player_id",
    "fumble_lost",
]
EVENTS = ["fum", "lost"]
KEYS = ["game_id", "team", "season", "week"]
COUNTS = ["fum_own", "lost_own", "fum_opp", "lost_opp"]
SUMS = ["lost_net", "fum_net", "games"]
SIDES = ("home", "away")
TEAMS_PER_GAME = len(SIDES)
NULL_STREAM = 0
ASOF_STREAM = 1
ASOF_CUTOFFS = 5
TOLERANCE = float(np.sqrt(np.finfo(float).eps))
WHAT = (
    "opponent fumbles recovered minus own fumbles lost, each less the league share of fumbles "
    "lost in earlier weeks of the same season (run and pass plays with a fumbling player, "
    "charged to the team in possession, regular season)"
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


def load_mod36():
    text = MOD36.read_text(encoding="utf-8")
    body, call, rest = text.rpartition("\nmain()")
    if not call or rest.strip():
        raise SystemExit("mod36_addone_noise_floor.py no longer ends with a bare main() call")
    mod = types.ModuleType("mod36_addone_noise_floor")
    mod.__file__ = str(MOD36)
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        exec(compile(body, str(MOD36), "exec"), mod.__dict__)
    finally:
        sys.argv = argv
    return mod


def latest_plays():
    latest = {}
    for snap in sorted(p for p in PBP_ROOT.iterdir() if p.is_dir()):
        for f in snap.glob("season=*/plays.parquet"):
            latest[int(f.parent.name.split("=")[1])] = f
    return latest


def season_table(season, path):
    f = pd.read_parquet(path, columns=PLAY_COLUMNS)
    f = f[f["season_type"].eq(REGULAR_SEASON) & f["posteam"].notna() & f["defteam"].notna()]
    f = f.reset_index(drop=True)
    scrimmage = f["play_type"].isin(SCRIMMAGE_TYPES)
    fum = (scrimmage & f["fumbled_1_player_id"].notna()).to_numpy(float)
    lost = fum * f["fumble_lost"].eq(1).to_numpy(float)
    ev = pd.DataFrame(
        {
            "game_id": f["game_id"],
            "pos": f["posteam"],
            "dfn": f["defteam"],
            "fum": fum,
            "lost": lost,
        }
    )
    own = ev.groupby(["game_id", "pos"])[EVENTS].sum().reset_index()
    own = own.rename(columns={"pos": "team"})
    opp = ev.groupby(["game_id", "dfn"])[EVENTS].sum().reset_index()
    opp = opp.rename(columns={"dfn": "team"})
    m = own.merge(opp, on=["game_id", "team"], how="outer", suffixes=("_p", "_d")).fillna(0.0)
    week = f.groupby("game_id")["week"].first()
    raw = pd.DataFrame(
        {
            "game_id": m["game_id"],
            "team": m["team"],
            "season": season,
            "week": m["game_id"].map(week).to_numpy(float),
            "fum_own": m["fum_p"],
            "lost_own": m["lost_p"],
            "fum_opp": m["fum_d"],
            "lost_opp": m["lost_d"],
        }
    )
    sides = f.groupby("game_id")[[f"{s}_team" for s in SIDES]].first().reset_index()
    per_game = raw.groupby("game_id")["team"].nunique()
    if not per_game.eq(TEAMS_PER_GAME).all():
        raise SystemExit(f"season {season}: a game without exactly two teams in play-by-play")
    diag = {
        "source": str(path.relative_to(ROOT)),
        "plays": len(f),
        "games": len(sides),
        "team_games": len(raw),
        "weeks": [int(raw["week"].min()), int(raw["week"].max())],
        "fumbles": int(ev["fum"].sum()),
        "fumbles_lost": int(ev["lost"].sum()),
        "lost_share": float(ev["lost"].sum() / ev["fum"].sum()),
    }
    return raw, sides, diag


def to_date(raw):
    r = raw.reset_index(drop=True)
    v = r[KEYS].copy()
    v["lost_net"] = r["lost_opp"] - r["lost_own"]
    v["fum_net"] = r["fum_opp"] - r["fum_own"]
    v["games"] = 1.0
    tw = v.groupby(["team", "season", "week"])[SUMS].sum().reset_index()
    prior = tw.groupby(["team", "season"])[SUMS].cumsum() - tw[SUMS]
    tot = r.groupby(["season", "week"])[["lost_own", "fum_own"]].sum().sort_index()
    c = tot.groupby(level="season").cumsum() - tot
    share = (c["lost_own"] / c["fum_own"].where(c["fum_own"] > 0)).rename("share")
    out = tw[["team", "season", "week"]].join(share, on=["season", "week"])
    played = prior["games"].where(prior["games"] > 0)
    luck = prior["lost_net"] - out["share"] * prior["fum_net"]
    out[NAME] = (luck / played).fillna(0.0)
    out["prior_games"] = prior["games"]
    out = out.drop(columns=["share"])
    return v[KEYS].merge(out, on=["team", "season", "week"], how="left")


def asof_check(seed, raw, values):
    rng = np.random.default_rng(seed)
    pool = values[["season", "week"]].drop_duplicates()
    picks = pool.sample(ASOF_CUTOFFS, random_state=rng)
    worst, rows, cutoffs = 0.0, 0, []
    key = ["game_id", "team"]
    for s, w in picks.itertuples(index=False):
        r2 = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["week"] <= w))].copy()
        at = (r2["season"] == s) & (r2["week"] == w)
        r2.loc[at, COUNTS] = 0.0
        v2 = to_date(r2)
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
    raws, sides, diag = [], [], {}
    for season in range(m36.FIRST_SEASON, m36.LAST_SEASON + 1):
        raw, side, d = season_table(season, files[season])
        raws.append(raw)
        sides.append(side)
        diag[str(season)] = d
        print(season, d, flush=True)
    raw = pd.concat(raws, ignore_index=True)
    values = to_date(raw)
    index = len(m36.TEMPLATES)
    check = asof_check([m36.SEED, index, ASOF_STREAM], raw, values)
    print("as-of", check, flush=True)
    if not check["max_abs_difference"] <= TOLERANCE:
        raise SystemExit("as-of check failed")
    values.to_parquet(OUT / "team_values.parquet", index=False)
    raw.to_parquet(OUT / "team_game_counts.parquet", index=False)
    pd.concat(sides, ignore_index=True).to_parquet(OUT / "game_teams.parquet", index=False)
    shape = {
        "team_games": len(values),
        "nonzero_team_games": int((values[NAME] != 0).sum()),
        "sd": float(values[NAME].std(ddof=1)),
        "mean": float(values[NAME].mean()),
    }
    print("values", shape, flush=True)
    out = {"candidate": NAME, "what": WHAT, "seasons": diag, "as_of_check": check, "values": shape}
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


def controls(m36, st):
    y, z, folds = st["y"], st["z"], st["folds"]
    const = np.empty(st["n"])
    for te in folds:
        const[te] = y[~te].mean()
    fitted, coefs = m36.loso(np.ones((st["n"], 1)), y, z, folds)
    out = {}
    for name, p in (("base", expit(z)), ("constant", const), ("base_plus_intercept", fitted)):
        m = m36.metric_matrix(p, y)
        out[name] = {"accuracy": float(m[:, 0].mean()), "log_loss": float(m[:, 1].mean())}
    out["fold_intercepts"] = [float(c) for c in coefs[:, 0]]
    return out


def coefficient_draws(m36, st):
    x, y, z, w = st["xs"], st["y"], st["z"], st["w"]
    b = np.zeros(len(w))
    for _ in range(m36.NEWTON_STEPS):
        p = expit(z[None, :] + b[:, None] * x[None, :])
        grad = (w * (x * (y - p))).sum(axis=1)
        hess = (w * (x * x * p * (1 - p))).sum(axis=1)
        step = grad / hess
        b = b + step
        if np.abs(step).max() < m36.NEWTON_TOL:
            break
    return b


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
                "name": f"xlg14_luck_nfl_{NAME}_{units}",
                "description": (
                    f"NFL confirmation of the college-first survivor {NAME}: {WHAT}. Per game "
                    "over each team's games in strictly earlier weeks of the same season, home "
                    "minus away; one offset-logistic coefficient per SD on the served "
                    f"out-of-season probability, opener grade {m36.FIRST_SEASON}-"
                    f"{m36.LAST_SEASON}, leave-one-season-out. Nonzero games "
                    f"{res['nonzero_games']}; changed picks {held['changed']}, record "
                    f"{held['arm_right']}-{held['ref_right']}; coefficient per SD "
                    f"{res['coefficient_per_sd']:+.4f} [{lo:+.4f}, {hi:+.4f}], negative in "
                    f"{res['folds_with_expected_sign']} of {res['folds']} folds (expected "
                    f"negative); this metric sits at percentile {res['null_percentile'][k]:.1f} "
                    f"of {res['draws']} within-season shuffles of the same column; in-sample "
                    f"minus held-out {res['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(held[k]["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": m36.FIRST_SEASON,
                "season_end": m36.LAST_SEASON,
                "family": FAMILY,
                "category": "onfield",
                "classification_evidence": (
                    "No resolved wrong sign (the coefficient's interval is not wholly on the "
                    "positive side), split-half reliability of the trait not measured, and no "
                    "positive control bounds an effect this small: the NFL add-one harness "
                    f"shows about .26 logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: the one pre-declared NFL confirmation of the college "
                    "survivor; 1 gating read (log-loss percentile in the matched null with the "
                    "full-sample coefficient sign), 3 recorded cells with accuracy and Brier "
                    "secondary, 3 control reads; terms resolved in the lane before any outcome "
                    f"was joined; outcome: {res['outcome']}. Not served."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": int(res["folds"]),
                "plain_summary": (
                    "Checks whether pro teams that have recovered more loose balls than chance "
                    "would give them so far this season go on to beat the opening spread less "
                    f"often, as college teams appeared to. {VERDICTS[res['outcome']]}"
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
    index = len(m36.TEMPLATES)
    tpl = {
        "name": NAME,
        "column": NAME,
        "intercept": False,
        "matched": "served",
        "reported": False,
        "what": "how far each team's fumble recoveries so far have run ahead of chance",
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
