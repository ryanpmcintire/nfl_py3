import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import xlg14_family_screen as x14  # noqa: E402

from nfl_ats.cfb_features import cfb_season_partitions  # noqa: E402

POWER = SCRIPTS / "xlg_harness_power.py"
OUT = ROOT / "artifacts" / "xlg14_luck_family"
LANE = "docs/lanes/xlg-14-xlg14-cfb-first-discovery.md"
POWER_LANE = "docs/lanes/xlg-harness-power-2026-11-14.md"
FAMILY = "xlg14_luck_family_u3"
STREAM_LUCK = 3
ASOF_STREAM = 1
ASOF_CUTOFFS = 5
ASOF_SEED = x14.u1.BOOTSTRAP_SEED
TOLERANCE = float(np.sqrt(np.finfo(float).eps))
TD_POINTS = 6
REGULAR_SEASON = 2
CALIBRATION_TYPE = "Rushing Touchdown"
PLAY_COLUMNS = [
    "game_id",
    "week",
    "seasonType",
    "pos_team_id",
    "def_pos_team_id",
    "type.text",
    "text",
    "start.yardsToEndzone",
    "pos_team_score",
    "def_pos_team_score",
]
INT_TYPES = ("Interception Return", "Interception Return Touchdown")
FG_TYPES = (
    "Field Goal Good",
    "Field Goal Missed",
    "Blocked Field Goal",
    "Missed Field Goal Return",
    "Blocked Field Goal Touchdown",
    "Missed Field Goal Return Touchdown",
)
BLOCK_TYPES = (
    "Blocked Punt",
    "Blocked Field Goal",
    "Blocked Punt Touchdown",
    "Blocked Field Goal Touchdown",
)
MUFF_TYPES = ("Punt Team Fumble Recovery", "Punt Team Fumble Recovery Touchdown")
NONOFF_TD_TYPES = (
    "Interception Return Touchdown",
    "Fumble Return Touchdown",
    "Fumble Recovery (Opponent) Touchdown",
    "Punt Return Touchdown",
    "Kickoff Return Touchdown",
    "Blocked Punt Touchdown",
    "Blocked Field Goal Touchdown",
    "Missed Field Goal Return Touchdown",
    "Punt Team Fumble Recovery Touchdown",
    "Kickoff Team Fumble Recovery Touchdown",
)
EVENTS = ["fum", "lost", "ints", "pd", "td_start", "td_end", "swing"]
RATE_COUNTS = ["lost_own", "fum_own", "int_own", "pd_own"]
COUNTS = [*RATE_COUNTS, "fum_opp", "lost_opp", "int_opp", "pd_opp", "td", "swing"]
DIRECT = ["non_offensive_td", "field_goal_luck", "kick_block_muff"]
SUMS = ["lost_net", "fum_net", "int_net", "pd_net", *DIRECT, "games"]
CANDIDATES = {
    "fumble_recovery_luck": {
        "what": (
            "opponent fumbles recovered minus own fumbles lost, each less the league share "
            "of fumbles lost in earlier weeks of the same season."
        ),
        "plain": "have recovered more loose balls than chance would give them",
    },
    "interception_luck": {
        "what": (
            "interceptions made minus thrown, each less the league share of defended passes "
            "(interceptions plus broken-up incompletions) intercepted in earlier weeks of "
            "the same season."
        ),
        "plain": "have turned more of the passes they got a hand on into interceptions than usual",
    },
    "non_offensive_td": {
        "what": "return and defensive touchdowns scored minus allowed.",
        "plain": "have scored more touchdowns on returns and defense than they have given up",
    },
    "field_goal_luck": {
        "what": (
            "field goals made minus expected from the same yard line in earlier seasons, own "
            "kicks less opponents' kicks."
        ),
        "plain": "have had field goals fall their way more often than the distances predict",
    },
    "kick_block_muff": {
        "what": (
            "blocked punts and field goals plus muffed punts recovered by the punting team, "
            "for minus against."
        ),
        "plain": "have come out ahead on blocked kicks and muffed punts",
    },
}
NAMES = list(CANDIDATES)


def season_tables(season, path):
    f = pd.read_parquet(path, columns=PLAY_COLUMNS)
    f = f[pd.to_numeric(f["seasonType"], errors="coerce").eq(REGULAR_SEASON)].copy()
    f["week"] = pd.to_numeric(f["week"], errors="coerce")
    f = f[f["pos_team_id"].notna() & f["def_pos_team_id"].notna() & f["week"].notna()]
    f = f.reset_index(drop=True)
    f["game_id"] = pd.to_numeric(f["game_id"]).astype("int64")
    f["pos"] = pd.to_numeric(f["pos_team_id"]).astype("int64")
    f["dfn"] = pd.to_numeric(f["def_pos_team_id"]).astype("int64")
    f["ps"] = pd.to_numeric(f["pos_team_score"], errors="coerce")
    f["ds"] = pd.to_numeric(f["def_pos_team_score"], errors="coerce")
    t = f["type.text"].fillna("")
    words = f["text"].fillna("").astype(str).str.lower()
    g = f.groupby("game_id", sort=False)
    net, hits = {}, {}
    for label, step in (("start", -1), ("end", 1)):
        same = g["pos"].shift(step).eq(f["pos"]).to_numpy()
        a = g["ps"].shift(step).to_numpy()
        b = g["ds"].shift(step).to_numpy()
        by_pos = (np.where(same, a, b) - f["ps"].to_numpy()) * -step >= TD_POINTS
        by_def = (np.where(same, b, a) - f["ds"].to_numpy()) * -step >= TD_POINTS
        net[label] = by_pos.astype(float) - by_def.astype(float)
        hits[label] = by_pos
    nonoff = t.isin(NONOFF_TD_TYPES).to_numpy()
    cal = t.eq(CALIBRATION_TYPE).to_numpy()
    ints = t.isin(INT_TYPES).to_numpy(float)
    pbu = (t.eq("Pass Incompletion") & words.str.contains("broken up", regex=False)).to_numpy(float)
    blocks = t.isin(BLOCK_TYPES).to_numpy(float)
    muffs = t.isin(MUFF_TYPES).to_numpy(float)
    ev = pd.DataFrame(
        {
            "game_id": f["game_id"],
            "week": f["week"],
            "pos": f["pos"],
            "dfn": f["dfn"],
            "fum": t.str.startswith("Fumble Recovery").to_numpy(float),
            "lost": t.str.startswith("Fumble Recovery (Opponent)").to_numpy(float),
            "ints": ints,
            "pd": ints + pbu,
            "td_start": np.where(nonoff, net["start"], 0.0),
            "td_end": np.where(nonoff, net["end"], 0.0),
            "swing": muffs - blocks,
        }
    )
    week = ev.groupby("game_id")["week"].first()
    own = ev.groupby(["game_id", "pos"])[EVENTS].sum().reset_index()
    own = own.rename(columns={"pos": "team"})
    opp = ev.groupby(["game_id", "dfn"])[EVENTS].sum().reset_index()
    opp = opp.rename(columns={"dfn": "team"})
    m = own.merge(opp, on=["game_id", "team"], how="outer", suffixes=("_p", "_d")).fillna(0.0)
    raw = pd.DataFrame(
        {
            "game_id": m["game_id"].astype("int64"),
            "team": m["team"].astype("int64"),
            "season": season,
            "week": m["game_id"].map(week).to_numpy(float),
            "fum_own": m["fum_p"],
            "lost_own": m["lost_p"],
            "int_own": m["ints_p"],
            "pd_own": m["pd_p"],
            "fum_opp": m["fum_d"],
            "lost_opp": m["lost_d"],
            "int_opp": m["ints_d"],
            "pd_opp": m["pd_d"],
            "td_start": m["td_start_p"] - m["td_start_d"],
            "td_end": m["td_end_p"] - m["td_end_d"],
            "swing": m["swing_p"] - m["swing_d"],
        }
    )
    fg = t.isin(FG_TYPES).to_numpy()
    kicks = pd.DataFrame(
        {
            "game_id": f["game_id"],
            "season": season,
            "week": f["week"],
            "team": f["pos"],
            "opp": f["dfn"],
            "yard": pd.to_numeric(f["start.yardsToEndzone"], errors="coerce").round(),
            "made": t.eq("Field Goal Good").to_numpy(float),
        }
    )[fg]
    kicks = kicks[kicks["yard"].notna()].copy()
    kicks["yard"] = kicks["yard"].astype("int64")
    diag = {
        "plays": len(f),
        "team_games": len(raw),
        "fumbles": int(ev["fum"].sum()),
        "fumbles_lost": int(ev["lost"].sum()),
        "interceptions": int(ints.sum()),
        "passes_broken_up": int(pbu.sum()),
        "field_goal_tries": int(fg.sum()),
        "field_goal_tries_with_yard_line": len(kicks),
        "field_goals_made": int(kicks["made"].sum()),
        "blocks": int(blocks.sum()),
        "muffs_recovered_by_punting_team": int(muffs.sum()),
        "non_offensive_td_plays": int(nonoff.sum()),
        "calibration_plays": int(cal.sum()),
        "calibration_hits": {k: int((cal & v).sum()) for k, v in hits.items()},
        "non_offensive_td_credited": {k: int((nonoff & (v != 0)).sum()) for k, v in net.items()},
        "non_offensive_td_to_defense": {k: int((nonoff & (v < 0)).sum()) for k, v in net.items()},
    }
    return raw, kicks, diag


def week_rates(raw):
    tot = raw.groupby(["season", "week"])[RATE_COUNTS].sum().sort_index()
    c = tot.groupby(level="season").cumsum() - tot
    return pd.DataFrame(
        {
            "r_fum": c["lost_own"] / c["fum_own"].where(c["fum_own"] > 0),
            "r_int": c["int_own"] / c["pd_own"].where(c["pd_own"] > 0),
        }
    )


def field_goal_luck(kicks):
    empty = pd.DataFrame(
        {
            "game_id": pd.Series(dtype="int64"),
            "team": pd.Series(dtype="int64"),
            "field_goal_luck": pd.Series(dtype="float64"),
        }
    )
    if kicks.empty:
        return empty
    made = kicks.pivot_table(index="season", columns="yard", values="made", aggfunc="sum")
    tries = kicks.pivot_table(index="season", columns="yard", values="made", aggfunc="count")
    made = made.sort_index().fillna(0.0).cumsum().shift(1)
    tries = tries.sort_index().fillna(0.0).cumsum().shift(1)
    share = (made / tries.where(tries > 0)).reset_index()
    share = share.melt(id_vars="season", var_name="yard", value_name="expected").dropna()
    if share.empty:
        return empty
    share["yard"] = share["yard"].astype("int64")
    k = kicks.merge(share, on=["season", "yard"], how="inner")
    if k.empty:
        return empty
    k["luck"] = k["made"] - k["expected"]
    own = k.groupby(["game_id", "team"])["luck"].sum().rename("own")
    opp = k.groupby(["game_id", "opp"])["luck"].sum().rename("opp")
    opp.index = opp.index.set_names(["game_id", "team"])
    both = pd.concat([own, opp], axis=1).fillna(0.0)
    both["field_goal_luck"] = both["own"] - both["opp"]
    return both.reset_index()[["game_id", "team", "field_goal_luck"]]


def to_date(raw, kicks):
    r = raw.reset_index(drop=True)
    v = r[["game_id", "team", "season", "week"]].copy()
    v["lost_net"] = r["lost_opp"] - r["lost_own"]
    v["fum_net"] = r["fum_opp"] - r["fum_own"]
    v["int_net"] = r["int_opp"] - r["int_own"]
    v["pd_net"] = r["pd_opp"] - r["pd_own"]
    v["non_offensive_td"] = r["td"]
    v["kick_block_muff"] = r["swing"]
    v = v.merge(field_goal_luck(kicks), on=["game_id", "team"], how="left")
    v["field_goal_luck"] = v["field_goal_luck"].fillna(0.0)
    v["games"] = 1.0
    tw = v.groupby(["team", "season", "week"])[SUMS].sum().reset_index()
    prior = tw.groupby(["team", "season"])[SUMS].cumsum() - tw[SUMS]
    out = tw[["team", "season", "week"]].join(week_rates(r), on=["season", "week"])
    played = prior["games"].where(prior["games"] > 0)
    fum = prior["lost_net"] - out["r_fum"] * prior["fum_net"]
    pick = prior["int_net"] - out["r_int"] * prior["pd_net"]
    out["fumble_recovery_luck"] = (fum / played).fillna(0.0)
    out["interception_luck"] = (pick / played).fillna(0.0)
    for n in DIRECT:
        out[n] = (prior[n] / played).fillna(0.0)
    out["prior_games"] = prior["games"]
    out = out.drop(columns=["r_fum", "r_int"])
    return v[["game_id", "team", "season", "week"]].merge(
        out, on=["team", "season", "week"], how="left"
    )


def asof_check(raw, kicks, values):
    rng = np.random.default_rng([ASOF_SEED, STREAM_LUCK, ASOF_STREAM])
    pool = values[values["season"].isin(x14.u1.SEASONS)][["season", "week"]].drop_duplicates()
    picks = pool.sample(ASOF_CUTOFFS, random_state=rng)
    worst, rows, cutoffs = 0.0, 0, []
    for s, w in picks.itertuples(index=False):
        r2 = raw[(raw["season"] < s) | ((raw["season"] == s) & (raw["week"] <= w))].copy()
        at = (r2["season"] == s) & (r2["week"] == w)
        r2.loc[at, COUNTS] = 0.0
        k2 = kicks[(kicks["season"] < s) | ((kicks["season"] == s) & (kicks["week"] < w))]
        v2 = to_date(r2, k2)
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
        "tolerance": TOLERANCE,
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = cfb_season_partitions(ROOT / "data" / "cfb", "pbp")
    last = max(x14.u1.SEASONS)
    raws, kicks, diag, skipped = [], [], {}, []
    for season in sorted(s for s in parts if s <= last):
        have = set(pq.ParquetFile(parts[season]).schema_arrow.names)
        if not set(PLAY_COLUMNS) <= have:
            skipped.append(int(season))
            continue
        r, k, d = season_tables(int(season), parts[season])
        raws.append(r)
        kicks.append(k)
        diag[str(season)] = d
    raw = pd.concat(raws, ignore_index=True)
    kick = pd.concat(kicks, ignore_index=True)
    hits = {k: sum(d["calibration_hits"][k] for d in diag.values()) for k in ("start", "end")}
    chosen = max(hits, key=hits.get)
    raw["td"] = raw[f"td_{chosen}"]
    raw = raw.drop(columns=["td_start", "td_end"])
    values = to_date(raw, kick)
    check = asof_check(raw, kick, values)
    tot = raw.groupby("season")[RATE_COUNTS].sum()
    recorded = {
        str(s): [
            round(float(row["lost_own"] / row["fum_own"]), 3) if row["fum_own"] else None,
            round(float(row["int_own"] / row["pd_own"]), 3) if row["pd_own"] else None,
        ]
        for s, row in tot.iterrows()
    }
    pop = values[values["season"].isin(x14.u1.SEASONS)]
    live = pop.groupby("season")[NAMES].agg(lambda c: float(c.ne(0).mean())).round(2)
    report = {
        "family": FAMILY,
        "skipped_seasons_missing_columns": skipped,
        "score_columns_read_as": chosen,
        "calibration_plays": sum(d["calibration_plays"] for d in diag.values()),
        "calibration_hits": hits,
        "season_share_fumbles_lost_and_defended_passes_intercepted": recorded,
        "population_team_games": len(pop),
        "population_nonzero_share": {n: float(pop[n].ne(0).mean()) for n in NAMES},
        "population_nonzero_share_by_season": {
            str(s): [float(x) for x in row] for s, row in live.iterrows()
        },
        "population_sd": {n: float(pop[n].std(ddof=1)) for n in NAMES},
        "asof_check": check,
        "per_season": diag,
    }
    (OUT / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("score columns read as", chosen, hits, "skipped", skipped)
    print("season lost share, intercepted share", recorded)
    print("nonzero share by season", NAMES, report["population_nonzero_share_by_season"])
    print("sd", {k: round(v, 3) for k, v in report["population_sd"].items()})
    print("asof", check, flush=True)
    if not check["max_abs_difference"] <= TOLERANCE:
        raise SystemExit("as-of check failed; screen not run")
    values.to_parquet(OUT / "team_values.parquet", index=False)


def load_power():
    text = POWER.read_text(encoding="utf-8")
    body, call, rest = text.rpartition("\nmain()")
    if not call or rest.strip():
        raise SystemExit("xlg_harness_power.py no longer ends with a bare main() call")
    mod = types.ModuleType("xlg_harness_power")
    mod.__file__ = str(POWER)
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        exec(compile(body, str(POWER), "exec"), mod.__dict__)
    finally:
        sys.argv = argv
    return mod


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


def controls(m36, st):
    y, z, folds = st["y"], st["z"], st["folds"]
    const = np.empty(st["n"])
    for te in folds:
        const[te] = y[~te].mean()
    fitted, _ = m36.loso(np.ones((st["n"], 1)), y, z, folds)
    out = {}
    for name, p in (("base", expit(z)), ("constant", const), ("base_plus_intercept", fitted)):
        m = m36.metric_matrix(p, y)
        out[name] = {"accuracy": float(m[:, 0].mean()), "log_loss": float(m[:, 1].mean())}
    return out


def read_candidate(hp, frame, name, index):
    m36 = hp.M36
    tpl = {"name": name, "column": name, "intercept": False}
    st = m36.setup(frame, tpl, index)
    real = m36.real_read(st, tpl)
    y, z = st["y"], st["z"]
    xd = m36.design(st["xs"], False)
    p, _ = m36.loso(xd, y, z, st["folds"])
    base = st["refs"]["served"]["m"]
    v = (m36.metric_matrix(p, y) - base) * m36.DIRECTION
    full = m36.fit_offset(xd, y, z)
    inside = ((m36.metric_matrix(expit(z + xd @ full), y) - base) * m36.DIRECTION).mean(axis=0)
    season = frame["season"].to_numpy()
    seasons = np.unique(season)
    cut = seasons[(len(seasons) - 1) // 2]
    halves = {}
    for label, mask in (("earlier", season <= cut), ("later", season > cut)):
        halves[label] = {
            "seasons": [int(s) for s in np.unique(season[mask])],
            "games": int(mask.sum()),
            "nonzero_games": int((st["xs"][mask] != 0).sum()),
            "coefficient_per_sd": float(m36.fit_offset(xd[mask], y[mask], z[mask])[-1]),
            "held_out_accuracy_points": float(v[mask, 0].mean()),
            "held_out_log_loss_improvement": float(v[mask, 1].mean()),
        }
    rng = np.random.default_rng([hp.SEED, STREAM_LUCK, index, 0])
    rows = []
    for i in range(hp.DRAWS):
        m36.draw_rows(st, tpl, "null", 0.0, i, m36.permuted(st, rng), rows)
    null = pd.DataFrame(rows)
    held = real["served"]
    signs = {float(np.sign(h["coefficient_per_sd"])) for h in halves.values()}
    pct = {k: m36.percentile_of(null[k], held[k]["effect"]) for k in m36.SHORT}
    bar = 100 * m36.NOTABLE[1]
    out = {
        "games": st["n"],
        "nonzero_games": int((st["xs"] != 0).sum()),
        "shape": st["shape"],
        "coefficient_per_sd": real["full_coefficient_per_sd"],
        "fold_coefficients_per_sd": real["fold_coefficients_per_sd"],
        "folds_with_full_sign": int(
            sum(
                np.sign(c) == np.sign(real["full_coefficient_per_sd"])
                for c in real["fold_coefficients_per_sd"]
            )
        ),
        "held_out": held,
        "in_sample": dict(zip(m36.SHORT, (float(x) for x in inside), strict=True)),
        "in_sample_minus_held_out": {
            k: float(inside[i] - held[k]["effect"]) for i, k in enumerate(m36.SHORT)
        },
        "halves": halves,
        "half_signs_agree": bool(len(signs) == 1 and 0.0 not in signs),
        "null": {k: m36.band(null[k]) for k in m36.SHORT},
        "null_changed_median": float(null["changed"].median()),
        "null_share_probability_at_least": {
            k: float((null[f"pp_{k}"] >= m36.NOTABLE[0]).mean()) for k in m36.SHORT
        },
        "null_percentile": pct,
        "survivor_bar_percentile": bar,
    }
    out["survivor"] = bool(pct["ll"] > bar and out["half_signs_agree"])
    return out, st


def cells(m36, name, res, seasons, draws):
    spec = CANDIDATES[name]
    held = res["held_out"]
    early = res["halves"]["earlier"]["coefficient_per_sd"]
    late = res["halves"]["later"]["coefficient_per_sd"]
    survived = "yes" if res["survivor"] else "no"
    verdict = (
        "It stood out from chance and goes to one pro-football check."
        if res["survivor"]
        else "No effect large enough for this test to see showed up."
    )
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"xlg14_luck_cfb_{name}_{units}",
                "description": (
                    f"College-first luck family, {name}: {spec['what']} Per game over each "
                    "team's games in strictly earlier weeks of the same season, home minus "
                    "away; one offset-logistic coefficient on a leave-one-season-out ridge "
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
                "category": "onfield",
                "classification_evidence": (
                    "No resolved wrong sign for a declared mechanism (a small held-out "
                    "log-loss cost is what a no-information term shows on this harness), "
                    "split-half reliability of the trait not measured, and no positive "
                    "control bounds an effect this small: the college add-one harness shows "
                    f"about .09 to .11 logit per SD four times in five ({POWER_LANE})."
                ),
                "notes": (
                    f"Family {FAMILY}: 5 candidates x 3 metrics = 15 recorded cells; 5 gating "
                    "reads (log-loss percentile in the matched null), 10 season-half "
                    "coefficients and 2 control reads; survivor rule declared in the lane "
                    f"before any outcome was joined; survivor: {survived}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    f"Checks whether college teams that {spec['plain']} so far this season go "
                    f"on to beat the opening spread more or less often. {verdict}"
                ),
            }
        )
    return out


def screen():
    started = time.time()
    hp = load_power()
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
        "seed": [hp.SEED, STREAM_LUCK],
        "candidates": {},
    }
    batch, lines = [], []
    for index, n in enumerate(NAMES):
        res, st = read_candidate(hp, frame, n, index)
        if index == 0:
            out["controls"] = controls(m36, st)
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
            f"pct {res['null_percentile']['ll']:.1f}; null ll mean "
            f"{res['null']['ll']['mean']:+.6f} [{res['null']['ll']['q05']:+.6f},"
            f"{res['null']['ll']['q95']:+.6f}]; in-sample ll {res['in_sample']['ll']:+.6f} gap "
            f"{res['in_sample_minus_held_out']['ll']:+.6f}; coef {res['coefficient_per_sd']:+.4f} "
            f"folds {res['folds_with_full_sign']}/{len(seasons)}; halves "
            f"{res['halves']['earlier']['coefficient_per_sd']:+.4f} / "
            f"{res['halves']['later']['coefficient_per_sd']:+.4f} (ll "
            f"{res['halves']['earlier']['held_out_log_loss_improvement']:+.6f} / "
            f"{res['halves']['later']['held_out_log_loss_improvement']:+.6f}); "
            f"survivor {res['survivor']}"
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
