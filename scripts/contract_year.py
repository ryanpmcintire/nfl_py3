import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "contracts"
OUT = ROOT / "artifacts" / "contract_year"
FEATURES = ROOT / "data" / "processed" / "game_features.parquet"
SNAP_ROOT = ROOT / "data" / "players" / "raw"
LANE = "docs/lanes/contract-year-2026-11-16.md"
FAMILY = "contract_year_family_v1"
NAME = "contract_year_snap_share_gap"
STREAM = 42
FAIR = 0.5
REGULAR = "REG"
ASOF_CUTOFFS = 5
TOLERANCE = float(np.sqrt(np.finfo(float).eps))
WHAT = (
    "home minus away share of offensive plus defensive snaps, over the team's strictly earlier "
    "regular-season games of the same season, played by players whose latest contract signed "
    "before the season's calendar year ends that season (nflverse OverTheCap contracts)"
)


def latest_snaps():
    return sorted(SNAP_ROOT.glob("*/snap_counts.parquet"))[-1]


def walk_years():
    c = pd.read_parquet(
        DATA / "historical_contracts.parquet", columns=["otc_id", "gsis_id", "year_signed", "years"]
    )
    c = c[(c["year_signed"] > 0) & c["years"].gt(0)].copy()
    c["last"] = c["year_signed"] + c["years"].astype(int) - 1
    return c


def player_map():
    p = pd.read_parquet(DATA / "players.parquet", columns=["pfr_id", "gsis_id", "otc_id"])
    p = p[p["pfr_id"].notna()].drop_duplicates("pfr_id")
    p["otc_id"] = pd.to_numeric(p["otc_id"], errors="coerce")
    return p


def flags(snaps, c, pm):
    s = snaps.merge(pm, left_on="pfr_player_id", right_on="pfr_id", how="left")
    by_otc = c[["otc_id", "year_signed", "last"]].dropna()
    by_gsis = c[["gsis_id", "year_signed", "last"]].dropna()
    keys = s[["pfr_player_id", "otc_id", "gsis_id", "season"]].drop_duplicates(
        ["pfr_player_id", "season"]
    )
    out = []
    for key, table in (("otc_id", by_otc), ("gsis_id", by_gsis)):
        k = keys[keys[key].notna()].merge(table, on=key, how="inner")
        k = k[k["year_signed"] < k["season"]]
        k = k.sort_values(["year_signed", "last"]).groupby(["pfr_player_id", "season"]).tail(1)
        out.append(k[["pfr_player_id", "season", "year_signed", "last"]].assign(via=key))
    f = pd.concat(out).drop_duplicates(["pfr_player_id", "season"], keep="first")
    f["walk"] = (f["last"] == f["season"]).astype(float)
    return f[["pfr_player_id", "season", "walk"]]


def team_values(snaps, f):
    s = snaps.merge(f, on=["pfr_player_id", "season"], how="left")
    s["matched"] = s["walk"].notna().astype(float)
    s["walk"] = s["walk"].fillna(0.0)
    s["snaps"] = s["offense_snaps"].fillna(0) + s["defense_snaps"].fillna(0)
    s["walk_snaps"] = s["snaps"] * s["walk"]
    s["matched_snaps"] = s["snaps"] * s["matched"]
    tg = s.groupby(["game_id", "team", "season", "week"])[
        ["snaps", "walk_snaps", "matched_snaps"]
    ].sum()
    tg = tg.reset_index().sort_values(["team", "season", "week"])
    prior = (
        tg.groupby(["team", "season"])[["snaps", "walk_snaps"]].cumsum()
        - tg[["snaps", "walk_snaps"]]
    )
    tg["prior_snaps"] = prior["snaps"]
    tg[NAME] = (prior["walk_snaps"] / prior["snaps"].where(prior["snaps"] > 0)).fillna(0.0)
    return tg


def load_snaps():
    s = pd.read_parquet(latest_snaps())
    return s[s["game_type"].eq(REGULAR)].copy()


def asof_check(snaps, f, values):
    rng = np.random.default_rng([36, STREAM, 1])
    picks = values[["season", "week"]].drop_duplicates().sample(ASOF_CUTOFFS, random_state=rng)
    worst, cut = 0.0, []
    for s, w in picks.itertuples(index=False):
        part = snaps[(snaps["season"] < s) | ((snaps["season"] == s) & (snaps["week"] <= w))].copy()
        at = (part["season"] == s) & (part["week"] == w)
        part.loc[at, ["offense_snaps", "defense_snaps"]] = 0
        v2 = team_values(part, f)
        a = values[(values["season"] == s) & (values["week"] == w)].set_index(["game_id", "team"])[
            NAME
        ]
        b = (
            v2[(v2["season"] == s) & (v2["week"] == w)]
            .set_index(["game_id", "team"])[NAME]
            .reindex(a.index)
        )
        gap = (a - b).abs().to_numpy()
        worst = float("inf") if np.isnan(gap).any() else max(worst, float(gap.max()))
        cut.append([int(s), int(w), len(a)])
    return {"cutoffs": cut, "max_abs_difference": worst, "tolerance": TOLERANCE}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = load_snaps()
    c, pm = walk_years(), player_map()
    f = flags(snaps, c, pm)
    values = team_values(snaps, f)
    check = asof_check(snaps, f, values)
    if not check["max_abs_difference"] <= TOLERANCE:
        raise SystemExit(f"as-of check failed {check}")
    s = snaps.merge(f, on=["pfr_player_id", "season"], how="left")
    s["snaps"] = s["offense_snaps"].fillna(0) + s["defense_snaps"].fillna(0)
    cov = s.groupby("season").apply(
        lambda g: pd.Series(
            {
                "matched_snap_share": float(
                    g.loc[g["walk"].notna(), "snaps"].sum() / g["snaps"].sum()
                ),
                "walk_snap_share": float(g.loc[g["walk"].eq(1), "snaps"].sum() / g["snaps"].sum()),
            }
        ),
        include_groups=False,
    )
    values.to_parquet(OUT / "team_values.parquet", index=False)
    out = {
        "snaps_source": str(latest_snaps().relative_to(ROOT)),
        "contracts_rows": len(c),
        "player_seasons_flagged": len(f),
        "coverage_by_season": cov.round(4).to_dict(orient="index"),
        "as_of_check": check,
        "team_games": len(values),
        "term_sd": float(values.loc[values["prior_snaps"] > 0, NAME].std(ddof=1)),
    }
    (OUT / "build.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


def game_term(games, values):
    look = values.set_index(["game_id", "team"])[[NAME, "prior_snaps"]]
    side = {}
    for t in ("home", "away"):
        key = pd.MultiIndex.from_arrays([games["game_id"], games[f"{t}_team"]])
        side[t] = look.reindex(key).reset_index(drop=True)
    ok = (side["home"]["prior_snaps"] > 0) & (side["away"]["prior_snaps"] > 0)
    diff = side["home"][NAME].to_numpy() - side["away"][NAME].to_numpy()
    return np.where(ok.to_numpy(), diff, 0.0), int(ok.sum())


def cells(m36, res, seasons, grade, power_lane):
    held = res["held_out"]
    early = res["halves"]["earlier"]["coefficient_per_sd"]
    late = res["halves"]["later"]["coefficient_per_sd"]
    out = []
    for k, units in zip(m36.SHORT, m36.METRICS, strict=True):
        out.append(
            {
                "name": f"contract_year_nfl_{grade}_{NAME}_{units}",
                "description": (
                    f"Contract-year family, {NAME}: {WHAT}. One offset-logistic coefficient per "
                    f"SD, leave-one-season-out, graded {grade}, {seasons[0]}-{seasons[-1]}; "
                    f"declared sign positive. Changed picks {held['changed']}, record "
                    f"{held['arm_right']}-{held['ref_right']}; coefficient per SD "
                    f"{res['coefficient_per_sd']:+.4f}; season halves {early:+.4f} and "
                    f"{late:+.4f}; percentile {res['null_percentile'][k]:.1f} of "
                    f"{res['draws']} within-season shuffles; in-sample minus held-out "
                    f"{res['in_sample_minus_held_out'][k]:+.6f}."
                ),
                "source": LANE,
                "effect": round(held[k]["effect"], 6),
                "effect_units": units,
                "classification": "unresolved_below_power",
                "league": "nfl",
                "season_start": int(seasons[0]),
                "season_end": int(seasons[-1]),
                "family": FAMILY,
                "category": "market",
                "classification_evidence": (
                    "No resolved wrong sign for the declared mechanism, split-half reliability "
                    "not measured, and no positive control bounds an effect this small "
                    f"({power_lane})."
                ),
                "notes": (
                    f"Family {FAMILY}: declared in {LANE} before the outcome join; 1 gating "
                    "log-loss read in the matched null, 3 recorded cells, 2 season-half "
                    f"coefficients; survivor: {'yes' if res['survivor'] else 'no'}."
                ),
                "interval_low": round(held[k]["low"], 6),
                "interval_high": round(held[k]["high"], 6),
                "probability_positive": round(held[k]["pp"], 4),
                "sample_games": int(res["games"]),
                "sample_blocks": len(seasons),
                "plain_summary": (
                    "Checks whether teams leaning more on players in the last year of their "
                    "contracts beat the spread more often. "
                    + (
                        "It stood out from chance and earns one declared served check."
                        if res["survivor"]
                        else "No effect large enough for this test to see showed up."
                    )
                ),
            }
        )
    return out


def harness():

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import xlg14_luck_family as u3

    u3.STREAM_LUCK = STREAM
    return u3, u3.load_power()


def report(name, res, seasons, both, extra):
    h = res["held_out"]
    line = (
        f"{name}: games {res['games']} both-with-prior {both} nonzero {res['nonzero_games']} "
        f"changed {h['changed']} record {h['arm_right']}-{h['ref_right']}; acc "
        f"{h['acc']['effect']:+.3f} [{h['acc']['low']:+.3f},{h['acc']['high']:+.3f}] pp "
        f"{h['acc']['pp']:.3f} pct {res['null_percentile']['acc']:.1f}; ll "
        f"{h['ll']['effect']:+.6f} [{h['ll']['low']:+.6f},{h['ll']['high']:+.6f}] pp "
        f"{h['ll']['pp']:.3f} pct "
        f"{res['null_percentile']['ll']:.1f}; brier pct {res['null_percentile']['brier']:.1f}; "
        f"in-sample ll gap {res['in_sample_minus_held_out']['ll']:+.6f}; coef "
        f"{res['coefficient_per_sd']:+.4f} folds {res['folds_with_full_sign']}/{len(seasons)} "
        f"{' '.join(f'{c:+.3f}' for c in res['fold_coefficients_per_sd'])}; halves "
        f"{res['halves']['earlier']['coefficient_per_sd']:+.4f} / "
        f"{res['halves']['later']['coefficient_per_sd']:+.4f}; survivor {res['survivor']}; {extra}"
    )
    print(line, flush=True)
    return line


def screen():
    started = time.time()
    u3, hp = harness()
    m36 = hp.M36
    values = pd.read_parquet(OUT / "team_values.parquet")
    g = pd.read_parquet(
        FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
            "home_cover",
        ],
    )
    g = g[g["game_id"].isin(values["game_id"]) & g["spread_line"].notna() & g["home_score"].notna()]
    margin = g["home_score"] - g["away_score"] - g["spread_line"]
    agree = float(((margin > 0).astype(float) == g["home_cover"]).loc[margin != 0].mean())
    g = g[margin != 0].copy()
    g["home_covered"] = (margin.loc[g.index] > 0).astype(float)
    g = g.sort_values(["season", "game_id"]).reset_index(drop=True)
    term, both = game_term(g, values)
    frame = g[["game_id", "season", "week", "home_covered"]].copy()
    frame[m36.SERVED_COLUMN] = FAIR
    frame[NAME] = term
    seasons = sorted(int(s) for s in frame["season"].unique())
    res, st = u3.read_candidate(hp, frame, NAME, 0)
    del st
    res["draws"] = hp.DRAWS
    res["home_cover_agreement"] = agree
    line = report("close", res, seasons, both, f"elapsed {time.time() - started:.0f}s")
    batch = cells(m36, res, seasons, "close", u3.POWER_LANE)
    (OUT / "screen_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "screen_summary.txt").write_text(line + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch.json").write_text(
        json.dumps({"cells": batch}, indent=1), encoding="utf-8"
    )


def served():
    started = time.time()
    u3, hp = harness()
    m36 = hp.M36
    pick, d = m36.games()
    values = pd.read_parquet(OUT / "team_values.parquet")
    teams = pd.read_parquet(FEATURES, columns=["game_id", "home_team", "away_team"])
    g = d[["game_id", "season", "week", "home_covered", m36.SERVED_COLUMN]].merge(
        teams, on="game_id", how="left", validate="one_to_one"
    )
    term, both = game_term(g, values)
    frame = g[["game_id", "season", "week", "home_covered", m36.SERVED_COLUMN]].copy()
    frame[NAME] = term
    seasons = sorted(int(s) for s in frame["season"].unique())
    res, st = u3.read_candidate(hp, frame, NAME, 1)
    del st
    res["draws"] = hp.DRAWS
    res["base_artifact"] = str(pick.relative_to(ROOT))
    line = report("served_opener", res, seasons, both, f"elapsed {time.time() - started:.0f}s")
    batch = cells(m36, res, seasons, "opener_served", u3.POWER_LANE)
    (OUT / "served_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "served_summary.txt").write_text(line + "\n", encoding="utf-8")
    (OUT / "weak_signals_batch_served.json").write_text(
        json.dumps({"cells": batch}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    {"build": build, "screen": screen, "served": served}[sys.argv[1]]()
