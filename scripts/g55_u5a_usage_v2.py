from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g55_u2_expected_snaps import KEYS, build_candidates, load_inputs
from g55_u4_neutral_usage_ceiling import ONE_SCORE, PBP_SNAPSHOT, participation_shares
from mod57_p2l_availability_ceiling import PLAYER_SNAPSHOT

from nfl_ats.pbp import load_pbp_snapshot
from nfl_ats.pbp import snapshot_from_root as pbp_snapshot_from_root

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/g55_u5a"
ACTIVES = ROOT / "artifacts/g55_u1/deadline_actives.parquet"
V1 = ROOT / "artifacts/g55_u3/u2_rebuilt/expected_snaps.parquet"
V1_MANIFEST = ROOT / "artifacts/g55_u3/u2_rebuilt/manifest.json"
INJURIES = PLAYER_SNAPSHOT / "injuries.parquet"
SEASONS = tuple(range(2020, 2026))
MEDIAN = 0.5
DESIGNATIONS = {"Questionable": "q", "Doubtful": "d", "Out": "o"}
PRACTICE = {
    "Did Not Participate In Practice": "dnp",
    "Limited Participation in Practice": "lp",
    "Full Participation in Practice": "fp",
}
TERMS = (
    "b",
    "r",
    "nb",
    "nb_missing",
    "no_history",
    "q",
    "d",
    "o",
    "q_b",
    "d_b",
    "o_b",
    "dnp",
    "lp",
    "fp",
    "dnp_b",
    "lp_b",
    "fp_b",
    "ret",
    "ret_b",
    "log_gap",
)


def neutral_tables():
    pbp = load_pbp_snapshot(pbp_snapshot_from_root(PBP_SNAPSHOT))
    stores, summary = participation_shares(pbp)
    del pbp
    store = stores["neutral"]
    records = [
        (game_id, team, gsis_id, offense + defense)
        for (game_id, team), players in store.items()
        for gsis_id, (offense, defense) in players.items()
    ]
    shares = pd.DataFrame(records, columns=[*KEYS, "neutral"])
    covered = pd.DataFrame(list(store), columns=["game_id", "team"]).assign(covered=True)
    return shares, covered, summary


def attach_neutral(frame, shares, covered):
    out = frame[KEYS].merge(shares, on=KEYS, how="left")
    out = out.merge(covered, on=["game_id", "team"], how="left")
    if len(out) != len(frame):
        raise SystemExit("neutral join changed row count")
    cov = out["covered"].fillna(False).astype(bool).to_numpy()
    return np.where(cov, out["neutral"].fillna(0.0).to_numpy(float), np.nan)


def injury_inputs(table):
    inj = pd.read_parquet(INJURIES)
    inj = inj.loc[inj["game_type"].eq("REG") & inj["season"].isin(SEASONS)].copy()
    inj["team"] = inj["team"].astype(str)
    inj["gsis_id"] = inj["gsis_id"].astype(str)
    deadlines = table.drop_duplicates(["season", "week", "team"])[
        ["season", "week", "team", "deadline_utc"]
    ]
    inj = inj.merge(deadlines, on=["season", "week", "team"], how="inner")
    proxy = inj["observed_at_is_proxy"].astype(bool)
    seen = inj["date_modified"].where(~proxy, inj["effective_observed_at"])
    visible = seen.notna() & (seen <= inj["deadline_utc"])
    summary = {
        "rows_matched": len(inj),
        "proxy_rows": int(proxy.sum()),
        "dropped_after_deadline_by_season": inj.loc[~visible]
        .groupby("season")
        .size()
        .astype(int)
        .to_dict(),
    }
    inj = inj.loc[visible].assign(seen=seen[visible])
    inj = inj.sort_values("seen").drop_duplicates(
        ["season", "week", "team", "gsis_id"], keep="last"
    )
    inj["designation"] = inj["report_status"].map(DESIGNATIONS).fillna("none")
    inj["practice"] = inj["practice_status"].map(PRACTICE).fillna("none")
    out = inj[["season", "week", "team", "gsis_id", "designation", "practice"]]
    return out.rename(columns={"gsis_id": "player_key"}), summary


def design(frame):
    b = np.nan_to_num(frame["baseline_share"].to_numpy(float))
    gap = frame["gap"].to_numpy(float)
    has = frame["has"].to_numpy(bool)
    cols = {
        "b": b,
        "r": np.nan_to_num(frame["redistributed"].to_numpy(float)),
        "nb": np.nan_to_num(frame["nb"].to_numpy(float)),
        "nb_missing": frame["nb"].isna().to_numpy(float),
        "no_history": (~has).astype(float),
    }
    for column, levels in (("designation", ("q", "d", "o")), ("practice", ("dnp", "lp", "fp"))):
        for level in levels:
            ind = frame[column].eq(level).to_numpy(float)
            cols[level] = ind
            cols[f"{level}_b"] = ind * b
    ret = (has & (gap > 0)).astype(float)
    cols["ret"] = ret
    cols["ret_b"] = ret * b
    cols["log_gap"] = np.log1p(gap)
    return np.column_stack([cols[t] for t in TERMS])


def fit_predict(x, y, train, test):
    keep = np.ptp(x[train], axis=0) > 0
    model = QuantileRegressor(quantile=MEDIAN, alpha=0.0, solver="highs")
    model.fit(x[train][:, keep], y[train])
    pred = np.clip(model.predict(x[test][:, keep]), 0.0, float(y[train].max()))
    coef = np.zeros(x.shape[1])
    coef[keep] = model.coef_
    return pred, float(model.intercept_), coef


def mae(values):
    return float(np.mean(np.abs(values))) if len(values) else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="G55-U5a pregame snap-share forecaster v2: v1 expected share plus final "
        "injury designation, practice status, return ramp and one-score usage history, one "
        "median regression per role, leave-one-season-out; reads no scores or results."
    )
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args()
    start = time.time()

    v1_manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    if v1_manifest.get("stand_in_for_uncovered") is not False:
        raise SystemExit("v1 rebuild carries the realised-snap stand-in")
    spans = {int(f["span_games"]) for f in v1_manifest["folds"].values()}
    if len(spans) != 1:
        raise SystemExit("v1 span differs across folds")
    span = spans.pop()

    table = pd.read_parquet(ACTIVES)
    table = table.loc[table["season"].isin(SEASONS)].copy()
    table["team"] = table["team"].astype(str)
    table["game_id"] = table["game_id"].astype(str)
    known = table.groupby(["game_id", "team"])["known_at_deadline"].all().reset_index()
    known = known.loc[known["known_at_deadline"], ["game_id", "team"]]
    listed = table.merge(known, on=["game_id", "team"], how="inner")
    listed = listed.loc[listed["active"].fillna(False).astype(bool)]
    actives = listed.assign(player_key=listed["gsis_id"].astype(str), active_source="actives_table")
    actives = actives[[*KEYS, "active_source"]].drop_duplicates(KEYS)

    rosters, snaps = load_inputs(PLAYER_SNAPSHOT)
    cand, hist = build_candidates(rosters, snaps, actives)
    cand = cand.merge(known, on=["game_id", "team"], how="inner").reset_index(drop=True)
    print(f"candidates {len(cand)} at {time.time() - start:.0f}s", flush=True)

    shares, covered, part_summary = neutral_tables()
    print(f"participation loaded at {time.time() - start:.0f}s", flush=True)
    hist["neutral"] = attach_neutral(hist, shares, covered)
    alpha = 2.0 / (span + 1.0)
    state = (
        hist.groupby("player_key", sort=False)["neutral"]
        .ewm(alpha=alpha, adjust=False, ignore_na=True)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(hist.index)
        .to_numpy(float)
    )
    has = cand["row"].notna().to_numpy()
    rows = cand["row"].fillna(0).astype(int).to_numpy()
    cand["has"] = has
    cand["nb"] = np.where(has, state[rows], np.nan)
    last_neutral = np.where(has, hist["neutral"].to_numpy(float)[rows], np.nan)
    cand["t1"] = attach_neutral(cand, shares, covered)

    v1 = pd.read_parquet(V1, columns=[*KEYS, "baseline_share", "redistributed", "expected_share"])
    v1 = v1.rename(columns={"expected_share": "v1_expected_share"})
    cand = cand.merge(v1, on=KEYS, how="left")
    active = cand["active"].to_numpy(bool)
    if cand.loc[active, "v1_expected_share"].isna().any():
        raise SystemExit("active candidate without a v1 forecast")
    inj, inj_summary = injury_inputs(table)
    cand = cand.merge(inj, on=["season", "week", "team", "player_key"], how="left")
    cand[["designation", "practice"]] = cand[["designation", "practice"]].fillna("none")
    if not (cand["row"].notna().to_numpy() == has).all():
        raise SystemExit("candidate order changed")

    x = design(cand)
    season = cand["season"].to_numpy(int)
    role = cand["role"].to_numpy(object)
    targets = {
        "one_score": cand["t1"].to_numpy(float),
        "full_game": cand["realized"].fillna(0.0).to_numpy(float),
    }
    preds = {}
    coefs = {}
    for target, y in targets.items():
        ok = active & np.isfinite(y)
        for scheme in ("loso", "forward"):
            pred = np.full(len(cand), np.nan)
            for held in SEASONS:
                train_seasons = [
                    s for s in SEASONS if (s != held if scheme == "loso" else s < held)
                ]
                if not train_seasons:
                    continue
                for name in sorted(set(role[active])):
                    train = ok & (role == name) & np.isin(season, train_seasons)
                    test = active & (role == name) & (season == held)
                    if not test.any():
                        continue
                    if not train.any():
                        pred[test] = cand.loc[test, "v1_expected_share"].to_numpy(float)
                        continue
                    p, intercept, coef = fit_predict(x, y, train, test)
                    pred[test] = p
                    if scheme == "loso":
                        coefs.setdefault(target, {}).setdefault(name, {})[str(held)] = {
                            "intercept": intercept,
                            **{t: float(c) for t, c in zip(TERMS, coef, strict=True)},
                        }
            pred[~active] = 0.0
            preds[(target, scheme)] = pred
            print(f"{target} {scheme} fitted at {time.time() - start:.0f}s", flush=True)

    v1_exp = np.where(active, cand["v1_expected_share"].fillna(0.0).to_numpy(float), 0.0)
    last_full = cand["last_share"].to_numpy(float)
    naive = {
        "one_score": np.where(np.isfinite(last_neutral), last_neutral, last_full),
        "full_game": last_full,
    }
    groups = cand["position_group"].to_numpy(object)
    report = {}
    for target, y in targets.items():
        ok = active & has & np.isfinite(y)
        v2 = preds[(target, "loso")]
        fwd = preds[(target, "forward")]
        entry = {}
        for name in ["all", *sorted(set(groups[ok]))]:
            m = ok if name == "all" else ok & (groups == name)
            mf = m & (season > SEASONS[0])
            entry[name] = {
                "n": int(m.sum()),
                "v2": mae(y[m] - v2[m]),
                "v1": mae(y[m] - v1_exp[m]),
                "naive_last_game": mae(y[m] - naive[target][m]),
                "v2_minus_v1": mae(y[m] - v2[m]) - mae(y[m] - v1_exp[m]),
                "forward_2021_25": {
                    "n": int(mf.sum()),
                    "v2_forward": mae(y[mf] - fwd[mf]),
                    "v2_loso": mae(y[mf] - v2[mf]),
                    "v1": mae(y[mf] - v1_exp[mf]),
                },
                "by_season": {
                    str(s): {
                        "n": int((m & (season == s)).sum()),
                        "v2": mae(y[m & (season == s)] - v2[m & (season == s)]),
                        "v1": mae(y[m & (season == s)] - v1_exp[m & (season == s)]),
                        "naive_last_game": mae(
                            y[m & (season == s)] - naive[target][m & (season == s)]
                        ),
                    }
                    for s in SEASONS
                },
            }
        fresh = active & ~has & np.isfinite(y)
        entry["no_history_actives"] = {
            "n": int(fresh.sum()),
            "v2": mae(y[fresh] - v2[fresh]),
            "v1": mae(y[fresh] - v1_exp[fresh]),
            "zero": mae(y[fresh]),
        }
        report[target] = entry

    stability = {}
    for target, by_role in coefs.items():
        for name, folds in by_role.items():
            for term in ("intercept", *TERMS):
                vals = [f[term] for f in folds.values()]
                stability.setdefault(target, {}).setdefault(name, {})[term] = [
                    float(min(vals)),
                    float(max(vals)),
                ]

    keep = active | (~active & (cand["baseline_share"].fillna(0.0).to_numpy(float) > 0.0))
    out = cand.loc[
        keep,
        ["game_id", "team", "season", "week", "player_key", "position", "position_group", "role"],
    ].copy()
    out["gsis_id"] = out["player_key"].where(~out["player_key"].str.startswith("pfr:"))
    out["active"] = active[keep]
    out["active_source"] = np.where(active[keep], "actives_table", "not_active")
    out["baseline_share"] = cand.loc[keep, "baseline_share"].to_numpy(float)
    out["redistributed"] = cand.loc[keep, "redistributed"].to_numpy(float)
    out["v1_expected_share"] = v1_exp[keep]
    out["expected_share"] = preds[("one_score", "loso")][keep]
    out["expected_share_full_game"] = preds[("full_game", "loso")][keep]
    out["expected_share_forward"] = preds[("one_score", "forward")][keep]
    out["designation"] = cand.loc[keep, "designation"].to_numpy(object)
    out["practice"] = cand.loc[keep, "practice"].to_numpy(object)
    out["games_missed"] = cand.loc[keep, "gap"].to_numpy(int)
    out["span_games"] = span
    out = out.sort_values(["season", "week", "game_id", "team", "player_key"]).reset_index(
        drop=True
    )
    if out.loc[out["active"], "expected_share"].isna().any():
        raise SystemExit("active row without a v2 forecast")
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "expected_snaps_v2.parquet", index=False)
    manifest = {
        "script": "scripts/g55_u5a_usage_v2.py",
        "declaration": "docs/lanes/g55-u5a-usage-v2-2026-11-21.md",
        "actives": str(ACTIVES.relative_to(ROOT)),
        "stand_in_for_uncovered": False,
        "team_games": "known_at_deadline only; no designation-only rows",
        "served_column": "expected_share = one-score share forecast, LOSO over 2020-25",
        "one_score_points": ONE_SCORE,
        "v1_span_games": span,
        "not_a_look": "reads no scores, margins or ATS results; score_differential only defines "
        "the one-score target subset; MAE is a build check",
        "injury_timing": inj_summary,
        "participation": part_summary,
        "rows": len(out),
        "active_rows": int(out["active"].sum()),
        "mae": report,
        "coefficient_range_across_loso_folds": stability,
        "coefficients_by_fold": coefs,
        "realized_share_stored": False,
        "seconds": round(time.time() - start, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    brief = {
        t: {
            g: {k: v for k, v in e.items() if k not in ("by_season",)} for g, e in report[t].items()
        }
        for t in report
    }
    print(json.dumps({"mae": brief, "rows": len(out)}, indent=1))


if __name__ == "__main__":
    main()
