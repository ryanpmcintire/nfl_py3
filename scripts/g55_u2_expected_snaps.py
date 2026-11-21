from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mod57_p2l_availability_ceiling import PLAYER_SNAPSHOT, ROLE_SPAN

from nfl_ats.players import (
    _position_group,
    attach_snap_player_ids,
    canonicalize_rosters,
    canonicalize_snaps,
    load_player_snapshot,
    player_snapshot_from_root,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/g55_u2"
CHECK_SEASONS = tuple(range(2020, 2026))
KEYS = ["game_id", "team", "player_key"]
ROLES = {
    "QB": ("QB",),
    "RB": ("RB", "HB", "FB"),
    "WR": ("WR",),
    "TE": ("TE",),
    "OL": ("C", "G", "OG", "OL", "OT", "T"),
    "DL": ("DE", "DL", "DT", "EDGE", "NT"),
    "LB": ("ILB", "LB", "MLB", "OLB"),
    "DB": ("CB", "DB", "FS", "S", "SAF", "SS"),
}
ROLE_OF = {position: role for role, positions in ROLES.items() for position in positions}


def role_of(position):
    if position is None or pd.isna(position):
        return "other"
    return ROLE_OF.get(str(position).split("/")[0].strip().upper(), "other")


def load_inputs(snapshot):
    _injuries, rosters_raw, snaps_raw = load_player_snapshot(player_snapshot_from_root(snapshot))
    rosters = canonicalize_rosters(rosters_raw)
    snaps = attach_snap_player_ids(canonicalize_snaps(snaps_raw), rosters)
    snaps["player_key"] = snaps["gsis_id"].astype(object).where(snaps["gsis_id"].notna(), None)
    snaps["player_key"] = snaps["player_key"].fillna("pfr:" + snaps["pfr_player_id"].astype(str))
    snaps["player_key"] = snaps["player_key"].astype(str)
    snaps["share"] = snaps["offense_pct"].fillna(0.0) + snaps["defense_pct"].fillna(0.0)
    played = snaps[["offense_snaps", "defense_snaps", "st_snaps"]].fillna(0.0).sum(axis=1)
    snaps["played"] = played.gt(0.0)
    snaps = snaps.drop_duplicates(KEYS).reset_index(drop=True)
    rosters = rosters.assign(player_key=rosters["gsis_id"].astype(str))
    return rosters, snaps


def load_actives(path, snaps):
    stand_in = snaps.loc[snaps["played"], KEYS].assign(active_source="snaps_gt_zero")
    if path is None:
        return stand_in
    given = pd.read_parquet(path)
    if "active" in given.columns:
        given = given.loc[given["active"].astype(bool)]
    given = given.assign(player_key=given["gsis_id"].astype(str), active_source="actives_table")
    given = given[[*KEYS, "active_source"]].drop_duplicates(KEYS)
    covered = given[["game_id", "team"]].drop_duplicates()
    keep = stand_in.merge(covered, on=["game_id", "team"], how="left", indicator=True)
    keep = keep.loc[keep["_merge"].eq("left_only")].drop(columns="_merge")
    return pd.concat([given, keep], ignore_index=True)


def build_candidates(rosters, snaps, actives):
    team_games = snaps.drop_duplicates(["game_id", "team"])[["game_id", "team", "season", "week"]]
    weeks = pd.concat([team_games[["season", "week"]], rosters[["season", "week"]]])
    weeks = weeks.drop_duplicates().sort_values(["season", "week"]).reset_index(drop=True)
    tick = {(int(s), int(w)): i for i, (s, w) in enumerate(weeks.itertuples(index=False))}
    team_games = team_games.assign(
        t=[
            tick[(int(s), int(w))]
            for s, w in zip(team_games["season"], team_games["week"], strict=False)
        ]
    )
    snaps = snaps.assign(
        t=[tick[(int(s), int(w))] for s, w in zip(snaps["season"], snaps["week"], strict=False)]
    )
    act = actives.merge(team_games, on=["game_id", "team"], how="inner")
    ros = rosters.merge(team_games, on=["season", "week", "team"], how="inner")
    ros = ros.rename(columns={"position": "roster_position"})
    cols = [*KEYS, "season", "week", "t"]
    cand = pd.concat([act[cols], ros[cols]], ignore_index=True).drop_duplicates(KEYS)
    cand = cand.merge(act[[*KEYS, "active_source"]], on=KEYS, how="left")
    cand["active"] = cand["active_source"].notna()
    cand = cand.merge(ros[[*KEYS, "roster_position"]], on=KEYS, how="left")
    current = snaps[[*KEYS, "position", "share"]].rename(
        columns={"position": "snap_position", "share": "realized"}
    )
    cand = cand.merge(current, on=KEYS, how="left")
    hist = snaps.loc[snaps["played"]].sort_values(["t", "player_key", "game_id"])
    hist = hist.reset_index(drop=True)
    hist["row"] = np.arange(len(hist))
    cand = cand.sort_values(["t", "player_key"]).reset_index(drop=True)
    cand = pd.merge_asof(
        cand,
        hist[["t", "player_key", "row"]],
        on="t",
        by="player_key",
        allow_exact_matches=False,
    )
    has = cand["row"].notna().to_numpy()
    rows = cand["row"].fillna(0).astype(int).to_numpy()
    last_position = np.where(has, hist["position"].to_numpy(object)[rows], None)
    cand["last_share"] = np.where(has, hist["share"].to_numpy(float)[rows], np.nan)
    last_t = np.where(has, hist["t"].to_numpy(int)[rows], -1)
    now = cand["t"].to_numpy(int)
    team = cand["team"].astype(str).to_numpy(object)
    gap = np.zeros(len(cand), dtype=int)
    for name, grp in team_games.groupby("team"):
        ts = np.sort(grp["t"].to_numpy(int))
        m = team == str(name)
        gap[m] = np.searchsorted(ts, now[m], side="left") - np.searchsorted(
            ts, last_t[m], side="right"
        )
    cand["gap"] = np.where(has, np.clip(gap, 0, None), 0)
    position = cand["snap_position"].astype(object)
    position = position.where(position.notna(), pd.Series(last_position, index=cand.index))
    position = position.where(position.notna(), cand["roster_position"].astype(object))
    cand["position"] = position
    cand["position_group"] = cand["position"].map(_position_group)
    cand["role"] = cand["position"].map(role_of)
    cand["group_code"] = cand.groupby(["game_id", "team", "role"], sort=False).ngroup()
    return cand, hist


def ewm_state(hist, span):
    alpha = 2.0 / (span + 1.0)
    state = hist.groupby("player_key", sort=False)["share"].ewm(alpha=alpha, adjust=False).mean()
    return state.reset_index(level=0, drop=True).reindex(hist.index).to_numpy(float)


def redistribute(code, active, baseline, carry):
    groups = int(code.max()) + 1
    b = np.nan_to_num(baseline, nan=0.0)
    freed = np.bincount(
        code, weights=np.where(~active & (b > 0.0), b * carry, 0.0), minlength=groups
    )
    weight = np.where(active, b, 0.0)
    total = np.bincount(code, weights=weight, minlength=groups)
    weight = np.where(active & (total[code] <= 0.0), 1.0, weight)
    cap = np.where(active, np.clip(1.0 - b, 0.0, None), 0.0)
    alloc = np.zeros_like(b)
    remaining = freed.copy()
    steps = int(np.bincount(code, weights=active.astype(float), minlength=groups).max()) + 1
    for _ in range(steps):
        open_ = active & (alloc < cap) & (weight > 0.0)
        mass = np.bincount(code, weights=np.where(open_, weight, 0.0), minlength=groups)
        give = np.divide(
            remaining[code] * weight,
            mass[code],
            out=np.zeros_like(b),
            where=open_ & (mass[code] > 0.0),
        )
        new = np.minimum(alloc + give, cap)
        remaining = np.maximum(
            remaining - np.bincount(code, weights=new - alloc, minlength=groups), 0.0
        )
        alloc = new
    return alloc


def fit_fraction(err, recv):
    pos = recv > 0.0
    if not pos.any():
        return 0.0
    z = err[pos] / recv[pos]
    order = np.argsort(z)
    cum = np.cumsum(recv[pos][order])
    k = int(np.searchsorted(cum, cum[-1] / 2.0))
    return float(np.clip(z[order][k], 0.0, 1.0))


def evaluate(cand, hist, span):
    state = ewm_state(hist, span)
    has = cand["row"].notna().to_numpy()
    rows = cand["row"].fillna(0).astype(int).to_numpy()
    baseline = np.where(has, state[rows], np.nan)
    carry = (1.0 - 2.0 / (span + 1.0)) ** cand["gap"].to_numpy(float)
    alloc = redistribute(
        cand["group_code"].to_numpy(), cand["active"].to_numpy(bool), baseline, carry
    )
    return baseline, alloc


def mae(values):
    return float(np.mean(np.abs(values))) if len(values) else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="G55-U2 expected offense+defense snap share per player-game from strictly "
        "earlier games plus the game's active list; decay and redistribution fraction chosen "
        "leave-one-season-out on snap-share error; reads no scores."
    )
    parser.add_argument(
        "--actives",
        type=Path,
        default=None,
        help="parquet with game_id, team, gsis_id (optional boolean active); team-games it "
        "does not cover fall back to the snaps-greater-than-zero stand-in",
    )
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args()

    rosters, snaps = load_inputs(PLAYER_SNAPSHOT)
    actives = load_actives(args.actives, snaps)
    cand, hist = build_candidates(rosters, snaps, actives)
    active = cand["active"].to_numpy(bool)
    has = cand["row"].notna().to_numpy()
    realized = cand["realized"].fillna(0.0).to_numpy(float)
    season = cand["season"].to_numpy(int)
    pop = active & has
    per_team_season = snaps.groupby(["season", "team"])["game_id"].nunique().max()
    span_max = int(2 * per_team_season)
    spans = list(range(1, span_max + 1))
    store = {}
    for span in spans:
        baseline, alloc = evaluate(cand, hist, span)
        store[span] = (
            (realized - np.nan_to_num(baseline))[pop].astype(np.float32),
            alloc[pop].astype(np.float32),
        )
    pop_season = season[pop]
    folds = {}
    for held in sorted(np.unique(season)):
        train = pop_season != held
        best = None
        for span in spans:
            err, recv = store[span]
            frac = fit_fraction(err[train], recv[train])
            score = mae(err[train] - frac * recv[train])
            if best is None or score < best[2]:
                best = (span, frac, score)
        folds[int(held)] = {
            "span_games": best[0],
            "redistribution_fraction": best[1],
            "train_mae": best[2],
            "span_on_grid_edge": best[0] in (spans[0], spans[-1]),
        }
    baseline = np.full(len(cand), np.nan)
    alloc = np.zeros(len(cand))
    frac_col = np.zeros(len(cand))
    span_col = np.zeros(len(cand), dtype=int)
    for span in sorted({f["span_games"] for f in folds.values()}):
        b_span, a_span = evaluate(cand, hist, span)
        for held, fit in folds.items():
            if fit["span_games"] != span:
                continue
            m = season == held
            baseline[m] = b_span[m]
            alloc[m] = a_span[m]
            frac_col[m] = fit["redistribution_fraction"]
            span_col[m] = span
    expected = np.where(active, np.nan_to_num(baseline) + frac_col * alloc, 0.0)
    harness_b, _ = evaluate(cand, hist, ROLE_SPAN)

    check = pop & np.isin(season, CHECK_SEASONS)
    groups = cand["position_group"].to_numpy(object)
    table = {}
    for name in ["all", *sorted(set(groups[check]))]:
        m = check if name == "all" else check & (groups == name)
        y = realized[m]
        table[name] = {
            "n": int(m.sum()),
            "forecast_mae": mae(y - expected[m]),
            "baseline_no_redistribution_mae": mae(y - baseline[m]),
            "span8_baseline_mae": mae(y - harness_b[m]),
            "naive_last_game_mae": mae(y - cand["last_share"].to_numpy(float)[m]),
            "n_received_mass": int((frac_col[m] * alloc[m] > 0.0).sum()),
            "received_rows_forecast_mae": mae((y - expected[m])[frac_col[m] * alloc[m] > 0.0]),
            "received_rows_baseline_mae": mae((y - baseline[m])[frac_col[m] * alloc[m] > 0.0]),
        }
        table[name]["forecast_minus_naive"] = (
            table[name]["forecast_mae"] - table[name]["naive_last_game_mae"]
        )
    fresh = active & ~has & np.isin(season, CHECK_SEASONS)
    table["no_history_actives"] = {
        "n": int(fresh.sum()),
        "forecast_mae": mae(realized[fresh] - expected[fresh]),
        "zero_forecast_mae": mae(realized[fresh]),
    }

    keep = active | (~active & (np.nan_to_num(baseline) > 0.0))
    out = cand.loc[keep, ["game_id", "team", "season", "week", "player_key", "position"]].copy()
    out["gsis_id"] = out["player_key"].where(~out["player_key"].str.startswith("pfr:"))
    out["position_group"] = cand.loc[keep, "position_group"].to_numpy()
    out["role"] = cand.loc[keep, "role"].to_numpy()
    out["active"] = active[keep]
    out["active_source"] = cand.loc[keep, "active_source"].fillna("not_active").to_numpy()
    out["baseline_share"] = baseline[keep]
    out["redistributed"] = (frac_col * alloc)[keep]
    out["expected_share"] = expected[keep]
    out["span_games"] = span_col[keep]
    out["redistribution_fraction"] = frac_col[keep]
    out = out.sort_values(["season", "week", "game_id", "team", "player_key"]).reset_index(
        drop=True
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "expected_snaps.parquet", index=False)
    manifest = {
        "script": "scripts/g55_u2_expected_snaps.py",
        "player_snapshot": str(PLAYER_SNAPSHOT.relative_to(ROOT)),
        "actives": str(args.actives) if args.actives else "snaps_gt_zero stand-in",
        "active_source_rows": {
            str(k): int(v) for k, v in out["active_source"].value_counts().items()
        },
        "not_a_look": "reads no scores, margins or ATS results; snap-share error is a build "
        "check, not a graded look",
        "unit": "offense_pct + defense_pct, the P2-L harness unit",
        "freed_mass": "inactive roster player's baseline x (1 - a)^gap, gap = his team's games "
        "missed since his last played game; moved to active same-role teammates in proportion "
        "to baseline, capped at 1, times the fitted fraction",
        "span_grid": f"integer spans 1..{span_max} (two seasons of team games from data)",
        "folds": folds,
        "check_seasons": list(CHECK_SEASONS),
        "mae_check": table,
        "rows": len(out),
        "seasons": [int(out["season"].min()), int(out["season"].max())],
        "realized_share_stored": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"folds": folds, "mae_check": table, "rows": len(out)}, indent=1))


if __name__ == "__main__":
    main()
