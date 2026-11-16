from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mod22_qb_input_repair as m22
import v4_candidate as v4
from mod27_fenced_repairs_replay import paired, record

from nfl_ats.pbp import analysis_plays

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/qb_starter_parity"
SCRATCH = ROOT / "tests/scratch/qb_starter_parity"
RELATIVE = "src/nfl_ats/players.py"
BASE_SOURCE = m22.SCRATCH / "repair_ab" / RELATIVE
SERVE_SEASON = v4.LIVE_SEASON
KNOWABLE_REPORT = ("out", "doubtful")
ON_TEAM = ("ACT", "INA")
QB_POSITION = "QB"
FAMILY = "qb_starter_parity_2026_11_16"
ARMS = {"A": "all", "A1": "knowable"}
POSITION_ANCHOR = "def _position_group(position: object) -> str:\n"
STARTER_ANCHOR = (
    '            if starter is not None:\n                player_id = str(starter["player_id"])\n'
)
HELPER = """_PARITY: dict = {"actual": {}, "present": {}, "audit": [], "mode": "", "statuses": ()}


def _parity_flagged(visible: pd.DataFrame | None) -> frozenset[str]:
    if visible is None or visible.empty:
        return frozenset()
    status = visible["report_status"].astype("string").str.strip().str.casefold()
    hit = status.isin(_PARITY["statuses"]).fillna(False)
    return frozenset(visible.loc[hit, "gsis_id"].astype(str))


"""
PATCH = """            _key = (str(game["game_id"]), team)
            _actual = _PARITY["actual"].get(_key)
            if _actual is not None:
                _previous = latest_qb_appearance.get(team)
                _previous_id = None if _previous is None else str(_previous["player_id"])
                _served_id = None if starter is None else str(starter["player_id"])
                _flagged = _previous_id is not None and _previous_id in _parity_flagged(
                    visible_injuries
                )
                _present = _PARITY["present"].get(_key)
                _gone = (
                    _previous_id is not None
                    and _present is not None
                    and _previous_id not in _present
                )
                _change = _previous_id is not None and _actual != _previous_id
                _PARITY["audit"].append(
                    (*_key, int(game["season"]), int(game["week"]), _previous_id, _served_id,
                     _actual, _flagged, _gone, _present is not None)
                )
                if _PARITY["mode"] == "all" or (_change and (_flagged or _gone)):
                    starter = pd.Series({"player_id": _actual})
"""
AUDIT_COLUMNS = (
    "game_id",
    "team",
    "season",
    "week",
    "previous",
    "served",
    "actual",
    "flagged",
    "gone",
    "roster_seen",
)


def patched_source():
    body = BASE_SOURCE.read_bytes().decode("utf-8").replace("\r\n", "\n")
    for anchor in (POSITION_ANCHOR, STARTER_ANCHOR):
        if body.count(anchor) != 1:
            raise ValueError(f"anchor count {body.count(anchor)}: {anchor[:40]!r}")
    body = body.replace(POSITION_ANCHOR, HELPER + POSITION_ANCHOR)
    body = body.replace(STARTER_ANCHOR, PATCH + STARTER_ANCHOR)
    target = SCRATCH / "parity" / RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def load_module(path, arm):
    name = f"nfl_ats._qb_parity_{arm}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def actual_starters(inputs):
    plays = analysis_plays(inputs["pbp"])
    plays = plays.loc[
        plays["passer_player_id"].notna()
        & pd.to_numeric(plays["qb_dropback"], errors="coerce").eq(1)
        & plays["season"].astype(int).lt(SERVE_SEASON)
    ].copy()
    rosters = inputs["rosters"]
    qb_ids = set(rosters.loc[rosters["position"].eq(QB_POSITION), "gsis_id"].dropna().astype(str))
    plays["is_qb"] = plays["passer_player_id"].astype(str).isin(qb_ids)
    plays["_left"] = pd.to_numeric(plays["game_seconds_remaining"], errors="coerce")
    plays = plays.sort_values(
        ["game_id", "posteam", "is_qb", "qtr", "_left"], ascending=[True, True, False, True, False]
    )
    first = plays.drop_duplicates(["game_id", "posteam"])
    actual = {
        (str(g), str(t)): str(p)
        for g, t, p in zip(
            first["game_id"], first["posteam"], first["passer_player_id"], strict=True
        )
    }
    return actual, {"team_games": len(first), "non_qb_first": int((~first["is_qb"]).sum())}


def present_sets(inputs):
    base = inputs["base"]
    rosters = inputs["rosters"]
    if "week" not in rosters.columns:
        return {}, {"roster_week": False}
    on = rosters.loc[rosters["status"].astype(str).isin(ON_TEAM)]
    groups = {
        (int(s), int(w), str(t)): frozenset(g["gsis_id"].dropna().astype(str))
        for (s, w, t), g in on.groupby(["season", "week", "team"], sort=False)
    }
    present = {}
    hist = base.loc[base["season"].astype(int).lt(SERVE_SEASON)]
    for g, s, w, h, a in zip(
        hist["game_id"],
        hist["season"],
        hist["week"],
        hist["home_team"],
        hist["away_team"],
        strict=True,
    ):
        for team in (str(h), str(a)):
            ids = groups.get((int(s), int(w), team))
            if ids is not None:
                present[(str(g), team)] = ids
    statuses = rosters["status"].astype(str).value_counts().head(12).to_dict()
    return present, {"roster_week": True, "keys": len(present), "status_counts": statuses}


def build_arms(table, qw):
    inputs = m22.load_inputs()
    actual, actual_info = actual_starters(inputs)
    present, present_info = present_sets(inputs)
    path = patched_source()
    frames = {}
    audit = None
    for arm, mode in ARMS.items():
        module = load_module(path, arm)
        module._PARITY.update(
            actual=actual, present=present, audit=[], mode=mode, statuses=KNOWABLE_REPORT
        )
        enriched = m22.enrich(module, inputs, list(v4.QB)).set_index("game_id")
        if audit is None:
            audit = pd.DataFrame(module._PARITY["audit"], columns=AUDIT_COLUMNS)
        out = qw.copy()
        hist = out["season"].lt(SERVE_SEASON).to_numpy()
        for c in v4.QB:
            new = enriched[c].reindex(out["game_id"]).to_numpy(float)
            out.loc[hist, c] = new[hist]
        frames[arm] = out
    return frames, audit, {"actual": actual_info, "present": present_info}


def audit_summary(audit, qw, frames):
    a = audit.copy()
    a["change"] = a["previous"].notna() & a["actual"].ne(a["previous"])
    a["knowable"] = a["flagged"] | a["gone"]
    a["diff_served"] = a["actual"].ne(a["served"])
    a["a1_override"] = a["change"] & a["knowable"]
    rows = {}
    for season, g in a.groupby("season"):
        rows[str(int(season))] = {
            "team_games": len(g),
            "changes": int(g["change"].sum()),
            "changes_knowable": int((g["change"] & g["knowable"]).sum()),
            "changes_flagged": int((g["change"] & g["flagged"]).sum()),
            "changes_gone": int((g["change"] & g["gone"]).sum()),
            "week1_changes": int((g["change"] & g["week"].eq(1)).sum()),
            "actual_ne_served": int(g["diff_served"].sum()),
            "a1_overrides_ne_served": int((g["a1_override"] & g["diff_served"]).sum()),
        }
    total = {k: int(sum(r[k] for r in rows.values())) for k in next(iter(rows.values()))}
    graded = a["season"].between(2020, 2025)
    identity = {}
    moved = set(a.loc[a["diff_served"], "game_id"])
    hist = qw["season"].lt(SERVE_SEASON)
    same = qw.loc[hist & ~qw["game_id"].isin(moved), "game_id"]
    for arm, frame in frames.items():
        diff = 0.0
        for c in v4.QB:
            x = frame.set_index("game_id")[c].reindex(same).to_numpy(float)
            y = qw.set_index("game_id")[c].reindex(same).to_numpy(float)
            both = ~np.isnan(x) & ~np.isnan(y)
            mismatch = int((np.isnan(x) != np.isnan(y)).sum())
            diff = max(diff, float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0)
        changed_rows = 0
        for c in v4.QB:
            x = frame[c].to_numpy(float)
            y = qw[c].to_numpy(float)
            changed_rows += int((~np.isclose(x, y, equal_nan=True)).sum())
        identity[arm] = {
            "unchanged_games": len(same),
            "unchanged_nan_mismatch_last_column": mismatch,
            "max_abs": diff,
            "changed_cells": changed_rows,
            "live_rows_changed": int(
                sum(
                    (
                        ~np.isclose(
                            frame.loc[~frame["season"].lt(SERVE_SEASON), c].to_numpy(float),
                            qw.loc[~qw["season"].lt(SERVE_SEASON), c].to_numpy(float),
                            equal_nan=True,
                        )
                    ).sum()
                    for c in v4.QB
                )
            ),
        }
    graded_rows = a.loc[graded]
    return {
        "per_season": rows,
        "total_2009_2025": total,
        "graded_2020_2025": {
            "actual_ne_served": int(graded_rows["diff_served"].sum()),
            "hindsight_overrides_in_A": int(
                (graded_rows["diff_served"] & ~graded_rows["a1_override"]).sum()
            ),
            "a1_overrides_ne_served": int(
                (graded_rows["diff_served"] & graded_rows["a1_override"]).sum()
            ),
        },
        "roster_seen_share": float(a["roster_seen"].mean()),
        "identity": identity,
    }


def versus(pop, probs, base, arm):
    y = pop["home_covered"].to_numpy(float)
    frame = pop.assign(week_block=pop["season"] * 100 + pop["week"])
    a, b = probs[arm], probs[base]
    changed = (a >= 0.5) != (b >= 0.5)
    return {
        "season_block": paired(frame, b, a, "season"),
        "week_block": paired(frame, b, a, "week_block"),
        "changed_picks": int(changed.sum()),
        "changed_record_arm": record(a[changed], y[changed]) if changed.any() else "0-0",
        "changed_record_base": record(b[changed], y[changed]) if changed.any() else "0-0",
    }


def cells(result):
    out = []
    for arm, label in (("A", "every actual starter"), ("A1", "knowable starter changes only")):
        vs = result[arm]["vs_QW"]
        acc = vs["season_block"]["accuracy_points_gain"]
        brier = vs["season_block"]["brier_gain"]
        week = vs["week_block"]["accuracy_points_gain"]
        out.append(
            {
                "name": f"{FAMILY}_{arm.lower()}_accuracy_points",
                "description": (
                    f"QW table with training quarterback columns rebuilt for {label} "
                    "(first quarterback dropback; form strictly before the game), serving "
                    "unchanged, minus QW; served four-term stage refit leave-one-season-out, "
                    f"opener grade 2020-2025, {result['games']} games, season-block bootstrap; "
                    f"record {result[arm]['record']} vs {result['QW']['record']}; changed picks "
                    f"{vs['changed_picks']} (arm {vs['changed_record_arm']}, QW "
                    f"{vs['changed_record_base']}); week-block P+ "
                    f"{week['probability_positive']:.3f}; Brier gain {brier['point']:.6f} "
                    f"P+ {brier['probability_positive']:.3f}"
                ),
                "effect": acc["point"],
                "interval_low": acc["lo"],
                "interval_high": acc["hi"],
                "probability_positive": acc["probability_positive"],
                "plain_summary": f"Train on the quarterback who actually started ({label}).",
            }
        )
    shared = {
        "family": FAMILY,
        "source": "artifacts/qb_starter_parity/results.json",
        "effect_units": "accuracy_points",
        "sample_games": result["games"],
        "season_start": 2020,
        "season_end": 2025,
        "league": "nfl",
        "category": "modeling",
        "classification": "unresolved_below_power",
        "classification_evidence": (
            "Train/serve parity of the quarterback input; interval and probability_positive "
            "reported; no resolved wrong sign and no positive control bound"
        ),
    }
    return {**shared, "cells": [{**shared, **cell} for cell in out]}


def main():
    parser = argparse.ArgumentParser(description="Train on the actual starting quarterback.")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v4.OUT = OUT
    v4.CACHED = False
    table = pd.read_parquet(v4.TABLE)
    table["game_id"] = table["game_id"].astype(str)
    q_table, _ = v4.qb_repaired(table)
    qw, _, _ = v4.forecast_weather(q_table)
    frames, audit, info = build_arms(table, qw)
    audit.to_parquet(OUT / "audit.parquet")
    summary = audit_summary(audit, qw, frames)
    summary["inputs"] = info
    (OUT / "audit.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    tables = {"S": (table, v4.FULL), "QW": (qw, v4.FULL)}
    tables.update({arm: (frame, v4.FULL) for arm, frame in frames.items()})
    pop = pd.read_parquet(v4.PICK / "per_game.parquet")
    pop["game_id"] = pop["game_id"].astype(str)
    pop = pop.reset_index(drop=True)
    result = v4.grade(tables, pop)
    ids = pop["game_id"].to_numpy()
    probs = {}
    for name in tables:
        raw = pd.read_parquet(OUT / f"opener_{name}.parquet")
        mp = raw["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
        probs[name] = v4.stage_two(pop, mp)[0]
    for arm in ARMS:
        result[arm]["vs_QW"] = versus(pop, probs, "QW", arm)
    result["audit"] = summary
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT / "cells.json").write_text(json.dumps(cells(result), indent=1), encoding="utf-8")
    brief = {n: {"record": result[n]["record"], "log_loss": result[n]["log_loss"]} for n in tables}
    brief.update({arm: {**brief[arm], "vs_QW": result[arm]["vs_QW"]} for arm in ARMS})
    print(
        json.dumps(
            {
                "brief": brief,
                "audit_total": summary["total_2009_2025"],
                "graded": summary["graded_2020_2025"],
                "identity": summary["identity"],
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
