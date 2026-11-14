from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mod57_p2c_uncorrected_error_model import (
    ARTIFACT,
    BASE_ALPHA,
    BASE_CONFIG,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    FULL_COLUMNS,
    MARKET_ROOT,
    SCORE_SEASONS,
    SERVED_FEATURES,
    SET_NAME,
    blocks_for,
    boot,
    paired,
    per_season,
)

from nfl_ats import clv, quarterbacks
from nfl_ats.cli_commands import features as feature_cli
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.pbp import load_pbp_snapshot

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/nfl_ats/players.py"
SCRATCH = ROOT / "tests/scratch/qb_input_repair"
OUT = ROOT / "artifacts/mod22_qb_input_repair"
UV = ROOT / ".tools/uv.exe"
MANIFEST = ROOT / "data/processed/game_features_weak_stack.manifest.json"
DEPTH_ROOT = ROOT / "data/quarterbacks/depth/raw"
FAMILY = "mod22_qb_input_repair"
VARIANTS = {"orig": (), "repair_a": ("a",), "repair_b": ("b",), "repair_ab": ("a", "b")}
REPAIRS = ("repair_a", "repair_b", "repair_ab")
MODEL_QB = tuple(c for c in FULL_COLUMNS if "_qb_" in c)
EXPECTED_BASE_RECORD = (802, 701)
IDENTITY_TOLERANCE = 1e-9
PICK_THRESHOLD = 0.5
DEAD_HEAT = 0.5
INTERVAL = (0.025, 0.975)
SIDES = ("home", "away")
LISTED = ("Out", "Doubtful")
OPENING_WEEK = 1
ID_SUFFIX = "_projected_qb_id"
RELATIVE = "src/nfl_ats/players.py"
POSITION_ANCHOR = "def _position_group(position: object) -> str:\n"
CONSTANT_ANCHOR = "_REPLACEMENT_QB_EPA = -0.15\n"
ORDER_ANCHOR = "    latest_qb_appearance: dict[str, pd.Series] = {}\n"
STARTER_ANCHOR = "            starter = latest_qb_appearance.get(team)\n"
DEPTH_ANCHOR = "                    depth_row = depth_visible.iloc[0]\n"
FILL_ANCHOR = "    depth_qbs_by_team: dict[str, pd.DataFrame] = {}\n"
STATE_MARKER = "qb_state = _latest_qb_state(qb_histories, player_id"
END_MARKER = '_qb_starter_experience_log"] = math.log1p(experience)'
BLOCK_INDENT = 20
DEDENT = 4

HELPER_A = """def _qb_ids_listed_out(visible: pd.DataFrame | None) -> frozenset[str]:
    if visible is None or visible.empty:
        return frozenset()
    status = visible["report_status"].astype("string").str.strip().str.casefold()
    listed = visible.loc[status.eq(_REPORT_STATUS_OUT).fillna(False), "gsis_id"]
    return frozenset(listed.astype(str))


"""
ORDER_A = "    latest_qb_order: dict[str, tuple[str, ...]] = {}\n"
STARTER_A = """            listed_out = _qb_ids_listed_out(visible_injuries)
            if starter is not None and str(starter["player_id"]) in listed_out:
                for candidate_id in latest_qb_order.get(team, ()):
                    if candidate_id not in listed_out:
                        starter = pd.Series({"player_id": candidate_id})
                        break
"""
DEPTH_A = """                    newest_depth = depth_visible.loc[
                        depth_visible["observed_at_utc"].eq(
                            depth_visible["observed_at_utc"].iloc[0]
                        )
                    ]
                    eligible_depth = newest_depth.loc[
                        ~newest_depth["gsis_id"].astype(str).isin(listed_out)
                    ]
                    depth_row = (
                        newest_depth if eligible_depth.empty else eligible_depth
                    ).iloc[0]
"""
ORDER_LINE = 'latest_qb_order[team] = tuple(appearances["player_id"].astype(str))\n'

HELPER_B = """def _debut_qb_fill(
    qb_states: pd.DataFrame, seasons: list[int]
) -> dict[int, tuple[float, float]]:
    ordered = qb_states.sort_values(["player_id", "gameday", "game_id"])
    prior_state = ordered.groupby("player_id", sort=False)["state_qb_epa_per_dropback"].shift(1)
    primary = (
        ordered.assign(_prior_state=prior_state)
        .sort_values("qb_dropbacks", ascending=False)
        .drop_duplicates(["game_id", "team"])
    )
    debuts = primary.loc[primary["_prior_state"].isna()]
    fill: dict[int, tuple[float, float]] = {}
    for season in seasons:
        prior = debuts.loc[debuts["season"].astype(int) < season]
        graded = prior.loc[prior["qb_cpoe"].notna()]
        dropbacks = float(prior["qb_dropbacks"].sum())
        graded_dropbacks = float(graded["qb_dropbacks"].sum())
        fill[season] = (
            float((prior["qb_epa_per_dropback"] * prior["qb_dropbacks"]).sum() / dropbacks)
            if dropbacks > 0.0
            else math.nan,
            float((graded["qb_cpoe"] * graded["qb_dropbacks"]).sum() / graded_dropbacks)
            if graded_dropbacks > 0.0
            else math.nan,
        )
    return fill


"""
FILL_B = """    debut_qb_fill = _debut_qb_fill(
        qb_states, sorted({int(season) for season in result["season"]})
    )
"""
HEAD_B = """                debut_epa, debut_cpoe = debut_qb_fill.get(
                    int(game["season"]), (math.nan, math.nan)
                )
                starter_epa = debut_epa
                starter_cpoe = debut_cpoe
                experience = 0.0
                if qb_state is not None:
                    experience = float(qb_state["career_dropbacks"])
                    if pd.notna(qb_state["state_qb_epa_per_dropback"]):
                        starter_epa = float(qb_state["state_qb_epa_per_dropback"])
                    if pd.notna(qb_state["state_qb_cpoe"]):
                        starter_cpoe = float(qb_state["state_qb_cpoe"])"""


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def insert_before(body, anchor, text):
    if body.count(anchor) != 1:
        raise ValueError(f"anchor found {body.count(anchor)} times: {anchor!r}")
    return body.replace(anchor, text + anchor)


def insert_after(body, anchor, text):
    if body.count(anchor) != 1:
        raise ValueError(f"anchor found {body.count(anchor)} times: {anchor!r}")
    return body.replace(anchor, anchor + text)


def add_order_line(match):
    return match.group(0) + match.group(1) + ORDER_LINE


def apply_a(body):
    body = insert_after(body, CONSTANT_ANCHOR, '_REPORT_STATUS_OUT = "out"\n')
    body = insert_before(body, POSITION_ANCHOR, HELPER_A)
    body = insert_after(body, ORDER_ANCHOR, ORDER_A)
    body = insert_after(body, STARTER_ANCHOR, STARTER_A)
    if body.count(DEPTH_ANCHOR) != 1:
        raise ValueError("depth anchor is not unique")
    body = body.replace(DEPTH_ANCHOR, DEPTH_A)
    body, count = re.subn(
        r"(?m)^([ ]*)latest_qb_appearance\[team\] = appearances\.iloc\[0\]\n", add_order_line, body
    )
    if count != 1:
        raise ValueError(f"appearance anchor found {count} times")
    return body


def apply_b(body):
    body = insert_before(body, POSITION_ANCHOR, HELPER_B)
    body = insert_before(body, FILL_ANCHOR, FILL_B)
    lines = body.split("\n")
    anchors = [i for i, line in enumerate(lines) if STATE_MARKER in line]
    ends = [i for i, line in enumerate(lines) if END_MARKER in line]
    if len(anchors) != 1 or len(ends) != 1:
        raise ValueError(f"state block markers found {len(anchors)} and {len(ends)} times")
    start = anchors[0] + 1
    end = ends[0]
    if lines[start].strip() != "if qb_state is not None:":
        raise ValueError(f"unexpected line after state lookup: {lines[start]!r}")
    block = lines[start + 1 : end + 1]
    heads = [line.strip().split(" = ")[0] for line in block[:3]]
    if heads != ["starter_epa", "starter_cpoe", "experience"]:
        raise ValueError(f"unexpected state block head: {heads}")
    rest = []
    for line in block[3:]:
        if line.strip() and not line.startswith(" " * BLOCK_INDENT):
            raise ValueError(f"unexpected indentation: {line!r}")
        rest.append(line[DEDENT:] if line.strip() else line)
    return "\n".join([*lines[:start], *HEAD_B.split("\n"), *rest, *lines[end + 1 :]])


APPLY = {"a": apply_a, "b": apply_b}


def run_tool(args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, check=False)


def ruff(*args):
    return run_tool([str(UV), "run", "--project", str(ROOT), "--no-sync", "ruff", *args])


def make_patches():
    raw = SRC.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    body = raw.replace("\r\n", "\n")
    paths = {}
    for name, repairs in VARIANTS.items():
        patched = body
        for repair in repairs:
            patched = APPLY[repair](patched)
        target = SCRATCH / name / RELATIVE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(patched.replace("\n", newline).encode("utf-8"))
        paths[name] = target
    clean = ruff("format", "--check", "--no-force-exclude", str(paths["orig"])).returncode == 0
    report = {"newline": repr(newline), "orig_format_clean": clean}
    for name in REPAIRS:
        if clean:
            ruff("format", "--no-force-exclude", str(paths[name]))
        lint = ruff("check", "--no-force-exclude", str(paths[name]))
        fmt = ruff("format", "--check", "--no-force-exclude", str(paths[name]))
        diff = run_tool(
            ["git", "diff", "--no-index", "--no-color", f"orig/{RELATIVE}", f"{name}/{RELATIVE}"],
            cwd=SCRATCH,
        )
        data = diff.stdout.replace(b"a/orig/src/", b"a/src/").replace(
            f"b/{name}/src/".encode(), b"b/src/"
        )
        patch = SCRATCH / f"{name}.patch"
        patch.write_bytes(data)
        applied = run_tool(["git", "-C", str(ROOT), "apply", "--check", str(patch)])
        body_lines = data.split(b"\n")
        report[name] = {
            "patch": str(patch),
            "added": sum(1 for x in body_lines if x.startswith(b"+") and not x.startswith(b"+++")),
            "removed": sum(
                1 for x in body_lines if x.startswith(b"-") and not x.startswith(b"---")
            ),
            "git_apply_check_exit": applied.returncode,
            "git_apply_check_stderr": applied.stderr.decode("utf-8", "replace")[-300:],
            "ruff_check_exit": lint.returncode,
            "ruff_check_tail": lint.stdout.decode("utf-8", "replace")[-400:],
            "ruff_format_check_exit": fmt.returncode,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patches.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def load_variant(name):
    module_name = f"nfl_ats._qb_input_{name}"
    spec = importlib.util.spec_from_file_location(module_name, SCRATCH / name / RELATIVE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_inputs():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    base = feature_cli._load_features(ROOT / manifest["source_features"])
    player_snapshot = feature_cli._resolve_player_snapshot(manifest["source_player_snapshot"])
    injuries, rosters, snaps = feature_cli.load_player_snapshot(player_snapshot)
    seasons = sorted(base["season"].astype(int).unique().tolist())
    return {
        "manifest": manifest,
        "base": base,
        "injuries": injuries,
        "rosters": rosters,
        "snaps": snaps,
        "pbp": load_pbp_snapshot(
            feature_cli._resolve_pbp_snapshot(manifest["source_pbp_snapshot"])
        ),
        "stats": feature_cli.load_player_value_snapshot(
            feature_cli._resolve_player_value_snapshot(manifest["source_player_value_snapshot"])
        ),
        "depth": feature_cli.load_depth_snapshot(
            feature_cli.depth_snapshot_from_root(DEPTH_ROOT / manifest["source_depth_snapshot"])
        ),
        "rates": pd.read_parquet(ROOT / manifest["rates_destination"]),
        "first_seen": feature_cli._injury_first_seen_index(seasons),
        "hours": int(manifest["availability_configuration"]["decision_hours_before_kickoff"]),
        "fallback": manifest["injury_timestamp_fallback"],
        "captured_at": feature_cli.parse_snapshot_capture(player_snapshot.snapshot_id),
    }


def enrich(module, inputs, columns):
    frame = module.enrich_with_player_features(
        inputs["base"],
        inputs["injuries"],
        inputs["rosters"],
        inputs["snaps"],
        inputs["pbp"],
        player_stats=inputs["stats"],
        availability_rates=inputs["rates"],
        depth_charts=inputs["depth"],
        decision_hours_before_kickoff=inputs["hours"],
        injury_snapshot_captured_at=inputs["captured_at"],
        injury_timestamp_fallback=inputs["fallback"],
        injury_first_seen=inputs["first_seen"],
    )
    return frame[["game_id", "season", "week", *columns]].copy()


def compare_columns(left, right, columns):
    out = {"max_abs": 0.0, "nan_mismatch": 0, "id_mismatch": 0, "mismatch_seasons": {}}
    seasons = left["season"].to_numpy()
    for column in columns:
        if column.endswith(ID_SUFFIX):
            a = left[column].astype("string").fillna("").to_numpy()
            b = right[column].astype("string").fillna("").to_numpy()
            bad = a != b
            out["id_mismatch"] += int(bad.sum())
        else:
            a = left[column].to_numpy(float)
            b = right[column].to_numpy(float)
            bad = np.isnan(a) != np.isnan(b)
            out["nan_mismatch"] += int(bad.sum())
            both = ~np.isnan(a) & ~np.isnan(b)
            if both.any():
                gap = np.abs(a[both] - b[both])
                out["max_abs"] = max(out["max_abs"], float(gap.max()))
                bad = bad.copy()
                bad[np.flatnonzero(both)[gap > IDENTITY_TOLERANCE]] = True
        for season in np.unique(seasons[bad]):
            key = str(int(season))
            out["mismatch_seasons"][key] = out["mismatch_seasons"].get(key, 0) + int(
                bad[seasons == season].sum()
            )
    return out


def changed_mask(base, arm, columns):
    mask = np.zeros(len(base), bool)
    for column in columns:
        a = base[column].to_numpy(float)
        b = arm[column].to_numpy(float)
        same = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= IDENTITY_TOLERANCE)
        mask |= ~same
    return mask


def team_game_table(served, pbp, qb_games, canon, hours):
    stamp = "effective_observed_at" if "effective_observed_at" in canon.columns else "date_modified"
    reports = {
        (int(s), int(w), str(t)): grp
        for (s, w, t), grp in canon.groupby(["season", "week", "team"], sort=False)
    }
    drops = pbp.loc[pbp["qb_dropback"].eq(1) & pbp["passer_player_id"].notna()]
    first = drops.sort_values(["game_id", "play_id"]).drop_duplicates(["game_id", "posteam"])
    first_qb = {
        (str(g), str(t)): str(p)
        for g, t, p in zip(
            first["game_id"], first["posteam"], first["passer_player_id"], strict=True
        )
    }
    by_team = {
        (str(g), str(t)): grp for (g, t), grp in qb_games.groupby(["game_id", "team"], sort=False)
    }
    columns = ["game_id", "season", "week", "kickoff", "result"]
    for side in SIDES:
        columns += [
            f"{side}_team",
            f"{side}{ID_SUFFIX}",
            f"{side}_qb_start_probability",
            f"{side}_qb_starter_epa_per_dropback",
        ]
    rows = []
    for game in served[columns].itertuples(index=False):
        decision_at = (
            pd.Timestamp(game.kickoff) - pd.Timedelta(hours=hours)
            if pd.notna(game.kickoff)
            else None
        )
        for side in SIDES:
            team = str(getattr(game, f"{side}_team"))
            raw_id = getattr(game, f"{side}{ID_SUFFIX}")
            projected = None if pd.isna(raw_id) else str(raw_id)
            status = None
            listed = reports.get((int(game.season), int(game.week), team))
            if listed is not None and projected is not None and decision_at is not None:
                seen = listed.loc[
                    listed[stamp].le(decision_at) & listed["gsis_id"].astype(str).eq(projected)
                ]
                if not seen.empty:
                    status = seen.sort_values(stamp).iloc[-1]["report_status"]
            grp = by_team.get((str(game.game_id), team))
            row = {
                "game_id": str(game.game_id),
                "season": int(game.season),
                "week": int(game.week),
                "team": team,
                "projected": projected,
                "status": status if isinstance(status, str) else None,
                "start_probability": float(getattr(game, f"{side}_qb_start_probability")),
                "starter_epa": float(getattr(game, f"{side}_qb_starter_epa_per_dropback")),
                "played": bool(pd.notna(game.result)) and grp is not None,
                "first_qb": first_qb.get((str(game.game_id), team)),
                "projected_dropbacks": 0.0,
                "projected_epa": 0.0,
                "other_dropbacks": 0.0,
                "other_epa": 0.0,
            }
            if grp is not None:
                dropbacks = grp["qb_dropbacks"].to_numpy(float)
                epa = grp["qb_epa_per_dropback"].to_numpy(float) * dropbacks
                own = np.array([str(pid) == projected for pid in grp["player_id"]], bool)
                row["projected_dropbacks"] = float(dropbacks[own].sum())
                row["projected_epa"] = float(epa[own].sum())
                row["other_dropbacks"] = float(dropbacks[~own].sum())
                row["other_epa"] = float(epa[~own].sum())
            rows.append(row)
    table = pd.DataFrame(rows)
    table["state_weighted"] = table["starter_epa"] * table["projected_dropbacks"]
    return table


def ratio_stat(frame, num, den, seasons, rng):
    sub = frame.loc[frame["season"].isin(seasons)]
    if sub.empty or float(sub[den].sum()) <= 0.0:
        return {"rows": len(sub)}
    folds = {}
    for season in seasons:
        rest = sub.loc[sub["season"].ne(season)]
        folds[str(season)] = float(rest[num].sum() / rest[den].sum())
    keys = (sub["season"].astype(int) * 100 + sub["week"].astype(int)).to_numpy()
    grouped = sub.groupby(keys)[[num, den]].sum()
    n = grouped[num].to_numpy(float)
    d = grouped[den].to_numpy(float)
    draws = rng.integers(0, len(n), size=(BOOTSTRAP_SAMPLES, len(n)))
    values = n[draws].sum(axis=1) / d[draws].sum(axis=1)
    return {
        "value": float(sub[num].sum() / sub[den].sum()),
        "ci95": [float(np.quantile(values, q)) for q in INTERVAL],
        "rows": len(sub),
        "weight": float(sub[den].sum()),
        "loso_min": min(folds.values()),
        "loso_max": max(folds.values()),
        "loso_by_held_out_season": folds,
    }


def measure_constant(table, windows, rng):
    played = table.loc[table["played"] & table["projected"].notna()]
    absent = played.loc[played["projected_dropbacks"].eq(0.0) & played["other_dropbacks"].gt(0.0)]
    benched = played.loc[
        played["first_qb"].notna()
        & played["first_qb"].ne(played["projected"])
        & played["other_dropbacks"].gt(0.0)
    ]
    started = played.loc[
        played["first_qb"].eq(played["projected"])
        & played["projected_dropbacks"].gt(0.0)
        & played["starter_epa"].notna()
    ]
    out = {}
    for label, seasons in windows.items():
        out[label] = {
            "seasons": [min(seasons), max(seasons)],
            "replacement_when_projected_has_no_dropbacks": ratio_stat(
                absent, "other_epa", "other_dropbacks", seasons, rng
            ),
            "non_projected_qbs_when_projected_did_not_take_first_dropback": ratio_stat(
                benched, "other_epa", "other_dropbacks", seasons, rng
            ),
            "projected_starter_realised_when_he_started": ratio_stat(
                started, "projected_epa", "projected_dropbacks", seasons, rng
            ),
            "projected_starter_state_when_he_started": ratio_stat(
                started, "state_weighted", "projected_dropbacks", seasons, rng
            ),
            "absent_starter_mean_state": float(
                absent.loc[absent["season"].isin(seasons), "starter_epa"].mean()
            ),
        }
    return out


def measure_debut(states, windows, rng):
    ordered = states.sort_values(["player_id", "gameday", "game_id"])
    prior = ordered.groupby("player_id", sort=False)["state_qb_epa_per_dropback"].shift(1)
    primary = (
        ordered.assign(prior_state=prior)
        .sort_values("qb_dropbacks", ascending=False)
        .drop_duplicates(["game_id", "team"])
    )
    debuts = primary.loc[primary["prior_state"].isna()].copy()
    debuts["epa_sum"] = debuts["qb_epa_per_dropback"] * debuts["qb_dropbacks"]
    graded = debuts.loc[debuts["qb_cpoe"].notna()].copy()
    graded["cpoe_sum"] = graded["qb_cpoe"] * graded["qb_dropbacks"]
    out = {}
    for label, seasons in windows.items():
        out[label] = {
            "epa_per_dropback": ratio_stat(debuts, "epa_sum", "qb_dropbacks", seasons, rng),
            "cpoe": ratio_stat(graded, "cpoe_sum", "qb_dropbacks", seasons, rng),
        }
    return out


def frequency(table, current):
    season, week = current
    recent = table.loc[
        table["season"].ge(min(SCORE_SEASONS))
        & ~(table["season"].eq(season) & table["week"].gt(week))
    ]
    rows = []
    for key, grp in recent.groupby("season"):
        played = grp.loc[grp["played"] & grp["projected"].notna()]
        out = grp["status"].eq("Out")
        rows.append(
            {
                "season": int(key),
                "team_games": len(grp),
                "listed_out": int(out.sum()),
                "listed_doubtful": int(grp["status"].eq("Doubtful").sum()),
                "start_probability_below_half": int(
                    grp["start_probability"].lt(PICK_THRESHOLD).sum()
                ),
                "played_team_games": len(played),
                "not_first_dropback": int(
                    (played["first_qb"].notna() & played["first_qb"].ne(played["projected"])).sum()
                ),
                "no_dropbacks": int(played["projected_dropbacks"].eq(0.0).sum()),
                "listed_out_but_took_first_dropback": int(
                    (played["status"].eq("Out") & played["first_qb"].eq(played["projected"])).sum()
                ),
                "all_five_missing": int(grp["start_probability"].isna().sum()),
                "epa_cpoe_missing_only": int(
                    (grp["starter_epa"].isna() & grp["start_probability"].notna()).sum()
                ),
            }
        )
    return rows


def evaluate(features, name):
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(features[list(MODEL_QB)], index=False).to_numpy().tobytes()
    ).hexdigest()[:12]
    original = FEATURE_SETS[SET_NAME]
    FEATURE_SETS[SET_NAME] = FULL_COLUMNS
    try:
        config = {
            **BASE_CONFIG,
            "ridge_alpha": BASE_ALPHA,
            "mod22_qb_input_arm": f"{name}_{digest}",
        }
        scored = clv.opener_pick_evaluation(
            MARKET_ROOT,
            features,
            active_model_config=config,
            min_train_games=DEFAULT_MIN_TRAIN_GAMES,
        )
    finally:
        FEATURE_SETS[SET_NAME] = original
    return scored.set_index("game_id")


def arm_report(per_game, base, arm, mask, rng):
    ids = per_game["game_id"]
    pb = base["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
    pa = arm["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
    rb = base["residual_at_open"].reindex(ids).to_numpy(float)
    ra = arm["residual_at_open"].reindex(ids).to_numpy(float)
    y = per_game["home_covered"].to_numpy(float)
    move = per_game["open_move"].to_numpy(float)
    actual = per_game["margin_vs_open"].to_numpy(float)
    gain_close = np.abs(rb - move) - np.abs(ra - move)
    gain_result = np.abs(rb - actual) - np.abs(ra - actual)
    out = {"missing": int(np.isnan(pa).sum())}
    for pop, chosen in (("all", np.ones(len(per_game), bool)), ("affected", mask)):
        games = int(chosen.sum())
        entry = {"games": games}
        if games:
            sub = per_game.loc[chosen].reset_index(drop=True)
            block = blocks_for(sub)
            hit_base = (pb[chosen] >= PICK_THRESHOLD) == (y[chosen] == 1)
            hit_arm = (pa[chosen] >= PICK_THRESHOLD) == (y[chosen] == 1)
            entry["base_record"] = f"{int(hit_base.sum())}-{int(games - hit_base.sum())}"
            entry["repair_record"] = f"{int(hit_arm.sum())}-{int(games - hit_arm.sum())}"
            entry.update(paired(sub, pa[chosen], pb[chosen], rng))
            entry["mae_vs_close"] = {
                "base": float(np.abs(rb[chosen] - move[chosen]).mean()),
                "repair": float(np.abs(ra[chosen] - move[chosen]).mean()),
            }
            entry["mae_vs_result"] = {
                "base": float(np.abs(rb[chosen] - actual[chosen]).mean()),
                "repair": float(np.abs(ra[chosen] - actual[chosen]).mean()),
            }
            entry["mae_improvement_vs_close"] = boot(gain_close[chosen], block, rng)
            entry["mae_improvement_vs_result"] = boot(gain_result[chosen], block, rng)
            entry["per_season_repair"] = per_season(sub, pa[chosen])
            entry["per_season_base"] = per_season(sub, pb[chosen])
        out[pop] = entry
    return out


def live_report(base, arm, features_base, features_arm, season):
    rows = base.loc[base["season"].eq(season) & base["margin_vs_open"].notna()]
    rows = rows.loc[rows["margin_vs_open"].ne(0.0)]
    if rows.empty:
        return {"games": 0}
    ids = rows.index
    covered = rows["margin_vs_open"].gt(0.0).to_numpy()
    hit_base = (rows["home_cover_probability_at_open"].to_numpy(float) >= PICK_THRESHOLD) == covered
    hit_arm = (
        arm["home_cover_probability_at_open"].reindex(ids).to_numpy(float) >= PICK_THRESHOLD
    ) == covered
    left = features_base.set_index("game_id").loc[ids]
    right = features_arm.set_index("game_id").loc[ids]
    mask = changed_mask(left, right, MODEL_QB)
    return {
        "games": len(rows),
        "base_record": f"{int(hit_base.sum())}-{int(len(rows) - hit_base.sum())}",
        "repair_record": f"{int(hit_arm.sum())}-{int(len(rows) - hit_arm.sum())}",
        "affected_games": int(mask.sum()),
        "affected_base_record": (
            f"{int(hit_base[mask].sum())}-{int(mask.sum() - hit_base[mask].sum())}"
        ),
        "affected_repair_record": (
            f"{int(hit_arm[mask].sum())}-{int(mask.sum() - hit_arm[mask].sum())}"
        ),
        "changed_picks": int((hit_base != hit_arm).sum()),
    }


def current_changes(orig, arm, current):
    season, week = current
    a = orig.set_index("game_id")
    b = arm.set_index("game_id")
    rows = []
    for game_id in a.index[(a["season"] == season) & (a["week"] == week)]:
        for side in SIDES:
            names = [c for c in a.columns if c.startswith(f"{side}_qb_")]
            before = [float(a.at[game_id, c]) for c in names]
            after = [float(b.at[game_id, c]) for c in names]
            ids = (
                str(a.at[game_id, f"{side}{ID_SUFFIX}"]),
                str(b.at[game_id, f"{side}{ID_SUFFIX}"]),
            )
            same = all(
                (np.isnan(x) and np.isnan(y)) or abs(x - y) <= IDENTITY_TOLERANCE
                for x, y in zip(before, after, strict=True)
            )
            if ids[0] != ids[1] or not same:
                rows.append(
                    {
                        "game_id": game_id,
                        "side": side,
                        "projected_before": ids[0],
                        "projected_after": ids[1],
                        "before": dict(zip(names, before, strict=True)),
                        "after": dict(zip(names, after, strict=True)),
                    }
                )
    return rows


def show(stat, scale=1.0):
    if not stat or "mean" not in stat:
        return "n/a"
    low, high = stat["ci95"]
    return (
        f"{stat['mean'] * scale:+.4f} [{low * scale:+.4f},{high * scale:+.4f}] "
        f"P+ {stat['probability_positive']:.3f}"
    )


def show_ratio(stat):
    if "value" not in stat:
        return f"n/a rows {stat.get('rows')}"
    return (
        f"{stat['value']:+.4f} [{stat['ci95'][0]:+.4f},{stat['ci95'][1]:+.4f}] "
        f"rows {stat['rows']} weight {stat['weight']:.0f} "
        f"held-out range {stat['loso_min']:+.4f}..{stat['loso_max']:+.4f}"
    )


def build_cells(reports):
    plain_text = {
        "repair_a": "When the listed starting quarterback is ruled out, the model reads the next "
        "quarterback in line instead of the one who will not play.",
        "repair_b": "When a quarterback has never started before, the model fills his numbers with "
        "what first-time starters averaged in earlier seasons instead of leaving them blank.",
        "repair_ab": "Both quarterback input fixes together: read the next quarterback when the "
        "starter is ruled out, and fill a first-time starter's numbers with the earlier-season "
        "average for first-time starters.",
    }
    metrics = (
        ("accuracy_points", "accuracy_points", "opener pick accuracy"),
        ("log_loss_improvement", "log_loss_improvement", "opener log loss"),
        ("mae_improvement_vs_close", "mae_improvement", "margin error against the closing line"),
    )
    cells = []
    for arm in REPAIRS:
        for pop in ("all", "affected"):
            entry = reports[arm][pop]
            if not entry["games"]:
                continue
            for key, units, words in metrics:
                stat = entry[key]
                low, high = stat["ci95"]
                probability = stat["probability_positive"]
                if low == high == 0.0:
                    probability = DEAD_HEAT
                cells.append(
                    {
                        "name": f"mod22_qb_input_{arm}_{pop}_{units}",
                        "description": (
                            f"Served base ridge refit with quarterback input {arm} minus served "
                            f"base, {pop} games ({entry['games']} of "
                            f"{reports[arm]['all']['games']}), "
                            f"{words}, opener grade {min(SCORE_SEASONS)}-{max(SCORE_SEASONS)}, "
                            f"week-block bootstrap; changed picks {entry['changed_pick_games']} "
                            f"(repair {entry['new_wins_on_changed']}, base "
                            f"{entry['old_wins_on_changed']})"
                        ),
                        "source": "artifacts/mod22_qb_input_repair/results.json",
                        "effect": stat["mean"],
                        "effect_units": units,
                        "interval_low": low,
                        "interval_high": high,
                        "probability_positive": probability,
                        "sample_games": entry["games"],
                        "season_start": min(SCORE_SEASONS),
                        "season_end": max(SCORE_SEASONS),
                        "league": "nfl",
                        "family": FAMILY,
                        "category": "health",
                        "classification": "unresolved_below_power",
                        "classification_evidence": (
                            "Input-definition repair touching a small share of games; interval and "
                            "probability_positive reported; no refuted mechanism and no positive "
                            "control bound, so no closing ground applies"
                        ),
                        "plain_summary": plain_text[arm],
                    }
                )
    return {"family": FAMILY, "cells": cells}


def write_outputs(state, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(state, indent=1, default=plain), encoding="utf-8")
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    state = {"family": FAMILY}
    lines = []
    served = pd.read_parquet(SERVED_FEATURES)
    inputs = load_inputs()
    manifest = inputs["manifest"]
    state["snapshots"] = {
        k: v for k, v in manifest.items() if k.startswith("source_") and isinstance(v, str)
    }
    qb_columns = [c for c in served.columns if "_qb_" in c]
    canon = feature_cli.canonicalize_injuries(
        inputs["injuries"], timestamp_fallback=inputs["fallback"], first_seen=inputs["first_seen"]
    )
    basis = pd.crosstab(canon["season"], canon["observed_at_basis"])
    observed = pd.to_datetime(inputs["depth"]["observed_at_utc"], utc=True, errors="coerce")
    state["timestamps"] = {
        "decision_hours_before_kickoff": inputs["hours"],
        "depth_chart_observed_from": str(observed.min()),
        "depth_chart_observed_to": str(observed.max()),
        "injury_timestamp_basis_by_season": {
            str(season): {k: int(v) for k, v in row.items() if v}
            for season, row in basis.loc[basis.index >= min(SCORE_SEASONS)].iterrows()
        },
    }
    lines.append(f"timestamps {json.dumps(state['timestamps'])}")
    qb_games = quarterbacks.build_qb_game_metrics(inputs["pbp"])
    table = team_game_table(served, inputs["pbp"], qb_games, canon, inputs["hours"])
    latest = int(served["season"].max())
    pending = served.loc[served["season"].eq(latest) & served["result"].isna(), "week"]
    current = (latest, int(pending.min()) if len(pending) else int(served["week"].max()))
    state["current_week"] = list(current)
    state["frequency"] = frequency(table, current)
    lines.append(f"frequency through {current}")
    lines.append(pd.DataFrame(state["frequency"]).to_string(index=False))
    completed = sorted(
        int(season) for season, grp in served.groupby("season") if grp["result"].notna().all()
    )
    windows = {"all_completed": completed, "score": list(SCORE_SEASONS)}
    state["constant"] = measure_constant(table, windows, rng)
    for label, block in state["constant"].items():
        lines.append(f"constant window {label} {block['seasons']}")
        for key, stat in block.items():
            if isinstance(stat, dict):
                lines.append(f"  {key}: {show_ratio(stat)}")
        lines.append(f"  absent_starter_mean_state: {block['absent_starter_mean_state']:+.4f}")
    defaults = inspect.signature(quarterbacks.build_qb_states).parameters
    enrich_defaults = inspect.signature(load_variant("orig").enrich_with_player_features).parameters
    state["qb_state_parameters"] = {
        "span": enrich_defaults["qb_span"].default,
        "min_dropbacks": enrich_defaults["qb_min_dropbacks"].default,
        "offseason_retention": enrich_defaults["offseason_retention"].default,
        "build_qb_states_own_default_min_dropbacks": defaults["min_dropbacks"].default,
    }
    states = quarterbacks.build_qb_states(
        qb_games,
        inputs["base"],
        span=enrich_defaults["qb_span"].default,
        min_dropbacks=enrich_defaults["qb_min_dropbacks"].default,
        offseason_retention=enrich_defaults["offseason_retention"].default,
    )
    state["debut"] = measure_debut(states, windows, rng)
    for label, block in state["debut"].items():
        for key, stat in block.items():
            lines.append(f"debut {label} {key}: {show_ratio(stat)}")
    write_outputs(state, lines)

    per_game = pd.read_parquet(ARTIFACT)
    per_game = per_game.loc[per_game["season"].isin(SCORE_SEASONS)].reset_index(drop=True)
    ids = per_game["game_id"]
    y = per_game["home_covered"].to_numpy(float)
    scored = {"base": evaluate(served, "base")}
    base_p = scored["base"]["home_cover_probability_at_open"].reindex(ids).to_numpy(float)
    hits = int(((base_p >= PICK_THRESHOLD) == (y == 1)).sum())
    state["identity_model"] = {
        "games": len(per_game),
        "record": [hits, len(per_game) - hits],
        "expected": list(EXPECTED_BASE_RECORD),
        "max_abs_vs_served_artifact": float(
            np.nanmax(np.abs(base_p - per_game["model_probability"].to_numpy(float)))
        ),
        "missing": int(np.isnan(base_p).sum()),
    }
    lines.append(f"identity_model {json.dumps(state['identity_model'])}")
    write_outputs(state, lines)
    if (hits, len(per_game) - hits) != EXPECTED_BASE_RECORD:
        lines.append("STOP: served base record not reproduced")
        write_outputs(state, lines)
        sys.exit(3)

    frames = {"orig": enrich(load_variant("orig"), inputs, qb_columns)}
    aligned = frames["orig"].set_index("game_id").loc[served["game_id"]].reset_index()
    state["identity_features"] = compare_columns(served, aligned, qb_columns)
    lines.append(f"identity_features {json.dumps(state['identity_features'])}")
    write_outputs(state, lines)
    played_seasons = {str(s) for s in completed}
    if played_seasons.intersection(state["identity_features"]["mismatch_seasons"]):
        lines.append("STOP: scratch copy of the served builder does not reproduce the served table")
        write_outputs(state, lines)
        sys.exit(4)

    fill_module = load_variant("repair_b")
    state["debut_fill_by_target_season"] = {
        str(season): list(values)
        for season, values in fill_module._debut_qb_fill(states, [*SCORE_SEASONS, latest]).items()
    }
    lines.append(f"debut_fill_by_target_season {json.dumps(state['debut_fill_by_target_season'])}")
    reports = {}
    state["repairs"] = reports
    state["live_season"] = {}
    state["current_week_changes"] = {}
    base_model = served.set_index("game_id").loc[ids, list(MODEL_QB)]
    for name in REPAIRS:
        frames[name] = enrich(load_variant(name), inputs, qb_columns)
        arm = frames[name].set_index("game_id").loc[served["game_id"]]
        features = served.copy()
        for column in qb_columns:
            features[column] = arm[column].to_numpy()
        scored[name] = evaluate(features, name)
        mask = changed_mask(
            base_model, features.set_index("game_id").loc[ids, list(MODEL_QB)], MODEL_QB
        )
        reports[name] = arm_report(per_game, scored["base"], scored[name], mask, rng)
        full_mask = changed_mask(served, features, MODEL_QB)
        reports[name]["changed_games_by_season"] = {
            str(int(season)): int(count)
            for season, count in served.loc[full_mask].groupby("season").size().items()
        }
        state["live_season"][name] = live_report(
            scored["base"], scored[name], served, features, latest
        )
        state["current_week_changes"][name] = current_changes(frames["orig"], frames[name], current)
        lines.append(
            f"== {name}: changed games by season {reports[name]['changed_games_by_season']}"
        )
        for pop in ("all", "affected"):
            entry = reports[name][pop]
            if not entry["games"]:
                lines.append(f"  {pop}: 0 games")
                continue
            lines.append(
                f"  {pop}: games {entry['games']} base {entry['base_record']} repair "
                f"{entry['repair_record']} changed picks {entry['changed_pick_games']} "
                f"(repair {entry['new_wins_on_changed']}, base {entry['old_wins_on_changed']})"
            )
            lines.append(f"    accuracy_points {show(entry['accuracy_points'])}")
            lines.append(f"    log_loss_improvement {show(entry['log_loss_improvement'])}")
            lines.append(f"    brier_improvement {show(entry['brier_improvement'])}")
            lines.append(
                f"    mae_vs_close base {entry['mae_vs_close']['base']:.4f} repair "
                f"{entry['mae_vs_close']['repair']:.4f} improvement "
                f"{show(entry['mae_improvement_vs_close'])}"
            )
            lines.append(
                f"    mae_vs_result base {entry['mae_vs_result']['base']:.4f} repair "
                f"{entry['mae_vs_result']['repair']:.4f} improvement "
                f"{show(entry['mae_improvement_vs_result'])}"
            )
        lines.append(f"  live season {latest}: {json.dumps(state['live_season'][name])}")
        for change in state["current_week_changes"][name]:
            lines.append(
                f"  week {current[1]} {change['game_id']} {change['side']} "
                f"{change['projected_before']} -> {change['projected_after']} "
                + " ".join(
                    f"{k.split('_qb_')[1]}={change['before'][k]:.3f}->{change['after'][k]:.3f}"
                    for k in change["before"]
                )
            )
        write_outputs(state, lines)
    batch = build_cells(reports)
    (OUT / "cells.json").write_text(json.dumps(batch, indent=1, default=plain), encoding="utf-8")
    state["looks"] = {
        "arms": len(REPAIRS),
        "populations": 2,
        "metrics": 5,
        "recorded_cells": len(batch["cells"]),
    }
    lines.append(f"looks {json.dumps(state['looks'])}")
    lines.append("DONE")
    write_outputs(state, lines)


def constant_by_listing():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    served = pd.read_parquet(SERVED_FEATURES)
    inputs = load_inputs()
    canon = feature_cli.canonicalize_injuries(
        inputs["injuries"], timestamp_fallback=inputs["fallback"], first_seen=inputs["first_seen"]
    )
    qb_games = quarterbacks.build_qb_game_metrics(inputs["pbp"])
    table = team_game_table(served, inputs["pbp"], qb_games, canon, inputs["hours"])
    completed = sorted(
        int(season) for season, grp in served.groupby("season") if grp["result"].notna().all()
    )
    windows = {"all_completed": completed, "score": list(SCORE_SEASONS)}
    played = table.loc[table["played"] & table["projected"].notna()]
    absent = played.loc[played["projected_dropbacks"].eq(0.0) & played["other_dropbacks"].gt(0.0)]
    listed = absent["status"].isin(LISTED)
    opening = absent["week"].eq(OPENING_WEEK)
    subsets = {
        "listed_out": absent.loc[absent["status"].eq(LISTED[0])],
        "listed_out_or_doubtful": absent.loc[listed],
        "not_listed_opening_week": absent.loc[~listed & opening],
        "not_listed_later_weeks": absent.loc[~listed & ~opening],
    }
    out = {"replacement_epa_per_dropback": {}, "absent_starter_mean_state": {}}
    lines = []
    for label, seasons in windows.items():
        out["replacement_epa_per_dropback"][label] = {}
        out["absent_starter_mean_state"][label] = {}
        for name, sub in subsets.items():
            stat = ratio_stat(sub, "other_epa", "other_dropbacks", seasons, rng)
            state = float(sub.loc[sub["season"].isin(seasons), "starter_epa"].mean())
            out["replacement_epa_per_dropback"][label][name] = stat
            out["absent_starter_mean_state"][label][name] = state
            lines.append(f"{label} {name}: {show_ratio(stat)} starter state {state:+.4f}")
    benched = played.loc[played["first_qb"].notna() & played["first_qb"].ne(played["projected"])]
    recent = benched.loc[benched["season"].ge(min(SCORE_SEASONS))]
    split = pd.crosstab(recent["season"], recent["week"].eq(OPENING_WEEK))
    out["not_first_dropback_by_opening_week"] = {
        str(int(season)): {("opening_week" if k else "later_weeks"): int(v) for k, v in row.items()}
        for season, row in split.iterrows()
    }
    lines.append(f"not_first_dropback {json.dumps(out['not_first_dropback_by_opening_week'])}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "constant_by_listing.json").write_text(
        json.dumps(out, indent=1, default=plain), encoding="utf-8"
    )
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("patches", "run", "constant"))
    args = parser.parse_args()
    if args.step == "patches":
        make_patches()
        return
    if args.step == "constant":
        constant_by_listing()
        return
    try:
        run()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
