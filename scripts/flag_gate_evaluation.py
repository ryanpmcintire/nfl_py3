from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, binomtest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nfl_ats import forecast_cold_visitor_tilt_overlay as cold_overlay  # noqa: E402
from nfl_ats import pick_probability_fit as ppf  # noqa: E402
from nfl_ats.pick_probability import (  # noqa: E402
    COUNTED_FLAG_COLUMNS,
    FLAG_DISCOVERY_DATES,
    FLAG_SUM_COLUMN,
    signed_composition_flags,
)

SEASON = 2026
SEED = 20261114
DRAWS = 20000
NULL_RATE = 0.50
TRUE_RATES = (0.52, 0.54, 0.56)
TARGET_PROBABILITY = 0.8
LAST_WEEK = 18
EXACT_HORIZON_FIRES = 20000
REPORT_WEEKS = (4, 7, 10, 11, 14, 18)
OUT = REPO / "artifacts" / "flag_gate_evaluation"
FIT_ROOT = REPO / "artifacts" / "pick_probability"
AUDIT_GAMES = REPO / "artifacts" / "flip_attribution_audit" / "games.csv"
DATA_ROOT = REPO / "data"


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _logit(probability):
    clipped = np.clip(
        np.asarray(probability, dtype=float),
        ppf.PROBABILITY_EPSILON,
        1.0 - ppf.PROBABILITY_EPSILON,
    )
    return np.log(clipped / (1.0 - clipped))


def exact_record(wins: int, losses: int) -> dict:
    games = wins + losses
    if games == 0:
        return {"wins": 0, "losses": 0, "games": 0}
    test = binomtest(wins, games, 0.5)
    interval = test.proportion_ci(confidence_level=0.95, method="exact")
    return {
        "wins": wins,
        "losses": losses,
        "games": games,
        "rate": wins / games,
        "p_two_sided": float(test.pvalue),
        "interval_low": float(interval.low),
        "interval_high": float(interval.high),
        "probability_positive": float(beta.sf(0.5, wins + 1, losses + 1)),
    }


def fit_history() -> pd.DataFrame:
    rows = []
    for directory in sorted(path for path in FIT_ROOT.iterdir() if path.is_dir()):
        meta_path = directory / "metadata.json"
        coef_path = directory / "coefficients.json"
        stamp = pd.to_datetime(directory.name, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
        if pd.isna(stamp) or not meta_path.is_file() or not coef_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        coef = json.loads(coef_path.read_text(encoding="utf-8"))
        counted = meta.get("counted_flag_columns")
        rows.append(
            {
                "artifact": directory.name,
                "stamp": stamp,
                "activated": bool(meta.get("activated", True)),
                "gated": bool(meta.get("post_discovery_flag_records")),
                "counted": tuple(counted) if counted is not None else tuple(COUNTED_FLAG_COLUMNS),
                "flag_coefficient": float(
                    (coef.get("coefficients") or {}).get("flag_sum", float("nan"))
                ),
                "records": meta.get("post_discovery_flag_records") or {},
            }
        )
    return pd.DataFrame(rows).sort_values("stamp").reset_index(drop=True)


def in_force(history: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    eligible = history.loc[history["activated"] & history["stamp"].le(when)]
    return eligible.iloc[-1]


def season_flags() -> pd.DataFrame:
    cold_overlay._write_cached_bulletin = lambda *args, **kwargs: None
    schedules, _team_stats = ppf.load_snapshot(ppf.latest_snapshot(DATA_ROOT / "raw"))
    incidents = ppf._arrest_incidents(DATA_ROOT)
    protection = ppf._protection_back_side(schedules, DATA_ROOT)
    games = schedules.loc[
        schedules["game_type"].astype(str).eq("REG")
        & pd.to_numeric(schedules["season"], errors="coerce").eq(SEASON)
    ].copy()
    games["game_id"] = games["game_id"].astype(str)
    lines: dict[str, float] = {}
    for week in sorted(int(value) for value in games["week"].unique()):
        capture = ppf.load_splash_capture(DATA_ROOT, SEASON, week)
        if capture is not None:
            lines.update(ppf.splash_decision_lines(capture))
    games["opener_home_spread"] = games["game_id"].map(lines)
    games["kickoff"] = ppf._kickoff_utc(games)
    lined = games.loc[games["opener_home_spread"].notna()].reset_index(drop=True)
    forecasts = ppf.fetch_tuesday_noon_forecast_temps_fail_open(
        ppf.games_for_forecast_fetch(lined, schedules),
        DATA_ROOT.resolve().parent / "registry" / ppf.STATION_MAP_RELATIVE_PATH,
    )
    games = games.reset_index(drop=True)
    flags = signed_composition_flags(
        games,
        schedules,
        incidents=incidents,
        forecasts_tuesday_noon=forecasts,
        protection_back_side=protection,
    )
    joined = games.merge(
        flags[["game_id", *COUNTED_FLAG_COLUMNS]], on="game_id", how="left", validate="one_to_one"
    )
    joined["week"] = joined["week"].astype(int)
    joined["gameday"] = pd.to_datetime(joined["gameday"], errors="coerce")
    result = pd.to_numeric(joined["result"], errors="coerce")
    joined["covered"] = np.sign(result - joined["opener_home_spread"])
    for column in COUNTED_FLAG_COLUMNS:
        value = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
        after = joined["gameday"].gt(pd.Timestamp(FLAG_DISCOVERY_DATES[column]))
        joined[column] = value.where(after, 0.0)
        joined[f"{column}_outcome"] = np.sign(value) * joined["covered"]
    return joined.sort_values(["gameday", "kickoff", "game_id"]).reset_index(drop=True)


def flag_tables(flagged: pd.DataFrame, current_week: int) -> tuple[dict, dict]:
    table: dict[str, dict] = {}
    computed: dict[str, str] = {}
    for column in COUNTED_FLAG_COLUMNS:
        fired = flagged.loc[flagged[column].ne(0.0)]
        outcome = fired[f"{column}_outcome"]
        by_week = {}
        running_wins = 0
        running_losses = 0
        for week, group in fired.groupby("week"):
            wins = int(group[f"{column}_outcome"].gt(0.0).sum())
            losses = int(group[f"{column}_outcome"].lt(0.0).sum())
            running_wins += wins
            running_losses += losses
            by_week[int(week)] = {
                "wins": wins,
                "losses": losses,
                "unsettled": int(group[f"{column}_outcome"].isna().sum()),
                "running": f"{running_wins}-{running_losses}",
                "winning_after": running_wins > running_losses,
            }
        wins = int(outcome.gt(0.0).sum())
        losses = int(outcome.lt(0.0).sum())
        computed[column] = f"{wins}-{losses}"
        table[column] = {
            "discovery": FLAG_DISCOVERY_DATES[column],
            "record": f"{wins}-{losses}",
            "exact": exact_record(wins, losses),
            "fires_through_current_week": int(fired["week"].le(current_week).sum()),
            "fires_known_later": int(fired["week"].gt(current_week).sum()),
            "last_fire_week": int(fired["week"].max()) if len(fired) else None,
            "by_week": by_week,
        }
    return table, computed


def gate_transitions(history: pd.DataFrame) -> list[dict]:
    rows = []
    previous = None
    for row in history.loc[history["activated"]].itertuples(index=False):
        state = (row.gated, row.counted)
        if state != previous:
            rows.append(
                {
                    "artifact": row.artifact,
                    "gated": row.gated,
                    "counted": list(row.counted),
                    "flag_coefficient": row.flag_coefficient,
                    "records": row.records,
                }
            )
        previous = state
    return rows


def refit_variants(per_game: pd.DataFrame, gate_now: tuple[str, ...]) -> dict:
    variants = {
        "gate_now": gate_now,
        "protection_only": ("flag_protection",),
        "all_flags": tuple(COUNTED_FLAG_COLUMNS),
        "none": (),
    }
    target_column = per_game["home_covered"].astype(float)
    seasons = sorted(int(value) for value in per_game["season"].unique())
    out: dict[str, dict] = {}
    for name, columns in variants.items():
        frame = per_game.copy()
        frame[FLAG_SUM_COLUMN] = (
            frame[list(columns)].astype(float).sum(axis=1).astype(float) if columns else 0.0
        )
        means, stds = ppf._standardisers(frame)
        coefficients = ppf._natural_coefficients(
            ppf._fit_logit(
                ppf._design(frame, means, stds), target_column.to_numpy(), ppf.FIT_RIDGE
            ),
            means,
            stds,
        )
        held_probability = pd.Series(np.nan, index=frame.index, dtype=float)
        fold_flag = {}
        for held in seasons:
            train = frame.loc[frame["season"].ne(held)]
            test = frame.loc[frame["season"].eq(held)]
            fold_means, fold_stds = ppf._standardisers(train)
            fold_beta = ppf._fit_logit(
                ppf._design(train, fold_means, fold_stds),
                train["home_covered"].astype(float).to_numpy(),
                ppf.FIT_RIDGE,
            )
            z = np.clip(ppf._design(test, fold_means, fold_stds) @ fold_beta, -35.0, 35.0)
            held_probability.loc[test.index] = 1.0 / (1.0 + np.exp(-z))
            fold_flag[str(held)] = float(
                ppf._natural_coefficients(fold_beta, fold_means, fold_stds)[FLAG_SUM_COLUMN]
            )
        right = held_probability.ge(0.5).astype(float).eq(target_column)
        log_loss = -float(
            np.mean(
                target_column * np.log(held_probability)
                + (1.0 - target_column) * np.log(1.0 - held_probability)
            )
        )
        out[name] = {
            "columns": list(columns),
            "coefficients": {key: float(value) for key, value in coefficients.items()},
            "flag_coefficient": float(coefficients[FLAG_SUM_COLUMN]),
            "fold_flag_coefficients": fold_flag,
            "out_of_season_record": f"{int(right.sum())}-{int((~right).sum())}",
            "out_of_season_log_loss": log_loss,
            "games_with_nonzero_sum": int(frame[FLAG_SUM_COLUMN].ne(0.0).sum()),
        }
    return out


def week_clock(flagged: pd.DataFrame) -> dict[int, dict]:
    out = {}
    for week, group in flagged.groupby("week"):
        first = group["gameday"].min()
        sunday = first + pd.Timedelta(days=(6 - first.weekday()) % 7)
        out[int(week)] = {
            "deadline": pd.Timestamp(f"{sunday.date()} 16:00", tz="America/New_York").tz_convert(
                "UTC"
            ),
            "tuesday": pd.Timestamp(f"{(sunday - pd.Timedelta(days=5)).date()} 23:59", tz="UTC"),
        }
    return out


def _side(value: float) -> str:
    return "HOME" if value >= 0.0 else "AWAY"


def replay_rows(
    flagged: pd.DataFrame, history: pd.DataFrame, refits: dict, active: pd.Series
) -> pd.DataFrame:
    audit = pd.read_csv(AUDIT_GAMES)
    audit["home_covered"] = audit["home_covered"].map(
        {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
    )
    merged = audit.merge(
        flagged[["game_id", "kickoff", *COUNTED_FLAG_COLUMNS]],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    clock = week_clock(flagged)
    rows = []
    for row in merged.to_dict("records"):
        week = int(row["week"])
        revisions = row.get("revisions")
        if isinstance(revisions, str) and revisions.strip():
            parts = revisions.split("|")[-1].strip().split()
            recorded = pd.Timestamp(f"{SEASON}-{parts[0]} {parts[1]}", tz="UTC")
        else:
            recorded = clock[week]["tuesday"]
        served_at = min(row["kickoff"], clock[week]["deadline"])
        record_fit = in_force(history, recorded)
        serve_fit = in_force(history, served_at)
        flags = {column: float(row[column]) for column in COUNTED_FLAG_COLUMNS}
        total = sum(flags.values())
        base = float(_logit(row["final_home_probability"])) - record_fit["flag_coefficient"] * sum(
            flags[column] for column in record_fit["counted"]
        )
        served_sum = sum(flags[column] for column in serve_fit["counted"])
        now_sum = sum(flags[column] for column in active["counted"])
        logits = {
            "gate": base + serve_fit["flag_coefficient"] * served_sum,
            "all_served_weight": base + serve_fit["flag_coefficient"] * total,
            "none": base,
            "fixed_all": base + refits["all_flags"]["flag_coefficient"] * total,
            "gate_now": base + refits["gate_now"]["flag_coefficient"] * now_sum,
            "protection_only": base
            + refits["protection_only"]["flag_coefficient"] * flags["flag_protection"],
        }
        entry = {
            "week": week,
            "game_id": row["game_id"],
            "pool_home_spread": row["pool_home_spread"],
            "final_side": row["final_side"],
            "final_home_probability": row["final_home_probability"],
            "home_covered": row["home_covered"],
            "record_fit": record_fit["artifact"],
            "serve_fit": serve_fit["artifact"],
            "gate_in_force": bool(serve_fit["gated"]),
            "counted_at_serve": "+".join(serve_fit["counted"]),
            "base_logit": base,
            **flags,
        }
        for name, value in logits.items():
            entry[f"logit_{name}"] = value
            entry[f"side_{name}"] = _side(value)
        rows.append(entry)
    return pd.DataFrame(rows)


def side_record(frame: pd.DataFrame, column: str) -> dict:
    settled = frame.loc[frame["home_covered"].notna()]
    right = settled[column].eq("HOME") == settled["home_covered"].eq(1.0)
    return exact_record(int(right.sum()), int((~right).sum()))


def replay_summary(rows: pd.DataFrame) -> dict:
    era = rows.loc[rows["gate_in_force"]]
    out: dict[str, dict] = {
        "games": len(rows),
        "gate_era_games": len(era),
        "gate_era_settled": int(era["home_covered"].notna().sum()),
        "served_side_reproduced": int(rows["side_gate"].eq(rows["final_side"]).sum()),
        "served_side_not_reproduced": rows.loc[
            rows["side_gate"].ne(rows["final_side"]), "game_id"
        ].tolist(),
        "gate_era_flag_games": int(era[list(COUNTED_FLAG_COLUMNS)].ne(0.0).any(axis=1).sum()),
        "totals": {},
        "differences": {},
    }
    for name in ("gate", "all_served_weight", "none", "fixed_all"):
        out["totals"][name] = side_record(era, f"side_{name}")
    for name in ("all_served_weight", "none", "fixed_all"):
        differing = era.loc[era["side_gate"].ne(era[f"side_{name}"])]
        settled = differing.loc[differing["home_covered"].notna()]
        out["differences"][name] = {
            "games": differing["game_id"].tolist(),
            "weeks": sorted(int(value) for value in settled["week"].unique()),
            "gate_side_record": side_record(differing, "side_gate"),
            "unsettled": differing.loc[differing["home_covered"].isna(), "game_id"].tolist(),
        }
    return out


def pending_card(rows: pd.DataFrame, current_week: int) -> list[dict]:
    pending = rows.loc[rows["week"].eq(current_week) & rows["home_covered"].isna()]
    names = ("gate_now", "protection_only", "fixed_all", "none", "all_served_weight")
    out = []
    for row in pending.to_dict("records"):
        sides = {name: row[f"side_{name}"] for name in names}
        out.append(
            {
                "game_id": row["game_id"],
                "final_side": row["final_side"],
                "final_home_probability": row["final_home_probability"],
                "flags": {
                    column: row[column] for column in COUNTED_FLAG_COLUMNS if row[column] != 0.0
                },
                "sides": sides,
                "logits": {name: round(row[f"logit_{name}"], 4) for name in names},
                "depends_on_counted_set": len(set(sides.values())) > 1,
            }
        )
    return out


def flip_share(base_logit: np.ndarray, weight: float) -> float:
    inside = (base_logit >= -weight) & (base_logit < weight)
    return 0.5 * float(np.mean(inside))


def simulate(fires: pd.DataFrame, rate: float, share: float, rng: np.random.Generator) -> dict:
    weeks = fires["week"].to_numpy(dtype=int)
    days = fires["gameday"].to_numpy()
    count = len(fires)
    tuesdays = np.arange(1, LAST_WEEK + 1)
    steps = np.where(rng.random((DRAWS, count)) < rate, 1, -1)
    walk = np.concatenate(
        [np.zeros((DRAWS, 1), dtype=int), np.cumsum(steps, axis=1, dtype=int)], axis=1
    )
    member_week = walk[:, np.searchsorted(weeks, tuesdays, side="left")] > 0
    member_game = walk[:, np.searchsorted(days, days, side="left")] > 0
    changes = (member_week[:, 1:] != member_week[:, :-1]).sum(axis=1)
    dropped = (member_week[:, :-1] & ~member_week[:, 1:]).any(axis=1)
    counted_fires = member_game.sum(axis=1)
    moved = rng.binomial(counted_fires, share)
    by_week = member_week.mean(axis=0)
    reached = np.flatnonzero(by_week >= TARGET_PROBABILITY)
    return {
        "fires": count,
        "p_counted_by_week": {int(week): float(by_week[week - 1]) for week in tuesdays},
        "first_week_at_target": int(tuesdays[reached[0]]) if len(reached) else None,
        "mean_changes": float(changes.mean()),
        "p_two_or_more_changes": float((changes >= 2).mean()),
        "p_ever_counted": float(member_week.any(axis=1).mean()),
        "p_dropped_after_counted": float(dropped.mean()),
        "mean_fires_counted": float(counted_fires.mean()),
        "share_fires_counted": float(counted_fires.mean() / count) if count else 0.0,
        "mean_picks_moved": float(moved.mean()),
        "picks_moved_5_95": [float(np.percentile(moved, 5)), float(np.percentile(moved, 95))],
        "_member_week": member_week,
        "_changes": changes,
        "_moved": moved,
    }


def fires_needed(rate: float) -> int | None:
    fires = np.arange(1, EXACT_HORIZON_FIRES + 1)
    reached = binom.sf(fires // 2, fires, rate) >= TARGET_PROBABILITY
    stays = np.flip(np.logical_and.accumulate(np.flip(reached)))
    return int(fires[np.argmax(stays)]) if stays.any() else None


def registry_batch(replay: dict) -> dict:
    labels = {
        "all_served_weight": (
            "flaggate_gated_side_vs_all_flags_2026_w7_w10",
            "every flag counted at the weight then served",
            "every situational flag had been counted",
        ),
        "none": (
            "flaggate_gated_side_vs_no_flags_2026_w7_w10",
            "no flag counted",
            "no situational flag had been counted",
        ),
        "fixed_all": (
            "flaggate_gated_side_vs_fixed_all_flags_2026_w7_w10",
            "every flag always counted at the weight fitted on 2020-25 with all flags",
            "every situational flag were always counted at one fixed weight",
        ),
    }
    cells = []
    for key, (name, technical, plain) in labels.items():
        record = replay["differences"][key]["gate_side_record"]
        if not record["games"]:
            continue
        cells.append(
            {
                "name": name,
                "description": (
                    "2026 games served while the post-discovery flag gate was in force whose "
                    f"side differs from the side with {technical}; effect is the gated side's "
                    "cover rate at the pool Tuesday line minus an even split, exact binomial "
                    "interval"
                ),
                "plain_summary": (
                    "The card only counts a situational flag once that flag has a winning "
                    f"record this season. This counts the games where the pick would differ if "
                    f"{plain}, and how often the pick actually made covered the pool number."
                ),
                "effect": round((record["rate"] - 0.5) * 100.0, 2),
                "interval_low": round((record["interval_low"] - 0.5) * 100.0, 2),
                "interval_high": round((record["interval_high"] - 0.5) * 100.0, 2),
                "probability_positive": round(record["probability_positive"], 3),
                "sample_games": record["games"],
                "sample_blocks": len(replay["differences"][key]["weeks"]),
                "season_start": SEASON,
                "season_end": SEASON,
                "effect_units": "accuracy_points",
                "notes": (
                    f"gated side {record['wins']}-{record['losses']}, exact two-sided p "
                    f"{record['p_two_sided']:.3f}; other terms held as served; three "
                    "comparisons, one family"
                ),
            }
        )
    return {
        "family": "flag_gate_evaluation_2026_11_14",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "category": "modeling",
        "source": "artifacts/flag_gate_evaluation/summary.json",
        "classification_evidence": (
            "No closing ground met: no resolved wrong sign and no positive control; a "
            "measurement of what the in-season flag membership gate did to served sides, "
            "recorded as measured, not a rule."
        ),
        "cells": cells,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    history = fit_history()
    active = history.loc[history["activated"]].iloc[-1]
    flagged = season_flags()
    current_week = int(flagged.loc[flagged["opener_home_spread"].notna(), "week"].max())
    table, computed = flag_tables(flagged, current_week)
    stored = {column: active["records"].get(column) for column in COUNTED_FLAG_COLUMNS}
    transitions = gate_transitions(history)
    per_game = pd.read_parquet(FIT_ROOT / active["artifact"] / "per_game.parquet")
    refits = refit_variants(per_game, tuple(active["counted"]))
    rows = replay_rows(flagged, history, refits, active)
    replay = replay_summary(rows)
    pending = pending_card(rows, current_week)

    weight = float(active["flag_coefficient"])
    share_served = flip_share(rows["base_logit"].to_numpy(dtype=float), weight)
    fold = (
        per_game["season"].astype(int).astype(str).map(refits["gate_now"]["fold_flag_coefficients"])
    )
    history_base = _logit(per_game["out_of_season_home_probability"]) - fold.to_numpy(
        dtype=float
    ) * per_game[list(active["counted"])].astype(float).sum(axis=1).to_numpy(dtype=float)
    share_history = flip_share(history_base, weight)

    simulation: dict[str, dict] = {}
    null_members = []
    null_changes = np.zeros(DRAWS)
    null_moved = np.zeros(DRAWS)
    for column in COUNTED_FLAG_COLUMNS:
        fires = flagged.loc[flagged[column].ne(0.0), ["week", "gameday"]]
        simulation[column] = {}
        for rate in (NULL_RATE, *TRUE_RATES):
            result = simulate(fires, rate, share_served, rng)
            if rate == NULL_RATE:
                null_members.append(result["_member_week"])
                null_changes = null_changes + result["_changes"]
                null_moved = null_moved + result["_moved"]
            simulation[column][f"{rate:.2f}"] = {
                key: value for key, value in result.items() if not key.startswith("_")
            }
    stacked = np.sum(np.stack(null_members), axis=0)
    joint = {
        "mean_flags_counted_by_week": {
            int(week): float(stacked[:, week - 1].mean()) for week in REPORT_WEEKS
        },
        "p_any_counted_by_week": {
            int(week): float((stacked[:, week - 1] > 0).mean()) for week in REPORT_WEEKS
        },
        "mean_membership_changes": float(null_changes.mean()),
        "membership_changes_5_95": [
            float(np.percentile(null_changes, 5)),
            float(np.percentile(null_changes, 95)),
        ],
        "mean_picks_moved": float(null_moved.mean()),
        "picks_moved_5_95": [
            float(np.percentile(null_moved, 5)),
            float(np.percentile(null_moved, 95)),
        ],
    }
    exact = {}
    for rate in TRUE_RATES:
        needed = fires_needed(rate)
        exact[f"{rate:.2f}"] = {
            "fires_needed": needed,
            "weeks_at_flag_rate": {
                column: round(
                    needed * current_week / table[column]["fires_through_current_week"], 1
                )
                for column in COUNTED_FLAG_COLUMNS
                if needed and table[column]["fires_through_current_week"]
            },
        }

    batch = registry_batch(replay)
    summary = {
        "seed": SEED,
        "draws": DRAWS,
        "active_fit": active["artifact"],
        "active_counted": list(active["counted"]),
        "active_flag_coefficient": weight,
        "current_week": current_week,
        "audit_games_written": pd.Timestamp(AUDIT_GAMES.stat().st_mtime, unit="s", tz="UTC"),
        "records_computed": computed,
        "records_in_fit_metadata": stored,
        "records_match": computed == stored,
        "flag_table": table,
        "gate_transitions": transitions,
        "refits": refits,
        "flip_share_served_2026": share_served,
        "flip_share_history": share_history,
        "simulation": simulation,
        "joint_null": joint,
        "exact_fires_needed": exact,
        "replay": replay,
        "pending_card": pending,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=_plain), "utf-8")
    (OUT / "record_batch.json").write_text(json.dumps(batch, indent=1, default=_plain), "utf-8")
    rows.to_csv(OUT / "replay_games.csv", index=False)
    keep = ["game_id", "week", "gameday", "opener_home_spread", "result", "covered"]
    keep += [*COUNTED_FLAG_COLUMNS, *(f"{column}_outcome" for column in COUNTED_FLAG_COLUMNS)]
    flagged.loc[flagged[list(COUNTED_FLAG_COLUMNS)].ne(0.0).any(axis=1), keep].to_csv(
        OUT / "flag_games_2026.csv", index=False
    )

    print("RECORDS match fit metadata:", computed == stored, computed, stored)
    for column, entry in table.items():
        weeks = " ".join(
            f"w{week}:{cell['wins']}-{cell['losses']}"
            + (f"+{cell['unsettled']}u" if cell["unsettled"] else "")
            for week, cell in entry["by_week"].items()
        )
        print(
            "FLAG",
            column,
            entry["discovery"],
            entry["record"],
            f"p={entry['exact'].get('p_two_sided', float('nan')):.3f}",
            f"fires<=w{current_week}:{entry['fires_through_current_week']}",
            f"later:{entry['fires_known_later']}",
            weeks,
        )
    for item in transitions:
        if item["gated"] or item is transitions[-1]:
            print(
                "GATE",
                item["artifact"],
                item["counted"],
                round(item["flag_coefficient"], 4),
                item["records"],
            )
    print("PRE-GATE states:", sum(1 for item in transitions if not item["gated"]))
    for name, entry in refits.items():
        folds = entry["fold_flag_coefficients"].values()
        print(
            "REFIT",
            name,
            {key: round(value, 4) for key, value in entry["coefficients"].items()},
            entry["out_of_season_record"],
            round(entry["out_of_season_log_loss"], 5),
            f"fold flag {min(folds):.3f}..{max(folds):.3f}",
            f"nonzero {entry['games_with_nonzero_sum']}",
        )
    print("FLIP SHARE served 2026", round(share_served, 3), "history", round(share_history, 3))
    for column in COUNTED_FLAG_COLUMNS:
        for rate, entry in simulation[column].items():
            print(
                "SIM",
                column,
                rate,
                f"fires {entry['fires']}",
                " ".join(
                    f"w{week}:{entry['p_counted_by_week'][week]:.2f}" for week in REPORT_WEEKS
                ),
                f"first>=.8 {entry['first_week_at_target']}",
                f"chg {entry['mean_changes']:.2f}",
                f"2+ {entry['p_two_or_more_changes']:.2f}",
                f"ever {entry['p_ever_counted']:.2f}",
                f"drop {entry['p_dropped_after_counted']:.2f}",
                f"counted {entry['share_fires_counted']:.2f}",
                f"moved {entry['mean_picks_moved']:.2f} {entry['picks_moved_5_95']}",
            )
    print("JOINT NULL", json.dumps(joint, default=_plain))
    print("EXACT", json.dumps(exact, default=_plain))
    print("REPLAY", json.dumps(replay, default=_plain))
    for item in pending:
        print("PENDING", json.dumps(item, default=_plain))
    print("BATCH cells", [cell["name"] for cell in batch["cells"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
