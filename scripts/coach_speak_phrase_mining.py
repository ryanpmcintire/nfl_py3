from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.io import atomic_parquet, run_id
from nfl_ats.players import attach_snap_player_ids
from nfl_ats.provenance import artifact_provenance, sha256_file, write_experiment_artifact
from nfl_ats.public_board import find_matching_opener_evaluation
from nfl_ats.transaction_wire_features import canonical_team

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/experiments/coach_speak_base_rates")
SEED = 20260910
SAMPLES = 20000

PHRASES: dict[str, str] = {
    "game_time_decision": "game time decision",
    "questionable": "questionable",
    "doubtful": "doubtful",
    "hopeful": "hopeful",
    "we_will_see": "we will see",
    "day_to_day": "day to day",
    "trending": "trending",
    "expected_to_play": "expected to play",
    "will_not_play": "will not play",
    "ruled_out": "ruled out",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
SEASON_START = 2013
SEASON_END = 2024
WINDOW_DAYS = 8.0
DAYS_BUCKETS = [(0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 8.0)]
INJURIES_PATH = Path("data/raw/nflverse_injuries/20260910T203021Z/injuries.parquet")
SNAP_ROOT = Path("data/players/raw/20260817T184901Z")
STARTER_SNAP_SHARE_THRESHOLD = 0.5


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace(".", "").replace("'", "").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def strip_suffix(tokens: tuple[str, ...]) -> tuple[str, ...]:
    if len(tokens) > 2 and tokens[-1] in SUFFIXES:
        return tokens[:-1]
    return tokens


def latest_injury_news_index(repo_root: Path) -> tuple[Path, pd.DataFrame]:
    root = repo_root / "data" / "raw" / "injury_news"
    candidates = sorted(p for p in root.glob("*") if p.is_dir() and (p / "index.parquet").is_file())
    if not candidates:
        raise FileNotFoundError(f"no injury_news snapshot with index.parquet under {root}")
    snapshot = candidates[-1]
    frame = pd.read_parquet(snapshot / "index.parquet")
    frame["headline_guess"] = frame["headline_guess"].fillna("").str.lower()
    frame["lastmod"] = pd.to_datetime(frame["lastmod"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["lastmod"]).reset_index(drop=True)
    frame["article_idx"] = frame.index
    return snapshot, frame


def tag_phrases(news: pd.DataFrame) -> list[str]:
    phrase_cols = []
    for phrase_id, phrase in PHRASES.items():
        col = f"phrase__{phrase_id}"
        news[col] = news["headline_guess"].str.contains(re.escape(phrase), regex=True)
        phrase_cols.append(col)
    return phrase_cols


def build_name_map(injuries: pd.DataFrame) -> dict[str, set[str]]:
    skill = injuries.loc[injuries["position"].isin(SKILL_POSITIONS)].copy()
    skill["norm_full"] = skill["full_name"].map(normalize_name)
    name_map: dict[str, set[str]] = {}
    for norm_full, gsis_id in (
        skill[["norm_full", "gsis_id"]].drop_duplicates().itertuples(index=False)
    ):
        if not norm_full:
            continue
        tokens = tuple(norm_full.split(" "))
        name_map.setdefault(norm_full, set()).add(gsis_id)
        stripped = strip_suffix(tokens)
        if stripped != tokens:
            name_map.setdefault(" ".join(stripped), set()).add(gsis_id)
    return name_map


def find_candidates(headline: str, name_map: dict[str, set[str]]) -> list[str]:
    tokens = headline.split(" ")
    found: set[str] = set()
    n = len(tokens)
    for length in (3, 2):
        for i in range(0, n - length + 1):
            span = " ".join(tokens[i : i + length])
            ids = name_map.get(span)
            if ids and len(ids) == 1:
                found.add(next(iter(ids)))
    return sorted(found)


def build_schedule_team_games(repo_root: Path) -> pd.DataFrame:
    candidates = sorted((repo_root / "data" / "raw").glob("*/schedules.parquet"))
    if not candidates:
        raise FileNotFoundError("no data/raw/*/schedules.parquet snapshot found")
    schedule = pd.read_parquet(candidates[-1])
    schedule = schedule.loc[schedule["game_type"].eq("REG")].copy()
    schedule["season"] = pd.to_numeric(schedule["season"], errors="raise").astype(int)
    schedule["week"] = pd.to_numeric(schedule["week"], errors="raise").astype(int)
    date_text = pd.to_datetime(schedule["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    time_text = schedule["gametime"].astype("string")
    local = pd.to_datetime(date_text + " " + time_text, errors="coerce")
    schedule["kickoff"] = local.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    home = schedule[["season", "week", "home_team", "game_id", "kickoff"]].rename(
        columns={"home_team": "team"}
    )
    away = schedule[["season", "week", "away_team", "game_id", "kickoff"]].rename(
        columns={"away_team": "team"}
    )
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["team"] = team_games["team"].astype(str).map(canonical_team)
    return team_games


def load_skill_report_population(injuries: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    frame = injuries.loc[
        injuries["position"].isin(SKILL_POSITIONS)
        & injuries["season"].between(SEASON_START, SEASON_END)
        & injuries["game_type"].eq("REG")
    ].copy()
    frame = frame.dropna(subset=["date_modified", "week"])
    frame["week"] = frame["week"].astype(int)
    frame = frame.merge(
        team_games, on=["season", "week", "team"], how="left", validate="many_to_one"
    )
    return frame


def attach_participation(frame: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    snaps = pd.read_parquet(repo_root / SNAP_ROOT / "snap_counts.parquet")
    rosters = pd.read_parquet(repo_root / SNAP_ROOT / "weekly_rosters.parquet")
    snaps = attach_snap_player_ids(snaps, rosters)
    snaps = snaps.loc[snaps["gsis_id"].notna()].copy()
    snaps["total_snaps"] = snaps[["offense_snaps", "defense_snaps", "st_snaps"]].sum(axis=1)
    snaps["snap_share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps = snaps.drop_duplicates(["game_id", "gsis_id"], keep="first")

    merged = frame.merge(
        snaps[["game_id", "gsis_id", "total_snaps"]],
        on=["game_id", "gsis_id"],
        how="left",
        validate="many_to_one",
    )
    merged["snap_row_matched"] = merged["total_snaps"].notna()
    merged["played"] = merged["total_snaps"].fillna(0.0).gt(0.0)
    merged["inactive"] = ~merged["played"]

    history = (
        snaps[["gsis_id", "season", "week", "snap_share"]].dropna(subset=["snap_share"]).copy()
    )
    history["season_week"] = history["season"] * 100 + history["week"]
    history = history.sort_values("season_week")

    merged["season_week"] = merged["season"] * 100 + merged["week"]
    left = merged[["season_week", "gsis_id"]].reset_index().sort_values("season_week")
    asof = pd.merge_asof(
        left,
        history[["season_week", "gsis_id", "snap_share"]],
        on="season_week",
        by="gsis_id",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("index")
    merged["trailing_share"] = asof["snap_share"].reindex(merged.index)
    merged = merged.drop(columns="season_week")
    return merged


def resolve_phrase_player_weeks(
    news: pd.DataFrame,
    phrase_cols: list[str],
    name_map: dict[str, set[str]],
    population: pd.DataFrame,
) -> pd.DataFrame:
    news = news.copy()
    mask_any = news[phrase_cols].any(axis=1)
    hits = news.loc[mask_any].copy()
    hits["gsis_candidates"] = hits["headline_guess"].map(
        lambda text: find_candidates(text, name_map)
    )
    exploded = (
        hits.loc[hits["gsis_candidates"].map(len) > 0]
        .explode("gsis_candidates")
        .rename(columns={"gsis_candidates": "gsis_id"})
    )

    long_rows = []
    for col in phrase_cols:
        phrase_id = col.replace("phrase__", "")
        sub = exploded.loc[
            exploded[col], ["article_idx", "gsis_id", "lastmod", "url", "slug"]
        ].copy()
        sub["phrase"] = phrase_id
        long_rows.append(sub)
    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    merged = long.merge(population, on="gsis_id", how="inner", suffixes=("", "_report"))
    merged["dt_days"] = (merged["kickoff"] - merged["lastmod"]).dt.total_seconds() / 86400.0
    inwin = merged.loc[merged["dt_days"].between(0.0, WINDOW_DAYS)].copy()
    inwin = inwin.sort_values("dt_days").drop_duplicates(
        ["article_idx", "phrase", "gsis_id"], keep="first"
    )
    inwin = inwin.sort_values("lastmod")
    player_week = inwin.drop_duplicates(["gsis_id", "season", "week", "phrase"], keep="first")
    return player_week.reset_index(drop=True)


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * np.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return float(max(0.0, low)), float(min(1.0, high))


def bucket_days(value: float) -> str:
    for low, high in DAYS_BUCKETS:
        if low <= value < high or (high == DAYS_BUCKETS[-1][1] and value == high):
            return f"{low:g}-{high:g}d"
    return "8+d"


def phrase_base_rate_table(player_week: pd.DataFrame) -> dict[str, Any]:
    scored = player_week.dropna(subset=["inactive"]).copy()
    scored["days_bucket"] = scored["dt_days"].map(bucket_days)
    by_phrase: dict[str, Any] = {}
    for phrase, group in scored.groupby("phrase", observed=True):
        n = len(group)
        k = int(group["inactive"].sum())
        low, high = wilson_interval(k, n)
        by_phrase[phrase] = {
            "n": n,
            "n_articles_raw": int(group["article_idx"].nunique()),
            "inactive_count": k,
            "inactive_rate": k / n if n else float("nan"),
            "wilson_low": low,
            "wilson_high": high,
            "snap_row_match_rate": float(group["snap_row_matched"].mean()),
        }
    by_phrase_bucket: dict[str, Any] = {}
    for (phrase, bucket), group in scored.groupby(["phrase", "days_bucket"], observed=True):
        n = len(group)
        k = int(group["inactive"].sum())
        low, high = wilson_interval(k, n)
        by_phrase_bucket[f"{phrase}__{bucket}"] = {
            "n": n,
            "inactive_count": k,
            "inactive_rate": k / n if n else float("nan"),
            "wilson_low": low,
            "wilson_high": high,
        }
    return {"by_phrase": by_phrase, "by_phrase_days_bucket": by_phrase_bucket}


def split_half_reliability(player_week: pd.DataFrame) -> dict[str, Any]:
    scored = player_week.dropna(subset=["inactive"]).copy()
    scored["month_parity"] = scored["kickoff"].dt.month % 2
    rows = []
    for (team, parity), group in scored.groupby(["team", "month_parity"], observed=True):
        rows.append(
            {
                "team": team,
                "parity": int(parity),
                "rate": float(group["inactive"].mean()),
                "n": len(group),
            }
        )
    long = pd.DataFrame(rows)
    if long.empty:
        return {"status": "no_data", "n_teams": 0, "correlation": None}
    pivot_rate = long.pivot(index="team", columns="parity", values="rate")
    pivot_n = long.pivot(index="team", columns="parity", values="n")
    eligible = (
        pivot_n.ge(5).all(axis=1)
        if {0, 1}.issubset(pivot_n.columns)
        else pd.Series(False, index=pivot_n.index)
    )
    eligible_teams = pivot_rate.loc[eligible].dropna()
    if (
        len(eligible_teams) < 4
        or eligible_teams[0].nunique() < 2
        or eligible_teams[1].nunique() < 2
    ):
        return {
            "status": "underpowered",
            "n_teams_min_5_per_half": int(eligible.sum()),
            "correlation": None,
        }
    correlation = float(eligible_teams[0].corr(eligible_teams[1]))
    return {
        "status": "measured",
        "unit": "team, odd/even-kickoff-month mean, pooled across all predeclared phrases",
        "n_teams_min_5_per_half": len(eligible_teams),
        "correlation": correlation,
    }


def assign_phrase_category(population: pd.DataFrame, player_week: pd.DataFrame) -> pd.DataFrame:
    ranked = player_week.sort_values("dt_days").drop_duplicates(
        ["gsis_id", "season", "week"], keep="first"
    )
    lookup = ranked.set_index(["gsis_id", "season", "week"])["phrase"]
    frame = population.copy()
    frame["phrase_category"] = "none"
    keys = list(zip(frame["gsis_id"], frame["season"], frame["week"], strict=True))
    values = []
    for key in keys:
        values.append(lookup.get(key, "none"))
    frame["phrase_category"] = values
    return frame


def _beta_rate_table(
    train: pd.DataFrame, group_cols: list[str], prior: float = 2.0
) -> dict[Any, float]:
    table: dict[Any, float] = {}
    grouped = train.groupby(group_cols, observed=True)["inactive"]
    for key, series in grouped:
        n = len(series)
        k = float(series.sum())
        table[key] = (k + prior) / (n + 2.0 * prior)
    return table


def _predict(
    frame: pd.DataFrame, table: dict[Any, float], group_cols: list[str], overall_rate: float
) -> np.ndarray:
    keys = list(zip(*(frame[col] for col in group_cols), strict=True))
    return np.array([table.get(key, overall_rate) for key in keys], dtype=float)


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def incremental_information_test(
    scored_population: pd.DataFrame, seed: int, samples: int
) -> dict[str, Any]:
    frame = scored_population.dropna(subset=["inactive", "kickoff", "report_status"]).copy()
    frame["inactive"] = frame["inactive"].astype(float)
    frame["month_parity"] = frame["kickoff"].dt.month % 2
    overall_rate = float(frame["inactive"].mean())

    baseline_pred = np.empty(len(frame))
    candidate_pred = np.empty(len(frame))
    for fold in (0, 1):
        train = frame.loc[frame["month_parity"].ne(fold)]
        test_mask = frame["month_parity"].eq(fold)
        baseline_table = _beta_rate_table(train, ["report_status"])
        candidate_table = _beta_rate_table(train, ["report_status", "phrase_category"])
        baseline_pred[test_mask.to_numpy()] = _predict(
            frame.loc[test_mask], baseline_table, ["report_status"], overall_rate
        )
        candidate_pred[test_mask.to_numpy()] = _predict(
            frame.loc[test_mask],
            candidate_table,
            ["report_status", "phrase_category"],
            overall_rate,
        )

    y = frame["inactive"].to_numpy()
    baseline_loss = _log_loss(y, baseline_pred)
    candidate_loss = _log_loss(y, candidate_pred)
    delta_loss = baseline_loss - candidate_loss
    baseline_correct = (baseline_pred.round() == y).astype(float)
    candidate_correct = (candidate_pred.round() == y).astype(float)
    delta_accuracy = (candidate_correct - baseline_correct) * 100.0

    blocks = pd.DataFrame(
        {
            "season": frame["season"].to_numpy(),
            "week": frame["week"].to_numpy(),
            "delta_loss": delta_loss,
            "delta_accuracy": delta_accuracy,
        }
    )
    groups = blocks.groupby(["season", "week"]).agg(
        sum_loss=("delta_loss", "sum"),
        sum_acc=("delta_accuracy", "sum"),
        count=("delta_loss", "size"),
    )
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    group_sum_loss = groups["sum_loss"].to_numpy()
    group_sum_acc = groups["sum_acc"].to_numpy()
    group_count = groups["count"].to_numpy()
    loss_draws = np.empty(samples)
    acc_draws = np.empty(samples)
    for start in range(0, samples, 500):
        batch = min(500, samples - start)
        indices = rng.integers(0, n_groups, size=(batch, n_groups))
        counts = group_count[indices].sum(axis=1)
        loss_draws[start : start + batch] = group_sum_loss[indices].sum(axis=1) / counts
        acc_draws[start : start + batch] = group_sum_acc[indices].sum(axis=1) / counts

    phrase_only_rows = frame.loc[frame["phrase_category"].ne("none")]
    return {
        "n_player_weeks_scored": len(frame),
        "n_phrase_fired_player_weeks": len(phrase_only_rows),
        "overall_inactive_rate": overall_rate,
        "baseline_mean_log_loss": float(baseline_loss.mean()),
        "candidate_mean_log_loss": float(candidate_loss.mean()),
        "log_loss_improvement_mean": float(delta_loss.mean()),
        "log_loss_improvement_interval_low": float(np.quantile(loss_draws, 0.025)),
        "log_loss_improvement_interval_high": float(np.quantile(loss_draws, 0.975)),
        "log_loss_improvement_probability_positive": float(
            probability_positive_from_draws(loss_draws)
        ),
        "accuracy_points_mean": float(delta_accuracy.mean()),
        "accuracy_points_interval_low": float(np.quantile(acc_draws, 0.025)),
        "accuracy_points_interval_high": float(np.quantile(acc_draws, 0.975)),
        "accuracy_points_probability_positive": float(probability_positive_from_draws(acc_draws)),
        "n_week_blocks": int(n_groups),
        "seed": seed,
        "samples": samples,
    }


def week_blocked_summary(frame: pd.DataFrame, seed: int, samples: int) -> dict[str, Any]:
    delta = (frame["candidate_correct"] - frame["production_correct"]) * 100.0
    blocks = pd.DataFrame({"season": frame["season"], "week": frame["week"], "delta": delta})
    groups = blocks.groupby(["season", "week"])["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    group_sums = groups["sum"].to_numpy()
    group_counts = groups["count"].to_numpy()
    n_groups = len(groups)
    for start in range(0, samples, 500):
        batch = min(500, samples - start)
        indices = rng.integers(0, n_groups, size=(batch, n_groups))
        draws[start : start + batch] = group_sums[indices].sum(axis=1) / group_counts[indices].sum(
            axis=1
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "n_games": len(frame),
        "n_week_blocks": int(n_groups),
        "production_accuracy": float(frame["production_correct"].mean() * 100.0),
        "candidate_accuracy": float(frame["candidate_correct"].mean() * 100.0),
        "effect_accuracy_points": float(delta.mean()),
        "interval_low": float(low),
        "interval_high": float(high),
        "probability_positive": float(probability_positive_from_draws(draws)),
    }


def positive_control(frame: pd.DataFrame, seed: int, samples: int) -> dict[str, Any]:
    control = frame.copy()
    control["candidate_pick_home"] = control["margin_home_positive"]
    control["candidate_correct"] = 1.0
    return week_blocked_summary(control, seed, samples)


def production_screen(
    scored_population: pd.DataFrame, seed: int, samples: int, repo_root: Path
) -> dict[str, Any] | None:
    from nfl_ats.transaction_flag_features import default_schedule

    match = find_matching_opener_evaluation(repo_root / "artifacts")
    if match is None:
        return {"skipped": True, "reason": "no opener evaluation matches the active model"}
    active_manifest, evaluation_dir = match
    per_game = pd.read_parquet(evaluation_dir / "per_game.parquet")

    schedule = default_schedule(repo_root)
    reg = schedule.loc[schedule["game_type"].eq("REG"), ["game_id", "home_team", "away_team"]]

    eligible = scored_population.loc[
        scored_population["phrase_category"].ne("none")
        & scored_population["trailing_share"].notna()
    ].copy()
    eligible["starter"] = eligible["trailing_share"].ge(STARTER_SNAP_SHARE_THRESHOLD)
    eligible = eligible.loc[eligible["starter"]]

    fired_rate = float(eligible["inactive"].mean()) if len(eligible) else 0.0
    team_frame = (
        eligible.groupby(["game_id", "team"], observed=True)
        .agg(n_flagged_starters=("gsis_id", "size"))
        .reset_index()
    )
    team_frame["expected_absence_value"] = fired_rate * team_frame["n_flagged_starters"]

    frame = per_game.merge(reg, on="game_id", how="left", validate="one_to_one")
    frame = frame.loc[frame["correct_at_open_probability_rule"].notna()].copy()
    frame["home_team"] = frame["home_team"].astype(str).map(canonical_team)
    frame["away_team"] = frame["away_team"].astype(str).map(canonical_team)

    lookup = team_frame.set_index(["game_id", "team"])["expected_absence_value"]
    for side in ("home", "away"):
        team_col = f"{side}_team"
        values = []
        for game_id, team in zip(frame["game_id"], frame[team_col], strict=True):
            values.append(lookup.get((game_id, team), 0.0))
        frame[f"{side}_expected_absence_value"] = values

    frame["diff_expected_absence_value"] = (
        frame["home_expected_absence_value"] - frame["away_expected_absence_value"]
    )
    frame["production_pick_home"] = frame["pick_home_at_open_probability_rule"].astype(bool)
    fade_home = frame["diff_expected_absence_value"] > 0
    fade_away = frame["diff_expected_absence_value"] < 0
    flip_condition = (fade_home & frame["production_pick_home"]) | (
        fade_away & ~frame["production_pick_home"]
    )
    frame["candidate_pick_home"] = np.where(
        flip_condition, ~frame["production_pick_home"], frame["production_pick_home"]
    )
    frame["margin_home_positive"] = frame["margin_vs_open"].gt(0)
    frame["production_correct"] = (
        frame["production_pick_home"].eq(frame["margin_home_positive"]).astype(float)
    )
    frame["candidate_correct"] = (
        frame["candidate_pick_home"].eq(frame["margin_home_positive"]).astype(float)
    )

    summary = week_blocked_summary(frame, seed, samples)
    control = positive_control(frame, seed, samples)
    return {
        "n_games_total": len(frame),
        "n_games_flipped": int(flip_condition.sum()),
        "active_model_id": active_manifest.get("active_model_id"),
        "opener_evaluation_dir": str(evaluation_dir),
        "summary": summary,
        "positive_control": control,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--skip-production-screen", action="store_true")
    args = parser.parse_args()

    news_snapshot, news = latest_injury_news_index(REPO_ROOT)
    phrase_cols = tag_phrases(news)

    injuries = pd.read_parquet(REPO_ROOT / INJURIES_PATH)
    injuries["date_modified"] = pd.to_datetime(injuries["date_modified"], utc=True, errors="coerce")
    injuries["season"] = pd.to_numeric(injuries["season"], errors="raise").astype(int)
    injuries["week"] = pd.to_numeric(injuries["week"], errors="coerce")
    injuries["team"] = injuries["team"].astype(str).map(canonical_team)

    name_map = build_name_map(injuries)
    team_games = build_schedule_team_games(REPO_ROOT)
    population = load_skill_report_population(injuries, team_games)
    population = attach_participation(population, REPO_ROOT)

    player_week = resolve_phrase_player_weeks(news, phrase_cols, name_map, population)

    base_rates = phrase_base_rate_table(player_week)
    reliability = split_half_reliability(player_week)

    scored_population = assign_phrase_category(population, player_week)
    incremental = incremental_information_test(scored_population, SEED, SAMPLES)

    screen_result: dict[str, Any] | None = None
    if (
        not args.skip_production_screen
        and incremental["accuracy_points_probability_positive"] > 0.5
    ):
        screen_result = production_screen(scored_population, SEED, SAMPLES, REPO_ROOT)

    coverage = {
        "news_snapshot_used": str(news_snapshot),
        "news_archive_rows": len(news),
        "news_archive_lastmod_min": str(news["lastmod"].min()),
        "news_archive_lastmod_max": str(news["lastmod"].max()),
        "articles_matching_any_phrase": int(news[phrase_cols].any(axis=1).sum()),
        "distinct_skill_position_full_names": int(
            injuries.loc[injuries["position"].isin(SKILL_POSITIONS), "full_name"].nunique()
        ),
        "population_rows_2013_2024_reg_skill": len(population),
        "population_rows_resolved_kickoff": int(population["kickoff"].notna().sum()),
        "resolved_player_week_phrase_rows": len(player_week),
        "resolved_player_week_phrase_rows_by_phrase": player_week["phrase"]
        .value_counts()
        .to_dict(),
    }

    results = {
        "predeclaration_doc": "docs/coach_speak_base_rates.md",
        "phrases": PHRASES,
        "season_window": [SEASON_START, SEASON_END],
        "window_days_before_kickoff": WINDOW_DAYS,
        "coverage": coverage,
        "base_rates": base_rates,
        "reliability": reliability,
        "incremental_information_test": incremental,
        "production_screen": screen_result,
    }
    print(json.dumps(results, indent=2, default=str), flush=True)

    destination = OUTPUT / run_id()
    metadata = {
        "news_snapshot": str(news_snapshot),
        "injuries_path": str(INJURIES_PATH),
        "injuries_sha256": sha256_file(REPO_ROOT / INJURIES_PATH),
        "results": results,
        "seed": SEED,
        "bootstrap_samples": SAMPLES,
        "predeclaration": "ROADMAP.md LEAD-56; docs/coach_speak_base_rates.md",
        "provenance": artifact_provenance(
            {"command": "coach-speak-phrase-mining", "seed": SEED, "samples": SAMPLES},
            REPO_ROOT / INJURIES_PATH,
        ),
    }
    write_experiment_artifact(
        destination,
        "results.json",
        metadata,
        command="coach-speak-phrase-mining",
        metrics={
            "log_loss_improvement_mean": incremental["log_loss_improvement_mean"],
            "accuracy_points_mean": incremental["accuracy_points_mean"],
        },
        registry_root=OUTPUT / "registry",
    )
    atomic_parquet(player_week, destination / "player_week_phrase_matches.parquet")
    atomic_parquet(scored_population, destination / "scored_population.parquet")
    print(json.dumps({"output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
