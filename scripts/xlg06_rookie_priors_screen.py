from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nfl_ats.clv import (  # noqa: E402
    HISTORICAL_CAPTURE_KIND,
    cached_pairing_table,
    opener_pick_evaluation,
    pick_correct,
    resolve_active_model_config,
    week_blocked_bootstrap,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.evidence_conventions import probability_positive_from_draws  # noqa: E402
from nfl_ats.rotation import confirmation_split, load_registry  # noqa: E402

DRAFT_ROOT = REPO / "data" / "cfb" / "draft_picks" / "raw" / "20260816T164451Z"
USAGE_ROOT = REPO / "data" / "cfb" / "usage" / "raw" / "20260818T214627Z"
PLAYERS_PARQUET = REPO / "data" / "players" / "raw" / "20260911T020107Z" / "players.parquet"
SNAP_COUNTS = REPO / "data" / "players" / "raw" / "20260905T123614Z" / "snap_counts.parquet"
WEEKLY_ROSTERS = REPO / "data" / "players" / "raw" / "20260905T123614Z" / "weekly_rosters.parquet"
PLAYER_STATS = (
    REPO / "data" / "players" / "values" / "raw" / "20260817T184911Z" / "player_stats.parquet"
)
GAME_FEATURES = REPO / "data" / "processed" / "game_features.parquet"
PRODUCTION_FEATURES = REPO / "data" / "processed" / "game_features_weak_stack.parquet"
MARKET_ROOT = REPO / "data" / "market" / "raw"
ARTIFACTS_ROOT = REPO / "artifacts"
OUT_DIR = REPO / "artifacts" / "xlg06_rookie_priors"

FAMILY = "xlg06_rookie_prior_surplus_on_production"
N0_SNAPS = 300.0
MIN_ROOKIE_SNAPS = 100.0
FLAG_PERCENTILE = 80.0
MIN_PRIOR_SEASONS = 2
MIN_TRAIN_CLASS_ROWS = 60
RIDGE_ALPHA = 1.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260911
FIRST_DRAFT_CLASS = 2014
LAST_DRAFT_CLASS = 2025

DEFENSE_DISRUPTION_WEIGHTS = (
    ("def_tackles_for_loss", 0.5),
    ("def_fumbles_forced", 2.0),
    ("def_sacks", 1.5),
    ("def_qb_hits", 0.25),
    ("def_interceptions", 4.0),
    ("def_pass_defended", 0.5),
)

OFFENSE_ARM_GROUPS = ("RB", "WR", "TE")
DEFENSE_ARM_GROUPS = ("DL", "LB", "DB")
SLOT_ONLY_GROUPS = ("QB", "OL")
PASS_RUSH_NFL_POSITIONS = ("DE", "OLB", "EDGE")
PASS_RUSH_DRAFT_POSITIONS = ("Defensive End", "Outside Linebacker", "Defensive Edge")


def _read_partitioned(root: Path) -> pd.DataFrame:
    frames = []
    for part in sorted(root.glob("season=*")):
        for parquet in sorted(part.glob("*.parquet")):
            frames.append(pd.read_parquet(parquet))
    if not frames:
        raise SystemExit(f"no parquet partitions under {root}")
    return pd.concat(frames, ignore_index=True)


def _surname(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    parts = [
        p for p in text.replace(".", " ").split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}
    ]
    return parts[-1] if parts else ""


def load_crosswalk() -> pd.DataFrame:
    draft = _read_partitioned(DRAFT_ROOT)
    draft = draft.loc[draft["year"].between(FIRST_DRAFT_CLASS, LAST_DRAFT_CLASS)].copy()
    draft["draft_year"] = draft["year"].astype(int)
    draft["overall"] = draft["overall"].astype(int)
    draft["draft_round"] = draft["round"].astype(int)
    draft["college_athlete_id"] = pd.to_numeric(draft["collegeAthleteId"], errors="coerce").astype(
        "Int64"
    )
    draft["pre_draft_grade"] = pd.to_numeric(draft["preDraftGrade"], errors="coerce")
    draft["pre_draft_ranking"] = pd.to_numeric(draft["preDraftRanking"], errors="coerce")
    draft["draft_position_label"] = draft["position"].astype(str)
    draft["draft_surname"] = draft["name"].map(_surname)

    players = pd.read_parquet(PLAYERS_PARQUET)
    nfl = players.loc[players["draft_year"].between(FIRST_DRAFT_CLASS, LAST_DRAFT_CLASS)].copy()
    nfl["draft_year"] = nfl["draft_year"].astype(int)
    nfl["overall"] = pd.to_numeric(nfl["draft_pick"], errors="coerce").astype("Int64")
    nfl = nfl.loc[nfl["overall"].notna()].copy()
    nfl["overall"] = nfl["overall"].astype(int)
    nfl["nfl_surname"] = nfl["display_name"].map(_surname)
    nfl["espn_numeric"] = pd.to_numeric(nfl["espn_id"], errors="coerce").astype("Int64")
    nfl["nfl_position_detail"] = nfl["position"].astype(str)
    nfl["nfl_position"] = nfl["position_group"].astype(str)
    nfl = nfl.drop_duplicates(subset=["draft_year", "overall"], keep="first")

    merged = draft.merge(
        nfl[
            [
                "draft_year",
                "overall",
                "gsis_id",
                "display_name",
                "nfl_surname",
                "espn_numeric",
                "college_name",
                "draft_team",
                "nfl_position_detail",
                "nfl_position",
            ]
        ],
        on=["draft_year", "overall"],
        how="left",
        validate="one_to_one",
    )
    merged["surname_match"] = merged["draft_surname"].eq(merged["nfl_surname"])
    merged["espn_match"] = merged["college_athlete_id"].eq(merged["espn_numeric"])
    merged["identity_ok"] = merged["gsis_id"].notna() & (
        merged["surname_match"] | merged["espn_match"]
    )
    merged["nfl_position"] = merged["nfl_position"].astype(str)
    merged["nfl_position_detail"] = merged["nfl_position_detail"].astype(str)
    merged["is_pass_rush"] = merged["nfl_position_detail"].isin(PASS_RUSH_NFL_POSITIONS) | merged[
        "draft_position_label"
    ].isin(PASS_RUSH_DRAFT_POSITIONS)
    return merged


def load_cfb_usage() -> pd.DataFrame:
    usage = _read_partitioned(USAGE_ROOT)
    usage["college_athlete_id"] = pd.to_numeric(usage["id"], errors="coerce").astype("Int64")
    usage["usage_overall"] = pd.to_numeric(usage["usage.overall"], errors="coerce")
    usage["season"] = usage["season"].astype(int)
    usage = usage.loc[usage["college_athlete_id"].notna()]
    usage = usage.sort_values(["college_athlete_id", "season", "usage_overall"])
    return usage.drop_duplicates(subset=["college_athlete_id", "season"], keep="last")[
        ["college_athlete_id", "season", "usage_overall"]
    ]


def _pfr_to_gsis() -> dict[str, str]:
    rosters = pd.read_parquet(WEEKLY_ROSTERS, columns=["gsis_id", "pfr_id"])
    links = rosters.loc[rosters["gsis_id"].notna() & rosters["pfr_id"].notna()].copy()
    links["gsis_id"] = links["gsis_id"].astype(str)
    links["pfr_id"] = links["pfr_id"].astype(str)
    counts = links.value_counts().rename("appearances").reset_index()
    selected = counts.sort_values(
        ["pfr_id", "appearances", "gsis_id"], ascending=[True, False, True]
    ).drop_duplicates("pfr_id")
    mapping = dict(zip(selected["pfr_id"], selected["gsis_id"], strict=True))

    master = pd.read_parquet(PLAYERS_PARQUET, columns=["gsis_id", "pfr_id"])
    master = master.loc[master["gsis_id"].notna() & master["pfr_id"].notna()]
    master = master.drop_duplicates(subset=["pfr_id"], keep="first")
    for pfr_id, gsis_id in zip(
        master["pfr_id"].astype(str), master["gsis_id"].astype(str), strict=True
    ):
        mapping.setdefault(pfr_id, gsis_id)
    return mapping


def load_snap_panel() -> pd.DataFrame:
    snaps = pd.read_parquet(SNAP_COUNTS)
    snaps = snaps.loc[snaps["game_type"].eq("REG")].copy()
    snaps["gsis_id"] = snaps["pfr_player_id"].astype(str).map(_pfr_to_gsis())
    snaps = snaps.loc[snaps["gsis_id"].notna()].copy()
    snaps["season"] = snaps["season"].astype(int)
    snaps["week"] = snaps["week"].astype(int)
    snaps["game_id"] = snaps["game_id"].astype(str)
    for column in ("offense_snaps", "defense_snaps", "offense_pct", "defense_pct"):
        snaps[column] = pd.to_numeric(snaps[column], errors="coerce").fillna(0.0)
    snaps["snap_position"] = snaps["position"].astype(str).str.upper()
    return snaps[
        [
            "gsis_id",
            "season",
            "week",
            "game_id",
            "team",
            "snap_position",
            "offense_snaps",
            "defense_snaps",
            "offense_pct",
            "defense_pct",
        ]
    ]


def load_value_panel(snaps: pd.DataFrame) -> pd.DataFrame:
    stats = pd.read_parquet(PLAYER_STATS)
    stats = stats.loc[stats["season_type"].eq("REG")].copy()
    stats["gsis_id"] = stats["player_id"].astype(str)
    stats["game_id"] = stats["game_id"].astype(str)
    for column in ("rushing_epa", "receiving_epa", *[c for c, _ in DEFENSE_DISRUPTION_WEIGHTS]):
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0.0)
    stats["skill_epa_raw"] = stats["rushing_epa"] + stats["receiving_epa"]
    stats["defense_disruption"] = sum(
        weight * stats[column] for column, weight in DEFENSE_DISRUPTION_WEIGHTS
    )
    merged = snaps.merge(
        stats[["gsis_id", "game_id", "skill_epa_raw", "defense_disruption"]],
        on=["gsis_id", "game_id"],
        how="left",
    )
    merged["skill_epa_raw"] = merged["skill_epa_raw"].fillna(0.0)
    merged["defense_disruption"] = merged["defense_disruption"].fillna(0.0)
    merged["skill_epa"] = np.where(merged["snap_position"].eq("QB"), 0.0, merged["skill_epa_raw"])
    return merged


def rookie_season_value(panel: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    rookie_map = crosswalk.loc[crosswalk["identity_ok"], ["gsis_id", "draft_year"]].copy()
    rookie_map["gsis_id"] = rookie_map["gsis_id"].astype(str)
    rows = panel.merge(rookie_map, on="gsis_id", how="inner")
    rows = rows.loc[rows["season"].eq(rows["draft_year"])].copy()
    rows["half"] = np.where(rows["week"] % 2 == 1, "odd", "even")

    def _agg(frame: pd.DataFrame) -> pd.DataFrame:
        grouped = frame.groupby("gsis_id", as_index=False).agg(
            offense_snaps=("offense_snaps", "sum"),
            defense_snaps=("defense_snaps", "sum"),
            skill_epa=("skill_epa", "sum"),
            defense_disruption=("defense_disruption", "sum"),
            games=("game_id", "nunique"),
        )
        grouped["skill_rate"] = np.where(
            grouped["offense_snaps"] > 0.0,
            100.0 * grouped["skill_epa"] / grouped["offense_snaps"],
            np.nan,
        )
        grouped["defense_rate"] = np.where(
            grouped["defense_snaps"] > 0.0,
            100.0 * grouped["defense_disruption"] / grouped["defense_snaps"],
            np.nan,
        )
        return grouped

    full = _agg(rows)
    odd = (
        _agg(rows.loc[rows["half"].eq("odd")])
        .add_suffix("_odd")
        .rename(columns={"gsis_id_odd": "gsis_id"})
    )
    even = (
        _agg(rows.loc[rows["half"].eq("even")])
        .add_suffix("_even")
        .rename(columns={"gsis_id_even": "gsis_id"})
    )
    return full.merge(odd, on="gsis_id", how="left").merge(even, on="gsis_id", how="left")


def build_prior_inputs() -> pd.DataFrame:
    crosswalk = load_crosswalk()
    usage = load_cfb_usage()
    usage = usage.rename(columns={"season": "final_college_season"})
    linked = crosswalk.copy()
    linked["final_college_season"] = linked["draft_year"] - 1
    linked = linked.merge(usage, on=["college_athlete_id", "final_college_season"], how="left")
    panel = load_value_panel(load_snap_panel())
    values = rookie_season_value(panel, linked)
    linked["gsis_id"] = linked["gsis_id"].astype(str)
    table = linked.merge(values, on="gsis_id", how="left")
    table["log_overall"] = np.log(table["overall"].astype(float))
    table["grade_z"] = table.groupby("draft_year")["pre_draft_grade"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
    )
    table["grade_z"] = table["grade_z"].fillna(0.0)
    table["grade_missing"] = table["pre_draft_grade"].isna().astype(float)
    table["usage_filled"] = table.groupby(["draft_year", "nfl_position"])[
        "usage_overall"
    ].transform(lambda s: s.fillna(s.mean()))
    table["usage_filled"] = table["usage_filled"].fillna(table["usage_overall"].mean()).fillna(0.0)
    table["usage_missing"] = table["usage_overall"].isna().astype(float)
    table["arm"] = np.select(
        [
            table["nfl_position"].isin(OFFENSE_ARM_GROUPS),
            table["nfl_position"].isin(DEFENSE_ARM_GROUPS),
            table["nfl_position"].isin(SLOT_ONLY_GROUPS),
        ],
        ["offense", "defense", "slot_only"],
        default="excluded",
    )
    table["target"] = np.where(
        table["arm"].eq("offense"),
        table["skill_rate"],
        np.where(table["arm"].eq("defense"), table["defense_rate"], np.nan),
    )
    enough = np.where(
        table["arm"].eq("offense"),
        table["offense_snaps"].fillna(0.0) >= MIN_ROOKIE_SNAPS,
        np.where(
            table["arm"].eq("defense"),
            table["defense_snaps"].fillna(0.0) >= MIN_ROOKIE_SNAPS,
            False,
        ),
    )
    table["target_usable"] = enough & table["target"].notna()
    table.loc[~table["target_usable"], "target"] = np.nan
    return table


def _arm_design(frame: pd.DataFrame, arm: str) -> np.ndarray:
    columns = [
        frame["log_overall"].to_numpy(dtype=float),
        frame["grade_z"].to_numpy(dtype=float),
        frame["grade_missing"].to_numpy(dtype=float),
    ]
    if arm == "offense":
        columns.append(frame["usage_filled"].to_numpy(dtype=float))
        columns.append(frame["usage_missing"].to_numpy(dtype=float))
        groups = OFFENSE_ARM_GROUPS
    else:
        groups = DEFENSE_ARM_GROUPS
    for group in groups[1:]:
        columns.append(frame["nfl_position"].eq(group).to_numpy(dtype=float))
    return np.column_stack(columns)


def fit_priors(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["prior_value"] = np.nan
    result["prior_sd"] = np.nan
    result["generic_replacement"] = np.nan
    result["surplus_z"] = np.nan
    result["prior_channel"] = "none"
    result["prior_train_rows"] = 0

    for draft_year in sorted(result["draft_year"].unique()):
        train_pool = result.loc[result["draft_year"].lt(draft_year) & result["target_usable"]]
        pooled_frames = []
        arm_fits: dict[str, tuple[Ridge, float, dict[str, float]]] = {}
        for arm in ("offense", "defense"):
            train = train_pool.loc[train_pool["arm"].eq(arm)]
            if len(train) < MIN_TRAIN_CLASS_ROWS:
                continue
            design = _arm_design(train, arm)
            target = train["target"].to_numpy(dtype=float)
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(design, target)
            residual_sd = float(np.sqrt(np.mean((target - model.predict(design)) ** 2)))
            generic = {
                str(group): float(train.loc[train["nfl_position"].eq(group), "target"].mean())
                for group in (OFFENSE_ARM_GROUPS if arm == "offense" else DEFENSE_ARM_GROUPS)
            }
            arm_fits[arm] = (model, residual_sd, generic)
            pooled = train.copy()
            pooled_mean = float(target.mean())
            pooled_sd = float(target.std(ddof=0)) or 1.0
            pooled["pooled_target_z"] = (target - pooled_mean) / pooled_sd
            pooled_frames.append(pooled)

        target_rows = result["draft_year"].eq(draft_year)
        for arm, (model, residual_sd, generic) in arm_fits.items():
            mask = target_rows & result["arm"].eq(arm)
            if not mask.any():
                continue
            frame = result.loc[mask]
            predicted = model.predict(_arm_design(frame, arm))
            baseline = frame["nfl_position"].map(generic).astype(float)
            sd = residual_sd if residual_sd > 0 else 1.0
            result.loc[mask, "prior_value"] = predicted
            result.loc[mask, "prior_sd"] = sd
            result.loc[mask, "generic_replacement"] = baseline.to_numpy()
            result.loc[mask, "surplus_z"] = (predicted - baseline.to_numpy()) / sd
            result.loc[mask, "prior_channel"] = "fitted"
            result.loc[mask, "prior_train_rows"] = len(train_pool.loc[train_pool["arm"].eq(arm)])

        if pooled_frames:
            pooled = pd.concat(pooled_frames, ignore_index=True)
            slot_design = np.column_stack(
                [
                    pooled["log_overall"].to_numpy(dtype=float),
                    pooled["grade_z"].to_numpy(dtype=float),
                    pooled["grade_missing"].to_numpy(dtype=float),
                ]
            )
            slot_target = pooled["pooled_target_z"].to_numpy(dtype=float)
            slot_model = Ridge(alpha=RIDGE_ALPHA)
            slot_model.fit(slot_design, slot_target)
            slot_sd = float(np.sqrt(np.mean((slot_target - slot_model.predict(slot_design)) ** 2)))
            mask = target_rows & result["arm"].eq("slot_only")
            if mask.any():
                frame = result.loc[mask]
                design = np.column_stack(
                    [
                        frame["log_overall"].to_numpy(dtype=float),
                        frame["grade_z"].to_numpy(dtype=float),
                        frame["grade_missing"].to_numpy(dtype=float),
                    ]
                )
                predicted = slot_model.predict(design)
                result.loc[mask, "prior_value"] = predicted
                result.loc[mask, "prior_sd"] = slot_sd if slot_sd > 0 else 1.0
                result.loc[mask, "generic_replacement"] = 0.0
                result.loc[mask, "surplus_z"] = predicted
                result.loc[mask, "prior_channel"] = "slot_only"
                result.loc[mask, "prior_train_rows"] = len(pooled)
    return result


def _block_bootstrap_stat(
    values: np.ndarray,
    blocks: list[np.ndarray],
    statistic,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        positions = np.concatenate([blocks[j] for j in selected])
        draws[i] = statistic(positions)
    finite = draws[~np.isnan(draws)]
    if finite.size == 0:
        return {"ci95": [float("nan"), float("nan")], "probability_positive": float("nan")}
    lower, upper = (float(x) for x in np.quantile(finite, [0.025, 0.975]))
    return {
        "ci95": [lower, upper],
        "probability_positive": float(probability_positive_from_draws(finite)),
        "samples": samples,
        "blocks": len(blocks),
        "nan_draws": int(np.isnan(draws).sum()),
    }


def _correlation_with_ci(
    frame: pd.DataFrame, x_column: str, y_column: str, block_column: str
) -> dict:
    working = frame.dropna(subset=[x_column, y_column]).reset_index(drop=True)
    if len(working) < 10:
        return {"n": len(working), "note": "fewer than 10 usable rows"}
    x = working[x_column].to_numpy(dtype=float)
    y = working[y_column].to_numpy(dtype=float)
    point = float(np.corrcoef(x, y)[0, 1])
    spearman = float(
        np.corrcoef(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1]
    )
    blocks = list(working.groupby(block_column, sort=False).indices.values())

    def _stat(positions: np.ndarray) -> float:
        sx = x[positions]
        sy = y[positions]
        if np.std(sx) == 0 or np.std(sy) == 0:
            return float("nan")
        return float(np.corrcoef(sx, sy)[0, 1])

    boot = _block_bootstrap_stat(y, blocks, _stat, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    return {"n": len(working), "pearson": point, "spearman": spearman, **boot}


def cmd_prior(args: argparse.Namespace) -> None:
    table = fit_priors(build_prior_inputs())
    crosswalk_rows = len(table)
    identity = {
        "draft_rows_2014_2025": crosswalk_rows,
        "linked_to_gsis": int(table["gsis_id"].notna().sum()),
        "identity_ok": int(table["identity_ok"].sum()),
        "surname_agreement_among_linked": float(
            table.loc[table["gsis_id"].notna(), "surname_match"].mean()
        ),
        "espn_id_confirms": int(table["espn_match"].fillna(False).sum()),
        "cfb_usage_linked": int(table["usage_overall"].notna().sum()),
        "pre_draft_grade_present": int(table["pre_draft_grade"].notna().sum()),
    }

    scored = table.loc[table["surplus_z"].notna()].copy()
    coverage_rows = []
    for season, group in scored.groupby("draft_year"):
        starters = group.loc[
            group["prior_channel"].eq("fitted") | group["prior_channel"].eq("slot_only")
        ]
        played = group.loc[group["games"].fillna(0) > 0]
        heavy = group.loc[
            (group["offense_snaps"].fillna(0.0) + group["defense_snaps"].fillna(0.0)) >= 300.0
        ]
        coverage_rows.append(
            {
                "draft_year": int(season),
                "drafted": len(group),
                "with_prior": len(starters),
                "played_a_rookie_game": len(played),
                "rookie_starters_300plus_snaps": len(heavy),
                "rookie_starters_with_prior": len(heavy.loc[heavy["surplus_z"].notna()]),
                "by_channel": heavy["prior_channel"].value_counts().to_dict(),
            }
        )

    measurable = scored.loc[scored["target_usable"]].copy()
    measurable["target_z"] = measurable.groupby("arm")["target"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
    )
    correlation_all = _correlation_with_ci(measurable, "surplus_z", "target_z", "draft_year")
    odd = measurable.loc[measurable["draft_year"] % 2 == 1]
    even = measurable.loc[measurable["draft_year"] % 2 == 0]
    correlation_odd = _correlation_with_ci(odd, "surplus_z", "target_z", "draft_year")
    correlation_even = _correlation_with_ci(even, "surplus_z", "target_z", "draft_year")

    by_arm = {
        arm: _correlation_with_ci(
            measurable.loc[measurable["arm"].eq(arm)], "surplus_z", "target_z", "draft_year"
        )
        for arm in ("offense", "defense")
    }

    halves = measurable.copy()
    halves["rate_odd"] = np.where(
        halves["arm"].eq("offense"), halves["skill_rate_odd"], halves["defense_rate_odd"]
    )
    halves["rate_even"] = np.where(
        halves["arm"].eq("offense"), halves["skill_rate_even"], halves["defense_rate_even"]
    )
    halves = halves.dropna(subset=["rate_odd", "rate_even"])
    half_r = (
        float(np.corrcoef(halves["rate_odd"], halves["rate_even"])[0, 1])
        if len(halves) > 10
        else float("nan")
    )
    spearman_brown = (
        (2.0 * half_r / (1.0 + half_r)) if half_r == half_r and half_r > -1 else float("nan")
    )

    result = {
        "identity_chain": identity,
        "coverage_by_draft_class": coverage_rows,
        "prior_vs_realised_first_season_value": {
            "note": (
                "surplus_z is the standardised prior surplus over the position-group generic "
                "replacement; target_z is the realised rookie-season value on the injury-value "
                "scale, standardised within arm so the two arms are commensurable"
            ),
            "all_measurable": correlation_all,
            "odd_draft_years": correlation_odd,
            "even_draft_years": correlation_even,
            "by_arm": by_arm,
        },
        "outcome_split_half_reliability_odd_even_weeks": {
            "n": len(halves),
            "pearson_odd_vs_even_weeks": half_r,
            "spearman_brown": spearman_brown,
        },
        "decay_rule": {
            "form": "weight(n) = N0 / (n + N0)",
            "N0_snaps": N0_SNAPS,
            "source": "frozen at XLG-06 Stage 3 (docs/xlg06_stage3_prior_spec.md), "
            "not retuned here",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "rookie_prior_table.parquet", index=False)
    (OUT_DIR / "prior_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'prior_results.json'}")


def _prior_table() -> pd.DataFrame:
    path = OUT_DIR / "rookie_prior_table.parquet"
    if not path.is_file():
        raise SystemExit("run `prior` first: rookie_prior_table.parquet is missing")
    return pd.read_parquet(path)


def rookie_week_state() -> pd.DataFrame:
    priors = _prior_table()
    priors = priors.loc[priors["identity_ok"]].copy()
    priors["gsis_id"] = priors["gsis_id"].astype(str)
    panel = load_snap_panel()
    panel = panel.merge(
        priors[
            [
                "gsis_id",
                "draft_year",
                "draft_round",
                "overall",
                "nfl_position",
                "nfl_position_detail",
                "is_pass_rush",
                "surplus_z",
                "prior_channel",
                "draft_team",
            ]
        ],
        on="gsis_id",
        how="inner",
    )
    rookie = panel.loc[panel["season"].eq(panel["draft_year"])].copy()
    rookie = rookie.sort_values(["gsis_id", "week"]).reset_index(drop=True)
    rookie["share"] = rookie["offense_pct"] + rookie["defense_pct"]
    rookie["snaps"] = rookie["offense_snaps"] + rookie["defense_snaps"]
    grouped = rookie.groupby("gsis_id", sort=False)
    rookie["prior_snaps"] = grouped["snaps"].cumsum() - rookie["snaps"]
    rookie["prior_share_sum"] = grouped["share"].cumsum() - rookie["share"]
    rookie["prior_games"] = grouped.cumcount()
    return rookie


def week1_expected_share(rookie: pd.DataFrame) -> pd.DataFrame:
    week1 = rookie.loc[rookie["week"].eq(1)].copy()
    rows = []
    for season in sorted(rookie["season"].unique()):
        history = week1.loc[week1["season"].lt(season)]
        if history.empty:
            continue
        table = history.groupby(["draft_round", "nfl_position"], as_index=False)["share"].mean()
        table["season"] = int(season)
        rows.append(table)
    if not rows:
        return pd.DataFrame(columns=["draft_round", "nfl_position", "share", "season"])
    return pd.concat(rows, ignore_index=True).rename(columns={"share": "expected_week1_share"})


def carry_forward_presence(rookie: pd.DataFrame) -> pd.DataFrame:
    weeks = sorted(int(w) for w in rookie["week"].unique())
    carried = []
    static_columns = [
        "gsis_id",
        "season",
        "draft_year",
        "draft_round",
        "overall",
        "nfl_position",
        "nfl_position_detail",
        "is_pass_rush",
        "surplus_z",
        "prior_channel",
        "draft_team",
    ]
    for (gsis_id, season), group in rookie.groupby(["gsis_id", "season"], sort=False):
        ordered = group.sort_values("week")
        first_week = int(ordered["week"].iloc[0])
        last_week = max(weeks)
        grid = pd.DataFrame({"week": [w for w in weeks if first_week <= w <= last_week]})
        grid["gsis_id"] = gsis_id
        grid["season"] = season
        merged = grid.merge(
            ordered[["week", "team", "prior_snaps", "prior_share_sum", "prior_games"]],
            on="week",
            how="left",
        )
        merged[["team", "prior_snaps", "prior_share_sum", "prior_games"]] = merged[
            ["team", "prior_snaps", "prior_share_sum", "prior_games"]
        ].ffill()
        for column in static_columns:
            if column in ("gsis_id", "season"):
                continue
            merged[column] = ordered[column].iloc[0]
        carried.append(merged)
    if not carried:
        return rookie
    result = pd.concat(carried, ignore_index=True)
    result["share"] = np.nan
    return result


def team_week_surplus(presence: str = "observed") -> pd.DataFrame:
    rookie = rookie_week_state()
    expected = week1_expected_share(rookie)
    if presence == "carry_forward":
        rookie = carry_forward_presence(rookie)
    merged = rookie.merge(expected, on=["season", "draft_round", "nfl_position"], how="left")
    merged["expected_week1_share"] = merged["expected_week1_share"].fillna(0.0)
    trailing = np.where(
        merged["prior_games"] > 0,
        merged["prior_share_sum"] / merged["prior_games"].replace(0, np.nan),
        np.nan,
    )
    merged["expected_share"] = np.where(
        merged["prior_games"] > 0,
        trailing,
        np.where(merged["week"].eq(1), merged["expected_week1_share"], 0.0),
    )
    merged["expected_share"] = pd.to_numeric(merged["expected_share"], errors="coerce").fillna(0.0)
    merged["decay_weight"] = N0_SNAPS / (merged["prior_snaps"].astype(float) + N0_SNAPS)
    merged["surplus_z"] = pd.to_numeric(merged["surplus_z"], errors="coerce").fillna(0.0)
    merged["contribution"] = merged["expected_share"] * merged["decay_weight"] * merged["surplus_z"]
    aggregated = merged.groupby(["season", "week", "team"], as_index=False).agg(
        team_surplus=("contribution", "sum"),
        rookies=("gsis_id", "nunique"),
    )
    return aggregated, merged


def game_level_feature(presence: str = "observed") -> pd.DataFrame:
    aggregated, players = team_week_surplus(presence=presence)
    features = pd.read_parquet(
        GAME_FEATURES,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "result",
            "spread_line",
        ],
    )
    games = features.loc[features["game_type"].eq("REG")].copy()
    games["game_id"] = games["game_id"].astype(str)
    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)
    home = aggregated.rename(
        columns={"team": "home_team", "team_surplus": "home_surplus", "rookies": "home_rookies"}
    )
    away = aggregated.rename(
        columns={"team": "away_team", "team_surplus": "away_surplus", "rookies": "away_rookies"}
    )
    games = games.merge(home, on=["season", "week", "home_team"], how="left")
    games = games.merge(away, on=["season", "week", "away_team"], how="left")
    for column in ("home_surplus", "away_surplus", "home_rookies", "away_rookies"):
        games[column] = games[column].fillna(0.0)
    games["surplus_diff"] = games["home_surplus"] - games["away_surplus"]

    games["threshold"] = np.nan
    for season in sorted(games["season"].unique()):
        history = games.loc[games["season"].lt(season)]
        if history["season"].nunique() < MIN_PRIOR_SEASONS:
            continue
        value = float(np.percentile(history["surplus_diff"].abs(), FLAG_PERCENTILE))
        games.loc[games["season"].eq(season), "threshold"] = value
    games["flag"] = 0
    fires = games["threshold"].notna()
    games.loc[fires & games["surplus_diff"].gt(games["threshold"]), "flag"] = 1
    games.loc[fires & games["surplus_diff"].lt(-games["threshold"]), "flag"] = -1
    return games, players


def _opener_pairing() -> pd.DataFrame:
    schedule = pd.read_parquet(GAME_FEATURES, columns=["game_id", "season", "week"])
    schedule["game_id"] = schedule["game_id"].astype(str)
    pairing = cached_pairing_table(
        MARKET_ROOT,
        capture_kind=HISTORICAL_CAPTURE_KIND,
        labels=("tue_open",),
        schedule=schedule,
    )
    pairing["game_id"] = pairing["game_id"].astype(str)
    return pairing[["game_id", "home_spread"]].rename(
        columns={"home_spread": "tue_open_home_spread"}
    )


def cmd_cover_rates(_args: argparse.Namespace) -> None:
    games, players = game_level_feature()
    pairing = _opener_pairing()
    graded = games.merge(pairing, on="game_id", how="inner")
    graded["margin_vs_open"] = graded["result"] - graded["tue_open_home_spread"]
    graded = graded.loc[graded["margin_vs_open"].ne(0.0)].copy()
    graded["home_cover_at_open"] = graded["margin_vs_open"].gt(0.0).astype(float)

    trailing = players.copy()
    trailing["trailing_share"] = np.where(
        trailing["prior_games"] > 0,
        trailing["prior_share_sum"] / trailing["prior_games"].replace(0, np.nan),
        np.nan,
    )
    is_high_leverage = (
        trailing["nfl_position_detail"].eq("QB")
        | trailing["nfl_position"].eq("OL")
        | trailing["is_pass_rush"].fillna(False)
    )
    starting = np.where(
        trailing["week"].eq(1),
        trailing["overall"].le(32),
        pd.to_numeric(trailing["trailing_share"], errors="coerce").fillna(0.0) >= 0.5,
    )
    trailing["rookie_high_leverage_starter"] = is_high_leverage & starting
    starters = (
        trailing.loc[trailing["rookie_high_leverage_starter"]]
        .groupby(["season", "week", "team"], as_index=False)
        .agg(n_rookie_starters=("gsis_id", "nunique"))
    )

    rows = []
    for side, opponent in (("home", "away"), ("away", "home")):
        frame = graded.merge(
            starters.rename(columns={"team": f"{side}_team"}),
            on=["season", "week", f"{side}_team"],
            how="inner",
        )
        frame = frame.assign(
            side=side,
            team=frame[f"{side}_team"],
            opponent=frame[f"{opponent}_team"],
            team_cover=np.where(
                side == "home", frame["home_cover_at_open"], 1.0 - frame["home_cover_at_open"]
            ),
        )
        rows.append(
            frame[
                [
                    "game_id",
                    "season",
                    "week",
                    "side",
                    "team",
                    "opponent",
                    "n_rookie_starters",
                    "team_cover",
                ]
            ]
        )
    starter_games = pd.concat(rows, ignore_index=True)
    starter_games["early"] = starter_games["week"].le(6)
    starter_games["week_block"] = starter_games["season"] * 100 + starter_games["week"]

    working = starter_games.reset_index(drop=True)
    cover = working["team_cover"].to_numpy(dtype=float)
    early = working["early"].to_numpy(dtype=bool)
    blocks = list(working.groupby("week_block", sort=False).indices.values())

    def _gap(positions: np.ndarray) -> float:
        c = cover[positions]
        e = early[positions]
        if e.sum() == 0 or (~e).sum() == 0:
            return float("nan")
        return float(c[e].mean() - c[~e].mean())

    def _early_rate(positions: np.ndarray) -> float:
        e = early[positions]
        if e.sum() == 0:
            return float("nan")
        return float(cover[positions][e].mean() - 0.5)

    def _late_rate(positions: np.ndarray) -> float:
        e = early[positions]
        if (~e).sum() == 0:
            return float("nan")
        return float(cover[positions][~e].mean() - 0.5)

    gap = _block_bootstrap_stat(cover, blocks, _gap, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    early_boot = _block_bootstrap_stat(
        cover, blocks, _early_rate, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
    )
    late_boot = _block_bootstrap_stat(cover, blocks, _late_rate, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)

    by_group = []
    for label, mask in (
        ("qb", trailing["nfl_position_detail"].eq("QB")),
        ("ol", trailing["nfl_position"].eq("OL")),
        ("pass_rusher", trailing["is_pass_rush"].fillna(False)),
    ):
        subset = trailing.loc[mask & trailing["rookie_high_leverage_starter"]]
        keys = subset.groupby(["season", "week", "team"], as_index=False).agg(
            n=("gsis_id", "nunique")
        )
        sub_rows = []
        for side in ("home", "away"):
            frame = graded.merge(
                keys.rename(columns={"team": f"{side}_team"}),
                on=["season", "week", f"{side}_team"],
                how="inner",
            )
            sub_rows.append(
                pd.DataFrame(
                    {
                        "week": frame["week"].to_numpy(),
                        "season": frame["season"].to_numpy(),
                        "team_cover": (
                            frame["home_cover_at_open"].to_numpy()
                            if side == "home"
                            else 1.0 - frame["home_cover_at_open"].to_numpy()
                        ),
                    }
                )
            )
        combined = pd.concat(sub_rows, ignore_index=True)
        if combined.empty:
            by_group.append({"group": label, "n": 0})
            continue
        combined["early"] = combined["week"].le(6)
        by_group.append(
            {
                "group": label,
                "n": len(combined),
                "n_early": int(combined["early"].sum()),
                "n_late": int((~combined["early"]).sum()),
                "cover_rate_weeks_1_6": float(combined.loc[combined["early"], "team_cover"].mean())
                if combined["early"].any()
                else float("nan"),
                "cover_rate_weeks_7plus": float(
                    combined.loc[~combined["early"], "team_cover"].mean()
                )
                if (~combined["early"]).any()
                else float("nan"),
            }
        )

    result = {
        "grade": "opener",
        "paired_seasons": sorted(int(s) for s in graded["season"].unique()),
        "n_paired_opener_games": len(graded),
        "starter_definition": (
            "weeks 2+: rookie mean (offense_pct + defense_pct) over completed games this "
            "season >= 0.50; week 1: first-round pick (overall <= 32) at QB / OL / pass-rusher"
        ),
        "n_team_games_with_rookie_high_leverage_starter": len(starter_games),
        "weeks_1_6": {
            "n": int(starter_games["early"].sum()),
            "cover_rate": float(starter_games.loc[starter_games["early"], "team_cover"].mean()),
            "excess_over_50pct_ci95": early_boot["ci95"],
            "probability_above_50pct": early_boot["probability_positive"],
        },
        "weeks_7plus": {
            "n": int((~starter_games["early"]).sum()),
            "cover_rate": float(starter_games.loc[~starter_games["early"], "team_cover"].mean()),
            "excess_over_50pct_ci95": late_boot["ci95"],
            "probability_above_50pct": late_boot["probability_positive"],
        },
        "early_minus_late_gap": {
            "point_estimate": float(
                starter_games.loc[starter_games["early"], "team_cover"].mean()
                - starter_games.loc[~starter_games["early"], "team_cover"].mean()
            ),
            **gap,
        },
        "by_position_group": by_group,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    starter_games.to_parquet(OUT_DIR / "rookie_starter_games.parquet", index=False)
    (OUT_DIR / "cover_rate_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'cover_rate_results.json'}")


def _accuracy_metric(frame: pd.DataFrame) -> dict:
    valid = frame.dropna(subset=["baseline_correct", "candidate_correct"])
    return {
        "accuracy_points": 100.0
        * float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": 100.0 * float(valid["candidate_correct"].mean()),
        "baseline_accuracy": 100.0 * float(valid["baseline_correct"].mean()),
    }


def _summarize(frame: pd.DataFrame) -> dict:
    point = _accuracy_metric(frame)
    week = week_blocked_bootstrap(
        frame, _accuracy_metric, block="week", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    season = week_blocked_bootstrap(
        frame, _accuracy_metric, block="season", samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    w = week.loc[week["metric"].eq("accuracy_points")].iloc[0]
    s = season.loc[season["metric"].eq("accuracy_points")].iloc[0]
    return {
        **point,
        "week_blocked_ci95": [float(w["lower"]), float(w["upper"])],
        "week_blocked_probability_positive": float(w["probability_positive"]),
        "season_blocked_ci95": [float(s["lower"]), float(s["upper"])],
        "season_blocked_probability_positive": float(s["probability_positive"]),
        "n_games": len(frame),
        "n_weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(frame["season"].nunique()),
    }


def _apply_tilt(scored: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["game_id"] = scored["game_id"].astype(str)
    flags = games[["game_id", "flag", "surplus_diff", "threshold"]].copy()
    flags["game_id"] = flags["game_id"].astype(str)
    scored = scored.merge(flags, on="game_id", how="left")
    scored["flag"] = scored["flag"].fillna(0).astype(int)
    scored["baseline_pick_home"] = scored["home_cover_probability_at_open"].ge(0.5)
    scored["candidate_pick_home"] = np.where(
        scored["flag"].eq(1),
        True,
        np.where(scored["flag"].eq(-1), False, scored["baseline_pick_home"]),
    )
    scored["baseline_correct"] = pick_correct(
        scored["baseline_pick_home"], scored["margin_vs_open"]
    )
    scored["candidate_correct"] = pick_correct(
        pd.Series(scored["candidate_pick_home"], index=scored.index), scored["margin_vs_open"]
    )
    return scored


def cmd_production(_args: argparse.Namespace) -> None:
    registry = load_registry()
    features = pd.read_parquet(PRODUCTION_FEATURES)
    training, window = confirmation_split(features, registry, FAMILY)
    if pd.to_datetime(training["gameday"]).max() >= pd.to_datetime(window["gameday"]).min():
        raise SystemExit("confirmation split leaked a training row into the assigned window")
    seasons = tuple(sorted(int(s) for s in window["season"].unique()))
    scoped = pd.concat([training, window], ignore_index=True)

    config = resolve_active_model_config(ARTIFACTS_ROOT)
    evaluated = opener_pick_evaluation(
        MARKET_ROOT,
        scoped,
        active_model_config=config,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
    )
    games, _players = game_level_feature()

    in_window = evaluated.loc[evaluated["season"].astype(int).isin(seasons)].reset_index(drop=True)
    window_scored = _apply_tilt(in_window, games)
    window_graded = window_scored.dropna(subset=["baseline_correct", "candidate_correct"])

    full_evaluated = opener_pick_evaluation(
        MARKET_ROOT,
        features.loc[features["game_type"].eq("REG")].reset_index(drop=True)
        if "game_type" in features.columns
        else features,
        active_model_config=config,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
    )
    archive = full_evaluated.reset_index(drop=True)
    archive_scored = _apply_tilt(archive, games)
    archive_graded = archive_scored.dropna(subset=["baseline_correct", "candidate_correct"])

    control = window_scored.copy()
    control["candidate_correct"] = pick_correct(
        control["margin_vs_open"].gt(0.0), control["margin_vs_open"]
    )
    control_graded = control.dropna(subset=["baseline_correct", "candidate_correct"])

    per_season = []
    for season, group in archive_graded.groupby("season"):
        metric = _accuracy_metric(group)
        boot = week_blocked_bootstrap(
            group,
            _accuracy_metric,
            block="week",
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        row = boot.loc[boot["metric"].eq("accuracy_points")].iloc[0]
        per_season.append(
            {
                "season": int(season),
                "n_games": len(group),
                "picks_changed": int(
                    (group["baseline_pick_home"] != group["candidate_pick_home"]).sum()
                ),
                **metric,
                "week_blocked_ci95": [float(row["lower"]), float(row["upper"])],
                "week_blocked_probability_positive": float(row["probability_positive"]),
            }
        )

    flagged_full = games.loc[games["threshold"].notna()]
    result = {
        "family": FAMILY,
        "grade": "opener",
        "window_seasons": list(seasons),
        "active_model_id": config.get("model_id"),
        "predeclared_direction": (
            "tilt TOWARD the side whose decayed, snap-share-weighted rookie prior surplus "
            "is larger; +1 favours home, -1 favours away"
        ),
        "decay": {"form": "N0 / (n + N0)", "N0_snaps": N0_SNAPS},
        "flag_threshold": {
            "rule": f"|surplus_diff| above the {FLAG_PERCENTILE:.0f}th percentile of strictly "
            f"prior seasons (needs {MIN_PRIOR_SEASONS} prior seasons)",
            "full_schedule_games_with_threshold": len(flagged_full),
            "full_schedule_flag_rate": float(flagged_full["flag"].ne(0).mean())
            if len(flagged_full)
            else float("nan"),
            "thresholds_by_season": {
                str(int(s)): float(v)
                for s, v in games.dropna(subset=["threshold"])
                .groupby("season")["threshold"]
                .first()
                .items()
            },
        },
        "window": {
            "picks_changed": int(
                (window_scored["baseline_pick_home"] != window_scored["candidate_pick_home"]).sum()
            ),
            "picks_changed_graded": int(
                (window_graded["baseline_pick_home"] != window_graded["candidate_pick_home"]).sum()
            ),
            "n_flagged": int(window_scored["flag"].ne(0).sum()),
            "summary": _summarize(window_graded),
        },
        "full_opener_archive_secondary": {
            "seasons": sorted(int(s) for s in archive_graded["season"].unique()),
            "picks_changed": int(
                (
                    archive_scored["baseline_pick_home"] != archive_scored["candidate_pick_home"]
                ).sum()
            ),
            "n_flagged": int(archive_scored["flag"].ne(0).sum()),
            "summary": _summarize(archive_graded),
            "per_season": per_season,
        },
        "positive_control": {
            "description": "candidate replaced by the realised opener margin sign",
            "summary": _summarize(control_graded),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    window_scored.to_parquet(OUT_DIR / "production_window_paired.parquet", index=False)
    archive_scored.to_parquet(OUT_DIR / "production_archive_paired.parquet", index=False)
    (OUT_DIR / "production_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'production_results.json'}")


def cmd_leakage_control(_args: argparse.Namespace) -> None:
    features = pd.read_parquet(PRODUCTION_FEATURES)
    config = resolve_active_model_config(ARTIFACTS_ROOT)
    evaluated = opener_pick_evaluation(
        MARKET_ROOT,
        features.loc[features["game_type"].eq("REG")].reset_index(drop=True)
        if "game_type" in features.columns
        else features,
        active_model_config=config,
        min_train_games=DEFAULT_MIN_TRAIN_GAMES,
    ).reset_index(drop=True)

    observed_games, _observed_players = game_level_feature(presence="observed")
    carried_games, _carried_players = game_level_feature(presence="carry_forward")

    observed = _apply_tilt(evaluated, observed_games)
    carried = _apply_tilt(evaluated, carried_games)
    observed_graded = observed.dropna(subset=["baseline_correct", "candidate_correct"])
    carried_graded = carried.dropna(subset=["baseline_correct", "candidate_correct"])

    agreement = float((observed["candidate_pick_home"] == carried["candidate_pick_home"]).mean())
    result = {
        "family": FAMILY,
        "grade": "opener",
        "read": "full paired 2020-2025 opener archive; NOT a rotation window draw",
        "active_model_id": config.get("model_id"),
        "question": (
            "a rookie appearing in snap counts in week w reveals he was active that week, "
            "which a Tuesday pick deadline does not know. This control removes that: once a "
            "rookie has appeared, his last-known trailing share and snap total are carried "
            "forward to every later week whether or not he appeared, so no game-day "
            "activation is used anywhere in the feature"
        ),
        "pick_agreement_between_arms": agreement,
        "observed_presence": {
            "picks_changed": int(
                (observed["baseline_pick_home"] != observed["candidate_pick_home"]).sum()
            ),
            "n_flagged": int(observed["flag"].ne(0).sum()),
            "summary": _summarize(observed_graded),
        },
        "carry_forward_presence": {
            "picks_changed": int(
                (carried["baseline_pick_home"] != carried["candidate_pick_home"]).sum()
            ),
            "n_flagged": int(carried["flag"].ne(0).sum()),
            "summary": _summarize(carried_graded),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    carried.to_parquet(OUT_DIR / "leakage_control_paired.parquet", index=False)
    (OUT_DIR / "leakage_control_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {OUT_DIR / 'leakage_control_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XLG-06 rookie priors: build the prior, read early-season cover rates, "
        "and score the rookie-prior-surplus tilt on production at the opener"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prior", help="build the rookie prior, its coverage, reliability and fit")
    sub.add_parser(
        "cover-rates", help="opener cover rates for teams starting rookies early vs late"
    )
    sub.add_parser("production", help="rookie-prior-surplus tilt stacked on production weak_stack")
    sub.add_parser(
        "leakage-control",
        help="same tilt with game-day activation removed from the feature, full archive",
    )
    args = parser.parse_args()
    if args.command == "prior":
        cmd_prior(args)
    elif args.command == "cover-rates":
        cmd_cover_rates(args)
    elif args.command == "production":
        cmd_production(args)
    elif args.command == "leakage-control":
        cmd_leakage_control(args)


if __name__ == "__main__":
    main()
