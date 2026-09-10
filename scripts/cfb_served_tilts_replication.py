"""The six served NFL tilts, replicated on COLLEGE FOOTBALL.

Predeclared in ``docs/cfb_served_tilts.md`` BEFORE this script was pointed at
any outcome column. Read that document first: it freezes the population, the
per-mechanism replicability verdicts, the flag definitions, the comparator,
the grading convention, the era split, the positive control and the recording
rules. A CFB replication is corroboration of a mechanism, never a second
independent NFL evidence point, and nothing here changes any served behaviour.

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands
on its own): an interval containing zero is NEVER grounds to reject, fail or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``,
never the binary "contains zero". If a record command errors, the verdict is
wrong, not the validator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"F:\Repos\nfl_py3")
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.cfb_benchmark import CFB_CLEAN_CORE_SEASONS  # noqa: E402
from nfl_ats.cfb_features import cfb_competitive_plays  # noqa: E402
from nfl_ats.cfb_qb_dependence import split_half_reliability  # noqa: E402
from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.provenance import write_stamped_artifact  # noqa: E402

FEATURES_PATH = DATA_ROOT / "data" / "processed" / "cfb_game_features.parquet"
BENCHMARK_PATH = (
    DATA_ROOT / "artifacts" / "cfb_benchmark" / "20260818T115149Z" / "predictions.parquet"
)
SCHEDULES_ROOT = DATA_ROOT / "data" / "cfb" / "schedules" / "raw"
PBP_ROOT = DATA_ROOT / "data" / "cfb" / "pbp" / "raw"
ARTIFACT_ROOT = DATA_ROOT / "artifacts" / "cfb_served_tilts"
COACHES_PATH = Path(
    r"C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3"
    r"\d0c6bd6d-414a-45cb-ab4f-bd3c0015f1dd\scratchpad\laneAL\cfb_coaches.parquet"
)

BOOTSTRAP_SAMPLES = 20_000
SEED = 20260821
HISTORY_START_SEASON = 2011

ERAS: tuple[tuple[str, int, int], ...] = (("2012_2019", 2012, 2019), ("2021_2025", 2021, 2025))

POST_BYE_GAP_DAYS = 12
WINDOW_GAMES = 4
MIN_WINDOW_OBS = 3
MIN_QUANTILE_POOL = 200
QUARTILE_UNASSIGNED = -1
QUARTILE_BOTTOM = 0
QUARTILE_MIDDLE = 1
QUARTILE_TOP = 2
TANK_ZONE_SIZE = 2
LAST_WEEKS = 5
BOWL_ELIGIBLE_WINS = 6

GRADES = ("opener", "close")


def latest_snapshot_dir(root: Path) -> Path:
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no snapshot under {root}")
    return candidates[0]


def load_cfb_schedules() -> pd.DataFrame:
    """Every CFB regular-season game in the newest schedules snapshot."""

    snapshot = latest_snapshot_dir(SCHEDULES_ROOT)
    frames = [pd.read_parquet(path) for path in sorted(snapshot.glob("season=*/schedules.parquet"))]
    schedules = pd.concat(frames, ignore_index=True)
    schedules = schedules.loc[schedules["season_type"].astype(str).eq("regular")].copy()
    schedules["season"] = pd.to_numeric(schedules["season"], errors="coerce").astype("Int64")
    schedules["week"] = pd.to_numeric(schedules["week"], errors="coerce").astype("Int64")
    schedules["kickoff"] = pd.to_datetime(schedules["start_date"], errors="coerce", utc=True)
    schedules["gameday"] = schedules["kickoff"].dt.tz_convert(None).dt.normalize()
    for column in ("home_id", "away_id"):
        schedules[column] = pd.to_numeric(schedules[column], errors="coerce").astype("Int64")
    schedules["game_id"] = pd.to_numeric(schedules["game_id"], errors="coerce").astype("int64")
    return schedules.dropna(subset=["season", "gameday", "home_id", "away_id"]).reset_index(
        drop=True
    )


def load_population() -> pd.DataFrame:
    """The frozen CFB benchmark ridge arm on the clean core, with both grades."""

    features = pd.read_parquet(FEATURES_PATH)
    predictions = pd.read_parquet(BENCHMARK_PATH)
    ridge = predictions.loc[predictions["method"].astype(str).eq("market_residual")].copy()
    keep = [
        "game_id",
        "home_id",
        "away_id",
        "home_team",
        "away_team",
        "spread_open",
        "total_line",
        "neutral_site",
    ]
    merged = ridge[
        ["game_id", "season", "week", "gameday", "spread_line", "result", "home_cover_probability"]
    ].merge(features[keep], on="game_id", how="left", validate="one_to_one")
    merged = merged.loc[merged["season"].isin(CFB_CLEAN_CORE_SEASONS)].copy()
    merged["season"] = merged["season"].astype(int)
    merged["week"] = merged["week"].astype(int)
    merged["baseline_pick_home"] = pd.to_numeric(
        merged["home_cover_probability"], errors="coerce"
    ).ge(0.5)
    merged["settle_opener"] = pd.to_numeric(merged["result"], errors="coerce") - pd.to_numeric(
        merged["spread_open"], errors="coerce"
    )
    merged["settle_close"] = pd.to_numeric(merged["result"], errors="coerce") - pd.to_numeric(
        merged["spread_line"], errors="coerce"
    )
    return merged.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _team_long(schedules: pd.DataFrame) -> pd.DataFrame:
    sides = []
    for side in ("home", "away"):
        sides.append(
            pd.DataFrame(
                {
                    "game_id": schedules["game_id"].to_numpy(),
                    "season": schedules["season"].astype(int).to_numpy(),
                    "week": schedules["week"].astype("Int64").to_numpy(),
                    "gameday": schedules["gameday"].to_numpy(),
                    "kickoff": schedules["kickoff"].to_numpy(),
                    "team_id": schedules[f"{side}_id"].astype(int).to_numpy(),
                    "opponent_id": schedules[f"{'away' if side == 'home' else 'home'}_id"]
                    .astype(int)
                    .to_numpy(),
                    "side": side,
                    "points_for": pd.to_numeric(
                        schedules[f"{side}_points"], errors="coerce"
                    ).to_numpy(),
                    "points_against": pd.to_numeric(
                        schedules[f"{'away' if side == 'home' else 'home'}_points"], errors="coerce"
                    ).to_numpy(),
                }
            )
        )
    return pd.concat(sides, ignore_index=True)


def bye_edge_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    """``home_off_bye``/``away_off_bye`` per game, the NFL rule with the league swapped.

    Ported from ``nfl_ats.bye_edge_fade_overlay.bye_edge_flag_by_game``: melt
    each regular-season game into two team rows, sort each team's rows within
    its own season by gameday, take the day gap to the immediately preceding
    game in that ``(team, season)`` group, and flag ``gap_days >= 12``. A
    team's first game of a season has no preceding game, so its gap is
    undefined and folds to False, exactly as the source does.
    """

    long_df = _team_long(schedules).sort_values(["team_id", "season", "gameday"])
    long_df["gap_days"] = long_df.groupby(["team_id", "season"])["gameday"].diff().dt.days
    long_df["off_bye"] = (long_df["gap_days"] >= POST_BYE_GAP_DAYS).fillna(False).astype(bool)
    wide = long_df.pivot_table(
        index="game_id", columns="side", values="off_bye", aggfunc="first"
    ).rename(columns={"home": "home_off_bye", "away": "away_off_bye"})
    for column in ("home_off_bye", "away_off_bye"):
        if column not in wide.columns:
            wide[column] = False
        wide[column] = wide[column].fillna(False).astype(bool)
    return wide.reset_index()[["game_id", "home_off_bye", "away_off_bye"]]


def load_cfb_pbp(seasons: list[int]) -> pd.DataFrame:
    """The newest CFB play-by-play snapshot for the requested seasons."""

    columns = [
        "game_id",
        "season",
        "week",
        "seasonType",
        "pos_team_id",
        "def_pos_team_id",
        "homeTeamId",
        "awayTeamId",
        "is_home",
        "EPA",
        "EPA_success",
        "rush",
        "pass",
        "kneel_down",
        "statYardage",
        "home_wp_before",
        "away_wp_before",
        "type.text",
    ]
    frames = []
    for season in seasons:
        matches = sorted(PBP_ROOT.glob(f"*/season={season}/plays.parquet"), reverse=True)
        if not matches:
            continue
        frames.append(pd.read_parquet(matches[0], columns=columns))
    if not frames:
        raise FileNotFoundError("no CFB play-by-play found")
    return pd.concat(frames, ignore_index=True)


def build_sack_traits(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game sack-allowed and sack-generated rate on competitive dropbacks.

    The CFB substitute for the NFL pressure rate: college play-by-play has no
    QB-hit field, so pressure = sack alone. Declared as a deviation in
    docs/cfb_served_tilts.md section 2.3, never as a faithful port.
    """

    plays = cfb_competitive_plays(pbp)
    plays = plays.loc[plays["competitive_play"]].copy()
    dropbacks = plays.loc[plays["pass"]].copy()
    dropbacks["sack"] = dropbacks["type.text"].astype(str).eq("Sack").astype(float)
    dropbacks["game_id"] = pd.to_numeric(dropbacks["game_id"], errors="coerce").astype("int64")
    dropbacks["pos_team_id"] = pd.to_numeric(dropbacks["pos_team_id"], errors="coerce").astype(
        "Int64"
    )
    dropbacks["def_pos_team_id"] = pd.to_numeric(
        dropbacks["def_pos_team_id"], errors="coerce"
    ).astype("Int64")
    allowed = (
        dropbacks.groupby(["game_id", "pos_team_id"], sort=False)["sack"]
        .mean()
        .rename("press_allow_g")
        .reset_index()
        .rename(columns={"pos_team_id": "team_id"})
    )
    generated = (
        dropbacks.groupby(["game_id", "def_pos_team_id"], sort=False)["sack"]
        .mean()
        .rename("press_gen_g")
        .reset_index()
        .rename(columns={"def_pos_team_id": "team_id"})
    )
    traits = allowed.merge(generated, on=["game_id", "team_id"], how="outer")
    traits["team_id"] = pd.to_numeric(traits["team_id"], errors="coerce").astype("Int64")
    return traits.dropna(subset=["team_id"])


def expanding_quartile_flags(values: pd.Series, blocks: pd.Series) -> np.ndarray:
    """Quartile code per row from STRICTLY EARLIER week blocks only, the screen's routine."""

    raw = values.to_numpy(dtype=np.float64)
    block_values = blocks.to_numpy()
    sort_order = np.argsort(block_values, kind="stable")
    sorted_values = raw[sort_order]
    sorted_blocks = block_values[sort_order]
    sorted_flags = np.full(len(raw), np.int8(QUARTILE_UNASSIGNED))

    pool: list[np.ndarray] = []
    start = 0
    total = len(raw)
    while start < total:
        end = start
        while end < total and sorted_blocks[end] == sorted_blocks[start]:
            end += 1
        if pool:
            pooled = np.concatenate(pool)
            if len(pooled) >= MIN_QUANTILE_POOL:
                q25, q75 = np.quantile(pooled, [0.25, 0.75])
                segment = sorted_values[start:end]
                assigned = ~np.isnan(segment)
                codes = np.where(
                    segment <= q25,
                    np.int8(QUARTILE_BOTTOM),
                    np.where(segment >= q75, np.int8(QUARTILE_TOP), np.int8(QUARTILE_MIDDLE)),
                )
                sorted_flags[start:end] = np.where(assigned, codes, np.int8(QUARTILE_UNASSIGNED))
        present = sorted_values[start:end]
        present = present[~np.isnan(present)]
        if len(present):
            pool.append(present)
        start = end

    result = np.empty(total, dtype=np.int8)
    result[sort_order] = sorted_flags
    return result


def protection_flags(
    schedules: pd.DataFrame, traits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``back_side`` per game under the sack-only protection-mismatch rule."""

    long_df = _team_long(schedules)
    long_df = long_df.merge(traits, on=["game_id", "team_id"], how="left")
    long_df["week_block"] = long_df["season"].astype(int) * 100 + long_df["week"].fillna(0).astype(
        int
    )
    long_df = long_df.sort_values(["team_id", "gameday", "game_id"]).reset_index(drop=True)
    for source, target in (("press_allow_g", "press_allow_w"), ("press_gen_g", "press_gen_w")):
        values = pd.to_numeric(long_df[source], errors="coerce")
        long_df[target] = values.groupby(long_df["team_id"]).transform(
            lambda series: series.shift(1).rolling(WINDOW_GAMES, min_periods=MIN_WINDOW_OBS).mean()
        )
    long_df["press_allow_q"] = expanding_quartile_flags(
        long_df["press_allow_w"], long_df["week_block"]
    )
    long_df["press_gen_q"] = expanding_quartile_flags(long_df["press_gen_w"], long_df["week_block"])

    home = long_df.loc[long_df["side"].eq("home")].set_index("game_id")
    away = long_df.loc[long_df["side"].eq("away")].set_index("game_id")
    index = schedules["game_id"].to_numpy()
    table = pd.DataFrame({"game_id": index}).set_index("game_id")
    home_flagged = home["press_allow_q"].eq(QUARTILE_TOP) & away["press_gen_q"].eq(QUARTILE_TOP)
    away_flagged = away["press_allow_q"].eq(QUARTILE_TOP) & home["press_gen_q"].eq(QUARTILE_TOP)
    table["home_offense_flagged"] = home_flagged.reindex(table.index).fillna(False).astype(bool)
    table["away_offense_flagged"] = away_flagged.reindex(table.index).fillna(False).astype(bool)
    only_home = table["home_offense_flagged"] & ~table["away_offense_flagged"]
    only_away = table["away_offense_flagged"] & ~table["home_offense_flagged"]
    table["back_side"] = np.where(only_home, "AWAY", np.where(only_away, "HOME", ""))
    panel = long_df.loc[
        :, ["team_id", "season", "week", "game_id", "side", "press_allow_q", "press_gen_q"]
    ].copy()
    panel["allow_top_quartile"] = panel["press_allow_q"].eq(QUARTILE_TOP).astype(float)
    panel.loc[panel["press_allow_q"].eq(QUARTILE_UNASSIGNED), "allow_top_quartile"] = np.nan
    return table.reset_index(), panel


def interim_flags(
    schedules: pd.DataFrame, coaches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First regular-season game under a mid-season head-coach replacement.

    A team-season carries a mid-season change when a coach's CFBD hire date
    falls strictly between that team's first and last regular-season kickoff
    of that season; the flagged game is that team's first regular-season game
    kicking off strictly after the hire date.
    """

    long_df = _team_long(schedules)
    long_df["kickoff"] = pd.to_datetime(long_df["kickoff"], utc=True)
    bounds = long_df.groupby(["team_id", "season"])["kickoff"].agg(["min", "max"])

    work = coaches.copy()
    work["hire_date"] = pd.to_datetime(work["hire_date"], errors="coerce", utc=True)
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce").astype("Int64")
    work["season"] = pd.to_numeric(work["season"], errors="coerce").astype(int)
    work = work.dropna(subset=["hire_date", "team_id"])

    flagged_games: list[dict[str, Any]] = []
    for row in work.itertuples():
        key = (int(row.team_id), int(row.season))
        if key not in bounds.index:
            continue
        first_kick, last_kick = bounds.loc[key, "min"], bounds.loc[key, "max"]
        if not (first_kick < row.hire_date < last_kick):
            continue
        candidates = long_df.loc[
            (long_df["team_id"] == key[0])
            & (long_df["season"] == key[1])
            & (long_df["kickoff"] > row.hire_date)
        ].sort_values("kickoff")
        if candidates.empty:
            continue
        first = candidates.iloc[0]
        flagged_games.append(
            {
                "game_id": int(first["game_id"]),
                "team_id": key[0],
                "season": key[1],
                "side": first["side"],
                "coach": row.coach,
                "hire_date": row.hire_date.isoformat(),
            }
        )
    events = pd.DataFrame(flagged_games)
    table = pd.DataFrame({"game_id": schedules["game_id"].to_numpy()}).set_index("game_id")
    table["home_interim_first"] = False
    table["away_interim_first"] = False
    if not events.empty:
        for side in ("home", "away"):
            ids = events.loc[events["side"].eq(side), "game_id"].astype("int64").unique()
            table.loc[table.index.isin(ids), f"{side}_interim_first"] = True
    return table.reset_index(), events


def interim_flags_gamecount(
    schedules: pd.DataFrame, coaches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same construct addressed by the PREDECESSOR'S OWN GAME COUNT, not the hire date.

    Correction of a date-resolution defect measured after the predeclared cell
    was scored: CFBD's hire date carries no time of day, so a game kicking at
    00:30 UTC on the hire date -- an evening game played the night BEFORE, and
    usually the game that got the previous coach fired -- is misattributed to
    the successor. Measured on the 81 predeclared events, 13 flag a game
    kicking within 12 hours of the hire date. This variant instead orders the
    team-season's coaches by hire date and starts the successor at game
    ``1 + sum(games of every preceding coach)``, chronologically. A fired
    coach never coaches the bowl, so his CFBD ``games`` count is exactly his
    regular-season games and the index is well defined.
    """

    long_df = _team_long(schedules)
    long_df["kickoff"] = pd.to_datetime(long_df["kickoff"], utc=True)
    bounds = long_df.groupby(["team_id", "season"])["kickoff"].agg(["min", "max"])

    work = coaches.copy()
    work["hire_date"] = pd.to_datetime(work["hire_date"], errors="coerce", utc=True)
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce").astype("Int64")
    work["season"] = pd.to_numeric(work["season"], errors="coerce").astype(int)
    work["games"] = pd.to_numeric(work["games"], errors="coerce")
    work = work.dropna(subset=["team_id"])

    flagged_games: list[dict[str, Any]] = []
    for (team_id, season), group in work.groupby(["team_id", "season"], sort=True):
        key = (int(team_id), int(season))
        if key not in bounds.index or len(group) < 2:
            continue
        first_kick, last_kick = bounds.loc[key, "min"], bounds.loc[key, "max"]
        ordered = group.sort_values("hire_date", na_position="first")
        team_games = (
            long_df.loc[(long_df["team_id"] == key[0]) & (long_df["season"] == key[1])]
            .sort_values("kickoff")
            .reset_index(drop=True)
        )
        cumulative = 0
        for row in ordered.itertuples():
            mid_season = (
                cumulative > 0
                and pd.notna(row.hire_date)
                and first_kick < row.hire_date < last_kick
                and cumulative < len(team_games)
            )
            if mid_season:
                first = team_games.iloc[cumulative]
                flagged_games.append(
                    {
                        "game_id": int(first["game_id"]),
                        "team_id": key[0],
                        "season": key[1],
                        "side": first["side"],
                        "coach": row.coach,
                        "hire_date": row.hire_date.isoformat(),
                        "game_index": cumulative + 1,
                    }
                )
            cumulative += int(row.games) if pd.notna(row.games) else 0

    events = pd.DataFrame(flagged_games)
    table = pd.DataFrame({"game_id": schedules["game_id"].to_numpy()}).set_index("game_id")
    table["home_interim_first"] = False
    table["away_interim_first"] = False
    if not events.empty:
        for side in ("home", "away"):
            ids = events.loc[events["side"].eq(side), "game_id"].astype("int64").unique()
            table.loc[table.index.isin(ids), f"{side}_interim_first"] = True
    return table.reset_index(), events


def _record_timeline(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per team-game wins/losses and games remaining from STRICTLY PRIOR WEEKS."""

    long_df = _team_long(schedules).copy()
    long_df["week"] = long_df["week"].fillna(0).astype(int)
    long_df["won"] = (long_df["points_for"] > long_df["points_against"]).astype(float)
    long_df["lost"] = (long_df["points_for"] < long_df["points_against"]).astype(float)
    long_df.loc[
        long_df["points_for"].isna() | long_df["points_against"].isna(), ["won", "lost"]
    ] = np.nan
    long_df = long_df.sort_values(["season", "team_id", "week", "gameday"]).reset_index(drop=True)

    per_week = (
        long_df.groupby(["season", "team_id", "week"], sort=True)
        .agg(wins=("won", "sum"), losses=("lost", "sum"), games=("won", "size"))
        .reset_index()
    )
    per_week["prior_wins"] = (
        per_week.groupby(["season", "team_id"])["wins"].cumsum() - per_week["wins"]
    )
    per_week["prior_losses"] = (
        per_week.groupby(["season", "team_id"])["losses"].cumsum() - per_week["losses"]
    )
    totals = per_week.groupby(["season", "team_id"])["games"].transform("sum")
    per_week["played_before"] = (
        per_week.groupby(["season", "team_id"])["games"].cumsum() - per_week["games"]
    )
    per_week["games_remaining"] = totals - per_week["played_before"]
    return long_df.merge(
        per_week[["season", "team_id", "week", "prior_wins", "prior_losses", "games_remaining"]],
        on=["season", "team_id", "week"],
        how="left",
    )


def motivation_flags(
    schedules: pd.DataFrame, fbs_team_ids: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Tank-zone (literal bottom two) and bowl-dead flags for the final five weeks."""

    timeline = _record_timeline(schedules)
    timeline = timeline.loc[timeline["team_id"].isin(fbs_team_ids)].copy()
    last_week = timeline.groupby("season")["week"].transform("max")
    timeline["in_late_window"] = timeline["week"] > (last_week - LAST_WEEKS)

    tank_rows: list[tuple[int, int, int, bool]] = []
    tie_counts: list[int] = []
    for (season, week), group in timeline.groupby(["season", "week"], sort=True):
        standings = (
            group.drop_duplicates(subset="team_id")
            .loc[:, ["team_id", "prior_wins", "prior_losses"]]
            .dropna()
        )
        if standings.empty:
            continue
        ordered = standings.sort_values(
            ["prior_wins", "prior_losses", "team_id"], ascending=[True, False, True]
        )
        zone = ordered.head(TANK_ZONE_SIZE)
        if len(ordered) > TANK_ZONE_SIZE:
            cut = ordered.iloc[TANK_ZONE_SIZE - 1]
            tied = int(
                (
                    (ordered["prior_wins"] == cut["prior_wins"])
                    & (ordered["prior_losses"] == cut["prior_losses"])
                ).sum()
            )
            tie_counts.append(tied)
        zone_ids = set(zone["team_id"].astype(int))
        for team_id in group["team_id"].astype(int).unique():
            tank_rows.append((int(season), int(week), int(team_id), team_id in zone_ids))
    tank = pd.DataFrame(tank_rows, columns=["season", "week", "team_id", "tank_zone"])

    timeline = timeline.merge(tank, on=["season", "week", "team_id"], how="left")
    timeline["tank_zone"] = (
        timeline["tank_zone"].fillna(False).astype(bool) & timeline["in_late_window"]
    )
    timeline["bowl_dead"] = (
        (timeline["prior_wins"] + timeline["games_remaining"]) < BOWL_ELIGIBLE_WINS
    ).fillna(False) & timeline["in_late_window"]

    wide = timeline.pivot_table(
        index="game_id", columns="side", values=["tank_zone", "bowl_dead"], aggfunc="first"
    )
    wide.columns = [f"{side}_{name}" for name, side in wide.columns]
    for column in ("home_tank_zone", "away_tank_zone", "home_bowl_dead", "away_bowl_dead"):
        if column not in wide.columns:
            wide[column] = False
        wide[column] = wide[column].fillna(False).astype(bool)
    panel = timeline.loc[:, ["team_id", "season", "week", "tank_zone", "bowl_dead"]].copy()
    panel["tank_zone"] = panel["tank_zone"].astype(float)
    panel["bowl_dead"] = panel["bowl_dead"].astype(float)
    mean_tied = float(np.mean(tie_counts)) if tie_counts else float("nan")
    return (
        wide.reset_index()[
            ["game_id", "home_tank_zone", "away_tank_zone", "home_bowl_dead", "away_bowl_dead"]
        ],
        panel,
        mean_tied,
    )


def apply_fade(
    frame: pd.DataFrame, home_flag: str, away_flag: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Exactly one side flagged; flip the pick OFF the flagged side."""

    only_home = frame[home_flag] & ~frame[away_flag]
    only_away = frame[away_flag] & ~frame[home_flag]
    eligible = only_home | only_away
    on_flagged = (only_home & frame["baseline_pick_home"]) | (
        only_away & ~frame["baseline_pick_home"]
    )
    candidate = frame["baseline_pick_home"].copy()
    candidate[on_flagged] = ~candidate[on_flagged]
    return candidate, eligible, only_home


def apply_back(
    frame: pd.DataFrame, home_flag: str, away_flag: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Exactly one side flagged; flip the pick ONTO the flagged side."""

    only_home = frame[home_flag] & ~frame[away_flag]
    only_away = frame[away_flag] & ~frame[home_flag]
    eligible = only_home | only_away
    off_flagged = (only_home & ~frame["baseline_pick_home"]) | (
        only_away & frame["baseline_pick_home"]
    )
    candidate = frame["baseline_pick_home"].copy()
    candidate[off_flagged] = ~candidate[off_flagged]
    return candidate, eligible, only_home


def apply_back_side(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Back the side named by ``back_side``, flipping only when the pick is off it."""

    eligible = frame["back_side"].isin(("HOME", "AWAY"))
    want_home = frame["back_side"].eq("HOME")
    candidate = frame["baseline_pick_home"].copy()
    mismatch = eligible & (want_home != frame["baseline_pick_home"])
    candidate[mismatch] = ~candidate[mismatch]
    return candidate, eligible, want_home


def subject_side_cover(
    frame: pd.DataFrame, eligible: pd.Series, subject_home: pd.Series
) -> dict[str, Any]:
    """Model-free read: how often the cell's subject side covers, against 50%.

    The subject is the side the NFL rule names -- the side a fade cell moves
    AWAY from, the side a back cell moves ONTO. Under no effect this is 0.5 at
    either grade; the NFL fade cells claim below, the back cells above.
    """

    out: dict[str, Any] = {}
    for grade in GRADES:
        settle = frame[f"settle_{grade}"]
        mask = eligible & settle.notna() & settle.ne(0.0)
        if not bool(mask.any()):
            out[grade] = None
            continue
        covered = settle.gt(0.0).eq(subject_home)
        out[grade] = {
            "subject_side_cover_rate": float(covered.loc[mask].mean()),
            "n_graded_eligible": int(mask.sum()),
        }
    return out


def perfect_foresight(frame: pd.DataFrame, eligible: pd.Series, settle: pd.Series) -> pd.Series:
    """Positive control: on the eligible set only, take the realised winning side."""

    candidate = frame["baseline_pick_home"].copy()
    truth = settle.gt(0.0)
    candidate[eligible] = truth[eligible]
    return candidate


def _metric(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["baseline_correct", "candidate_correct"])
    if valid.empty:
        return {
            "delta_accuracy": float("nan"),
            "candidate_accuracy": float("nan"),
            "baseline_accuracy": float("nan"),
        }
    return {
        "delta_accuracy": float((valid["candidate_correct"] - valid["baseline_correct"]).mean()),
        "candidate_accuracy": float(valid["candidate_correct"].mean()),
        "baseline_accuracy": float(valid["baseline_correct"].mean()),
    }


def summarize(
    graded: pd.DataFrame, *, samples: int, seed: int, season_blocked: bool
) -> dict[str, Any] | None:
    valid = graded.dropna(subset=["baseline_correct", "candidate_correct"])
    if valid.empty:
        return None
    point = _metric(valid)
    week = week_blocked_bootstrap(valid, _metric, block="week", samples=samples, seed=seed)
    row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    summary: dict[str, Any] = {
        "delta_accuracy_points": point["delta_accuracy"] * 100.0,
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["baseline_accuracy"],
        "week_blocked_ci95_points": [float(row["lower"]) * 100.0, float(row["upper"]) * 100.0],
        "week_blocked_probability_positive": float(row["probability_positive"]),
        "n_games": len(valid),
        "n_weeks": int(valid[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(valid["season"].nunique()),
        "n_eligible": int(valid["eligible"].sum()),
        "n_flips": int((valid["candidate_pick_home"] != valid["baseline_pick_home"]).sum()),
    }
    if season_blocked and valid["season"].nunique() >= 2:
        season = week_blocked_bootstrap(valid, _metric, block="season", samples=samples, seed=seed)
        srow = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
        summary["season_blocked_ci95_points"] = [
            float(srow["lower"]) * 100.0,
            float(srow["upper"]) * 100.0,
        ]
        summary["season_blocked_probability_positive"] = float(srow["probability_positive"])
    return summary


def grade_arm(
    frame: pd.DataFrame, candidate: pd.Series, eligible: pd.Series, settle: pd.Series
) -> pd.DataFrame:
    graded = frame.loc[:, ["season", "week", "game_id", "baseline_pick_home"]].copy()
    graded["candidate_pick_home"] = candidate.to_numpy()
    graded["eligible"] = eligible.to_numpy()
    graded["baseline_correct"] = pick_correct(frame["baseline_pick_home"], settle).to_numpy()
    graded["candidate_correct"] = pick_correct(candidate, settle).to_numpy()
    graded.loc[settle.isna().to_numpy(), ["baseline_correct", "candidate_correct"]] = np.nan
    return graded


def score_cell(
    frame: pd.DataFrame,
    candidate: pd.Series,
    eligible: pd.Series,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for grade in GRADES:
        settle = frame[f"settle_{grade}"]
        graded = grade_arm(frame, candidate, eligible, settle)
        control_pick = perfect_foresight(frame, eligible, settle)
        control = grade_arm(frame, control_pick, eligible, settle)
        block: dict[str, Any] = {
            "pooled": summarize(graded, samples=samples, seed=seed, season_blocked=True),
            "positive_control_pooled": summarize(
                control, samples=samples, seed=seed, season_blocked=True
            ),
            "eras": {},
            "positive_control_eras": {},
        }
        for label, start, end in ERAS:
            era = graded.loc[graded["season"].between(start, end)]
            block["eras"][label] = (
                summarize(era, samples=samples, seed=seed, season_blocked=False)
                if not era.empty
                else None
            )
            era_control = control.loc[control["season"].between(start, end)]
            block["positive_control_eras"][label] = (
                summarize(era_control, samples=samples, seed=seed, season_blocked=False)
                if not era_control.empty
                else None
            )
        out[grade] = block
    return out


def flag_panel_reliability(panel: pd.DataFrame, metric: str, seed: int) -> dict[str, Any]:
    frame = panel.copy()
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["week"])
    frame["week"] = frame["week"].astype(int)
    return split_half_reliability(frame, metric, seed=seed)


def _print_summary(label: str, summary: dict[str, Any] | None) -> None:
    if summary is None:
        print(f"  {label}: no scored games")
        return
    low, high = summary["week_blocked_ci95_points"]
    print(
        f"  {label}: delta {summary['delta_accuracy_points']:+.4f} pts  "
        f"P+ {summary['week_blocked_probability_positive']:.4f}  "
        f"week 95% [{low:+.4f}, {high:+.4f}]  n={summary['n_games']}, "
        f"eligible={summary['n_eligible']}, flips={summary['n_flips']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-protection", action="store_true")
    parser.add_argument(
        "--interim-addressing", choices=("hire_date", "gamecount"), default="hire_date"
    )
    args = parser.parse_args()

    started = time.time()
    print("=== loading population ===", flush=True)
    population = load_population()
    print(f"clean-core benchmark rows: {len(population)}", flush=True)

    schedules = load_cfb_schedules()
    schedules = schedules.loc[schedules["season"].astype(int).ge(HISTORY_START_SEASON)].copy()
    print(f"regular-season schedule rows {HISTORY_START_SEASON}+: {len(schedules)}", flush=True)

    fbs_team_ids = set(
        pd.to_numeric(population["home_id"], errors="coerce").dropna().astype(int)
    ) | set(pd.to_numeric(population["away_id"], errors="coerce").dropna().astype(int))
    print(f"FBS team ids in the benchmark population: {len(fbs_team_ids)}", flush=True)

    results: dict[str, Any] = {}
    coverage: dict[str, Any] = {}

    print("=== bye-edge fade ===", flush=True)
    bye = bye_edge_flags(schedules)
    frame = population.merge(bye, on="game_id", how="left")
    for column in ("home_off_bye", "away_off_bye"):
        frame[column] = frame[column].fillna(False).astype(bool)
    candidate, eligible, subject_home = apply_fade(frame, "home_off_bye", "away_off_bye")
    coverage["bye_edge_fade"] = {
        "n_home_off_bye": int(frame["home_off_bye"].sum()),
        "n_away_off_bye": int(frame["away_off_bye"].sum()),
        "n_exactly_one": int(eligible.sum()),
        "n_both": int((frame["home_off_bye"] & frame["away_off_bye"]).sum()),
    }
    print(f"  coverage: {coverage['bye_edge_fade']}", flush=True)
    bye_panel = _team_long(schedules).merge(
        bye.melt(id_vars="game_id", var_name="side_col", value_name="off_bye").assign(
            side=lambda d: d["side_col"].str.split("_").str[0]
        )[["game_id", "side", "off_bye"]],
        on=["game_id", "side"],
        how="left",
    )
    bye_panel["off_bye_metric"] = bye_panel["off_bye"].astype(float)
    results["bye_edge_fade"] = {
        "role": "faithful_port",
        "direction": "fade the strict-bye-holding side",
        "coverage": coverage["bye_edge_fade"],
        "subject_side_cover": subject_side_cover(frame, eligible, subject_home),
        "reliability": flag_panel_reliability(
            bye_panel.loc[bye_panel["season"].isin(CFB_CLEAN_CORE_SEASONS)],
            "off_bye_metric",
            args.seed,
        ),
        "grades": score_cell(
            frame, candidate, eligible, samples=args.bootstrap_samples, seed=args.seed
        ),
    }
    for grade in GRADES:
        print(f" [{grade}]", flush=True)
        _print_summary("pooled", results["bye_edge_fade"]["grades"][grade]["pooled"])
        _print_summary(
            "control", results["bye_edge_fade"]["grades"][grade]["positive_control_pooled"]
        )

    print("=== interim head coach, first game ===", flush=True)
    coaches = pd.read_parquet(COACHES_PATH)
    interim, events = (
        interim_flags_gamecount(schedules, coaches)
        if args.interim_addressing == "gamecount"
        else interim_flags(schedules, coaches)
    )
    frame = population.merge(interim, on="game_id", how="left")
    for column in ("home_interim_first", "away_interim_first"):
        frame[column] = frame[column].fillna(False).astype(bool)
    candidate, eligible, subject_home = apply_back(
        frame, "home_interim_first", "away_interim_first"
    )
    coverage["interim_hc_first_game"] = {
        "n_change_events_all_seasons": len(events),
        "n_flagged_in_population": int(
            (frame["home_interim_first"] | frame["away_interim_first"]).sum()
        ),
        "n_exactly_one": int(eligible.sum()),
    }
    print(f"  coverage: {coverage['interim_hc_first_game']}", flush=True)
    interim_panel = _team_long(schedules).merge(
        interim.melt(id_vars="game_id", var_name="side_col", value_name="interim").assign(
            side=lambda d: d["side_col"].str.split("_").str[0]
        )[["game_id", "side", "interim"]],
        on=["game_id", "side"],
        how="left",
    )
    interim_panel["interim_metric"] = interim_panel["interim"].fillna(False).astype(float)
    results["interim_hc_first_game"] = {
        "role": "faithful_port_with_disclosed_label_assumption",
        "direction": "back the team in its first game under a mid-season replacement",
        "coverage": coverage["interim_hc_first_game"],
        "subject_side_cover": subject_side_cover(frame, eligible, subject_home),
        "reliability": flag_panel_reliability(
            interim_panel.loc[interim_panel["season"].isin(CFB_CLEAN_CORE_SEASONS)],
            "interim_metric",
            args.seed,
        ),
        "grades": score_cell(
            frame, candidate, eligible, samples=args.bootstrap_samples, seed=args.seed
        ),
    }
    for grade in GRADES:
        print(f" [{grade}]", flush=True)
        _print_summary("pooled", results["interim_hc_first_game"]["grades"][grade]["pooled"])
        _print_summary(
            "control",
            results["interim_hc_first_game"]["grades"][grade]["positive_control_pooled"],
        )

    print("=== motivation cells ===", flush=True)
    motivation, motivation_panel, mean_tied = motivation_flags(schedules, fbs_team_ids)
    frame = population.merge(motivation, on="game_id", how="left")
    for column in ("home_tank_zone", "away_tank_zone", "home_bowl_dead", "away_bowl_dead"):
        frame[column] = frame[column].fillna(False).astype(bool)
    for name, home_flag, away_flag, role in (
        (
            "tank_zone_literal",
            "home_tank_zone",
            "away_tank_zone",
            "mechanical_flag_port_falsification_probe_no_cfb_draft_incentive",
        ),
        (
            "bowl_dead_analogue",
            "home_bowl_dead",
            "away_bowl_dead",
            "declared_analogue_different_mechanism_not_a_replication",
        ),
    ):
        candidate, eligible, subject_home = apply_fade(frame, home_flag, away_flag)
        coverage[name] = {
            "n_home_flagged": int(frame[home_flag].sum()),
            "n_away_flagged": int(frame[away_flag].sum()),
            "n_exactly_one": int(eligible.sum()),
            "n_both": int((frame[home_flag] & frame[away_flag]).sum()),
            "mean_teams_tied_at_cut": mean_tied if name == "tank_zone_literal" else None,
        }
        print(f"  {name} coverage: {coverage[name]}", flush=True)
        metric = "tank_zone" if name == "tank_zone_literal" else "bowl_dead"
        results[name] = {
            "role": role,
            "direction": "fade the flagged side",
            "coverage": coverage[name],
            "subject_side_cover": subject_side_cover(frame, eligible, subject_home),
            "reliability": flag_panel_reliability(
                motivation_panel.loc[motivation_panel["season"].isin(CFB_CLEAN_CORE_SEASONS)],
                metric,
                args.seed,
            ),
            "grades": score_cell(
                frame, candidate, eligible, samples=args.bootstrap_samples, seed=args.seed
            ),
        }
        for grade in GRADES:
            print(f" [{grade}]", flush=True)
            _print_summary("pooled", results[name]["grades"][grade]["pooled"])
            _print_summary("control", results[name]["grades"][grade]["positive_control_pooled"])

    if not args.skip_protection:
        print("=== protection mismatch (sack-only proxy) ===", flush=True)
        seasons = list(range(HISTORY_START_SEASON, 2026))
        pbp = load_cfb_pbp(seasons)
        print(f"  pbp rows: {len(pbp)}", flush=True)
        traits = build_sack_traits(pbp)
        del pbp
        print(f"  team-game trait rows: {len(traits)}", flush=True)
        protection, protection_panel = protection_flags(schedules, traits)
        frame = population.merge(protection, on="game_id", how="left")
        frame["back_side"] = frame["back_side"].fillna("")
        for column in ("home_offense_flagged", "away_offense_flagged"):
            frame[column] = frame[column].fillna(False).astype(bool)
        candidate, eligible, subject_home = apply_back_side(frame)
        coverage["protection_mismatch_sackonly"] = {
            "n_home_offense_flagged": int(frame["home_offense_flagged"].sum()),
            "n_away_offense_flagged": int(frame["away_offense_flagged"].sum()),
            "n_exactly_one": int(eligible.sum()),
            "n_both": int((frame["home_offense_flagged"] & frame["away_offense_flagged"]).sum()),
        }
        print(f"  coverage: {coverage['protection_mismatch_sackonly']}", flush=True)
        results["protection_mismatch_sackonly"] = {
            "role": "degraded_port_pressure_is_sack_only_no_qb_hit_field_in_cfb_pbp",
            "direction": "back the defence in a top-quartile allow x top-quartile generate matchup",
            "coverage": coverage["protection_mismatch_sackonly"],
            "subject_side_cover": subject_side_cover(frame, eligible, subject_home),
            "reliability": flag_panel_reliability(
                protection_panel.loc[protection_panel["season"].isin(CFB_CLEAN_CORE_SEASONS)],
                "allow_top_quartile",
                args.seed,
            ),
            "grades": score_cell(
                frame, candidate, eligible, samples=args.bootstrap_samples, seed=args.seed
            ),
        }
        for grade in GRADES:
            print(f" [{grade}]", flush=True)
            _print_summary(
                "pooled", results["protection_mismatch_sackonly"]["grades"][grade]["pooled"]
            )
            _print_summary(
                "control",
                results["protection_mismatch_sackonly"]["grades"][grade]["positive_control_pooled"],
            )

    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        "predeclaration": "docs/cfb_served_tilts.md",
        "interim_addressing": args.interim_addressing,
        "league": "cfb",
        "population": {
            "features": str(FEATURES_PATH),
            "benchmark_predictions": str(BENCHMARK_PATH),
            "arm": "market_residual ridge forced pick, home_cover_probability >= 0.5",
            "clean_core_seasons": list(CFB_CLEAN_CORE_SEASONS),
            "n_games": len(population),
            "baseline_home_pick_rate": float(population["baseline_pick_home"].mean()),
        },
        "grades": {
            "opener": "result - spread_open (primary, per AGENTS.md)",
            "close": "result - spread_line (median-book close proxy, secondary)",
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "eras": [list(era) for era in ERAS],
        "not_replicable": {
            "forecast_cold_visitor": (
                "no CFB temperature source: no local snapshot carries a weather field and "
                "CFBD /games/weather returned HTTP 401 (Patreon-gated) when probed this "
                "session. Coverage block, never a mechanism verdict."
            ),
            "forecast_precip_high_total": (
                "same weather block; total_line is available on 99.97% of clean-core rows "
                "but no precipitation source exists. Coverage block, never a mechanism verdict."
            ),
        },
        "results": results,
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = ARTIFACT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    write_stamped_artifact(
        json.loads(json.dumps(payload, sort_keys=True, default=float)), output_dir / "results.json"
    )
    print(f"wrote {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
