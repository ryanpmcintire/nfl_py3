from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import pick_correct
from nfl_ats.io import atomic_parquet

RESULT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "result",
    "actual_total",
)

GRADED_COLUMNS: tuple[str, ...] = (
    "ledger",
    "arm",
    "game_id",
    "season",
    "week",
    "pick_side",
    "decision_home_spread",
    "result",
    "settle_margin",
    "outcome",
    "recorded_at_utc",
    "passes_recorded",
    "graded_at_utc",
)

OUTCOME_WON = "won"
OUTCOME_LOST = "lost"
OUTCOME_PUSHED = "pushed"
OUTCOME_PENDING = "pending"

TOTALS_CLOSER = "closer"
TOTALS_FURTHER = "further"
TOTALS_TIED = "tied"

PICK_SIDES = ("HOME", "AWAY")


def results_artifact_path(artifacts_root: Path) -> Path:
    return artifacts_root / "settlement" / "game_results.parquet"


def graded_index_path(artifacts_root: Path) -> Path:
    return artifacts_root / "settlement" / "graded_decisions.parquet"


def normalise_results(schedules: pd.DataFrame) -> pd.DataFrame:

    frame = schedules.copy()
    for column in ("game_id", "season", "week", "away_team", "home_team"):
        if column not in frame.columns:
            frame[column] = pd.NA
    home = pd.to_numeric(_column(frame, "home_score"), errors="coerce")
    away = pd.to_numeric(_column(frame, "away_score"), errors="coerce")
    recorded = pd.to_numeric(_column(frame, "result"), errors="coerce")
    frame["home_score"] = home
    frame["away_score"] = away
    frame["result"] = recorded.where(recorded.notna(), home - away)
    frame["actual_total"] = home + away
    frame["game_id"] = frame["game_id"].astype(str)
    return frame.loc[:, list(RESULT_COLUMNS)].drop_duplicates("game_id").reset_index(drop=True)


def newest_local_schedules(data_root: Path) -> Path | None:
    hits = sorted((data_root / "raw").glob("*/schedules.parquet"))
    return hits[-1] if hits else None


def fetch_live_results(seasons: Sequence[int]) -> pd.DataFrame:

    import nflreadpy as nfl

    frame = nfl.load_schedules(seasons=[int(season) for season in seasons])
    converted = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    return normalise_results(pd.DataFrame(converted))


def seasons_in_scope(results: pd.DataFrame, *, start_season: int) -> list[int]:

    if results.empty:
        return []
    seasons = pd.to_numeric(results["season"], errors="coerce").dropna().astype(int)
    return sorted({int(season) for season in seasons if season >= start_season})


def load_results(
    data_root: Path,
    *,
    seasons: Sequence[int] = (),
    refresh: bool = True,
    results_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    if results_path is not None:
        return normalise_results(pd.read_parquet(results_path)), {
            "source": "file",
            "path": str(results_path),
        }
    local_path = newest_local_schedules(data_root)
    local = (
        normalise_results(pd.read_parquet(local_path))
        if local_path is not None
        else pd.DataFrame(columns=list(RESULT_COLUMNS))
    )
    fallback_reason = "live refresh not requested"
    if refresh and seasons:
        try:
            live = fetch_live_results(seasons)
        except Exception as error:
            fallback_reason = f"{type(error).__name__}: {error}"
        else:
            if not live.empty:
                return live, {
                    "source": "nflverse_schedules",
                    "seasons": [int(season) for season in seasons],
                }
            fallback_reason = f"nflverse returned no schedule rows for {list(seasons)}"
    elif refresh:
        fallback_reason = "no season in scope to refresh"
    if local_path is None:
        return local, {"source": "none", "reason": fallback_reason}
    return local, {
        "source": "local_snapshot",
        "path": str(local_path),
        "reason": fallback_reason,
    }


@dataclass(frozen=True)
class Arm:
    label: str
    pick_column: str
    line_column: str = "decision_home_spread"
    game_column: str = "game_id"
    split_by: str | None = None
    filter_column: str | None = None


@dataclass(frozen=True)
class TotalsArm:
    label: str
    total_column: str
    game_column: str = "game_id"


@dataclass(frozen=True)
class LedgerSpec:
    key: str
    relative_path: str
    arms: tuple[Arm, ...] = ()
    totals_arms: tuple[TotalsArm, ...] = ()
    order_column: str = "recorded_at_utc"
    deadline_columns: tuple[str, ...] = ("deadline", "kickoff")
    row_keys: tuple[str, ...] = ("game_id",)
    notes: str = ""

    def path(self, artifacts_root: Path) -> Path:
        return artifacts_root / self.relative_path

    def graded_path(self, artifacts_root: Path) -> Path:
        target = self.path(artifacts_root)
        return target.with_name(f"{target.stem}.graded.parquet")


LEDGERS: tuple[LedgerSpec, ...] = (
    LedgerSpec(
        key="paper_decisions",
        relative_path="clv_ledger/decisions.parquet",
        arms=(
            Arm("played", "pick_side"),
            Arm("raw_model", "model_pick_side"),
            Arm("pre_arrest_policy", "pre_arrest_pick_side"),
            Arm("former_policy", "former_policy_pick_side"),
            Arm("best_pick", "pick_side", filter_column="is_best_pick"),
        ),
        deadline_columns=("kickoff",),
        notes="the played card and the frozen arms recorded beside it",
    ),
    LedgerSpec(
        key="challenger_decisions",
        relative_path="prospective/challenger_decisions.parquet",
        arms=(Arm("{value}", "pick_side", split_by="challenger_id"),),
        deadline_columns=("kickoff",),
        row_keys=("game_id", "challenger_id"),
        notes="one arm per registered challenger",
    ),
    LedgerSpec(
        key="weak_stack_deadline_drag_paired",
        relative_path="prospective/weak_stack_deadline_drag_paired_decisions.parquet",
        arms=(
            Arm("{value}", "pick_side", split_by="challenger_id"),
            Arm("{value}__baseline", "baseline_pick_side", split_by="challenger_id"),
        ),
        deadline_columns=("kickoff",),
        row_keys=("game_id", "challenger_id"),
    ),
    LedgerSpec(
        key="weak_stack_expected_lineup_loss_paired",
        relative_path="prospective/weak_stack_expected_lineup_loss_paired_decisions.parquet",
        arms=(
            Arm("{value}", "pick_side", split_by="challenger_id"),
            Arm("{value}__baseline", "baseline_pick_side", split_by="challenger_id"),
        ),
        deadline_columns=("kickoff",),
        row_keys=("game_id", "challenger_id"),
    ),
    LedgerSpec(
        key="pick_revisions",
        relative_path="prospective/pick_revisions.parquet",
        arms=(
            Arm("refreshed", "new_pick_side"),
            Arm("before_refresh", "previous_pick_side"),
            Arm("model_only_off_arm", "model_only_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
        deadline_columns=("kickoff",),
        notes="the active model's own late-week revisions",
    ),
    LedgerSpec(
        key="late_week_move_follow_refresh",
        relative_path="prospective/late_week_move_follow_refresh_decisions.parquet",
        arms=(
            Arm("tuesday", "tuesday_pick_side"),
            Arm("movement_follow", "movement_would_be_pick_side"),
            Arm("equal_book_off_arm", "equal_would_be_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
    ),
    LedgerSpec(
        key="handle_follow_refresh",
        relative_path="prospective/handle_follow_refresh_decisions.parquet",
        arms=(
            Arm("tuesday", "tuesday_pick_side"),
            Arm("served", "served_pick_side"),
            Arm("off_arm", "off_arm_pick_side"),
            Arm("handle_follow", "handle_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
    ),
    LedgerSpec(
        key="consensus_movement_refresh",
        relative_path="prospective/consensus_movement_refresh_decisions.parquet",
        arms=(
            Arm("tuesday", "tuesday_pick_side"),
            Arm("served", "served_pick_side"),
            Arm("consensus_arm", "consensus_arm_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
    ),
    LedgerSpec(
        key="crew_tilt_refresh",
        relative_path="prospective/crew_tilt_refresh_decisions.parquet",
        arms=(
            Arm("played", "played_pick_side"),
            Arm("crew_tilt", "crew_would_be_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
    ),
    LedgerSpec(
        key="inactives_refresh",
        relative_path="prospective/inactives_refresh_decisions.parquet",
        arms=(
            Arm("tuesday", "tuesday_pick_side"),
            Arm("played", "played_pick_side"),
            Arm("inactives", "inactives_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
    ),
    LedgerSpec(
        key="injury_signal_refresh",
        relative_path="prospective/injury_signal_refresh_decisions.parquet",
        arms=(
            Arm("hold", "hold_pick_side"),
            Arm("injury_tilt", "injury_tilt_pick_side"),
            Arm("movement", "movement_pick_side"),
            Arm("played", "played_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
        deadline_columns=("kickoff",),
    ),
    LedgerSpec(
        key="nflcom_friday_refresh",
        relative_path="prospective/nflcom_friday_refresh_decisions.parquet",
        arms=(
            Arm("played", "played_pick_side"),
            Arm("nflcom_starters", "nflcom_would_be_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
        deadline_columns=("kickoff",),
    ),
    LedgerSpec(
        key="specialist_absence_fade_refresh",
        relative_path="prospective/specialist_absence_fade_refresh_decisions.parquet",
        arms=(
            Arm("played", "played_pick_side"),
            Arm("specialist_fade", "specialist_would_be_pick_side"),
        ),
        order_column="revision_recorded_at_utc",
    ),
    LedgerSpec(
        key="best_pick_refresh",
        relative_path="prospective/best_pick_refresh_decisions.parquet",
        arms=(
            Arm(
                "tuesday_nominee",
                "tuesday_pick_side",
                line_column="tuesday_decision_home_spread",
                game_column="tuesday_game_id",
            ),
            Arm(
                "sunday_nominee",
                "sunday_pick_side",
                line_column="sunday_decision_home_spread",
                game_column="sunday_game_id",
            ),
        ),
        order_column="paired_at_utc",
        deadline_columns=(),
        row_keys=(),
        notes="one Tuesday/Sunday nomination pair per week",
    ),
    LedgerSpec(
        key="tiebreaker_shade",
        relative_path="prospective/tiebreaker_shade_decisions.parquet",
        totals_arms=(
            TotalsArm("served_total", "served_total"),
            TotalsArm("shaded_total", "shaded_total"),
        ),
        deadline_columns=(),
        row_keys=(),
        notes="totals, graded on absolute error against the final combined score",
    ),
    LedgerSpec(
        key="totals_served_method",
        relative_path="prospective/totals_served_method_decisions.parquet",
        totals_arms=(
            TotalsArm("blend_k01", "served_total_blend_k01"),
            TotalsArm("joint_residual", "served_total_joint_residual"),
        ),
        deadline_columns=(),
        row_keys=(),
        notes="totals, graded on absolute error against the final combined score",
    ),
)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _deadline(frame: pd.DataFrame, spec: LedgerSpec) -> pd.Series:

    for column in spec.deadline_columns:
        if column in frame.columns:
            return _utc(frame[column])
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")


def latest_pre_deadline_rows(frame: pd.DataFrame, spec: LedgerSpec) -> pd.DataFrame:

    if frame.empty:
        return frame.assign(passes_recorded=pd.Series(dtype="int64"))
    working = frame.copy()
    working["_order"] = (
        _utc(working[spec.order_column])
        if spec.order_column in working.columns
        else pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns, UTC]")
    )
    deadline = _deadline(working, spec)
    playable = deadline.isna() | working["_order"].isna() | working["_order"].lt(deadline)
    working = working.loc[playable]
    keys = [key for key in spec.row_keys if key in working.columns]
    if working.empty or not keys:
        return working.drop(columns=["_order"]).assign(passes_recorded=1)
    working["passes_recorded"] = working.groupby(keys, sort=False)[keys[0]].transform("size")
    latest = (
        working.sort_values("_order", na_position="first")
        .groupby(keys, as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest.drop(columns=["_order"])


def _arm_rows(frame: pd.DataFrame, arm: Arm) -> list[tuple[str, pd.DataFrame]]:
    if arm.pick_column not in frame.columns or arm.game_column not in frame.columns:
        return []
    rows = frame
    if arm.filter_column is not None:
        if arm.filter_column not in rows.columns:
            return []
        rows = rows.loc[rows[arm.filter_column].fillna(False).astype(bool)]
    if arm.split_by is None:
        return [(arm.label, rows)]
    if arm.split_by not in rows.columns:
        return []
    values = rows[arm.split_by].astype(str)
    return [
        (arm.label.format(value=value), rows.loc[values.eq(value)]) for value in sorted(set(values))
    ]


def _graded_frame(
    rows: pd.DataFrame,
    *,
    spec: LedgerSpec,
    label: str,
    games: pd.Series,
    pick_side: np.ndarray,
    line: pd.Series,
    result: pd.Series,
    margin: pd.Series,
    outcome: np.ndarray,
) -> pd.DataFrame:
    order = (
        _utc(rows[spec.order_column])
        if spec.order_column in rows.columns
        else pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
    )
    passes = (
        pd.to_numeric(rows["passes_recorded"], errors="coerce")
        if "passes_recorded" in rows.columns
        else pd.Series(1, index=rows.index, dtype="int64")
    )
    return pd.DataFrame(
        {
            "ledger": spec.key,
            "arm": label,
            "game_id": games.to_numpy(),
            "season": pd.to_numeric(_column(rows, "season"), errors="coerce").to_numpy(),
            "week": pd.to_numeric(_column(rows, "week"), errors="coerce").to_numpy(),
            "pick_side": pick_side,
            "decision_home_spread": line.to_numpy(),
            "result": result.to_numpy(),
            "settle_margin": margin.to_numpy(),
            "outcome": outcome,
            "recorded_at_utc": order.to_numpy(),
            "passes_recorded": passes.to_numpy(),
        }
    )


def _graded_ats_arm(
    rows: pd.DataFrame, arm: Arm, label: str, results: pd.DataFrame, spec: LedgerSpec
) -> pd.DataFrame:
    picks = rows[arm.pick_column].astype("string").fillna("").str.upper()
    rows = rows.loc[picks.isin(PICK_SIDES)]
    if rows.empty:
        return pd.DataFrame()
    picks = picks.loc[rows.index]
    games = rows[arm.game_column].astype(str)
    lookup = results.set_index("game_id")["result"] if not results.empty else pd.Series(dtype=float)
    result = pd.to_numeric(games.map(lookup), errors="coerce")
    line = pd.to_numeric(_column(rows, arm.line_column), errors="coerce")
    margin = result - line
    correct = pick_correct(picks.eq("HOME"), margin)
    outcome = np.where(
        margin.isna(),
        OUTCOME_PENDING,
        np.where(
            margin.eq(0.0),
            OUTCOME_PUSHED,
            np.where(correct.eq(1.0), OUTCOME_WON, OUTCOME_LOST),
        ),
    )
    return _graded_frame(
        rows,
        spec=spec,
        label=label,
        games=games,
        pick_side=picks.to_numpy(),
        line=line,
        result=result,
        margin=margin,
        outcome=outcome,
    )


def _graded_totals(rows: pd.DataFrame, results: pd.DataFrame, spec: LedgerSpec) -> pd.DataFrame:
    lookup = (
        results.set_index("game_id")["actual_total"]
        if not results.empty
        else pd.Series(dtype=float)
    )
    errors: dict[str, pd.Series] = {}
    for arm in spec.totals_arms:
        if arm.total_column not in rows.columns or arm.game_column not in rows.columns:
            continue
        actual = pd.to_numeric(rows[arm.game_column].astype(str).map(lookup), errors="coerce")
        served = pd.to_numeric(rows[arm.total_column], errors="coerce")
        errors[arm.label] = (served - actual).abs()
    if not errors:
        return pd.DataFrame()
    table = pd.DataFrame(errors)
    complete = table.notna().all(axis=1)
    best = table.min(axis=1)
    tied = complete & table.nunique(axis=1).eq(1)
    frames: list[pd.DataFrame] = []
    for arm in spec.totals_arms:
        if arm.label not in errors:
            continue
        absolute = errors[arm.label]
        games = rows[arm.game_column].astype(str)
        actual = pd.to_numeric(games.map(lookup), errors="coerce")
        outcome = np.where(
            ~complete,
            OUTCOME_PENDING,
            np.where(tied, TOTALS_TIED, np.where(absolute.eq(best), TOTALS_CLOSER, TOTALS_FURTHER)),
        )
        frames.append(
            _graded_frame(
                rows,
                spec=spec,
                label=arm.label,
                games=games,
                pick_side=pd.to_numeric(rows[arm.total_column], errors="coerce")
                .astype("string")
                .to_numpy(),
                line=pd.Series(np.nan, index=rows.index, dtype=float),
                result=actual,
                margin=absolute,
                outcome=outcome,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def grade_ledger(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    spec: LedgerSpec,
    *,
    graded_at: datetime,
) -> pd.DataFrame:

    empty = pd.DataFrame(columns=list(GRADED_COLUMNS))
    if frame.empty:
        return empty
    latest = latest_pre_deadline_rows(frame, spec)
    if latest.empty:
        return empty
    frames: list[pd.DataFrame] = []
    for arm in spec.arms:
        for label, rows in _arm_rows(latest, arm):
            if rows.empty:
                continue
            frames.append(_graded_ats_arm(rows, arm, label, results, spec))
    if spec.totals_arms:
        frames.append(_graded_totals(latest, results, spec))
    frames = [candidate for candidate in frames if not candidate.empty]
    if not frames:
        return empty
    graded = pd.concat(frames, ignore_index=True)
    instant = pd.Timestamp(graded_at)
    graded["graded_at_utc"] = (
        instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")
    )
    return graded.loc[:, list(GRADED_COLUMNS)]


def arm_summary(graded: pd.DataFrame) -> list[dict[str, Any]]:

    if graded.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (ledger, arm), group in graded.groupby(["ledger", "arm"], sort=True):
        counts = group["outcome"].value_counts()
        won = int(counts.get(OUTCOME_WON, 0)) + int(counts.get(TOTALS_CLOSER, 0))
        lost = int(counts.get(OUTCOME_LOST, 0)) + int(counts.get(TOTALS_FURTHER, 0))
        pushed = int(counts.get(OUTCOME_PUSHED, 0)) + int(counts.get(TOTALS_TIED, 0))
        decided = won + lost
        rows.append(
            {
                "ledger": str(ledger),
                "arm": str(arm),
                "games": len(group),
                "won": won,
                "lost": lost,
                "pushed": pushed,
                "pending": int(counts.get(OUTCOME_PENDING, 0)),
                "accuracy": float(won) / decided if decided else float("nan"),
            }
        )
    return rows


def _scope(
    frame: pd.DataFrame, *, season: int | None, week: int | None, start_season: int
) -> pd.DataFrame:
    scoped = frame
    if "season" in scoped.columns:
        seasons = pd.to_numeric(scoped["season"], errors="coerce")
        scoped = scoped.loc[
            seasons.eq(float(season)) if season is not None else seasons.ge(float(start_season))
        ]
    if week is not None and "week" in scoped.columns:
        scoped = scoped.loc[pd.to_numeric(scoped["week"], errors="coerce").eq(float(week))]
    return scoped.reset_index(drop=True)


def settle_ledgers(
    artifacts_root: Path,
    results: pd.DataFrame,
    *,
    season: int | None = None,
    week: int | None = None,
    start_season: int = 2026,
    write: bool = True,
    graded_at: datetime | None = None,
    specs: Sequence[LedgerSpec] = LEDGERS,
    reader: Callable[[Path], pd.DataFrame] = pd.read_parquet,
) -> dict[str, Any]:

    instant = graded_at or datetime.now(UTC)
    ledgers: list[dict[str, Any]] = []
    graded_frames: list[pd.DataFrame] = []
    for spec in specs:
        path = spec.path(artifacts_root)
        entry: dict[str, Any] = {
            "ledger": spec.key,
            "path": str(path),
            "recorded_rows": 0,
            "rows_in_scope": 0,
            "graded_rows": 0,
            "notes": spec.notes,
        }
        if not path.is_file():
            entry["reason"] = "no ledger written yet"
            ledgers.append(entry)
            continue
        frame = reader(path)
        entry["recorded_rows"] = len(frame)
        scoped = _scope(frame, season=season, week=week, start_season=start_season)
        entry["rows_in_scope"] = len(scoped)
        graded = grade_ledger(scoped, results, spec, graded_at=instant)
        entry["graded_rows"] = len(graded)
        entry["arms"] = arm_summary(graded)
        if write and not graded.empty:
            atomic_parquet(graded, spec.graded_path(artifacts_root))
            entry["graded_path"] = str(spec.graded_path(artifacts_root))
        graded_frames.append(graded)
        ledgers.append(entry)
    populated = [frame for frame in graded_frames if not frame.empty]
    combined = (
        pd.concat(populated, ignore_index=True)
        if populated
        else pd.DataFrame(columns=list(GRADED_COLUMNS))
    )
    if write and not combined.empty:
        atomic_parquet(combined, graded_index_path(artifacts_root))
    return {
        "graded_at_utc": instant.isoformat(),
        "season": season,
        "week": week,
        "ledgers": ledgers,
        "arms": arm_summary(combined),
        "graded": combined,
    }


def render_arm_table(arms: Sequence[dict[str, Any]]) -> str:

    header = f"{'ledger':<38} {'arm':<46} {'won':>4} {'lost':>4} {'push':>4} {'pend':>5}"
    lines = [header, "-" * len(header)]
    for row in arms:
        lines.append(
            f"{row['ledger'][:38]:<38} {row['arm'][:46]:<46} "
            f"{row['won']:>4} {row['lost']:>4} {row['pushed']:>4} {row['pending']:>5}"
        )
    return "\n".join(lines)
