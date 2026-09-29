from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS, sharp_book_movement_features

POPULATION = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
QUOTES = Path("artifacts/sharp_book_weighted_movement/spread_quotes.parquet")
REPORT = Path("docs/lead69_results.md")
PREDICTIONS = Path("docs/lead69_predictions.md")
SEASONS = (2023, 2024, 2025)
SLOTS = ("thursday", "friday", "saturday", "sunday_early", "sunday_late", "other")
METRICS = ("accuracy", "log_loss", "brier", "line_move")
ARMS = ("base", "horizon", "slots", "model_only", "market")
RESAMPLES = np.asarray(list(itertools.product(range(3), repeat=3)))


class SourceGateError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceGateError(message)


def build_predictors(inventory: dict) -> pd.DataFrame:
    require(POPULATION.is_file(), f"Missing population: {POPULATION}")
    require(QUOTES.is_file(), f"Missing quotes: {QUOTES}")
    columns = ["game_id", "season", "week", "game_type", "model_probability", *FIT_FEATURES]
    frame = pd.read_parquet(POPULATION, columns=columns, use_threads=False)
    frame = frame.loc[frame.season.isin(SEASONS) & frame[MOVE_AVAILABLE_COLUMN].eq(1)].copy()
    inventory["starting_games"] = len(frame)
    inventory["starting_by_season"] = frame.groupby("season").size().to_dict()
    require(len(frame) == 799, "Frozen source does not reproduce the declared 799 games")
    require(not frame.game_id.duplicated().any(), "Duplicate population game IDs")
    require(frame.game_type.eq("REG").all(), "Non-regular-season population row")
    quotes = pd.read_parquet(
        QUOTES,
        filters=[
            ("decision_label", "==", "intraday_hourly"),
            ("archive_season", "in", list(SEASONS)),
            ("bookmaker_key", "in", list(LEADER_BOOKS)),
            ("market", "==", "spreads"),
        ],
        use_threads=False,
    )
    quotes = quotes.loc[quotes.nflverse_game_id.isin(frame.game_id)].copy()
    inventory["leader_capture_rows"] = len(quotes)
    require(not quotes.empty, "No historical leader captures for the population")
    times = [
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "snapshot_timestamp_utc",
        "commence_time_utc",
    ]
    for column in times:
        quotes[column] = pd.to_datetime(quotes[column], utc=True, errors="coerce")
    valid = (
        quotes[times].notna().all(axis=1)
        & quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc)
        & quotes.observed_at_utc.lt(quotes.commence_time_utc)
        & quotes.snapshot_timestamp_utc.lt(quotes.commence_time_utc)
        & np.isfinite(quotes.home_spread_line)
    )
    inventory["invalid_or_postkick_capture_rows"] = int((~valid).sum())
    quotes = quotes.loc[valid].copy()
    kickoff = (
        quotes.sort_values(["observed_at_utc", "snapshot_timestamp_utc"])
        .drop_duplicates("nflverse_game_id")[["nflverse_game_id", "commence_time_utc"]]
        .rename(columns={"nflverse_game_id": "game_id"})
    )
    frame = frame.merge(kickoff, on="game_id", how="left", validate="one_to_one")
    missing = frame.loc[frame.commence_time_utc.isna(), "game_id"].tolist()
    inventory["missing_kickoffs"] = missing
    require(not missing, "Games lack kickoff supported by pregame captures")
    frame["week_first_commence_utc"] = frame.groupby(
        ["season", "week"]
    ).commence_time_utc.transform("min")
    locks = {
        key: sunday_pick_lock(group.commence_time_utc)
        for key, group in frame.groupby(["season", "week"])
    }
    frame["sunday_lock"] = [locks[(row.season, row.week)] for row in frame.itertuples()]
    frame["deadline"] = [
        pick_deadline(row.commence_time_utc, row.sunday_lock) for row in frame.itertuples()
    ]
    local_sunday = frame.sunday_lock.dt.tz_convert("America/New_York").dt.tz_localize(None)
    freeze = local_sunday.dt.normalize() - pd.Timedelta(days=5) + pd.Timedelta(hours=12)
    frame["freeze"] = freeze.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    local = frame.commence_time_utc.dt.tz_convert("America/New_York")
    excluded = local.dt.weekday.eq(0) | (local.dt.weekday.eq(6) & local.dt.hour.ge(19))
    inventory["structural_snf_mnf_exclusions"] = int(excluded.sum())
    inventory["structural_exclusions_by_season"] = (
        frame.loc[excluded].groupby("season").size().to_dict()
    )
    frame["slot"] = np.select(
        [
            local.dt.weekday.eq(3),
            local.dt.weekday.eq(4),
            local.dt.weekday.eq(5),
            local.dt.weekday.eq(6) & local.dt.hour.lt(16),
            local.dt.weekday.eq(6) & local.dt.hour.ge(16),
        ],
        list(SLOTS[:-1]),
        default="other",
    )
    frame = frame.loc[~excluded].copy()
    frame["hours_to_deadline"] = (frame.deadline - frame.freeze).dt.total_seconds() / 3600
    inventory["eligible_games"] = len(frame)
    inventory["eligible_by_season"] = frame.groupby("season").size().to_dict()
    inventory["slot_counts"] = frame.groupby("slot").size().to_dict()
    inventory["hours_range"] = [
        float(frame.hours_to_deadline.min()),
        float(frame.hours_to_deadline.max()),
    ]
    require(frame.hours_to_deadline.gt(0).all(), "Nonpositive freeze-to-deadline horizon")
    require(set(frame.season) == set(SEASONS), "A declared season has no eligible games")
    quotes = quotes.merge(
        frame[["game_id", "deadline", "commence_time_utc"]].rename(
            columns={"commence_time_utc": "scheduled_kickoff"}
        ),
        left_on="nflverse_game_id",
        right_on="game_id",
        validate="many_to_one",
    )
    timely = quotes.observed_at_utc.lt(quotes.deadline) & quotes.snapshot_timestamp_utc.lt(
        quotes.deadline
    )
    inventory["postdeadline_capture_rows_removed"] = int((~timely).sum())
    quotes = quotes.loc[timely].copy()
    conflicts = (
        quotes.loc[quotes.commence_time_utc.ne(quotes.scheduled_kickoff), "game_id"]
        .unique()
        .tolist()
    )
    inventory["kickoff_revision_games"] = len(conflicts)
    inventory["kickoff_revision_examples"] = conflicts[:5]
    earliest = quotes.groupby("game_id").commence_time_utc.min()
    previous_deadline = frame.deadline.copy()
    frame["commence_time_utc"] = frame.game_id.map(earliest)
    require(frame.commence_time_utc.notna().all(), "Missing predeadline kickoff source")
    frame["deadline"] = [
        pick_deadline(row.commence_time_utc, row.sunday_lock) for row in frame.itertuples()
    ]
    frame["hours_to_deadline"] = (frame.deadline - frame.freeze).dt.total_seconds() / 3600
    require(frame.hours_to_deadline.gt(0).all(), "Nonpositive conservative horizon")
    inventory["maximum_deadline_contraction_minutes"] = float(
        (previous_deadline - frame.deadline).dt.total_seconds().max() / 60
    )
    inventory["hours_range"] = [
        float(frame.hours_to_deadline.min()),
        float(frame.hours_to_deadline.max()),
    ]
    conservative_local = frame.commence_time_utc.dt.tz_convert("America/New_York")
    require(
        conservative_local.dt.weekday.eq(local.loc[frame.index].dt.weekday).all(),
        "Conservative kickoff changes the structural game day",
    )
    slots = np.select(
        [
            conservative_local.dt.weekday.eq(3),
            conservative_local.dt.weekday.eq(4),
            conservative_local.dt.weekday.eq(5),
            conservative_local.dt.weekday.eq(6) & conservative_local.dt.hour.lt(16),
            conservative_local.dt.weekday.eq(6) & conservative_local.dt.hour.ge(16),
        ],
        list(SLOTS[:-1]),
        default="other",
    )
    inventory["slot_labels_updated_after_deadline_contraction"] = int(frame.slot.ne(slots).sum())
    frame["slot"] = slots
    inventory["slot_counts"] = frame.groupby("slot").size().to_dict()
    quotes["deadline"] = quotes.game_id.map(frame.set_index("game_id").deadline)
    timely = quotes.observed_at_utc.lt(quotes.deadline) & quotes.snapshot_timestamp_utc.lt(
        quotes.deadline
    )
    inventory["conservative_deadline_additional_removals"] = int((~timely).sum())
    quotes = quotes.loc[timely].copy()
    require(
        quotes.observed_at_utc.lt(quotes.deadline).all()
        and quotes.snapshot_timestamp_utc.lt(quotes.deadline).all()
        and quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc).all(),
        "Timestamp guard failed",
    )
    games = frame[["game_id", "commence_time_utc", "week_first_commence_utc"]].copy()
    games["cutoff_utc"] = frame.deadline
    rebuilt = sharp_book_movement_features(quotes, games, include_sunday=True)
    missing_moves = rebuilt.loc[rebuilt.leader_books.le(0), "game_id"].tolist()
    inventory["missing_safe_move_games"] = missing_moves
    require(not missing_moves, "Eligible games lack a deadline-safe leader movement source")
    frame["stored_move"] = frame[MOVE_COLUMN]
    mapping = rebuilt.set_index("game_id")
    frame[MOVE_COLUMN] = frame.game_id.map(mapping.leader_median_net_move)
    frame[MOVE_AVAILABLE_COLUMN] = frame.game_id.map(mapping.leader_books).gt(0).astype(float)
    inventory["rebuilt_move_differs_from_stored"] = int(
        (~np.isclose(frame[MOVE_COLUMN], frame.stored_move)).sum()
    )
    inventory["largest_move_rebuild_difference"] = float(
        (frame[MOVE_COLUMN] - frame.stored_move).abs().max()
    )
    frame["move_log_hours"] = frame[MOVE_COLUMN] * np.log(frame.hours_to_deadline)
    for slot in SLOTS:
        if slot != "sunday_early":
            frame[f"move_slot_{slot}"] = frame[MOVE_COLUMN] * frame.slot.eq(slot).astype(float)
    frame["latest_capture"] = frame.game_id.map(quotes.groupby("game_id").observed_at_utc.max())
    frame["latest_snapshot"] = frame.game_id.map(
        quotes.groupby("game_id").snapshot_timestamp_utc.max()
    )
    require(
        np.isfinite(frame[[*FIT_FEATURES, "model_probability", "move_log_hours"]]).all().all(),
        "Nonfinite inputs",
    )
    return frame.reset_index(drop=True)


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> tuple:
    means = train[features].mean().to_numpy(dtype=float)
    scales = train[features].std(ddof=0).to_numpy(dtype=float)
    scales = np.where(scales > 0, scales, 1.0)
    x = np.column_stack([np.ones(len(train)), (train[features].to_numpy() - means) / scales])
    beta = _fit_logit(x, train.home_covered.to_numpy(dtype=float), FIT_RIDGE)
    target_x = np.column_stack(
        [np.ones(len(target)), (target[features].to_numpy() - means) / scales]
    )
    p = 1 / (1 + np.exp(-np.clip(target_x @ beta, -35, 35)))
    coefficients = {"intercept": float(beta[0] - np.sum(beta[1:] * means / scales))}
    coefficients.update(dict(zip(features, (beta[1:] / scales).tolist(), strict=True)))
    return p, coefficients


def metric_arrays(frame: pd.DataFrame, p: np.ndarray) -> dict[str, np.ndarray]:
    y = frame.home_covered.to_numpy(dtype=float)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "accuracy": ((p >= 0.5) == y).astype(float),
        "log_loss": -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)),
        "brier": (p - y) ** 2,
        "line_move": np.where(p >= 0.5, 1.0, -1.0) * frame.open_move.to_numpy(dtype=float),
    }


def interval(frame: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    sums = np.asarray([values[frame.season.eq(s)].sum() for s in SEASONS])
    counts = np.asarray([frame.season.eq(s).sum() for s in SEASONS])
    boot = sums[RESAMPLES].sum(axis=1) / counts[RESAMPLES].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "effect": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "probability_positive": float(np.mean(boot > 0) + 0.5 * np.mean(boot == 0)),
    }


def fmt(summary: dict) -> str:
    return "{effect:.6f} [{low:.6f}, {high:.6f}]".format(**summary)


def table(headers: list[str], rows: list[list]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        "",
    ]


def evaluate(frame: pd.DataFrame, inventory: dict) -> None:
    outcomes = pd.read_parquet(
        POPULATION,
        columns=["game_id", "home_covered", "margin_vs_open", "open_move"],
        use_threads=False,
    )
    frame = frame.merge(outcomes, on="game_id", validate="one_to_one")
    require(frame.margin_vs_open.ne(0).all(), "Unexpected push")
    require(frame.home_covered.isin([0.0, 1.0]).all(), "Invalid opener target")
    require(
        frame.home_covered.eq(frame.margin_vs_open.gt(0).astype(float)).all(),
        "Target differs from margin",
    )
    require(np.isfinite(frame.open_move).all(), "Missing evaluation-only opener-to-close move")
    features = {
        "base": list(FIT_FEATURES),
        "horizon": [*FIT_FEATURES, "move_log_hours"],
        "slots": [*FIT_FEATURES, *(f"move_slot_{s}" for s in SLOTS if s != "sunday_early")],
    }
    coefficients = []
    for arm, terms in features.items():
        frame[f"{arm}_is"], full_coef = fit_predict(frame, frame, terms)
        coefficients.append({"arm": arm, "fold": "in_sample", "n_train": len(frame), **full_coef})
        frame[f"{arm}_oos"] = np.nan
        for season in SEASONS:
            train, held = frame.loc[frame.season.ne(season)], frame.loc[frame.season.eq(season)]
            p, fold_coef = fit_predict(train, held, terms)
            frame.loc[held.index, f"{arm}_oos"] = p
            coefficients.append({"arm": arm, "fold": season, "n_train": len(train), **fold_coef})
    for mode in ("is", "oos"):
        frame[f"model_only_{mode}"] = frame.model_probability
        frame[f"market_{mode}"] = 0.5
    metrics = {
        (mode, arm): metric_arrays(frame, frame[f"{arm}_{mode}"].to_numpy())
        for mode in ("is", "oos")
        for arm in ARMS
    }
    lines = [
        "# LEAD-69 unit 1",
        "",
        "**Measured:** final verification of the protocol saved before outcomes in",
        "`lead69_protocol.md` and the lane.",
        f"Source population: `{POPULATION.as_posix()}`; quote source: `{QUOTES.as_posix()}`.",
        "The source artifact was active when inspected; it is pinned for this experiment.",
        "",
        "## Decisive games first",
        "",
    ]
    decisive = []
    for mode in ("is", "oos"):
        for arm in ("horizon", "slots"):
            delta = metrics[(mode, arm)]["accuracy"] - metrics[(mode, "base")]["accuracy"]
            decisive.append(
                [mode, arm, int((delta > 0).sum()), int((delta < 0).sum()), int((delta != 0).sum())]
            )
    lines += table(["Mode", "Arm vs base", "Wins", "Losses", "Different picks"], decisive)
    lines += [
        "## Source inventory",
        "",
        "```json",
        json.dumps(inventory, indent=2),
        "```",
        "",
        "The starting 799 is the source population; SNF/MNF exclusions are structural,",
        "before outcomes.",
        "The comparator is the served four-term specification refitted on identical",
        "eligible rows and rebuilt",
        "deadline-safe moves. This is not a replay of the historical served card.",
        "The production move begins Wednesday; the horizon begins at the Tuesday-noon freeze.",
        "Snapshot and observation timestamps precede the deadline; book updates precede",
        "observation.",
        "",
        "## Aggregate performance and in-sample/out-of-sample gap",
        "",
        "Accuracy is a fraction; line movement is points. Intervals are exact season-",
        "cluster bootstrap 95%",
        "percentiles across all 27 ordered resamples. Only three seasons are available.",
        "",
    ]
    aggregate = []
    for arm in ARMS:
        for metric in METRICS:
            ins, oos = (interval(frame, metrics[(mode, arm)][metric]) for mode in ("is", "oos"))
            gap = ins["effect"] - oos["effect"]
            aggregate.append([arm, metric, fmt(ins), fmt(oos), f"{gap:.6f}"])
    lines += table(
        ["Arm", "Metric", "IS [95% interval]", "OOS [95% interval]", "IS minus OOS"], aggregate
    )
    lines += [
        "## Paired gains versus four-term base",
        "",
        "Positive means improvement for every metric.",
        "",
    ]
    gains, primary = [], {}
    for mode in ("is", "oos"):
        for arm in ("horizon", "slots"):
            for metric in METRICS:
                sign = -1 if metric in ("log_loss", "brier") else 1
                delta = sign * (metrics[(mode, arm)][metric] - metrics[(mode, "base")][metric])
                summary = interval(frame, delta)
                gains.append(
                    [
                        mode,
                        arm,
                        metric,
                        fmt(summary),
                        format(summary["probability_positive"], ".6f"),
                    ]
                )
                if mode == "oos" and arm == "horizon":
                    primary[metric] = summary
    lines += table(["Mode", "Arm", "Metric", "Gain [95% interval]", "probability_positive"], gains)
    lines += ["## Season stability", ""]
    rows = []
    for season in SEASONS:
        mask = frame.season.eq(season)
        for arm in ARMS:
            values = metrics[("oos", arm)]
            wins = int(values["accuracy"][mask].sum())
            rows.append(
                [
                    season,
                    arm,
                    int(mask.sum()),
                    f"{wins}-{int(mask.sum()) - wins}",
                    *(f"{values[m][mask].mean():.6f}" for m in METRICS),
                ]
            )
    lines += table(["Season", "Arm", "N", "Record", *METRICS], rows)
    lines += [
        "## Natural-unit coefficients",
        "",
        "No held-out row affects its fold means, scales, or coefficients.",
        "",
    ]
    for arm, terms in features.items():
        lines += [f"### {arm}", ""]
        rows = [
            [row["fold"], row["n_train"], *(f"{row[t]:.9f}" for t in ["intercept", *terms])]
            for row in coefficients
            if row["arm"] == arm
        ]
        lines += table(["Held season / IS", "Training N", "intercept", *terms], rows)
    lines += ["### Fold coefficient stability", ""]
    rows = []
    for arm, terms in features.items():
        for term in ["intercept", *terms]:
            values = np.asarray(
                [
                    row[term]
                    for row in coefficients
                    if row["arm"] == arm and row["fold"] != "in_sample"
                ]
            )
            rows.append(
                [
                    arm,
                    term,
                    f"{values.min():.9f}",
                    f"{values.max():.9f}",
                    int((values > 0).sum()),
                    int((values < 0).sum()),
                ]
            )
    lines += table(["Arm", "Term", "Minimum", "Maximum", "Positive folds", "Negative folds"], rows)
    lines += [
        "## OOS reliability",
        "",
        "Fixed equal-width home-probability bins; upper edge 1.0 is included.",
        "",
    ]
    rows, bins = [], np.linspace(0, 1, 6)
    for arm in ARMS:
        p = frame[f"{arm}_oos"].to_numpy()
        band = np.minimum(np.searchsorted(bins, p, side="right") - 1, 4)
        for index in range(5):
            mask = band == index
            observed = frame.home_covered.to_numpy()[mask]
            rows.append(
                [
                    arm,
                    f"{bins[index]:.1f}-{bins[index + 1]:.1f}",
                    int(mask.sum()),
                    f"{p[mask].mean():.6f}" if mask.any() else "NA",
                    f"{observed.mean():.6f}" if mask.any() else "NA",
                ]
            )
    lines += table(["Arm", "Bin", "N", "Mean forecast", "Observed home cover"], rows)
    lines += [
        "## Interpretation, looks, and limitations",
        "",
        "**Measured:** 2 predeclared research looks: one-parameter horizon and descriptive",
        "slot sensitivity.",
        "Diagnostic accounting: 5 probability arms (3 fitted, 2 fixed); 12 numerical fits",
        "(3 arms x 4 folds/IS);",
        "25 fixed calibration cells; 40 aggregate metric cells; 60 season-metric cells; 16",
        "paired gain cells;",
        "4 decisive-record cells. Disclosed looks are not independent confirmations or",
        "selection searches.",
        "**Inferred:** classification remains `unresolved_below_power`; a zero crossing",
        "cannot close a signal.",
        "Only three seasons and previously selected base features limit inference. LOSO is",
        "the declared season",
        "holdout design, not an untouched chronological outer test. The descriptive arm is",
        "not promotable here.",
        "Line-move grading shares market information with the fitted move term and is descriptive.",
        "Move availability is constant here: its coefficient is unidentifiable separately",
        "from the intercept;",
        "standardized ridge returns zero. No deployment or registry mutation occurred.",
        "Registry commands await the orchestrator. See `lead69_predictions.md` for every",
        "scored row.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    columns = [
        "game_id",
        "season",
        "week",
        "slot",
        "freeze",
        "deadline",
        "latest_capture",
        "latest_snapshot",
        "hours_to_deadline",
        "home_covered",
        "open_move",
        "model_logit",
        "composition_flag_sum",
        MOVE_COLUMN,
        MOVE_AVAILABLE_COLUMN,
        "move_log_hours",
        *(f"{arm}_{mode}" for arm in ARMS for mode in ("is", "oos")),
    ]
    rows = [
        [f"{value:.12g}" if isinstance(value, float) else str(value) for value in row]
        for row in frame.sort_values(["season", "week", "game_id"])[columns].itertuples(
            index=False, name=None
        )
    ]
    PREDICTIONS.write_text(
        "\n".join(
            [
                "# LEAD-69 prediction-level evidence",
                "",
                "**Measured:** pinned inputs, deadline checks, targets and all predictions",
                "from the single unit run.",
                "",
                *table(columns, rows),
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "unresolved_below_power",
                "starting_games": inventory["starting_games"],
                "eligible_games": len(frame),
                "decisive_oos": decisive[2],
                "primary_oos_gains": primary,
                "research_looks": 2,
                "report": REPORT.as_posix(),
                "predictions": PREDICTIONS.as_posix(),
            },
            indent=2,
        )
    )


def main() -> None:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    inventory = {"population": POPULATION.as_posix(), "quotes": QUOTES.as_posix()}
    try:
        frame = build_predictors(inventory)
    except (SourceGateError, FileNotFoundError, KeyError) as error:
        inventory["source_gate"] = str(error)
        REPORT.write_text(
            "\n".join(
                [
                    "# LEAD-69 source inventory",
                    "",
                    "**Measured:** source gate stopped this unit before outcomes were read",
                    "or fitted.",
                    "",
                    "```json",
                    json.dumps(inventory, indent=2),
                    "```",
                    "",
                    "No accuracy, interval, probability_positive, coefficient or decisive",
                    "record was computed.",
                    "Research looks executed: 0 of 2 declared. No empirical signal closure",
                    "and no registry command.",
                    "The declared experiment remains pending complete, consistent,",
                    "predeadline source coverage.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"status": "source_gated", **inventory, "report": REPORT.as_posix()}, indent=2
            )
        )
        return
    evaluate(frame, inventory)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
