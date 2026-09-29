from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sunday_market_probability_eval import sunday_move
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

QUOTES = Path("artifacts/sharp_book_weighted_movement/spread_quotes.parquet")
FIT_ROOT = Path("artifacts/pick_probability/20260929T192747Z")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
OUTPUT = Path("tests/scratch/codex/lead82_unit1")
REPORT = Path("docs/lead82_unit1.md")
LANE = Path("docs/lanes/lead82.md")
ZONE = "America/New_York"
KEYS = ["game_id", "bookmaker_key"]


def read_table(path: Path, columns: list[str], filters=None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, filters=filters, use_threads=False).to_pandas(
        use_threads=False
    )


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def load_population(opener: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["game_id", "season", "week", "tue_open_home_spread"]
    openers = read_table(opener, columns)
    openers = openers.loc[openers.season.between(2020, 2025)].copy()
    fitted = read_table(FIT_ROOT / "per_game.parquet", columns + [MOVE_COLUMN, MOVE_AVAILABLE_COLUMN])
    fitted = fitted.loc[fitted.season.between(2020, 2025)].copy()
    if openers.game_id.duplicated().any() or fitted.game_id.duplicated().any():
        raise ValueError("Duplicate frozen game identifiers")
    if not fitted.game_id.isin(openers.game_id).all():
        raise ValueError("Frozen fit games are absent from their opener source")
    schedule = read_table(
        SCHEDULE, ["game_id", "season", "week", "game_type", "gameday", "gametime"]
    )
    schedule = schedule.loc[schedule.season.between(2020, 2025) & schedule.game_type.eq("REG")].copy()
    if schedule.game_id.duplicated().any():
        raise ValueError("Duplicate schedule game identifiers")
    local = pd.to_datetime(schedule.gameday.astype(str) + " " + schedule.gametime.astype(str), errors="raise")
    schedule["kickoff"] = local.dt.tz_localize(ZONE, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    first = local.dt.normalize().groupby([schedule.season, schedule.week]).transform("min")
    sunday = first + pd.to_timedelta((6 - first.dt.weekday) % 7, unit="D")
    for name, offset in {
        "monday": pd.Timedelta(days=-6),
        "freeze": pd.Timedelta(days=-5, hours=12),
        "wednesday": pd.Timedelta(days=-4),
        "sunday_deadline": pd.Timedelta(hours=12, minutes=45),
    }.items():
        schedule[name] = (sunday + offset).dt.tz_localize(ZONE).dt.tz_convert("UTC")
    schedule["deadline"] = schedule[["kickoff", "sunday_deadline"]].min(axis=1)
    schedule["week_first_commence_utc"] = schedule.groupby(["season", "week"]).kickoff.transform("min")
    games = openers.merge(schedule, on=["game_id", "season", "week"], validate="one_to_one", how="left")
    if games.kickoff.isna().any() or not np.isfinite(games.tue_open_home_spread).all():
        raise ValueError("Frozen openers lack finite lines or regular-season schedule clocks")
    if not (
        games.monday.lt(games.freeze)
        & games.freeze.lt(games.wednesday)
        & games.wednesday.lt(games.deadline)
        & games.deadline.le(games.kickoff)
    ).all():
        raise ValueError("Game clocks do not support the declared Tuesday block")
    frozen = fitted[columns].merge(openers, on=["game_id", "season", "week"], validate="one_to_one")
    if not np.allclose(frozen.tue_open_home_spread_x, frozen.tue_open_home_spread_y, atol=0, rtol=0):
        raise ValueError("Frozen fit and opener source lines disagree")
    games["in_frozen_fit"] = games.game_id.isin(fitted.game_id)
    return games, fitted


def load_quotes(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = [
        "nflverse_game_id", "bookmaker_key", "observed_at_utc", "bookmaker_last_update_utc",
        "home_spread_line", "commence_time_utc", "market", "snapshot_timestamp_utc",
        "manifest_season", "archive_season", "decision_label",
    ]
    quotes = read_table(
        QUOTES, columns,
        filters=[("bookmaker_key", "in", list(LEADER_BOOKS)), ("nflverse_game_id", "in", games.game_id.tolist())],
    ).rename(columns={"nflverse_game_id": "game_id"})
    quotes = quotes.merge(
        games[["game_id", "season", "week", "kickoff", "monday", "freeze", "wednesday", "deadline"]],
        on="game_id", validate="many_to_one",
    )
    for name in ["observed_at_utc", "bookmaker_last_update_utc", "commence_time_utc", "snapshot_timestamp_utc"]:
        quotes[name] = pd.to_datetime(quotes[name], utc=True, errors="coerce")
    gates = {
        "spread_market": quotes.market.eq("spreads"),
        "finite_line": np.isfinite(quotes.home_spread_line),
        "source_season": quotes.manifest_season.eq(quotes.season) & quotes.archive_season.eq(quotes.season),
        "target_week": quotes.observed_at_utc.ge(quotes.monday) & quotes.snapshot_timestamp_utc.ge(quotes.monday),
        "book_before_observation": quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc),
        "observation_before_snapshot": quotes.observed_at_utc.le(quotes.snapshot_timestamp_utc),
        "before_decision": quotes.observed_at_utc.lt(quotes.deadline) & quotes.snapshot_timestamp_utc.lt(quotes.deadline),
        "before_source_kickoff": quotes.observed_at_utc.lt(quotes.commence_time_utc) & quotes.snapshot_timestamp_utc.lt(quotes.commence_time_utc),
    }
    eligible = pd.DataFrame(gates, index=quotes.index).all(axis=1)
    inventory = {
        "selected_rows": len(quotes), "quote_schedule_kickoff_differences": int(quotes.commence_time_utc.ne(quotes.kickoff).sum()),
        "rejected_rows": int((~eligible).sum()),
        "gate_failures_nonexclusive": {name: int((~valid).sum()) for name, valid in gates.items()},
    }
    quotes = quotes.loc[eligible].copy()
    keys = KEYS + ["observed_at_utc"]
    if quotes.groupby(keys).home_spread_line.nunique().gt(1).any():
        raise ValueError("Conflicting same-book lines at one observation time")
    quotes = quotes.sort_values(keys + ["snapshot_timestamp_utc", "bookmaker_last_update_utc"])
    quotes = quotes.drop_duplicates(keys + ["decision_label"], keep="first").reset_index(drop=True)
    inventory["unique_eligible_rows"] = len(quotes)
    return quotes, inventory


def boundary(quotes: pd.DataFrame, eligible: pd.Series, prefix: str) -> pd.DataFrame:
    columns = KEYS + ["home_spread_line", "observed_at_utc", "snapshot_timestamp_utc", "bookmaker_last_update_utc", "decision_label"]
    frame = quotes.loc[eligible, columns].groupby(KEYS, sort=False).tail(1)
    return frame.rename(columns={name: f"{prefix}_{name}" for name in columns if name not in KEYS})


def extract(games: pd.DataFrame, quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hourly = quotes.loc[quotes.decision_label.eq("intraday_hourly")].copy()
    anchors = boundary(
        hourly,
        hourly.observed_at_utc.le(hourly.freeze) & hourly.snapshot_timestamp_utc.le(hourly.freeze),
        "noon",
    )
    terminals = boundary(
        hourly,
        hourly.observed_at_utc.lt(hourly.wednesday) & hourly.snapshot_timestamp_utc.lt(hourly.wednesday),
        "tuesday_end",
    )
    latest = boundary(quotes, pd.Series(True, index=quotes.index), "decision")
    pairs = anchors.merge(terminals, on=KEYS, validate="one_to_one").merge(
        latest, on=KEYS, validate="one_to_one"
    )
    pairs = pairs.merge(
        games[["game_id", "season", "week", "monday", "freeze", "wednesday", "deadline", "kickoff", "in_frozen_fit"]],
        on="game_id", validate="many_to_one",
    )
    pairs["tuesday_early_move_toward_home"] = pairs.tuesday_end_home_spread_line - pairs.noon_home_spread_line
    pairs["late_boundary_move_toward_home"] = pairs.decision_home_spread_line - pairs.tuesday_end_home_spread_line
    if not np.allclose(
        pairs.tuesday_early_move_toward_home + pairs.late_boundary_move_toward_home,
        pairs.decision_home_spread_line - pairs.noon_home_spread_line, atol=1e-12, rtol=0,
    ):
        raise ValueError("Same-book Tuesday and later blocks do not telescope")
    for prefix, cutoff, inclusive in [("noon", "freeze", True), ("tuesday_end", "wednesday", False), ("decision", "deadline", False)]:
        observed = pairs[f"{prefix}_observed_at_utc"]
        snapshot = pairs[f"{prefix}_snapshot_timestamp_utc"]
        update = pairs[f"{prefix}_bookmaker_last_update_utc"]
        within = observed.le(pairs[cutoff]) & snapshot.le(pairs[cutoff]) if inclusive else observed.lt(pairs[cutoff]) & snapshot.lt(pairs[cutoff])
        if not (within & update.le(observed) & observed.le(snapshot) & observed.ge(pairs.monday) & observed.lt(pairs.kickoff)).all():
            raise ValueError(f"Invalid {prefix} boundary clocks")
    early = pairs.groupby("game_id", as_index=False).agg(
        tuesday_early_move_toward_home=("tuesday_early_move_toward_home", "median"),
        tuesday_early_books=("bookmaker_key", "size"),
    )
    bounds = games[["game_id", "kickoff", "week_first_commence_utc"]].rename(columns={"kickoff": "commence_time_utc"})
    source = quotes.rename(columns={"game_id": "nflverse_game_id"})
    later = sunday_move(source, bounds).rename(
        columns={"sunday_move": "reconstructed_late_move_toward_home", "sunday_books": "reconstructed_late_books"}
    )
    features = games.merge(early, on="game_id", how="left", validate="one_to_one").merge(
        later, on="game_id", how="left", validate="one_to_one"
    )
    features["early_available"] = features.tuesday_early_books.notna()
    features["late_available"] = features.reconstructed_late_books.notna()
    features["source_complete"] = features.early_available & features.late_available
    inventory = {
        "noon_book_anchors": len(anchors),
        "tuesday_end_book_anchors": len(terminals),
        "same_book_boundary_pairs": len(pairs),
        "frozen_fit_boundary_pairs_2023_2025": int((pairs.in_frozen_fit & pairs.season.between(2023, 2025)).sum()),
        "boundary_clock_violations": 0,
        "telescope_violations": 0,
    }
    return features, pairs, inventory


def main() -> None:
    declaration = LANE.read_text(encoding="utf-8-sig")
    required = ["Predeclared protocol", "fit 2023, calibrate 2024, outer 2025", "261 looks"]
    if not all(value in declaration for value in required):
        raise ValueError("The frozen lane declaration must precede execution")
    metadata = json.loads((FIT_ROOT / "metadata.json").read_text(encoding="utf-8-sig"))
    opener_root = Path("artifacts") / metadata["opener_evaluation"]
    sources = [QUOTES, FIT_ROOT / "per_game.parquet", FIT_ROOT / "metadata.json", SCHEDULE, opener_root / "per_game.parquet", opener_root / "metadata.json"]
    missing = [str(path) for path in sources if not path.is_file()]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if missing:
        payload = {"state": "source_inventory_only", "missing_sources": missing, "outcome_looks_executed": 0}
        (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        REPORT.write_text("# LEAD-82 unit 1\n\n**Measured:** required local sources absent: " + ", ".join(missing) + ".\nNo extraction, fit or score was run. Declared family: 261 looks; executed: 0.\n", encoding="utf-8")
        print(json.dumps(payload))
        return
    if metadata["base_probability_policy"] != "discrete_conditional_non_push_v1":
        raise ValueError("The frozen model is not the declared discrete conditional probability")
    if metadata["market_move_feature_version"] != "leader_median_through_sunday_prekick_v1" or metadata["ridge"] != 0.001:
        raise ValueError("The frozen four-term recipe differs from the declaration")
    games, fitted = load_population(opener_root / "per_game.parquet")
    quotes, quote_inventory = load_quotes(games)
    features, pairs, boundaries = extract(games, quotes)
    comparison = features.merge(fitted[["game_id", MOVE_COLUMN, MOVE_AVAILABLE_COLUMN]], on="game_id", validate="one_to_one")
    late = comparison.loc[comparison.season.between(2023, 2025) & comparison.late_available]
    parity_error = float((late.reconstructed_late_move_toward_home - late[MOVE_COLUMN]).abs().max()) if len(late) else None
    coverage = []
    for season in range(2020, 2026):
        frame = features.loc[features.season.eq(season)]
        frozen = frame.loc[frame.in_frozen_fit]
        coverage.append({
            "season": season, "opener_games_including_pushes": len(frame), "frozen_fit_games": len(frozen),
            "opener_early_complete": int(frame.early_available.sum()),
            "opener_late_complete": int(frame.late_available.sum()),
            "frozen_fit_early_complete": int(frozen.early_available.sum()),
            "frozen_fit_late_complete": int(frozen.late_available.sum()),
            "frozen_fit_both_complete": int(frozen.source_complete.sum()),
            "frozen_fit_book_pairs": int((pairs.season.eq(season) & pairs.in_frozen_fit).sum()),
        })
    features.to_parquet(OUTPUT / "features.parquet", index=False)
    pairs.to_parquet(OUTPUT / "boundary_pairs.parquet", index=False)
    summary = {
        "state": "boundary_clock_unit_complete", "sources": {str(path): digest(path) for path in sources},
        "quote_cache_rows": pq.ParquetFile(QUOTES).metadata.num_rows,
        "leader_books": list(LEADER_BOOKS), "coverage": coverage,
        "quotes": quote_inventory, "boundaries": boundaries,
        "late_move_parity_games_2023_2025": len(late), "late_move_max_abs_error_2023_2025": parity_error,
        "late_move_changed_games_2023_2025": int((late.reconstructed_late_move_toward_home - late[MOVE_COLUMN]).abs().gt(1e-12).sum()),
        "late_move_availability_mismatches_2023_2025": int((late[MOVE_AVAILABLE_COLUMN] != 1).sum()),
        "planned_looks": 261, "outcome_looks_executed": 0, "fits_executed": 0,
        "outcomes_loaded": False, "closing_inputs_loaded": False,
        "upstream_cutoff_verification": "pending row-level training and discrete-distribution provenance before fitted replay",
        "source_sign": "positive changes are movement toward home; parser convention checked in scripts/lead73_unit2.py:97-101",
        "frozen_protocol_sha256": hashlib.sha256(declaration.split("### Predeclared protocol", 1)[1].split("## Tried", 1)[0].encode()).hexdigest(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# LEAD-82 unit 1: Tuesday boundary and clock extraction", "",
        "**Measured:** boundary/clock unit only; no outcomes or closing inputs loaded, no fitting or scoring.",
        "**Read:** ROADMAP.md:865 splits boundary/clock extraction from the later cached fitted replay.",
        "The declaration was saved in `docs/lanes/lead82.md` before execution and remains unchanged.", "",
        "## Frozen protocol", "",
        "Median same-leader-book signed change from the last quote at/before Tuesday noon Eastern",
        "to the last quote before Wednesday 00:00 Eastern. Add this term beside Wednesday-to-decision",
        "movement in the existing four-term discrete conditional probability. Historical opener is",
        "the frozen pool-line proxy. Fit 2023, calibrate 2024, outer 2025; existing ridge 0.001.",
        "Primary Brier; accuracy/log loss/RPS, five arms, B=6, K=4, F=1, 261 declared looks.",
        "**Measured:** 0 outcome looks and 0 fits executed. One outer season cannot establish season stability.", "",
        "## Source and coverage inventory", "",
        f"**Measured:** quote cache {summary['quote_cache_rows']:,} rows; frozen fit {len(fitted):,} games; opener source {len(games):,} games including pushes.",
        "No outcome filter was applied to the opener source. Source gaps remain missing rather than becoming zero moves.", "Tuesday boundaries use only `intraday_hourly`; scheduled `tue_open` captures are not noon-boundary evidence.",
        "**Measured:** exact counts below; confidence intervals are not applicable to this finite local inventory.", "",
        table(
            ["Season", "Openers incl. pushes", "Frozen fit", "Opener early", "Opener later", "Fit early", "Fit later", "Fit both", "Fit book pairs"],
            [[row[key] for key in ["season", "opener_games_including_pushes", "frozen_fit_games", "opener_early_complete", "opener_late_complete", "frozen_fit_early_complete", "frozen_fit_late_complete", "frozen_fit_both_complete", "frozen_fit_book_pairs"]] for row in coverage],
        ), "",
        f"**Measured:** {boundaries['frozen_fit_boundary_pairs_2023_2025']:,} same-book boundary pairs for frozen-fit games in 2023-2025; {boundaries['same_book_boundary_pairs']:,} including the wider opener source.",
        f"**Measured:** reconstructed later moves compared on {len(late)} frozen-fit games; {summary['late_move_changed_games_2023_2025']} changed; maximum absolute difference from the saved feature = {parity_error}; availability mismatches = {summary['late_move_availability_mismatches_2023_2025']}.", "",
        "## Runtime timing checks", "",
        "**Measured:** every saved pair passes target-week, source-season, observation, snapshot,",
        "bookmaker-update and both source/schedule kickoff gates. Noon is inclusive; Wednesday and decision boundaries",
        "are strict. Decisions are capped at the earlier of kickoff and Sunday 12:45 Eastern.",
        "Same-book early plus later changes telescope to noon-to-decision movement; zero violations.",
        "Both arms' later move is rebuilt with the existing `sunday_move` implementation after these gates.",
        f"**Measured:** {quote_inventory['selected_rows']:,} selected quote rows; {quote_inventory['rejected_rows']:,} outside one or more gates; {quote_inventory['unique_eligible_rows']:,} eligible observation/source rows.",
        "Gate rejection counts overlap and include snapshots outside the target-week decision window.", "Archived kickoff timestamps may differ from the final schedule; quotes must precede both clocks, with no exact-equality requirement.", "",
        table(["Gate", "Rejected quote rows"], [[key, value] for key, value in quote_inventory["gate_failures_nonexclusive"].items()]), "",
        "## Replay handoff", "",
        "**Measured:** in-sample/out-of-sample metrics, gaps, coefficients, decisive-game records,",
        "intervals and `probability_positive` are not estimated in this outcome-free unit.",
        "Before the separately declared fitted replay, verify upstream model/discrete-distribution",
        "training cutoffs at row level and retain the opener pushes for distribution/RPS work.",
        "Then use the unchanged fit/calibration/test split and paired baselines from the lane.",
        "**Inferred:** this inventory establishes feature availability only; it makes no claim of a useful signal.",
        "No research closure, registry entry, promotion or served-card change follows from coverage.", "",
        "Saved outcome-free feature rows: `tests/scratch/codex/lead82_unit1/features.parquet`.",
        "Saved per-book boundary evidence: `tests/scratch/codex/lead82_unit1/boundary_pairs.parquet`.",
        "Source hashes, exact counts and protocol hash: `tests/scratch/codex/lead82_unit1/summary.json`.",
        "Run: `.tools/uv.exe run --no-sync python scripts/lead82_unit1.py` with `UV_CACHE_DIR` set to a writable temporary directory.", "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "state": summary["state"], "quote_cache_rows": summary["quote_cache_rows"],
        "coverage": coverage, "boundaries": boundaries,
        "late_move_max_abs_error_2023_2025": parity_error, "outcome_looks_executed": 0,
        "report": str(REPORT), "cache": str(OUTPUT),
    }))


if __name__ == "__main__":
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with threadpool_limits(limits=2):
        main()
