from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from lead69_unit1 import (
    ARMS,
    METRICS,
    POPULATION,
    QUOTES,
    SLOTS,
    build_predictors,
    fit_predict,
    metric_arrays,
    require,
    table,
)
from lead69_unit1 import (
    interval as season_interval,
)
from lead73_unit1 import load_games
from lead73_unit2 import SCHEDULE, digest, read_parquet
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_probability_fit import FIT_FEATURES
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

OUTPUT = Path("tests/scratch/codex/lead69_unit2")
REPORT = Path("docs/lead69_unit2.md")
ALIGNMENT = Path("tests/scratch/codex/lead73_unit2")
SEASONS = tuple(range(2020, 2026))
BOOTSTRAPS = 10000
SEED = 20260929
FEATURES = {
    "base": list(FIT_FEATURES),
    "horizon": [*FIT_FEATURES, "move_log_hours"],
    "slots": [*FIT_FEATURES, *(f"move_slot_{s}" for s in SLOTS if s != "sunday_early")],
}


def add_slots(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.commence_time_utc.dt.tz_convert("America/New_York")
    excluded = local.dt.weekday.eq(0) | (local.dt.weekday.eq(6) & local.dt.hour.ge(19))
    result = frame.loc[~excluded].copy()
    local = local.loc[result.index]
    result["slot"] = np.select(
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
    for slot in SLOTS:
        if slot != "sunday_early":
            result[f"move_slot_{slot}"] = result[MOVE_COLUMN] * result.slot.eq(slot)
    return result


def quote_terms(quotes: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = quotes.copy().rename(columns={"nflverse_game_id": "game_id"})
    q = q.drop(columns=["deadline"], errors="ignore").merge(
        games[["game_id", "deadline", "week_first_commence_utc"]],
        on="game_id",
        validate="many_to_one",
    )
    for column in ("observed_at_utc", "bookmaker_last_update_utc"):
        q[column] = pd.to_datetime(q[column], utc=True, errors="coerce")
    if "snapshot_timestamp_utc" not in q:
        q["snapshot_timestamp_utc"] = q.observed_at_utc
    q["snapshot_timestamp_utc"] = pd.to_datetime(
        q.snapshot_timestamp_utc, utc=True, errors="coerce"
    )
    if "source_scan_at_utc" in q:
        q["source_scan_at_utc"] = pd.to_datetime(q.source_scan_at_utc, utc=True, errors="coerce")
        q["quote_as_of_utc"] = q.source_scan_at_utc.fillna(q.observed_at_utc)
    else:
        q["quote_as_of_utc"] = q.observed_at_utc
    q["availability"] = q[
        ["observed_at_utc", "snapshot_timestamp_utc", "bookmaker_last_update_utc"]
    ].max(axis=1)
    local = q.week_first_commence_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    sunday = local.dt.normalize() + pd.to_timedelta((6 - local.dt.weekday) % 7, unit="D")
    q["monday"] = (
        (sunday - pd.Timedelta(days=6)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    q["wednesday"] = (
        (sunday - pd.Timedelta(days=4)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    if "commence_time_utc" in q:
        q["commence_time_utc"] = pd.to_datetime(q.commence_time_utc, utc=True, errors="coerce")
        q = q.loc[
            q.observed_at_utc.lt(q.commence_time_utc)
            & q.snapshot_timestamp_utc.lt(q.commence_time_utc)
        ].copy()
    q = q.loc[
        q.observed_at_utc.ge(q.monday)
        & q.availability.lt(q.deadline)
        & q.bookmaker_last_update_utc.le(q.observed_at_utc)
        & q.quote_as_of_utc.le(q.observed_at_utc)
        & q.snapshot_timestamp_utc.notna()
        & np.isfinite(q.home_spread_line)
    ].copy()
    keys = ["game_id", "bookmaker_key", "quote_as_of_utc"]
    require(not q.groupby(keys).home_spread_line.nunique().gt(1).any(), "Conflicting book quotes")
    q = q.sort_values([*keys, "observed_at_utc"]).drop_duplicates(keys, keep="last")
    grouped = q.groupby(["game_id", "bookmaker_key"])
    q["move"] = grouped.home_spread_line.diff()
    q["previous_availability"] = grouped.availability.shift()
    increments = q.loc[q.quote_as_of_utc.ge(q.wednesday) & q.move.notna()].copy()
    require(increments.availability.lt(increments.deadline).all(), "Late quote in move")
    books = increments.groupby(["game_id", "bookmaker_key"], as_index=False).agg(
        net_move=("move", "sum"),
        start_availability=("previous_availability", "first"),
        last_availability=("availability", "max"),
        deadline=("deadline", "first"),
    )
    books["hours"] = (books.deadline - books.start_availability).dt.total_seconds() / 3600
    require(books.hours.gt(0).all() and books.hours.notna().all(), "Invalid quote horizon")
    books["weighted_move"] = books.net_move * np.log(books.hours)
    terms = books.groupby("game_id", as_index=False).agg(
        quote_move=("net_move", "median"),
        quote_horizon=("weighted_move", "median"),
        horizon_min=("hours", "min"),
        horizon_max=("hours", "max"),
    )
    return terms, books


def predictors() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    provenance = {"unit1": {}}
    recent = build_predictors(provenance["unit1"])
    source = json.loads((ALIGNMENT / "provenance.json").read_text(encoding="utf-8"))
    require(digest(SCHEDULE) == source["schedule_sha256"], "Schedule changed after LEAD-73")
    require(digest(POPULATION) == source["frozen_population_sha256"], "Frozen population changed")
    for entry in source["sources"]:
        manifest = Path(entry["manifest"])
        require(digest(manifest) == entry["manifest_sha256"], "Historical manifest changed")
        require(
            digest(manifest.parent / "quotes.parquet") == entry["quotes_sha256"],
            "Historical quotes changed",
        )
    evidence = read_parquet(ALIGNMENT / "quote_evidence.parquet")
    moves = read_parquet(ALIGNMENT / "recovered_moves.parquet")
    schedule = load_games(SCHEDULE)
    schedule = schedule.loc[schedule.season.between(2020, 2022)].copy()
    sunday = schedule.monday.dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(
        days=6, hours=16
    )
    schedule["deadline"] = pd.concat(
        [schedule.kickoff, sunday.dt.tz_localize("America/New_York").dt.tz_convert("UTC")], axis=1
    ).min(axis=1)
    schedule = schedule.rename(columns={"kickoff": "commence_time_utc"})
    schedule["week_first_commence_utc"] = schedule.monday
    columns = ["game_id", "season", "week", "game_type", "model_probability", *FIT_FEATURES]
    early = read_parquet(POPULATION, columns)
    early = early.loc[early.season.between(2020, 2022)].copy()
    early = early.merge(
        moves[["game_id", "recovered_move"]], on="game_id", how="left", validate="one_to_one"
    )
    provenance["older_missing_moves"] = early.loc[early.recovered_move.isna(), "game_id"].tolist()
    provenance["older_starting"] = early.groupby("season").size().to_dict()
    early = early.loc[early.recovered_move.notna()].copy()
    early[MOVE_COLUMN] = early.recovered_move
    early[MOVE_AVAILABLE_COLUMN] = 1.0
    early = early.merge(
        schedule[["game_id", "commence_time_utc", "deadline", "week_first_commence_utc"]],
        on="game_id",
        validate="one_to_one",
    )
    early = add_slots(early)
    terms, early_books = quote_terms(evidence, early)
    early = early.merge(terms, on="game_id", how="left", validate="one_to_one")
    require(
        np.allclose(early.quote_move, early[MOVE_COLUMN]),
        "Older move reconstruction differs from LEAD-73",
    )
    early["move_log_hours"] = early.quote_horizon
    q = pd.read_parquet(
        QUOTES,
        filters=[
            ("decision_label", "==", "intraday_hourly"),
            ("archive_season", "in", [2023, 2024, 2025]),
            ("bookmaker_key", "in", list(LEADER_BOOKS)),
            ("market", "==", "spreads"),
        ],
        use_threads=False,
    )
    terms, recent_books = quote_terms(q, recent)
    extended_recent = recent.merge(terms, on="game_id", how="left", validate="one_to_one")
    require(
        np.allclose(extended_recent.quote_move, extended_recent[MOVE_COLUMN]),
        "Recent move reconstruction differs from unit 1",
    )
    extended_recent["move_log_hours"] = extended_recent.quote_horizon
    frame = (
        pd.concat([early, extended_recent], ignore_index=True)
        .sort_values(["season", "week", "game_id"])
        .reset_index(drop=True)
    )
    require(not frame.game_id.duplicated().any(), "Duplicate games")
    require(set(frame.season) == set(SEASONS), "Missing season")
    require(
        np.isfinite(frame[[*FIT_FEATURES, "move_log_hours", "model_probability"]]).all().all(),
        "Nonfinite predictors",
    )
    books = pd.concat([early_books, recent_books], ignore_index=True)
    books.to_parquet(OUTPUT / "quote_horizons.parquet", index=False)
    provenance.update(
        {
            "alignment_sources": source["sources"],
            "alignment_evidence_sha256": digest(ALIGNMENT / "quote_evidence.parquet"),
            "population_sha256": digest(POPULATION),
            "eligible_by_season": frame.groupby("season").size().to_dict(),
            "slot_counts": frame.groupby("slot").size().to_dict(),
            "book_horizon_hours": [float(books.hours.min()), float(books.hours.max())],
            "book_rows": len(books),
            "script_sha256": digest(Path(__file__)),
        }
    )
    return frame, recent, provenance


class WeekBootstrap:
    def __init__(self, frame: pd.DataFrame):
        index = pd.MultiIndex.from_frame(frame[["season", "week"]])
        self.codes, unique = pd.factorize(index, sort=True)
        self.blocks = len(unique)
        self.weights = np.zeros((BOOTSTRAPS, self.blocks), dtype=float)
        rng = np.random.default_rng(SEED)
        for season in sorted(frame.season.unique()):
            positions = np.flatnonzero(unique.get_level_values(0) == season)
            self.weights[:, positions] = rng.multinomial(
                len(positions), np.full(len(positions), 1 / len(positions)), BOOTSTRAPS
            )

    def summarize(self, values: np.ndarray, eligible: np.ndarray | None = None) -> dict:
        values = np.asarray(values, dtype=float)
        selected = np.ones(len(values), dtype=bool) if eligible is None else eligible
        counts = np.bincount(self.codes[selected], minlength=self.blocks)
        sums = np.bincount(self.codes[selected], weights=values[selected], minlength=self.blocks)
        denominator = self.weights @ counts
        if not selected.any():
            return {
                "effect": None,
                "low": None,
                "high": None,
                "probability_positive": None,
                "standard_error": None,
            }
        draws = (self.weights @ sums)[denominator > 0] / denominator[denominator > 0]
        low, high = np.quantile(draws, [0.025, 0.975])
        return {
            "effect": float(values[selected].mean()),
            "low": float(low),
            "high": float(high),
            "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
            "standard_error": float(np.std(draws, ddof=1)),
        }


def fit(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[dict]]:
    frame = frame.copy()
    coefficients = []
    for arm, terms in FEATURES.items():
        for held in ["IS", *sorted(frame.season.unique())]:
            test = (
                np.ones(len(frame), dtype=bool)
                if held == "IS"
                else frame.season.eq(held).to_numpy()
            )
            train = frame if held == "IS" else frame.loc[~test]
            p, coefficient = fit_predict(train, frame.loc[test], terms)
            frame.loc[test, f"{arm}_{'is' if held == 'IS' else 'oos'}"] = p
            coefficients.append(
                {
                    "population": name,
                    "arm": arm,
                    "fold": str(held),
                    "n_train": len(train),
                    **coefficient,
                }
            )
    for mode in ("is", "oos"):
        frame[f"model_only_{mode}"] = frame.model_probability
        frame[f"market_{mode}"] = 0.5
    return frame, coefficients


def score(frame: pd.DataFrame) -> dict:
    boot = WeekBootstrap(frame)
    arrays = {
        (mode, arm): metric_arrays(frame, frame[f"{arm}_{mode}"].to_numpy())
        for mode in ("is", "oos")
        for arm in ARMS
    }
    result = {
        "games": len(frame),
        "blocks": boot.blocks,
        "metrics": {},
        "contrasts": {},
        "gaps": {},
        "decisive": [],
        "seasons": [],
        "calibration": [],
    }
    for mode in ("is", "oos"):
        for arm in ARMS:
            result["metrics"][f"{arm}_{mode}"] = {
                metric: boot.summarize(values) for metric, values in arrays[mode, arm].items()
            }
        for arm in ("horizon", "slots"):
            diff = arrays[mode, arm]["accuracy"] - arrays[mode, "base"]["accuracy"]
            result["decisive"].append(
                {
                    "arm": arm,
                    "mode": mode,
                    "wins": int((diff > 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "rate": boot.summarize((diff > 0).astype(float), diff != 0),
                }
            )
            for baseline in ("base", "model_only", "market"):
                result["contrasts"][f"{arm}_{baseline}_{mode}"] = {
                    metric: boot.summarize(
                        (1 if metric in ("accuracy", "line_move") else -1)
                        * (arrays[mode, arm][metric] - arrays[mode, baseline][metric])
                    )
                    for metric in METRICS
                }
    for arm in FEATURES:
        result["gaps"][arm] = {
            metric: boot.summarize(
                (1 if metric in ("accuracy", "line_move") else -1)
                * (arrays["is", arm][metric] - arrays["oos", arm][metric])
            )
            for metric in METRICS
        }
    for season in sorted(frame.season.unique()):
        mask = frame.season.eq(season).to_numpy()
        for arm in ARMS:
            result["seasons"].append(
                {
                    "season": int(season),
                    "arm": arm,
                    "games": int(mask.sum()),
                    **{
                        metric: float(values[mask].mean())
                        for metric, values in arrays["oos", arm].items()
                    },
                }
            )
    for arm in ARMS:
        p = frame[f"{arm}_oos"].to_numpy()
        bins = np.minimum((p * 5).astype(int), 4)
        for band in range(5):
            mask = bins == band
            result["calibration"].append(
                {
                    "arm": arm,
                    "band": f"{band / 5:.1f}-{(band + 1) / 5:.1f}",
                    "games": int(mask.sum()),
                    "predicted": float(p[mask].mean()) if mask.any() else None,
                    "actual": float(frame.loc[mask, "home_covered"].mean()) if mask.any() else None,
                }
            )
    return result


def estimate(row: dict, scale: float = 1.0) -> str:
    if row["effect"] is None:
        return "empty"
    return f"{row['effect'] * scale:.6f} [{row['low'] * scale:.6f}, {row['high'] * scale:.6f}]"


def render(results: dict, coefficients: list[dict], provenance: dict, replay: dict) -> str:
    lines = [
        "# LEAD-69 unit 2: quote-availability horizon",
        "",
        "Protocol amendment saved in docs/lanes/lead69.md before outcomes.",
        "**Measured:** two research looks; 33 numerical fits, 660 reporting looks; no tuning.",
        "Historical frozen lines are openers, not pre-2026 Splash captures.",
        "",
        "## Decisive games first",
        "",
    ]
    rows = []
    for name, result in results.items():
        for row in result["decisive"]:
            rows.append(
                [
                    name,
                    row["mode"],
                    row["arm"],
                    f"{row['wins']}-{row['losses']}",
                    estimate(row["rate"], 100),
                ]
            )
    lines += table(
        ["Population", "Mode", "Arm vs base", "Record", "Decisive accuracy % [95%]"], rows
    )
    lines += [
        "## Protocol and sources",
        "",
        "**Read:** unit-1 terms/metrics: docs/lead69_protocol.md; older alignment: "
        "docs/lead73_unit2.md.",
        "The pre-outcome amendment changes horizon timing to the start quote's observed "
        "availability.",
        "For each book, sum Wednesday-onward changes, multiply by log(hours from start-quote",
        "availability to min(kickoff, Sunday 16:00 ET)), then take the median across books.",
        "Availability=max(capture, snapshot, book update). The base move remains the median",
        "unweighted book net move. The original Tuesday-noon definition is replayed separately.",
        "Those two interactions need not agree when book start times differ.",
        "Older evidence ends at 12:45 Sunday; this archive limit was not a tuned cutoff.",
        "**Measured:** source manifest/quote hashes, schedule/population hashes, timestamp gates",
        "and move reconstruction parity against both prior units passed before outcome loading.",
        "Regular-season nonpush rows with verified moves only; retain unit-1 SNF/MNF exclusions.",
        f"Missing older moves: {len(provenance['older_missing_moves'])}; book horizons",
        f"{provenance['book_horizon_hours'][0]:.3f}-{provenance['book_horizon_hours'][1]:.3f} "
        f"hours over {provenance['book_rows']} book/game pairs.",
        "",
    ]
    lines += table(
        ["Season", "Eligible games"],
        [[season, count] for season, count in provenance["eligible_by_season"].items()],
    )
    lines += [
        "**Read:** base is intercept plus model logit, signed move, availability, composition sum;",
        "horizon adds one term; slots add five fixed non-reference interactions. Ridge=0.001;",
        "training-only standardization; six LOSO folds plus IS, original three-season replay "
        "plus IS.",
        "One fitted probability chooses each side at 0.5. Model-only uses the same discrete",
        "conditional nonpush probability; neutral market p=0.5 selects home on ties.",
        "Opener-to-close movement is evaluation only, never an input.",
        "**Measured:** 10,000 paired season-stratified week-block draws, seed 20260929;",
        "95% percentile intervals; probability_positive=P(gain>0)+0.5P(gain=0); predictions fixed.",
        "Reporting budget: 33 fits + 120 aggregate metrics + 144 contrasts + 36 gaps +",
        "240 season metric cells + 75 calibration bands + 12 decisive records = 660 looks.",
        "These correlated diagnostics are not independent discoveries.",
        "",
        "## Unit-1 continuity",
        "",
        f"**Measured:** original 685-game replay accuracy gain {estimate(replay, 100)} pp",
        f"using its exact season bootstrap; "
        f"probability_positive={replay['probability_positive']:.6f}.",
        "Tables below use week-block intervals. extended_recent is the same 685 games under",
        "six-season fits and quote-availability horizons; original_replay keeps three-season",
        "fits and the Tuesday horizon. This separates extended training from exact reproduction.",
        "",
    ]
    for name, result in results.items():
        lines += [
            f"## {name}: {result['games']} games, {result['blocks']} season/week blocks",
            "",
            "**Measured:** accuracy is percent; movement is spread points; losses are natural "
            "units.",
            "",
        ]
        rows = []
        for arm in ARMS:
            for metric in METRICS:
                scale = 100 if metric == "accuracy" else 1
                gap = (
                    estimate(result["gaps"][arm][metric], scale)
                    if arm in FEATURES
                    else "same baseline"
                )
                rows.append(
                    [
                        arm,
                        metric,
                        estimate(result["metrics"][f"{arm}_is"][metric], scale),
                        estimate(result["metrics"][f"{arm}_oos"][metric], scale),
                        gap,
                    ]
                )
        lines += table(
            ["Arm", "Metric", "IS [95%]", "OOS [95%]", "Optimistic IS/OOS gap [95%]"], rows
        )
        lines += [
            "Positive gap means higher IS accuracy/movement or lower IS loss.",
            "Positive contrast means challenger improves on the comparator.",
            "",
        ]
        rows = []
        for contrast, metrics in result["contrasts"].items():
            for metric, row in metrics.items():
                rows.append(
                    [
                        contrast,
                        metric,
                        estimate(row, 100 if metric == "accuracy" else 1),
                        f"{row['probability_positive']:.6f}",
                    ]
                )
        lines += table(["Contrast", "Metric", "Gain [95%]", "probability_positive"], rows)
        lines += ["OOS season stability, descriptive:", ""]
        lines += table(
            ["Season", "Arm", "Games", "Accuracy %", "Log loss", "Brier", "Line move"],
            [
                [
                    row["season"],
                    row["arm"],
                    row["games"],
                    f"{100 * row['accuracy']:.4f}",
                    *[f"{row[m]:.6f}" for m in METRICS[1:]],
                ]
                for row in result["seasons"]
            ],
        )
        lines += ["Fixed OOS reliability bins; no bin selected:", ""]
        lines += table(
            ["Arm", "Band", "Games", "Mean p(home)", "Home-cover rate"],
            [
                [
                    row["arm"],
                    row["band"],
                    row["games"],
                    *[
                        "empty" if row[k] is None else f"{row[k]:.6f}"
                        for k in ("predicted", "actual")
                    ],
                ]
                for row in result["calibration"]
            ],
        )
    lines += [
        "## Natural-unit coefficients",
        "",
        "**Measured:** fold identifies the held-out season; IS is descriptive. Availability",
        "is constant one in this population, so its standardized coefficient is zero.",
        "",
    ]
    terms = [
        "intercept",
        *FIT_FEATURES,
        "move_log_hours",
        *(f"move_slot_{s}" for s in SLOTS if s != "sunday_early"),
    ]
    lines += table(
        ["Population", "Arm", "Fold", "Train n", *terms],
        [
            [
                row["population"],
                row["arm"],
                row["fold"],
                row["n_train"],
                *[f"{row[t]:.6f}" if t in row else "—" for t in terms],
            ]
            for row in coefficients
        ],
    )
    for population in ("extended", "original_replay"):
        rows = [
            row
            for row in coefficients
            if row["population"] == population and row["arm"] == "horizon" and row["fold"] != "IS"
        ]
        values = np.array([row["move_log_hours"] for row in rows])
        lines += [
            f"**Measured:** {population} horizon coefficient range [{values.min():.6f}, "
            f"{values.max():.6f}], mean {values.mean():.6f}, fold SD {values.std(ddof=1):.6f}; "
            f"positive {int((values > 0).sum())}/{len(values)}, "
            f"negative {int((values < 0).sum())}/{len(values)}.",
            "",
        ]
    primary = results["extended"]["contrasts"]["horizon_base_oos"]["accuracy"]
    loss = results["extended"]["contrasts"]["horizon_base_oos"]["log_loss"]
    lines += [
        "## Interpretation and limits",
        "",
        f"**Measured:** primary accuracy gain {estimate(primary, 100)} pp, "
        f"probability_positive={primary['probability_positive']:.6f};",
        f"log-loss gain {estimate(loss)}, probability_positive={loss['probability_positive']:.6f}.",
        "**Inferred:** unresolved_below_power pending the owner's serial registry record;",
        "no terminal closure or serving proposal. AGENTS.md separates promotion from closure:",
        "a zero-crossing interval closes nothing. No powered control or split-half study ran.",
        "The accuracy interval is wholly adverse for this declared version; its proper-score",
        "contrasts remain unresolved. This does not support changing the served picks or",
        "closing the broader horizon mechanism, especially given the timing amendment.",
        "LOSO coefficients are out of season but use future seasons and inherited frozen upstream",
        "model/composition features. This is not a chronological outer test. Bootstrap omits",
        "refit/selection uncertainty. Archive quote density and early 12:45 cutoff differ.",
        "Historical forced-pick accuracy establishes neither profit nor game-level probability.",
        "",
        "## Reproduction",
        "",
        ".tools/uv.exe run --no-sync --no-cache python scripts/lead69_unit2.py",
        "",
        "Prediction rows, quote horizons, coefficients, provenance and summaries:",
        "tests/scratch/codex/lead69_unit2/. Owner-run record commands are in the lane; none ran.",
        "",
    ]
    return "\n".join(lines)


def registry(result: dict) -> dict:
    cells = []
    for arm in ("horizon", "slots"):
        for metric, row in result["contrasts"][f"{arm}_base_oos"].items():
            scale = 100 if metric == "accuracy" else 1
            units = {
                "accuracy": "accuracy_points",
                "log_loss": "log_loss_improvement",
                "brier": "brier_improvement",
                "line_move": "ats_points",
            }[metric]
            cells.append(
                {
                    "name": f"LEAD69-unit2-{arm}-{metric}",
                    "description": f"Six-season LOSO {arm} versus served four-term "
                    "specification; opener grade; positive gain is better.",
                    "effect": row["effect"] * scale,
                    "effect_units": units,
                    "interval_low": row["low"] * scale,
                    "interval_high": row["high"] * scale,
                    "standard_error": row["standard_error"] * scale,
                    "probability_positive": row["probability_positive"],
                    "sample_games": result["games"],
                    "sample_blocks": result["blocks"],
                    "classification": "unresolved_below_power",
                    "classification_evidence": "No refuted mechanism or powered positive "
                    "control established; owner reviews correlated metrics together.",
                    "plain_summary": "This check asks whether a sportsbook move should count "
                    "differently when its starting quote had more time before picks were due. "
                    "The evidence does not yet settle whether that improves the picks.",
                }
            )
    return {
        "source": "docs/lead69_unit2.md",
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "family": "LEAD69-unit2-horizon",
        "category": "market",
        "cells": cells,
        "notes": "Two research looks; 660 correlated reporting looks; 33 fits; 10000 "
        "season-stratified week-block draws; no tuning or serving action. Slots descriptive; "
        "continuity replay is not independent evidence.",
    }


def main() -> int:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, recent, provenance = predictors()
    print(f"Predictor gates passed: {len(frame)} games; recent replay {len(recent)}", flush=True)
    outcomes = read_parquet(POPULATION, ["game_id", "home_covered", "margin_vs_open", "open_move"])
    frame = frame.merge(outcomes, on="game_id", validate="one_to_one")
    recent = recent.merge(outcomes, on="game_id", validate="one_to_one")
    require(
        frame.margin_vs_open.ne(0).all()
        and frame.home_covered.eq(frame.margin_vs_open.gt(0)).all(),
        "Invalid opener target",
    )
    require(np.isfinite(frame.open_move).all(), "Missing evaluation move")
    with threadpool_limits(limits=1):
        frame, coefficients = fit(frame, "extended")
        recent, recent_coefficients = fit(recent, "original_replay")
        coefficients.extend(recent_coefficients)
        populations = {
            "extended": frame,
            "extended_recent": frame.loc[frame.season.ge(2023)].reset_index(drop=True),
            "original_replay": recent,
        }
        results = {name: score(rows) for name, rows in populations.items()}
        delta = (
            metric_arrays(recent, recent.horizon_oos.to_numpy())["accuracy"]
            - metric_arrays(recent, recent.base_oos.to_numpy())["accuracy"]
        )
        replay = season_interval(recent, delta)
    require(
        len(recent) == 685 and int((delta > 0).sum()) == 8 and int((delta < 0).sum()) == 6,
        "Original unit-1 continuity failed",
    )
    require(np.isclose(replay["effect"], 2 / 685), "Original primary gain changed")
    for name, rows in populations.items():
        rows.to_parquet(OUTPUT / f"{name}_predictions.parquet", index=False)
    for name, value in (
        ("summary", results),
        ("coefficients", coefficients),
        ("provenance", provenance),
        ("original_season_interval", replay),
        ("registry_batch", registry(results["extended"])),
    ):
        (OUTPUT / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    REPORT.write_text(render(results, coefficients, provenance, replay), encoding="utf-8")
    for name, result in results.items():
        primary = result["contrasts"]["horizon_base_oos"]["accuracy"]
        decisive = next(
            row for row in result["decisive"] if row["arm"] == "horizon" and row["mode"] == "oos"
        )
        print(
            f"{name}: n={result['games']} accuracy gain={estimate(primary, 100)} pp "
            f"P+={primary['probability_positive']:.6f} "
            f"decisive={decisive['wins']}-{decisive['losses']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
