from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lead73_unit1 import instant, load_games
from sunday_market_probability_eval import sunday_move
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import FLAG_SUM_COLUMN, MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    _design,
    _fit_logit,
    _natural_coefficients,
    _predict,
    _standardisers,
)
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

FIT_ROOT = Path("artifacts/pick_probability/20260929T192747Z")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
OUTPUT = Path("tests/scratch/codex/lead73_unit2")
REPORT = Path("docs/lead73_unit2.md")
SEASONS = tuple(range(2020, 2026))
SEED = 20260929
BOOTSTRAPS = 10000
EDGES = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0)
METRICS = ("accuracy_points", "log_loss", "brier")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def recover_moves() -> tuple[pd.DataFrame, dict]:
    policies = json.loads(Path("config/source_policies.json").read_text(encoding="utf-8-sig"))
    if "quota_headers_required" not in policies["sources"]["the_odds_api"]["conditions"]:
        raise ValueError("Source policy changed")
    games = load_games(SCHEDULE)
    games = games.loc[games.season.between(2020, 2022)].copy()
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "market",
        "home_spread_line",
        "line",
        "outcome_side",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "home_team",
        "away_team",
    ]
    frames = []
    sources = []
    invalid_rows = 0
    for path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        stamp = instant(manifest.get("snapshot_timestamp_utc"))
        if (
            manifest.get("provider") != "the-odds-api"
            or pd.isna(stamp)
            or not 2009 <= stamp.year <= 2023
            or request.get("sport") != "americanfootball_nfl"
            or "spreads" not in str(request.get("markets", "")).split(",")
            or manifest.get("capture_kind") != "historical_backfill"
            or int(request.get("season", 0)) not in range(2020, 2023)
        ):
            continue
        quotes_path = path.parent / "quotes.parquet"
        expected = manifest.get("files", {}).get("quotes.parquet", {}).get("sha256")
        if not expected or digest(quotes_path) != expected:
            raise ValueError(f"Quote integrity failure: {path}")
        q = read_parquet(quotes_path, columns)
        for name in ("observed_at_utc", "bookmaker_last_update_utc"):
            q[name] = pd.to_datetime(q[name], utc=True, errors="coerce", format="mixed")
        q = q.loc[q.market.eq("spreads") & q.bookmaker_key.isin(LEADER_BOOKS)].copy()
        valid = (
            q.observed_at_utc.eq(stamp)
            & q.bookmaker_last_update_utc.le(q.observed_at_utc)
            & np.isfinite(pd.to_numeric(q.home_spread_line, errors="coerce"))
        )
        invalid_rows += int((~valid).sum())
        q = q.loc[valid].copy()
        signed = np.where(q.outcome_side.eq("HOME"), -q.line, q.line)
        if not q.outcome_side.isin(("HOME", "AWAY")).all() or not np.allclose(
            signed, q.home_spread_line, atol=1e-9, rtol=0
        ):
            raise ValueError(f"Signed home spread disagrees with bookmaker handicap: {path}")
        frames.append(q)
        sources.append(
            {"manifest": str(path), "manifest_sha256": digest(path), "quotes_sha256": expected}
        )
    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.merge(
        games[["game_id", "home_team", "away_team", "monday", "deadline"]],
        left_on="nflverse_game_id",
        right_on="game_id",
        validate="many_to_one",
        suffixes=("", "_schedule"),
    )
    if not (
        quotes.home_team.eq(quotes.home_team_schedule)
        & quotes.away_team.eq(quotes.away_team_schedule)
    ).all():
        raise ValueError("Quote teams disagree with schedule")
    quotes = quotes.loc[
        quotes.observed_at_utc.ge(quotes.monday) & quotes.observed_at_utc.lt(quotes.deadline)
    ].copy()
    bounds = games[["game_id", "kickoff", "monday"]].rename(
        columns={"kickoff": "commence_time_utc"}
    )
    bounds["week_first_commence_utc"] = bounds.monday
    moves = sunday_move(quotes, bounds).rename(columns={"sunday_move": "recovered_move"})
    moves = moves.merge(games[["game_id", "season", "week"]], validate="one_to_one", on="game_id")
    keys = ["nflverse_game_id", "bookmaker_key", "observed_at_utc"]
    evidence = quotes.sort_values(keys).drop_duplicates(keys)[
        [*keys, "home_spread_line", "bookmaker_last_update_utc", "deadline"]
    ]
    return moves, {
        "sources": sources,
        "invalid_rows": invalid_rows,
        "admissible_quote_rows": len(evidence),
        "evidence": evidence,
    }


def load_population(moves: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads((FIT_ROOT / "metadata.json").read_text(encoding="utf-8"))
    if metadata["market_move_feature_version"] != "leader_median_through_sunday_prekick_v1":
        raise ValueError("Frozen fit uses a different market feature")
    frame = read_parquet(FIT_ROOT / "per_game.parquet")
    required = [
        *FIT_FEATURES,
        "model_probability",
        "home_covered",
        "margin_vs_open",
        "season",
        "week",
    ]
    if (
        frame.game_id.duplicated().any()
        or not np.isfinite(frame[required].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Frozen population is incomplete or duplicated")
    if not frame.season.isin(SEASONS).all() or not frame.game_type.eq("REG").all():
        raise ValueError("Frozen population does not match the declared seasons/population")
    if (
        frame.margin_vs_open.eq(0).any()
        or not frame.home_covered.eq(frame.margin_vs_open.gt(0)).all()
    ):
        raise ValueError("Opener target mismatch")
    if not frame.base_probability_policy.eq("discrete_conditional_non_push_v1").all():
        raise ValueError("Model probability is not the served discrete read")
    if not np.allclose(
        frame.model_logit, np.log(frame.model_probability / (1 - frame.model_probability))
    ):
        raise ValueError("Model logit mismatch")
    early = frame.season.lt(2023)
    if (
        not frame.loc[early, MOVE_COLUMN].eq(0).all()
        or not frame.loc[early, MOVE_AVAILABLE_COLUMN].eq(0).all()
    ):
        raise ValueError("Current fit already has early moves")
    move_path = Path("artifacts/sunday_market_probability/20260920_fixed/market_move.parquet")
    move_summary = json.loads(move_path.with_name("summary.json").read_text(encoding="utf-8"))
    if digest(move_path) != move_summary["market_move_sha256"]:
        raise ValueError("Current move artifact integrity failure")
    current = read_parquet(move_path).set_index("game_id").leader_median_net
    exposed = frame[MOVE_AVAILABLE_COLUMN].eq(1)
    expected = frame.loc[exposed, "game_id"].map(current)
    if not np.allclose(expected, frame.loc[exposed, MOVE_COLUMN], atol=1e-9, rtol=0):
        raise ValueError("Current signed move artifact parity failure")
    frame = frame.merge(
        moves[["game_id", "recovered_move", "sunday_books"]],
        how="left",
        on="game_id",
        validate="one_to_one",
    )
    frame["extended_move"] = frame.recovered_move.fillna(frame[MOVE_COLUMN])
    frame["extended_available"] = np.where(
        frame.recovered_move.notna(), 1.0, frame[MOVE_AVAILABLE_COLUMN]
    )
    if not frame.loc[~early, "extended_move"].equals(frame.loc[~early, MOVE_COLUMN]):
        raise ValueError("Late-season move changed")
    coverage = []
    for season in SEASONS:
        rows = frame.loc[frame.season.eq(season)]
        source = moves.loc[moves.season.eq(season)]
        coverage.append(
            {
                "season": season,
                "fit_games": len(rows),
                "source_games": len(source),
                "newly_available": int(rows.recovered_move.notna().sum()),
                "source_outside_fit": int((~source.game_id.isin(frame.game_id)).sum()),
                "current_available": int(rows[MOVE_AVAILABLE_COLUMN].sum()),
                "extended_available": int(rows.extended_available.sum()),
            }
        )
    opener_path = Path("artifacts") / metadata["opener_evaluation"] / "per_game.parquet"
    opener = read_parquet(opener_path, ["game_id", "margin_vs_open"])
    excluded = moves.loc[~moves.game_id.isin(frame.game_id)].merge(
        opener, how="left", on="game_id", validate="one_to_one", indicator=True
    )
    excluded["reason"] = np.select(
        [
            excluded["_merge"].eq("left_only"),
            excluded.margin_vs_open.eq(0),
            excluded.margin_vs_open.isna(),
        ],
        ["absent_from_opener", "opener_push", "ungraded"],
        default="other_nonfit",
    )
    exclusions = excluded.groupby(["season", "reason"]).size().rename("games").reset_index()
    return frame, {
        "coverage": coverage,
        "exclusion_counts": exclusions.to_dict("records"),
        "opener_sha256": digest(opener_path),
        "excluded_source_games": moves.loc[~moves.game_id.isin(frame.game_id)].to_dict("records"),
        "frozen_population_sha256": digest(FIT_ROOT / "per_game.parquet"),
        "metadata_sha256": digest(FIT_ROOT / "metadata.json"),
        "schedule_sha256": digest(SCHEDULE),
        "current_moves_sha256": digest(move_path),
        "model_id": metadata["active_model_id"],
        "base_policy": metadata["base_probability_policy"],
        "upstream_exclusions": {
            key: metadata[key] for key in ("pushes_dropped", "ungraded_dropped") if key in metadata
        },
    }


def fit_arms(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    probabilities = {
        "model": frame.model_probability.to_numpy(),
        "market": np.full(len(frame), 0.5),
    }
    coefficients = []
    parity = {}
    for arm in ("current", "extended"):
        work = frame.copy()
        if arm == "extended":
            work[MOVE_COLUMN] = work.extended_move
            work[MOVE_AVAILABLE_COLUMN] = work.extended_available
        probabilities[arm] = np.full(len(work), np.nan)
        for held in (*SEASONS, "IS"):
            test = (
                np.ones(len(work), dtype=bool) if held == "IS" else work.season.eq(held).to_numpy()
            )
            train = work if held == "IS" else work.loc[~test]
            means, stds = _standardisers(train)
            beta = _fit_logit(
                _design(train, means, stds), train.home_covered.to_numpy(dtype=float), FIT_RIDGE
            )
            prediction = _predict(work.loc[test], beta, means, stds)
            coefficients.append(
                {
                    "arm": arm,
                    "held_out": held,
                    "train_games": len(train),
                    **_natural_coefficients(beta, means, stds),
                }
            )
            if held == "IS":
                probabilities[f"{arm}_IS"] = prediction
            else:
                probabilities[arm][test] = prediction
        if arm == "current":
            for output, original in (
                (arm, "out_of_season_home_probability"),
                (f"{arm}_IS", "in_sample_home_probability"),
            ):
                error = float(np.max(np.abs(probabilities[output] - frame[original].to_numpy())))
                parity[output] = error
                if error > 1e-9:
                    raise ValueError("Current refit does not reproduce frozen fit")
    return probabilities, coefficients, parity


class Bootstrap:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.blocks = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        self.block_index = pd.MultiIndex.from_frame(self.blocks)
        self.weights = np.zeros((BOOTSTRAPS, len(self.blocks)), dtype=float)
        rng = np.random.default_rng(SEED)
        for season in SEASONS:
            positions = np.flatnonzero(self.blocks.season.eq(season))
            self.weights[:, positions] = rng.multinomial(
                len(positions), np.full(len(positions), 1 / len(positions)), size=BOOTSTRAPS
            )

    def estimate(self, values: np.ndarray, select: np.ndarray | None = None) -> dict:
        keep = np.ones(len(self.frame), dtype=bool) if select is None else select
        grouped = self.frame.loc[keep, ["season", "week"]].copy()
        grouped["value"] = values[keep]
        sums = (
            grouped.groupby(["season", "week"])
            .value.agg(["sum", "count"])
            .reindex(self.block_index, fill_value=0)
        )
        denominator = self.weights @ sums["count"].to_numpy(dtype=float)
        draws = np.divide(
            self.weights @ sums["sum"].to_numpy(dtype=float),
            denominator,
            out=np.full(BOOTSTRAPS, np.nan),
            where=denominator > 0,
        )
        draws = draws[np.isfinite(draws)]
        if not len(draws):
            return {
                "effect": None,
                "interval_low": None,
                "interval_high": None,
                "probability_positive": None,
            }
        return {
            "effect": float(values[keep].mean()),
            "interval_low": float(np.quantile(draws, 0.025)),
            "interval_high": float(np.quantile(draws, 0.975)),
            "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
            "standard_error": float(draws.std(ddof=1)),
            "draws": len(draws),
        }


def score(frame: pd.DataFrame, probabilities: dict[str, np.ndarray]) -> dict:
    bootstrap = Bootstrap(frame)
    y = frame.home_covered.to_numpy(dtype=float)
    arrays = {}
    for arm, raw in probabilities.items():
        p = np.clip(raw, 1e-9, 1 - 1e-9)
        arrays[arm] = {
            "accuracy_points": 100.0 * ((p >= 0.5) == y),
            "log_loss": -y * np.log(p) - (1 - y) * np.log1p(-p),
            "brier": (p - y) ** 2,
        }
    pooled = {
        arm: {metric: bootstrap.estimate(values) for metric, values in cells.items()}
        for arm, cells in arrays.items()
    }
    contrasts = {}
    for baseline in ("current", "model", "market"):
        contrasts[baseline] = {
            metric: bootstrap.estimate(
                (1 if metric == "accuracy_points" else -1)
                * (arrays["extended"][metric] - arrays[baseline][metric])
            )
            for metric in METRICS
        }
    gaps = {
        arm: {
            metric: bootstrap.estimate(
                (1 if metric == "accuracy_points" else -1)
                * (arrays[f"{arm}_IS"][metric] - arrays[arm][metric])
            )
            for metric in METRICS
        }
        for arm in ("current", "extended")
    }
    decisive = (probabilities["extended"] >= 0.5) != (probabilities["current"] >= 0.5)
    decisive_result = {
        "games": int(decisive.sum()),
        "extended_wins": int((arrays["extended"]["accuracy_points"][decisive] > 0).sum()),
        "current_wins": int((arrays["current"]["accuracy_points"][decisive] > 0).sum()),
        "accuracy": bootstrap.estimate(arrays["extended"]["accuracy_points"], decisive),
    }
    seasons = []
    reliability = []
    for arm in ("current", "extended", "model", "market"):
        for season in SEASONS:
            selected = frame.season.eq(season).to_numpy()
            seasons.append(
                {
                    "arm": arm,
                    "season": season,
                    "games": int(selected.sum()),
                    **{
                        metric: bootstrap.estimate(values, selected)
                        for metric, values in arrays[arm].items()
                    },
                }
            )
        p = probabilities[arm]
        for low, high in itertools.pairwise(EDGES):
            selected = (p >= low) & ((p < high) if high < 1 else (p <= high))
            reliability.append(
                {
                    "arm": arm,
                    "low": low,
                    "high": high,
                    "games": int(selected.sum()),
                    "predicted": float(p[selected].mean()) if selected.any() else None,
                    "actual": float(y[selected].mean()) if selected.any() else None,
                }
            )
    return {
        "pooled": pooled,
        "contrasts": contrasts,
        "gaps": gaps,
        "decisive": decisive_result,
        "seasons": seasons,
        "reliability": reliability,
        "blocks": len(bootstrap.blocks),
        "reporting_looks": 144,
    }


def interval(cell: dict, places: int = 4) -> str:
    if cell["effect"] is None:
        return "unavailable"
    return (
        f"{cell['effect']:.{places}f} [{cell['interval_low']:.{places}f}, "
        f"{cell['interval_high']:.{places}f}]"
    )


def render(frame: pd.DataFrame, result: dict, coefficients: list[dict], provenance: dict) -> str:
    decisive = result["decisive"]
    lines = [
        "# LEAD-73 unit 2: extended dated market moves",
        "",
        "Protocol was saved in docs/lanes/lead73.md before outcomes were loaded.",
        "**Measured:** one frozen-population comparison; 14 fits, 144 declared reporting "
        "looks; no tuning.",
        f"Both arms score the same {len(frame)} regular-season non-push opener games, 2020-2025.",
        "**Measured, decisive games first:** the extended arm won "
        f"{decisive['extended_wins']}-{decisive['current_wins']} on {decisive['games']} side "
        f"disagreements; "
        f"accuracy {interval(decisive['accuracy'], 2)}%.",
        "",
        "## Population and source alignment",
        "",
        "| Season | Fit games | Early source pairs | Added | Outside fit | Current available "
        "| Extended available |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    protocol = [
        "## Declared protocol",
        "",
        "Saved before outcomes in the lane; copied here without changing the design.",
        "Population: frozen pick_probability/20260929T192747Z/per_game.parquet; identical",
        "complete 2020-2025 regular-season rows in both arms; opener pushes excluded.",
        "Target: home covers the Tuesday opener, conditional on no push.",
        "Terms: model logit, signed move toward home, availability, composition sum, intercept;",
        "ridge 0.001, six LOSO folds with train-fold standardization; one full-data IS fit.",
        "Current retains all inputs. Extended adds only verified 2020-2022 leader moves and",
        "availability. Monday history; Wednesday-onward differences; capture strictly before",
        "min(kickoff, Sunday 12:45 ET); book update no later than capture; median book net move.",
        "The 2023-2025 inputs stay identical. Fail on timestamp, quote, sign or schema conflict.",
        "Metrics: opener accuracy, log loss and Brier against current, raw model and 0.5 market.",
        "Primary contrast: extended-minus-current accuracy. Fixed home-probability bins:",
        "0, .40, .45, .50, .55, .60, 1. No tuning, new subgroup, challenger or standalone flip.",
        "10,000 paired season-stratified week-block bootstraps; seed 20260929; 95% percentile",
        "intervals; probability_positive=P(gain>0)+0.5P(gain=0); hold predictions fixed.",
        "Look accounting: 14 fits + 12 OOS metric cells + 6 IS metric cells + 72 season metric",
        "cells + 24 reliability cells + 9 paired contrasts + 1 decisive comparison + 6 gaps",
        "= 144 reporting looks, one primary comparison; coefficients shown for every fit.",
        "No multiple-comparison adjustment; diagnostics are not independent discoveries.",
        "",
    ]
    lines[7:7] = protocol
    for row in provenance["coverage"]:
        lines.append(
            "| "
            + " | ".join(
                str(row[key])
                for key in (
                    "season",
                    "fit_games",
                    "source_games",
                    "newly_available",
                    "source_outside_fit",
                    "current_available",
                    "extended_available",
                )
            )
            + " |"
        )
    lines.extend(["", "**Measured, exclusions from the 757 source games:**", ""])
    for exclusion in provenance["exclusion_counts"]:
        lines.append(f"- {exclusion['season']}: {exclusion['games']} {exclusion['reason']}.")
    lines.extend(
        [
            "",
            "**Measured:** quote hashes, schedule team mapping, HOME/AWAY handicap sign, "
            "capture/update gates,",
            "and current-artifact move parity passed. Bookmaker home handicap is negated by "
            "the parser;",
            "a positive feature is movement toward home. Both arms retain identical "
            "late-season inputs.",
            "Source games outside the frozen model/composition population are excluded from "
            "both arms;",
            "their outcomes are not used. The frozen fit already excludes opener pushes and "
            "ungraded rows.",
            f"Verified manifests: {provenance['source_manifests']}; rejected quote rows: "
            f"{provenance['invalid_rows']};",
            f"admissible unique leader quotes: {provenance['admissible_quote_rows']}.",
            f"Current refit maximum probability error versus saved fit: "
            f"{max(provenance['refit_parity'].values()):.3g}.",
            "",
            "## Paired out-of-season results",
            "",
            "All intervals are 95% percentile intervals from 10,000 paired week-block "
            "resamples within season.",
            f"There are {result['blocks']} season/week blocks; seed {SEED}. Accuracy is "
            f"percent; gains are percentage points.",
            "| Arm | Accuracy [95% interval] | Log loss [95% interval] | Brier [95% interval] |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm in ("current", "extended", "model", "market"):
        lines.append(
            f"| {arm} | "
            + " | ".join(
                interval(result["pooled"][arm][metric], 2 if metric == "accuracy_points" else 4)
                for metric in METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Positive gain means extended is better. The market comparator is neutral 0.5 at "
            "the opener,",
            "with home selected on exact ties; this is not a tradable price or profitability "
            "estimate.",
            "",
            "| Baseline | Metric | Extended gain [95% interval] | probability_positive |",
            "|---|---|---:|---:|",
        ]
    )
    for baseline, cells in result["contrasts"].items():
        for metric, cell in cells.items():
            lines.append(
                f"| {baseline} | {metric} | {interval(cell, 6)} | "
                f"{cell['probability_positive']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## In-sample versus out-of-season",
            "",
            "Gap is optimistic IS gain: IS-OOS accuracy; OOS-IS loss. Both use the same games.",
            "",
            "| Arm | Metric | IS [95% interval] | OOS [95% interval] | Gap [95% interval] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for arm in ("current", "extended"):
        for metric in METRICS:
            lines.append(
                f"| {arm} | {metric} | {interval(result['pooled'][arm + '_IS'][metric])} | "
                f"{interval(result['pooled'][arm][metric])} | "
                f"{interval(result['gaps'][arm][metric])} |"
            )
    lines.extend(
        [
            "",
            "## Coefficients and stability",
            "",
            "Natural units; ridge 0.001; train-fold standardization. IS fits are descriptive only.",
            "",
            "| Arm | Held out | Train n | Intercept | Model logit | Move | Available | "
            "Composition |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coefficients:
        values = [
            row[key]
            for key in (
                "intercept",
                "model_logit",
                MOVE_COLUMN,
                MOVE_AVAILABLE_COLUMN,
                FLAG_SUM_COLUMN,
            )
        ]
        lines.append(
            f"| {row['arm']} | {row['held_out']} | {row['train_games']} | "
            + " | ".join(f"{value:.6f}" for value in values)
            + " |"
        )
    for arm in ("current", "extended"):
        rows = [row for row in coefficients if row["arm"] == arm and row["held_out"] != "IS"]
        values = np.array([row[MOVE_COLUMN] for row in rows])
        lines.extend(
            [
                "",
                f"**Measured:** {arm} move coefficient is positive in "
                f"{int((values > 0).sum())}/6 folds;",
                f"range {values.min():.6f} to {values.max():.6f}; mean {values.mean():.6f}; "
                f"fold SD {values.std(ddof=1):.6f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Season stability",
            "",
            "| Season | Arm | n | Accuracy [95% interval] | Log loss [95% interval] | Brier "
            "[95% interval] |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(result["seasons"], key=lambda row: (row["season"], row["arm"])):
        lines.append(
            f"| {row['season']} | {row['arm']} | {row['games']} | "
            + " | ".join(
                interval(row[metric], 2 if metric == "accuracy_points" else 4) for metric in METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Fixed reliability table",
            "",
            "Probabilities and observed rates below refer to the home side; empty bins remain "
            "reported.",
            "",
            "| Arm | Home probability bin | n | Mean probability | Home cover rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in result["reliability"]:
        values = ["—" if row[key] is None else f"{row[key]:.4f}" for key in ("predicted", "actual")]
        lines.append(
            f"| {row['arm']} | {row['low']:.2f}-{row['high']:.2f} | {row['games']} | "
            f"{' | '.join(values)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "**Inferred:** this is a source-extension estimate conditional on the frozen "
            "upstream model and",
            "composition features. LOSO keeps calibration parameters out of each scored "
            "season, but uses",
            "future seasons in training and is not a chronological deployment simulation. "
            "Week resampling",
            "holds fitted predictions fixed; it does not include full refit or "
            "source-selection uncertainty.",
            "Equal aggregation rules do not establish equal capture density across archives.",
            "No source-frequency adjustment was selected.",
            "One calibrated fitted probability chooses each side. Historical forced-pick "
            "accuracy does not",
            "establish a profitable edge or represent an individual game's probability.",
            "**Measured:** primary accuracy gain is "
            + interval(result["contrasts"]["current"]["accuracy_points"])
            + " percentage points; probability_positive="
            + f"{result['contrasts']['current']['accuracy_points']['probability_positive']:.4f}.",
            "**Inferred:** the source extension does not support replacing the current fit.",
            "For this fixed accuracy comparison, the entire interval is adverse: the proposed",
            "registry row names wrong_sign_resolved under AGENTS.md. This is a claim about",
            "the tested extension, not a rejection of book moves or pre-2023 source research.",
            "Proper-score differences versus current remain unresolved_below_power. No",
            "positive-control or split-half reliability experiment was run; no broader closure.",
            "All registry entries await the owner. No served fit was changed.",
            "",
            "## Reproduction and artifacts",
            "",
            ".tools/uv.exe run --no-sync --no-cache python scripts/lead73_unit2.py",
            "",
            "Prediction rows, quote evidence, coefficients, provenance hashes and bootstrap "
            "summaries:",
            "tests/scratch/codex/lead73_unit2/. No registry writes or served artifacts changed.",
            "The lane contains the exact owner-run record commands; --no-cache avoids the "
            "inaccessible uv cache.",
        ]
    )
    return "\n".join(lines) + "\n"


def registry_payload(result: dict, games: int) -> dict:
    cells = []
    for baseline, metrics in result["contrasts"].items():
        for metric, estimate in metrics.items():
            terminal = estimate["interval_high"] < 0
            units = metric if metric == "accuracy_points" else f"{metric}_improvement"
            plain = (
                "Adding older sportsbook line moves to the same game predictions produced "
                "fewer winning picks than the current version. Keep the current version."
                if terminal
                else "This check compares the version with older sportsbook line moves against "
                + {
                    "current": "the current picks",
                    "model": "the game model alone",
                    "market": "a fifty-fifty opener prediction",
                }[baseline]
                + ". It does not establish that older moves should change the current picks."
            )
            cells.append(
                {
                    "name": f"LEAD73-unit2-extended-vs-{baseline}-{metric}",
                    "description": f"LOSO extended versus {baseline}: positive {units} gain.",
                    "effect": estimate["effect"],
                    "effect_units": units,
                    "interval_low": estimate["interval_low"],
                    "interval_high": estimate["interval_high"],
                    "standard_error": estimate["standard_error"],
                    "probability_positive": estimate["probability_positive"],
                    "sample_games": games,
                    "sample_blocks": result["blocks"],
                    "classification": "refuted_mechanism" if terminal else "unresolved_below_power",
                    "classification_evidence": (
                        "Predeclared extension accuracy comparison; entire week-block interval "
                        "is below zero. This tests the fixed extension, not all book moves."
                        if terminal
                        else "No closure ground established for this metric; no positive "
                        "control or split-half reliability experiment was run."
                    ),
                    "closing_ground": "wrong_sign_resolved" if terminal else None,
                    "plain_summary": plain,
                }
            )
    return {
        "source": "docs/lead73_unit2.md",
        "league": "nfl",
        "season_start": 2020,
        "season_end": 2025,
        "family": "LEAD73-unit2-source-extension",
        "category": "market",
        "notes": (
            "One primary source-extension comparison; 144 reporting looks including "
            "diagnostics; 10000 season-stratified paired week-block resamples. Frozen upstream "
            "model/composition; six LOSO calibration folds; fixed-prediction uncertainty. "
            "Nine correlated contrasts, not independent evidence and not pooled across units."
        ),
        "plain_summary": (
            "Adding older sportsbook line moves did not improve the current picks. "
            "Keep the current version while the older moves remain available for research."
        ),
        "cells": cells,
    }


def main() -> int:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    moves, sources = recover_moves()
    print(f"Verified {len(sources['sources'])} manifests; recovered {len(moves)} games", flush=True)
    evidence = sources.pop("evidence")
    frame, provenance = load_population(moves)
    with threadpool_limits(limits=1):
        probabilities, coefficients, parity = fit_arms(frame)
        result = score(frame, probabilities)
    provenance.update(sources)
    provenance["source_manifests"] = len(sources["sources"])
    provenance["refit_parity"] = parity
    provenance["script_sha256"] = digest(Path(__file__))
    predictions = frame[
        [
            "game_id",
            "season",
            "week",
            "home_covered",
            "model_logit",
            FLAG_SUM_COLUMN,
            MOVE_COLUMN,
            MOVE_AVAILABLE_COLUMN,
            "extended_move",
            "extended_available",
        ]
    ].copy()
    for arm, p in probabilities.items():
        predictions[f"{arm}_probability"] = p
    predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
    evidence.to_parquet(OUTPUT / "quote_evidence.parquet", index=False)
    moves.to_parquet(OUTPUT / "recovered_moves.parquet", index=False)
    for name, data in (
        ("summary", result),
        ("coefficients", coefficients),
        ("provenance", provenance),
        ("registry_batch", registry_payload(result, len(frame))),
    ):
        (OUTPUT / f"{name}.json").write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    REPORT.write_text(render(frame, result, coefficients, provenance), encoding="utf-8")
    primary = result["contrasts"]["current"]["accuracy_points"]
    print(
        f"Games={len(frame)} added={int(frame.recovered_move.notna().sum())} "
        f"accuracy_gain={interval(primary)} "
        f"probability_positive={primary['probability_positive']:.4f}"
    )
    print(f"Decisive={result['decisive']}; report={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
