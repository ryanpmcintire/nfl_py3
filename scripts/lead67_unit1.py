from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_ats.mass_preserving_lattice import DiscretePushReader, prior_pool
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit
from nfl_ats.sharp_book_movement_features import LEADER_BOOKS

POPULATION = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z/per_game.parquet")
FEATURES = Path("data/processed/game_features_pbp.parquet")
QUOTES = Path("artifacts/sharp_book_weighted_movement/spread_quotes.parquet")
REPORT = Path("docs/lead67_report.md")
PREDICTIONS = Path("docs/lead67_predictions.md")
SEASONS = (2023, 2024, 2025)
ARMS = {
    "base": list(FIT_FEATURES),
    "replacement": [name if name != "market_move_toward_home" else "dP" for name in FIT_FEATURES],
    "additive": [*FIT_FEATURES, "dP"],
}


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def latest_lines(frame):
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "home_spread_line",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "commence_time_utc",
        "decision_label",
        "archive_season",
    ]
    quotes = pd.read_parquet(
        QUOTES,
        columns=columns,
        use_threads=False,
        filters=[
            ("decision_label", "==", "intraday_hourly"),
            ("archive_season", "in", list(SEASONS)),
            ("nflverse_game_id", "in", frame.game_id.tolist()),
        ],
    ).rename(columns={"nflverse_game_id": "game_id"})
    for column in ("observed_at_utc", "bookmaker_last_update_utc", "commence_time_utc"):
        quotes[column] = pd.to_datetime(quotes[column], utc=True, errors="raise")
    games = quotes.groupby("game_id", as_index=False).agg(kickoff=("commence_time_utc", "min"))
    games = frame[["game_id", "season", "week"]].merge(games, validate="one_to_one")
    anchor = (
        games.groupby(["season", "week"]).kickoff.transform("min").dt.tz_convert("America/New_York")
    )
    sunday = anchor.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(
        (6 - anchor.dt.weekday) % 7, unit="D"
    )
    games["monday"] = (
        (sunday - pd.Timedelta(days=6)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    )
    deadline = (
        (sunday + pd.Timedelta(hours=12, minutes=45))
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("UTC")
    )
    games["cutoff"] = pd.concat([games.kickoff, deadline], axis=1).min(axis=1)
    quotes = quotes.merge(games[["game_id", "monday", "cutoff"]], validate="many_to_one")
    quotes = quotes.loc[
        quotes.bookmaker_key.isin(LEADER_BOOKS)
        & quotes.observed_at_utc.ge(quotes.monday)
        & quotes.observed_at_utc.lt(quotes.cutoff)
        & quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc)
        & np.isfinite(quotes.home_spread_line)
    ].copy()
    keys = ["game_id", "bookmaker_key", "observed_at_utc"]
    if quotes.groupby(keys).home_spread_line.nunique().gt(1).any():
        raise ValueError("Conflicting leader quotes")
    last = quotes.sort_values(keys).drop_duplicates(["game_id", "bookmaker_key"], keep="last")
    lines = last.groupby("game_id", as_index=False).agg(
        late_line=("home_spread_line", "median"),
        leader_books=("bookmaker_key", "size"),
        latest_quote=("observed_at_utc", "max"),
    )
    result = frame.merge(games[["game_id", "cutoff"]], how="left", validate="one_to_one")
    result = result.merge(lines, how="left", validate="one_to_one")
    if result.late_line.isna().any() or not result.latest_quote.lt(result.cutoff).all():
        raise ValueError("A declared game lacks a dated pre-deadline leader median")
    result["late_minus_open"] = result.late_line - result.tue_open_home_spread
    return result


def lattice_feature(frame, pool, held):
    result = frame.copy()
    result["dP"] = np.nan
    result["lattice_prior_games"] = 0
    for season, indexes in result.groupby("season").groups.items():
        eligible = pool.loc[pool.season.lt(season) & pool.season.ne(held)].copy()
        for index in indexes:
            row = result.loc[index]
            cutoff = pd.Timestamp(row.cutoff).tz_convert("UTC").tz_localize(None)
            reader = DiscretePushReader.for_week(
                eligible,
                season=int(season),
                week=int(row.week),
                cutoff=cutoff,
                exclude_game_ids=(str(row.game_id),),
            )
            opener, late = float(row.tue_open_home_spread), float(row.late_line)
            at_open = reader.read(opener, late, conditioning_line=late)
            at_late = reader.read(late, late, conditioning_line=late)
            result.loc[index, "dP"] = (
                at_open.home_cover_probability - at_late.home_cover_probability
            )
            result.loc[index, "lattice_prior_games"] = reader.prior_rows
    if not np.isfinite(result.dP).all():
        raise ValueError("Nonfinite probability-valued move")
    if (result.dP * result.late_minus_open < -1e-12).any():
        raise ValueError("Probability move orientation conflicts with quoted movement")
    return result


def fit(train, test, terms):
    x = train[terms].to_numpy(dtype=float)
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales[scales < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(train)), (x - means) / scales])
    beta = _fit_logit(design, train.home_covered.to_numpy(dtype=float), FIT_RIDGE)
    test_design = np.column_stack(
        [np.ones(len(test)), (test[terms].to_numpy(dtype=float) - means) / scales]
    )
    p = 1.0 / (1.0 + np.exp(-np.clip(test_design @ beta, -35, 35)))
    natural = dict(zip(terms, (beta[1:] / scales).tolist(), strict=True))
    natural["intercept"] = float(beta[0] - np.sum(beta[1:] * means / scales))
    return p, natural


def values(frame, probability):
    y = frame.home_covered.to_numpy(dtype=float)
    p = np.clip(probability, 1e-9, 1 - 1e-9)
    home = p >= 0.5
    return {
        "accuracy": (home == y).astype(float),
        "brier": (p - y) ** 2,
        "log_loss": -y * np.log(p) - (1 - y) * np.log1p(-p),
        "line_move": np.where(home, 1.0, -1.0)
        * (frame.close_home_spread - frame.tue_open_home_spread).to_numpy(dtype=float),
    }


def interval(frame, delta, block):
    data = frame[block].copy()
    data["delta"] = delta
    grouped = data.groupby(block).delta.agg(["sum", "count"])
    indexes = np.random.default_rng(6701).integers(0, len(grouped), size=(10000, len(grouped)))
    draws = grouped["sum"].to_numpy()[indexes].sum(axis=1) / grouped["count"].to_numpy()[
        indexes
    ].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "effect": float(delta.mean()),
        "low": float(low),
        "high": float(high),
        "probability_positive": float(
            (draws > 1e-12).mean() + 0.5 * (np.abs(draws) <= 1e-12).mean()
        ),
        "standard_error": float(draws.std(ddof=1)),
        "blocks": len(grouped),
    }


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        ]
    )


def main():
    sources = [POPULATION, OPENER, FEATURES, QUOTES]
    absent = [str(path) for path in sources if not path.is_file()]
    if absent:
        REPORT.write_text(
            "# LEAD-67 inventory\n\n**Measured:** missing local inputs: "
            + ", ".join(absent)
            + ". No outcomes computed.\n",
            encoding="utf-8",
        )
        raise SystemExit("Required local source absent; inventory saved")
    hashes = {str(path): fingerprint(path) for path in sources}
    source = pd.read_parquet(POPULATION, use_threads=False)
    frame = source.loc[source.season.isin(SEASONS) & source.market_move_available.eq(1)].copy()
    frame = frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    if len(frame) != 799 or frame.game_id.duplicated().any():
        raise ValueError("The declared unique 799-game population was not reproduced")
    if frame.margin_vs_open.eq(0).any() or not np.array_equal(
        frame.home_covered, frame.margin_vs_open.gt(0).astype(float)
    ):
        raise ValueError("The frozen-opener target or push exclusion differs")
    needed = [*FIT_FEATURES, "model_probability", "close_home_spread", "tue_open_home_spread"]
    if not np.isfinite(frame[needed].to_numpy(dtype=float)).all():
        raise ValueError("Missing declared predictors or grading lines")
    frame = latest_lines(frame)
    features = pd.read_parquet(
        FEATURES,
        columns=["game_id", "season", "week", "game_type", "gameday", "spread_line", "result"],
        use_threads=False,
    )
    openers = pd.read_parquet(
        OPENER, columns=["game_id", "tue_open_home_spread"], use_threads=False
    )
    pool = prior_pool(features, openers.set_index("game_id").tue_open_home_spread)
    coefficients = []
    all_features = lattice_feature(frame, pool, None)
    frame["dP_in_sample"] = all_features.dP
    for arm, terms in ARMS.items():
        frame[f"{arm}_is"], natural = fit(all_features, all_features, terms)
        coefficients.append({"arm": arm, "fold": "in_sample", **natural})
        frame[f"{arm}_oos"] = np.nan
    for held in SEASONS:
        fold = lattice_feature(frame, pool, held)
        train, test = fold.loc[fold.season.ne(held)], fold.loc[fold.season.eq(held)]
        frame.loc[test.index, "dP_oos"] = test.dP
        frame.loc[test.index, "lattice_prior_games"] = test.lattice_prior_games
        for arm, terms in ARMS.items():
            predicted, natural = fit(train, test, terms)
            frame.loc[test.index, f"{arm}_oos"] = predicted
            coefficients.append({"arm": arm, "fold": str(held), **natural})
    for protocol in ("is", "oos"):
        frame[f"model_{protocol}"] = frame.model_probability
        frame[f"market_{protocol}"] = 0.5
    metrics = {
        (arm, protocol): values(frame, frame[f"{arm}_{protocol}"].to_numpy())
        for arm in (*ARMS, "model", "market")
        for protocol in ("is", "oos")
    }
    decisive, comparisons, registry = [], [], []
    for arm in ("replacement", "additive"):
        delta = metrics[arm, "oos"]["accuracy"] - metrics["base", "oos"]["accuracy"]
        decisive.append(
            [arm, int((delta > 0).sum()), int((delta < 0).sum()), int((delta != 0).sum())]
        )
        for metric in ("accuracy", "brier", "log_loss", "line_move"):
            sign = -1 if metric in ("brier", "log_loss") else 1
            change = sign * (metrics[arm, "oos"][metric] - metrics["base", "oos"][metric])
            insample = sign * (metrics[arm, "is"][metric] - metrics["base", "is"][metric]).mean()
            for label, blocks in (("season", ["season"]), ("season-week", ["season", "week"])):
                result = interval(frame, change, blocks)
                comparisons.append(
                    [
                        arm,
                        metric,
                        label,
                        f"{result['effect']:.6f}",
                        f"[{result['low']:.6f}, {result['high']:.6f}]",
                        f"{result['probability_positive']:.4f}",
                        f"{insample:.6f}",
                        f"{insample - result['effect']:.6f}",
                    ]
                )
                if metric == "accuracy" and label == "season":
                    registry.append({"arm": arm, **result})
    rows, folds, reliability = [], [], []
    for arm in (*ARMS, "model", "market"):
        for protocol in ("is", "oos"):
            m = metrics[arm, protocol]
            rows.append(
                [
                    arm,
                    protocol,
                    f"{int(m['accuracy'].sum())}/799",
                    *(
                        f"{m[name].mean():.6f}"
                        for name in ("accuracy", "brier", "log_loss", "line_move")
                    ),
                ]
            )
        p = frame[f"{arm}_oos"].to_numpy()
        bins = np.minimum((p * 10).astype(int), 9)
        for band in range(10):
            chosen = bins == band
            if chosen.any():
                reliability.append(
                    [
                        arm,
                        f"{band / 10:.1f}-{(band + 1) / 10:.1f}",
                        int(chosen.sum()),
                        f"{p[chosen].mean():.6f}",
                        f"{frame.home_covered.to_numpy()[chosen].mean():.6f}",
                    ]
                )
    for season in SEASONS:
        chosen = frame.season.eq(season).to_numpy()
        for arm in ARMS:
            m = metrics[arm, "oos"]
            delta = m["accuracy"][chosen] - metrics["base", "oos"]["accuracy"][chosen]
            folds.append(
                [
                    season,
                    arm,
                    int(chosen.sum()),
                    int(m["accuracy"][chosen].sum()),
                    int((delta > 0).sum()),
                    int((delta < 0).sum()),
                    f"{m['brier'][chosen].mean():.6f}",
                    f"{m['log_loss'][chosen].mean():.6f}",
                ]
            )
    terms = ["intercept", *FIT_FEATURES, "dP"]
    coefficient_rows = [
        [
            item["arm"],
            item["fold"],
            *(f"{item[term]:.6f}" if term in item else "absent" for term in terms),
        ]
        for item in coefficients
    ]
    stability = []
    for arm in ARMS:
        for term in terms:
            vals = [
                item[term]
                for item in coefficients
                if item["arm"] == arm and item["fold"] != "in_sample" and term in item
            ]
            if vals:
                stability.append(
                    [
                        arm,
                        term,
                        f"{min(vals):.6f}",
                        f"{max(vals):.6f}",
                        int(sum(v > 1e-12 for v in vals)),
                        int(sum(v < -1e-12 for v in vals)),
                    ]
                )
    if hashes != {str(path): fingerprint(path) for path in sources}:
        raise ValueError("A source changed during the run")
    text = [
        "# LEAD-67 unit 1",
        "",
        "**Measured:** 799 opener-graded games; seasons 2023/2024/2025 = 266/266/267. "
        "Two predeclared experimental looks: replacement and additive. All results are offline.",
        "",
        "## Decisive-game record first",
        "",
        table(["Arm vs base", "Wins", "Losses", "Disagreements"], decisive),
        "",
        "## Metrics",
        "",
        "**Measured:** IS fits all 799 rows; OOS holds each season out. Accuracy is a fraction; "
        "line move is points toward the selected side. A 0.5 market tie selects home solely for "
        "the mandatory accuracy/movement benchmark; it has no directional edge.",
        "",
        table(["Arm", "Protocol", "Correct", "Accuracy", "Brier", "Log loss", "Line move"], rows),
        "",
        "## Paired gains, intervals, and in-sample optimism",
        "",
        "**Measured:** positive effects favor the challenger. Gap = IS gain minus OOS gain. "
        "Season blocks are primary; season-week blocks are sensitivity. 10,000 draws, seed 6701; "
        "95% percentile intervals; zero draws receive half credit in probability_positive.",
        "",
        table(
            [
                "Arm",
                "Metric",
                "Block",
                "OOS gain",
                "95% interval",
                "probability_positive",
                "IS gain",
                "IS-OOS gap",
            ],
            comparisons,
        ),
        "",
        "## Season stability",
        "",
        table(
            ["Season", "Arm", "N", "Correct", "Decisive W", "Decisive L", "Brier", "Log loss"],
            folds,
        ),
        "",
        "## Natural coefficients",
        "",
        "**Measured:** availability is constant one; its standardized coefficient is zero.",
        "",
        table(["Arm", "Fold", *terms], coefficient_rows),
        "",
        table(
            ["Arm", "Term", "LOSO min", "LOSO max", "Positive folds", "Negative folds"], stability
        ),
        "",
        "## Reliability",
        "",
        "**Measured:** fixed 0.1-wide home-probability bins; all nonempty bins shown. "
        "No bin outcome selected a model.",
        "",
        table(["Arm", "Bin", "N", "Mean predicted", "Observed home cover"], reliability),
        "",
        "## Interpretation and limits",
        "",
        "**Inferred:** neither serving nor research closure is authorized by this unit. Candidate "
        "registry status is unresolved_below_power pending orchestrator serial registration and "
        "adjudication under AGENTS.md:65-105. No positive-control or split-half reliability study "
        "was run. Only three season blocks support the primary interval.",
        "",
        "**Read:** the frozen model probabilities retain their walk-forward production lineage. "
        "LOSO combining coefficients are retrospective; this is not a prospective deployment test. "
        "Lattice priors exclude the held season and every row's current/future seasons. "
        "The endpoint "
        "is the actual leader-median quote; the base input sums within-book moves. Consequently "
        "this declared endpoint study includes the frozen-pool/leader basis difference as well as "
        "key-number pricing and cannot attribute any gain solely to units.",
        "",
        "**Measured:** all 799 latest quote times precede min(kickoff, Sunday 12:45 Eastern). "
        f"LOSO lattice prior rows range {int(frame.lattice_prior_games.min())}-"
        f"{int(frame.lattice_prior_games.max())}. Close lines enter the yardstick only, never "
        "predictors. The yardstick overlaps the movement input and is not independent evidence.",
        "",
        "## Reproduction and local prediction output",
        "",
        "Command: .tools/uv.exe run --no-sync python scripts/lead67_unit1.py "
        "(UV_NO_CACHE=1 avoids an inaccessible shared uv cache). No network or pipeline rebuild. "
        "Prediction rows: docs/lead67_predictions.md (local research artifact; do not commit).",
        "",
        "Protocol: docs/lead67_protocol.md; lane: docs/lanes/lead67.md.",
        "",
        table(["Read-only source", "SHA256"], [[name, sha] for name, sha in hashes.items()]),
        "",
        "## Registry estimates for the orchestrator",
        "",
        json.dumps(registry, indent=2),
        "",
    ]
    REPORT.write_text("\n".join(text), encoding="utf-8")
    keep = [
        "game_id",
        "season",
        "week",
        "home_covered",
        "tue_open_home_spread",
        "late_line",
        "latest_quote",
        "cutoff",
        "leader_books",
        "market_move_toward_home",
        "dP_in_sample",
        "dP_oos",
        "lattice_prior_games",
        "model_probability",
        "base_is",
        "base_oos",
        "replacement_is",
        "replacement_oos",
        "additive_is",
        "additive_oos",
    ]
    prediction_rows = [
        [f"{value:.10g}" if isinstance(value, float) else str(value) for value in row]
        for row in frame[keep].itertuples(index=False, name=None)
    ]
    PREDICTIONS.write_text(
        "# LEAD-67 local prediction artifact - do not commit\n\n"
        + table(keep, prediction_rows)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "games": len(frame),
                "looks": 2,
                "decisive": decisive,
                "accuracy_gain": registry,
                "report": str(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
