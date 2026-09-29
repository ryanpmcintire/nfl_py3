from __future__ import annotations

import hashlib
import json
import os
import shlex
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from quote_provenance_check import check_snapshot, validate_frame
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

ROOT = Path("artifacts/pick_probability/20260929T192747Z")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
OUT = Path("tests/scratch/codex/lead74_unit2")
REPORT = Path("docs/lead74_unit2.md")
LANE = Path("docs/lanes/lead74.md")
SEASONS = (2023, 2024, 2025)
ARMS = ("combined", "four_term", "model", "market")
METRICS = ("accuracy_points", "log_loss", "brier")
BOOTSTRAPS = 10000
SEED = 20260929


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parquet(path, columns=None):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def schedules():
    games = parquet(
        SCHEDULE,
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "gametime",
            "home_team",
            "away_team",
        ],
    )
    games = games.loc[games.season.isin(SEASONS) & games.game_type.eq("REG")].copy()
    games["gameday"] = pd.to_datetime(games.gameday)
    local = pd.to_datetime(games.gameday.dt.strftime("%Y-%m-%d") + " " + games.gametime)
    games["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    anchor = games.groupby(["season", "week"]).gameday.transform("min")
    sunday = anchor + pd.to_timedelta((6 - anchor.dt.dayofweek) % 7, unit="D")
    for name, delta in {
        "tuesday": pd.Timedelta(days=-5),
        "wednesday": pd.Timedelta(days=-4),
        "deadline": pd.Timedelta(hours=12, minutes=45),
    }.items():
        games[name] = (sunday + delta).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    games["deadline"] = games[["deadline", "kickoff"]].min(axis=1)
    if games.game_id.duplicated().any() or games.kickoff.isna().any():
        raise ValueError("Schedule is incomplete or duplicated")
    return games


def pressure(games):
    frames, hashes = [], []
    audit = Counter()
    for path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        if (
            manifest.get("capture_kind") != "historical_backfill"
            or request.get("sport") != "americanfootball_nfl"
            or request.get("season") not in SEASONS
            or "spreads" not in str(request.get("markets", "")).split(",")
        ):
            continue
        audit.update(check_snapshot(path.parent, manifest, None))
        quotes = parquet(path.with_name("quotes.parquet"))
        quotes = quotes.loc[
            quotes.market.eq("spreads") & quotes.nflverse_game_id.isin(games.game_id)
        ].copy()
        quotes = validate_frame(quotes, manifest)
        if not quotes.clock_consistent.all():
            raise ValueError(f"Inconsistent quote clocks: {path}")
        audit["snapshots"] += 1
        hashes.append({"path": str(path), "sha256": digest(path)})
        frames.append(
            quotes.loc[
                quotes.asof_consistent,
                [
                    "nflverse_game_id",
                    "provider_event_id",
                    "bookmaker_key",
                    "observed_at_utc",
                    "commence_time_utc",
                    "home_team",
                    "away_team",
                    "outcome_side",
                    "line",
                    "price",
                ],
            ]
        )
        if audit["snapshots"] % 1000 == 0:
            print(f"Audited {audit['snapshots']} snapshots", flush=True)
    if not frames:
        raise ValueError("No quote snapshots")
    q = pd.concat(frames, ignore_index=True).drop_duplicates()
    q = q.rename(columns={"nflverse_game_id": "game_id"})
    q["observed_at_utc"] = pd.to_datetime(q.observed_at_utc, utc=True)
    q["commence_time_utc"] = pd.to_datetime(q.commence_time_utc, utc=True)
    q = q.merge(games, on="game_id", suffixes=("", "_schedule"), validate="many_to_one")
    if not (q.home_team.eq(q.home_team_schedule) & q.away_team.eq(q.away_team_schedule)).all():
        raise ValueError("Quote/schedule team disagreement")
    q = q.loc[
        q.observed_at_utc.ge(q.tuesday)
        & q.observed_at_utc.lt(q.deadline)
        & q.observed_at_utc.lt(q.commence_time_utc)
        & q.outcome_side.isin(["HOME", "AWAY"])
        & np.isfinite(q.price)
        & q.price.abs().ge(100)
        & np.isfinite(q.line)
    ].copy()
    keys = ["game_id", "provider_event_id", "bookmaker_key", "observed_at_utc"]
    duplicates = q.duplicated([*keys, "outcome_side"], keep=False)
    audit["duplicate_side_rows_excluded"] = int(duplicates.sum())
    q = q.loc[~duplicates]
    home = q.loc[q.outcome_side.eq("HOME")]
    away = q.loc[q.outcome_side.eq("AWAY"), [*keys, "line", "price"]]
    pairs = home.merge(away, on=keys, suffixes=("", "_away"), validate="one_to_one")
    pairs = pairs.loc[np.isclose(pairs.line + pairs.line_away, 0, atol=1e-9, rtol=0)].copy()
    for side in ("", "_away"):
        price = pairs[f"price{side}"].to_numpy(dtype=float)
        implied = np.empty(len(price))
        positive = price > 0
        implied[positive] = 100 / (price[positive] + 100)
        implied[~positive] = -price[~positive] / (100 - price[~positive])
        pairs[f"implied{side}"] = implied
    pairs["price_logit"] = np.log(pairs.implied / pairs.implied_away)
    book_keys = keys[:-1]
    pairs = pairs.sort_values("observed_at_utc")
    tuesday = pairs.loc[pairs.observed_at_utc.lt(pairs.wednesday)].drop_duplicates(
        book_keys, keep="last"
    )
    deadline = pairs.loc[pairs.observed_at_utc.ge(pairs.wednesday)].drop_duplicates(
        book_keys, keep="last"
    )
    endpoints = tuesday[[*book_keys, "observed_at_utc", "line", "price_logit"]].merge(
        deadline[[*book_keys, "observed_at_utc", "line", "price_logit"]],
        on=book_keys,
        suffixes=("_tuesday", "_deadline"),
        validate="one_to_one",
    )
    audit["book_endpoint_pairs"] = len(endpoints)
    matched = endpoints.loc[
        np.isclose(
            endpoints.line_tuesday,
            endpoints.line_deadline,
            atol=1e-9,
            rtol=0,
        )
    ].copy()
    if matched.duplicated(["game_id", "bookmaker_key"]).any():
        raise ValueError("Multiple provider events for one game/book")
    matched["pressure"] = matched.price_logit_deadline - matched.price_logit_tuesday
    matched.to_parquet(OUT / "book_pairs.parquet", index=False)
    feature = (
        matched.groupby("game_id")
        .agg(
            pressure=("pressure", "median"),
            matched_books=("bookmaker_key", "size"),
        )
        .reset_index()
    )
    audit["matched_book_pairs"] = len(matched)
    audit["source_games"] = len(feature)
    return feature, {"counts": dict(audit), "manifests": hashes}


def population(feature):
    columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "home_covered",
        "margin_vs_open",
        "model_probability",
        "base_probability_policy",
        "residual_at_open_served",
        *FIT_FEATURES,
    ]
    frozen = parquet(ROOT / "per_game.parquet", columns)
    if (
        frozen.game_id.duplicated().any()
        or not frozen.season.between(2020, 2025).all()
        or not frozen.game_type.eq("REG").all()
        or frozen.margin_vs_open.eq(0).any()
        or not frozen.home_covered.eq(frozen.margin_vs_open.gt(0)).all()
        or not frozen.base_probability_policy.eq("discrete_conditional_non_push_v1").all()
    ):
        raise ValueError("Frozen population/target/discrete probability conflict")
    if not np.allclose(
        frozen.model_logit, np.log(frozen.model_probability / (1 - frozen.model_probability))
    ):
        raise ValueError("Model probability/logit conflict")
    frame = frozen.merge(feature, on="game_id", validate="one_to_one")
    numeric = [
        *FIT_FEATURES,
        "pressure",
        "model_probability",
        "home_covered",
        "margin_vs_open",
        "residual_at_open_served",
    ]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Incomplete study rows")
    coverage = []
    for season in SEASONS:
        rows = frame.loc[frame.season.eq(season)]
        coverage.append(
            {
                "season": season,
                "frozen_games": int(frozen.season.eq(season).sum()),
                "paired_games": len(rows),
                "books": int(rows.matched_books.sum()),
                "nonzero_pressure": int(rows.pressure.ne(0).sum()),
            }
        )
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True), coverage


def fit(train, test, terms):
    means = train[terms].mean().to_numpy()
    stds = train[terms].std(ddof=0).replace(0, 1).to_numpy()
    x = np.column_stack([np.ones(len(train)), (train[terms].to_numpy() - means) / stds])
    z = np.column_stack([np.ones(len(test)), (test[terms].to_numpy() - means) / stds])
    beta = _fit_logit(x, train.home_covered.to_numpy(dtype=float), FIT_RIDGE)
    natural = beta[1:] / stds
    coefficients = {
        "intercept": float(beta[0] - np.dot(natural, means)),
        **dict(zip(terms, natural, strict=True)),
    }
    return (
        1 / (1 + np.exp(-np.clip(z @ beta, -35, 35))),
        1 / (1 + np.exp(-np.clip(x @ beta, -35, 35))),
        coefficients,
    )


def losses(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "accuracy_points": 100 * ((p >= 0.5) == y),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
    }


def replay(frame):
    coefficients, fold_metrics, unavailable = [], [], {}
    for arm in ARMS:
        frame[f"p_{arm}"] = np.nan
        frame[f"is_{arm}"] = np.nan
        frame[f"band_{arm}"] = -1
    for held in (*SEASONS, "IS"):
        train = frame if held == "IS" else frame.loc[frame.season.lt(held)]
        select = (
            np.ones(len(frame), dtype=bool) if held == "IS" else frame.season.eq(held).to_numpy()
        )
        test = frame.loc[select]
        if len(train) < 2 or train.pressure.nunique() < 2 or train.home_covered.nunique() < 2:
            unavailable[str(held)] = (
                "No earlier covered season with identifiable pressure and both targets"
            )
            continue
        for arm in ARMS:
            if arm in ("combined", "four_term"):
                terms = [*FIT_FEATURES, *(["pressure"] if arm == "combined" else [])]
                p, train_p, beta = fit(train, test, terms)
                coefficients.append(
                    {
                        "arm": arm,
                        "fold": str(held),
                        "train_games": len(train),
                        "test_games": len(test),
                        **beta,
                    }
                )
            else:
                p = test.model_probability.to_numpy() if arm == "model" else np.full(len(test), 0.5)
                train_p = (
                    train.model_probability.to_numpy()
                    if arm == "model"
                    else np.full(len(train), 0.5)
                )
            frame.loc[select, f"{'is' if held == 'IS' else 'p'}_{arm}"] = p
            if held == "IS":
                continue
            edges = np.quantile(train_p, [0.2, 0.4, 0.6, 0.8])
            frame.loc[select, f"band_{arm}"] = np.searchsorted(edges, p, side="right")
            train_loss = losses(train.home_covered.to_numpy(), train_p)
            test_loss = losses(test.home_covered.to_numpy(), p)
            for metric in METRICS:
                fold_metrics.append(
                    {
                        "fold": held,
                        "arm": arm,
                        "metric": metric,
                        "train_games": len(train),
                        "test_games": len(test),
                        "IS": float(train_loss[metric].mean()),
                        "OOS": float(test_loss[metric].mean()),
                        "gap_OOS_minus_IS": float(
                            test_loss[metric].mean() - train_loss[metric].mean()
                        ),
                    }
                )
    return frame, coefficients, fold_metrics, unavailable


class Bootstrap:
    def __init__(self, frame):
        self.frame = frame
        self.blocks = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        self.index = pd.MultiIndex.from_frame(self.blocks)
        self.weights = np.zeros((BOOTSTRAPS, len(self.blocks)))
        rng = np.random.default_rng(SEED)
        for season in sorted(self.blocks.season.unique()):
            positions = np.flatnonzero(self.blocks.season.eq(season))
            self.weights[:, positions] = rng.multinomial(
                len(positions), np.full(len(positions), 1 / len(positions)), size=BOOTSTRAPS
            )

    def estimate(self, values, select=None):
        keep = np.ones(len(self.frame), dtype=bool) if select is None else select
        if not keep.any():
            return {
                "effect": None,
                "low": None,
                "high": None,
                "probability_positive": None,
                "se": None,
            }
        grouped = self.frame.loc[keep, ["season", "week"]].copy()
        grouped["value"] = values[keep]
        summary = (
            grouped.groupby(["season", "week"])
            .value.agg(["sum", "count"])
            .reindex(self.index, fill_value=0)
        )
        denominator = self.weights @ summary["count"].to_numpy()
        numerator = self.weights @ summary["sum"].to_numpy()
        draws = numerator[denominator > 0] / denominator[denominator > 0]
        return {
            "effect": float(values[keep].mean()),
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
            "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
            "se": float(draws.std(ddof=1)),
        }


def score(frame):
    frame = frame.loc[frame.p_combined.notna()].copy()
    if frame.empty:
        raise ValueError("No eligible chronological held-out rows")
    boot = Bootstrap(frame)
    y = frame.home_covered.to_numpy()
    arrays = {arm: losses(y, frame[f"p_{arm}"].to_numpy()) for arm in ARMS}
    results = {
        "games": len(frame),
        "blocks": len(boot.blocks),
        "metrics": {},
        "contrasts": {},
        "gaps": {},
        "seasons": {},
        "reliability": [],
    }
    for arm in ARMS:
        results["metrics"][arm] = {
            metric: boot.estimate(values) for metric, values in arrays[arm].items()
        }
        insample = losses(y, frame[f"is_{arm}"].to_numpy())
        results["gaps"][arm] = {
            metric: {
                "IS": float(insample[metric].mean()),
                "OOS_minus_IS": boot.estimate(arrays[arm][metric] - insample[metric]),
            }
            for metric in METRICS
        }
        for band in range(5):
            selected = frame.loc[frame[f"band_{arm}"].eq(band)]
            results["reliability"].append(
                {
                    "arm": arm,
                    "band": band + 1,
                    "n": len(selected),
                    "prediction": None if selected.empty else float(selected[f"p_{arm}"].mean()),
                    "observed": None if selected.empty else float(selected.home_covered.mean()),
                }
            )
    for comparator in ARMS[1:]:
        results["contrasts"][comparator] = {
            metric: boot.estimate(
                (1 if metric == "accuracy_points" else -1)
                * (arrays["combined"][metric] - arrays[comparator][metric])
            )
            for metric in METRICS
        }
    for season in sorted(frame.season.unique()):
        select = frame.season.eq(season).to_numpy()
        results["seasons"][str(season)] = {
            "games": int(select.sum()),
            "metrics": {
                arm: {metric: boot.estimate(arrays[arm][metric], select) for metric in METRICS}
                for arm in ARMS
            },
            "gains": {
                metric: boot.estimate(
                    (1 if metric == "accuracy_points" else -1)
                    * (arrays["combined"][metric] - arrays["four_term"][metric]),
                    select,
                )
                for metric in METRICS
            },
        }
    decisive = (frame.p_combined.to_numpy() >= 0.5) != (frame.p_four_term.to_numpy() >= 0.5)
    wins = int((arrays["combined"]["accuracy_points"][decisive] / 100).sum())
    results["decisive"] = {
        "n": int(decisive.sum()),
        "wins": wins,
        "losses": int(decisive.sum()) - wins,
        "accuracy": boot.estimate(arrays["combined"]["accuracy_points"], decisive),
        "advantage": boot.estimate(arrays["combined"]["accuracy_points"] - 50, decisive),
    }
    results["margin_mae"] = boot.estimate(
        np.abs(frame.margin_vs_open.to_numpy() - frame.residual_at_open_served.to_numpy())
    )
    return results


def interval(value, digits=4):
    if value["effect"] is None:
        return "unavailable"
    return f"{value['effect']:.{digits}f} [{value['low']:.{digits}f}, {value['high']:.{digits}f}]"


def report(result, coverage, coefficients, folds, unavailable, audit, declaration):
    decisive = result["decisive"]
    lines = [
        "# LEAD-74 unit 2: price changes at unchanged spreads",
        "",
        ("**Measured:** one replay; historical opener is the frozen pool-line proxy;"),
        ("all findings are conditional on provider observed_at_utc being availability."),
        (
            f"**Measured, decisive games first:** combined "
            f"{decisive['wins']}-{decisive['losses']} on {decisive['n']} "
            f"disagreements;"
        ),
        (
            f"accuracy {interval(decisive['accuracy'], 2)}%; "
            f"probability_positive versus 50% = "
            f"{decisive['advantage']['probability_positive']}."
        ),
        "",
        "## Frozen declaration",
        "",
        declaration,
        "",
        "## Coverage and source audit",
        "",
        ("| Season | Frozen non-push games | Paired games | Matched books | Nonzero pressure |"),
        "|---|---:|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['season']} | {row['frozen_games']} | "
            f"{row['paired_games']} | {row['books']} | "
            f"{row['nonzero_pressure']} |"
        )
    counts = audit["counts"]
    lines.extend(
        [
            "",
            (
                f"**Measured:** {counts['snapshots']} snapshots passed hash, "
                f"identity and clock checks;"
            ),
            (
                f"{counts['book_endpoint_pairs']} same-book endpoint pairs, "
                f"{counts['matched_book_pairs']} unchanged-spread pairs,"
            ),
            (
                f"{counts['source_games']} source games; "
                f"{counts['duplicate_side_rows_excluded']} duplicate-side rows "
                f"excluded."
            ),
            (f"OOS population: {result['games']} games and {result['blocks']} season/week blocks."),
            f"Unavailable folds: {json.dumps(unavailable)}.",
            (
                "**Read:** the frozen probability artifact has no cached Elo "
                "probability; Elo cells remain unavailable."
            ),
            (
                "**Measured:** no games or books were selected by outcome. "
                "Snapshot availability_proven remains false"
            ),
            (
                "under the strict receipt validator; this unit uses "
                "asof_consistent under the root's conditional rule."
            ),
            "",
            "## Paired chronological out-of-season results",
            "",
            (
                "**Measured:** 95% percentile intervals from 10,000 paired "
                "week-block resamples within season."
            ),
            (
                "Accuracy is percent; accuracy gains are percentage points. "
                "Positive gains always favor combined."
            ),
            (
                "Market 0.5 ties choose home consistently; its accuracy is a "
                "tie-convention diagnostic."
            ),
            "",
            ("| Arm | Accuracy [95% interval] | Log loss [95% interval] | Brier [95% interval] |"),
            "|---|---:|---:|---:|",
        ]
    )
    for arm, metrics in result["metrics"].items():
        lines.append(
            f"| {arm} | {interval(metrics['accuracy_points'], 2)} | "
            f"{interval(metrics['log_loss'])} | "
            f"{interval(metrics['brier'])} |"
        )
    lines.extend(
        [
            "",
            ("| Comparator | Metric gain | Estimate [95% interval] | probability_positive |"),
            "|---|---|---:|---:|",
        ]
    )
    for arm, metrics in result["contrasts"].items():
        for metric, cell in metrics.items():
            lines.append(
                f"| {arm} | {metric} | {interval(cell)} | {cell['probability_positive']:.4f} |"
            )
    lines.extend(
        [
            "",
            (
                f"**Measured:** unchanged cached model margin MAE "
                f"{interval(result['margin_mae'])} points."
            ),
            (
                "Cover recalibration supplies no new expected margin; no "
                "challenger margin gain is claimed."
            ),
            "",
            "## In-sample/out-of-season gaps",
            "",
            ("**Measured:** full-population IS diagnostics below use the same OOS rows."),
            "Gap is OOS minus IS. All-season IS fits never select held-out picks.",
            "",
            "| Arm | Metric | IS | OOS minus IS [95% interval] |",
            "|---|---|---:|---:|",
        ]
    )
    for arm, metrics in result["gaps"].items():
        for metric, cell in metrics.items():
            lines.append(
                f"| {arm} | {metric} | {cell['IS']:.6f} | {interval(cell['OOS_minus_IS'], 6)} |"
            )
    lines.extend(
        [
            "",
            "Each fold's training IS and held-out OOS metrics:",
            "",
            "| Fold | Arm | Metric | Train/test | IS | OOS | Gap |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in folds:
        lines.append(
            f"| {row['fold']} | {row['arm']} | {row['metric']} | "
            f"{row['train_games']}/{row['test_games']} | {row['IS']:.6f} | "
            f"{row['OOS']:.6f} | {row['gap_OOS_minus_IS']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Season stability",
            "",
            (
                "| Season | Games | Metric gain versus four-term | Estimate "
                "[95% interval] | probability_positive |"
            ),
            "|---|---:|---|---:|---:|",
        ]
    )
    for season, values in result["seasons"].items():
        for metric, cell in values["gains"].items():
            lines.append(
                f"| {season} | {values['games']} | {metric} | {interval(cell)} "
                f"| {cell['probability_positive']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Natural coefficients",
            "",
            ("**Measured:** slopes use original feature units; absent pressure is the base arm."),
            "",
            (
                "| Fold | Arm | Train/test | Intercept | Model logit | "
                "Composition | Move | Availability | Pressure |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coefficients:
        values = [row.get(key, 0.0) for key in ("intercept", *FIT_FEATURES, "pressure")]
        lines.append(
            (f"| {row['fold']} | {row['arm']} | {row['train_games']}/{row['test_games']} | ")
            + " | ".join(f"{value:.6f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reliability",
            "",
            ("**Measured:** bands use earlier-season training prediction quintiles for each arm."),
            ("Tied cut points can leave empty bands; held-out outcomes never set cut points."),
            "",
            "| Arm | Band | Games | Mean home probability | Home-cover fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["reliability"]:
        values = (
            "unavailable | unavailable"
            if not row["n"]
            else f"{row['prediction']:.4f} | {row['observed']:.4f}"
        )
        lines.append(f"| {row['arm']} | {row['band']} | {row['n']} | {values} |")
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            (
                "**Inferred:** retain unresolved_below_power pending the "
                "orchestrator's serial registry write;"
            ),
            (
                "no positive-control bound or split-half reliability test was "
                "run. AGENTS.md permits no closure"
            ),
            (
                "merely from an interval crossing zero. Research closure and "
                "card promotion remain separate."
            ),
            ("This conditional historical comparison does not authorize changing the served card."),
            (
                "**Measured:** B=1, F=3, E=0; 173 reserved fitting/reporting "
                "looks under the original formula,"
            ),
            (
                "including unavailable 2023/Elo cells; one primary accuracy "
                "contrast, no outcome-driven tuning."
            ),
            (
                "Coefficients, calibration cells and correlated comparators "
                "are not independent discoveries."
            ),
            (
                "**Inferred:** only two held-out seasons limit stability "
                "assessment. Intervals condition on the"
            ),
            (
                "frozen upstream model and predictions; they omit refitting "
                "and archive-correction uncertainty."
            ),
            (
                "Earlier-season fitting here does not re-certify upstream "
                "model selection or feature construction."
            ),
            (
                "Unchanged-spread book prices can be at a different line from "
                "the opener; their pressure is"
            ),
            (
                "a fitted opener predictor, never an alternative-line "
                "probability or automatic side flip."
            ),
            "",
            "## Reproduction",
            "",
            "\x60\x60\x60bash",
            ("UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/lead74_unit2.py"),
            "\x60\x60\x60",
            "",
            (
                "**Measured:** predictions, quote endpoints, metrics, hashes "
                "and declaration are preserved"
            ),
            (
                "in tests/scratch/codex/lead74_unit2/. Record commands are in "
                "the lane; not executed here."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def records(result):
    cells = []
    for comparator, metrics in result["contrasts"].items():
        for metric, cell in metrics.items():
            cells.append(
                {
                    "name": f"LEAD74-unit2-pressure-vs-{comparator}-{metric}",
                    "description": (
                        f"Chronological opener comparison: combined versus "
                        f"{comparator}; positive is improvement"
                    ),
                    "effect": cell["effect"],
                    "effect_units": metric
                    if metric == "accuracy_points"
                    else f"{metric}_improvement",
                    "interval_low": cell["low"],
                    "interval_high": cell["high"],
                    "standard_error": cell["se"],
                    "sample_games": result["games"],
                    "sample_blocks": result["blocks"],
                    "probability_positive": cell["probability_positive"],
                    "plain_summary": (
                        "Books sometimes change the price while leaving the spread "
                        "alone. This check asks whether those price changes help our "
                        "opener picks; keep the current picks while the evidence "
                        "remains unresolved."
                    ),
                }
            )
    payload = {
        "source": "docs/lead74_unit2.md",
        "league": "nfl",
        "season_start": 2024,
        "season_end": 2025,
        "family": "LEAD74-unit2-within-spread-price",
        "category": "market",
        "sample_games": result["games"],
        "sample_blocks": result["blocks"],
        "classification": "unresolved_below_power",
        "classification_evidence": (
            "Conditional historical availability; no positive-control or "
            "reliability closure ground established; descriptive baseline "
            "contrasts do not refute the price-pressure mechanism"
        ),
        "notes": (
            "173 reserved looks; one primary accuracy contrast; paired "
            "week bootstrap; provider observed time assumed available; 9 "
            "correlated metric contrasts are not independent evidence"
        ),
        "cells": cells,
    }
    (OUT / "record_payload.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args = [
        "nfl-ats",
        "weak-signals",
        "record",
        "--batch",
        str(OUT / "record_payload.json").replace("\\", "/"),
        "--source",
        "docs/lead74_unit2.md",
        "--plain-summary",
        (
            "Books sometimes change prices without changing the spread. "
            "Keep the current picks while we check whether those price "
            "changes help."
        ),
    ]
    command = "UV_NO_CACHE=1 .tools/uv.exe run --no-sync " + shlex.join(args)
    return command.replace(" --", " " + chr(92) + chr(10) + "  --")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    lane = LANE.read_text(encoding="utf-8-sig")
    if REPORT.exists():
        saved = REPORT.read_text(encoding="utf-8")
        declaration = (
            saved.split("## Frozen declaration\n", 1)[1]
            .split("\n## Coverage and source audit", 1)[0]
            .strip()
        )
    else:
        declaration = lane.split("## Unit-2 declaration\n", 1)[1].split("\n## Tried", 1)[0].strip()
    (OUT / "declaration.txt").write_text(declaration + "\n", encoding="utf-8")
    games = schedules()
    feature, audit = pressure(games)
    feature.to_parquet(OUT / "features.parquet", index=False)
    frame, coverage = population(feature)
    print(f"Outcome join complete: {len(frame)} quote-covered rows", flush=True)
    with threadpool_limits(limits=2):
        frame, coefficients, folds, unavailable = replay(frame)
        result = score(frame)
    audit["source_hashes"] = {
        str(path): digest(path)
        for path in (ROOT / "per_game.parquet", ROOT / "metadata.json", SCHEDULE)
    }
    audit["declaration_sha256"] = digest(OUT / "declaration.txt")
    frame.to_parquet(OUT / "predictions.parquet", index=False)
    payload = {
        "result": result,
        "coverage": coverage,
        "coefficients": coefficients,
        "fold_metrics": folds,
        "unavailable": unavailable,
        "audit": audit,
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    REPORT.write_text(
        report(result, coverage, coefficients, folds, unavailable, audit, declaration),
        encoding="utf-8",
    )
    command = records(result)
    (OUT / "record_commands.sh").write_text(command + "\n", encoding="utf-8")
    lane = lane.replace(
        ("Pending measured results; orchestrator runs exact bash commands serially."),
        (
            "Orchestrator only; NOT run here. Nine correlated metric "
            "records in the batch.\n\n\x60\x60\x60bash\n"
        )
        + command
        + "\n\x60\x60\x60",
    )
    LANE.write_text(lane, encoding="utf-8")
    print(
        json.dumps(
            {
                "games": result["games"],
                "blocks": result["blocks"],
                "decisive": result["decisive"],
                "primary": result["contrasts"]["four_term"],
                "coefficients": coefficients,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
