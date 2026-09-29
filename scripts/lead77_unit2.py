from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from quote_provenance_check import check_snapshot, validate_frame
from scipy.special import expit
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

ROOT = Path("tests/scratch/codex/lead77_unit2")
REPORT = Path("docs/lead77_unit2.md")
LANE = Path("docs/lanes/lead77.md")
SOURCE = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
SEASONS = (2023, 2024, 2025)
SPORTS = {"americanfootball_nfl": "NFL", "americanfootball_ncaaf": "CFB"}
ARMS = ("freshness", "four_term", "model", "market")
METRICS = ("accuracy_points", "log_loss", "brier")
BOOTSTRAPS = 10000
SEED = 20260929


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    )


def schedule() -> pd.DataFrame:
    columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "gametime",
        "home_team",
        "away_team",
    ]
    games = read_parquet(SCHEDULE, columns)
    games = games.loc[games.season.isin(SEASONS) & games.game_type.eq("REG")].copy()
    day = pd.to_datetime(games.gameday)
    local = pd.to_datetime(day.dt.strftime("%Y-%m-%d") + " " + games.gametime.astype(str))
    games["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    first_day = day.groupby([games.season, games.week]).transform("min")
    sunday = first_day + pd.to_timedelta((6 - first_day.dt.dayofweek) % 7, unit="D")
    deadline = (sunday + pd.Timedelta(hours=12, minutes=45)).dt.tz_localize("America/New_York")
    games["deadline"] = pd.concat([games.kickoff, deadline.dt.tz_convert("UTC")], axis=1).min(
        axis=1
    )
    if games.game_id.duplicated().any() or games.deadline.isna().any():
        raise ValueError("Schedule identity or deadline is invalid")
    return games.set_index("game_id")


def quote_panel(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected = {}
    inventory = Counter()
    for path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        season, sport = request.get("season"), request.get("sport")
        if (
            season not in SEASONS
            or sport not in SPORTS
            or manifest.get("capture_kind") != "historical_backfill"
        ):
            continue
        observed = pd.Timestamp(manifest["observed_at_utc"])
        key = (SPORTS[sport], season, observed.floor("6h"))
        inventory[f"{SPORTS[sport]}_source_snapshots"] += 1
        if key not in selected or observed > selected[key][0]:
            selected[key] = observed, path.parent, manifest
    frames = []
    source_hash = hashlib.sha256()
    for count, (key, (observed, path, manifest)) in enumerate(sorted(selected.items()), 1):
        league, season, _ = key
        checks = check_snapshot(path, manifest, None)
        inventory["validated_snapshots"] += 1
        inventory["validator_asof_pairs"] += checks["asof_consistent_pairs"]
        source_hash.update(path.as_posix().encode())
        source_hash.update(manifest["files"]["quotes.parquet"]["sha256"].encode())
        checked = validate_frame(read_parquet(path / "quotes.parquet"), manifest)
        quotes = checked.loc[
            checked.asof_consistent
            & checked.market.eq("spreads")
            & checked.outcome_side.isin(["HOME", "AWAY"])
        ].copy()
        quotes["line"] = pd.to_numeric(quotes.line, errors="coerce")
        quotes["price"] = pd.to_numeric(quotes.price, errors="coerce")
        quotes["updated"] = pd.to_datetime(
            quotes.market_last_update_utc, utc=True, errors="coerce"
        ).fillna(pd.to_datetime(quotes.bookmaker_last_update_utc, utc=True, errors="coerce"))
        quotes["kickoff"] = pd.to_datetime(quotes.commence_time_utc, utc=True, errors="coerce")
        if league == "NFL":
            quotes = quotes.loc[quotes.nflverse_game_id.isin(games.index)].copy()
            quotes["game_id"] = quotes.nflverse_game_id
            for side in ("home", "away"):
                if not quotes[f"{side}_team"].eq(quotes.game_id.map(games[f"{side}_team"])).all():
                    raise ValueError("NFL quote team identity mismatch")
            if not quotes.game_id.map(games.season).eq(season).all():
                raise ValueError("NFL quote season mismatch")
            quotes["deadline"] = quotes.game_id.map(games.deadline)
            quotes["deadline"] = quotes[["deadline", "kickoff"]].min(axis=1)
        else:
            quotes["game_id"] = "CFB:" + str(season) + ":" + quotes.provider_event_id.astype(str)
            quotes["deadline"] = quotes.kickoff
        quotes = quotes.loc[
            np.isfinite(quotes.line)
            & np.isfinite(quotes.price)
            & quotes.price.abs().ge(100)
            & quotes.updated.notna()
            & quotes.updated.le(observed)
            & quotes.deadline.gt(observed)
            & (quotes.deadline - observed).le(pd.Timedelta(days=7))
        ].copy()
        quotes["home_line"] = quotes.line.where(quotes.outcome_side.eq("HOME"), -quotes.line)
        keys = ["provider_event_id", "bookmaker_key", "home_line"]
        quotes = quotes.loc[~quotes.duplicated([*keys, "outcome_side"], keep=False)]
        fields = [*keys, "game_id", "price", "updated", "deadline"]
        home = quotes.loc[quotes.outcome_side.eq("HOME"), fields]
        away = quotes.loc[quotes.outcome_side.eq("AWAY"), fields]
        paired = home.merge(away, on=keys, suffixes=("_h", "_a"), validate="one_to_one")
        if not paired.game_id_h.eq(paired.game_id_a).all():
            raise ValueError("Opposing quote game identity mismatch")
        for side in ("h", "a"):
            price = paired[f"price_{side}"]
            paired[f"implied_{side}"] = np.where(
                price > 0, 100 / (price + 100), -price / (100 - price)
            )
        paired["q"] = paired.implied_h / (paired.implied_h + paired.implied_a)
        paired["updated"] = paired[["updated_h", "updated_a"]].min(axis=1)
        paired["deadline"] = paired[["deadline_h", "deadline_a"]].min(axis=1)
        paired["game_id"] = paired.game_id_h
        paired["observed"] = observed
        paired["season"] = season
        paired["league"] = league
        paired["line"] = paired.home_line
        fields = ["game_id", "season", "league", "observed", "updated", "deadline", "line", "q"]
        frames.append(paired[fields])
        if count % 150 == 0:
            print(f"Validated {count}/{len(selected)} capped snapshots", flush=True)
    if not frames:
        raise ValueError("No quote snapshots selected")
    panel = pd.concat(frames, ignore_index=True)
    if not (panel.updated.le(panel.observed) & panel.observed.lt(panel.deadline)).all():
        raise ValueError("Quote cutoff guard failed")
    inventory["paired_quotes"] = len(panel)
    return panel, {**dict(inventory), "selected_snapshot_hash": source_hash.hexdigest()}


def prepare_response(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    keys = ["game_id", "season", "league", "observed", "line"]
    grouped = panel.groupby(keys, sort=False)
    panel["group"] = grouped.ngroup()
    panel["books"] = grouped.q.transform("size")
    panel = panel.loc[panel.books.ge(3)].copy()
    latest = panel.groupby("group").updated.transform("max")
    fresh_q = panel.loc[panel.updated.eq(latest)].groupby("group").q.median()
    panel["distance"] = panel.group.map(fresh_q) - panel.q
    age = (panel.observed - panel.updated).dt.total_seconds() / 3600
    panel["age_distance"] = np.log1p(age) * panel.distance
    summary = (
        panel.groupby("group")
        .agg(
            game_id=("game_id", "first"),
            season=("season", "first"),
            league=("league", "first"),
            observed=("observed", "first"),
            line=("line", "first"),
            books=("q", "size"),
            median_q=("q", "median"),
            oldest=("updated", "min"),
            deadline=("deadline", "min"),
        )
        .reset_index()
    )
    future, label_time = {}, {}
    for _, rows in summary.groupby(["game_id", "line"], sort=False):
        rows = rows.sort_values("observed")
        stamps = rows.observed.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        updates = rows.oldest.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        deadlines = rows.deadline.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        consensus, groups = rows.median_q.to_numpy(), rows.group.to_numpy()
        for index, stamp in enumerate(stamps):
            valid = np.flatnonzero(
                (stamps > stamp)
                & (stamps - stamp <= pd.Timedelta(hours=48).value)
                & (updates > stamp)
                & (stamps < deadlines[index])
            )
            if len(valid):
                following = valid[0]
                future[groups[index]] = consensus[following]
                label_time[groups[index]] = pd.Timestamp(stamps[following], tz="UTC")
    panel["target"] = panel.group.map(future) - panel.q
    panel["label_at"] = pd.to_datetime(panel.group.map(label_time), utc=True)
    response = panel.loc[panel.target.notna()].copy()
    if not (
        response.label_at.gt(response.observed) & response.label_at.lt(response.deadline)
    ).all():
        raise ValueError("Response label cutoff guard failed")
    latest = summary.sort_values(
        ["game_id", "observed", "books", "line"], ascending=[True, False, False, True]
    ).drop_duplicates("game_id")
    features = panel.loc[panel.group.isin(latest.group)].copy()
    coverage = {}
    for league in SPORTS.values():
        for season in SEASONS:
            source = panel.loc[panel.league.eq(league) & panel.season.eq(season)]
            labels = response.loc[response.league.eq(league) & response.season.eq(season)]
            coverage[f"{league}_{season}"] = {
                "games": int(source.game_id.nunique()),
                "quotes": len(source),
                "response_games": int(labels.game_id.nunique()),
                "response_rows": len(labels),
            }
    return response, features, coverage


def fit_response(train: pd.DataFrame, pooled: bool) -> dict:
    rows = train if pooled else train.loc[train.league.eq("NFL")]
    if rows.empty or (pooled and set(rows.league) != {"NFL", "CFB"}):
        raise ValueError("Response training population is unavailable")
    x = rows[["distance", "age_distance"]].to_numpy()
    weight = 1 / rows.groupby("game_id").game_id.transform("size").to_numpy()
    scale = np.sqrt(np.average(x**2, axis=0, weights=weight))
    scale = np.where(scale > 1e-12, scale, 1)
    x = x / scale
    if pooled:
        x = np.column_stack([x, x * rows.league.eq("NFL").to_numpy()[:, None]])
    penalty = np.diag([0.001, 0.001, 1.0, 1.0] if pooled else [0.001, 0.001])
    beta = np.linalg.solve((x.T * weight) @ x + penalty, (x.T * weight) @ rows.target.to_numpy())
    natural = beta / np.tile(scale, 2 if pooled else 1)
    return {
        "natural": natural,
        "pooled": pooled,
        "train_games": int(rows.game_id.nunique()),
        "train_rows": len(rows),
    }


def predict_response(rows: pd.DataFrame, fit: dict) -> np.ndarray:
    x = rows[["distance", "age_distance"]].to_numpy()
    if fit["pooled"]:
        x = np.column_stack([x, x * rows.league.eq("NFL").to_numpy()[:, None]])
    return x @ fit["natural"]


def game_mae(rows: pd.DataFrame, fit: dict) -> float:
    error = np.abs(predict_response(rows, fit) - rows.target.to_numpy())
    return float(pd.Series(error, index=rows.game_id).groupby(level=0).mean().mean())


def fit_probability(train: pd.DataFrame, test: pd.DataFrame, names: list[str]) -> tuple:
    means = train[names].mean().to_numpy()
    scales = train[names].std(ddof=0).to_numpy()
    scales = np.where(scales > 1e-12, scales, 1)
    x = np.column_stack([np.ones(len(train)), (train[names].to_numpy() - means) / scales])
    z = np.column_stack([np.ones(len(test)), (test[names].to_numpy() - means) / scales])
    beta = _fit_logit(x, train.home_covered.to_numpy(dtype=float), FIT_RIDGE)
    natural = beta[1:] / scales
    coefficients = {
        "intercept": float(beta[0] - natural @ means),
        **dict(zip(names, natural.tolist(), strict=True)),
    }
    return expit(x @ beta), expit(z @ beta), coefficients


def loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.column_stack(
        [100 * ((p >= 0.5) == y), -(y * np.log(p) + (1 - y) * np.log(1 - p)), (p - y) ** 2]
    )


def replay(response: pd.DataFrame, features: pd.DataFrame) -> tuple:
    frame = read_parquet(SOURCE)
    if (
        frame.game_id.duplicated().any()
        or not frame.season.between(2020, 2025).all()
        or not frame.game_type.eq("REG").all()
    ):
        raise ValueError("Frozen base population mismatch")
    if not frame.base_probability_policy.eq("discrete_conditional_non_push_v1").all():
        raise ValueError("Frozen model is not the discrete conditional probability")
    if (
        frame.margin_vs_open.eq(0).any()
        or not frame.home_covered.eq(frame.margin_vs_open.gt(0)).all()
    ):
        raise ValueError("Historical opener grade mismatch")
    if not np.allclose(
        frame.model_logit, np.log(frame.model_probability / (1 - frame.model_probability))
    ):
        raise ValueError("Frozen model logit mismatch")
    frame = frame.loc[
        frame.season.isin(SEASONS)
        & frame.game_id.isin(features.loc[features.league.eq("NFL"), "game_id"])
    ].copy()
    if not np.isfinite(
        frame[[*FIT_FEATURES, "home_covered", "model_probability"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Incomplete ATS inputs")
    predictions, train_scores, fold_scores, coefficients, response_scores = [], [], [], [], []
    for held in SEASONS:
        if held == 2023:
            continue
        source_train = response.loc[response.season.lt(held)]
        nfl_train = source_train.loc[source_train.league.eq("NFL")]
        nfl_test = response.loc[response.season.eq(held) & response.league.eq("NFL")]
        pooled_fit = None
        for name, pooled in (("nfl_response", False), ("pooled_response", True)):
            fitted = fit_response(source_train, pooled)
            if pooled:
                pooled_fit = fitted
            response_scores.append(
                {
                    "held": held,
                    "arm": name,
                    "train_mae": game_mae(nfl_train, fitted),
                    "test_mae": game_mae(nfl_test, fitted),
                    "train_games": int(nfl_train.game_id.nunique()),
                    "test_games": int(nfl_test.game_id.nunique()),
                    "fit_games": fitted["train_games"],
                    "fit_rows": fitted["train_rows"],
                }
            )
            labels = (
                [
                    "shared_distance",
                    "shared_age_distance",
                    "nfl_distance_deviation",
                    "nfl_age_distance_deviation",
                ]
                if pooled
                else ["distance", "age_distance"]
            )
            coefficients.append(
                {
                    "held": held,
                    "arm": name,
                    **dict(zip(labels, fitted["natural"].tolist(), strict=True)),
                }
            )
        eligible = features.loc[features.league.eq("NFL") & features.season.le(held)].copy()
        eligible["next_q"] = eligible.q + predict_response(eligible, pooled_fit)
        correction = (
            eligible.groupby("game_id").next_q.mean() - eligible.groupby("game_id").q.median()
        )
        work = frame.loc[frame.season.le(held)].copy()
        work["freshness_correction"] = work.game_id.map(correction)
        if work.freshness_correction.isna().any():
            raise ValueError("Missing preregistered freshness feature")
        train, test = work.loc[work.season.lt(held)].copy(), work.loc[work.season.eq(held)].copy()
        train_prob = {
            "model": train.model_probability.to_numpy(),
            "market": np.full(len(train), 0.5),
        }
        test_prob = {"model": test.model_probability.to_numpy(), "market": np.full(len(test), 0.5)}
        for name in ("four_term", "freshness"):
            terms = list(FIT_FEATURES) + (["freshness_correction"] if name == "freshness" else [])
            train_prob[name], test_prob[name], natural = fit_probability(train, test, terms)
            coefficients.append({"held": held, "arm": name, **natural})
        for arm in ARMS:
            train_loss = loss(train.home_covered.to_numpy(), train_prob[arm])
            test_loss = loss(test.home_covered.to_numpy(), test_prob[arm])
            train_scores.append(
                {
                    "held": held,
                    "arm": arm,
                    "n": len(train),
                    "mean": train_loss.mean(axis=0).tolist(),
                }
            )
            fold_scores.append(
                {
                    "held": held,
                    "arm": arm,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train": train_loss.mean(axis=0).tolist(),
                    "test": test_loss.mean(axis=0).tolist(),
                }
            )
            test[f"p_{arm}"] = test_prob[arm]
            edges = np.quantile(train_prob[arm], [0.2, 0.4, 0.6, 0.8])
            test[f"band_{arm}"] = np.searchsorted(edges, test_prob[arm], side="right")
        test["held"] = held
        predictions.append(test)
        print(
            f"Completed chronological holdout {held}: {len(train)} train / {len(test)} test",
            flush=True,
        )
    return (
        pd.concat(predictions, ignore_index=True),
        train_scores,
        fold_scores,
        coefficients,
        response_scores,
    )


def estimate(point: float, draws: np.ndarray) -> dict:
    draws = draws[np.isfinite(draws)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "effect": float(point),
        "low": float(low),
        "high": float(high),
        "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
    }


def summarize(frame: pd.DataFrame) -> dict:
    y = frame.home_covered.to_numpy(dtype=float)
    blocks = pd.MultiIndex.from_frame(frame[["season", "week"]])
    unique = blocks.unique()
    ids = unique.get_indexer(blocks)
    counts = np.bincount(ids, minlength=len(unique))
    losses = np.stack([loss(y, frame[f"p_{arm}"].to_numpy()) for arm in ARMS], axis=1)
    totals = np.zeros((len(unique), len(ARMS), len(METRICS)))
    np.add.at(totals, ids, losses)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((BOOTSTRAPS, len(unique)))
    for season in sorted(frame.season.unique()):
        indexes = np.flatnonzero(unique.get_level_values("season") == season)
        weights[:, indexes] = rng.multinomial(
            len(indexes), np.full(len(indexes), 1 / len(indexes)), size=BOOTSTRAPS
        )
    denominator = weights @ counts
    boot = np.einsum("rb,bam->ram", weights, totals) / denominator[:, None, None]
    point = losses.mean(axis=0)
    absolute, contrasts = {}, {}
    for index, arm in enumerate(ARMS):
        absolute[arm] = {
            metric: estimate(point[index, j], boot[:, index, j]) for j, metric in enumerate(METRICS)
        }
        if arm != "freshness":
            direction = np.array([1, -1, -1])
            gain = (point[0] - point[index]) * direction
            samples = (boot[:, 0] - boot[:, index]) * direction
            contrasts[arm] = {
                metric: estimate(gain[j], samples[:, j]) for j, metric in enumerate(METRICS)
            }
    decisive = frame.p_freshness.ge(0.5).ne(frame.p_four_term.ge(0.5)).to_numpy()
    wins = decisive & (frame.p_freshness.ge(0.5).to_numpy() == y)
    decisive_counts = np.bincount(ids, weights=decisive, minlength=len(unique))
    win_counts = np.bincount(ids, weights=wins, minlength=len(unique))
    draw_n = weights @ decisive_counts
    decisive_draws = np.divide(
        100 * (weights @ win_counts), draw_n, out=np.full(BOOTSTRAPS, np.nan), where=draw_n > 0
    )
    decisive_summary = {
        "wins": int(wins.sum()),
        "losses": int(decisive.sum() - wins.sum()),
        "games": int(decisive.sum()),
    }
    if decisive.any():
        decisive_summary["accuracy"] = estimate(100 * wins.sum() / decisive.sum(), decisive_draws)
        decisive_summary["probability_above_half"] = estimate(
            100 * wins.sum() / decisive.sum() - 50, decisive_draws - 50
        )["probability_positive"]
    return {
        "n": len(frame),
        "blocks": len(unique),
        "absolute": absolute,
        "contrasts": contrasts,
        "decisive": decisive_summary,
    }


def cell(value: dict) -> str:
    return f"{value['effect']:.6f} [{value['low']:.6f}, {value['high']:.6f}]"


def render(frame: pd.DataFrame, payload: dict, protocol: str) -> str:
    result = payload["summary"]
    decisive = result["decisive"]
    lines = [
        "# LEAD-77 unit 2: quote-freshness reliability",
        "",
        (
            "**Conditional assumption:** historical provider `observed_at_utc` "
            "is availability time, as authorized in the provenance decision. "
            "Historical opener is the frozen pool-line proxy; no pre-2026 "
            "Splash captures are used."
        ),
        "",
        (
            f"**Measured, decisive games first:** freshness won "
            f"{decisive['wins']}-{decisive['losses']} on {decisive['games']} "
            f"side disagreements with the four-term base."
        ),
    ]
    if "accuracy" in decisive:
        lines.append(
            f"**Measured:** decisive accuracy {cell(decisive['accuracy'])}%; "
            f"probability_positive versus 50% = "
            f"{decisive['probability_above_half']:.6f}."
        )
    lines += [
        "",
        (
            f"**Measured:** {result['n']} held-out non-push opener games; "
            f"{result['blocks']} season-week blocks; chronological 2024 and 2025 "
            f"folds. 2023 supplies training only. No forecast/card changed."
        ),
        "",
        "## Declaration saved before outcomes",
        "",
        protocol.strip(),
        "",
        "## Paired held-out results",
        "",
        (
            "**Measured:** brackets are 95% season-stratified week-block "
            "bootstrap intervals, 10,000 draws, fixed predictions. Loss gains "
            "are comparator loss minus freshness loss; accuracy gains are "
            "freshness minus comparator, in percentage points."
        ),
        "",
        table(
            ["Arm", "Accuracy %", "Log loss", "Brier"],
            [[arm, *[cell(result["absolute"][arm][metric]) for metric in METRICS]] for arm in ARMS],
        ),
        "",
        table(
            ["Comparator", "Metric", "Gain [95% interval]", "probability_positive"],
            [
                [arm, metric, cell(value), f"{value['probability_positive']:.6f}"]
                for arm, metrics in result["contrasts"].items()
                for metric, value in metrics.items()
            ],
        ),
        "",
        "## Training and held-out gap",
        "",
        (
            "**Measured:** training metrics pool each outer fit training "
            "appearance; 2023 appears twice. Gap is held-out minus training, so "
            "a positive loss gap means worse held-out loss. No held-out season "
            "enters its response or calibration fit."
        ),
        "",
    ]
    rows = []
    for arm in ARMS:
        entries = [entry for entry in payload["training"] if entry["arm"] == arm]
        means = np.average(
            [entry["mean"] for entry in entries], axis=0, weights=[entry["n"] for entry in entries]
        )
        for index, metric in enumerate(METRICS):
            held = result["absolute"][arm][metric]["effect"]
            rows.append(
                [arm, metric, f"{means[index]:.6f}", f"{held:.6f}", f"{held - means[index]:.6f}"]
            )
    lines += [
        table(["Arm", "Metric", "IS", "OOS", "OOS-IS"], rows),
        "",
        "**Measured:** per-fold values are IS / OOS / gap.",
        "",
    ]
    rows = [
        [
            entry["held"],
            entry["arm"],
            f"{entry['train_n']}/{entry['test_n']}",
            *[
                f"{a:.6f} / {b:.6f} / {b - a:.6f}"
                for a, b in zip(entry["train"], entry["test"], strict=True)
            ],
        ]
        for entry in payload["folds"]
    ]
    lines += [
        table(["Held season", "Arm", "Train/test n", *METRICS], rows),
        "",
        "## Response fit and coverage",
        "",
        (
            "**Measured:** response MAE uses identical NFL response games for "
            "both response arms, averaging within game first. Pooled training "
            "additionally uses CFB. Rows are repeated quotes, never extra ATS "
            "games."
        ),
        "",
    ]
    rows = [
        [
            entry["held"],
            entry["arm"],
            f"{entry['fit_games']}/{entry['fit_rows']}",
            f"{entry['train_games']}/{entry['test_games']}",
            f"{entry['train_mae']:.6f}",
            f"{entry['test_mae']:.6f}",
            f"{entry['test_mae'] - entry['train_mae']:.6f}",
        ]
        for entry in payload["response_scores"]
    ]
    lines += [
        table(
            [
                "Held",
                "Response",
                "Fit games/rows",
                "NFL train/test games",
                "IS MAE",
                "OOS MAE",
                "Gap",
            ],
            rows,
        ),
        "",
        table(
            ["League/season", "Feature games", "Book rows", "Response games", "Response rows"],
            [[name, *entry.values()] for name, entry in payload["coverage"].items()],
        ),
        "",
        "## Coefficients and calibration",
        "",
        (
            "**Measured:** natural-unit coefficients below; response intercept "
            "fixed at zero. Pooled NFL response uses shared plus NFL-deviation "
            "slopes. ATS coefficients include the intercept and all four base "
            "terms; the correction is fitted jointly."
        ),
        "",
        table(
            ["Held", "Fit", "Term", "Coefficient"],
            [
                [entry["held"], entry["arm"], term, f"{value:.9f}"]
                for entry in payload["coefficients"]
                for term, value in entry.items()
                if term not in {"held", "arm"}
            ],
        ),
        "",
        (
            "**Measured:** five training-quantile bands per arm, pooled across "
            "held-out seasons. Ties can leave bands empty; no band chooses the "
            "side."
        ),
        "",
    ]
    reliability = []
    for arm in ARMS:
        for band in range(5):
            rows = frame.loc[frame[f"band_{arm}"].eq(band)]
            reliability.append(
                [
                    arm,
                    band + 1,
                    len(rows),
                    f"{rows[f'p_{arm}'].mean():.6f}" if len(rows) else "n/a",
                    f"{rows.home_covered.mean():.6f}" if len(rows) else "n/a",
                ]
            )
    lines += [
        table(
            ["Arm", "Training-quantile band", "n", "Mean home p", "Home-cover rate"], reliability
        ),
        "",
        "## Interpretation and limits",
        "",
        "**Measured:** correction coefficients by held season: "
        + "; ".join(
            f"{entry['held']}: {entry['freshness_correction']:.6f}"
            for entry in payload["coefficients"]
            if entry["arm"] == "freshness"
        )
        + ". Compare the per-fold table for accuracy and loss stability.",
        "",
        (
            "**Inferred:** this is conditional research evidence, not a serving "
            "decision. Per AGENTS.md, an interval crossing zero is not a "
            "closing ground. The mechanism remains unresolved_below_power "
            "pending the orchestrator serial registry record; no "
            "positive-control or split-half-reliability closure was tested."
        ),
        "",
        (
            "**Read:** the reused upstream artifact supplies a discrete "
            "conditional non-push model probability. This script never maps "
            "across spread lines or estimates a smooth residual margin "
            "distribution; a single fitted probability selects each side at the "
            "historical opener. Push and alternative-line serving are outside "
            "this unit."
        ),
        "",
        (
            "**Inferred:** same-line synchronization excludes line-change "
            "responses, and six-hour source capping limits freshness "
            "resolution. CFB and NFL use identical paired-price probability "
            "units, but their sporting response may differ; penalized NFL "
            "deviations address that declared difference without selecting an "
            "arm afterward. Frozen upstream model/composition fits are "
            "inherited, not recertified as fully chronological here. Week-block "
            "intervals hold all fitted predictions fixed and do not quantify "
            "refit uncertainty. Two scored seasons cannot establish broad "
            "season stability."
        ),
        "",
        (
            "**Measured:** 158 reporting looks within the 162-look ceiling: 8 "
            "fits + 72 IS/OOS metric cells + 8 response-MAE cells + 20 "
            "reliability bands + 9 paired contrasts + 1 decisive comparison + "
            "12 pooled gaps + 24 fold gaps + 4 response gaps. Derived gaps are "
            "counted conservatively. One primary comparison; correlated "
            "diagnostics are not separate discoveries."
        ),
        "",
        "## Reproduction and provenance",
        "",
        "```bash",
        "UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/lead77_unit2.py",
        "```",
        "",
        "**Measured:** source hashes and structural counts:",
        "",
        "```json",
        json.dumps(payload["provenance"], indent=2),
        "```",
        "",
        (
            "Prediction rows, response rows, result JSON and a registry batch "
            "are saved only under `tests/scratch/codex/lead77_unit2/`. The lane "
            "contains the exact registry command; this script never runs it."
        ),
    ]
    return "\n".join(lines) + "\n"


def registry(result: dict) -> dict:
    cells = []
    for arm, metrics in result["contrasts"].items():
        for metric, value in metrics.items():
            cells.append(
                {
                    "name": f"LEAD77-unit2-freshness-vs-{arm}-{metric}",
                    "description": (
                        f"Chronology-purged quote-freshness correction versus {arm}; "
                        f"positive means improvement in {metric}."
                    ),
                    "effect": value["effect"],
                    "effect_units": metric
                    if metric == "accuracy_points"
                    else f"{metric}_improvement",
                    "interval_low": value["low"],
                    "interval_high": value["high"],
                    "probability_positive": value["probability_positive"],
                    "classification": "unresolved_below_power",
                    "sample_games": result["n"],
                    "sample_blocks": result["blocks"],
                    "classification_evidence": (
                        "No mechanism closure or detecting positive control established; "
                        "conditional historical availability and fixed-prediction "
                        "week-block uncertainty."
                    ),
                    "plain_summary": (
                        "Older sportsbook prices were compared with recently updated prices "
                        "before the pick deadline. This check does not establish that quote "
                        "age should change the current pool picks."
                    ),
                }
            )
    return {
        "source": REPORT.as_posix(),
        "league": "nfl",
        "season_start": 2024,
        "season_end": 2025,
        "family": "LEAD77-unit2-quote-freshness",
        "category": "market",
        "sample_games": result["n"],
        "sample_blocks": result["blocks"],
        "notes": (
            "Historical opener proxy, conditional provider availability; 158 "
            "reporting looks within 162 ceiling, one primary freshness versus "
            "four-term accuracy contrast. Nine correlated contrasts are not "
            "independent evidence."
        ),
        "cells": cells,
    }


def main() -> None:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    lane = LANE.read_text(encoding="utf-8-sig")
    heading = "## Unit-2 amendment \u2014 declared before outcomes\n"
    if heading in lane:
        protocol = lane.split(heading, 1)[1].split("\n## Goal", 1)[0]
    else:
        protocol = (
            REPORT.read_text(encoding="utf-8")
            .split("## Declaration saved before outcomes\n\n", 1)[1]
            .split("\n## Paired held-out results", 1)[0]
            .rstrip()
            + "\n"
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "declaration.md").write_text(protocol, encoding="utf-8")
    provenance = {
        "base": SOURCE.as_posix(),
        "base_sha256": digest(SOURCE),
        "schedule": SCHEDULE.as_posix(),
        "schedule_sha256": digest(SCHEDULE),
        "declaration_sha256": hashlib.sha256(protocol.encode()).hexdigest(),
        "script_sha256": digest(Path(__file__)),
    }
    games = schedule()
    panel, inventory = quote_panel(games)
    provenance.update(inventory)
    response, features, coverage = prepare_response(panel)
    del panel
    response.to_parquet(ROOT / "response_rows.parquet", index=False)
    features.to_parquet(ROOT / "feature_rows.parquet", index=False)
    with threadpool_limits(limits=2):
        frame, training, folds, coefficients, response_scores = replay(response, features)
        result = summarize(frame)
    frame.to_parquet(ROOT / "predictions.parquet", index=False)
    payload = {
        "summary": result,
        "training": training,
        "folds": folds,
        "coefficients": coefficients,
        "response_scores": response_scores,
        "coverage": coverage,
        "provenance": provenance,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ROOT / "registry_batch.json").write_text(
        json.dumps(registry(result), indent=2), encoding="utf-8"
    )
    REPORT.write_text(render(frame, payload, protocol), encoding="utf-8")
    print(
        json.dumps(
            {
                "games": result["n"],
                "decisive": result["decisive"],
                "primary": result["contrasts"]["four_term"],
                "report": REPORT.as_posix(),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
