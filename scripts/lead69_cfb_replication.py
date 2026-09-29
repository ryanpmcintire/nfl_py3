from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability_fit import FIT_RIDGE, _fit_logit

SEASONS = (2023, 2024, 2025)
SEED = 6902026
RESAMPLES = 4000
SCHEDULES = Path("data/cfb/schedules/raw/20260816T162105Z")
QUOTES = Path("data/market/raw")
REPORT = Path("docs/lead69_cfb_replication.md")
SCRATCH = Path("tests/scratch/codex/lead69_cfb_replication")
BASE = ["opener_logit", "market_move_toward_home"]
ARMS = ("opener", "base", "horizon")
NFL_BETAS = {2023: -0.434823142, 2024: -0.029133083, 2025: -0.653789708}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def implied(price: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(price < 0, -price / (100 - price), 100 / (100 + price)), index=price.index
    )


def load_predictors() -> tuple[pd.DataFrame, dict]:
    audit: dict = {"opener_basis": "first_observed_paired_quote_proxy", "quote_files": 0}
    columns = [
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date",
        "completed",
        "neutral_site",
        "home_team",
        "away_team",
        "home_division",
        "away_division",
    ]
    schedule = pd.concat(
        [
            pd.read_parquet(
                SCHEDULES / f"season={s}/schedules.parquet", columns=columns, use_threads=False
            )
            for s in SEASONS
        ],
        ignore_index=True,
    )
    names = set(schedule.home_team.dropna()) | set(schedule.away_team.dropna())
    normalized = {normalize(name): name for name in sorted(names)}
    aliases = {"appalachianstatemountaineers": "App State", "umassminutemen": "Massachusetts"}

    def team(value: str) -> str | None:
        key = normalize(value)
        if key in aliases:
            return aliases[key]
        matches = [prefix for prefix in normalized if key.startswith(prefix)]
        return normalized[max(matches, key=len)] if matches else None

    schedule = schedule.loc[
        schedule.season_type.eq("regular")
        & schedule.completed.fillna(False)
        & (schedule.home_division.eq("fbs") | schedule.away_division.eq("fbs"))
    ].copy()
    schedule["schedule_kickoff"] = pd.to_datetime(schedule.start_date, utc=True)
    require(not schedule.game_id.duplicated().any(), "Duplicate schedule game IDs")
    audit["fbs_regular_completed_by_season"] = schedule.groupby("season").size().to_dict()
    quote_columns = [
        "provider_event_id",
        "sport_key",
        "observed_at_utc",
        "commence_time_utc",
        "home_team_name",
        "away_team_name",
        "bookmaker_key",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
        "market",
        "outcome_side",
        "line",
        "price",
    ]
    frames = []
    for path in sorted(QUOTES.glob("*-ncaaf/quotes.parquet")):
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8-sig"))
        if manifest.get("request", {}).get("season") not in SEASONS:
            continue
        available = set(pq.read_schema(path).names)
        require(
            (set(quote_columns) - {"market_last_update_utc"}).issubset(available),
            f"Missing quote schema: {path}",
        )
        q = pd.read_parquet(
            path, columns=[c for c in quote_columns if c in available], use_threads=False
        )
        q = q.loc[
            q.sport_key.eq("americanfootball_ncaaf")
            & q.market.eq("spreads")
            & q.outcome_side.isin(("HOME", "AWAY"))
        ].copy()
        q["snapshot"] = pd.to_datetime(manifest["snapshot_timestamp_utc"], utc=True)
        frames.append(q)
        audit["quote_files"] += 1
    require(bool(frames), "No CFB historical quote files")
    quotes = pd.concat(frames, ignore_index=True)
    audit["spread_quote_rows"] = len(quotes)
    for column in (
        "observed_at_utc",
        "commence_time_utc",
        "bookmaker_last_update_utc",
        "market_last_update_utc",
    ):
        if column not in quotes:
            quotes[column] = pd.NaT
        quotes[column] = pd.to_datetime(quotes[column], utc=True, errors="coerce")
    mapping = {name: team(name) for name in set(quotes.home_team_name) | set(quotes.away_team_name)}
    quotes["home_team"] = quotes.home_team_name.map(mapping)
    quotes["away_team"] = quotes.away_team_name.map(mapping)
    events = quotes.groupby("provider_event_id", as_index=False).agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        quote_kickoff=("commence_time_utc", "min"),
    )
    audit["quote_events"] = len(events)
    candidates = events.merge(schedule, on=["home_team", "away_team"], how="inner")
    candidates = candidates.loc[
        (candidates.quote_kickoff - candidates.schedule_kickoff).abs().le(pd.Timedelta(hours=36))
    ].copy()
    require(not candidates.provider_event_id.duplicated().any(), "Ambiguous event-to-schedule join")
    duplicate_games = candidates.game_id.duplicated(keep=False)
    audit["ambiguous_multi_event_games_excluded"] = int(
        candidates.loc[duplicate_games, "game_id"].nunique()
    )
    candidates = candidates.loc[~duplicate_games].copy()
    candidates["kickoff"] = candidates[["quote_kickoff", "schedule_kickoff"]].min(axis=1)
    audit["matched_games_by_season"] = candidates.groupby("season").size().to_dict()
    quotes = quotes.merge(
        candidates[["provider_event_id", "game_id", "kickoff"]],
        on="provider_event_id",
        how="inner",
        validate="many_to_one",
    )
    provider = quotes[["market_last_update_utc", "bookmaker_last_update_utc"]].max(axis=1)
    safe = (
        quotes.observed_at_utc.notna()
        & quotes.snapshot.notna()
        & provider.notna()
        & provider.le(quotes.observed_at_utc)
        & quotes.observed_at_utc.le(quotes.snapshot)
        & quotes.snapshot.lt(quotes.kickoff)
        & quotes.commence_time_utc.ge(quotes.kickoff)
    )
    audit["unsafe_quote_rows_removed"] = int((~safe).sum())
    quotes = quotes.loc[safe].copy()
    quotes["provider_time"] = provider.loc[safe]
    quotes["line"] = pd.to_numeric(quotes.line, errors="coerce")
    quotes["price"] = pd.to_numeric(quotes.price, errors="coerce")
    quotes = quotes.loc[
        np.isfinite(quotes.line) & np.isfinite(quotes.price) & quotes.price.abs().ge(100)
    ].copy()
    keys = ["provider_event_id", "game_id", "bookmaker_key", "observed_at_utc"]
    side_columns = [*keys, "line", "price", "snapshot", "provider_time"]
    quotes = quotes.drop_duplicates([*keys, "outcome_side", "line", "price"])
    require(not quotes.duplicated([*keys, "outcome_side"]).any(), "Conflicting book-side quotes")
    pairs = quotes.loc[quotes.outcome_side.eq("HOME"), side_columns].merge(
        quotes.loc[quotes.outcome_side.eq("AWAY"), side_columns],
        on=keys,
        suffixes=("_home", "_away"),
        validate="one_to_one",
    )
    pairs = pairs.loc[np.isclose(pairs.line_home + pairs.line_away, 0)].copy()
    home_p, away_p = implied(pairs.price_home), implied(pairs.price_away)
    pairs["opener_probability"] = home_p / (home_p + away_p)
    pairs["snapshot"] = pairs[["snapshot_home", "snapshot_away"]].max(axis=1)
    pairs["provider_time"] = pairs[["provider_time_home", "provider_time_away"]].max(axis=1)
    rows, no_later = [], 0
    for game_id, group in pairs.groupby("game_id", sort=True):
        group = group.sort_values(["observed_at_utc", "bookmaker_key"])
        opening_time = group.observed_at_utc.min()
        opening = group.loc[group.observed_at_utc.eq(opening_time)]
        opening_lines = np.sort(opening.line_home.to_numpy())
        opener = float(opening_lines[(len(opening_lines) - 1) // 2])
        p_open = float(opening.loc[opening.line_home.eq(opener), "opener_probability"].median())
        latest = group.groupby("bookmaker_key", sort=True).tail(1)
        moves = opening[["bookmaker_key", "line_home"]].merge(
            latest[["bookmaker_key", "line_home", "observed_at_utc"]],
            on="bookmaker_key",
            suffixes=("_open", "_late"),
            validate="one_to_one",
        )
        moves = moves.loc[moves.observed_at_utc.gt(opening_time)]
        if moves.empty:
            no_later += 1
            continue
        rows.append(
            {
                "game_id": game_id,
                "opener_timestamp": opening_time,
                "home_open": opener,
                "opener_probability": p_open,
                "market_move_toward_home": float(
                    (moves.line_home_open - moves.line_home_late).median()
                ),
                "opening_books": len(opening),
                "movement_books": len(moves),
                "latest_quote": group.observed_at_utc.max(),
                "latest_snapshot": group.snapshot.max(),
                "latest_provider_time": group.provider_time.max(),
            }
        )
    audit["games_without_later_same_book_pair"] = no_later
    require(bool(rows), "No eligible paired opener/later quote games")
    frame = candidates.merge(pd.DataFrame(rows), on="game_id", validate="one_to_one")
    frame["hours_to_kickoff"] = (frame.kickoff - frame.opener_timestamp).dt.total_seconds() / 3600
    frame["opener_logit"] = np.log(frame.opener_probability / (1 - frame.opener_probability))
    frame["move_log_hours"] = frame.market_move_toward_home * np.log(frame.hours_to_kickoff)
    require(frame.hours_to_kickoff.gt(0).all(), "Nonpositive opener horizon")
    require(
        frame.latest_snapshot.lt(frame.kickoff).all()
        and frame.latest_provider_time.le(frame.latest_quote).all()
        and frame.latest_quote.lt(frame.kickoff).all(),
        "Feature timestamp guard failed",
    )
    require(np.isfinite(frame[[*BASE, "move_log_hours"]]).all().all(), "Nonfinite predictors")
    require(set(frame.season.unique()) == set(SEASONS), "Missing a declared season")
    audit["eligible_before_outcomes_by_season"] = frame.groupby("season").size().to_dict()
    audit["hours_min_median_max"] = [
        float(x) for x in frame.hours_to_kickoff.agg(["min", "median", "max"])
    ]
    age = (frame.kickoff - frame.latest_quote).dt.total_seconds() / 3600
    audit["quote_to_kickoff_hours_min_median_max"] = [
        float(x) for x in age.agg(["min", "median", "max"])
    ]
    audit["neutral_site_games"] = int(frame.neutral_site.fillna(False).sum())
    audit["fbs_vs_fbs_games"] = int(
        (frame.home_division.eq("fbs") & frame.away_division.eq("fbs")).sum()
    )
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True), audit


def attach_outcomes(frame: pd.DataFrame, audit: dict) -> pd.DataFrame:
    outcomes = pd.concat(
        [
            pd.read_parquet(
                SCHEDULES / f"season={s}/schedules.parquet",
                columns=["game_id", "home_points", "away_points"],
                use_threads=False,
            )
            for s in SEASONS
        ],
        ignore_index=True,
    )
    frame = frame.merge(outcomes, on="game_id", validate="one_to_one")
    require(frame[["home_points", "away_points"]].notna().all().all(), "Completed game lacks score")
    frame["margin_vs_open"] = frame.home_points - frame.away_points + frame.home_open
    audit["pushes_by_season"] = (
        frame.loc[frame.margin_vs_open.eq(0)].groupby("season").size().to_dict()
    )
    frame = frame.loc[frame.margin_vs_open.ne(0)].copy().reset_index(drop=True)
    frame["home_covered"] = frame.margin_vs_open.gt(0).astype(float)
    audit["decisive_by_season"] = frame.groupby("season").size().to_dict()
    return frame


def fit(train: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict]:
    means = train[features].mean().to_numpy(dtype=float)
    scales = train[features].std(ddof=0).to_numpy(dtype=float)
    scales = np.where(scales > 0, scales, 1.0)
    design = np.column_stack([np.ones(len(train)), (train[features].to_numpy() - means) / scales])
    beta = _fit_logit(design, train.home_covered.to_numpy(), FIT_RIDGE)
    target_design = np.column_stack(
        [np.ones(len(target)), (target[features].to_numpy() - means) / scales]
    )
    probability = 1 / (1 + np.exp(-np.clip(target_design @ beta, -35, 35)))
    coefficients = {"intercept": float(beta[0] - np.sum(beta[1:] * means / scales))}
    coefficients.update(dict(zip(features, (beta[1:] / scales).tolist(), strict=True)))
    return probability, coefficients


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, np.ndarray]:
    p, y = np.clip(probability, 1e-12, 1 - 1e-12), frame.home_covered.to_numpy()
    return {
        "accuracy_points": 100 * ((p >= 0.5) == y).astype(float),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
    }


def bootstrap_weights(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, blocks = pd.factorize(pd.MultiIndex.from_frame(frame[["season", "week"]]), sort=True)
    weights = np.zeros((RESAMPLES, len(blocks)))
    rng = np.random.default_rng(SEED)
    for season in SEASONS:
        indices = np.flatnonzero(blocks.get_level_values(0) == season)
        weights[:, indices] = rng.multinomial(
            len(indices), np.full(len(indices), 1 / len(indices)), size=RESAMPLES
        )
    return ids, weights, weights @ np.bincount(ids, minlength=len(blocks))


def summary(values: np.ndarray, boot: tuple, directional: bool = False) -> dict:
    ids, weights, sizes = boot
    draws = (weights @ np.bincount(ids, weights=values, minlength=weights.shape[1])) / sizes
    result = {
        "estimate": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "standard_error": float(draws.std(ddof=1)),
    }
    if directional:
        result["probability_positive"] = float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0))
    return result


def table(headers: list[str], rows: list[list]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        "",
    ]


def interval(value: dict) -> str:
    return f"{value['estimate']:+.6f} [{value['low']:+.6f}, {value['high']:+.6f}]"


def main() -> None:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with threadpool_limits(limits=2):
        frame, audit = load_predictors()
        frame = attach_outcomes(frame, audit)
        coefficients = []
        for mode in ("is", "oos"):
            frame[f"{mode}_opener"] = frame.opener_probability
            for arm in ("base", "horizon"):
                features = BASE + (["move_log_hours"] if arm == "horizon" else [])
                if mode == "is":
                    p, beta = fit(frame, frame, features)
                    frame[f"{mode}_{arm}"] = p
                    coefficients.append(
                        {"held_season": "in_sample", "arm": arm, "training_n": len(frame), **beta}
                    )
                else:
                    for season in SEASONS:
                        held = frame.season.eq(season)
                        p, beta = fit(frame.loc[~held], frame.loc[held], features)
                        frame.loc[held, f"{mode}_{arm}"] = p
                        coefficients.append(
                            {
                                "held_season": season,
                                "arm": arm,
                                "training_n": int((~held).sum()),
                                **beta,
                            }
                        )
        boot = bootstrap_weights(frame)
        arrays, absolute, effects = {}, {}, {}
        for mode in ("is", "oos"):
            for arm in ARMS:
                arrays[mode, arm] = metrics(frame, frame[f"{mode}_{arm}"].to_numpy())
                for metric, values in arrays[mode, arm].items():
                    absolute[f"{mode}_{arm}_{metric}"] = summary(values, boot)
            for metric in ("accuracy_points", "log_loss", "brier"):
                delta = arrays[mode, "horizon"][metric] - arrays[mode, "base"][metric]
                if metric != "accuracy_points":
                    delta = -delta
                effects[f"{mode}_{metric}"] = summary(delta, boot, True)
        season_rows = []
        for season, group in frame.groupby("season"):
            results = {arm: metrics(group, group[f"oos_{arm}"].to_numpy()) for arm in ARMS}
            season_rows.append(
                [
                    int(season),
                    len(group),
                    *[f"{results[arm]['accuracy_points'].mean():.3f}" for arm in ARMS],
                    (
                        f"{
                            (
                                results['horizon']['accuracy_points']
                                - results['base']['accuracy_points']
                            ).mean():+.3f}"
                    ),
                    f"{(results['base']['log_loss'] - results['horizon']['log_loss']).mean():+.6f}",
                    f"{(results['base']['brier'] - results['horizon']['brier']).mean():+.6f}",
                ]
            )
        calibration = []
        bins = np.linspace(0, 1, 11)
        for arm in ARMS:
            p = frame[f"oos_{arm}"].to_numpy()
            cells = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 9)
            for cell in range(10):
                mask = cells == cell
                if mask.any():
                    calibration.append(
                        [
                            arm,
                            f"{bins[cell]:.1f}-{bins[cell + 1]:.1f}",
                            int(mask.sum()),
                            f"{p[mask].mean():.6f}",
                            f"{frame.loc[mask, 'home_covered'].mean():.6f}",
                        ]
                    )
        SCRATCH.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(SCRATCH / "predictions.parquet", index=False)
        payload = {
            "audit": audit,
            "coefficients": coefficients,
            "absolute": absolute,
            "effects": effects,
            "games": len(frame),
            "blocks": boot[1].shape[1],
            "looks": 1,
            "nfl_looks": 0,
        }
        (SCRATCH / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_report(frame, payload, season_rows, calibration)
        print(
            json.dumps(
                {
                    "status": "completed_proxy_replication",
                    "games": len(frame),
                    "week_blocks": boot[1].shape[1],
                    "effects": {k: v for k, v in effects.items() if k.startswith("oos_")},
                    "horizon_coefficients": [r for r in coefficients if r["arm"] == "horizon"],
                    "report": str(REPORT),
                    "predictions": str(SCRATCH / "predictions.parquet"),
                },
                indent=2,
            )
        )


def write_report(frame: pd.DataFrame, payload: dict, season_rows: list, calibration: list) -> None:
    audit, effects = payload["audit"], payload["effects"]
    decisive_rows = []
    for mode in ("is", "oos"):
        for arm in ARMS:
            wins = int(((frame[f"{mode}_{arm}"] >= 0.5) == frame.home_covered).sum())
            decisive_rows.append([mode, arm, wins, len(frame) - wins, len(frame)])
    lines = [
        "# LEAD-69 CFB horizon replication",
        "",
        (
            "**Measured:** first-observed-opener proxy study completed. Exact "
            "market-opening-time replication remains unavailable: the stored CFB opener table "
            "has no opener timestamp."
        ),
        "",
        "## Decisive games before the effect",
        "",
        *table(["Fit", "Probability", "Correct", "Incorrect", "Decisive games"], decisive_rows),
        (
            f"**Measured:** LOSO horizon-minus-base opener-graded accuracy: "
            f"{interval(effects['oos_accuracy_points'])} percentage points; probability_positive "
            f"{effects['oos_accuracy_points']['probability_positive']:.6f}."
        ),
        "",
        (
            "**Inferred:** provisional unresolved_below_power, pending the orchestrator's serial "
            "registry entry. The proxy study supplies no consistent NFL-sign replication and no "
            "demonstrated improvement; it does not refute the mechanism. No admissible closing "
            "ground has been established. AGENTS.md research rules distinguish signal closure "
            "from serving; this retrospective study changes no served pick. No positive control "
            "or split-half closure was attempted."
        ),
        "",
        "## Protocol and source limits",
        "",
        (
            "Declared in docs/lanes/lead69-cfb-replication.md before CFB outcome access. "
            "Population: completed 2023-2025 regular-season games with at least one FBS team, "
            "including neutral sites, a valid paired opener quote and a later same-book quote. "
            "The frozen grade is the historical first-observed opener proxy, not a 2026 pool "
            "capture."
        ),
        "",
        (
            "**Read:** scripts/lead69_unit1.py:212 and :227 implement move x log(hours) and "
            "training-only standardization followed by ridge logistic regression. This study "
            "uses that form and FIT_RIDGE=0.001, with no tuning. Base = intercept + "
            "opener-implied home-cover logit + market move toward home; challenger adds move x "
            "log(hours from observed opener to conservative kickoff). One fitted probability "
            "chooses the side at 0.5."
        ),
        "",
        (
            "**Measured:** each opener is an actual home spread at the earliest paired "
            "historical snapshot (lower observed median). Its implied probability is the median "
            "of home/away de-vigged prices only at that exact line. Movement is the median "
            "same-book opening-home-line minus latest-home-line among opener books with a later "
            "valid pair. Positive movement means the market moved toward home. This avoids "
            "inventing a smooth spread-to-probability mapping."
        ),
        "",
        (
            "**Measured:** features are built before scores are loaded. Provider clocks must be "
            "at or before capture; captures and snapshots must precede the earlier of schedule "
            "kickoff and quoted kickoff. Schedule join uses normalized team names and kickoff "
            "separation at most 36 hours; ambiguous matches fail closed. Prices must have valid "
            "two-sided opposite spreads. Pushes are excluded from binary fitting and scoring."
        ),
        "",
        (
            "**Read:** the CFB line loader in scripts/opener_error_transfer_eval.py:89 uses the "
            "stored opening_lines field but supplies no opening timestamp. Schedule source: "
            "data/cfb/schedules/raw/20260816T162105Z; quote source: "
            "data/market/raw/*-ncaaf/quotes.parquet and adjacent historical-response manifests. "
            "Retrospective ingestion does not certify real-time availability in an archive that "
            "existed then. First observation may be days after actual opening, and some last "
            "quotes precede kickoff by days; those limits prevent a claim of exact NFL timing "
            "replication."
        ),
        "",
        (
            "One declared CFB look, zero NFL looks: one continuous interaction against a fixed "
            "base, three predeclared metrics. The three LOSO folds and fixed reliability bins "
            "are descriptive diagnostics, not searched alternatives. Each held season is "
            "excluded from coefficient and standardization fitting. In-sample fits use all "
            "seasons and are diagnostic only. LOSO can train on later seasons, so this is "
            "season-held-out replication, not a forward deployment simulation."
        ),
        "",
        (
            "Uncertainty: 4,000 paired bootstrap resamples of (season, week) blocks, sampled "
            "within season, seed 6902026, 95% percentile intervals. These resample fixed "
            "held-out predictions and do not refit models. probability_positive gives half "
            "credit to exact ties. Positive improvements always favor the horizon term. Only "
            "three seasons limit season-level inference; no cross-league pooling was performed."
        ),
        "",
        "## Population audit",
        "",
        *table(
            ["Quantity", "Measured value"],
            [[key, json.dumps(value)] for key, value in audit.items()],
        ),
        "## Accuracy, probability quality, and fitting gap",
        "",
    ]
    absolute_rows = []
    for arm in ARMS:
        for metric in ("accuracy_points", "log_loss", "brier"):
            inside = payload["absolute"][f"is_{arm}_{metric}"]
            outside = payload["absolute"][f"oos_{arm}_{metric}"]
            absolute_rows.append(
                [
                    arm,
                    metric,
                    interval(inside),
                    interval(outside),
                    f"{inside['estimate'] - outside['estimate']:+.6f}",
                ]
            )
    lines += table(
        ["Probability", "Metric", "In sample [95% CI]", "LOSO [95% CI]", "IS - OOS"], absolute_rows
    )
    lines += [
        (
            "**Measured:** paired improvements below use identical games and week draws. "
            "Log-loss and Brier improvements are base minus challenger; accuracy is challenger "
            "minus base in percentage points."
        ),
        "",
    ]
    lines += table(
        ["Fit", "Improvement", "Estimate [95% CI]", "probability_positive"],
        [
            [
                mode,
                metric,
                interval(effects[f"{mode}_{metric}"]),
                f"{effects[f'{mode}_{metric}']['probability_positive']:.6f}",
            ]
            for mode in ("is", "oos")
            for metric in ("accuracy_points", "log_loss", "brier")
        ],
    )
    lines += [
        "## Season stability",
        "",
        *table(
            [
                "Held season",
                "N",
                "Opener accuracy %",
                "Base accuracy %",
                "Horizon accuracy %",
                "Accuracy gain pp",
                "Log-loss gain",
                "Brier gain",
            ],
            season_rows,
        ),
        "## Natural-unit coefficients",
        "",
        (
            "**Measured:** all coefficients below are on the original feature scale. NFL "
            "comparison values are read from docs/lead69_results.md:173-175; negative is the NFL "
            "direction in all three folds."
        ),
        "",
    ]
    coefficient_rows, sign_agreements = [], 0
    for row in payload["coefficients"]:
        year, same = row["held_season"], "-"
        if row["arm"] == "horizon" and year in SEASONS:
            same = "yes" if row["move_log_hours"] < 0 else "no"
            sign_agreements += int(same == "yes")
        coefficient_rows.append(
            [
                year,
                row["arm"],
                row["training_n"],
                *[f"{row.get(term, 0):+.9f}" for term in ("intercept", *BASE, "move_log_hours")],
                f"{NFL_BETAS[year]:+.9f}" if year in SEASONS else "-",
                same,
            ]
        )
    lines += table(
        [
            "Held season",
            "Fit",
            "Training N",
            "Intercept",
            "Opener logit",
            "Move",
            "Move x log(hours)",
            "NFL horizon",
            "Same sign",
        ],
        coefficient_rows,
    )
    lines += [
        (
            f"**Measured:** horizon sign agrees with the corresponding NFL fold in "
            f"{sign_agreements}/3 held seasons. Sign agreement is mechanism context; it is not "
            f"three independent looks or evidence of profitable betting."
        ),
        "",
        "## Reliability on held-out games",
        "",
        (
            "**Measured:** fixed 0.1-wide bins describe home-cover calibration. Probability "
            "scores above compare both fitted models with the raw opener-implied market "
            "baseline. Sparse bins are not selected findings."
        ),
        "",
        *table(
            ["Probability", "Bin", "N", "Mean predicted home cover", "Observed home cover"],
            calibration,
        ),
        "## Reproduction and artifacts",
        "",
        (
            "Command: `.tools/uv.exe run --no-sync --no-cache python "
            "scripts/lead69_cfb_replication.py` (one scoring run; cache disabled because the "
            "shared user cache is outside the writable workspace)."
        ),
        "",
        (
            "Prediction rows and full summary: "
            "`tests/scratch/codex/lead69_cfb_replication/predictions.parquet` and "
            "`summary.json`. No prediction-row dump is written under docs/. BLAS and Arrow are "
            "limited to two threads. The script never writes the registry, publishes, or changes "
            "the served card. Exact bash-compatible registry commands are handed to the "
            "orchestrator in the lane."
        ),
        "",
    ]
    long_horizons = int(frame.hours_to_kickoff.gt(336).sum())
    quote_age = (frame.kickoff - frame.latest_quote).dt.total_seconds() / 3600
    stale_quotes = int(quote_age.gt(168).sum())
    lines += [
        "## Review and preserved predeclaration",
        "",
        (
            f"**Measured, source review:** {long_horizons} decisive games have observed-opener "
            f"horizons above 14 days; {stale_quotes} last quotes are more than seven days old "
            f"(maximum {quote_age.max():,.2f} hours). Maximum opener horizon is "
            f"{frame.hours_to_kickoff.max():,.2f} hours. No freshness cutoff was added after "
            f"outcomes were seen. Ambiguous provider joins are excluded before scoring, as "
            f"counted above."
        ),
        "",
        (
            "Protocol declared before outcome inspection. Population: FBS regular-season games "
            "in 2023\u20132025 with a historical opener and a pre-kickoff quote. Frozen "
            "pool-line proxy: historical opener. Horizon: hours from opener timestamp to "
            "kickoff. Target: home covers the opener, excluding pushes from binary fitting and "
            "accuracy. Base: opener-implied probability plus market move toward home. "
            "Challenger: the same fitted form as LEAD-69, adding its market-move \u00d7 horizon "
            "interaction. All fitted parameters and transformations use training seasons only; "
            "leave one season out (LOSO). Report in-sample and out-of-sample metrics and gap, "
            "fold coefficients and sign agreement with NFL, log loss, Brier, opener-graded "
            "accuracy, season stability, paired week-blocked bootstrap intervals and "
            "probability_positive. One declared CFB look; no NFL look. A zero-crossing interval "
            "cannot close the signal. One fitted probability selects the side; no standalone "
            "flip.\n\nPre-outcome implementation declaration: include regular-season completed "
            "games with at least one FBS team (neutral sites included). At the earliest valid "
            "event snapshot, opener = an actual quoted median home line (lower middle if tied), "
            "probability = median two-sided de-vigged home-cover price among books quoting that "
            "exact line. Move toward home = median same-book (opening home line \u2212 latest "
            "home line), using opener books with a later pre-kickoff pair. Both "
            "snapshot/provider clocks precede the conservative kickoff. Fit intercept + opener "
            "logit + move; add move \u00d7 log(opener-to-kickoff hours), with the NFL ridge and "
            "training-only standardization. Binary pushes excluded; one 2023\u20132025 LOSO fit "
            "family, no tuning. Use 4,000 paired season-stratified week-block bootstraps (seed "
            "6902026), 95% percentile intervals and half-credit ties for probability_positive. "
            "Fixed probability bins report calibration descriptively; fold/metric summaries are "
            "diagnostics of the single declared look."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
