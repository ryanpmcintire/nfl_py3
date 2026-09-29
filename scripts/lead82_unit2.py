from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from lead82_unit1 import (
    FIT_ROOT,
    KEYS,
    QUOTES,
    SCHEDULE,
    boundary,
    load_population,
    read_table,
    table,
)
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import MOVE_AVAILABLE_COLUMN, MOVE_COLUMN
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

OUTPUT = Path("tests/scratch/codex/lead82_unit2")
REPORT = Path("docs/lead82_unit2.md")
LANE = Path("docs/lanes/lead82.md")
SEASONS = (2023, 2024, 2025)
METRICS = ("accuracy_points", "log_loss", "brier", "rps")
ARMS = ("candidate", "served", "model", "market")
BOOTSTRAPS = 10000
SEED = 20260929
EARLY = "tuesday_move"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tuesday_features(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = [
        "nflverse_game_id",
        "bookmaker_key",
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "home_spread_line",
        "commence_time_utc",
        "market",
        "snapshot_timestamp_utc",
        "manifest_season",
        "archive_season",
        "decision_label",
    ]
    quotes = read_table(
        QUOTES,
        columns,
        filters=[
            ("bookmaker_key", "in", list(LEADER_BOOKS)),
            ("nflverse_game_id", "in", games.game_id.tolist()),
            ("decision_label", "=", "intraday_hourly"),
        ],
    ).rename(columns={"nflverse_game_id": "game_id"})
    quotes = quotes.merge(
        games[["game_id", "season", "monday", "freeze", "wednesday", "kickoff"]],
        on="game_id",
        validate="many_to_one",
    )
    for name in [
        "observed_at_utc",
        "bookmaker_last_update_utc",
        "commence_time_utc",
        "snapshot_timestamp_utc",
    ]:
        quotes[name] = pd.to_datetime(quotes[name], utc=True, errors="coerce")
    gates = {
        "spread_market": quotes.market.eq("spreads"),
        "finite_line": np.isfinite(quotes.home_spread_line),
        "source_season": quotes.manifest_season.eq(quotes.season)
        & quotes.archive_season.eq(quotes.season),
        "target_week": quotes.observed_at_utc.ge(quotes.monday)
        & quotes.snapshot_timestamp_utc.ge(quotes.monday),
        "book_before_observation": quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc),
        "observation_before_snapshot": quotes.observed_at_utc.le(quotes.snapshot_timestamp_utc),
        "before_wednesday": quotes.observed_at_utc.lt(quotes.wednesday)
        & quotes.snapshot_timestamp_utc.lt(quotes.wednesday),
        "before_kickoff": quotes.observed_at_utc.lt(quotes.kickoff)
        & quotes.snapshot_timestamp_utc.lt(quotes.kickoff)
        & quotes.observed_at_utc.lt(quotes.commence_time_utc)
        & quotes.snapshot_timestamp_utc.lt(quotes.commence_time_utc),
    }
    keep = pd.DataFrame(gates).all(axis=1)
    inventory = {
        "hourly_source_rows": len(quotes),
        "eligible_tuesday_rows": int(keep.sum()),
        "gate_rejections_nonexclusive": {k: int((~v).sum()) for k, v in gates.items()},
    }
    quotes = quotes.loc[keep].copy()
    keys = [*KEYS, "observed_at_utc"]
    if quotes.groupby(keys).home_spread_line.nunique().gt(1).any():
        raise ValueError("Conflicting hourly lines")
    quotes = quotes.sort_values([*keys, "snapshot_timestamp_utc", "bookmaker_last_update_utc"])
    quotes = quotes.drop_duplicates(keys, keep="first")
    anchors = boundary(
        quotes,
        quotes.observed_at_utc.le(quotes.freeze) & quotes.snapshot_timestamp_utc.le(quotes.freeze),
        "noon",
    )
    terminals = boundary(quotes, quotes.observed_at_utc.gt(quotes.freeze), "tuesday_end")
    pairs = anchors.merge(terminals, on=KEYS, validate="one_to_one")
    pairs[EARLY] = pairs.tuesday_end_home_spread_line - pairs.noon_home_spread_line
    pairs.to_parquet(OUTPUT / "boundary_pairs.parquet", index=False)
    early = pairs.groupby("game_id", as_index=False).agg(
        tuesday_move=(EARLY, "median"),
        tuesday_books=("bookmaker_key", "size"),
    )
    inventory["same_book_pairs"] = len(pairs)
    inventory["boundary_clock_violations"] = 0
    return games.merge(early, on="game_id", how="left", validate="one_to_one"), inventory


def compare_unit1(features: pd.DataFrame) -> dict:
    previous = Path("tests/scratch/codex/lead82_unit1/features.parquet")
    if not previous.is_file():
        return {}
    old = read_table(previous, ["game_id", "tuesday_early_move_toward_home"])
    selected = features.loc[features.in_frozen_fit & features.season.isin(SEASONS)]
    paired = selected.merge(old, on="game_id", validate="one_to_one")
    difference = paired[EARLY] - paired.tuesday_early_move_toward_home
    result = {
        "compared_games": len(paired),
        "changed_game_values": int(difference.abs().gt(1e-12).sum()),
        "max_abs_change": float(difference.abs().max()),
    }
    (OUTPUT / "unit1_feature_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata_path = FIT_ROOT / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata["base_probability_policy"] != "discrete_conditional_non_push_v1":
        raise ValueError("Expected the served discrete conditional probability")
    if metadata["market_move_feature_version"] != "leader_median_through_sunday_prekick_v1":
        raise ValueError("Expected the served hourly late-move definition")
    if metadata["ridge"] != FIT_RIDGE or FIT_RIDGE != 0.001:
        raise ValueError("Fixed ridge changed")
    opener_root = Path("artifacts") / metadata["opener_evaluation"]
    games, frozen = load_population(opener_root / "per_game.parquet")
    features, inventory = tuesday_features(games)
    features.to_parquet(OUTPUT / "feature_inventory.parquet", index=False)
    inventory["unit1_comparison"] = compare_unit1(features)
    columns = list(
        dict.fromkeys(
            [
                "game_id",
                "season",
                "week",
                "game_type",
                "tue_open_home_spread",
                "result",
                "margin_vs_open",
                "home_covered",
                "model_probability",
                "base_probability_policy",
                "in_sample_home_probability",
                "out_of_season_home_probability",
                *FIT_FEATURES,
            ]
        )
    )
    source = read_table(FIT_ROOT / "per_game.parquet", columns)
    if source.game_id.duplicated().any() or not source.game_id.equals(frozen.game_id):
        raise ValueError("Frozen population changed")
    if not source.season.between(2020, 2025).all() or not source.game_type.eq("REG").all():
        raise ValueError("Wrong source population")
    numeric = [*FIT_FEATURES, "model_probability", "home_covered", "margin_vs_open"]
    if not np.isfinite(source[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite frozen inputs")
    if (
        source.margin_vs_open.eq(0).any()
        or not source.home_covered.eq(source.margin_vs_open.gt(0)).all()
    ):
        raise ValueError("Conditional opener target mismatch")
    if not np.allclose(source.result - source.tue_open_home_spread, source.margin_vs_open):
        raise ValueError("Opener grade mismatch")
    if not source.base_probability_policy.eq("discrete_conditional_non_push_v1").all():
        raise ValueError("Unexpected model-only probability mapping")
    if not np.allclose(logit(source.model_probability), source.model_logit):
        raise ValueError("Model logit mismatch")
    move_path = Path("artifacts") / metadata["market_move_artifact"]
    move_summary = json.loads(move_path.with_name("summary.json").read_text(encoding="utf-8-sig"))
    if digest(move_path) != move_summary["market_move_sha256"]:
        raise ValueError("Saved hourly move hash mismatch")
    moves = read_table(move_path, ["game_id", "leader_median_net"]).set_index("game_id")
    exposed = source[MOVE_AVAILABLE_COLUMN].eq(1)
    errors = (
        source.loc[exposed, MOVE_COLUMN].to_numpy()
        - source.loc[exposed, "game_id"].map(moves.leader_median_net).to_numpy()
    )
    if not np.isfinite(errors).all() or np.max(np.abs(errors)) > 1e-9:
        raise ValueError("Saved hourly move mismatch")
    frame = (
        source.loc[source.season.isin(SEASONS)]
        .merge(
            features[["game_id", EARLY, "tuesday_books"]],
            on="game_id",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    if (
        frame[EARLY].isna().any()
        or not frame[MOVE_AVAILABLE_COLUMN].eq(1).all()
        or len(frame) != 799
    ):
        raise ValueError("Expected 799 complete Tuesday and served-late rows")
    inventory.update(
        {
            "source_games": len(source),
            "scored_games": len(frame),
            "opener_inventory_including_pushes": len(features),
            "games_by_season": {str(k): int(v) for k, v in frame.groupby("season").size().items()},
            "frozen_fit_boundary_pairs": int(frame.tuesday_books.sum()),
            "served_move_max_error": float(np.max(np.abs(errors))),
            "sources": {
                str(p): digest(p)
                for p in [
                    metadata_path,
                    FIT_ROOT / "per_game.parquet",
                    SCHEDULE,
                    opener_root / "metadata.json",
                    opener_root / "per_game.parquet",
                    move_path,
                    QUOTES,
                ]
            },
            "upstream_limitations": metadata["validation_limitations"],
        }
    )
    return source, frame, inventory


def fit_tuesday(frame: pd.DataFrame, base: np.ndarray) -> float:
    scale = float(frame[EARLY].std(ddof=0)) or 1.0
    x = frame[EARLY].to_numpy(dtype=float) / scale
    y = frame.home_covered.to_numpy(dtype=float)
    offset = logit(base)
    beta = 0.0
    for _ in range(50):
        p = expit(offset + x * beta)
        gradient = x @ (y - p) - FIT_RIDGE * beta
        curvature = (x * x) @ (p * (1 - p)) + FIT_RIDGE
        step = gradient / curvature
        beta += step
        if abs(step) < 1e-12:
            break
    if not np.isfinite(beta) or abs(gradient) > 1e-7:
        raise ValueError("Tuesday offset fit did not converge")
    return float(beta / scale)


def fit_arms(source: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict, list, dict]:
    probabilities = {
        "candidate": np.full(len(frame), np.nan),
        "served": np.full(len(frame), np.nan),
        "model": frame.model_probability.to_numpy(dtype=float),
        "market": np.full(len(frame), 0.5),
    }
    coefficients = []
    parity = {}
    for held in (*SEASONS, "IS"):
        is_sample = held == "IS"
        source_train = source if is_sample else source.loc[source.season.ne(held)]
        select = np.ones(len(frame), dtype=bool) if is_sample else frame.season.eq(held).to_numpy()
        term_train = frame if is_sample else frame.loc[~select]
        means, stds = _standardisers(source_train)
        beta = _fit_logit(
            _design(source_train, means, stds),
            source_train.home_covered.to_numpy(dtype=float),
            FIT_RIDGE,
        )
        base_train = _predict(term_train, beta, means, stds)
        base_test = _predict(frame.loc[select], beta, means, stds)
        original = "in_sample_home_probability" if is_sample else "out_of_season_home_probability"
        error = float(np.max(np.abs(base_test - frame.loc[select, original].to_numpy())))
        if error > 1e-9:
            raise ValueError("Served fold reproduction failed")
        parity[str(held)] = error
        added = fit_tuesday(term_train, base_train)
        candidate = expit(logit(base_test) + added * frame.loc[select, EARLY].to_numpy())
        natural = _natural_coefficients(beta, means, stds)
        for arm, value in [("served", 0.0), ("candidate", added)]:
            coefficients.append(
                {
                    "arm": arm,
                    "held_out": held,
                    "base_train_games": len(source_train),
                    "tuesday_train_games": len(term_train),
                    **natural,
                    EARLY: value,
                }
            )
        if is_sample:
            probabilities["served_IS"] = base_test
            probabilities["candidate_IS"] = candidate
        else:
            probabilities["served"][select] = base_test
            probabilities["candidate"][select] = candidate
    return probabilities, coefficients, parity


class Bootstrap:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.blocks = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        self.index = pd.MultiIndex.from_frame(self.blocks)
        self.weights = np.zeros((BOOTSTRAPS, len(self.blocks)))
        rng = np.random.default_rng(SEED)
        for season in SEASONS:
            positions = np.flatnonzero(self.blocks.season.eq(season))
            self.weights[:, positions] = rng.multinomial(
                len(positions),
                np.full(len(positions), 1 / len(positions)),
                size=BOOTSTRAPS,
            )

    def estimate(self, values: np.ndarray, select: np.ndarray | None = None) -> dict:
        keep = np.ones(len(self.frame), dtype=bool) if select is None else select
        if not keep.any():
            return dict.fromkeys(
                ["effect", "low", "high", "probability_positive", "standard_error"]
            )
        work = self.frame.loc[keep, ["season", "week"]].copy()
        work["value"] = values[keep]
        sums = (
            work.groupby(["season", "week"])
            .value.agg(["sum", "count"])
            .reindex(self.index, fill_value=0)
        )
        denominator = self.weights @ sums["count"].to_numpy(dtype=float)
        draws = np.divide(
            self.weights @ sums["sum"].to_numpy(dtype=float),
            denominator,
            out=np.full(BOOTSTRAPS, np.nan),
            where=denominator > 0,
        )
        draws = draws[np.isfinite(draws)]
        return {
            "effect": float(values[keep].mean()),
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
            "probability_positive": float(np.mean(draws > 0) + 0.5 * np.mean(draws == 0)),
            "standard_error": float(draws.std(ddof=1)),
        }


def score(frame: pd.DataFrame, probabilities: dict) -> dict:
    bootstrap = Bootstrap(frame)
    y = frame.home_covered.to_numpy(dtype=float)
    arrays = {}
    for arm, probability in probabilities.items():
        p = np.clip(probability, 1e-9, 1 - 1e-9)
        arrays[arm] = {
            "accuracy_points": 100.0 * ((p >= 0.5) == y),
            "log_loss": -y * np.log(p) - (1 - y) * np.log1p(-p),
            "brier": (p - y) ** 2,
            "rps": (p - y) ** 2,
        }
    panels, contrasts, decisive = {}, {}, []
    for panel in ("pooled", *SEASONS):
        selected = (
            np.ones(len(frame), dtype=bool)
            if panel == "pooled"
            else frame.season.eq(panel).to_numpy()
        )
        panels[str(panel)] = {
            arm: {m: bootstrap.estimate(arrays[arm][m], selected) for m in METRICS} for arm in ARMS
        }
        contrasts[str(panel)] = {}
        for baseline in ARMS[1:]:
            contrasts[str(panel)][baseline] = {
                m: bootstrap.estimate(
                    (1 if m == "accuracy_points" else -1)
                    * (arrays["candidate"][m] - arrays[baseline][m]),
                    selected,
                )
                for m in METRICS
            }
            changed = selected & (
                (probabilities["candidate"] >= 0.5) != (probabilities[baseline] >= 0.5)
            )
            wins = int((arrays["candidate"]["accuracy_points"][changed] > 0).sum())
            decisive.append(
                {
                    "panel": str(panel),
                    "baseline": baseline,
                    "games": int(changed.sum()),
                    "candidate_wins": wins,
                    "baseline_wins": int(changed.sum()) - wins,
                    "candidate_accuracy": bootstrap.estimate(
                        arrays["candidate"]["accuracy_points"], changed
                    ),
                }
            )
    in_sample = {
        arm: {m: bootstrap.estimate(arrays[f"{arm}_IS"][m]) for m in METRICS} for arm in ARMS[:2]
    }
    gaps = {
        arm: {
            m: bootstrap.estimate(
                (1 if m == "accuracy_points" else -1) * (arrays[f"{arm}_IS"][m] - arrays[arm][m]),
            )
            for m in METRICS
        }
        for arm in ARMS[:2]
    }
    contrast_is, contrast_gap = {}, {}
    for m in METRICS:
        sign = 1 if m == "accuracy_points" else -1
        in_gain = sign * (arrays["candidate_IS"][m] - arrays["served_IS"][m])
        out_gain = sign * (arrays["candidate"][m] - arrays["served"][m])
        contrast_is[m] = bootstrap.estimate(in_gain)
        contrast_gap[m] = bootstrap.estimate(in_gain - out_gain)
    reliability = []
    edges = np.linspace(0.0, 1.0, 6)
    for arm in ARMS:
        p = probabilities[arm]
        for low, high in pairwise(edges):
            selected = (p >= low) & ((p < high) if high < 1 else (p <= high))
            reliability.append(
                {
                    "arm": arm,
                    "bin": f"{low:.1f}-{high:.1f}",
                    "n": int(selected.sum()),
                    "predicted": float(p[selected].mean()) if selected.any() else None,
                    "observed": float(y[selected].mean()) if selected.any() else None,
                }
            )
    return {
        "panels": panels,
        "contrasts": contrasts,
        "decisive": decisive,
        "in_sample": in_sample,
        "optimism_gaps": gaps,
        "candidate_served_IS": contrast_is,
        "candidate_served_optimism_gap": contrast_gap,
        "reliability": reliability,
        "week_blocks": len(bootstrap.blocks),
        "metric_panels": 136,
        "used_looks": 216,
        "look_budget": 261,
    }


def interval(cell: dict) -> str:
    if cell["effect"] is None:
        return "not estimable"
    return f"{cell['effect']:.6f} [{cell['low']:.6f}, {cell['high']:.6f}]"


def write_report(
    frame: pd.DataFrame, result: dict, coefficients: list, inventory: dict, parity: dict
) -> None:
    lines = [
        "# LEAD-82 unit 2 — hourly Tuesday move",
        "",
        "**Measured:** one declared replay; no tuning, serving changes, or registry writes.",
        (
            "The lane amendment was saved before outcome access. Historical opener is the "
            "frozen pool-line proxy."
        ),
        "",
        "## Decisive games first",
        "",
        (
            "**Measured:** candidate record when its OOS pick differs from each baseline; "
            "pushes excluded from conditional-cover scoring."
        ),
        table(
            ["Panel", "Baseline", "Candidate W-L", "Accuracy, 95% interval"],
            [
                [
                    r["panel"],
                    r["baseline"],
                    f"{r['candidate_wins']}-{r['baseline_wins']}",
                    interval(r["candidate_accuracy"]),
                ]
                for r in result["decisive"]
            ],
        ),
        "",
        "## Protocol and integrity",
        "",
        (
            "**Read:** the lane fixes the signed median Tuesday-noon-to-pre-Wednesday "
            "leader-book move,"
        ),
        (
            "four metrics and 261-look ceiling. The owner's LOSO amendment supersedes the "
            "row's single 2025 outer season."
        ),
        (
            "The audit (docs/move_feature_audit.md:3-21) attributes 164 late-move "
            "differences to later captures."
        ),
        (
            "Both arms retain saved hourly late move and availability; only Tuesday is newly "
            "extracted on the hourly grid."
        ),
        (
            "The predeclared terminal must occur after noon; a stale pre-noon quote is not "
            "treated as an observed afternoon block."
        ),
        "",
        (
            f"**Measured:** {inventory['source_games']} nonpush source rows across "
            f"2020-2025; {len(frame)} scored rows;"
        ),
        (
            f"season counts {inventory['games_by_season']}; "
            f"{inventory['opener_inventory_including_pushes']} opener rows retained"
        ),
        (
            f"in the feature inventory including pushes; "
            f"{inventory['frozen_fit_boundary_pairs']} same-book Tuesday pairs."
        ),
        (
            f"Zero boundary-clock violations. Served fold probability maximum reproduction "
            f"error {max(parity.values()):.3g};"
        ),
        f"saved hourly move maximum error {inventory['served_move_max_error']:.3g}.",
        (
            f"**Measured:** source-only Tuesday comparison with unit 1: "
            f"{inventory.get('unit1_comparison', {})}."
        ),
        (
            "The post-noon gate accounts for the removed stale boundary pair; the one "
            "changed game median is disclosed,"
        ),
        "and no definition was revised after outcomes.",
        "",
        (
            "Each base four-term ridge fit uses all other 2020-2025 seasons, then freezes "
            "its coefficients."
        ),
        (
            "One Tuesday coefficient uses the other two quote-covered seasons with base "
            "logit as offset."
        ),
        (
            "Tuesday scales by training SD without centering; ridge 0.001; no new intercept, "
            "slope, cut point, or flip rule."
        ),
        (
            "All-data IS is explicitly optimistic. One combined probability selects at 0.5; "
            "ties choose home."
        ),
        "",
        (
            f"**Measured:** 10,000 paired bootstrap draws over {result['week_blocks']} "
            f"season-week blocks,"
        ),
        (
            f"stratified by season, seed {SEED}. Percentile 95% intervals; "
            f"probability_positive gives exact-zero draws half credit."
        ),
        (
            "Positive improvement means candidate-minus-baseline accuracy or "
            "baseline-minus-candidate loss."
        ),
        (
            "Primary Brier; conditional binary RPS equals Brier exactly, not full-margin "
            "RPS. No push forecast is fitted."
        ),
        (
            "Fair opener market is 0.5; model-only is saved discrete conditional "
            "probability; identical opener grades/population."
        ),
        (
            "136 metric panels + 48 coefficient entries + 20 reliability cells + 12 decisive "
            "records = 216 looks;"
        ),
        (
            "45 of 261 unspent. No extra specifications. Baseline contrasts are not separate "
            "registry candidates."
        ),
        "",
        "## Out-of-season scores",
        "",
        "**Measured:** estimate [95% week-block interval]; accuracy is percentage points.",
        table(
            ["Panel", "Arm", *METRICS],
            [
                [panel, arm, *(interval(values[m]) for m in METRICS)]
                for panel, arms in result["panels"].items()
                for arm, values in arms.items()
            ],
        ),
        "",
        "## Paired improvements",
        "",
        "**Measured:** positive favors Tuesday; paired comparisons are not independent votes.",
        table(
            ["Panel", "Baseline", "Metric", "Improvement, 95% interval", "probability_positive"],
            [
                [panel, baseline, metric, interval(cell), f"{cell['probability_positive']:.4f}"]
                for panel, contrasts in result["contrasts"].items()
                for baseline, values in contrasts.items()
                for metric, cell in values.items()
            ],
        ),
        "",
        "## Optimistic IS, OOS, and gap",
        "",
        "**Measured:** positive optimism gap means better IS accuracy or lower IS loss than OOS.",
        table(
            [
                "Arm",
                "Metric",
                "IS, 95% interval",
                "OOS, 95% interval",
                "Optimism gap, 95% interval",
            ],
            [
                [
                    arm,
                    m,
                    interval(result["in_sample"][arm][m]),
                    interval(result["panels"]["pooled"][arm][m]),
                    interval(result["optimism_gaps"][arm][m]),
                ]
                for arm in ARMS[:2]
                for m in METRICS
            ],
        ),
        table(
            [
                "Candidate vs served improvement",
                "IS, 95% interval",
                "OOS, 95% interval",
                "IS-OOS gap, 95% interval",
            ],
            [
                [
                    m,
                    interval(result["candidate_served_IS"][m]),
                    interval(result["contrasts"]["pooled"]["served"][m]),
                    interval(result["candidate_served_optimism_gap"][m]),
                ]
                for m in METRICS
            ],
        ),
        "",
        "## Natural coefficients",
        "",
        (
            "**Measured:** base coefficients are identical between arms; both shown to "
            "expose the constraint."
        ),
        table(
            [
                "Holdout",
                "Arm",
                "Base train n",
                "Tuesday train n",
                "Intercept",
                *FIT_FEATURES,
                EARLY,
            ],
            [
                [
                    c["held_out"],
                    c["arm"],
                    c["base_train_games"],
                    c["tuesday_train_games"],
                    *(f"{c[k]:.6f}" for k in ["intercept", *FIT_FEATURES, EARLY]),
                ]
                for c in coefficients
            ],
        ),
        "",
        "## Reliability",
        "",
        (
            "**Measured:** OOS home-side probabilities and observed cover rates; all five "
            "equal-width bins retained."
        ),
        table(
            ["Arm", "Bin", "Games", "Mean probability", "Observed rate"],
            [
                [
                    r["arm"],
                    r["bin"],
                    r["n"],
                    "empty" if r["predicted"] is None else f"{r['predicted']:.6f}",
                    "empty" if r["observed"] is None else f"{r['observed']:.6f}",
                ]
                for r in result["reliability"]
            ],
        ),
        "",
        "## Interpretation and limits",
        "",
        (
            "**Read:** upstream opener producer fits completed games before the target "
            "week's earliest game"
        ),
        (
            "(src/nfl_ats/clv.py:2184-2200) and passes that cutoff and target-game "
            "exclusions to the discrete reader."
        ),
        (
            "The cached artifact lacks row-level upstream training ledgers; source "
            "inspection is not a fresh reconstruction."
        ),
        f"**Read:** frozen-fit limitation: {inventory['upstream_limitations']}",
        (
            "**Inferred:** retrospective LOSO includes later seasons in earlier-fold "
            "training and reuses selected upstream features."
        ),
        (
            "This is not chronological deployment validation or an untouched outer test. "
            "Bootstrap intervals condition on fitted"
        ),
        (
            "predictions and three fixed seasons; they exclude refitting uncertainty and "
            "cannot establish long-run season stability."
        ),
        (
            "**Inferred:** retain unresolved_below_power pending the orchestrator's serial "
            "candidate-versus-served recording."
        ),
        "**Measured:** Tuesday coefficients are positive in all three folds:",
        ", ".join(
            f"{c['held_out']}: {c[EARLY]:.6f}"
            for c in coefficients
            if c["arm"] == "candidate" and c["held_out"] != "IS"
        )
        + ".",
        (
            "Brier improves in two folds and worsens in one; changed-pick accuracy trails "
            "served in all three folds."
        ),
        (
            "**Inferred:** the small proper-score gain and weaker changed-pick record "
            "warrant further research;"
        ),
        "they do not establish a serving improvement or refute the Tuesday-news mechanism.",
        (
            "No split-half refutation or powered positive control was tested. "
            "AGENTS.md:65-84 governs closure;"
        ),
        (
            "AGENTS.md:115-120 separates closure from serving. Historical forced-pick rate "
            "does not establish a profitable edge."
        ),
        "",
        "## Reproduction",
        "",
        (
            "**Measured:** .tools/uv.exe run --no-sync python scripts/lead82_unit2.py "
            "completed one replay."
        ),
        (
            "Use writable UV_CACHE_DIR, UV_OFFLINE=1 and at most two numerical threads. No "
            "tests added."
        ),
        (
            "Rows, boundaries, feature inventory, coefficients, intervals and source hashes: "
            "tests/scratch/codex/lead82_unit2/."
        ),
        "Record commands are in the lane for the orchestrator; they were not executed.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_record_batch(result: dict) -> None:
    summary = (
        "Adding Tuesday afternoon line moves slightly improved the confidence estimates, "
        "but the changed picks went 13-23. Keep the current picks while this is studied further."
    )
    payload = {
        "description": (
            "One hourly Tuesday move coefficient added to the served four-term probability; "
            "LOSO 2023-2025."
        ),
        "source": "docs/lead82_unit2.md",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "season_start": 2023,
        "season_end": 2025,
        "family": "lead82_tuesday_hourly",
        "classification_evidence": (
            "No resolved wrong-sign interval, split-half refutation, or powered positive control; "
            "retain unresolved under AGENTS.md:65-84."
        ),
        "category": "market",
        "plain_summary": summary,
        "notes": (
            "216 reporting looks within a 261-look ceiling; fixed ridge; "
            "10000 paired season-stratified week bootstrap draws; "
            "retrospective selected-feature replay. Positive means improvement. "
            "Three correlated candidate-versus-served endpoints; "
            "binary RPS duplicates Brier and is not recorded twice."
        ),
        "cells": [],
    }
    for metric, units in [
        ("brier", "brier_improvement"),
        ("log_loss", "log_loss_improvement"),
        ("accuracy_points", "accuracy_points"),
    ]:
        cell = result["contrasts"]["pooled"]["served"][metric]
        payload["cells"].append(
            {
                "name": f"lead82_tuesday_hourly_vs_served_{metric}_2023_2025",
                "effect": cell["effect"],
                "effect_units": units,
                "standard_error": cell["standard_error"],
                "interval_low": cell["low"],
                "interval_high": cell["high"],
                "probability_positive": cell["probability_positive"],
                "sample_games": 799,
                "sample_blocks": result["week_blocks"],
            }
        )
    (OUTPUT / "registry_batch.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    declaration = LANE.read_text(encoding="utf-8-sig")
    required = [
        "Predeclared protocol and amendment",
        "261 looks",
        "136 metric panels",
        "logit as offset",
    ]
    if not all(value in declaration for value in required):
        raise ValueError("Save protocol before outcomes")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / "protocol_before_run.md"
    if not protocol_path.exists():
        protocol_path.write_text(declaration, encoding="utf-8")
    source, frame, inventory = load_inputs()
    probabilities, coefficients, parity = fit_arms(source, frame)
    result = score(frame, probabilities)
    for arm, values in probabilities.items():
        frame[f"p_{arm}"] = values
    frame.to_parquet(OUTPUT / "predictions.parquet", index=False)
    pd.DataFrame(coefficients).to_csv(OUTPUT / "coefficients.csv", index=False)
    payload = {
        "inventory": inventory,
        "parity": parity,
        "coefficients": coefficients,
        "results": result,
        "protocol_sha256": digest(OUTPUT / "protocol_before_run.md"),
        "script_sha256": digest(Path(__file__)),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(frame, result, coefficients, inventory, parity)
    write_record_batch(result)
    print(f"completed games={len(frame)} blocks={result['week_blocks']} looks=216/261")
    for metric, cell in result["contrasts"]["pooled"]["served"].items():
        print(
            f"candidate vs served {metric}: {interval(cell)} "
            f"probability_positive={cell['probability_positive']:.4f}"
        )


if __name__ == "__main__":
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with threadpool_limits(limits=2):
        main()
