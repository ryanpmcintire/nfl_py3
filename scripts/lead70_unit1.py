from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import BASE_PROBABILITY_POLICY
from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

SEASONS = tuple(range(2020, 2026))
SAMPLES = 4000
SEED = 20260929
POPULATION = Path("artifacts/pick_probability/20260929T192747Z/per_game.parquet")
INTERACTION = "model_logit_x_disagreement"
GROUP_COLUMNS = ("method", "model_name", "feature_profile", "availability_method")
ARMS = ("market", "model_only", "four_term", "disagreement")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.12g}" if math.isfinite(value) else "NA"
        return str(value).replace("|", "/").replace("\n", " ")

    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
        "",
    ]


def inventory(population: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    active = read_json(Path("artifacts/active_ats_model.json"))
    selected: dict[str, tuple[dict[str, Any], pd.Series]] = {}
    rows: list[list[Any]] = []
    totals: Counter[str] = Counter()
    target_ids = set(population.game_id.astype(str))
    required = {"game_id", "season", "gameday", "predicted_margin", "train_max_gameday"}
    for path in sorted(Path("artifacts").rglob("predictions.parquet")):
        counts: Counter[str] = Counter()
        totals["files"] += 1
        if "cfb" in path.as_posix().lower():
            totals["college_files"] += 1
            rows.append([path.as_posix(), "college football; excluded", 0])
            continue
        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = required.difference(names)
        if missing:
            totals["missing_required_columns"] += 1
            rows.append([path.as_posix(), "missing " + ", ".join(sorted(missing)), 0])
            continue
        metadata_path = path.with_name("metadata.json")
        if not metadata_path.exists():
            totals["missing_metadata"] += 1
            rows.append([path.as_posix(), "no model configuration metadata", 0])
            continue
        metadata = read_json(metadata_path)
        columns = sorted(required | (set(GROUP_COLUMNS) & names))
        frame = pq.read_table(path, columns=columns, use_threads=False).to_pandas()
        frame["game_id"] = frame.game_id.astype(str)
        frame = frame.loc[frame.game_id.isin(target_ids) & frame.season.isin(SEASONS)].copy()
        if frame.empty:
            rows.append([path.as_posix(), "no target NFL games", 0])
            totals["no_target_games"] += 1
            continue
        groups = [column for column in GROUP_COLUMNS if column in frame]
        if "method" not in groups:
            rows.append([path.as_posix(), "no verifiable margin method", 0])
            totals["missing_method"] += 1
            continue
        for _, group in frame.groupby(groups, dropna=False, sort=True):
            configuration = {
                "method": str(group.method.iloc[0]),
                "model_name": str(group.model_name.iloc[0])
                if "model_name" in group
                else metadata.get("regressor"),
                "feature_profile": str(group.feature_profile.iloc[0])
                if "feature_profile" in group
                else metadata.get("feature_profile"),
                "ridge_alpha": metadata.get("ridge_alpha"),
                "availability_method": str(group.availability_method.iloc[0])
                if "availability_method" in group
                else None,
            }
            if configuration["method"] not in {"fair_margin", "market_residual"}:
                counts["not_margin_regression"] += 1
                continue
            if not configuration["feature_profile"] or not configuration["model_name"]:
                counts["missing_model_identity"] += 1
                continue
            if (
                configuration["method"] == active["method"]
                and configuration["feature_profile"] == active["feature_profile"]
                and configuration["model_name"] == active["regressor"]
                and configuration["ridge_alpha"] == active["ridge_alpha"]
                and configuration["availability_method"] is None
            ):
                counts["served_configuration"] += 1
                continue
            margin = pd.to_numeric(group.predicted_margin, errors="coerce")
            cutoff = pd.to_datetime(group.train_max_gameday, utc=True, errors="coerce")
            gameday = pd.to_datetime(group.gameday, utc=True, errors="coerce")
            safe = np.isfinite(margin) & cutoff.notna() & gameday.notna() & cutoff.lt(gameday)
            if not safe.all():
                counts["invalid_margin_or_cutoff"] += 1
                continue
            if not set(SEASONS).issubset(set(group.season.astype(int))):
                counts["missing_seasons"] += 1
                continue
            group = group.assign(predicted_margin=margin)
            if group.groupby("game_id").predicted_margin.nunique().gt(1).any():
                counts["ambiguous_game_predictions"] += 1
                continue
            series = group.drop_duplicates("game_id").set_index("game_id").predicted_margin
            configuration_key = json.dumps(configuration, sort_keys=True)
            record = {
                "path": path.as_posix(),
                "created": metadata.get("created_at_utc", path.parent.name),
                "configuration": configuration,
                "games": len(series),
                "minimum_cutoff_lag_days": float(
                    (gameday - cutoff).dt.total_seconds().min() / 86400
                ),
            }
            counts["qualifying_streams"] += 1
            if configuration_key not in selected or (str(record["created"]), record["path"]) > (
                str(selected[configuration_key][0]["created"]),
                selected[configuration_key][0]["path"],
            ):
                selected[configuration_key] = record, series
        totals.update(counts)
        rows.append(
            [
                path.as_posix(),
                json.dumps(dict(counts), sort_keys=True),
                counts["qualifying_streams"],
            ]
        )
    selected_items = sorted(selected.items())
    matrix = (
        pd.concat(
            [series.rename(f"challenger_{i}") for i, (_, (_, series)) in enumerate(selected_items)],
            axis=1,
            join="inner",
        )
        if selected_items
        else pd.DataFrame()
    )
    keep: list[str] = []
    records: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    if not matrix.empty:
        matrix = matrix.sort_index()
        for i, (_, (record, _)) in enumerate(selected_items):
            column = f"challenger_{i}"
            fingerprint = matrix[column].to_numpy(dtype=np.float64).tobytes()
            if fingerprint in seen:
                totals["duplicate_margin_streams"] += 1
                continue
            seen.add(fingerprint)
            keep.append(column)
            record["column"] = column
            record["sha256"] = digest(Path(record["path"]))
            records.append(record)
        matrix = matrix[keep]
    totals["unique_qualifying_models"] = len(records)
    totals["common_games"] = len(matrix)
    report = [
        "# LEAD-70 source inventory",
        "",
        "**Measured:** metadata and margin predictions only; no outcomes selected these sources.",
        "The gate needs three distinct challenger configurations with all six seasons and",
        "strictly pregame training cutoffs. Market/classifier outputs, college football, the",
        "served configuration, and duplicate margin vectors cannot satisfy the gate.",
        "",
        *table(
            ["Inventory count", "Value"], [[key, value] for key, value in sorted(totals.items())]
        ),
        "## Selected challenger streams",
        "",
        *table(
            ["Column", "File", "Configuration", "Games", "Minimum cutoff lag (days)", "SHA256"],
            [
                [
                    r["column"],
                    r["path"],
                    json.dumps(r["configuration"], sort_keys=True),
                    r["games"],
                    r["minimum_cutoff_lag_days"],
                    r["sha256"],
                ]
                for r in records
            ],
        ),
        "## File inventory",
        "",
        *table(["File", "Disposition", "Qualifying streams before deduplication"], rows),
    ]
    Path("docs/lead70_inventory.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"inventory": dict(totals)}, sort_keys=True), flush=True)
    return matrix, records


def fit(
    train: pd.DataFrame, test: pd.DataFrame, augmented: bool
) -> tuple[np.ndarray, dict[str, float]]:
    features = list(FIT_FEATURES)
    train = train.copy()
    test = test.copy()
    mean = float(train.disagreement.mean())
    std = float(train.disagreement.std(ddof=0))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Training disagreement has no finite positive variance")
    if augmented:
        for frame in (train, test):
            frame[INTERACTION] = frame.model_logit * ((frame.disagreement - mean) / std)
        features.append(INTERACTION)
    means = train[features].mean().to_numpy(dtype=float)
    stds = train[features].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    x_train = np.column_stack([np.ones(len(train)), (train[features].to_numpy() - means) / stds])
    x_test = np.column_stack([np.ones(len(test)), (test[features].to_numpy() - means) / stds])
    beta = _fit_logit(x_train, train.home_covered.to_numpy(dtype=float), FIT_RIDGE)
    coefficients = dict(zip(features, beta[1:] / stds, strict=True))
    coefficients["intercept"] = float(beta[0] - np.sum(beta[1:] * means / stds))
    coefficients["disagreement_mean"] = mean
    coefficients["disagreement_std"] = std
    coefficients["d"] = (
        float(coefficients[INTERACTION] / coefficients["model_logit"])
        if augmented and abs(coefficients["model_logit"]) > 1e-12
        else float("nan")
    )
    return 1.0 / (1.0 + np.exp(-np.clip(x_test @ beta, -35, 35))), coefficients


def losses(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    p = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "log_loss": -(target * np.log(p) + (1.0 - target) * np.log1p(-p)),
        "brier": (p - target) ** 2,
        "accuracy": (p >= 0.5) == target,
    }


def bootstrap(values: np.ndarray, season: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    sums = np.array([values[season == year].sum() for year in SEASONS])
    counts = np.array([(season == year).sum() for year in SEASONS])
    draws = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {
        "effect": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "se": float(draws.std(ddof=1)),
        "probability_positive": float(
            (np.count_nonzero(draws > 0) + 0.5 * np.count_nonzero(draws == 0)) / len(draws)
        ),
    }


def record_commands(effects: list[dict[str, Any]], games: int) -> list[str]:
    commands = ["$lead70Cells = @("]
    for index, effect in enumerate(effects):
        units = {
            "log_loss": "log_loss_improvement",
            "brier": "brier_improvement",
            "accuracy": "accuracy_points",
        }[effect["metric"]]
        commands.append(
            "    @('"
            + effect["reference"]
            + "_"
            + effect["metric"]
            + "', '"
            + units
            + "', "
            + ", ".join(
                f"{effect[key]:.12g}"
                for key in ("effect", "se", "low", "high", "probability_positive")
            )
            + (")" if index == len(effects) - 1 else "),")
        )
    commands.extend(
        [
            ")",
            "$lead70Cells | ForEach-Object {",
            "    $lead70Verdict = @('--classification', 'unresolved_below_power')",
            "    if ($_[0] -like 'four_term_*' -and $_[5] -lt 0) {",
            "        $lead70Verdict = @('--classification', 'refuted_mechanism', "
            "'--closing-ground', 'wrong_sign_resolved')",
            "    }",
            "    .tools/uv.exe run --no-sync nfl-ats weak-signals record @lead70Verdict `",
            "      --name ('lead70_disagreement_vs_' + $_[0]) --family lead70_disagreement_v1 `",
            "      --description 'LOSO disagreement modifier; fixed preserved challenger panel' `",
            "      --source docs/lead70_results.md --league nfl "
            "--season-start 2020 --season-end 2025 `",
            "      --classification-evidence 'Incremental effects have wholly adverse intervals; "
            "other references do not isolate the modifier' `",
            "      --effect-units $_[1] --effect $_[2] --standard-error $_[3] `",
            f"      --interval-low $_[4] --interval-high $_[5] --probability-positive $_[6] "
            f"--sample-games {games} --sample-blocks 6",
            "}",
        ]
    )
    return commands


def evaluate(frame: pd.DataFrame, records: list[dict[str, Any]]) -> None:
    target = frame.home_covered.to_numpy(dtype=float)
    season = frame.season.to_numpy(dtype=int)
    samples = np.random.default_rng(SEED).integers(0, len(SEASONS), (SAMPLES, len(SEASONS)))
    probabilities = {
        "market": frame.neutral_market_probability.to_numpy(dtype=float),
        "model_only": frame.model_probability.to_numpy(dtype=float),
    }
    coefficients: list[list[Any]] = []
    coefficient_keys = [
        "intercept",
        *FIT_FEATURES,
        INTERACTION,
        "d",
        "disagreement_mean",
        "disagreement_std",
    ]
    fold_rows: list[list[Any]] = []
    frame["tercile"] = -1
    is_probabilities: dict[str, np.ndarray] = {}
    for augmented, name in ((False, "four_term"), (True, "disagreement")):
        probabilities[name] = np.full(len(frame), np.nan)
        for year in SEASONS:
            test_mask = season == year
            train = frame.loc[~test_mask]
            test = frame.loc[test_mask]
            if train.empty or test.empty:
                raise ValueError(f"Missing LOSO fold {year}")
            combined, beta = fit(train, pd.concat([train, test]), augmented)
            p_train, p_test = combined[: len(train)], combined[len(train) :]
            probabilities[name][test_mask] = p_test
            coefficients.append(
                [
                    year,
                    name,
                    len(train),
                    len(test),
                    *(beta.get(key, float("nan")) for key in coefficient_keys),
                ]
            )
            train_metrics = losses(train.home_covered.to_numpy(dtype=float), p_train)
            test_metrics = losses(test.home_covered.to_numpy(dtype=float), p_test)
            fold_rows.append(
                [
                    year,
                    name,
                    len(test),
                    *(float(train_metrics[m].mean()) for m in ("log_loss", "brier", "accuracy")),
                    *(float(test_metrics[m].mean()) for m in ("log_loss", "brier", "accuracy")),
                ]
            )
            if augmented:
                thresholds = train.disagreement.quantile([1 / 3, 2 / 3]).to_numpy()
                frame.loc[test_mask, "tercile"] = np.searchsorted(
                    thresholds, test.disagreement, side="right"
                )
                coefficients[-1].extend(thresholds.tolist())
            else:
                coefficients[-1].extend([float("nan"), float("nan")])
        is_probabilities[name], beta = fit(frame, frame, augmented)
        coefficients.append(
            [
                "full IS",
                name,
                len(frame),
                0,
                *(beta.get(key, float("nan")) for key in coefficient_keys),
                float("nan"),
                float("nan"),
            ]
        )
    metrics = {name: losses(target, p) for name, p in probabilities.items()}
    effects: list[dict[str, Any]] = []
    for reference in ("four_term", "model_only", "market"):
        for metric in ("log_loss", "brier", "accuracy"):
            improvement = metrics[reference][metric].astype(float) - metrics["disagreement"][
                metric
            ].astype(float)
            if metric == "accuracy":
                improvement = -100.0 * improvement
            effects.append(
                {
                    "reference": reference,
                    "metric": metric,
                    **bootstrap(improvement, season, samples),
                }
            )
    decisive = (probabilities["disagreement"] >= 0.5) != (probabilities["four_term"] >= 0.5)
    decisive_n = int(decisive.sum())
    decisive_wins = int(metrics["disagreement"]["accuracy"][decisive].sum())
    if decisive_n:
        exact = binomtest(decisive_wins, decisive_n)
        decisive_ci = exact.proportion_ci(confidence_level=0.95, method="wilson")
        decisive_text = (
            f"{decisive_wins}-{decisive_n - decisive_wins} on {decisive_n} changed-side games; "
            f"95% Wilson accuracy interval [{decisive_ci.low:.6f}, {decisive_ci.high:.6f}], "
            f"exact two-sided equal-success null p={exact.pvalue:.6f}. "
            f"The four-term record on those same games is "
            f"{decisive_n - decisive_wins}-{decisive_wins}."
        )
    else:
        decisive_text = (
            "0 changed-side games; the decisive-game interval and exact null are not estimable."
        )
    reproduction = float(
        np.max(np.abs(probabilities["four_term"] - frame.out_of_season_home_probability.to_numpy()))
    )
    metric_rows = []
    for name in ARMS:
        for metric in ("log_loss", "brier", "accuracy"):
            measured = bootstrap(metrics[name][metric].astype(float), season, samples)
            inside = (
                float(losses(target, is_probabilities[name])[metric].mean())
                if name in is_probabilities
                else float("nan")
            )
            gap = measured["effect"] - inside
            metric_rows.append(
                [name, metric, inside, measured["effect"], gap, measured["low"], measured["high"]]
            )
    reliability_rows = []
    for band in range(3):
        mask = frame.tercile.to_numpy() == band
        for name in ARMS:
            p = probabilities[name][mask]
            reliability_rows.append(
                [
                    band + 1,
                    name,
                    int(mask.sum()),
                    float(p.mean()),
                    float(target[mask].mean()),
                    float(np.maximum(p, 1.0 - p).mean()),
                    *(
                        float(metrics[name][m][mask].mean())
                        for m in ("accuracy", "brier", "log_loss")
                    ),
                ]
            )
    report = [
        "# LEAD-70: preserved-model disagreement",
        "",
        f"**Measured:** {len(frame)} opener-graded non-push games across 2020-2025; "
        f"{len(records)} distinct challenger streams.",
        "**Measured decisive-game record:** " + decisive_text,
        "",
        "## Protocol and interpretation",
        "",
        "The preregistration is in `docs/lanes/lead70.md`; source inventory is in "
        "`docs/lead70_inventory.md`.",
        "The two fitted forms are the existing four-term logistic model and that same model with",
        "`model_logit * z(disagreement)` added. This is `a*logit*(1+d*z)` with `d=interaction/a`.",
        "All coefficients, feature standardisers, and disagreement tercile cuts are fitted on "
        "the other",
        "five seasons. The full-population fit supplies descriptive IS metrics only. The fixed "
        "existing",
        f"ridge is {FIT_RIDGE}; no parameter search occurred. Disagreement is the population "
        f"standard deviation",
        "of challenger predicted margins in points; it is never an independent side-selection "
        "rule.",
        "The existing discrete opener probability supplies the model logit; no pooled-residual "
        "mapping is fitted.",
        "**Look accounting:** 2 predeclared fitted forms, 12 LOSO fits plus 2 descriptive full "
        "fits,",
        "3 predeclared disagreement bands, 12 arm-by-band diagnostic cells, 9 paired "
        "arm-by-metric contrasts.",
        "The broader reporting surface is explicit; the nominal two looks is not a claim of "
        "two independent tests.",
        "The market/model-only references are fixed and evaluated on exactly the same games.",
        f"Intervals are 95% percentile intervals from {SAMPLES} season-cluster draws (seed "
        f"{SEED});",
        "`probability_positive` counts positive draws plus half the zero draws. Six clusters "
        "limit precision.",
        "Positive paired effects mean lower loss or greater accuracy; accuracy effects are "
        "percentage points.",
        "These are reused research seasons and previously developed challengers, not an "
        "untouched outer test.",
        "Different fitting configurations share training data and are not independent "
        "statistical replicates.",
        "A training cutoff before game day establishes the declared fold condition, not the "
        "provenance of",
        "every historical feature or an exact Tuesday-noon forecast capture; those remain "
        "source limitations.",
        "No serving or stable-edge claim follows from the positive baseline comparisons.",
        "**Inferred disposition, pending serial recording:** the incremental comparisons satisfy",
        "`wrong_sign_resolved` only when their entire paired improvement interval is negative.",
        "This applies to the tested panel and modifier, not every disagreement mechanism.",
        "AGENTS.md permits that closure ground; other comparisons remain",
        "`unresolved_below_power` and do not isolate the added disagreement term.",
        "",
        f"**Measured:** four-term OOS probability difference from the saved served fit on "
        f"these games: {reproduction:.12g}.",
        "A nonzero difference can reflect common-game restriction; exact agreement is expected "
        "on the full population.",
        f"**Read source:** `{POPULATION.as_posix()}`, SHA256 `{digest(POPULATION)}`.",
        "",
        "## IS, OOS and gap",
        "",
        "Gap is OOS minus IS (lower is favorable for losses, higher for accuracy). Accuracy is "
        "a fraction here.",
        "",
        *table(
            [
                "Arm",
                "Metric",
                "Full-fit IS",
                "LOSO OOS",
                "OOS minus IS",
                "OOS CI low",
                "OOS CI high",
            ],
            metric_rows,
        ),
        "## Paired OOS improvement",
        "",
        *table(
            ["Reference", "Metric", "Effect", "95% low", "95% high", "SE", "probability_positive"],
            [
                [
                    e["reference"],
                    e["metric"],
                    *(e[k] for k in ("effect", "low", "high", "se", "probability_positive")),
                ]
                for e in effects
            ],
        ),
        "## Coefficients and stability",
        "",
        "Coefficients are on natural feature scales, except the interaction uses "
        "training-standardised disagreement.",
        "`d` is unstable when the fitted model-logit coefficient is near zero; the interaction "
        "coefficient is primary.",
        "",
        *table(
            [
                "Held season",
                "Arm",
                "Train n",
                "Test n",
                *coefficient_keys,
                "Tercile q1",
                "Tercile q2",
            ],
            coefficients,
        ),
        "## Season stability and fold IS/OOS",
        "",
        *table(
            [
                "Season",
                "Arm",
                "Held n",
                "Train LL",
                "Train Brier",
                "Train accuracy",
                "OOS LL",
                "OOS Brier",
                "OOS accuracy",
            ],
            fold_rows,
        ),
        "## Reliability by training-defined disagreement tercile",
        "",
        *table(
            [
                "Tercile",
                "Arm",
                "n",
                "Mean home p",
                "Home cover rate",
                "Mean pick p",
                "Accuracy",
                "Brier",
                "Log loss",
            ],
            reliability_rows,
        ),
    ]
    Path("docs/lead70_results.md").write_text("\n".join(report), encoding="utf-8")
    prediction_rows = [
        [
            frame.game_id.iloc[i],
            int(season[i]),
            int(frame.week.iloc[i]),
            int(target[i]),
            float(frame.disagreement.iloc[i]),
            int(frame.tercile.iloc[i]) + 1,
            *(float(probabilities[name][i]) for name in ARMS),
        ]
        for i in range(len(frame))
    ]
    Path("docs/lead70_predictions.md").write_text(
        "\n".join(
            [
                "# LEAD-70 preserved LOSO predictions",
                "",
                "Research output; probabilities are held out by season, except the fixed "
                "market/model-only references.",
                "Preserved for audit within the packet Markdown output scope; do not publish "
                "as a served card.",
                "",
                *table(
                    ["Game", "Season", "Week", "Home covered", "Disagreement", "Tercile", *ARMS],
                    prediction_rows,
                ),
            ]
        ),
        encoding="utf-8",
    )
    lane = Path("docs/lanes/lead70.md")
    lane_text = lane.read_text(encoding="utf-8-sig").split("## Record commands")[0]
    lane.write_text(
        lane_text
        + "## Record commands\nPending orchestrator; not executed by this worker.\n```powershell\n"
        + "\n".join(record_commands(effects, len(frame)))
        + "\n```\n",
        encoding="utf-8",
    )
    primary = next(
        e for e in effects if e["reference"] == "four_term" and e["metric"] == "log_loss"
    )
    print(
        json.dumps(
            {
                "games": len(frame),
                "challengers": len(records),
                "primary_log_loss_improvement": primary,
                "decisive_record": decisive_text,
                "baseline_reproduction_max_error": reproduction,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    metadata = read_json(POPULATION.with_name("metadata.json"))
    active = read_json(Path("artifacts/active_ats_model.json"))
    if metadata["active_model_id"] != active["model_id"]:
        raise ValueError("The frozen population no longer matches the active model")
    if metadata["base_probability_policy"] != BASE_PROBABILITY_POLICY:
        raise ValueError("The frozen population does not use the served discrete probability")
    keys = ["game_id", "season"]
    identifiers = pq.read_table(POPULATION, columns=keys, use_threads=False).to_pandas()
    identifiers = identifiers.loc[identifiers.season.isin(SEASONS)].copy()
    if identifiers.game_id.duplicated().any():
        raise ValueError("The frozen population contains duplicate games")
    matrix, records = inventory(identifiers)
    if len(records) < 3 or matrix.empty:
        Path("docs/lead70_results.md").write_text(
            "# LEAD-70 data gap\n\n**Measured:** fewer than three admissible challenger "
            "streams or no common games.\n"
            "No outcomes loaded, fit performed, metric estimated, or hypothesis look consumed.\n"
            "See `docs/lead70_inventory.md`; this is a source gap, not a research result.\n",
            encoding="utf-8",
        )
        print("DATA_GAP: inventory only; no outcome computation", flush=True)
        return
    columns = [
        *keys,
        "week",
        "home_covered",
        *FIT_FEATURES,
        "model_probability",
        "neutral_market_probability",
        "out_of_season_home_probability",
    ]
    frame = pq.read_table(POPULATION, columns=columns, use_threads=False).to_pandas()
    frame = frame.loc[frame.game_id.isin(matrix.index) & frame.season.isin(SEASONS)].reset_index(
        drop=True
    )
    frame["disagreement"] = matrix.loc[frame.game_id].std(axis=1, ddof=0).to_numpy()
    if set(frame.season.astype(int)) != set(SEASONS):
        raise ValueError("The common population does not contain every declared season")
    if (
        not frame.home_covered.isin([0.0, 1.0]).all()
        or not np.isfinite(frame[list(FIT_FEATURES)].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Invalid target or fit features in the frozen population")
    with threadpool_limits(limits=2):
        evaluate(frame, records)


if __name__ == "__main__":
    main()
