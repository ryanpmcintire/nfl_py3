from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SOURCE = Path("tests/scratch/codex/lead71_unit2")
OUTPUT = Path("tests/scratch/codex/lead71_unit3")
REPORT = Path("docs/lead71_unit3.md")
LANE = Path("docs/lanes/lead71.md")
FEATURES = Path("data/processed/game_features_weak_stack.parquet")
OPENER = Path("artifacts/opener_evaluation/20260929T192743Z/per_game.parquet")
MARGINS = Path("artifacts/margins/20260929T192312Z/predictions.parquet")
ARMS = ("implied_lattice", "served_lattice", "market_residual")
SCHEMES = ("IS", "OOS")
METRICS = ("push_log_loss", "push_brier")
REPLICATES = 10_000
SEED = 7103
EPSILON = 1e-12


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs():
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    pinned = {Path(key).as_posix(): value for key, value in summary["hashes"].items()}
    hashes = {}
    for path in (FEATURES, OPENER, MARGINS):
        hashes[path.as_posix()] = digest(path)
        require(hashes[path.as_posix()] == pinned[path.as_posix()], f"Source changed: {path}")
    for name in ("scores.parquet", "quotes.parquet", "folds.parquet", "summary.json"):
        path = SOURCE / name
        hashes[path.as_posix()] = digest(path)
    scores = pd.read_parquet(SOURCE / "scores.parquet")
    scores = scores.loc[scores.look.eq(1) & scores.arm.isin(ARMS)].copy()
    require(not scores.duplicated(["game_id", "scheme", "arm"]).any(), "Duplicate scores")
    reference = scores.loc[scores.scheme.eq("OOS") & scores.arm.eq(ARMS[0])]
    ids = set(reference.game_id)
    require(len(ids) == 1480, "Unit 2 population changed")
    require(set(scores.scheme) == set(SCHEMES), "Missing IS/OOS scheme")
    for scheme in SCHEMES:
        for arm in ARMS:
            selected = scores.loc[scores.scheme.eq(scheme) & scores.arm.eq(arm)]
            require(set(selected.game_id) == ids, "Unpaired game population")
    features = pd.read_parquet(
        FEATURES, columns=["game_id", "season", "week", "gameday", "result", "game_type"]
    )
    games = features.loc[features.game_id.isin(ids)].copy()
    opener = pd.read_parquet(
        OPENER, columns=["game_id", "season", "week", "tue_open_home_spread", "result"]
    )
    games = games.merge(opener, on=["game_id", "season", "week", "result"], validate="one_to_one")
    require(len(games) == len(ids), "Pool line, outcome or schedule join failed")
    games = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    require(set(games.season) == set(range(2020, 2026)), "Unexpected seasons")
    require(games.game_type.eq("REG").all(), "Unexpected game population")
    require(games.notna().all().all(), "Missing game metadata")
    require(np.equal(games.result, np.rint(games.result)).all(), "Noninteger result")
    require(
        np.isfinite(games.tue_open_home_spread).all(),
        "Nonfinite pool line",
    )
    quotes = pd.read_parquet(SOURCE / "quotes.parquet")
    quotes = quotes.loc[quotes.game_id.isin(ids)]
    timing = games[["game_id", "gameday"]].merge(quotes, on="game_id", validate="one_to_one")
    require(len(timing) == len(ids), "Missing frozen quote")
    for name in ("snapshot", "cutoff", "kickoff"):
        timing[name] = pd.to_datetime(timing[name], utc=True)
    require(
        (timing.snapshot.le(timing.cutoff) & timing.cutoff.lt(timing.kickoff)).all(),
        "Quote crossed the prediction timestamp",
    )
    archived = pd.read_parquet(
        MARGINS, columns=["game_id", "method", "model_name", "train_max_gameday"]
    )
    archived = archived.loc[archived.method.eq("market_residual") & archived.model_name.eq("ridge")]
    timing = timing.merge(archived, on="game_id", validate="one_to_one")
    require(len(timing) == len(ids), "Missing model chronology")
    training_end = pd.to_datetime(timing.train_max_gameday, utc=True) + pd.Timedelta(days=1)
    require(training_end.lt(timing.cutoff).all(), "Model training crossed opener cutoff")
    folds = pd.read_parquet(SOURCE / "folds.parquet")
    require(set(folds.season) == set(games.season) and len(folds) == 6, "Missing LOSO fold")
    for fold in folds.itertuples(index=False):
        held = games.loc[games.season.eq(fold.season)]
        require(len(held) == fold.games, "Fold population mismatch")
        require(
            pd.Timestamp(fold.training_last_day) + pd.Timedelta(days=1)
            < pd.to_datetime(held.gameday).min(),
            "Lattice history crossed held-out season",
        )
    scores = scores.merge(
        games[["game_id", "season", "week", "tue_open_home_spread", "result"]],
        on=["game_id", "season"],
        validate="many_to_one",
    )
    actual = scores.result.eq(scores.tue_open_home_spread)
    require(actual.eq(scores.actual_push).all(), "Push target does not match signed pool line")
    require(
        scores.actual_home_cover.eq(scores.result.gt(scores.tue_open_home_spread)).all(),
        "Cover sign mismatch",
    )
    p = scores.push_probability.to_numpy(dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid push probability")
    half = scores.tue_open_home_spread.ne(np.rint(scores.tue_open_home_spread))
    require(
        scores.loc[half, "push_probability"].eq(0).all(), "Noninteger-line push mass is nonzero"
    )
    bounded = np.clip(p, EPSILON, 1 - EPSILON)
    y = actual.to_numpy(dtype=float)
    scores["push_log_loss"] = -np.where(y == 1, np.log(bounded), np.log1p(-bounded))
    scores["push_brier"] = np.square(p - y)
    scores["raw_impossible"] = ((p == 0) & (y == 1)) | ((p == 1) & (y == 0))
    scores["clipped"] = (p < EPSILON) | (p > 1 - EPSILON)
    return games, scores, folds, hashes


def bootstrap_design(games):
    labels = pd.MultiIndex.from_frame(games[["season", "week"]])
    codes, blocks = pd.factorize(labels, sort=True)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((REPLICATES, len(blocks)), dtype=float)
    seasons = blocks.get_level_values(0).to_numpy()
    for season in sorted(set(seasons)):
        positions = np.flatnonzero(seasons == season)
        weights[:, positions] = rng.multinomial(
            len(positions), np.full(len(positions), 1 / len(positions)), size=REPLICATES
        )
    return codes, weights


def boot_means(values, mask, codes, weights):
    totals = np.zeros((weights.shape[1], values.shape[1]), dtype=float)
    np.add.at(totals, codes[mask], values[mask])
    counts = np.bincount(codes[mask], minlength=weights.shape[1])
    denominator = weights @ counts
    valid = denominator > 0
    return (weights[valid] @ totals) / denominator[valid, None], int((~valid).sum())


def interval(values):
    if len(values) == 0:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def wilson(wins, games):
    if games == 0:
        return None, None
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * np.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return float(center - radius), float(center + radius)


def evaluate(games, scores):
    codes, weights = bootstrap_design(games)
    panels = {"all": np.ones(len(games), dtype=bool)}
    panels["integer"] = games.tue_open_home_spread.eq(
        np.rint(games.tue_open_home_spread)
    ).to_numpy()
    panels.update({str(s): games.season.eq(s).to_numpy() for s in sorted(games.season.unique())})
    summary, contrasts, gaps, reliability, records = [], [], [], [], []
    ordered = {}
    for scheme in SCHEMES:
        for arm in ARMS:
            ordered[scheme, arm] = (
                scores.loc[scores.scheme.eq(scheme) & scores.arm.eq(arm)]
                .set_index("game_id")
                .loc[games.game_id]
            )
    for panel, mask in panels.items():
        schemes = SCHEMES if panel in ("all", "integer") else ("OOS",)
        for scheme in schemes:
            for metric in METRICS:
                values = np.column_stack([ordered[scheme, arm][metric] for arm in ARMS])
                estimates = values[mask].mean(axis=0)
                draws, empty = boot_means(values, mask, codes, weights)
                for index, arm in enumerate(ARMS):
                    raw = ordered[scheme, arm]
                    low, high = interval(draws[:, index])
                    summary.append(
                        {
                            "panel": panel,
                            "scheme": scheme,
                            "arm": arm,
                            "metric": metric,
                            "games": int(mask.sum()),
                            "pushes": int(raw.actual_push.to_numpy()[mask].sum()),
                            "score": float(estimates[index]),
                            "low": low,
                            "high": high,
                            "raw_impossible": int(raw.raw_impossible.to_numpy()[mask].sum()),
                            "clipped": int(raw.clipped.to_numpy()[mask].sum()),
                            "empty_draws": empty,
                        }
                    )
                for index, comparator in enumerate(ARMS[1:], start=1):
                    differences = draws[:, index] - draws[:, 0]
                    low, high = interval(differences)
                    contrasts.append(
                        {
                            "panel": panel,
                            "scheme": scheme,
                            "comparator": comparator,
                            "metric": metric,
                            "primary": panel == "all"
                            and scheme == "OOS"
                            and comparator == "served_lattice",
                            "games": int(mask.sum()),
                            "blocks": len(np.unique(codes[mask])),
                            "effect": float(estimates[index] - estimates[0]),
                            "low": low,
                            "high": high,
                            "standard_error": float(differences.std(ddof=1)),
                            "probability_positive": float(
                                np.mean(differences > 0) + 0.5 * np.mean(differences == 0)
                            ),
                            "empty_draws": empty,
                        }
                    )
    for panel in ("all", "integer"):
        for arm in ARMS:
            for metric in METRICS:
                pair = {
                    row["scheme"]: row["score"]
                    for row in summary
                    if row["panel"] == panel and row["arm"] == arm and row["metric"] == metric
                }
                gaps.append(
                    {"panel": panel, "arm": arm, "metric": metric, "gap": pair["OOS"] - pair["IS"]}
                )
    for key in (3, 7, 10, 14):
        mask = games.tue_open_home_spread.abs().eq(key).to_numpy()
        n = int(mask.sum())
        for arm in ARMS:
            raw = ordered["OOS", arm]
            p, y = raw.push_probability.to_numpy(), raw.actual_push.to_numpy(dtype=float)
            events = int(y[mask].sum())
            observed_low, observed_high = wilson(events, n)
            draws, empty = boot_means((p - y)[:, None], mask, codes, weights)
            low, high = interval(draws[:, 0])
            reliability.append(
                {
                    "key": key,
                    "arm": arm,
                    "games": n,
                    "pushes": events,
                    "predicted": float(p[mask].mean()) if n else None,
                    "observed": events / n if n else None,
                    "observed_low": observed_low,
                    "observed_high": observed_high,
                    "calibration_error": float((p - y)[mask].mean()) if n else None,
                    "low": low,
                    "high": high,
                    "empty_draws": empty,
                }
            )
    for arm in ARMS:
        raw = ordered["OOS", arm]
        decisive = raw.loc[~raw.actual_push]
        wins = int(decisive.accuracy.sum())
        low, high = wilson(wins, len(decisive))
        records.append(
            {
                "arm": arm,
                "games": len(decisive),
                "wins": wins,
                "losses": len(decisive) - wins,
                "accuracy": wins / len(decisive),
                "low": low,
                "high": high,
            }
        )
    require(
        sum(map(len, (summary, contrasts, gaps, reliability, records))) == 127,
        "Look budget changed",
    )
    return {
        "scores": summary,
        "contrasts": contrasts,
        "gaps": gaps,
        "reliability": reliability,
        "records": records,
        "games": len(games),
        "weeks": weights.shape[1],
        "look_count": 127,
        "primary_looks": 2,
        "replicates": REPLICATES,
        "seed": SEED,
        "epsilon": EPSILON,
    }


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def band(row, field="effect"):
    return f"{fmt(row[field])} [{fmt(row['low'])}, {fmt(row['high'])}]"


def table(lines, headers, rows):
    lines.extend(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    )
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    lines.append("")


def record_command(row):
    units = "log_loss_improvement" if row["metric"] == "push_log_loss" else "brier_improvement"
    name = "logloss" if row["metric"] == "push_log_loss" else "brier"
    return (
        f".tools/uv.exe run --no-sync nfl-ats weak-signals record --name LEAD-71-unit3-{name} "
        f"--description 'Frozen-opener push {name}; 2 primary and 125 descriptive looks' "
        "--source docs/lead71_unit3.md --league nfl --category market "
        f"--effect={row['effect']:.12g} --effect-units {units} "
        f"--standard-error={row['standard_error']:.12g} "
        f"--interval-low={row['low']:.12g} --interval-high={row['high']:.12g} "
        f"--probability-positive={row['probability_positive']:.12g} "
        f"--sample-games {row['games']} --sample-blocks {row['blocks']} "
        "--season-start 2020 --season-end 2025 --family LEAD-71-unit3 "
        "--classification unresolved_below_power "
        "--classification-evidence 'No closure or serving decision; paired week "
        "bootstrap with inherited LOSO folds' "
        "--plain-summary 'We checked how often a game lands exactly on the pool spread. "
        "The market prices and the current margin table give different chances. "
        "This check leaves the idea open and does not change any pool pick.' "
        "--notes '2 primary endpoints; 127 total estimands; unadjusted intervals; no new fits'"
    )


def write_report(result, folds, protocol, hashes):
    lines = [
        "# LEAD-71 unit 3: push chances at the frozen pool line",
        "",
        "**Measured:** saved Unit 2 probabilities were rescored without fitting "
        "or changing served code.",
        "",
        "## Decisive opener record",
        "",
        "**Measured:** pushes are excluded only from this inherited side-selection record. "
        "All push scores below retain them. Accuracy intervals are Wilson 95%; "
        "rates are fractions.",
        "",
    ]
    table(
        lines,
        ["Arm", "W-L", "Accuracy [95% CI]"],
        [[r["arm"], f"{r['wins']}-{r['losses']}", band(r, "accuracy")] for r in result["records"]],
    )
    primary = [row for row in result["contrasts"] if row["primary"]]
    lines.extend(
        [
            "## Primary push comparisons",
            "",
            f"**Measured:** {result['games']} games and {result['weeks']} season/week blocks. "
            "Positive improvement is served minus implied loss; smaller loss is better. "
            "Intervals use 10,000 paired week-block draws within season.",
            "",
        ]
    )
    table(
        lines,
        ["Metric", "Improvement [95% CI]", "probability_positive"],
        [[r["metric"], band(r), fmt(r["probability_positive"])] for r in primary],
    )
    lines.extend(
        [
            "**Inferred:** neither metric establishes better push calibration against served. "
            "These push results describe distribution quality, not "
            "a new pick rule or "
            "proof of a profitable edge. The lattice mechanism remains open "
            "pending serial recording. "
            "No promotion or closure is made here. AGENTS.md:54-59 requires the "
            "discrete distribution "
            "at the line; AGENTS.md:65-79 governs closure; AGENTS.md:89-100 requires one fitted "
            "probability and out-of-season parameters before side selection changes.",
            "",
            "## IS/OOS push scores and gaps",
            "",
            "**Measured:** each score has a week-block 95% CI. The integer panel "
            "excludes noninteger "
            "lines with structurally zero push chance. Gap = OOS minus optimistic"
            " IS; positive is worse. "
            "Log loss clips only for numeric evaluation at 1e-12. Raw impossible counts reveal any "
            "infinite exact log losses; zero support is not smoothed in the probabilities.",
            "",
        ]
    )
    table(
        lines,
        ["Panel", "Scheme", "Arm", "Games / pushes", "Metric", "Score [95% CI]", "Raw impossible"],
        [
            [
                r["panel"],
                r["scheme"],
                r["arm"],
                f"{r['games']} / {r['pushes']}",
                r["metric"],
                band(r, "score"),
                r["raw_impossible"],
            ]
            for r in result["scores"]
            if r["panel"] in ("all", "integer")
        ],
    )
    table(
        lines,
        ["Panel", "Arm", "Metric", "OOS minus IS"],
        [[r["panel"], r["arm"], r["metric"], fmt(r["gap"])] for r in result["gaps"]],
    )
    lines.extend(
        [
            "## All predeclared paired comparisons",
            "",
            "**Measured:** comparator minus implied loss. Market residual is the simple market "
            "baseline; served lattice is the model-only baseline. Season panels "
            "use OOS probabilities.",
            "",
        ]
    )
    table(
        lines,
        ["Panel", "Scheme", "Comparator", "Metric", "Improvement [95% CI]", "probability_positive"],
        [
            [
                r["panel"],
                r["scheme"],
                r["comparator"],
                r["metric"],
                band(r),
                fmt(r["probability_positive"]),
            ]
            for r in result["contrasts"]
        ],
    )
    lines.extend(
        [
            "## Key-number reliability",
            "",
            "**Measured:** each cell contains games whose absolute pool spread equals that key. "
            "A push still means the signed margin equals the signed pool threshold; opposite-side "
            "margins are not combined as events. These are line-conditional push masses, not the "
            "unconditional chance of either team winning by that number. Rates are fractions. "
            "Observed intervals are Wilson; predicted minus observed intervals "
            "use paired week blocks.",
            "",
        ]
    )
    table(
        lines,
        [
            "Key",
            "Arm",
            "Games / pushes",
            "Predicted",
            "Observed [Wilson 95% CI]",
            "Predicted minus observed [block 95% CI]",
            "Empty draws",
        ],
        [
            [
                r["key"],
                r["arm"],
                f"{r['games']} / {r['pushes']}",
                fmt(r["predicted"]),
                f"{fmt(r['observed'])} [{fmt(r['observed_low'])}, {fmt(r['observed_high'])}]",
                band(r, "calibration_error"),
                r["empty_draws"],
            ]
            for r in result["reliability"]
        ],
    )
    lines.extend(
        [
            "## Season stability and inherited folds",
            "",
            "**Measured:** no season or key cell was selected after scoring. "
            "Sparse-event bootstrap "
            "intervals condition on observed events: a zero-push cell or season cannot "
            "generate unseen pushes. "
            "Reliability Wilson intervals show event-rate uncertainty in those cells. Week-block "
            "intervals do not measure between-season sampling or refit uncertainty.",
            "",
        ]
    )
    table(
        lines,
        ["Season", "Arm", "Games / pushes", "Metric", "Score [95% CI]"],
        [
            [r["panel"], r["arm"], f"{r['games']} / {r['pushes']}", r["metric"], band(r, "score")]
            for r in result["scores"]
            if r["panel"] not in ("all", "integer")
        ],
    )
    table(
        lines,
        [
            "Held season",
            "Games",
            "Prior training games",
            "Last training day",
            "Optimistic IS games",
        ],
        [
            [r.season, r.games, r.training_games, r.training_last_day, r.in_sample_games]
            for r in folds.itertuples(index=False)
        ],
    )
    lines.extend(
        [
            "**Read:** scripts/lead71_unit2.py:276-324 constructs each OOS lattice only from "
            "completed prior seasons; its IS companion adds the held season. It is chronological "
            "LOSO, not training on future seasons. Archived model forecasts are "
            "inherited, not refitted. "
            "There are no Unit 3 coefficients or fitted cut points; coefficient stability is "
            "inapplicable. Price constraints and lattice parameters remain "
            "exactly those from Unit 2.",
            "",
            "## Reproduction and limits",
            "",
            "Run `.tools/uv.exe run --no-sync python scripts/lead71_unit3.py` with "
            "`UV_CACHE_DIR=tests/scratch/codex/lead71_unit3/uv-cache` and "
            "`PYTHONDONTWRITEBYTECODE=1`. Threads are capped at two.",
            "",
            "**Measured:** 127 predeclared estimands: 60 score cells, 40 paired "
            "contrasts, 12 IS/OOS "
            "gaps, 12 reliability cells and 3 decisive records. Only the two "
            "all-game OOS comparisons "
            "against served are primary; all other results are descriptive. "
            "Intervals are unadjusted "
            "and correlated; no minimum cell or best season is selected. All "
            "primary and score-panel "
            "bootstrap draws retain a nonempty denominator.",
            "",
            "**Measured:** source hashes, frozen protocol, scored prediction rows"
            " and full summary are "
            "saved in tests/scratch/codex/lead71_unit3/. Hashes for the three joined original data "
            "sources match Unit 2; saved score/quote/fold bytes are hashed for reproducibility. "
            "Raw quote archives are not re-parsed. Runtime checks recheck game "
            "IDs, pool-line signs, "
            "push outcomes, noninteger-line support, prediction-time and fold chronology.",
            "",
            "**Inferred:** LOSO reuse prevents choosing push parameters on these "
            "outcomes, but this "
            "is a new endpoint on the already researched Unit 2 population, not "
            "an independent outer "
            "test. Key cells can have few or no pushes; neither an unadjusted cell interval nor "
            "an overall push score establishes reliable alternative-line or flip-line behavior. "
            "The interval rule leaves the mechanism unresolved_below_power "
            "pending the orchestrator's "
            "serial records. Proposed commands are in docs/lanes/lead71.md; none was executed.",
            "",
            "## Predeclared protocol",
            "",
            protocol,
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    result["hashes"] = hashes
    result["record_commands"] = [record_command(row) for row in primary]
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    lane = LANE.read_text(encoding="utf-8-sig")
    if "### Frozen unit 3 protocol" in lane:
        protocol = lane.split("### Frozen unit 3 protocol", 1)[1].split("## Tried", 1)[0].strip()
    else:
        protocol = (
            REPORT.read_text(encoding="utf-8").split("## Predeclared protocol\n\n", 1)[1].strip()
        )
    require("127 estimands" in protocol and "seed 7103" in protocol, "Missing frozen protocol")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "protocol.txt").write_text(protocol + "\n", encoding="utf-8")
    games, scores, folds, hashes = load_inputs()
    result = evaluate(games, scores)
    require(all(row["empty_draws"] == 0 for row in result["scores"]), "Empty score bootstrap")
    scores.to_parquet(OUTPUT / "push_scores.parquet", index=False)
    hashes["scripts/lead71_unit3.py"] = digest(Path("scripts/lead71_unit3.py"))
    hashes["protocol"] = hashlib.sha256(protocol.encode()).hexdigest()
    write_report(result, folds, protocol, hashes)
    print(
        json.dumps(
            {
                "primary": [row for row in result["contrasts"] if row["primary"]],
                "games": result["games"],
                "weeks": result["weeks"],
                "looks": result["look_count"],
                "report": REPORT.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
