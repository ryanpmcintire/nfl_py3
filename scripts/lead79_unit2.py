from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from quote_provenance_check import REQUIRED, instant, validate_frame
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

FIT_ROOT = Path("artifacts/pick_probability/20260929T192747Z")
SCHEDULE = Path("data/raw/20260908T162105Z/schedules.parquet")
OUTPUT = Path("tests/scratch/codex/lead79_unit2")
REPORT = Path("docs/lead79_unit2.md")
LANE = Path("docs/lanes/lead79.md")
SEASONS = tuple(range(2020, 2026))
ARMS = ("anchor", "four_term", "model", "market")
METRICS = ("accuracy_points", "log_loss", "brier")
BOOTSTRAPS = 10000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def schedule() -> pd.DataFrame:
    fields = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "gametime",
        "home_team",
        "away_team",
    ]
    frame = read_table(SCHEDULE, fields)
    frame = frame.loc[frame.season.isin(SEASONS) & frame.game_type.eq("REG")].copy()
    local = pd.to_datetime(
        frame.gameday.astype(str) + " " + frame.gametime.astype(str), errors="coerce"
    )
    frame["kickoff"] = local.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    if frame.game_id.duplicated().any() or frame.kickoff.isna().any():
        raise ValueError("Schedule identity or kickoff is incomplete")
    teams = pd.concat(
        [
            frame[["game_id", "season", "kickoff", side]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ],
        ignore_index=True,
    ).sort_values(["season", "team", "kickoff"])
    teams["previous"] = teams.groupby(["season", "team"]).kickoff.shift()
    for side in ("home", "away"):
        previous = teams[["game_id", "team", "previous"]].rename(
            columns={"team": f"{side}_team", "previous": f"{side}_previous"}
        )
        frame = frame.merge(previous, on=["game_id", f"{side}_team"], validate="one_to_one")
    first = pd.to_datetime(frame.gameday).groupby([frame.season, frame.week]).transform("min")
    tuesday = first - pd.to_timedelta(first.dt.dayofweek, unit="D") + pd.Timedelta(days=1, hours=9)
    frame["tuesday"] = tuesday.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    frame["gate"] = frame[["home_previous", "away_previous", "tuesday"]].min(axis=1)
    frame.loc[frame.home_previous.isna() | frame.away_previous.isna(), "gate"] = pd.NaT
    return frame


def lookaheads(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = Counter()
    candidates = []
    source_hashes = []
    columns = sorted(REQUIRED | {"nflverse_game_id", "home_team", "away_team", "capture_kind"})
    identity = games[["game_id", "home_team", "away_team", "kickoff", "gate"]]
    for manifest_path in sorted(Path("data/market/raw").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        request = manifest.get("request", {})
        if (
            manifest.get("provider") != "the-odds-api"
            or manifest.get("capture_kind") != "historical_backfill"
            or request.get("sport") != "americanfootball_nfl"
            or "spreads" not in str(request.get("markets", "")).split(",")
        ):
            continue
        observed = instant(manifest.get("observed_at_utc"))
        if (
            not pd.Timestamp("2020-01-01", tz="UTC")
            <= observed
            < pd.Timestamp("2026-03-01", tz="UTC")
        ):
            continue
        counts["snapshots"] += 1
        quote_path = manifest_path.parent / "quotes.parquet"
        quotes = read_table(quote_path, columns)
        counts["quote_rows"] += len(quotes)
        quotes = quotes.loc[quotes.market.eq("spreads")].rename(
            columns={"nflverse_game_id": "game_id"}
        )
        joined = quotes.merge(
            identity, on=["game_id", "home_team", "away_team"], how="inner", validate="many_to_one"
        )
        counts["same_fixture_spread_rows"] += len(joined)
        counts["prior_game_available_rows"] += int(joined.gate.notna().sum())
        joined = joined.loc[joined.gate.gt(observed) & joined.kickoff.gt(observed)].copy()
        counts["before_both_prior_games_rows"] += len(joined)
        if joined.empty:
            continue
        expected = manifest.get("files", {}).get("quotes.parquet", {}).get("sha256")
        if expected is None or digest(quote_path) != expected:
            raise ValueError(f"Quote hash mismatch: {quote_path}")
        checked = validate_frame(joined, manifest, cutoff=observed)
        counts["clock_rejected_rows"] += int((~checked.asof_consistent).sum())
        checked = checked.loc[
            checked.asof_consistent & checked.outcome_side.isin(["HOME", "AWAY"])
        ].copy()
        keys = ["game_id", "provider_event_id", "bookmaker_key"]
        for key, pair in checked.groupby(keys, sort=False):
            if len(pair) != 2 or set(pair.outcome_side) != {"HOME", "AWAY"}:
                counts["incomplete_pairs"] += 1
                continue
            sides = pair.set_index("outcome_side")
            lines = pd.to_numeric(sides.line, errors="coerce")
            if not np.isfinite(lines).all() or not np.isclose(lines.sum(), 0, atol=1e-9):
                counts["invalid_lines"] += 1
                continue
            candidates.append(
                {
                    "game_id": key[0],
                    "event": key[1],
                    "book": key[2],
                    "observed": observed,
                    "lookahead_margin": -float(lines["HOME"]),
                    "quote_path": quote_path.as_posix(),
                    "quote_sha256": expected,
                }
            )
            counts["valid_book_pairs"] += 1
        source_hashes.append(
            {
                "manifest": manifest_path.as_posix(),
                "sha256": digest(manifest_path),
                "quote_sha256": expected,
            }
        )
    pairs = pd.DataFrame(
        candidates,
        columns=[
            "game_id",
            "event",
            "book",
            "observed",
            "lookahead_margin",
            "quote_path",
            "quote_sha256",
        ],
    )
    if pairs.empty:
        anchors = pd.DataFrame(columns=["game_id", "observed", "lookahead_margin", "books"])
    else:
        duplicate = pairs.groupby(["game_id", "observed", "book"]).lookahead_margin.nunique()
        if duplicate.gt(1).any():
            raise ValueError("Conflicting same-book snapshot lines")
        pairs = pairs.drop_duplicates(["game_id", "observed", "book"])
        anchors = (
            pairs.groupby(["game_id", "observed"])
            .agg(lookahead_margin=("lookahead_margin", "median"), books=("book", "nunique"))
            .reset_index()
            .sort_values(["game_id", "observed"])
            .drop_duplicates("game_id", keep="last")
        )
    pairs.to_parquet(OUTPUT / "quote_pairs.parquet", index=False)
    counts["lookahead_games"] = len(anchors)
    return anchors, {"counts": dict(counts), "sources": source_hashes}


def population(games: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metadata = json.loads((FIT_ROOT / "metadata.json").read_text(encoding="utf-8"))
    frame = read_table(FIT_ROOT / "per_game.parquet")
    required = [
        *FIT_FEATURES,
        "model_probability",
        "home_covered",
        "margin_vs_open",
        "tue_open_home_spread",
    ]
    if (
        frame.game_id.duplicated().any()
        or not np.isfinite(frame[required].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Frozen probability rows are incomplete")
    if not frame.season.isin(SEASONS).all() or not frame.game_type.eq("REG").all():
        raise ValueError("Frozen population is outside the declared seasons")
    if not frame.base_probability_policy.eq("discrete_conditional_non_push_v1").all():
        raise ValueError("Base probability is not the served discrete read")
    if (
        frame.margin_vs_open.eq(0).any()
        or not frame.home_covered.eq(frame.margin_vs_open.gt(0)).all()
    ):
        raise ValueError("Opener grading mismatch")
    if not np.allclose(frame.result - frame.tue_open_home_spread, frame.margin_vs_open):
        raise ValueError("Opener sign mismatch")
    if not np.allclose(
        frame.model_logit, np.log(frame.model_probability / (1 - frame.model_probability))
    ):
        raise ValueError("Model logit mismatch")
    coverage = []
    for season in SEASONS:
        ids = set(games.loc[games.season.eq(season), "game_id"])
        available = anchors.game_id.isin(ids)
        coverage.append(
            {
                "season": season,
                "schedule_games": len(ids),
                "frozen_games": int(frame.season.eq(season).sum()),
                "lookahead_games": int(available.sum()),
                "eligible_games": int((available & anchors.game_id.isin(frame.game_id)).sum()),
            }
        )
    frame = frame.merge(anchors, on="game_id", how="inner", validate="one_to_one")
    frame = frame.merge(games[["game_id", "kickoff", "gate"]], on="game_id", validate="one_to_one")
    if len(frame) and not (frame.observed.lt(frame.gate) & frame.observed.lt(frame.kickoff)).all():
        raise ValueError("Lookahead timing leakage")
    frame["anchor_points"] = frame.lookahead_margin - frame.tue_open_home_spread
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True), {
        "coverage": coverage,
        "frozen_sha256": digest(FIT_ROOT / "per_game.parquet"),
        "metadata_sha256": digest(FIT_ROOT / "metadata.json"),
        "schedule_sha256": digest(SCHEDULE),
        "active_model_id": metadata["active_model_id"],
        "elo_available": False,
        "margin_mae_available": "residual_at_open_served" in frame,
    }


def fit(
    train: pd.DataFrame, test: pd.DataFrame, terms: list[str]
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = train[terms].to_numpy(dtype=float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1
    design = np.column_stack([np.ones(len(train)), (x - means) / stds])
    beta = _fit_logit(design, train.home_covered.to_numpy(dtype=float), FIT_RIDGE)

    def predict(data: pd.DataFrame) -> np.ndarray:
        z = beta[0] + (data[terms].to_numpy(dtype=float) - means) / stds @ beta[1:]
        return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

    natural = {"intercept": float(beta[0] - (beta[1:] * means / stds).sum())}
    natural.update(dict(zip(terms, (beta[1:] / stds).tolist(), strict=True)))
    return predict(test), predict(train), natural


def fit_arms(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    work = frame.copy()
    coefficients = []
    folds = []
    for arm in ARMS:
        for kind in ("oos", "is"):
            work[f"{arm}_{kind}"] = np.nan
        work[f"{arm}_band"] = -1
    for held in SEASONS:
        train = work.loc[work.season.lt(held)]
        test = work.loc[work.season.eq(held)]
        reason = "available"
        if test.empty:
            reason = "no_test_quotes"
        elif train.empty or train.home_covered.nunique() < 2:
            reason = "no_earlier_training_classes"
        elif train.anchor_points.nunique() < 2:
            reason = "no_earlier_anchor_variation"
        folds.append(
            {"held_out": held, "train_games": len(train), "test_games": len(test), "status": reason}
        )
        if reason != "available":
            continue
        for arm in ARMS:
            if arm in ("anchor", "four_term"):
                terms = list(FIT_FEATURES) + (["anchor_points"] if arm == "anchor" else [])
                prediction, train_prediction, beta = fit(train, test, terms)
                coefficients.append(
                    {"arm": arm, "held_out": held, "train_games": len(train), **beta}
                )
            else:
                prediction = (
                    test.model_probability.to_numpy() if arm == "model" else np.full(len(test), 0.5)
                )
                train_prediction = (
                    train.model_probability.to_numpy()
                    if arm == "model"
                    else np.full(len(train), 0.5)
                )
            edges = np.quantile(train_prediction, [0.2, 0.4, 0.6, 0.8])
            work.loc[test.index, f"{arm}_oos"] = prediction
            work.loc[test.index, f"{arm}_band"] = np.searchsorted(edges, prediction, side="right")
    if len(work) and work.home_covered.nunique() == 2 and work.anchor_points.nunique() > 1:
        for arm in ARMS:
            if arm in ("anchor", "four_term"):
                terms = list(FIT_FEATURES) + (["anchor_points"] if arm == "anchor" else [])
                prediction, _, beta = fit(work, work, terms)
                coefficients.append(
                    {"arm": arm, "held_out": "IS", "train_games": len(work), **beta}
                )
            else:
                prediction = (
                    work.model_probability.to_numpy() if arm == "model" else np.full(len(work), 0.5)
                )
            work[f"{arm}_is"] = prediction
    return work, coefficients, folds


class Bootstrap:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.blocks = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        self.index = pd.MultiIndex.from_frame(self.blocks)
        self.weights = np.zeros((BOOTSTRAPS, len(self.blocks)))
        rng = np.random.default_rng(79)
        for season in sorted(self.blocks.season.unique()):
            positions = np.flatnonzero(self.blocks.season.eq(season))
            self.weights[:, positions] = rng.multinomial(
                len(positions), np.full(len(positions), 1 / len(positions)), size=BOOTSTRAPS
            )

    def estimate(self, values: np.ndarray, select: np.ndarray | None = None) -> dict:
        values = np.asarray(values, dtype=float)
        keep = np.isfinite(values)
        if select is not None:
            keep &= select
        if not keep.any():
            return {
                "effect": None,
                "interval_low": None,
                "interval_high": None,
                "probability_positive": None,
                "n": 0,
            }
        grouped = self.frame.loc[keep, ["season", "week"]].copy()
        grouped["value"] = values[keep]
        sums = (
            grouped.groupby(["season", "week"])
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
            "interval_low": float(np.quantile(draws, 0.025)),
            "interval_high": float(np.quantile(draws, 0.975)),
            "probability_positive": float((draws > 0).mean() + 0.5 * (draws == 0).mean()),
            "standard_error": float(draws.std(ddof=1)),
            "n": int(keep.sum()),
        }


def losses(frame: pd.DataFrame, p: np.ndarray, arm: str) -> dict[str, np.ndarray]:
    y = frame.home_covered.to_numpy(dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    accuracy = 100 * ((p >= 0.5) == y).astype(float)
    accuracy[~np.isfinite(p)] = np.nan
    if arm == "market":
        accuracy[:] = np.nan
    return {
        "accuracy_points": accuracy,
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (y - p) ** 2,
    }


def score(frame: pd.DataFrame) -> dict:
    bootstrap = Bootstrap(frame)
    result = {
        "metrics": [],
        "gains": [],
        "gaps": [],
        "decisive": [],
        "reliability": [],
        "margin_mae": [],
    }
    scored = frame.anchor_oos.notna().to_numpy()
    values = {
        (arm, kind): losses(frame, frame[f"{arm}_{kind}"].to_numpy(), arm)
        for arm in ARMS
        for kind in ("oos", "is")
    }
    for kind in ("oos", "is"):
        for arm in ARMS:
            for metric in METRICS:
                result["metrics"].append(
                    {
                        "arm": arm,
                        "sample": kind,
                        "season": "pooled",
                        "metric": metric,
                        **bootstrap.estimate(values[arm, kind][metric], scored),
                    }
                )
        for comparator in ARMS[1:]:
            for metric in METRICS:
                gain = values["anchor", kind][metric] - values[comparator, kind][metric]
                if metric != "accuracy_points":
                    gain = -gain
                result["gains"].append(
                    {
                        "comparator": comparator,
                        "sample": kind,
                        "metric": metric,
                        **bootstrap.estimate(gain, scored),
                    }
                )
    for season in SEASONS:
        for arm in ARMS:
            for metric in METRICS:
                result["metrics"].append(
                    {
                        "arm": arm,
                        "sample": "oos",
                        "season": season,
                        "metric": metric,
                        **bootstrap.estimate(
                            values[arm, "oos"][metric], frame.season.eq(season).to_numpy()
                        ),
                    }
                )
    for arm in ARMS:
        for metric in METRICS:
            result["gaps"].append(
                {
                    "arm": arm,
                    "metric": metric,
                    **bootstrap.estimate(
                        values[arm, "is"][metric] - values[arm, "oos"][metric], scored
                    ),
                }
            )
        for band in range(5):
            select = scored & frame[f"{arm}_band"].eq(band).to_numpy()
            result["reliability"].append(
                {
                    "arm": arm,
                    "band": band + 1,
                    "n": int(select.sum()),
                    "predicted": float(frame.loc[select, f"{arm}_oos"].mean())
                    if select.any()
                    else None,
                    "actual": bootstrap.estimate(frame.home_covered.to_numpy(dtype=float), select),
                }
            )
    for season in ("pooled", *SEASONS):
        select = scored.copy()
        if season != "pooled":
            select &= frame.season.eq(season).to_numpy()
        changed = select & (
            (frame.anchor_oos.to_numpy() >= 0.5) != (frame.four_term_oos.to_numpy() >= 0.5)
        )
        correct = values["anchor", "oos"]["accuracy_points"]
        result["decisive"].append(
            {
                "season": season,
                "n": int(changed.sum()),
                "wins": int(np.sum(correct[changed] == 100)),
                "losses": int(np.sum(correct[changed] == 0)),
                "accuracy": bootstrap.estimate(correct, changed),
            }
        )
    if "residual_at_open_served" in frame:
        for arm, predicted in (
            ("model", frame.tue_open_home_spread + frame.residual_at_open_served),
            ("market", frame.tue_open_home_spread),
        ):
            result["margin_mae"].append(
                {
                    "arm": arm,
                    **bootstrap.estimate(np.abs(frame.result - predicted).to_numpy(), scored),
                }
            )
    result["oos_games"] = int(scored.sum())
    result["oos_week_blocks"] = int(
        frame.loc[scored, ["season", "week"]].drop_duplicates().shape[0]
    )
    result["outcome_looks"] = sum(
        row.get("n", 0) > 0
        for key in ("metrics", "gains", "gaps", "margin_mae")
        for row in result[key]
    )
    result["outcome_looks"] += sum(row["n"] > 0 for row in result["decisive"])
    result["outcome_looks"] += 2 * sum(row["n"] > 0 for row in result["reliability"])
    return result


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def estimate_text(row: dict) -> str:
    if row["effect"] is None:
        return "unavailable"
    return f"{row['effect']:.5f} [{row['interval_low']:.5f}, {row['interval_high']:.5f}]"


def report(
    frame: pd.DataFrame,
    provenance: dict,
    pop: dict,
    coefficients: list[dict],
    folds: list[dict],
    results: dict,
) -> None:
    decisive = results["decisive"][0]
    lines = [
        "# LEAD-79 unit 2: earlier same-fixture price",
        "",
        "Protocol: `docs/lanes/lead79.md`, saved before outcomes. Historical opener is the frozen",
        "pool-line proxy on 2020-2025 REG games; base is the served four-term feature set.",
        "**Conditional assumption:** provider `observed_at_utc` counts as historical availability",
        (
            "under `docs/lanes/quote-provenance.md` Decision. This does not prove "
            "archive immutability."
        ),
        "",
        (
            "**Measured decisive record first:** "
            f"{decisive['wins']}-{decisive['losses']} on {decisive['n']} changed picks;"
        ),
        f"accuracy {estimate_text(decisive['accuracy'])}. Eligible games: {len(frame)};",
        (
            f"chronological OOS games: {results['oos_games']}; week blocks: "
            f"{results['oos_week_blocks']}."
        ),
        "",
        "## Source coverage",
        "",
        table(
            ["Season", "Schedule", "Frozen non-push", "Lookahead", "Eligible"],
            [
                [
                    r[k]
                    for k in (
                        "season",
                        "schedule_games",
                        "frozen_games",
                        "lookahead_games",
                        "eligible_games",
                    )
                ]
                for r in pop["coverage"]
            ],
        ),
        "",
        "**Measured:** " + "; ".join(f"{k}={v:,}" for k, v in provenance["counts"].items()) + ".",
        "",
        "Each anchor is the latest complete opposite spread pair snapshot before BOTH teams'",
        "previous same-season kickoffs and Tuesday 09:00 Eastern, aggregated equally across books.",
        "Week-one games cannot qualify. Home margin equals minus the HOME handicap, so the",
        (
            "anchor term is lookahead margin minus the canonical opener margin. No "
            "largest-move screen."
        ),
        "",
        "## Estimation and limits",
        "",
        "One joint ridge-logistic calibration fits the four served inputs plus the anchor term.",
        "The comparator refits the same four served inputs on the same rows. Six forward-only",
        "season-held-out folds use earlier seasons only; fixed existing ridge is 0.001. There is",
        "no tuning or selection sample; calibration is the fitted logistic layer, not a separate",
        "held-out calibration stage. These are research fits and do not authorize serving.",
        "The base movement horizon is through the existing pregame cutoff: opener-graded is not",
        "a claim that all base inputs existed Tuesday. Raw model probabilities use the cached",
        "discrete conditional non-push read; upstream model-selection chronology is inherited.",
        "Neutral market probability is 0.5 and has no directional accuracy. Elo is unavailable",
        "in the frozen paired artifact. Candidate/four-term margin MAE is undefined for a",
        "binary probability calibration; raw-model and opener-margin MAE are reported when scored.",
        "IS predictions come from a full-population fit and are evaluated on the same OOS-eligible",
        (
            "rows for the IS/OOS gap. Positive accuracy gap is optimism; negative "
            "loss gap is optimism."
        ),
        (
            "Intervals are paired 10,000-draw week-block bootstraps within season "
            "(seed 79), conditional"
        ),
        "on the fitted predictions; they do not include refit uncertainty. probability_positive",
        "is bootstrap mass above zero plus half the mass at zero, not a Bayesian posterior.",
        "",
        table(
            ["Held out", "Earlier training", "Test quotes", "Status"],
            [[r[k] for k in ("held_out", "train_games", "test_games", "status")] for r in folds],
        ),
        "",
        "## Pooled scores and IS/OOS gap",
        "",
        table(
            ["Sample", "Arm", "Metric", "Estimate [95% interval]"],
            [
                [r["sample"], r["arm"], r["metric"], estimate_text(r)]
                for r in results["metrics"]
                if r["season"] == "pooled"
            ],
        ),
        "",
        table(
            ["Arm", "Metric", "IS minus OOS [95% interval]"],
            [[r["arm"], r["metric"], estimate_text(r)] for r in results["gaps"]],
        ),
        "",
        "## Paired gains",
        "",
        "Positive gains favor the anchor arm; all scored comparisons share the same rows.",
        "",
        table(
            ["Sample", "Comparator", "Metric", "Gain [95% interval]", "probability_positive"],
            [
                [
                    r["sample"],
                    r["comparator"],
                    r["metric"],
                    estimate_text(r),
                    "unavailable" if r["effect"] is None else f"{r['probability_positive']:.5f}",
                ]
                for r in results["gains"]
            ],
        ),
        "",
        "## Fold coefficients",
        "",
    ]
    if coefficients:
        terms = ["intercept", *FIT_FEATURES, "anchor_points"]
        lines.append(
            table(
                ["Arm", "Held out", "Training", *terms],
                [
                    [
                        r["arm"],
                        r["held_out"],
                        r["train_games"],
                        *[f"{r[t]:.6f}" if t in r else "—" for t in terms],
                    ]
                    for r in coefficients
                ],
            )
        )
    else:
        lines.append(
            "**Measured:** no admissible fit; coefficients and coefficient "
            "stability are unavailable."
        )
    lines += [
        "",
        "## Season stability",
        "",
        table(
            ["Season", "Arm", "Metric", "Estimate [95% interval]"],
            [
                [r["season"], r["arm"], r["metric"], estimate_text(r)]
                for r in results["metrics"]
                if r["season"] != "pooled" and r["n"]
            ],
        ),
        "",
        "## Reliability",
        "",
        (
            "Five quantile bands use only each fold's training probabilities; ties "
            "may leave empty bands."
        ),
        "",
        table(
            ["Arm", "Band", "Games", "Mean prediction", "Home cover [95% interval]"],
            [
                [
                    r["arm"],
                    r["band"],
                    r["n"],
                    "unavailable" if r["predicted"] is None else f"{r['predicted']:.5f}",
                    estimate_text(r["actual"]),
                ]
                for r in results["reliability"]
            ],
        ),
        "",
        "## Margin MAE",
        "",
        table(
            ["Cached margin", "MAE [95% interval]"],
            [[r["arm"], estimate_text(r)] for r in results["margin_mae"]],
        ),
        "",
        (
            f"**Measured looks:** {results['outcome_looks']} numeric reporting "
            "looks, including fitted coefficients;"
        ),
        "291-look ceiling retained (B=2, six scheduled folds, no extra endpoints). Missing cells",
        (
            "are reported unavailable and consume no outcome look. Correlated "
            "looks are not independent evidence."
        ),
        "",
        "## Decision and handoff",
        "",
    ]
    if not results["oos_games"]:
        lines += [
            (
                "**Inferred:** the archive/training support cannot estimate this term "
                "under the declared"
            ),
            (
                "chronological protocol. Accuracy, Brier, log loss, their intervals, "
                "probability_positive,"
            ),
            (
                "and IS/OOS gaps are unavailable, not zero. This is a source/coverage "
                "gap, not a negative"
            ),
            (
                "research result or a closed signal. No numerical registry record can "
                "honestly be written."
            ),
        ]
    else:
        lines += [
            (
                "**Inferred:** unresolved_below_power pending serial registry review; "
                "no positive control"
            ),
            (
                "or admissible closing ground is claimed. Under AGENTS.md, a "
                "zero-crossing interval does"
            ),
            (
                "not close a signal, and a favorable probability_positive alone does "
                "not authorize serving."
            ),
        ]
    lines += [
        "",
        (
            "**Measured command:** `UV_NO_CACHE=1 .tools/uv.exe run --no-sync "
            "python scripts/lead79_unit2.py`."
        ),
        (
            "Detailed provenance, coefficients, bootstrap summaries, and "
            "prediction rows are saved under"
        ),
        (
            "`tests/scratch/codex/lead79_unit2/`; no row dumps are stored in docs. "
            "Registry commands belong"
        ),
        "to the orchestrator. No served card, registry, publication, or Git history was changed.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    declaration = LANE.read_text(encoding="utf-8-sig")
    if "Unit-2 amendment" not in declaration or "291-look" not in declaration:
        raise ValueError("Missing pre-outcome declaration")
    (OUTPUT / "declaration.md").write_text(declaration, encoding="utf-8")
    with threadpool_limits(limits=2):
        games = schedule()
        anchors, provenance = lookaheads(games)
        frame, pop = population(games, anchors)
        frame, coefficients, folds = fit_arms(frame)
        results = score(frame)
    results["outcome_looks"] += sum(len(r) - 3 for r in coefficients)
    if results["outcome_looks"] > 291:
        raise ValueError("Predeclared reporting look budget exceeded")
    frame.to_parquet(OUTPUT / "predictions.parquet", index=False)
    summary = {
        "provenance": provenance,
        "population": pop,
        "coefficients": coefficients,
        "folds": folds,
        "results": results,
        "declaration_sha256": hashlib.sha256(declaration.encode()).hexdigest(),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    report(frame, provenance, pop, coefficients, folds, results)
    print(
        json.dumps(
            {
                "report": REPORT.as_posix(),
                "eligible_games": len(frame),
                "oos_games": results["oos_games"],
                "outcome_looks": results["outcome_looks"],
                "counts": provenance["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
