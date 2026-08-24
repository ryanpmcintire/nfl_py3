"""RWB-12 drift monitoring: feature, missingness, probability and calibration drift.

Every week the pipeline rebuilds the feature tables, re-fits the walk-forward
models, and publishes a card. Nothing before this module asked whether the
*inputs* the card was built from still look like the data the model was
designed for. A silent schema regression, a snapshot whose columns went
half-null, or a probability distribution drifting away from its training-era
shape would all reach the published card unremarked -- and the 2026-08-19
Tuesday-visibility audit (``docs/prospective_evidence.md``) showed exactly how
a live arm can quietly stop seeing its own inputs while everything keeps
"working".

Four signals, one read-only report:

1. **Feature drift** -- per-column standardized mean shift and PSI (population
   stability index) of this week's pregame rows against a reference window of
   recent completed weeks, organized by the feature-family registry
   (``nfl_ats.constants.FEATURE_FAMILIES``, RWB-02).
2. **Missingness drift** -- per-column null-rate change in percentage points.
   This is the signal that would have caught a broken snapshot join.
3. **Probability drift** -- the published ``home_cover_probability``
   distribution versus the same method's recent history: mean shift plus the
   share of games outside the reference central 90% band.
4. **Calibration drift** -- Brier score and expected calibration error (ECE)
   over the most recent settled weeks versus the prior settled history, for
   already-published probabilities.

What this is NOT, and must never become: evidence. Drift reports are
operational telemetry about whether the machine is still pointed where it was
aimed. They compare distributions and published probabilities; they adjudicate
no candidate against any baseline, so they spend no rotation-registry window
and may never be cited as a result about any signal. Any claim that a feature
family's *effect* changed goes through the weak-signal / rotation registries,
not through here. The report carries that note on every artifact it writes.

Reference windows use strictly pre-target-week completed games -- the same
leak rule every pregame consumer follows -- but nothing here feeds a model or
an evaluation, so the leak contract is observational hygiene rather than a
scoring guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.constants import FEATURE_FAMILIES
from nfl_ats.io import atomic_csv, atomic_json, run_id

# ---------------------------------------------------------------------------
# Alert thresholds. Deliberately conservative: monitoring that cries wolf gets
# ignored, so the warn tier marks "look at this", the alert tier marks "do not
# publish blind". Neither tier is a verdict about any signal's effect.
# ---------------------------------------------------------------------------

#: PSI tiers (standard convention: <0.10 stable, 0.10-0.25 moderate, >0.25 major).
FEATURE_PSI_WARN = 0.10
FEATURE_PSI_ALERT = 0.25

#: Standardized mean shift ((current mean - reference mean) / reference sd).
MEAN_SHIFT_SD_WARN = 0.50
MEAN_SHIFT_SD_ALERT = 1.00

#: Null-rate change in percentage points versus the reference window.
MISSINGNESS_DELTA_PP_WARN = 10.0
MISSINGNESS_DELTA_PP_ALERT = 25.0

#: Probability drift: share of current probabilities outside the reference
#: central 90% band, and the absolute mean shift.
PROBABILITY_BAND_SHARE_WARN = 0.20
PROBABILITY_MEAN_SHIFT_WARN = 0.05

#: Calibration drift: recent-window Brier minus prior-history Brier.
CALIBRATION_BRIER_DELTA_WARN = 0.02
CALIBRATION_BRIER_DELTA_ALERT = 0.04

#: Minimum settled games before a calibration comparison means anything at all.
CALIBRATION_MIN_RECENT_GAMES = 32
CALIBRATION_MIN_PRIOR_GAMES = 200

#: Probability drift needs at least this many games on each side to be scored.
PROBABILITY_MIN_GAMES = 5

#: Minimum current-window games before a PSI number means anything. A 16-game
#: week against decile bins averages 1.6 games per bin, which reads ~0.2 PSI
#: under a TRUE null -- measured this session on gaussian draws -- so below
#: this floor the value is reported but its status stays unscored rather than
#: crying wolf every September.
FEATURE_PSI_MIN_GAMES = 50

#: ECE bins.
_CALIBRATION_ECE_BINS = 10

#: Floor used inside the PSI ratio to avoid division by zero.
_PSI_EPSILON = 1e-6

_STATUS_RANK = {"ok": 0, "insufficient_history": 1, "warn": 2, "alert": 3}

_FEATURE_FAMILY_OF_COLUMN: dict[str, str] = {
    column: family for family, columns in FEATURE_FAMILIES.items() for column in columns
}


def worst_status(statuses: Iterable[str]) -> str:
    """Fold several section statuses into one overall status."""

    ranked = [status for status in statuses if status in _STATUS_RANK]
    if not ranked:
        return "ok"
    return max(ranked, key=lambda status: _STATUS_RANK[status])


def feature_family_of_column(column: str) -> str:
    """The registered feature family a column belongs to, or ``"unregistered"``."""

    return _FEATURE_FAMILY_OF_COLUMN.get(column, "unregistered")


def registered_feature_columns() -> tuple[str, ...]:
    """Every registered feature-family column, in registry order, deduplicated."""

    ordered: tuple[str, ...] = ()
    for family_columns in FEATURE_FAMILIES.values():
        ordered = ordered + tuple(column for column in family_columns if column not in ordered)
    return ordered


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    series = pd.to_numeric(frame[column], errors="coerce")
    return series.to_numpy(dtype=float)


def _missing_rate_pct(frame: pd.DataFrame, column: str) -> float:
    """Fraction of rows that are null OR non-numeric, as a percentage.

    A numeric feature column carrying text garbage is operationally identical
    to a null column, so both count as missing here.
    """

    total = len(frame)
    if total == 0:
        return 100.0
    usable = pd.to_numeric(frame[column], errors="coerce")
    return float(100.0 * int(usable.isna().sum()) / total)


def psi(current: pd.Series, reference: pd.Series, bins: int = 10) -> float:
    """Population stability index of one column, binned on reference quantiles.

    Out-of-range current values fall into the extreme bins by construction
    (the outer edges are +/-inf), so a wholesale level shift registers as a
    large PSI even when it moves entirely outside the reference range. A
    constant reference column is compared as two bins (equal vs not equal,
    within float tolerance) so a constant going non-constant is still caught.
    """

    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size == 0 or cur.size == 0:
        return float("nan")

    if np.all(ref == ref[0]):
        constant = ref[0]
        cur_equal = float(np.mean(np.isclose(cur, constant, rtol=0.0, atol=1e-12)))
        shares_ref = np.array([1.0, _PSI_EPSILON])
        shares_cur = np.array([max(cur_equal, _PSI_EPSILON), max(1.0 - cur_equal, _PSI_EPSILON)])
        return float(np.sum((shares_cur - shares_ref) * np.log(shares_cur / shares_ref)))

    quantiles = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    inner_edges = np.unique(quantiles[1:-1])
    histogram_edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    counts_ref = np.histogram(ref, bins=histogram_edges)[0].astype(float)
    counts_cur = np.histogram(cur, bins=histogram_edges)[0].astype(float)
    shares_ref = np.maximum(counts_ref / counts_ref.sum(), _PSI_EPSILON)
    shares_cur = np.maximum(counts_cur / counts_cur.sum(), _PSI_EPSILON)
    return float(np.sum((shares_cur - shares_ref) * np.log(shares_cur / shares_ref)))


# ---------------------------------------------------------------------------
# 1. Feature and missingness drift
# ---------------------------------------------------------------------------


def feature_drift_table(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per-column feature and missingness drift for one week versus a reference.

    One row per monitored column with the standardized mean shift, PSI, both
    missing rates, and per-signal statuses. A column absent from either frame
    is reported as fully missing rather than dropped: a column that vanished
    from this week's table is precisely the regression worth shouting about.
    """

    if columns is None:
        columns = [
            column
            for column in registered_feature_columns()
            if column in current.columns or column in reference.columns
        ]
    rows: list[dict[str, Any]] = []
    for column in columns:
        in_current = column in current.columns
        in_reference = column in reference.columns
        missing_current = _missing_rate_pct(current, column) if in_current else 100.0
        missing_reference = _missing_rate_pct(reference, column) if in_reference else 100.0
        delta_pp = missing_current - missing_reference

        ref_values = (
            _numeric_values(reference, column) if in_reference else np.empty(0, dtype=float)
        )
        cur_values = _numeric_values(current, column) if in_current else np.empty(0, dtype=float)
        ref_clean = ref_values[~np.isnan(ref_values)]
        cur_clean = cur_values[~np.isnan(cur_values)]

        if ref_clean.size and cur_clean.size:
            ref_mean = float(np.mean(ref_clean))
            ref_std = float(np.std(ref_clean))
            cur_mean = float(np.mean(cur_clean))
            if ref_std > 0:
                mean_shift_sd = (cur_mean - ref_mean) / ref_std
                if abs(mean_shift_sd) >= MEAN_SHIFT_SD_ALERT:
                    shift_status = "alert"
                elif abs(mean_shift_sd) >= MEAN_SHIFT_SD_WARN:
                    shift_status = "warn"
                else:
                    shift_status = "ok"
            else:
                mean_shift_sd = float("nan")
                shift_status = "ok"
            psi_value = psi(pd.Series(cur_values), pd.Series(ref_values))
            psi_status: str
            if cur_clean.size < FEATURE_PSI_MIN_GAMES:
                psi_status = "insufficient_history"
            elif psi_value >= FEATURE_PSI_ALERT:
                psi_status = "alert"
            elif psi_value >= FEATURE_PSI_WARN:
                psi_status = "warn"
            else:
                psi_status = "ok"
        else:
            ref_mean = float("nan")
            ref_std = float("nan")
            cur_mean = float("nan")
            mean_shift_sd = float("nan")
            psi_value = float("nan")
            shift_status = "insufficient_history"
            psi_status = "insufficient_history"

        if delta_pp >= MISSINGNESS_DELTA_PP_ALERT:
            missingness_status = "alert"
        elif delta_pp >= MISSINGNESS_DELTA_PP_WARN:
            missingness_status = "warn"
        else:
            missingness_status = "ok"

        rows.append(
            {
                "column": column,
                "feature_family": feature_family_of_column(column),
                "reference_games": len(reference),
                "current_games": len(current),
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_shift_sd": mean_shift_sd,
                "psi": psi_value,
                "missingness_reference_pct": missing_reference,
                "missingness_current_pct": missing_current,
                "missingness_delta_pp": delta_pp,
                "shift_status": shift_status,
                "psi_status": psi_status,
                "missingness_status": missingness_status,
            }
        )
    return pd.DataFrame(rows)


def summarize_feature_drift(table: pd.DataFrame) -> dict[str, Any]:
    """Collapse the per-column drift table into the report's summary section."""

    if table.empty:
        return {"status": "insufficient_history", "columns_monitored": 0}
    alerts = sorted(
        table.loc[
            table["psi_status"].eq("alert")
            | table["shift_status"].eq("alert")
            | table["missingness_status"].eq("alert"),
            "column",
        ].tolist()
    )
    warned = set(table.loc[table["psi_status"].eq("warn"), "column"])
    warned |= set(table.loc[table["shift_status"].eq("warn"), "column"])
    warned |= set(table.loc[table["missingness_status"].eq("warn"), "column"])
    warnings = sorted(warned.difference(alerts))
    status = "alert" if alerts else ("warn" if warnings else "ok")
    summary: dict[str, Any] = {
        "status": status,
        "columns_monitored": len(table),
        "alerts": alerts,
        "warnings": warnings,
    }
    if table["psi"].notna().any():
        summary["max_psi"] = float(table["psi"].max())
    if table["missingness_delta_pp"].notna().any():
        summary["max_missingness_delta_pp"] = float(table["missingness_delta_pp"].max())
    return summary


# ---------------------------------------------------------------------------
# 2. Probability drift
# ---------------------------------------------------------------------------


def probability_drift_summary(
    current_probabilities: pd.Series,
    reference_probabilities: pd.Series,
) -> dict[str, Any]:
    """Published-probability distribution drift versus recent history.

    Reports the mean shift and the share of current probabilities outside the
    reference central 90% band. The status flags "the card no longer looks
    like its own recent past"; it says nothing about whether any pick is good.
    """

    current = pd.to_numeric(current_probabilities, errors="coerce").dropna()
    reference = pd.to_numeric(reference_probabilities, errors="coerce").dropna()
    if len(current) < PROBABILITY_MIN_GAMES or len(reference) < PROBABILITY_MIN_GAMES:
        return {
            "status": "insufficient_history",
            "n_current": len(current),
            "n_reference": len(reference),
        }
    band_low = float(reference.quantile(0.05))
    band_high = float(reference.quantile(0.95))
    share_outside = float(((current < band_low) | (current > band_high)).mean())
    mean_shift = float(current.mean() - reference.mean())
    status = (
        "warn"
        if share_outside > PROBABILITY_BAND_SHARE_WARN
        or abs(mean_shift) > PROBABILITY_MEAN_SHIFT_WARN
        else "ok"
    )
    return {
        "status": status,
        "n_current": len(current),
        "n_reference": len(reference),
        "mean_current": float(current.mean()),
        "mean_reference": float(reference.mean()),
        "delta_mean": mean_shift,
        "reference_band_low": band_low,
        "reference_band_high": band_high,
        "share_outside_band": share_outside,
    }


# ---------------------------------------------------------------------------
# 3. Calibration drift
# ---------------------------------------------------------------------------


def _brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def _ece(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = _CALIBRATION_ECE_BINS,
) -> float:
    """Expected calibration error over equal-width probability bins."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if not mask.any():
            continue
        gap = abs(float(probabilities[mask].mean()) - float(outcomes[mask].mean()))
        error += (int(mask.sum()) / total) * gap
    return float(error)


def calibration_drift_summary(
    settled: pd.DataFrame,
    *,
    recent_weeks: int = 4,
    probability_column: str = "home_cover_probability",
    outcome_column: str = "home_cover",
) -> dict[str, Any]:
    """Recent settled Brier/ECE versus prior settled history.

    ``settled`` must already contain only rows where both the published
    probability and the ATS outcome are known, and must carry ``gameday``,
    ``season`` and ``week``. The most recent ``recent_weeks`` distinct
    (season, week) pairs form the recent window; every earlier row forms the
    prior window. This is telemetry about already-published probabilities --
    not a backtest, spending no window, adjudicating nothing.
    """

    frame = settled.copy()
    frame["_gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    frame = frame.dropna(subset=["_gameday"]).sort_values("_gameday")
    ordered_keys = list(dict.fromkeys(map(tuple, frame[["season", "week"]].to_numpy())))
    if not ordered_keys:
        return {"status": "insufficient_history"}
    recent_keys = set(ordered_keys[-recent_weeks:])
    key_series = list(map(tuple, frame[["season", "week"]].to_numpy()))
    is_recent = np.fromiter(
        (key in recent_keys for key in key_series), dtype=bool, count=len(key_series)
    )
    recent = frame.loc[is_recent]
    prior = frame.loc[~is_recent]
    if len(recent) < CALIBRATION_MIN_RECENT_GAMES or len(prior) < CALIBRATION_MIN_PRIOR_GAMES:
        return {
            "status": "insufficient_history",
            "n_recent": len(recent),
            "n_prior": len(prior),
        }
    recent_p = pd.to_numeric(recent[probability_column], errors="coerce").to_numpy(dtype=float)
    recent_y = pd.to_numeric(recent[outcome_column], errors="coerce").to_numpy(dtype=float)
    prior_p = pd.to_numeric(prior[probability_column], errors="coerce").to_numpy(dtype=float)
    prior_y = pd.to_numeric(prior[outcome_column], errors="coerce").to_numpy(dtype=float)
    recent_brier = _brier(recent_p, recent_y)
    prior_brier = _brier(prior_p, prior_y)
    delta_brier = recent_brier - prior_brier
    summary: dict[str, Any] = {
        "status": "ok",
        "n_recent": len(recent),
        "n_prior": len(prior),
        "recent_weeks": sorted(recent_keys),
        "brier_recent": recent_brier,
        "brier_prior": prior_brier,
        "delta_brier": delta_brier,
        "ece_recent": _ece(recent_p, recent_y),
        "ece_prior": _ece(prior_p, prior_y),
    }
    if delta_brier >= CALIBRATION_BRIER_DELTA_ALERT:
        summary["status"] = "alert"
    elif delta_brier >= CALIBRATION_BRIER_DELTA_WARN:
        summary["status"] = "warn"
    return summary


# ---------------------------------------------------------------------------
# 4. The weekly report
# ---------------------------------------------------------------------------


def reference_window(
    features: pd.DataFrame,
    *,
    season: int,
    week: int,
    reference_weeks: int = 6,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Strictly pre-target weeks used as the drift reference window.

    The target week's earliest gameday is the cutoff, mirroring
    ``score_outcome_week``'s leak-safe training rule; the reference is the
    ``reference_weeks`` most recent distinct (season, week) pairs strictly
    before it. Raises ``ValueError`` when the target week is absent or no
    completed game precedes it.
    """

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    target = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)]
    if target.empty:
        raise ValueError(f"No games found for {season} week {week} in the feature table")
    cutoff = target["gameday"].min()
    prior = frame.loc[frame["gameday"].lt(cutoff)]
    if prior.empty:
        raise ValueError(f"No completed games precede {season} week {week}")
    keys = list(dict.fromkeys(map(tuple, prior[["season", "week"]].to_numpy())))[-reference_weeks:]
    keep = [tuple(key) in set(keys) for key in prior[["season", "week"]].to_numpy()]
    selected = prior.loc[np.asarray(keep)]
    return selected.reset_index(drop=True), keys


def build_drift_report(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    history_predictions: pd.DataFrame | None,
    *,
    season: int,
    week: int,
    feature_profile: str,
    probability_method: str = "gaussian",
    ats_method: str = "market_residual",
    reference_weeks: int = 6,
    calibration_recent_weeks: int = 4,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Assemble the full drift report for one published week.

    ``predictions`` is the current week's outcome card (as written by
    ``margin-predict``); ``history_predictions`` gathers the same profile's
    earlier cards, deduplicated per game (first write wins, mirroring the
    ledger convention). Returns ``(report, feature_drift_table)`` so callers
    can persist the wide table beside the JSON without recomputing it.
    """

    reference_frame, reference_keys = reference_window(
        features, season=season, week=week, reference_weeks=reference_weeks
    )
    target = features.loc[features["season"].eq(season) & features["week"].eq(week)]

    monitored = [
        column
        for column in registered_feature_columns()
        if column in target.columns or column in reference_frame.columns
    ]
    drift_table = feature_drift_table(target, reference_frame, columns=monitored)
    feature_section = summarize_feature_drift(drift_table)

    current_card = predictions.loc[predictions["method"].eq(ats_method)]
    if history_predictions is None or history_predictions.empty:
        probability_section: dict[str, Any] = {
            "status": "insufficient_history",
            "reason": "no earlier cards found for this feature profile",
        }
        calibration_section: dict[str, Any] = {
            "status": "insufficient_history",
            "reason": "no earlier cards found for this feature profile",
        }
    else:
        history_card = history_predictions.loc[history_predictions["method"].eq(ats_method)]
        probability_section = probability_drift_summary(
            current_card["home_cover_probability"],
            history_card["home_cover_probability"],
        )
        combined = pd.concat([history_card, current_card], ignore_index=True)
        settled = combined.loc[
            combined["home_cover_probability"].notna() & combined["home_cover"].notna()
        ]
        try:
            calibration_section = calibration_drift_summary(
                settled, recent_weeks=calibration_recent_weeks
            )
        except (KeyError, ValueError):
            calibration_section = {
                "status": "insufficient_history",
                "reason": "settled probabilities unavailable in the gathered cards",
            }

    overall = worst_status(
        [
            feature_section.get("status", "ok"),
            probability_section.get("status", "ok"),
            calibration_section.get("status", "ok"),
        ]
    )
    report: dict[str, Any] = {
        "command": "drift-report",
        "season": season,
        "week": week,
        "feature_profile": feature_profile,
        "probability_method": probability_method,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "overall_status": overall,
        "reference": {
            "weeks": [list(key) for key in reference_keys],
            "games": len(reference_frame),
        },
        "sections": {
            "features_and_missingness": feature_section,
            "probability_drift": probability_section,
            "calibration_drift": calibration_section,
        },
        "notes": [
            "Monitoring telemetry only: this report adjudicates no candidate "
            "against any baseline, spends no rotation-registry window, and may "
            "never be cited as evidence about any signal's effect."
        ],
    }
    return report, drift_table


def write_drift_artifacts(
    report: dict[str, Any],
    drift_table: pd.DataFrame,
    output_root: Path,
) -> Path:
    """Persist one drift run under ``<root>/<season>-week-NN-<run_id>/``."""

    output = output_root / f"{report['season']}-week-{report['week']:02d}-{run_id()}"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(report, output / "drift_report.json")
    atomic_csv(drift_table, output / "feature_drift.csv")
    return output
