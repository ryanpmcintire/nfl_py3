"""Leak-safe, caller-configured transaction-aware preseason priors.

This module performs deterministic as-of arithmetic only.  It does not learn
weights, inspect game outcomes, or decide whether a feature should be used by
an ATS model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
import pandas as pd

from nfl_ats.constants import TEAM_ABBREVIATION_ALIASES

COMPONENTS: tuple[str, ...] = ("qb", "roster", "coaching", "draft", "free_agency")
APPLICATIONS: frozenset[str] = frozenset({"additive", "override"})

_DECISION_COLUMNS = frozenset({"season", "team", "decision_at_utc"})
_ADJUSTMENT_COLUMNS = frozenset(
    {
        "season",
        "team",
        "component",
        "adjustment",
        "uncertainty",
        "units",
        "source_id",
        "source_observed_at_utc",
        "effective_at_utc",
        "application",
        "override_priority",
    }
)


@dataclass(frozen=True)
class PriorComponentRule:
    """Caller-declared scaling and optional time decay for one component."""

    weight: float = 0.0
    half_life_days: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.weight):
            raise ValueError("component weight must be finite")
        if self.half_life_days is not None and (
            not isfinite(self.half_life_days) or self.half_life_days <= 0
        ):
            raise ValueError("half_life_days must be finite and positive when supplied")


@dataclass(frozen=True)
class PreseasonPriorConfig:
    """Explicit prior settings; all component weights default to neutral."""

    units: str = "rating_points"
    baseline_prior: float = 0.0
    qb: PriorComponentRule = field(default_factory=PriorComponentRule)
    roster: PriorComponentRule = field(default_factory=PriorComponentRule)
    coaching: PriorComponentRule = field(default_factory=PriorComponentRule)
    draft: PriorComponentRule = field(default_factory=PriorComponentRule)
    free_agency: PriorComponentRule = field(default_factory=PriorComponentRule)

    def __post_init__(self) -> None:
        if not self.units.strip():
            raise ValueError("units must be non-empty")
        if not isfinite(self.baseline_prior):
            raise ValueError("baseline_prior must be finite")

    def rule_for(self, component: str) -> PriorComponentRule:
        rules = {
            "qb": self.qb,
            "roster": self.roster,
            "coaching": self.coaching,
            "draft": self.draft,
            "free_agency": self.free_agency,
        }
        try:
            return rules[component]
        except KeyError as exc:
            raise ValueError(f"unsupported component: {component!r}") from exc


@dataclass(frozen=True)
class PreseasonPriorResult:
    """One prior per decision plus a source-level calculation audit."""

    priors: pd.DataFrame
    source_audit: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: frozenset[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _canonical_team(value: object) -> str:
    code = str(value).strip().upper()
    if not code:
        raise ValueError("team values must be non-empty")
    return TEAM_ABBREVIATION_ALIASES.get(code, code)


def _utc(series: pd.Series, label: str) -> pd.Series:
    try:
        converted = pd.to_datetime(series, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain valid timestamps") from exc
    if converted.isna().any():
        raise ValueError(f"{label} must not contain missing timestamps")
    return converted


def _numeric(series: pd.Series, label: str) -> pd.Series:
    try:
        converted = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if not np.isfinite(converted).all():
        raise ValueError(f"{label} must contain finite values")
    return converted


def _prepare_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(decisions, _DECISION_COLUMNS, "decisions")
    result = decisions.copy().reset_index(drop=True)
    result["season"] = pd.to_numeric(result["season"], errors="raise").astype(int)
    result["team"] = result["team"].map(_canonical_team)
    result["decision_at_utc"] = _utc(result["decision_at_utc"], "decision_at_utc")
    key = ["season", "team", "decision_at_utc"]
    if result.duplicated(key).any():
        raise ValueError(f"decisions must be unique on {key}")
    result["decision_order"] = np.arange(len(result), dtype=int)
    return result


def _prepare_adjustments(adjustments: pd.DataFrame, config: PreseasonPriorConfig) -> pd.DataFrame:
    _require_columns(adjustments, _ADJUSTMENT_COLUMNS, "adjustments")
    result = adjustments.copy().reset_index(drop=True)
    result["season"] = pd.to_numeric(result["season"], errors="raise").astype(int)
    result["team"] = result["team"].map(_canonical_team)
    result["component"] = result["component"].astype(str).str.strip().str.lower()
    unknown = sorted(set(result["component"]).difference(COMPONENTS))
    if unknown:
        raise ValueError(f"unsupported adjustment components: {unknown}")
    result["application"] = result["application"].astype(str).str.strip().str.lower()
    invalid_applications = sorted(set(result["application"]).difference(APPLICATIONS))
    if invalid_applications:
        raise ValueError(f"application must be additive or override: {invalid_applications}")
    result["adjustment"] = _numeric(result["adjustment"], "adjustment")
    result["uncertainty"] = _numeric(result["uncertainty"], "uncertainty")
    if (result["uncertainty"] < 0).any():
        raise ValueError("uncertainty must be non-negative")
    result["override_priority"] = _numeric(result["override_priority"], "override_priority")
    result["source_observed_at_utc"] = _utc(
        result["source_observed_at_utc"], "source_observed_at_utc"
    )
    result["effective_at_utc"] = _utc(result["effective_at_utc"], "effective_at_utc")
    result["source_id"] = result["source_id"].astype(str).str.strip()
    if (result["source_id"] == "").any():
        raise ValueError("source_id must be non-empty")
    result["units"] = result["units"].astype(str).str.strip()
    wrong_units = sorted(set(result.loc[result["units"] != config.units, "units"]))
    if wrong_units:
        raise ValueError(
            f"all adjustments must use configured units {config.units!r}; got {wrong_units}"
        )
    source_key = ["season", "team", "component", "source_id"]
    if result.duplicated(source_key).any():
        raise ValueError(f"adjustments must be unique on {source_key}")
    return result


def _decay_factor(age_days: pd.Series, half_life_days: float | None) -> pd.Series:
    if half_life_days is None:
        return pd.Series(1.0, index=age_days.index, dtype=float)
    return pd.Series(np.power(0.5, age_days / half_life_days), index=age_days.index, dtype=float)


def _audit_columns() -> list[str]:
    return [
        "season",
        "team",
        "decision_at_utc",
        "component",
        "source_id",
        "source_observed_at_utc",
        "effective_at_utc",
        "application",
        "override_priority",
        "adjustment",
        "uncertainty",
        "units",
        "age_days",
        "decay_factor",
        "configured_weight",
        "half_life_days",
        "selected",
        "selection_reason",
        "weighted_adjustment",
        "weighted_uncertainty",
    ]


def build_transaction_aware_preseason_prior(
    decisions: pd.DataFrame,
    adjustments: pd.DataFrame,
    *,
    config: PreseasonPriorConfig | None = None,
) -> PreseasonPriorResult:
    """Build deterministic as-of priors from caller-supplied adjustments.

    A source is visible only when both its observation and effective timestamp
    are at or before that row's decision cutoff.  Within a component, any
    visible ``override`` source suppresses additive sources; the winning
    override is highest priority, then newest observation, then lexical
    ``source_id``.  Without an override, every visible additive source is used.
    """

    settings = config or PreseasonPriorConfig()
    decision_frame = _prepare_decisions(decisions)
    adjustment_frame = _prepare_adjustments(adjustments, settings)

    audit_parts: list[pd.DataFrame] = []
    prior_rows: list[dict[str, object]] = []
    decision_fields = decision_frame[
        ["decision_order", "season", "team", "decision_at_utc"]
    ].itertuples(index=False, name=None)
    for decision_order, season, team, decision_at_value in decision_fields:
        decision_at = pd.Timestamp(decision_at_value)
        visible = adjustment_frame.loc[
            (adjustment_frame["season"] == season)
            & (adjustment_frame["team"] == team)
            & (adjustment_frame["source_observed_at_utc"] <= decision_at)
            & (adjustment_frame["effective_at_utc"] <= decision_at)
        ].copy()

        prior_row: dict[str, object] = {
            "decision_order": decision_order,
            "baseline_prior": settings.baseline_prior,
            "prior_units": settings.units,
            "visible_source_count": len(visible),
            "override_count": 0,
        }
        total_adjustment = 0.0
        total_variance = 0.0
        override_count = 0
        latest_timestamp: pd.Timestamp | None = None

        for component in COMPONENTS:
            rule = settings.rule_for(component)
            rows = visible.loc[visible["component"] == component].copy()
            if rows.empty:
                contribution = 0.0
                component_uncertainty = 0.0
            else:
                age_days = (decision_at - rows["effective_at_utc"]).dt.total_seconds() / 86400.0
                rows["age_days"] = age_days
                rows["decision_at_utc"] = decision_at
                rows["decay_factor"] = _decay_factor(age_days, rule.half_life_days)
                rows["configured_weight"] = rule.weight
                rows["half_life_days"] = rule.half_life_days
                rows["selected"] = False
                rows["selection_reason"] = "suppressed_by_override"

                overrides = rows.loc[rows["application"] == "override"].sort_values(
                    ["override_priority", "source_observed_at_utc", "source_id"],
                    ascending=[False, False, True],
                    kind="stable",
                )
                if overrides.empty:
                    selected_index = rows.index[rows["application"] == "additive"]
                    rows.loc[selected_index, "selected"] = True
                    rows.loc[selected_index, "selection_reason"] = "additive"
                else:
                    winner = overrides.index[0]
                    rows.loc[winner, "selected"] = True
                    rows.loc[winner, "selection_reason"] = "override_winner"
                    losing_overrides = overrides.index[1:]
                    rows.loc[losing_overrides, "selection_reason"] = "lower_ranked_override"
                    override_count += 1

                rows["weighted_adjustment"] = np.where(
                    rows["selected"],
                    rows["adjustment"] * rows["decay_factor"] * rule.weight,
                    0.0,
                )
                rows["weighted_uncertainty"] = np.where(
                    rows["selected"],
                    rows["uncertainty"] * rows["decay_factor"] * abs(rule.weight),
                    0.0,
                )
                contribution = float(rows["weighted_adjustment"].sum())
                component_uncertainty = float(
                    np.sqrt(np.square(rows["weighted_uncertainty"]).sum())
                )
                component_latest = rows["source_observed_at_utc"].max()
                if latest_timestamp is None or component_latest > latest_timestamp:
                    latest_timestamp = component_latest
                audit_parts.append(rows[_audit_columns()])

            prior_row[f"{component}_contribution"] = contribution
            prior_row[f"{component}_uncertainty"] = component_uncertainty
            total_adjustment += contribution
            total_variance += component_uncertainty**2

        prior_row["prior_adjustment"] = total_adjustment
        prior_row["prior_value"] = settings.baseline_prior + total_adjustment
        prior_row["prior_uncertainty"] = float(np.sqrt(total_variance))
        prior_row["latest_source_observed_at_utc"] = latest_timestamp
        prior_row["override_count"] = override_count
        prior_rows.append(prior_row)

    calculated = pd.DataFrame(prior_rows)
    priors = (
        decision_frame.merge(calculated, on="decision_order", validate="one_to_one")
        .sort_values("decision_order", kind="stable")
        .drop(columns="decision_order")
        .reset_index(drop=True)
    )
    if audit_parts:
        source_audit = pd.concat(audit_parts, ignore_index=True).sort_values(
            ["season", "team", "decision_at_utc", "component", "source_id"], kind="stable"
        )
        source_audit = source_audit.reset_index(drop=True)
    else:
        source_audit = pd.DataFrame(columns=_audit_columns())
    return PreseasonPriorResult(priors=priors, source_audit=source_audit)
