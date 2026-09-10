from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.market_decomposition import FAMILY_PHRASES
from nfl_ats.reporting import artifact_directories, read_json

STABILITY_JUMPY_RATIO = 0.15

CLASSIFICATION_CAPTIONS: dict[str, str] = {
    "unpriced_predictive": (
        "the model leans on this and the market does not seem to price it -- unconfirmed"
    ),
    "overpriced": "the market leans on this more than reality rewards it",
    "priced": "market and reality roughly agree here",
    "noise": "neither says much about this",
}

HONESTY_NOTES: tuple[tuple[str, str], ...] = (
    (
        "A diagnostic, not a discovery",
        "This table generates hypotheses about what the market might be missing. It does "
        "not adjudicate them -- only the project's outer-season, prospectively scored "
        'record does that. Nothing here should be read as "found an edge."',
    ),
    (
        "Family-level only",
        "Coefficients are only meaningful once added up to a whole feature family -- ridge "
        "regression smears weight across correlated features, so an individual feature's "
        "number would not mean what it looks like it means.",
    ),
    (
        "Explains a pattern, scores no new picks",
        "This is fit on seasons the project has already looked at and scored elsewhere. It "
        "explains what the model leans on in hindsight; it does not grade a new stream of "
        "picks.",
    ),
    (
        "Blind to the market's own line, on purpose",
        "The model behind these weights never sees the market's line -- that is what makes "
        '"what does the market not see" a meaningful question instead of a circular one. It '
        "is therefore not numerically identical to the deployed model behind this week's "
        "picks; it shows what a model blind to the market's line would have prioritized.",
    ),
)


@dataclass(frozen=True)
class FamilyExplanation:
    family: str
    label: str
    margin_share: float
    spread_share: float
    weight_in_spread: float
    refit_std_in_spread: float
    classification: str

    @property
    def stability_word(self) -> str:

        ratio = self._stability_ratio()
        return (
            "jumps around between refits"
            if ratio >= STABILITY_JUMPY_RATIO
            else ("steady across refits")
        )

    @property
    def stability_detail(self) -> str:

        ratio = self._stability_ratio()
        return (
            f"Week-to-week standard deviation {self.refit_std_in_spread:.2f}, "
            f"{ratio:.0%} of the mean weight ({self.weight_in_spread:.2f}) across "
            "every weekly refit."
        )

    @property
    def caption(self) -> str:
        return CLASSIFICATION_CAPTIONS.get(self.classification, "not enough data to say either way")

    def _stability_ratio(self) -> float:
        return self.refit_std_in_spread / self.weight_in_spread if self.weight_in_spread else 0.0


@dataclass(frozen=True)
class ModelExplanation:
    run_directory: str
    created_at_utc: str | None
    feature_profile: str | None
    ridge_alpha: float | None
    start_season: int | None
    end_season: int | None
    refit_weeks: int | None
    feature_table_sha256: str | None
    matches_active_feature_table: bool | None
    families: tuple[FamilyExplanation, ...]


def _number(value: Any) -> float:

    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if result != result else result


def _metadata_feature_table_sha256(metadata: dict[str, Any]) -> str | None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        return None
    feature_table = provenance.get("feature_table")
    if not isinstance(feature_table, dict):
        return None
    value = feature_table.get("sha256")
    return str(value) if value else None


def _active_feature_table_sha256(artifacts_root: Path) -> str | None:

    path = artifacts_root / "active_ats_model.json"
    if not path.is_file():
        return None
    try:
        manifest = read_json(path)
    except (ValueError, OSError):
        return None
    value = manifest.get("feature_table_sha256")
    return str(value) if value else None


def _parse_family_row(row: pd.Series) -> FamilyExplanation:
    family = str(row.get("family", ""))
    return FamilyExplanation(
        family=family,
        label=FAMILY_PHRASES.get(family, family.replace("_", " ")),
        margin_share=_number(row.get("margin_share")),
        spread_share=_number(row.get("spread_share")),
        weight_in_spread=_number(row.get("weight_in_spread")),
        refit_std_in_spread=_number(row.get("refit_std_in_spread")),
        classification=str(row.get("classification", "")),
    )


def load_model_explanation(artifacts_root: Path) -> ModelExplanation | None:

    root = artifacts_root / "market_decomposition"
    directories = artifact_directories(root, "classification.csv")
    if not directories:
        return None
    active_sha256 = _active_feature_table_sha256(artifacts_root)
    for directory in directories:
        metadata_path = directory / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = read_json(metadata_path)
            except (ValueError, OSError):
                metadata = {}
        try:
            classification = pd.read_csv(directory / "classification.csv")
        except (ValueError, OSError):
            continue
        required = {"family", "margin_share", "spread_share", "classification"}
        if classification.empty or not required.issubset(classification.columns):
            continue
        feature_sha256 = _metadata_feature_table_sha256(metadata)
        matches_active = (
            None
            if active_sha256 is None or feature_sha256 is None
            else feature_sha256 == active_sha256
        )
        created_at = metadata.get("created_at_utc")
        return ModelExplanation(
            run_directory=directory.name,
            created_at_utc=str(created_at) if created_at else None,
            feature_profile=(
                str(metadata.get("feature_profile")) if metadata.get("feature_profile") else None
            ),
            ridge_alpha=(
                float(metadata["ridge_alpha"])
                if isinstance(metadata.get("ridge_alpha"), (int, float))
                else None
            ),
            start_season=(
                int(metadata["start_season"]) if metadata.get("start_season") is not None else None
            ),
            end_season=(
                int(metadata["end_season"]) if metadata.get("end_season") is not None else None
            ),
            refit_weeks=(
                int(metadata["refit_weeks"]) if metadata.get("refit_weeks") is not None else None
            ),
            feature_table_sha256=feature_sha256,
            matches_active_feature_table=matches_active,
            families=tuple(_parse_family_row(row) for _, row in classification.iterrows()),
        )
    return None


def market_decomposition_run_count(artifacts_root: Path) -> int:

    return len(artifact_directories(artifacts_root / "market_decomposition", "classification.csv"))


def render_model_explanation_section(explanation: ModelExplanation) -> str:

    out = [
        '<div style="margin-top:40px;max-width:80ch;">',
        '<p class="kicker">HOW THE MODEL DECIDES</p>',
        '<h3 class="title page-title" style="margin-bottom:6px;">What the model leans on</h3>',
        '<p class="sub">Reality&#8217;s share of the model&#8217;s weight against the '
        "market&#8217;s own share, family by family, with how steadily each weight holds "
        "up across weekly refits.</p>",
        "</div>",
    ]
    if explanation.matches_active_feature_table is False:
        out.append(
            '<p class="status warning" style="margin-top:12px;">'
            "<span>&#9888;</span><span>These weights were measured on an earlier build of "
            "the model&#8217;s inputs than the one behind this week&#8217;s picks -- treat "
            "them as describing that earlier build until "
            "<code>nfl-ats market-decomposition</code> is re-run.</span></p>"
        )
    out.append(
        '<p class="fine" style="margin-top:12px;max-width:75ch;">This model never sees '
        "the market&#8217;s own line -- on purpose, so the comparison below is not "
        "circular. That also means it is not the exact model that made this week&#8217;s "
        "picks; it shows what a model blind to the market&#8217;s line would have leaned "
        "on.</p>"
    )
    out.append('<table class="data" style="margin-top:14px;max-width:100%;">')
    out.append(
        "<thead><tr><th>Feature family</th><th>Reality weight share</th>"
        "<th>Market weight share</th><th>What that reads as</th>"
        "<th>Weight stability</th></tr></thead><tbody>"
    )
    for row in explanation.families:
        out.append(
            "<tr>"
            f"<td>{escape(row.label)}</td>"
            f'<td class="num">{row.margin_share * 100:.1f}%</td>'
            f'<td class="num">{row.spread_share * 100:.1f}%</td>'
            f"<td>{escape(row.caption)}</td>"
            "<td>"
            f"<strong>{escape(row.stability_word)}</strong>"
            f'<br><span class="fine">{escape(row.stability_detail)}</span>'
            "</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    provenance_bits = []
    if explanation.start_season is not None and explanation.end_season is not None:
        provenance_bits.append(
            f"fit on {explanation.start_season}&ndash;{explanation.end_season} completed games"
        )
    if explanation.refit_weeks is not None:
        provenance_bits.append(f"{explanation.refit_weeks} weekly refits")
    if explanation.ridge_alpha is not None:
        provenance_bits.append(f"ridge alpha {explanation.ridge_alpha:g}")
    if explanation.feature_profile:
        provenance_bits.append(f"feature profile {escape(explanation.feature_profile)}")
    stamp = ""
    if explanation.created_at_utc:
        parsed = _parse_utc(explanation.created_at_utc)
        if parsed is not None:
            stamp = f"; artifact written {parsed.strftime('%Y-%m-%d %H:%M UTC')}"
    if provenance_bits:
        out.append(
            f'<p class="fine" style="margin-top:10px;">Provenance: '
            f"{', '.join(provenance_bits)}{stamp}. Per-family weights come from the "
            "walk-forward market decomposition; the raw named coefficients live in that "
            "artifact&#8217;s <code>coefficients.csv</code>.</p>"
        )
    out.append('<div class="card" style="margin-top:18px;">')
    out.append('<p class="title" style="margin-bottom:8px;">Read this table honestly</p>')
    for title, body in HONESTY_NOTES:
        out.append(f"<p class='prose'><strong>{escape(title)}.</strong> {escape(body)}</p>")
    out.append("</div>")
    return "".join(out)


def _parse_utc(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


_EXPLANATION_UNAVAILABLE_HTML = (
    '<div class="card" style="border-left:3px solid var(--warning);margin-top:14px;">'
    '<p class="kicker" style="color:var(--warning);font-weight:700;">'
    "&#9888; MODEL EXPLANATION UNAVAILABLE</p>"
    '<p class="fine">A market-decomposition run exists but could not be read '
    "({detail}); the rest of this page is unaffected.</p></div>"
)

_EXPLANATION_EMPTY_HTML = (
    '<div class="card" style="margin-top:14px;">'
    '<p class="title" style="margin-bottom:6px;">No model-explanation run saved yet</p>'
    '<p class="fine">This section fills in once <code>nfl-ats market-decomposition</code> '
    "has been run. That command is optional and manual, so a fresh clone legitimately has "
    "nothing here yet -- nothing is hidden.</p></div>"
)


def load_model_explanation_html(artifacts_root: Path) -> str:

    count = market_decomposition_run_count(artifacts_root)
    if count == 0:
        return _EXPLANATION_EMPTY_HTML
    explanation = load_model_explanation(artifacts_root)
    if explanation is None:
        detail = escape(f"{count} run(s) present, none readable")
        return _EXPLANATION_UNAVAILABLE_HTML.replace("{detail}", detail)
    return render_model_explanation_section(explanation)
