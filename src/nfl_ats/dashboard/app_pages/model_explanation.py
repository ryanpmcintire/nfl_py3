"""How the model decides -- the market-decomposition explanation view.

Two sections, both read from the latest ``nfl-ats market-decomposition`` run
(optional and manual -- most trees, and every fresh clone, have not run it
yet, so the whole page feature-detects and renders an honest empty state):

1. **What the model weighs, family by family** -- reality's share of weight
   vs. the market's share, plus a plain-English caveat label. This mirrors
   the public findings page's honesty-first framing: the ``unpriced_predictive``
   bucket -- the one that looks most like a lead -- reads as an unconfirmed
   lean the market does not seem to price, never as a discovered edge. See
   the honesty block at the foot of this page and
   :mod:`nfl_ats.market_decomposition`'s own module docstring, which this
   page's wording is required to stay consistent with.
2. **This week's breakdown, game by game** -- the exact per-game,
   per-family attribution the public picks page sources its single
   strong-lean sentence from (:func:`nfl_ats.market_decomposition.explain_game_structured`),
   shown here in full for *every* game, not just strong leans -- the
   pick-relevant strong-lean gate stays on the public site; this page is the
   research view.

One asymmetry worth restating plainly, every time this page renders: the
decomposition's residual model is refit *without* ``spread_line`` as an
input, on purpose, so the market-vs-reality comparison is not circular. That
also means it is **not** numerically identical to the deployed
``market_residual`` model that actually made this week's picks (which does
condition on the market's own line) -- this page explains what a model blind
to the market's line would have prioritized, not "why the deployed model
picked X."

Sync/staleness idiom matches every other optional artifact
(:mod:`nfl_ats.dashboard.state`): the artifact's own
``provenance.feature_table.sha256`` is compared against the active model's,
mirroring ``_opener_state``'s pattern. On a mismatch the page still shows the
numbers -- they are real measurements -- but says plainly, once, up top, that
they were taken on an earlier build than this week's picks.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from nfl_ats.dashboard import theme, viz
from nfl_ats.dashboard.state import load_csv, load_parquet, project_state
from nfl_ats.market_decomposition import FAMILY_PHRASES, explain_game_structured

# A family's market-weight refit-to-refit standard deviation, as a fraction of
# its mean weight (a coefficient-of-variation-style ratio), at or above which
# the stability note reads "jumps around between refits" rather than "steady
# across refits". This is page-wording judgment (which caption a number
# earns), not a modeling threshold, so it is declared here rather than in
# nfl_ats.market_decomposition -- same spirit as that module's own declared,
# non-magic thresholds, just scoped to this page's caption instead of its math.
STABILITY_JUMPY_RATIO = 0.15

# Plain-English translations of the four classification buckets. The
# "unpriced_predictive" wording is deliberate and fixed: this bucket is the
# one families most resembling "a lead" fall into, and the module's own
# honesty notes are emphatic that it is a diagnostic, not evidence of edge --
# so the caption says "unconfirmed" out loud rather than reading as a find.
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
        "This page generates hypotheses about what the market might be missing. It does "
        "not adjudicate them -- only the project's own outer-season, prospectively scored "
        'record does that. Nothing on this page should be read as "found an edge."',
    ),
    (
        "Family-level only",
        "Coefficients are only meaningful once added up to a whole feature family -- ridge "
        "regression smears weight across correlated features, so an individual feature's "
        "number, if you went looking for one, would not mean what it looks like it means.",
    ),
    (
        "Explains a pattern, scores no new picks",
        "This is fit on seasons the project has already looked at and scored, repeatedly, "
        "elsewhere. It explains what the model leans on in hindsight; it does not grade a "
        "new stream of picks.",
    ),
    (
        "Blind to the market's own line, on purpose",
        "The model behind this page never sees the market's line -- that is what makes "
        '"what does the market not see" a meaningful question instead of a circular one. '
        "It is therefore not the exact model that made this week's picks; it shows what a "
        "model blind to the market's line would have prioritized.",
    ),
)


def _html(inner: str) -> None:
    """Render one composed fragment inside the scoped ``.ats`` root."""

    st.html(f'<div class="ats">{inner}</div>', unsafe_allow_javascript=True)


def _family_label(family: str) -> str:
    return FAMILY_PHRASES.get(family, family.replace("_", " "))


def _number(value: Any, default: float = 0.0) -> float:
    """Coerce an artifact cell to a float, treating NaN the same as missing."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result  # NaN != NaN


def _stability_note(refit_std: float, mean_weight: float) -> tuple[str, str]:
    """(headline word, hover/secondary detail) -- exact numbers stay off the headline."""

    ratio = (refit_std / mean_weight) if mean_weight else 0.0
    word = (
        "jumps around between refits" if ratio >= STABILITY_JUMPY_RATIO else "steady across refits"
    )
    detail = (
        f"Week-to-week standard deviation {refit_std:.2f}, {ratio:.0%} of the mean weight "
        f"({mean_weight:.2f}) across every weekly refit."
    )
    return word, detail


def _section_header(kicker: str, title: str, sub: str) -> str:
    return (
        '<div style="margin:30px 0 14px;max-width:70ch;">'
        f'<p class="kicker">{escape(kicker)}</p>'
        f'<h3 class="title" style="font-size:20px;margin-bottom:6px;">{escape(title)}</h3>'
        f'<p class="sub">{escape(sub)}</p></div>'
    )


# ---------------------------------------------------------------------------
# The anchor
# ---------------------------------------------------------------------------

state = project_state()
root_directory = state.market_decomposition.directory

st.html(theme.stylesheet())

_html(
    viz.page_header(
        "How the model decides",
        "What the model leans on, and what the market seems to miss",
        "A research view, not a new pick stream: family-level weights and the "
        "per-game breakdown behind this week's card, with the same honesty rules "
        "as the rest of this dashboard.",
    )
)

if state.market_decomposition.consistent_with_active is False:
    _html(
        viz.card(
            viz.status_line(
                "warning",
                "This breakdown was measured on an earlier build of the model's inputs "
                "than the one behind this week's picks -- treat the numbers below as "
                "stale until `nfl-ats market-decomposition` is re-run.",
            )
        )
    )

_html(
    viz.card(
        '<p class="fine">This model never sees the market\'s own line -- on purpose, so '
        "the comparison below is not circular. That also means it is not the exact model "
        "that made this week's picks; it shows what a model blind to the market's line "
        "would have leaned on.</p>"
    )
)

if root_directory is None:
    _html(
        viz.empty_state(
            "No model-explanation run saved yet",
            "This page fills in once `nfl-ats market-decomposition` has been run. It is "
            "optional and manual, so a fresh clone or an early season legitimately has "
            "nothing here yet.",
        )
    )
    st.html(theme.theme_sync_script(), unsafe_allow_javascript=True)
    st.stop()

classification = load_csv(root_directory / "classification.csv")
attribution = load_parquet(root_directory / "attribution.parquet")

# ---------------------------------------------------------------------------
# 1. What the model weighs, family by family
# ---------------------------------------------------------------------------

_html(
    _section_header(
        "By feature family",
        "What the model weighs, family by family",
        "Reality's weight (how much the outcome actually rewards this) against the "
        "market's own weight (how much the line already prices it in).",
    )
)

if classification.empty:
    _html(
        viz.empty_state(
            "No family weights saved",
            "This run predates the family-weight table, or it has not finished writing.",
        )
    )
else:
    scale_max = (
        max(
            max(_number(row.get("margin_share")), _number(row.get("spread_share")))
            for _, row in classification.iterrows()
        )
        * 1.15
    ) or 1.0

    family_cards = []
    for _, row in classification.iterrows():
        family = str(row.get("family", ""))
        label = _family_label(family)
        classification_word = str(row.get("classification", ""))
        caption = CLASSIFICATION_CAPTIONS.get(
            classification_word, "not enough data to say either way"
        )
        stability_word, stability_detail = _stability_note(
            _number(row.get("refit_std_in_spread")), _number(row.get("weight_in_spread"))
        )
        bars = viz.family_comparison_bars(
            reality_share=_number(row.get("margin_share")),
            market_share=_number(row.get("spread_share")),
            scale_max=scale_max,
        )
        family_cards.append(
            viz.card(
                f'<p class="title" style="font-size:16px;margin-bottom:10px;">'
                f"{escape(label)}</p>"
                f"{bars}"
                '<div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;'
                'align-items:center;">'
                f'<span class="chip">{escape(caption)}</span>'
                f'<span class="fine" title="{escape(stability_detail)}">'
                f"The market&#8217;s own weight on this is "
                f"<strong>{escape(stability_word)}</strong>.</span>"
                "</div>"
            )
        )
    _html('<div class="stack">' + "".join(family_cards) + "</div>")

honesty_cards = [
    viz.card(
        f'<p class="title" style="margin-bottom:8px;">{escape(title)}</p>'
        f'<div class="prose"><p>{escape(body)}</p></div>'
    )
    for title, body in HONESTY_NOTES
]
_html(
    '<div class="row" style="margin-top:16px;">'
    + "".join(f'<div style="display:grid;">{card}</div>' for card in honesty_cards)
    + "</div>"
)

# ---------------------------------------------------------------------------
# 2. This week's breakdown, game by game
# ---------------------------------------------------------------------------

_html(
    _section_header(
        "By game",
        "This week's breakdown, game by game",
        "Every game on this week's card, not just the strong leans -- the pick-relevant "
        "gate stays on the picks page. This is the full, research-level view.",
    )
)

if state.forecast is None:
    _html(
        viz.empty_state(
            "No pick card yet",
            "This section fills in once this week's card exists (`nfl-ats margin-predict`).",
        )
    )
else:
    recommendations = state.forecast.recommendations
    ordered = recommendations.sort_values(["kickoff", "game_id"], na_position="last")
    game_cards = []
    for _, row in ordered.iterrows():
        game_id = str(row["game_id"])
        home, away = str(row["home_team"]), str(row["away_team"])
        game_attribution = (
            attribution.loc[attribution["game_id"].astype(str).eq(game_id)]
            if not attribution.empty and "game_id" in attribution.columns
            else pd.DataFrame()
        )
        if game_attribution.empty:
            game_cards.append(
                viz.card(
                    f'<p class="title" style="font-size:17px;">{escape(away)} at '
                    f"{escape(home)}</p>"
                    '<p class="fine" style="margin-top:8px;">No per-family attribution '
                    "saved for this game yet.</p>"
                )
            )
            continue
        family_totals = {
            str(family): float(contribution)
            for family, contribution in zip(
                game_attribution["family"], game_attribution["contribution"], strict=True
            )
        }
        predicted_residual = float(game_attribution["predicted_residual"].iloc[0])
        explanation = explain_game_structured(
            game_id=game_id,
            home_team=home,
            away_team=away,
            predicted_residual=predicted_residual,
            family_contributions=family_totals,
            max_drivers=len(family_totals),
            max_offsets=len(family_totals),
        )
        combined = sorted(
            (
                (driver.phrase, driver.points)
                for driver in (*explanation.drivers, *explanation.offsets)
            ),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        bars = viz.contribution_bars(combined, pick_text=f"{explanation.pick_team} to cover")
        game_cards.append(
            viz.card(
                f'<p class="title" style="font-size:17px;">{escape(away)} at {escape(home)}</p>'
                f'<p class="sub" style="margin:4px 0 12px;">{escape(explanation.sentence)}</p>'
                f"{bars}"
            )
        )
    _html('<div class="stack">' + "".join(game_cards) + "</div>")

st.html(theme.theme_sync_script(), unsafe_allow_javascript=True)
