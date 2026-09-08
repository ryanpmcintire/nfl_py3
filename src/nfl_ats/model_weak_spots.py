"""Opener-only spread diagnostics; no pick changes or fitted parameters.

Two tables come out of one saved opener evaluation: the per-bucket record
(:class:`WeakSpotRow`) and, since 2026-09-07, the same buckets split by
whether the HOME team opened as the favourite or the underdog
(:class:`HomeSplitRow`) -- lane L's diagnosis localised the big-spread
weakness to home underdogs, so the split is published as a diagnosis, never
as a pick flip (AGENTS.md, "No unexplained threshold flips").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

# Disjoint half-point ranges: 7.5 belongs to the fourth bucket.
SPREAD_BUCKETS: tuple[tuple[str, float, float, Literal["both", "right", "neither"]], ...] = (
    ("0-3", 0.0, 3.0, "both"),
    ("3.5-6.5", 3.0, 6.5, "right"),
    ("7-7.5", 6.5, 7.5, "neither"),
    ("7.5-10", 7.5, 10.0, "both"),
    ("10.5+", 10.0, float("inf"), "right"),
)
UNAVAILABLE = "No matching opener record is available for the current model yet."
EXPLANATION = (
    "The model's confidence barely changes with the size of the spread. "
    "Final margins pile up on 3, 7, 10 and 14 points, so lines just inside those "
    "numbers can expose weaknesses that an average confidence hides. "
    "Since the 2026 opener the forecast includes an adjustment for how home teams do on big "
    "spreads, learned from past seasons; the numbers here are the record before that change."
)
BUCKET_NOTE = (
    "Opening lines, ties excluded; 7.5-point lines belong to 7.5-10, not 7-7.5. "
    "At level odds there is no favourite or underdog."
)
HOME_SPLIT_LEAD = (
    "The same games, split by whether the home team opened as the favourite or the underdog, "
    "with how often the home team covered against what the model expected. "
    "The honest takeaway: home underdogs on big spreads have covered more often than the "
    "model expected; a fix is being built, and until then this is the record."
)
HOME_SPLIT_NOTE = "Level-odds lines have no home favourite or underdog and are left out."
HOME_FAVOURITE = "Home favourite"
HOME_UNDERDOG = "Home underdog"


def percent(value: float | None) -> str:
    return "--" if value is None else f"{value:.1%}"


@dataclass(frozen=True)
class WeakSpotRow:
    spread: str
    games: int
    accuracy: float | None
    favourite_pick_rate: float | None
    favourite_cover_rate: float | None
    confidence: float | None
    favourite_accuracy: float | None
    underdog_accuracy: float | None

    @property
    def cells(self) -> tuple[str, ...]:
        return (
            self.spread,
            str(self.games),
            *(
                percent(v)
                for v in (
                    self.accuracy,
                    self.favourite_pick_rate,
                    self.favourite_cover_rate,
                    self.confidence,
                    self.favourite_accuracy,
                    self.underdog_accuracy,
                )
            ),
        )

    @property
    def reliability(self) -> str:
        if self.accuracy is None or self.confidence is None:
            return f"On {self.spread} point spreads, no decided games are available."
        return (
            f"On {self.spread} point spreads the model said about {self.confidence:.0%} "
            f"and was right {self.accuracy:.0%} of the time."
        )


@dataclass(frozen=True)
class HomeSplitRow:
    """One spread bucket for games where the home team opened on one side of the line."""

    spread: str
    home_side: str
    games: int
    home_cover_rate: float | None
    home_confidence: float | None
    accuracy: float | None

    @property
    def cells(self) -> tuple[str, ...]:
        return (
            self.spread,
            self.home_side,
            str(self.games),
            percent(self.home_cover_rate),
            percent(self.home_confidence),
            percent(self.accuracy),
        )

    @property
    def plain(self) -> str:
        side = self.home_side.lower()
        if self.home_cover_rate is None or self.home_confidence is None or self.accuracy is None:
            return f"On {self.spread} point spreads with a {side}, no decided games are available."
        games = f"{self.games} game{'s' if self.games != 1 else ''}"
        return (
            f"On {self.spread} point spreads with a {side} ({games}) the home team "
            f"covered {self.home_cover_rate:.0%} of the time against the model's expected "
            f"{self.home_confidence:.0%}, and the model's pick was right {self.accuracy:.0%}."
        )


@dataclass(frozen=True)
class WeakSpots:
    rows: tuple[WeakSpotRow, ...] = ()
    home_split: tuple[HomeSplitRow, ...] = ()

    @property
    def home_split_text(self) -> str:
        if not self.home_split:
            return UNAVAILABLE
        return (
            HOME_SPLIT_LEAD
            + " "
            + HOME_SPLIT_NOTE
            + " "
            + " ".join(row.plain for row in self.home_split)
        )

    @property
    def text(self) -> str:
        if not self.rows:
            return UNAVAILABLE
        return (
            EXPLANATION
            + " "
            + BUCKET_NOTE
            + " "
            + " ".join(
                row.reliability
                + f" {row.games} games; stated confidence {percent(row.confidence)}; "
                f"favourite picks {percent(row.favourite_pick_rate)}; "
                f"favourites covered {percent(row.favourite_cover_rate)}; right on favourite picks "
                f"{percent(row.favourite_accuracy)}, on underdog picks "
                f"{percent(row.underdog_accuracy)}."
                for row in self.rows
            )
        )


def build_weak_spots(frame: pd.DataFrame) -> WeakSpots:
    """Summarize the saved probability picks; pushes never enter a denominator."""
    spread = pd.to_numeric(frame["tue_open_home_spread"], errors="coerce")
    margin = pd.to_numeric(frame["margin_vs_open"], errors="coerce")
    correct = pd.to_numeric(frame["correct_at_open_probability_rule"], errors="coerce")
    probability = pd.to_numeric(frame["home_cover_probability_at_open"], errors="coerce")
    pick = frame["pick_home_at_open_probability_rule"]
    valid = (
        spread.notna()
        & margin.notna()
        & margin.ne(0)
        & correct.isin([0, 1])
        & probability.between(0, 1)
        & pick.notna()
    )
    home_pick = pick.eq(True)
    # Favourite identity follows the saved home line. nflverse convention,
    # verified on the archive 2026-09-07 (positive home spread -> mean home
    # result +5.85 over 908 games): a POSITIVE home spread means the HOME team
    # is favoured. The first cut of this module had the sign reversed.
    favourite = home_pick.eq(spread.gt(0)) & spread.ne(0)
    underdog = ~favourite & spread.ne(0)
    favourite_cover = margin.gt(0).eq(spread.gt(0))
    home_cover = margin.gt(0)
    confidence = probability.where(home_pick, 1 - probability)

    def mean(series: pd.Series, mask: pd.Series) -> float | None:
        return float(series[mask].mean()) if mask.any() else None

    rows = []
    home_split = []
    for label, lower, upper, inclusive in SPREAD_BUCKETS:
        mask = valid & spread.abs().between(lower, upper, inclusive=inclusive)
        sided = mask & spread.ne(0)
        for side_label, side_mask in (
            (HOME_FAVOURITE, spread.gt(0)),
            (HOME_UNDERDOG, spread.lt(0)),
        ):
            side = mask & side_mask
            home_split.append(
                HomeSplitRow(
                    label,
                    side_label,
                    int(side.sum()),
                    mean(home_cover, side),
                    mean(probability, side),
                    mean(correct, side),
                )
            )
        rows.append(
            WeakSpotRow(
                label,
                int(mask.sum()),
                mean(correct, mask),
                mean(favourite, sided),
                mean(favourite_cover, sided),
                mean(confidence, mask),
                mean(correct, mask & favourite),
                mean(correct, mask & underdog),
            )
        )
    return WeakSpots(tuple(rows), tuple(home_split))
