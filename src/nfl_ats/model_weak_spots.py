"""Opener-only spread diagnostics; no pick changes or fitted parameters.

Two tables come out of one saved opener evaluation: the per-bucket record
(:class:`WeakSpotRow`) and, since 2026-09-07, the same buckets split by
whether the HOME team opened as the favourite or the underdog
(:class:`HomeSplitRow`) -- lane L's diagnosis localised the big-spread
weakness to home underdogs, so the split is published as a diagnosis, never
as a pick flip (AGENTS.md, "No unexplained threshold flips").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from nfl_ats.spread_regime import BUCKETS, spread_bucket

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
    "Since the 2026 opener the forecast includes a small push toward the home team, sized by "
    "the spread and learned from past seasons; the numbers here are the record with that push on."
)
EXPLANATION_UNALIGNED = (
    "The model's confidence barely changes with the size of the spread. "
    "Final margins pile up on 3, 7, 10 and 14 points, so lines just inside those "
    "numbers can expose weaknesses that an average confidence hides. "
    "These numbers are the saved opening-line record before the home-team push was measured."
)
BUCKET_NOTE = (
    "Opening lines, ties excluded; 7.5-point lines belong to 7.5-10, not 7-7.5. "
    "At level odds there is no favourite or underdog."
)
HOME_SPLIT_LEAD = (
    "The same games, split by whether the home team opened as the favourite or the underdog, "
    "with how often the home team covered against what the model expected. "
    "The honest takeaway: home underdogs on big spreads have covered more often than the "
    "model expected; the home-side push below is the fix, and this is the record with it on."
)
HOME_SPLIT_LEAD_UNALIGNED = (
    "The same games, split by whether the home team opened as the favourite or the underdog, "
    "with how often the home team covered against what the model expected. "
    "The honest takeaway: home underdogs on big spreads have covered more often than the "
    "model expected; this is the record before the home-side push was measured."
)
HOME_CORRECTION_LEAD = (
    "Since the 2026 opener the model's point forecast gets a small push toward the home team on "
    "spreads of seven points or more, sized by the spread and learned only from games already "
    "played; smaller spreads are left alone, because that is where the error was found. The push "
    "for this week's card is listed by spread size, with what the same rule did on the 2020-2025 "
    "archive."
)
HOME_CORRECTION_NOTE = (
    "A positive push moves the forecast toward the home team. On its own the push is a small "
    "change either way; the published score above is measured with it on."
)
HOME_CORRECTION_UNAVAILABLE = (
    "The home-side push has not been measured on the current model's opener record yet."
)
HOME_SPLIT_NOTE = "Level-odds lines have no home favourite or underdog and are left out."
HOME_FAVOURITE = "Home favourite"
HOME_UNDERDOG = "Home underdog"


def percent(value: float | None) -> str:
    return "--" if value is None else f"{value:.1%}"


def points(value: float | None) -> str:
    return "--" if value is None else f"{value:+.2f}"


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
class HomeCorrectionRow:
    """One spread bucket of the served home-side push (MOD-18 lane S)."""

    spread: str
    this_week_points: float | None
    learned_from_games: int | None
    archive_games: int
    picks_changed: int
    accuracy_with: float | None
    accuracy_without: float | None

    @property
    def cells(self) -> tuple[str, ...]:
        return (
            self.spread,
            points(self.this_week_points),
            "--" if self.learned_from_games is None else str(self.learned_from_games),
            str(self.archive_games),
            str(self.picks_changed),
            percent(self.accuracy_with),
            percent(self.accuracy_without),
        )

    @property
    def plain(self) -> str:
        push = (
            "this week's push is unavailable"
            if self.this_week_points is None
            else f"this week's push {points(self.this_week_points)} points"
        )
        return (
            f"On {self.spread} point spreads: {push}; on the archive it changed "
            f"{self.picks_changed} of {self.archive_games} picks, right "
            f"{percent(self.accuracy_with)} with the push and "
            f"{percent(self.accuracy_without)} without."
        )


@dataclass(frozen=True)
class HomeCorrection:
    """The served home-side push: this week's values plus its archive record."""

    rows: tuple[HomeCorrectionRow, ...] = ()
    archive_games: int = 0
    picks_changed: int = 0
    accuracy_with: float | None = None
    accuracy_without: float | None = None
    this_week_available: bool = False

    @property
    def summary(self) -> str:
        return (
            f"Across the archive the push changed {self.picks_changed} of {self.archive_games} "
            f"picks; the model alone was right {percent(self.accuracy_with)} with it and "
            f"{percent(self.accuracy_without)} without."
        )


@dataclass(frozen=True)
class WeakSpots:
    rows: tuple[WeakSpotRow, ...] = ()
    home_split: tuple[HomeSplitRow, ...] = ()
    home_correction: HomeCorrection | None = None

    @property
    def explanation(self) -> str:
        """The lead sentence for the bucket table: says the push is in the
        record only when the push was actually measured on it."""

        return EXPLANATION if self.home_correction is not None else EXPLANATION_UNALIGNED

    @property
    def home_split_lead(self) -> str:
        return HOME_SPLIT_LEAD if self.home_correction is not None else HOME_SPLIT_LEAD_UNALIGNED

    @property
    def home_correction_text(self) -> str:
        if self.home_correction is None or not self.home_correction.rows:
            return HOME_CORRECTION_UNAVAILABLE
        return (
            HOME_CORRECTION_LEAD
            + " "
            + self.home_correction.summary
            + " "
            + " ".join(row.plain for row in self.home_correction.rows)
            + " "
            + HOME_CORRECTION_NOTE
        )

    @property
    def home_split_text(self) -> str:
        if not self.home_split:
            return UNAVAILABLE
        return (
            self.home_split_lead
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
            self.explanation
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


def build_home_correction(
    frame: pd.DataFrame,
    this_week_offsets: Mapping[str, float] | None = None,
    this_week_prior_games: Mapping[str, int] | None = None,
) -> HomeCorrection | None:
    """The served home-side push by spread bucket, from a saved opener evaluation.

    ``frame`` is the evaluation's ``per_game`` table; it must carry the raw
    twins the aligned evaluation writes (``*_probability_rule_raw``) or the
    push was not measured on this record and ``None`` is returned. This
    week's values come from the served forecast's sidecar when given.
    """

    needed = {
        "tue_open_home_spread",
        "margin_vs_open",
        "home_side_offset_at_open",
        "pick_home_at_open_probability_rule",
        "pick_home_at_open_probability_rule_raw",
        "correct_at_open_probability_rule",
        "correct_at_open_probability_rule_raw",
    }
    if not needed.issubset(frame.columns):
        return None
    spread = pd.to_numeric(frame["tue_open_home_spread"], errors="coerce")
    margin = pd.to_numeric(frame["margin_vs_open"], errors="coerce")
    with_push = pd.to_numeric(frame["correct_at_open_probability_rule"], errors="coerce")
    without = pd.to_numeric(frame["correct_at_open_probability_rule_raw"], errors="coerce")
    changed = frame["pick_home_at_open_probability_rule"].astype(bool) != frame[
        "pick_home_at_open_probability_rule_raw"
    ].astype(bool)
    valid = spread.notna() & margin.notna() & margin.ne(0) & with_push.isin([0, 1])
    buckets = spread_bucket(spread.fillna(0.0)).where(spread.notna())

    def mean(series: pd.Series, mask: pd.Series) -> float | None:
        return float(series[mask].mean()) if mask.any() else None

    rows = []
    for bucket in BUCKETS:
        mask = valid & buckets.eq(bucket)
        offset = None if this_week_offsets is None else this_week_offsets.get(bucket)
        prior = None if this_week_prior_games is None else this_week_prior_games.get(bucket)
        rows.append(
            HomeCorrectionRow(
                spread=bucket,
                this_week_points=None if offset is None else float(offset),
                learned_from_games=None if prior is None else int(prior),
                archive_games=int(mask.sum()),
                picks_changed=int((changed & mask).sum()),
                accuracy_with=mean(with_push, mask),
                accuracy_without=mean(without, mask),
            )
        )
    return HomeCorrection(
        rows=tuple(rows),
        archive_games=int(valid.sum()),
        picks_changed=int((changed & valid).sum()),
        accuracy_with=mean(with_push, valid),
        accuracy_without=mean(without, valid),
        this_week_available=this_week_offsets is not None,
    )
