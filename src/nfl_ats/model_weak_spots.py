from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.spread_regime import BUCKETS, spread_bucket

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
DISPLAYED_CONFIDENCE_NOTE = (
    "The stated confidence column is the model's own number, before any correction. The score "
    "shown beside each pick on the picks page is not: it is pulled toward what picks of that "
    "size and that confidence have really hit, so a big favourite or big underdog is shown a "
    "smaller number than the model asked for. It is never shown below 50% on a side the card is "
    "picking. Past games that pointed that far down were few, and they finished nearer a coin "
    "flip than the number suggested."
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
SEASON_TIMING_EARLY_LABEL = "Weeks 1-9"
SEASON_TIMING_LATE_LABEL = "Weeks 10-18"
SEASON_TIMING_EARLY_MAX_WEEK = 9
SEASON_TIMING_SAMPLES = 2_000
SEASON_TIMING_SEED = 20260817
SEASON_TIMING_UNAVAILABLE = (
    "The time-of-season split has not been measured on the current model's opener record yet."
)
SEASON_TIMING_LEAD = (
    "The same served picks, split by whether the game fell in the first nine weeks of a "
    "season or the weeks after that, with a resampled range on the gap between the two halves."
)
SEASON_TIMING_PLAIN = (
    "The model has been sharper early in the season than late; the confidence beside a "
    "late-season pick does not yet say so."
)


def percent(value: float | None) -> str:
    return "--" if value is None else f"{value:.1%}"


def signed_percent(value: float | None) -> str:
    return "--" if value is None else f"{value:+.1%}"


def range_text(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "--"
    return f"[{lower:.1%}, {upper:.1%}]"


def likely_real(value: float | None) -> str:
    return "--" if value is None else f"{value:.0%} likely real"


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
class SeasonTimingRow:
    label: str
    games: int
    accuracy: float | None
    accuracy_lower: float | None
    accuracy_upper: float | None
    stated_confidence: float | None

    @property
    def cells(self) -> tuple[str, ...]:
        return (
            self.label,
            str(self.games),
            percent(self.accuracy),
            range_text(self.accuracy_lower, self.accuracy_upper),
            percent(self.stated_confidence),
        )


@dataclass(frozen=True)
class SeasonTiming:
    rows: tuple[SeasonTimingRow, ...] = ()
    accuracy_gap: float | None = None
    accuracy_gap_lower: float | None = None
    accuracy_gap_upper: float | None = None
    accuracy_gap_probability_positive: float | None = None
    confidence_gap: float | None = None
    confidence_gap_lower: float | None = None
    confidence_gap_upper: float | None = None
    confidence_gap_probability_positive: float | None = None

    @property
    def available(self) -> bool:
        return len(self.rows) == 2 and self.accuracy_gap is not None

    @property
    def summary(self) -> str:
        if not self.available:
            return SEASON_TIMING_UNAVAILABLE
        return (
            f"Early minus late, model right: {signed_percent(self.accuracy_gap)}, resampled "
            f"range {range_text(self.accuracy_gap_lower, self.accuracy_gap_upper)} "
            f"({likely_real(self.accuracy_gap_probability_positive)}). Stated confidence moved "
            f"{signed_percent(self.confidence_gap)} over the same split, resampled range "
            f"{range_text(self.confidence_gap_lower, self.confidence_gap_upper)} "
            f"({likely_real(self.confidence_gap_probability_positive)})."
        )

    @property
    def plain(self) -> str:
        return SEASON_TIMING_PLAIN if self.available else SEASON_TIMING_UNAVAILABLE


@dataclass(frozen=True)
class WeakSpots:
    rows: tuple[WeakSpotRow, ...] = ()
    home_split: tuple[HomeSplitRow, ...] = ()
    home_correction: HomeCorrection | None = None
    season_timing: SeasonTiming = field(default_factory=SeasonTiming)

    @property
    def explanation(self) -> str:

        return EXPLANATION if self.home_correction is not None else EXPLANATION_UNALIGNED

    @property
    def season_timing_text(self) -> str:
        if not self.season_timing.available:
            return SEASON_TIMING_UNAVAILABLE
        return (
            SEASON_TIMING_LEAD + " " + self.season_timing.summary + " " + self.season_timing.plain
        )

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
            + DISPLAYED_CONFIDENCE_NOTE
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


def build_season_timing(frame: pd.DataFrame) -> SeasonTiming:
    needed = {
        "season",
        "week",
        "tue_open_home_spread",
        "margin_vs_open",
        "correct_at_open_probability_rule",
        "home_cover_probability_at_open",
        "pick_home_at_open_probability_rule",
    }
    if not needed.issubset(frame.columns):
        return SeasonTiming()
    spread = pd.to_numeric(frame["tue_open_home_spread"], errors="coerce")
    margin = pd.to_numeric(frame["margin_vs_open"], errors="coerce")
    correct = pd.to_numeric(frame["correct_at_open_probability_rule"], errors="coerce")
    probability = pd.to_numeric(frame["home_cover_probability_at_open"], errors="coerce")
    pick = frame["pick_home_at_open_probability_rule"]
    week = pd.to_numeric(frame["week"], errors="coerce")
    season = pd.to_numeric(frame["season"], errors="coerce")
    valid = (
        spread.notna()
        & margin.notna()
        & margin.ne(0)
        & correct.isin([0, 1])
        & probability.between(0, 1)
        & pick.notna()
        & week.notna()
        & season.notna()
    )
    if not valid.any():
        return SeasonTiming()
    confidence = probability.where(pick.eq(True), 1 - probability)
    scored = pd.DataFrame(
        {
            "season": season[valid],
            "week": week[valid],
            "correct": correct[valid],
            "confidence": confidence[valid],
        }
    )
    scored["early"] = scored["week"] <= SEASON_TIMING_EARLY_MAX_WEEK
    if scored["early"].nunique() < 2:
        return SeasonTiming()

    def group_metric(inner: pd.DataFrame) -> dict[str, float]:
        return {
            "accuracy": float(inner["correct"].mean()),
            "confidence": float(inner["confidence"].mean()),
        }

    rows = []
    for label, early_flag in (
        (SEASON_TIMING_EARLY_LABEL, True),
        (SEASON_TIMING_LATE_LABEL, False),
    ):
        group = scored.loc[scored["early"].eq(early_flag)]
        if group.empty:
            rows.append(SeasonTimingRow(label, 0, None, None, None, None))
            continue
        point = group_metric(group)
        boot = week_blocked_bootstrap(
            group,
            group_metric,
            block="week",
            samples=SEASON_TIMING_SAMPLES,
            seed=SEASON_TIMING_SEED,
        )
        accuracy_row = boot.loc[boot["metric"].eq("accuracy")].iloc[0]
        rows.append(
            SeasonTimingRow(
                label,
                len(group),
                point["accuracy"],
                float(accuracy_row["lower"]),
                float(accuracy_row["upper"]),
                point["confidence"],
            )
        )

    def gap_metric(inner: pd.DataFrame) -> dict[str, float]:
        early = inner.loc[inner["early"]]
        late = inner.loc[~inner["early"]]
        if early.empty or late.empty:
            return {"accuracy_gap": 0.0, "confidence_gap": 0.0}
        return {
            "accuracy_gap": float(early["correct"].mean() - late["correct"].mean()),
            "confidence_gap": float(early["confidence"].mean() - late["confidence"].mean()),
        }

    gap_point = gap_metric(scored)
    gap_boot = week_blocked_bootstrap(
        scored, gap_metric, block="week", samples=SEASON_TIMING_SAMPLES, seed=SEASON_TIMING_SEED
    )
    accuracy_gap_row = gap_boot.loc[gap_boot["metric"].eq("accuracy_gap")].iloc[0]
    confidence_gap_row = gap_boot.loc[gap_boot["metric"].eq("confidence_gap")].iloc[0]
    return SeasonTiming(
        rows=tuple(rows),
        accuracy_gap=gap_point["accuracy_gap"],
        accuracy_gap_lower=float(accuracy_gap_row["lower"]),
        accuracy_gap_upper=float(accuracy_gap_row["upper"]),
        accuracy_gap_probability_positive=float(accuracy_gap_row["probability_positive"]),
        confidence_gap=gap_point["confidence_gap"],
        confidence_gap_lower=float(confidence_gap_row["lower"]),
        confidence_gap_upper=float(confidence_gap_row["upper"]),
        confidence_gap_probability_positive=float(confidence_gap_row["probability_positive"]),
    )
