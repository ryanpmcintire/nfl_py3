from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.displayed_confidence import (
    DISPLAYED_PICK_PROBABILITY_COLUMN,
    DISPLAYED_STRENGTH_WORD_COLUMN,
    PICK_SIDE_FLOOR,
)
from nfl_ats.published_picks import FrozenPick, frozen_picks
from nfl_ats.publishing import _publication_context
from nfl_ats.score_lattice import (
    _MAX_CENTRE_DISTANCE,
    pick_consistent_top_score,
    score_lattice,
)
from nfl_ats.served_total import (
    SERVED_TOTAL_METHOD,
    joint_residual_total_view,
    served_total,
    served_total_blend_k01,
)
from nfl_ats.served_total_challenger import ledger_path as totals_method_ledger_path
from nfl_ats.tiebreaker import (
    TOTAL_LOW_SIDE_SHADE_POINTS,
    TOTALS_RESIDUAL_WEIGHT,
    MarketConsensus,
    ModelView,
    TiebreakerConsistencyError,
    build_report,
    lined_finals,
    newest_schedules_path,
    snapshot_consensus,
)
from nfl_ats.totals import TotalsDataError
from nfl_ats.totals_wave2 import model_total_view_wave2

BEST_PICK_MARK = "★ "

TOLERANCE = 1e-9


@dataclass
class GameAudit:
    game_id: str
    home: str
    away: str
    pick_team: str
    pick_side: str
    printed_line: float
    spread_line: float
    home_cover_probability: float
    served_pick_cover: float
    displayed: float | None
    strength_word: str | None
    predicted_margin: float
    market_total: float
    method_total: float
    method_name: str
    served_total_value: float
    comparison_blend: float
    cell_home: int | None
    cell_away: int | None
    cell_total: int | None
    cell_margin: int | None
    cell_tolerance: float | None
    lattice_pick_cover: float | None
    lattice_push: float | None
    lattice_label: str
    note: str
    frozen: bool
    raw_home_cover_probability: float
    mechanism: str
    recorder_joint_arm: float


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pick_from_card(row: pd.Series) -> tuple[str, str, float]:
    raw = row.get("ATS prediction")
    if not isinstance(raw, str) or not raw.strip():
        home_pick = _number(row["home_cover_probability"]) >= PICK_SIDE_FLOOR
        team = str(row["home_team"]) if home_pick else str(row["away_team"])
        spread = _number(row["spread_line"])
        return team, ("HOME" if home_pick else "AWAY"), (-spread if home_pick else spread)
    printed = raw.replace(BEST_PICK_MARK, "").strip()
    team, _, line_text = printed.rpartition(" ")
    line = 0.0 if line_text.strip().upper() == "PK" else float(line_text)
    side = "HOME" if team == str(row["home_team"]) else "AWAY"
    return team, side, line


def _consensus(game_id: str, game: pd.Series, data_root: Path) -> MarketConsensus:
    found = snapshot_consensus(game_id, data_root)
    if found is not None:
        return found
    return MarketConsensus(
        game_id=game_id,
        home_expected_margin=float(game["spread_line"]),
        total_line=float(game["total_line"]),
        source="schedules (fallback -- possibly stale)",
    )


def _totals_views(game_id: str, data_root: Path) -> tuple[Any, Any]:
    wave2_path = data_root / "processed" / "game_features_pbp.parquet"
    try:
        blend_view = model_total_view_wave2(game_id, data_root, wave2_path)
    except (TotalsDataError, KeyError, OSError, TypeError, ValueError):
        blend_view = None
    try:
        joint_view = joint_residual_total_view(game_id, data_root)
    except (TotalsDataError, KeyError, OSError, TypeError, ValueError):
        joint_view = None
    return blend_view, joint_view


def audit_game(
    row: pd.Series,
    game: pd.Series,
    finals: pd.DataFrame,
    data_root: Path,
    frozen: FrozenPick | None,
    raw_home_probability: float,
) -> tuple[GameAudit, list[str]]:
    violations: list[str] = []
    game_id = str(row["game_id"])
    home, away = str(row["home_team"]), str(row["away_team"])
    pick_team, pick_side, printed_line = _pick_from_card(row)
    spread_line = _number(row["spread_line"])
    home_probability = _number(row["home_cover_probability"])
    served_pick_cover = home_probability if pick_side == "HOME" else 1.0 - home_probability
    displayed = row.get(DISPLAYED_PICK_PROBABILITY_COLUMN)
    displayed_value = None if displayed is None else _number(displayed)
    if displayed_value is not None and not np.isfinite(displayed_value):
        displayed_value = None
    word = row.get(DISPLAYED_STRENGTH_WORD_COLUMN)
    strength_word = str(word) if isinstance(word, str) and word.strip() else None
    predicted_margin = _number(row["predicted_margin"])

    if pick_team not in (home, away):
        violations.append(
            f"{game_id}: printed pick team {pick_team!r} is neither {home} nor {away}"
        )
    expected_line = -spread_line if pick_side == "HOME" else spread_line
    if abs(printed_line - expected_line) > TOLERANCE:
        violations.append(
            f"{game_id}: printed line {printed_line:+g} for the {pick_side} pick does not match "
            f"the served spread {spread_line:+g} (expected {expected_line:+g})"
        )

    consensus = _consensus(game_id, game, data_root)
    blend_view, joint_view = _totals_views(game_id, data_root)
    method_total, method_name = served_total(
        SERVED_TOTAL_METHOD,
        market_total=consensus.total_line,
        blend_view=blend_view,
        joint_view=joint_view,
        blend_weight=TOTALS_RESIDUAL_WEIGHT,
    )
    comparison_blend = served_total_blend_k01(
        consensus.total_line, blend_view, weight=TOTALS_RESIDUAL_WEIGHT
    )

    model_view = ModelView(
        predicted_margin=predicted_margin,
        forecast_line=spread_line,
        residual=_number(row["predicted_market_residual"]),
        source="audit (published card row)",
    )

    note = ""
    cell_home = cell_away = cell_total = cell_margin = None
    cell_tolerance: float | None = None
    served_total_value = method_total + TOTAL_LOW_SIDE_SHADE_POINTS
    lattice_pick_cover: float | None = None
    lattice_push: float | None = None
    lattice_label = ""
    try:
        report = build_report(
            game,
            consensus,
            finals,
            model_view,
            blend_view,
            joint_view,
            published_pick_side=pick_side,
        )
    except TiebreakerConsistencyError as error:
        note = f"no pick-consistent lattice cell: {error}"
        violations.append(f"{game_id}: {note}")
        report = None
    recorder_joint_arm = float("nan")
    if report is not None:
        served_total_value = report.served_total
        recorder_joint_arm = (
            report.served_total_before_shade
            if report.served_total_method == "joint_residual"
            else float("nan")
        )
        cell_home, cell_away = int(report.guess_home), int(report.guess_away)
        cell_total = cell_home + cell_away
        cell_margin = cell_home - cell_away
        lattice_pick_cover = report.pick_cover_probability
        lattice_push = report.pick_push_probability
        note = report.consistency_note

    lattice = score_lattice(finals, predicted_margin, served_total_value)
    lattice_label = lattice.label
    chosen = pick_consistent_top_score(
        lattice,
        pick_side=pick_side,
        spread_line=spread_line,
        served_total=served_total_value,
        centre_margin=predicted_margin,
    )
    if chosen is not None:
        cell_tolerance = float(chosen[3])

    raw_side = "HOME" if raw_home_probability >= PICK_SIDE_FLOOR else "AWAY"
    served_side = "HOME" if home_probability >= PICK_SIDE_FLOOR else "AWAY"
    if raw_side != served_side:
        mechanism = f"overlay flip (raw model picked {raw_side})"
    elif served_side != pick_side:
        mechanism = "frozen published pick"
    else:
        mechanism = "probability mapping centre is offset from the point estimate"
    margin_on_side = (
        predicted_margin > spread_line + TOLERANCE
        if pick_side == "HOME"
        else predicted_margin < spread_line - TOLERANCE
    )
    if not margin_on_side:
        violations.append(
            f"{game_id}: projected margin {predicted_margin:+.2f} is not strictly on the "
            f"{pick_side} side of {spread_line:+g}  [{mechanism}]"
        )
    else:
        mechanism = ""
    if served_pick_cover < PICK_SIDE_FLOOR - TOLERANCE:
        violations.append(
            f"{game_id}: served cover probability on the picked {pick_team} side is "
            f"{served_pick_cover:.4f}, below {PICK_SIDE_FLOOR:.0%}"
        )
    if displayed_value is None:
        violations.append(f"{game_id}: no displayed confidence on the card")
    elif displayed_value < PICK_SIDE_FLOOR - TOLERANCE:
        violations.append(
            f"{game_id}: displayed confidence {displayed_value:.4f} is below "
            f"{PICK_SIDE_FLOOR:.0%} on the picked {pick_team} side"
        )
    printed_score = str(row.get("Decision score") or "").strip()
    if printed_score and printed_score.endswith("%"):
        if float(printed_score.rstrip("%")) < PICK_SIDE_FLOOR * 100.0 - 1e-6:
            violations.append(
                f"{game_id}: the reader-facing Decision score {printed_score} is below "
                f"{PICK_SIDE_FLOOR:.0%} on the picked {pick_team} side"
            )
        expected_score = (
            f"{frozen.displayed_score:.1%}"
            if frozen is not None
            else (f"{displayed_value:.1%}" if displayed_value is not None else "")
        )
        if expected_score and printed_score != expected_score:
            source = "frozen published pick" if frozen is not None else "served displayed"
            violations.append(
                f"{game_id}: the reader-facing Decision score {printed_score} is not the "
                f"{source} confidence {expected_score}"
            )
    if frozen is not None and frozen.pick_team != pick_team:
        violations.append(
            f"{game_id}: the printed pick {pick_team} is not the frozen published pick "
            f"{frozen.pick_team}"
        )
    standalone_side = "HOME" if model_view.residual > 0.0 else "AWAY"
    if standalone_side != pick_side:
        violations.append(
            f"{game_id}: `nfl-ats tiebreaker` with no published row would build its guess for "
            f"the {standalone_side} side while the card picks {pick_side}"
        )
    if abs(served_total_value - (method_total + TOTAL_LOW_SIDE_SHADE_POINTS)) > 1e-6:
        violations.append(
            f"{game_id}: served total {served_total_value:.4f} is not the {method_name} total "
            f"{method_total:.4f} plus the {TOTAL_LOW_SIDE_SHADE_POINTS:+.1f} shade"
        )
    if cell_home is not None and cell_away is not None:
        support = {int(value) for value in lattice.scores}
        if cell_home not in support or cell_away not in support:
            violations.append(
                f"{game_id}: projected score {cell_home}-{cell_away} is not a cell of the "
                "lattice centred on the served numbers"
            )
        cell_margin_value = float(cell_home - cell_away)
        on_side = (
            cell_margin_value > spread_line
            if pick_side == "HOME"
            else cell_margin_value < spread_line
        )
        if not on_side:
            violations.append(
                f"{game_id}: projected score margin {cell_margin_value:+g} is not strictly on "
                f"the {pick_side} side of {spread_line:+g}"
            )
        if abs(cell_margin_value - predicted_margin) > _MAX_CENTRE_DISTANCE + TOLERANCE:
            violations.append(
                f"{game_id}: projected score margin {cell_margin_value:+g} is more than "
                f"{_MAX_CENTRE_DISTANCE:g} from the projected margin {predicted_margin:+.2f}"
            )
        cell_total_value = float(cell_home + cell_away)
        if abs(cell_total_value - served_total_value) > _MAX_CENTRE_DISTANCE + TOLERANCE:
            violations.append(
                f"{game_id}: projected score total {cell_total_value:g} is more than "
                f"{_MAX_CENTRE_DISTANCE:g} from the served total {served_total_value:.2f}"
            )
        if cell_tolerance is not None and (
            abs(cell_total_value - served_total_value) > cell_tolerance + TOLERANCE
        ):
            violations.append(
                f"{game_id}: projected score total {cell_total_value:g} is outside the "
                f"{cell_tolerance:g}-point total tolerance the lattice selection used"
            )

    return (
        GameAudit(
            game_id=game_id,
            home=home,
            away=away,
            pick_team=pick_team,
            pick_side=pick_side,
            printed_line=printed_line,
            spread_line=spread_line,
            home_cover_probability=home_probability,
            served_pick_cover=served_pick_cover,
            displayed=displayed_value,
            strength_word=strength_word,
            predicted_margin=predicted_margin,
            market_total=consensus.total_line,
            method_total=method_total,
            method_name=str(method_name),
            served_total_value=served_total_value,
            comparison_blend=comparison_blend,
            cell_home=cell_home,
            cell_away=cell_away,
            cell_total=cell_total,
            cell_margin=cell_margin,
            cell_tolerance=cell_tolerance,
            lattice_pick_cover=lattice_pick_cover,
            lattice_push=lattice_push,
            lattice_label=lattice_label,
            note=note,
            frozen=frozen is not None,
            raw_home_cover_probability=raw_home_probability,
            mechanism=mechanism,
            recorder_joint_arm=recorder_joint_arm,
        ),
        violations,
    )


def audit_published_tiebreaker(
    audits: list[GameAudit], forecast_dir: Path, tiebreaker_game_id: str
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    lines: list[str] = []
    path = forecast_dir / "tiebreaker.json"
    if not path.is_file():
        violations.append(f"no published tiebreaker.json beside {forecast_dir}")
        return violations, lines
    payload = json.loads(path.read_text(encoding="utf-8"))
    match = [audit for audit in audits if audit.game_id == tiebreaker_game_id]
    if not match:
        violations.append(f"tiebreaker game {tiebreaker_game_id} is not on the audited card")
        return violations, lines
    audit = match[0]
    published_total = _number(payload.get("projected_total"))
    published_home = _number(payload.get("guess_home"))
    published_away = _number(payload.get("guess_away"))
    published_margin = _number(payload.get("implied_margin"))
    published_served_total = _number(payload.get("served_total"))
    published_shade = payload.get("total_low_side_shade_points")
    published_cover = _number(payload.get("pick_cover_probability"))

    lines.append(f"published tiebreaker.json  {path}")
    lines.append(
        f"  published: {payload.get('home')} {published_home:g} - {payload.get('away')} "
        f"{published_away:g}, projected_total {published_total:g}, served_total "
        f"{published_served_total:.4f}, shade "
        f"{'ABSENT' if published_shade is None else f'{float(published_shade):+.1f}'}, "
        f"pick {payload.get('pick_side')} at {_number(payload.get('pick_spread_line')):+g}, "
        f"P(cover) {published_cover:.4f}"
    )
    served_cell = (
        f"{audit.cell_home:g} - {audit.cell_away:g}"
        if audit.cell_home is not None and audit.cell_away is not None
        else "none"
    )
    lines.append(
        f"  served now: {audit.home} {served_cell.split(' - ')[0]} - {audit.away} "
        f"{served_cell.split(' - ')[-1]}, projected_total "
        f"{audit.cell_total if audit.cell_total is not None else float('nan')}, served_total "
        f"{audit.served_total_value:.4f}, shade {TOTAL_LOW_SIDE_SHADE_POINTS:+.1f}, "
        f"pick {audit.pick_side} at {audit.spread_line:+g}, P(cover) "
        f"{audit.lattice_pick_cover if audit.lattice_pick_cover is not None else float('nan'):.4f}"
    )

    if abs(published_total - (published_home + published_away)) > TOLERANCE:
        violations.append(
            f"{tiebreaker_game_id}: published projected_total {published_total:g} is not "
            f"guess_home + guess_away ({published_home + published_away:g})"
        )
    if abs(published_margin - (published_home - published_away)) > TOLERANCE:
        violations.append(
            f"{tiebreaker_game_id}: published implied_margin {published_margin:g} is not "
            f"guess_home - guess_away ({published_home - published_away:g})"
        )
    if published_shade is None:
        violations.append(
            f"{tiebreaker_game_id}: published tiebreaker.json carries no "
            "total_low_side_shade_points -- it predates the shade wiring, so its served total "
            "is not the total the code serves now"
        )
    elif abs(float(published_shade) - TOTAL_LOW_SIDE_SHADE_POINTS) > TOLERANCE:
        violations.append(
            f"{tiebreaker_game_id}: published shade {float(published_shade):+.1f} is not the "
            f"served {TOTAL_LOW_SIDE_SHADE_POINTS:+.1f}"
        )
    if abs(published_served_total - audit.served_total_value) > 1e-6:
        violations.append(
            f"{tiebreaker_game_id}: published served_total {published_served_total:.4f} differs "
            f"from the total the code serves now ({audit.served_total_value:.4f})"
        )
    if audit.cell_home is not None and (
        int(published_home) != audit.cell_home or int(published_away) != audit.cell_away
    ):
        violations.append(
            f"{tiebreaker_game_id}: published score {published_home:g}-{published_away:g} "
            f"differs from the lattice cell the code serves now "
            f"{audit.cell_home}-{audit.cell_away}"
        )
    published_pick_cover_side = published_cover >= PICK_SIDE_FLOOR
    if published_cover == published_cover and not published_pick_cover_side:
        violations.append(
            f"{tiebreaker_game_id}: published pick_cover_probability {published_cover:.4f} is "
            "below 50% on the side the card picks"
        )
    lines.append(
        f"  card P(cover) on the picked side {audit.served_pick_cover:.4f} vs lattice "
        f"{audit.lattice_pick_cover if audit.lattice_pick_cover is not None else float('nan'):.4f}"
        f"  (gap {abs(audit.served_pick_cover - (audit.lattice_pick_cover or float('nan'))):.4f})"
    )
    if audit.lattice_pick_cover is not None and (
        (audit.served_pick_cover >= PICK_SIDE_FLOOR)
        != (audit.lattice_pick_cover >= PICK_SIDE_FLOOR)
    ):
        violations.append(
            f"{tiebreaker_game_id}: the card's cover probability and the lattice's disagree on "
            f"the SIDE ({audit.served_pick_cover:.4f} vs {audit.lattice_pick_cover:.4f})"
        )
    return violations, lines


def audit_prospective_arms(
    artifacts_root: Path, tiebreaker: GameAudit | None
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    lines: list[str] = []
    if tiebreaker is not None:
        recorded_joint = tiebreaker.recorder_joint_arm
        recorded_blend = tiebreaker.comparison_blend
        lines.append(
            "next recording would write: blend_k01 "
            f"{recorded_blend:.4f} (market{recorded_blend - tiebreaker.market_total:+.4f}), "
            f"joint_residual {recorded_joint:.4f} "
            f"(market{recorded_joint - tiebreaker.market_total:+.4f})"
        )
        if abs(recorded_joint - tiebreaker.method_total) > 1e-6:
            violations.append(
                f"the joint_residual arm the recorder would write ({recorded_joint:.4f}) is not "
                f"the unshaded method total ({tiebreaker.method_total:.4f}), while the blend arm "
                f"({recorded_blend:.4f}) is unshaded -- the two arms would not be commensurable"
            )
    path = totals_method_ledger_path(artifacts_root)
    if not path.is_file():
        lines.append("totals_served_method ledger: absent")
        return violations, lines
    ledger = pd.read_parquet(path)
    lines.append(f"totals_served_method ledger: {len(ledger)} row(s)  {path}")
    for _, row in ledger.iterrows():
        blend = _number(row["served_total_blend_k01"])
        joint = _number(row["served_total_joint_residual"])
        market = _number(row["market_total"])
        blend_shade = blend - market
        joint_shade = joint - market
        lines.append(
            f"  {row['game_id']} week {int(row['week'])}: market {market:g}, blend_k01 "
            f"{blend:.4f} (market{blend_shade:+.4f}), joint_residual {joint:.4f} "
            f"(market{joint_shade:+.4f}), method {row['served_total_method']}"
        )
        mismatched = (blend_shade <= -0.9) != (joint_shade <= -0.9)
        if np.isfinite(blend) and np.isfinite(joint) and mismatched:
            violations.append(
                f"{row['game_id']}: the two recorded totals arms are not on the same shade "
                f"scale (blend {blend:.4f}, joint {joint:.4f}, market {market:g}) -- a "
                "paired comparison of incommensurable numbers"
            )
    return violations, lines


def format_table(audits: list[GameAudit]) -> str:
    header = (
        f"{'game':>16} {'pick':>5} {'line':>6} {'P(cov)':>7} {'shown':>7} {'word':>7} "
        f"{'margin':>7} {'mkt tot':>8} {'method':>8} {'served':>8} {'cell':>9} "
        f"{'cm':>4} {'ct':>4} {'tol':>4} {'latP':>6}"
    )
    rows = [header, "-" * len(header)]
    for audit in audits:
        cell = f"{audit.cell_home}-{audit.cell_away}" if audit.cell_home is not None else "-"
        lattice_probability = (
            audit.lattice_pick_cover if audit.lattice_pick_cover is not None else float("nan")
        )
        rows.append(
            f"{audit.game_id.replace('2026_01_', ''):>16} "
            f"{audit.pick_team:>5} "
            f"{audit.printed_line:>+6g} "
            f"{audit.served_pick_cover:>7.4f} "
            f"{audit.displayed if audit.displayed is not None else float('nan'):>7.4f} "
            f"{audit.strength_word or '-':>7} "
            f"{audit.predicted_margin:>+7.2f} "
            f"{audit.market_total:>8.1f} "
            f"{audit.method_total:>8.3f} "
            f"{audit.served_total_value:>8.3f} "
            f"{cell:>9} "
            f"{audit.cell_margin if audit.cell_margin is not None else float('nan'):>+4.0f} "
            f"{audit.cell_total if audit.cell_total is not None else float('nan'):>4.0f} "
            f"{audit.cell_tolerance if audit.cell_tolerance is not None else float('nan'):>4.0f} "
            f"{lattice_probability:>6.3f}"
        )
    return "\n".join(rows)


def run(artifacts_root: Path, data_root: Path) -> int:
    (
        active,
        metadata,
        card,
        nomination,
        _overlay,
        _arrest_overlay,
        _composition,
        served,
        displayed_confidence,
    ) = _publication_context(artifacts_root, data_root, require_fresh_arrest_overlay=False)
    season, week = int(metadata["season"]), int(metadata["week"])
    schedules = pd.read_parquet(newest_schedules_path(data_root))
    finals = lined_finals(schedules)
    schedule_rows = schedules.set_index(schedules["game_id"].astype(str))

    printed = card.set_index(card["Matchup"].astype(str))
    merged = served.copy()
    merged["Matchup"] = merged["away_team"].astype(str) + " at " + merged["home_team"].astype(str)
    merged["ATS prediction"] = merged["Matchup"].map(printed["ATS prediction"])
    merged["Decision score"] = merged["Matchup"].map(printed["Decision score"])
    forecast_dir = artifacts_root / str(active.get("weekly_forecast", {}).get("artifact", ""))
    raw_probability = (
        pd.read_csv(forecast_dir / "recommendations.csv")
        .drop_duplicates("game_id")
        .set_index("game_id")["home_cover_probability"]
    )
    frozen = frozen_picks(
        artifacts_root, now=datetime.now(UTC), season=season, week=week, include_open=False
    )

    print(f"MOD-17 served-number audit -- {season} week {week}")
    print(f"active model {active.get('model_id')}  method {active.get('method')}")
    print(f"forecast {active.get('weekly_forecast', {}).get('artifact')}")
    print(
        f"served total method {SERVED_TOTAL_METHOD}, low-side shade "
        f"{TOTAL_LOW_SIDE_SHADE_POINTS:+.1f}, displayed confidence policy "
        f"{displayed_confidence.policy} (served={displayed_confidence.served})"
    )
    print(f"best pick {nomination.active_game_id}")
    print()

    audits: list[GameAudit] = []
    violations: list[str] = []
    for _, row in merged.iterrows():
        game_id = str(row["game_id"])
        if game_id not in schedule_rows.index:
            violations.append(f"{game_id}: not present in the newest schedules")
            continue
        game = schedule_rows.loc[game_id]
        if isinstance(game, pd.DataFrame):
            game = game.iloc[0]
        audit, game_violations = audit_game(
            row,
            game,
            finals,
            data_root,
            frozen.get(game_id),
            _number(raw_probability.get(game_id)),
        )
        audits.append(audit)
        violations.extend(game_violations)

    print(format_table(audits))
    print()
    for audit in audits:
        flags = " FROZEN" if audit.frozen else ""
        print(f"{audit.game_id}{flags}: lattice {audit.lattice_label}; {audit.note}")
    print()

    week_games = schedules.loc[
        schedules["season"].eq(season)
        & schedules["week"].eq(week)
        & schedules["game_type"].astype(str).eq("REG")
    ]
    keys = week_games["gameday"].astype(str)
    if "gametime" in week_games.columns:
        keys = keys + " " + week_games["gametime"].astype(str).fillna("")
    tiebreaker_game_id = str(week_games.loc[keys.sort_values().index[-1]]["game_id"])

    published_violations, published_lines = audit_published_tiebreaker(
        audits, forecast_dir, tiebreaker_game_id
    )
    for line in published_lines:
        print(line)
    violations.extend(published_violations)
    print()

    tiebreaker_audit = next(
        (audit for audit in audits if audit.game_id == tiebreaker_game_id), None
    )
    ledger_violations, ledger_lines = audit_prospective_arms(artifacts_root, tiebreaker_audit)
    for line in ledger_lines:
        print(line)
    violations.extend(ledger_violations)
    print()

    if violations:
        print(f"VIOLATIONS ({len(violations)}):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("no violations")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every served number on the current week's card for the MOD-17 "
            "one-lattice / one-margin / one-total invariants."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(os.environ.get("NFL_ATS_ARTIFACTS_DIR", "artifacts")),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("NFL_ATS_DATA_DIR", "data")),
    )
    args = parser.parse_args(argv)
    return run(args.artifacts_root, args.data_root)


if __name__ == "__main__":
    sys.exit(main())
