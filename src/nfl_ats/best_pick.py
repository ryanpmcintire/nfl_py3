"""Best Pick selection: the one confirmed pool-play lever (POL-09).

The pool pays one "Best Pick" per regular-season week. Our residual magnitude
does not rank pick quality -- the weekly top-|residual| pick scored 48.6% over
107 weeks -- so choosing one arbitrarily costs nothing and choosing one well is
free points. ``sweep_robustness`` is the signal that survived SPEC-5:

* screened on the registry window [2013, 2015] against the nflverse spread:
  top-1 accuracy +5.57 points over all picks, ``probability_positive`` 0.7955;
* confirmed on [2020, 2021] against the Tuesday opener the pool grades on:
  top-1 60.0% vs 51.32%, +8.68 points, ``probability_positive`` 0.865.

Both windows are permanently spent (``registry/rotation_registry.json``) and
the write-up is ``docs/best_pick_ranker.md``. Two disjoint windows, two grades,
same direction -- but 86 top-1 picks in total and a Kendall tau of +0.067
(p = 0.099), so this is a lever worth pulling, not an established effect.

**The definition below is frozen.** It is the exact function that was scored;
``scripts/best_pick_ranker.py`` imports it rather than keeping its own copy, so
the signal we deploy each Tuesday cannot drift from the signal that was
confirmed. Changing the grid, the 0.50 threshold, or the anchoring rule makes
this a different, unconfirmed signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The confirmed signal was measured on the full sweep grid that
# ``margin-predict`` writes (+/-4 points in 0.5 steps). The public and internal
# pick pages narrow the sweep for PLOTTING; the ranking must not see that
# narrowed frame, or it silently censors a signal that is already censored at
# the grid edge (see docs/best_pick_ranker.md, "Known limitation of signal 3").
SWEEP_PROBABILITY_FLOOR = 0.50


def sweep_robustness(sweep: pd.DataFrame, picks: pd.DataFrame) -> pd.Series:
    """Width in points of the contiguous interval around the quote where the
    pick's cover probability stays >= 0.50.

    ``line_sweep`` reports ``home_cover_probability`` at each alternative line;
    the PICK side is fixed at the quote, so the pick's probability is that value
    for HOME picks and its complement for AWAY picks. The run is anchored at
    offset 0 (where the condition holds by construction) and extended over
    contiguous 0.5-point grid steps; the width is max(alt) - min(alt) of the run.

    ``picks`` needs ``game_id`` and a ``pick`` column of ``"HOME"``/``"AWAY"``.

    ``sweep`` must contain ONE row per (game_id, line_offset). ``line_sweep``
    artifacts hold every method side by side, so a caller that forgets to filter
    to the active method hands this function three interleaved curves per game;
    the contiguity walk would then read a neighbour method's probability as the
    next half-point and return a silently wrong width. That is checked, not
    trusted.
    """

    side = picks.set_index("game_id")["pick"].to_dict()
    widths: dict[str, float] = {}
    for raw_game_id, group in sweep.groupby("game_id", sort=False):
        game_id = str(raw_game_id)
        pick = side.get(game_id)
        if pick is None:
            continue
        if group["line_offset"].duplicated().any():
            raise ValueError(
                f"sweep for {game_id} has repeated line offsets; filter it to one method"
            )
        ordered = group.sort_values("line_offset")
        probability = ordered["home_cover_probability"].to_numpy(dtype=float)
        if pick == "AWAY":
            probability = 1.0 - probability
        offsets = ordered["line_offset"].to_numpy(dtype=float)
        holds = probability >= SWEEP_PROBABILITY_FLOOR
        anchor = int(np.argmin(np.abs(offsets)))
        if not holds[anchor]:
            widths[game_id] = 0.0
            continue
        low = anchor
        while low - 1 >= 0 and holds[low - 1]:
            low -= 1
        high = anchor
        while high + 1 < len(holds) and holds[high + 1]:
            high += 1
        widths[game_id] = float(offsets[high] - offsets[low])
    return pd.Series(widths, name="sweep_robustness")


def _pick_sides(predictions: pd.DataFrame) -> pd.DataFrame:
    picks = predictions[["game_id"]].copy()
    picks["game_id"] = picks["game_id"].astype(str)
    picks["pick"] = np.where(
        pd.to_numeric(predictions["home_cover_probability"], errors="coerce") >= 0.5,
        "HOME",
        "AWAY",
    )
    return picks


def best_pick_scores(predictions: pd.DataFrame, sweep: pd.DataFrame) -> pd.Series:
    """``sweep_robustness`` per game for one week's card, indexed by game_id.

    Returns an empty series when the sweep artifact is missing or malformed, so
    every caller degrades to "no Best Pick marked" rather than failing a page.
    """

    required = {"game_id", "line_offset", "home_cover_probability"}
    if predictions.empty or sweep.empty or not required.issubset(sweep.columns):
        return pd.Series(dtype=float, name="sweep_robustness")
    if "home_cover_probability" not in predictions.columns:
        return pd.Series(dtype=float, name="sweep_robustness")
    work = sweep.copy()
    work["game_id"] = work["game_id"].astype(str)
    # An unfiltered multi-method artifact is malformed for this purpose. Both
    # production callers filter, but a page with no Best Pick badge is a visible
    # failure while a wrong badge is an invisible one, so degrade rather than
    # rank a mixture of three models' curves.
    if work.duplicated(subset=["game_id", "line_offset"]).any():
        return pd.Series(dtype=float, name="sweep_robustness")
    return sweep_robustness(work, _pick_sides(predictions))


def select_best_pick(predictions: pd.DataFrame, sweep: pd.DataFrame) -> str | None:
    """The game_id whose forced pick is the week's Best Pick, or ``None``.

    Ties break on game_id so the choice is reproducible: two runs of the same
    card must never nominate different games.
    """

    scores = best_pick_scores(predictions, sweep)
    if scores.empty:
        return None
    ranked = scores.reset_index()
    ranked.columns = ["game_id", "sweep_robustness"]
    ranked = ranked.dropna(subset=["sweep_robustness"])
    if ranked.empty:
        return None
    ranked = ranked.sort_values(["sweep_robustness", "game_id"], ascending=[False, True])
    return str(ranked.iloc[0]["game_id"])


def best_pick_tie_count(predictions: pd.DataFrame, sweep: pd.DataFrame) -> int:
    """How many games tie for the top ``sweep_robustness`` score this week.

    ``0`` means there is no Best Pick at all (empty or malformed sweep -- see
    ``best_pick_scores``). ``1`` means the nomination is an unambiguous top
    score. Anything higher means ``select_best_pick``'s alphabetical
    ``game_id`` tie-break decided arbitrarily among that many games: the
    signal is censored at ``SWEEP_PROBABILITY_FLOOR``'s grid edge (8.0
    points), so weeks routinely tie there, and removing that tie-break luck
    cuts the recorded Best Pick edge from +8.68 to +0.92
    (``docs/pool_format_levers.md``).
    """

    scores = best_pick_scores(predictions, sweep)
    if scores.empty:
        return 0
    return int((scores == scores.max()).sum())


def best_pick_tie_note(predictions: pd.DataFrame, sweep: pd.DataFrame) -> str:
    """A plain-language disclosure sentence when the Best Pick is a tie.

    Returns ``""`` when the nomination is unambiguous (or there is none), so
    every surface that shows a Best Pick -- the dashboard and the published
    card -- can share the exact same wording and the exact same definition of
    "tied" rather than risking the two silently drifting apart.
    """

    tied = best_pick_tie_count(predictions, sweep)
    if tied <= 1:
        return ""
    return (
        f"This week {tied} games tie at the top of that signal, so choosing "
        "between them is arbitrary -- reproducible, but not a lean."
    )
