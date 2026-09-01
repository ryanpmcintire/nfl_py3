"""Leakage regression + unit tests for the FluView home-market illness battery
(``docs/fluview_battery.md``): proves a revision issued AFTER a game's
decision-cutoff Tuesday can never reach that game's AS-OF ``ili`` feature,
that an upstream all-null-``release_date`` region (the measured ``ny`` gap,
section 1 of the doc) resolves to missing rather than a leaked value, and
that "missing" is never silently defaulted to "not elevated" in a way that
would leave a leakage-tainted row inside a scored population.

Modeled on ``tests/test_respiratory_battery_leakage.py``'s canary pattern
(itself modeled on ``tests/test_gdelt_attention_screen.py``): load the real
script module (not a reimplementation) and feed it a synthetic fixture with
a deliberately extreme "future" value that must not leak.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("fluview_battery_screen_test", "fluview_battery_screen.py")


# ---------------------------------------------------------------------------
# cdc_epiweek: validated against real Delphi rows (docs/fluview_battery.md
# section 3's own worked example), not hand-picked week numbers.
# ---------------------------------------------------------------------------


def test_cdc_epiweek_matches_delphi_validated_example() -> None:
    """``docs/fluview_battery.md`` / the module docstring cites a live-checked
    row: ``ca`` epiweek 201840 has ``release_date`` 2018-10-12 (a Friday), 6
    days after the computed week-end of Saturday 2018-10-06. The Saturday
    must resolve to 201840 and the very next day (Sunday) must already be
    the following epiweek 201841 -- Sunday-start weeks, not ISO Monday-start."""

    assert screen.cdc_epiweek(pd.Timestamp("2018-10-06")) == 201840
    assert screen.cdc_epiweek(pd.Timestamp("2018-10-07")) == 201841
    assert screen.cdc_epiweek(pd.Timestamp("2018-10-12")) == 201841


def test_cdc_epiweek_handles_53_week_years() -> None:
    """2020 is a 53-CDC-week year; the last day of the year (and the first
    two days of the FOLLOWING calendar year, since that week has fewer than
    4 days in 2021) must land in week 53 of 2020, not silently overflow into
    week 1 of 2021 (a plausible off-by-one if the year-boundary search only
    checked +/-0 years). Measured this session against the module's own
    implementation."""

    assert screen.cdc_epiweek(pd.Timestamp("2020-12-31")) == 202053
    assert screen.cdc_epiweek(pd.Timestamp("2021-01-01")) == 202053
    # The ordinary (non-53-week) year boundary: Sat 2019-12-28 is the last
    # day of week 52 of 2019; the very next day, Sun 2019-12-29, already
    # starts week 1 of 2020 (that week has >=4 days in the new year).
    assert screen.cdc_epiweek(pd.Timestamp("2019-12-28")) == 201952
    assert screen.cdc_epiweek(pd.Timestamp("2019-12-29")) == 202001


# ---------------------------------------------------------------------------
# Point-in-time-safety canary: a later-arriving revision must never be
# visible to a decision cutoff strictly before its own release_date.
# ---------------------------------------------------------------------------


def test_checkpoint_table_never_lets_a_later_revision_leak_before_its_release_date() -> None:
    """Canary: epiweek 201801's value is revised from 1.0 to 99.0 (an extreme,
    unmistakable value) by a much later issue. A decision cutoff strictly
    before that later issue's release date must see the OLD value, never
    99.0; a cutoff on/after that release date must see the revision. This is
    the exact property the doc's section 3 as-of algorithm claims."""

    raw = pd.DataFrame(
        {
            "region": ["zz", "zz"],
            "epiweek": [201801, 201801],
            "issue": [201805, 201820],
            "lag": [4, 19],
            "release_date": pd.to_datetime(["2018-02-02", "2018-05-18"]),
            "ili": [1.0, 99.0],
        }
    )
    checkpoints = screen.build_checkpoint_tables(raw)
    checkpoint = checkpoints["zz"]

    release_late = pd.Timestamp("2018-05-18")

    cutoffs = pd.Series([release_late - pd.Timedelta(days=1), release_late])
    looked_up = screen.asof_lookup(checkpoint, cutoffs)

    assert looked_up.loc[0, "known_ili"] == 1.0, "revision leaked BEFORE its release date"
    assert looked_up.loc[1, "known_ili"] == 99.0, "revision not visible on its own release date"


def test_checkpoint_table_carries_forward_newest_known_epiweek_not_latest_revision() -> None:
    """A stale re-issue of an OLD epiweek arriving late must not override a
    NEWER epiweek already known as of that release_date -- the running-max
    construction (section 3) must key on ``epiweek``, not on release order
    alone."""

    raw = pd.DataFrame(
        {
            "region": ["zz", "zz", "zz"],
            "epiweek": [201801, 201802, 201801],
            "issue": [201801, 201802, 201810],
            "lag": [0, 0, 8],
            "release_date": pd.to_datetime(["2018-01-06", "2018-01-13", "2018-03-01"]),
            "ili": [1.0, 2.0, 1.5],
        }
    )
    checkpoints = screen.build_checkpoint_tables(raw)
    checkpoint = checkpoints["zz"]

    # As of 2018-03-01, the freshest KNOWN epiweek is still 201802 (week 2's
    # ili=2.0) even though a late, stale re-issue of week 1 (ili=1.5) arrived
    # more recently -- a late revision of an already-superseded old week must
    # not overwrite the newer week's own as-of reading.
    looked_up = screen.asof_lookup(checkpoint, pd.Series([pd.Timestamp("2018-03-01")]))
    assert looked_up.loc[0, "known_epiweek"] == 201802
    assert looked_up.loc[0, "known_ili"] == 2.0


def test_checkpoint_table_drops_rows_with_null_release_date_measured_ny_gap() -> None:
    """Measured this session (docs/fluview_battery.md section 1): Delphi
    returns ``release_date: null`` on EVERY row for the ``ny`` region. Because
    the as-of algorithm keys entirely on ``release_date``, such a region must
    resolve to an EMPTY checkpoint table (zero point-in-time information),
    never fall back to using ``issue`` or ``epiweek`` as a release-date
    substitute, and never leak a final/most-recent value."""

    raw = pd.DataFrame(
        {
            "region": ["ny", "ny", "zz"],
            "epiweek": [201801, 201802, 201801],
            "issue": [201801, 201802, 201801],
            "lag": [0, 0, 0],
            "release_date": [pd.NaT, pd.NaT, pd.Timestamp("2018-01-06")],
            "ili": [5.0, 6.0, 1.0],
        }
    )
    checkpoints = screen.build_checkpoint_tables(raw)

    assert checkpoints["ny"].empty
    assert list(checkpoints["ny"].columns) == ["release_date", "known_epiweek", "known_ili"]

    # Routed through the actual production entry point (``attach_asof_ili``),
    # which is what guards an empty/all-null-release_date checkpoint from
    # ever reaching ``asof_lookup``'s ``merge_asof`` at all -- calling
    # ``merge_asof`` directly against a genuinely empty right frame is not
    # itself a safe operation (a dtype-only empty frame can raise), so the
    # production code must short-circuit before that point, not rely on
    # ``merge_asof`` to fail safe.
    games = pd.DataFrame(
        {
            "home_state": ["ny"],
            "away_state": ["zz"],
            "cutoff_date": [pd.Timestamp("2026-01-01")],  # far-future cutoff
        }
    )
    scored = screen.attach_asof_ili(games, checkpoints)
    assert pd.isna(scored.loc[0, "home_ili"]), (
        "an all-null-release_date region must resolve to missing, never a leaked value, "
        "even at a cutoff far in the future"
    )


def test_attach_asof_ili_end_to_end_never_leaks_a_future_revision() -> None:
    """End-to-end production entry point (``attach_asof_ili``, what
    ``load_schedules``'s output is actually merged against): a game whose
    decision cutoff is strictly before a late revision's release_date must
    see the OLD as-of value on both sides, never the revised one."""

    raw = pd.DataFrame(
        {
            "region": ["az", "az", "ca", "ca"],
            "epiweek": [201801, 201801, 201801, 201801],
            "issue": [201802, 201820, 201802, 201820],
            "lag": [1, 19, 1, 19],
            "release_date": pd.to_datetime(
                ["2018-01-13", "2018-05-18", "2018-01-13", "2018-05-18"]
            ),
            "ili": [1.0, 99.0, 2.0, 88.0],
        }
    )
    checkpoints = screen.build_checkpoint_tables(raw)

    games = pd.DataFrame(
        {
            "home_state": ["az"],
            "away_state": ["ca"],
            "cutoff_date": [pd.Timestamp("2018-05-17")],  # one day before the late revision
        }
    )
    scored = screen.attach_asof_ili(games, checkpoints)
    assert scored.loc[0, "home_ili"] == 1.0, "future home revision (99.0) leaked before release"
    assert scored.loc[0, "away_ili"] == 2.0, "future away revision (88.0) leaked before release"

    games_after = games.copy()
    games_after["cutoff_date"] = pd.Timestamp("2018-05-18")
    scored_after = screen.attach_asof_ili(games_after, checkpoints)
    assert scored_after.loc[0, "home_ili"] == 99.0
    assert scored_after.loc[0, "away_ili"] == 88.0


# ---------------------------------------------------------------------------
# Missing-data handling: a missing AS-OF value must poison exclusion, never
# be silently defaulted to "not elevated" and left inside a scored subset.
# ---------------------------------------------------------------------------


def test_attach_elevated_flags_marks_missing_rather_than_defaulting_silently() -> None:
    """docs/fluview_battery.md section 3: rows with a missing AS-OF value (or
    a state with no frozen threshold, e.g. below the >=10-observation floor)
    must be flagged ``*_missing=True`` so ``build_cells`` can exclude them --
    ``elevated`` itself is allowed to read False for a missing row, but ONLY
    because the missing flag independently gates population membership, not
    because missing was treated as evidence of non-elevation."""

    df = pd.DataFrame(
        {
            "home_ili": [5.0, np.nan, 3.0],
            "away_ili": [1.0, 2.0, np.nan],
            "home_state": ["az", "az", "ca"],
            "away_state": ["ca", "ca", "az"],
        }
    )
    thresholds = {"az": 4.0, "ca": 4.0}
    scored = screen.attach_elevated_flags(df, thresholds)

    assert list(scored["home_missing"]) == [False, True, False]
    assert list(scored["away_missing"]) == [False, False, True]
    assert list(scored["home_elevated"]) == [True, False, False]
    assert list(scored["away_elevated"]) == [False, False, False]

    # A state with no frozen threshold at all (below the 10-observation
    # floor, section 3) must also count as missing, not as "never elevated".
    df2 = pd.DataFrame(
        {"home_ili": [10.0], "away_ili": [10.0], "home_state": ["zz"], "away_state": ["az"]}
    )
    scored2 = screen.attach_elevated_flags(df2, {"az": 4.0})
    assert bool(scored2.loc[0, "home_missing"]) is True
    assert bool(scored2.loc[0, "home_elevated"]) is False


def test_compute_state_thresholds_ignores_missing_values_and_requires_floor() -> None:
    """Section 3: the per-state top-decile threshold is computed on the
    state's own AS-OF panel with missing values dropped first, and requires
    at least 10 non-missing observations (a state with fewer must have NO
    threshold at all, which ``attach_elevated_flags`` then treats as
    missing, not as "never elevated")."""

    plenty = pd.DataFrame({"state": ["az"] * 12, "ili": [*range(10), np.nan, np.nan]})
    few = pd.DataFrame({"state": ["zz"] * 5, "ili": [1.0, 2.0, 3.0, 4.0, 5.0]})
    panel = pd.concat([plenty, few], ignore_index=True)

    thresholds = screen.compute_state_thresholds(panel)
    assert "az" in thresholds
    assert "zz" not in thresholds, "a state below the 10-observation floor must get no threshold"
    # The threshold must be computed on the 10 non-missing values only
    # (dropna before quantile), not on all 12 rows including the two NaNs.
    assert thresholds["az"] == pd.Series(range(10)).quantile(0.90)


def test_build_state_week_panel_dedupes_shared_state_across_home_and_away() -> None:
    """Two-team states (section 2, e.g. CA hosting LA/LAC/SF) must contribute
    exactly one panel row per (state, season, week), not one per game, since
    both franchises' home games in the same week share the identical
    state-level AS-OF reading."""

    df = pd.DataFrame(
        {
            "home_state": ["ca", "ca"],
            "away_state": ["az", "nv"],
            "season": [2019, 2019],
            "week": [3, 3],
            "cutoff_date": [pd.Timestamp("2019-09-17")] * 2,
            "home_ili": [4.0, 4.0],
            "away_ili": [1.0, 2.0],
        }
    )
    panel = screen.build_state_week_panel(df)
    ca_rows = panel.loc[panel["state"] == "ca"]
    assert len(ca_rows) == 1, "duplicate (state, season, week) rows were not collapsed"
