"""Construction, sign-convention, franchise-normalisation and leakage
contracts for the two LEAD-20/LEAD-25 quarterback-identity flags, plus the
on-production confirmation wrapper's duck-typed reuse of
``scripts/on_production_opener_confirmation.py``.

Predeclared in ``docs/schedule_flag_battery.md`` "Wave 5". Every fixture is
built in memory: these tests must pass in a fresh clone with no local data
snapshots (no schedules/weekly_rosters/combine snapshot is ever read).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import qb_identity_on_production as qiop  # noqa: E402

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402
from nfl_ats.qb_identity_features import (  # noqa: E402
    QB_REVENGE_COLUMN,
    ROOKIE_QB_DEBUT_FADE_COLUMN,
    _canonical_schedule_team,
    attach_qb_revenge_features,
    attach_rookie_qb_debut_fade_features,
    derive_qb_revenge_features,
    derive_rookie_qb_debut_fade_features,
    describe_rookie_qb_debut_population,
    draft_team_by_gsis_id,
    qb_revenge_join_diagnostics,
)


def _game(
    game_id: str,
    season: int,
    gameday: str,
    home: str,
    away: str,
    home_qb: str | None,
    away_qb: str | None,
    game_type: str = "REG",
) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "gameday": gameday,
        "game_type": game_type,
        "home_team": home,
        "away_team": away,
        "home_qb_id": home_qb,
        "away_qb_id": away_qb,
    }


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _rosters(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """(season, gsis_id, years_exp) rows -- the only columns
    ``_season_years_exp`` reads."""

    return pd.DataFrame(rows, columns=["season", "gsis_id", "years_exp"])


# ---------------------------------------------------------------------------
# LEAD-20: rookie-QB debut fade
# ---------------------------------------------------------------------------


def _debut_schedule() -> pd.DataFrame:
    return _schedule(
        [
            # H1 already has an earlier start (h_prior); R1 is making his very
            # first archived start, as AWAY, and is a rookie (years_exp 0) ->
            # away debut rookie, home is not first -> +1.
            _game("h_prior", 2020, "2020-09-06", "AAA", "ZZZ", "H1", "ZQ"),
            _game("g1", 2020, "2020-09-10", "AAA", "BBB", "H1", "R1"),
            # Mirror: R2 making his very first archived start, as HOME, rookie
            # -> home debut rookie, away (H1) already started -> -1.
            _game("g2", 2020, "2020-09-13", "CCC", "AAA", "R2", "H1"),
            # V1's very first ARCHIVED start (this tiny fixture's own
            # left-censoring), but years_exp is 5 that season -- an
            # established veteran, NOT a debut.
            _game("g_vet", 2020, "2020-09-06", "DDD", "EEE", "V1", "ZQ2"),
            # R1's SECOND start (still years_exp 0 this season) -- must NOT be
            # flagged a second time; only the FIRST start counts.
            _game("g3", 2020, "2020-09-20", "FFF", "BBB", "H1", "R1"),
            # U1's very first archived start has NO weekly_rosters row at all
            # for (season, U1) -- unresolved years_exp, never guessed. Dated
            # after h_prior so H1's own first-archived start is unambiguously
            # h_prior, not this game.
            _game("g_unresolved", 2020, "2020-09-08", "GGG", "HHH", "H1", "U1"),
            # Both sides debut as rookies simultaneously -> 0.
            _game("g_both", 2021, "2021-09-12", "III", "JJJ", "R3", "R4"),
            # A team's only archived appearance is POSTSEASON -- never enters
            # the REG-only debut population, so this game (and its own
            # "first-looking" starter) must read 0.
            _game("g_post", 2020, "2021-01-10", "AAA", "KKK", "H1", "PP1", game_type="WC"),
        ]
    )


def _debut_rosters() -> pd.DataFrame:
    return _rosters(
        [
            (2020, "H1", 6.0),
            (2020, "ZQ", 3.0),
            (2020, "R1", 0.0),
            (2020, "R2", 0.0),
            (2020, "V1", 5.0),
            (2020, "ZQ2", 4.0),
            (2021, "R3", 0.0),
            (2021, "R4", 0.0),
            # U1 deliberately has NO row -- unresolved years_exp.
            # PP1 deliberately has NO row either (postseason-only starter).
        ]
    )


def test_debut_rookie_sign_convention_away_is_positive() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g1", ROOKIE_QB_DEBUT_FADE_COLUMN] == 1.0


def test_debut_rookie_sign_convention_home_is_negative() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g2", ROOKIE_QB_DEBUT_FADE_COLUMN] == -1.0


def test_veteran_whose_first_archived_start_is_not_a_debut() -> None:
    """A player's first-archived start is not automatically a debut: the
    rookie gate (``years_exp == 0``) must exclude an established veteran
    whose true career debut predates the archive (the entire reason this
    gate exists, per the LEAD-20 predeclaration)."""

    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_vet", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_second_start_is_never_flagged_as_a_debut() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g3", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_unresolved_years_exp_is_never_flagged_a_debut() -> None:
    """A first-archived start that cannot be joined to weekly_rosters is
    treated as NOT a debut -- never guessed."""

    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_unresolved", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_both_sides_debuting_simultaneously_is_zero() -> None:
    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_both", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_postseason_game_is_never_flagged() -> None:
    """A debut is only ever defined against a REG start; a postseason game
    (even one with a starter who has never had a REG start in the archive)
    must read 0."""

    derived = derive_rookie_qb_debut_fade_features(_debut_schedule(), _debut_rosters()).set_index(
        "game_id"
    )
    assert derived.loc["g_post", ROOKIE_QB_DEBUT_FADE_COLUMN] == 0.0


def test_describe_rookie_qb_debut_population_diagnostic() -> None:
    diagnostic = describe_rookie_qb_debut_population(_debut_schedule(), _debut_rosters())
    # First-archived REG starts: H1 (g_prior), ZQ (g_prior), R1 (g1), R2 (g2),
    # V1 (g_vet), ZQ2 (g_vet), U1 (g_unresolved), R3 (g_both), R4 (g_both) = 9.
    # (g3 is R1's SECOND start, not counted again; g_post is non-REG.)
    assert diagnostic["n_first_archived_reg_starts"] == 9
    assert diagnostic["n_confirmed_rookie_debuts"] == 4  # R1, R2, R3, R4
    assert diagnostic["n_confirmed_non_rookie_first_starts"] == 4  # H1, ZQ, V1, ZQ2
    assert diagnostic["n_unresolved_years_exp"] == 1  # U1


def test_rookie_debut_leakage_ignores_unrelated_outcome_columns() -> None:
    """Mutating an unrelated outcome-shaped column (e.g. a result/score field
    that no LEAD-20 code path ever reads) must never change the flag --
    neither ``derive_rookie_qb_debut_fade_features`` nor its required-column
    set references any such column at all."""

    schedule = _debut_schedule()
    schedule["result"] = 3.0
    schedule["home_score"] = 20
    schedule["away_score"] = 17
    baseline = derive_rookie_qb_debut_fade_features(schedule, _debut_rosters()).set_index("game_id")

    mutated = schedule.copy()
    mutated["result"] = -14.0
    mutated["home_score"] = 3
    mutated["away_score"] = 41
    after = derive_rookie_qb_debut_fade_features(mutated, _debut_rosters()).set_index("game_id")
    pd.testing.assert_series_equal(
        baseline[ROOKIE_QB_DEBUT_FADE_COLUMN], after[ROOKIE_QB_DEBUT_FADE_COLUMN]
    )


def test_rookie_debut_attach_is_purely_additive() -> None:
    schedule = _debut_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_rookie_qb_debut_fade_features(
        features, schedule=schedule, rosters=_debut_rosters()
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [ROOKIE_QB_DEBUT_FADE_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_rookie_debut_attach_requires_the_join_key() -> None:
    schedule = _debut_schedule()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_rookie_qb_debut_fade_features(features, schedule=schedule, rosters=_debut_rosters())


def test_rookie_debut_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _debut_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], ROOKIE_QB_DEBUT_FADE_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=ROOKIE_QB_DEBUT_FADE_COLUMN):
        attach_rookie_qb_debut_fade_features(features, schedule=schedule, rosters=_debut_rosters())


def test_rookie_debut_derive_requires_every_schedule_column() -> None:
    schedule = _debut_schedule().drop(columns=["home_qb_id"])
    with pytest.raises(DataContractError, match="home_qb_id"):
        derive_rookie_qb_debut_fade_features(schedule, _debut_rosters())


# ---------------------------------------------------------------------------
# LEAD-25: quarterback revenge game
# ---------------------------------------------------------------------------


def test_franchise_code_normalization_current_and_historical_codes_match() -> None:
    """The schedule's own historical codes (OAK/SD/STL) and the CURRENT
    codes (LV/LAC/LA) must canonicalize to the identical code space."""

    codes = pd.Series(["OAK", "LV", "SD", "LAC", "STL", "SL", "LA", "WAS", "ARI"])
    canonical = _canonical_schedule_team(codes)
    assert list(canonical) == ["LV", "LV", "LAC", "LAC", "LA", "LA", "LA", "WAS", "ARI"]


def test_draft_team_name_to_code_covers_every_relocation_variant() -> None:
    combine = pd.DataFrame(
        {
            "pfr_id": ["p_oak", "p_lv", "p_sd", "p_lac", "p_stl", "p_lar", "p_wr", "p_wf", "p_wc"],
            "draft_team": [
                "Oakland Raiders",
                "Las Vegas Raiders",
                "San Diego Chargers",
                "Los Angeles Chargers",
                "St. Louis Rams",
                "Los Angeles Rams",
                "Washington Redskins",
                "Washington Football Team",
                "Washington Commanders",
            ],
            "draft_year": [2005, 2021, 2005, 2021, 2005, 2021, 2005, 2019, 2022],
        }
    )
    rosters = pd.DataFrame(
        {
            "pfr_id": combine["pfr_id"],
            "gsis_id": [f"g_{pfr}" for pfr in combine["pfr_id"]],
        }
    )
    lookup = draft_team_by_gsis_id(combine, rosters)
    assert lookup["g_p_oak"] == "LV"
    assert lookup["g_p_lv"] == "LV"
    assert lookup["g_p_sd"] == "LAC"
    assert lookup["g_p_lac"] == "LAC"
    assert lookup["g_p_stl"] == "LA"
    assert lookup["g_p_lar"] == "LA"
    assert lookup["g_p_wr"] == "WAS"
    assert lookup["g_p_wf"] == "WAS"
    assert lookup["g_p_wc"] == "WAS"


def test_draft_team_by_gsis_id_rejects_unrecognized_names() -> None:
    combine = pd.DataFrame(
        {"pfr_id": ["p1"], "draft_team": ["Los Angeles Xtreme"], "draft_year": [2001]}
    )
    rosters = pd.DataFrame({"pfr_id": ["p1"], "gsis_id": ["g1"]})
    with pytest.raises(DataContractError, match="Los Angeles Xtreme"):
        draft_team_by_gsis_id(combine, rosters)


def test_draft_team_by_gsis_id_keeps_earliest_draft_year_on_duplicate() -> None:
    """A rare supplemental/re-entry-draft edge case: keep the ORIGINAL draft."""

    combine = pd.DataFrame(
        {
            "pfr_id": ["p1", "p1"],
            "draft_team": ["Oakland Raiders", "Kansas City Chiefs"],
            "draft_year": [2010, 2012],
        }
    )
    rosters = pd.DataFrame({"pfr_id": ["p1"], "gsis_id": ["g1"]})
    lookup = draft_team_by_gsis_id(combine, rosters)
    assert lookup["g1"] == "LV"  # the 2010 (earlier) draft, not the 2012 one


def _revenge_schedule() -> pd.DataFrame:
    return _schedule(
        [
            # Home QB Q1 (drafted by LV) faces away team OAK (-> LV) -> +1.
            _game("r1", 2020, "2020-09-10", "SEA", "OAK", "Q1", "Q9"),
            # Away QB Q2 (drafted by LAC) faces home team SD (-> LAC) -> -1.
            _game("r2", 2020, "2020-09-13", "SD", "DEN", "Q9", "Q2"),
            # Mutual revenge: home Q4 drafted by LAC (== away team LAC), away
            # Q3 drafted by LV (== home team LV) -> both true -> 0.
            _game("r3", 2020, "2020-09-20", "LV", "LAC", "Q4", "Q3"),
            # Neither drafted by the opponent -> 0.
            _game("r4", 2020, "2020-09-27", "DEN", "SEA", "Q9", "Q9"),
            # Home QB is not in the lookup at all (unjoined) -> treated as 0
            # for that side; away also not a revenge -> overall 0.
            _game("r5", 2020, "2020-10-04", "KC", "DEN", "QUNK", "Q9"),
        ]
    )


def _revenge_lookup() -> dict[str, str]:
    return {
        "Q1": "LV",
        "Q2": "LAC",
        "Q3": "LV",
        "Q4": "LAC",
        "Q9": "GB",  # never the opponent in any fixture game above
    }


def test_qb_revenge_sign_convention_home_is_positive() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r1", QB_REVENGE_COLUMN] == 1.0


def test_qb_revenge_sign_convention_away_is_negative() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r2", QB_REVENGE_COLUMN] == -1.0


def test_qb_revenge_both_sides_simultaneously_is_zero() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r3", QB_REVENGE_COLUMN] == 0.0


def test_qb_revenge_neither_side_is_zero() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r4", QB_REVENGE_COLUMN] == 0.0


def test_qb_revenge_unjoined_qb_is_treated_as_zero_never_guessed() -> None:
    derived = derive_qb_revenge_features(_revenge_schedule(), _revenge_lookup()).set_index(
        "game_id"
    )
    assert derived.loc["r5", QB_REVENGE_COLUMN] == 0.0


def test_qb_revenge_join_diagnostics_counts() -> None:
    diagnostic = qb_revenge_join_diagnostics(_revenge_schedule(), _revenge_lookup())
    # 5 games x 2 sides = 10 non-null QB-side starts; QUNK is the only one
    # absent from the lookup.
    assert diagnostic["n_qb_side_starts"] == 10
    assert diagnostic["n_resolved_draft_team"] == 9
    assert diagnostic["join_rate"] == pytest.approx(0.9)


def test_qb_revenge_leakage_ignores_unrelated_outcome_columns() -> None:
    """``qb_revenge_flag`` never reads any outcome column at all (it is a
    pure function of starter identity, team codes, and a static draft-team
    lookup); mutating an unrelated result/score column must never change
    it."""

    schedule = _revenge_schedule()
    schedule["result"] = 3.0
    schedule["home_score"] = 24
    schedule["away_score"] = 10
    lookup = _revenge_lookup()
    baseline = derive_qb_revenge_features(schedule, lookup).set_index("game_id")

    mutated = schedule.copy()
    mutated["result"] = -21.0
    mutated["home_score"] = 6
    mutated["away_score"] = 27
    after = derive_qb_revenge_features(mutated, lookup).set_index("game_id")
    pd.testing.assert_series_equal(baseline[QB_REVENGE_COLUMN], after[QB_REVENGE_COLUMN])


def test_qb_revenge_attach_is_purely_additive() -> None:
    schedule = _revenge_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_qb_revenge_features(
        features, schedule=schedule, draft_team_lookup=_revenge_lookup()
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [QB_REVENGE_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_qb_revenge_attach_requires_the_join_key() -> None:
    schedule = _revenge_schedule()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_qb_revenge_features(features, schedule=schedule, draft_team_lookup=_revenge_lookup())


def test_qb_revenge_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _revenge_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], QB_REVENGE_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=QB_REVENGE_COLUMN):
        attach_qb_revenge_features(features, schedule=schedule, draft_team_lookup=_revenge_lookup())


def test_qb_revenge_derive_requires_every_schedule_column() -> None:
    schedule = _revenge_schedule().drop(columns=["home_qb_id"])
    with pytest.raises(DataContractError, match="home_qb_id"):
        derive_qb_revenge_features(schedule, _revenge_lookup())


# ---------------------------------------------------------------------------
# Registered candidate profiles: production plus exactly the one column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(qiop.CANDIDATES))
def test_registered_profile_is_production_plus_the_declared_one_column(key: str) -> None:
    candidate = qiop.CANDIDATES[key]
    baseline = set(margin_feature_columns("market_residual", qiop.BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", candidate.profile))
    assert treatment - baseline == {candidate.column}
    assert baseline - treatment == set()


@pytest.mark.parametrize("key", sorted(qiop.CANDIDATES))
def test_candidate_duck_types_with_the_template_profile_identity(key: str) -> None:
    """``on_production_opener_confirmation.profile_identity`` is reused
    unmodified: our ``QbIdentityCandidate`` need only carry the same
    ``profile``/``column`` attribute names."""

    candidate = qiop.CANDIDATES[key]
    columns = margin_feature_columns("market_residual", candidate.profile)
    frame = pd.DataFrame({column: [0.0] for column in columns})
    observed = qiop.confirmation.profile_identity(candidate, frame)
    assert observed["only_added_column"] == candidate.column
