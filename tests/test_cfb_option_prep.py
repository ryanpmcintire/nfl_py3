"""LEAD-45 flag definition: frozen option identity, signed encoding."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cfb_option_prep_screen import attach_option_flag, is_option_team


def test_option_identity_is_era_scoped() -> None:
    assert is_option_team("Army", 2012)
    assert is_option_team("Navy", 2025)
    assert is_option_team("Air Force", 2006)
    assert is_option_team("Georgia Tech", 2008)
    assert is_option_team("Georgia Tech", 2018)
    assert not is_option_team("Georgia Tech", 2019)
    assert not is_option_team("Georgia Tech", 2007)
    assert not is_option_team("Alabama", 2015)


def test_flag_is_signed_and_option_vs_option_is_zero() -> None:
    frame = pd.DataFrame(
        {
            "season": [2015, 2015, 2015, 2015],
            "home_team": ["Army", "Alabama", "Army", "Alabama"],
            "away_team": ["Alabama", "Navy", "Navy", "Georgia"],
        }
    )
    flagged = attach_option_flag(frame)
    assert list(flagged["cfb_option_side"]) == [1.0, -1.0, 0.0, 0.0]
