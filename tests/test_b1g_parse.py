"""Unit tests for the Big Ten availability-report PDF parser (scripts/b1g_parse.py).

The fixtures below replicate the two measured layout generations verbatim
(standard game page, bye page) plus the extraction artifacts pypdf produces on
the real snapshot (split capital T/Y, curly apostrophes, "(Season)" trailing
annotations). The parser's contract is to fail loudly on anything outside the
known layouts -- several tests pin that behavior.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "b1g_parse.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("b1g_parse", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves its own string annotations via sys.modules; without
    # registering the module first, @dataclass crashes on load.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def b1g() -> ModuleType:
    return _load_module()


STANDARD_PAGE = """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 1: Aug. 31 - Sept. 3
ILLINOIS vs. Toledo
7:30 p.m. | Big Ten Network
OUT
7  Matthew Bailey
9   T yson Rooks
QUESTIONABLE
12  Cade McNamara
"""

ANNOTATED_PAGE = """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 4: Sept. 22-23
ILLINOIS vs. Florida Atlantic
3:30 p.m. | Big Ten Network
OUT
2   Matthew Bailey (Season)
QUESTIONABLE
None
"""

AWAY_PAGE = """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 1: Aug. 31 - Sept. 3
NEBRASKA at Minnesota
8 p.m. | FOX
OUT
3  Nick Henrich
QUESTIONABLE
None
"""

BYE_PAGE = """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 5: Sept. 30
OHIO STATE
BYE
OUT QUESTIONABLE
"""


def test_standard_page_splits_designations_and_repairs_split_letters(
    b1g: ModuleType,
) -> None:
    page = b1g.parse_page_text(STANDARD_PAGE, 0, context="test")
    assert (page.team_raw, page.team_code, page.cfb_display_name) == (
        "ILLINOIS",
        "ILL",
        "Illinois",
    )
    assert page.opponent_raw == "Toledo"
    assert page.venue_side == "home"
    assert not page.is_bye
    assert [(entry.designation_raw, entry.name_raw, entry.name) for entry in page.entries] == [
        ("OUT", "Matthew Bailey", "Matthew Bailey"),
        ("OUT", "T yson Rooks", "Tyson Rooks"),
        ("QUESTIONABLE", "Cade McNamara", "Cade McNamara"),
    ]
    assert [entry.number for entry in page.entries] == [7, 9, 12]


def test_away_matchup_records_venue_side(b1g: ModuleType) -> None:
    page = b1g.parse_page_text(AWAY_PAGE, 0, context="test")
    assert page.team_code == "NEB"
    assert page.opponent_raw == "Minnesota"
    assert page.venue_side == "away"


def test_season_annotation_is_lifted_into_its_own_column(b1g: ModuleType) -> None:
    page = b1g.parse_page_text(ANNOTATED_PAGE, 3, context="test")
    assert len(page.entries) == 1
    entry = page.entries[0]
    assert entry.annotation == "season"
    assert entry.name == "Matthew Bailey"


def test_bye_page_yields_no_entries(b1g: ModuleType) -> None:
    page = b1g.parse_page_text(BYE_PAGE, 8, context="test")
    assert page.is_bye
    assert page.team_code == "OSU"
    assert page.opponent_raw is None
    assert page.venue_side is None
    assert page.entries == []


def test_week_mismatch_fails_loudly(b1g: ModuleType) -> None:
    page = b1g.parse_page_text(STANDARD_PAGE, 0, context="test")
    with pytest.raises(b1g.PageParseError, match="manifest says Week 6"):
        b1g.assert_week_matches(page, 6, "test")


@pytest.mark.parametrize(
    "page_text",
    [
        # Unknown section header where a designation header belongs.
        """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 1: Aug. 31 - Sept. 3
ILLINOIS vs. Toledo
7:30 p.m. | Big Ten Network
OUT
7  Matthew Bailey
DOUBTFUL
None
""",
        # Missing QUESTIONABLE section entirely.
        """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 1: Aug. 31 - Sept. 3
IOWA vs. Utah State
Noon | FS1
OUT
27  Jermari Harris
""",
        # Player line without a jersey number.
        """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 1: Aug. 31 - Sept. 3
PURDUE vs. Fresno State
Noon | Big Ten Network
OUT
Salim Turner-Muhammad
QUESTIONABLE
None
""",
        # Unmapped team name.
        """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 1: Aug. 31 - Sept. 3
FLORIDA vs. Toledo
7:30 p.m. | Big Ten Network
OUT
None
QUESTIONABLE
None
""",
        # Wrong report header entirely.
        """SOME OTHER REPORT
Week 1: Aug. 31 - Sept. 3
ILLINOIS vs. Toledo
OUT
None
QUESTIONABLE
None
""",
        # Bye page missing its BYE line.
        """BIG TEN FOOTBALL AVAILABILITY REPORT
Week 5: Sept. 30
OHIO STATE
OUT QUESTIONABLE
""",
    ],
)
def test_unparseable_pages_raise_instead_of_dropping(b1g: ModuleType, page_text: str) -> None:
    with pytest.raises(b1g.PageParseError):
        b1g.parse_page_text(page_text, 0, context="test")


def test_all_snapshot_teams_map_to_unique_stable_codes(b1g: ModuleType) -> None:
    codes = {code for code, _ in b1g.TEAM_CODES.values()}
    displays = {display for _, display in b1g.TEAM_CODES.values()}
    assert len(codes) == len(b1g.TEAM_CODES) == 14
    assert len(displays) == len(b1g.TEAM_CODES)


def test_clean_player_name_normalizes_apostrophes_and_whitespace(
    b1g: ModuleType,
) -> None:
    name, annotation = b1g.clean_player_name("Peyton \u2019O\u2019Leary\u2019s  Spot ")
    assert name == "Peyton 'O'Leary's Spot"
    assert annotation is None
