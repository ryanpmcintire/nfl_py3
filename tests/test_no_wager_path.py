from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"


WAGER_CLIENT_DENYLIST = {
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "caesarssportsbook",
    "pointsbet",
    "kalshi",
    "polymarket",
    "betfair",
    "pinnacle",
    "sportsbook",
    "wager",
    "bet365",
    "unibet",
    "bovada",
    "mybookie",
    "prizepicks",
    "underdogfantasy",
    "espnbet",
    "barstoolsportsbook",
    "wynnbet",
    "betrivers",
    "betway",
    "williamhill",
    "ladbrokes",
    "skybook",
    "888sport",
    "betonline",
    "sxbet",
    "novig",
}


def _dependency_names() -> set[str]:

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    raw_specs: list[str] = list(project.get("dependencies", []))
    for extra_deps in project.get("optional-dependencies", {}).values():
        raw_specs.extend(extra_deps)
    for group_deps in data.get("dependency-groups", {}).values():
        raw_specs.extend(group_deps)

    normalized: set[str] = set()
    for spec in raw_specs:
        name = re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].strip()
        normalized.add(name.lower().replace("_", "-"))
    return normalized


def test_no_dependency_matches_a_known_wagering_client_name() -> None:
    deps = _dependency_names()
    assert deps, "expected at least one declared dependency to check"
    hits = {dep: banned for dep in deps for banned in WAGER_CLIENT_DENYLIST if banned in dep}
    assert not hits, (
        "pyproject.toml declares a dependency matching the wagering-client "
        f"denylist: {hits}. This is a research and paper-decision project "
        "(AGENTS.md); do not add automated wagering."
    )


WAGER_PLACEMENT_PATTERN = re.compile(
    r"\bplace_bet\b|\bplace_wager\b|\bsubmit_bet\b|\bplace_order\b"
    r"""|['"]/bets['"]|\.post\([^)]*['"]/bets"""
)

ALLOWLISTED_READONLY_ODDS_MODULES = {
    SRC_DIR / "nfl_ats" / "odds.py",
    SRC_DIR / "nfl_ats" / "odds_backfill.py",
    SRC_DIR / "nfl_ats" / "market_data.py",
}


def _python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_no_wager_placement_verb_outside_the_readonly_odds_client() -> None:
    candidates = _python_files(SRC_DIR) + _python_files(SCRIPTS_DIR)
    assert candidates, "expected to scan at least one Python file under src/ or scripts/"

    offenders: list[str] = []
    for path in candidates:
        if path in ALLOWLISTED_READONLY_ODDS_MODULES:
            continue
        text = path.read_text(encoding="utf-8")
        if WAGER_PLACEMENT_PATTERN.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "wager-placement verb found outside the allowlisted read-only odds "
        f"client: {offenders}. This is a research and paper-decision "
        "project; do not add automated wagering."
    )


def test_allowlisted_odds_modules_actually_exist_and_stay_read_only() -> None:

    for path in ALLOWLISTED_READONLY_ODDS_MODULES:
        assert path.is_file(), f"expected allowlisted odds module to exist: {path}"
        text = path.read_text(encoding="utf-8")
        assert not WAGER_PLACEMENT_PATTERN.search(text), (
            f"{path} is allowlisted as read-only-quotes but now contains a "
            "wager-placement verb; re-audit before keeping it allowlisted"
        )


RESPONSIBLE_USE_DOC = REPO_ROOT / "docs" / "responsible_use.md"


def test_a_paper_only_limitations_statement_exists() -> None:

    assert RESPONSIBLE_USE_DOC.is_file(), (
        "docs/responsible_use.md is required as the paper-only/limitations "
        "statement this guard checks"
    )
    lowered = RESPONSIBLE_USE_DOC.read_text(encoding="utf-8").lower()
    assert "paper" in lowered
    assert "no code path from a model prediction to money" in lowered
    assert "wager" in lowered
    assert "automated wagering" in lowered
