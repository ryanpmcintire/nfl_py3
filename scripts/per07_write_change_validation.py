from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_ats.provenance import write_stamped_artifact

ROOT = Path("data/raw/coordinators")
STAFF_STATE = Path("artifacts/staff_state")
INSEASON_BASIS = "inseason_revision_change"
PERSON_STATES = ("filled", "interim")
ROLES = ("OC", "DC")


def revision_key(change: dict) -> tuple[int, str]:
    return int(str(change["source_url"]).rsplit("=", 1)[1]), str(change["role"])


def latest_validation() -> Path:
    return max(ROOT.glob("*/change_validation.json"))


def latest_staff_state() -> Path:
    return max(p for p in STAFF_STATE.iterdir() if (p / "staff_state.parquet").exists())


def new_person_changes(spells: pd.DataFrame, known: set[tuple[int, str]], now: datetime) -> list:
    spells = spells.loc[spells.role.isin(ROLES)].sort_values("first_observed_at")
    rows = []
    for _, group in spells.groupby(["season", "team", "role"]):
        previous = None
        for spell in group.itertuples(index=False):
            if (
                previous is not None
                and spell.first_observation_basis == INSEASON_BASIS
                and spell.state in PERSON_STATES
                and previous.state in PERSON_STATES
                and spell.person != previous.person
                and spell.first_observed_at.to_pydatetime() < now
                and (int(spell.first_revision_id), spell.role) not in known
            ):
                rows.append(
                    {
                        "season": int(spell.season),
                        "team": spell.team,
                        "role": spell.role,
                        "person": spell.person,
                        "previous_person": previous.person,
                        "previous_observed_at": previous.first_observed_at.isoformat(sep=" "),
                        "revision_at": spell.first_observed_at.isoformat(sep=" "),
                        "source_url": spell.first_source_url,
                        "event_date": spell.independent_event_date
                        if isinstance(spell.independent_event_date, str)
                        else None,
                        "independent_source": spell.independent_source
                        if isinstance(spell.independent_source, str)
                        else None,
                        "validation_status": spell.adjudication
                        if isinstance(spell.adjudication, str)
                        else "unadjudicated_staff_state_person_change",
                        "validation_note": "Person change in the dated staff-state table.",
                    }
                )
            previous = spell
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a stamped change_validation.json from the adjudicated staff-state table"
    )
    parser.add_argument("--staff-state", type=Path, default=None)
    parser.add_argument("--base", type=Path, default=None)
    args = parser.parse_args()
    now = datetime.now(UTC)
    base_path = args.base or latest_validation()
    state_dir = args.staff_state or latest_staff_state()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    carried = [c for c in base["changes"] if pd.Timestamp(c["revision_at"]).to_pydatetime() < now]
    known = {revision_key(c) for c in carried}
    spells = pd.read_parquet(state_dir / "staff_state.parquet")
    added = new_person_changes(spells, known, now)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    payload = {k: v for k, v in base.items() if k not in ("changes", "_provenance_stamp")}
    payload["changes"] = carried + added
    payload["written_at"] = now.isoformat()
    payload["carried_from"] = str(base_path)
    payload["staff_state_source"] = str(state_dir)
    payload["added_from_staff_state"] = len(added)
    payload["method"] = (
        f"Carried forward verbatim from {base_path.parent.name}; appended only in-season "
        "person changes in the dated staff-state table whose first observation precedes the "
        "write time. Never backdated; unadjudicated changes are not counted by the overlay."
    )
    destination = ROOT / stamp
    destination.mkdir(parents=True, exist_ok=False)
    write_stamped_artifact(payload, destination / "change_validation.json")
    print(destination / "change_validation.json", len(carried), len(added))


if __name__ == "__main__":
    main()
