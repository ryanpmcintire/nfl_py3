from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.provenance import write_stamped_artifact

ROOT = Path("data/raw/coordinators")
OUT = Path("artifacts/staff_state")
INGEST = Path(__file__).with_name("ingest_coordinator_history.py")
ROLE_LABELS = {"offensive coordinator": "OC", "defensive coordinator": "DC", "head coach": "HC"}
TEMPLATE_ROLES = {"offensive_coordinator": "OC", "defensive_coordinator": "DC", "coach": "HC"}
SIDE_BY_ROLE = {"OC": "offence", "DC": "defence"}
COUNTED_STATUSES = ("staff_change", "playcaller_role")
LINE_PATTERN = re.compile(
    r"^\s*\*\s*(?:(?:assistant|associate) head coach\s*[/&]\s*)?"
    r"(?P<interim_prefix>interim\s+)?"
    r"(?P<role>head coach|offensive coordinator|defensive coordinator)"
    r"(?P<label>(?:\s*\(interim\))?\s*(?:[/&][A-Za-z /&-]+)?)\s*[–—-]\s*(?P<person>.+)$",
    re.I,
)
TEMPLATE_PATTERN = re.compile(
    r"^\s*\|\s*(?P<role>offensive_coordinator|defensive_coordinator|coach)\s*=\s*(?P<person>.+)$",
    re.I,
)
VACANT_PATTERN = re.compile(r"\bvacant\b|\btbd\b|\bto be determined\b", re.I)
REF_PATTERN = re.compile(r"<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", re.S)

PLAYCALLER_CITATIONS = {
    1115297677: ("Al Holcomb", "playcaller_is_title_holder"),
    1120928130: ("Parks Frazier", "playcaller_is_title_holder"),
    1186218091: ("Mike Sullivan", "playcaller_differs_from_title"),
    1323963640: ("Greg Olson", "playcaller_is_title_holder"),
    1321786588: ("Mike Kafka", "playcaller_differs_from_title"),
}


def load_ingest() -> Any:
    spec = importlib.util.spec_from_file_location("ingest_coordinator_history", INGEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_revision(
    content: str, linked_person: Any
) -> dict[str, list[tuple[str | None, bool, bool]]]:
    found: dict[str, list[tuple[str | None, bool, bool]]] = {}
    for line in content.splitlines():
        line = re.sub(
            r"^(\s*\*\s*)\[\[(?:[^\]|]+\|)?(head coach|offensive coordinator|defensive coordinator)\]\]",
            r"\1\2",
            line,
            flags=re.I,
        )
        match = LINE_PATTERN.match(line)
        if match:
            role = ROLE_LABELS[match["role"].lower()]
            text = match["person"]
            interim = bool(match["interim_prefix"]) or "interim" in match["label"].lower()
        else:
            match = TEMPLATE_PATTERN.match(line)
            if not match:
                continue
            role = TEMPLATE_ROLES[match["role"].lower()]
            text = match["person"]
            interim = False
        bare = REF_PATTERN.sub("", text)
        interim = interim or bool(re.search(r"\(interim\)", bare, re.I))
        vacant = bool(VACANT_PATTERN.search(bare))
        person = None if vacant else linked_person(text)
        found.setdefault(role, []).append((person, interim, vacant))
    return found


def role_state(entries: list[tuple[str | None, bool, bool]]) -> tuple[str, str | None, str | None]:
    people = {p for p, _, v in entries if p and not v}
    interim_people = {p for p, i, v in entries if p and i and not v}
    if any(v for _, _, v in entries) and not people:
        return "vacant", None, None
    if any(p is None and not v for p, _, v in entries) and not people:
        return "unlinked", None, None
    if len(interim_people) == 1:
        others = sorted(people - interim_people)
        return "interim", next(iter(interim_people)), " | ".join(others) or None
    if len(people) == 1:
        return "filled", next(iter(people)), None
    return "multiple", " | ".join(sorted(people)), None


def collect_observations(source: Path, ingest: Any) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(source.glob("*.request.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != 200 or not meta.get("api_ok"):
            continue
        body = meta_path.with_name(meta_path.name.replace(".request.json", ".response"))
        payload = json.loads(body.read_bytes())
        for page in payload.get("query", {}).get("pages", []):
            for revision in page.get("revisions", []):
                content = revision.get("slots", {}).get("main", {}).get("content", "")
                stamp = pd.Timestamp(revision["timestamp"])
                for role, entries in scan_revision(content, ingest.linked_person).items():
                    state, person, listed_alongside = role_state(entries)
                    rows.append(
                        {
                            "season": int(meta["season"]),
                            "team": meta["team"],
                            "role": role,
                            "state": state,
                            "person": person,
                            "listed_alongside": listed_alongside,
                            "observed_at": stamp,
                            "revision_id": int(revision["revid"]),
                            "mode": meta["mode"],
                        }
                    )
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(["season", "team", "role", "revision_id"]).sort_values(
        ["season", "team", "role", "observed_at", "revision_id"]
    )


def build_spells(observations: pd.DataFrame) -> pd.DataFrame:
    spells = []
    for (season, team, role), group in observations.groupby(["season", "team", "role"], sort=True):
        group = group.reset_index(drop=True)
        key = group[["state", "person", "listed_alongside"]].fillna("").agg("|".join, axis=1)
        block = key.ne(key.shift()).cumsum()
        for _, part in group.groupby(block, sort=True):
            first, last = part.iloc[0], part.iloc[-1]
            following = group.index[group.index > part.index[-1]]
            nxt = group.loc[following[0]] if len(following) else None
            spells.append(
                {
                    "season": season,
                    "team": team,
                    "role": role,
                    "state": first["state"],
                    "person": first["person"],
                    "listed_alongside": first["listed_alongside"],
                    "first_observed_at": first["observed_at"],
                    "first_revision_id": first["revision_id"],
                    "first_source_url": f"https://en.wikipedia.org/w/index.php?oldid={first['revision_id']}",
                    "first_observation_basis": (
                        "preseason_sample_latest_revision_before_cutoff"
                        if first["mode"] == "preseason"
                        else "inseason_revision_change"
                    ),
                    "last_confirmed_at": last["observed_at"],
                    "ended_observed_at": None if nxt is None else nxt["observed_at"],
                    "ended_revision_id": None if nxt is None else nxt["revision_id"],
                    "observations_in_spell": len(part),
                }
            )
    frame = pd.DataFrame(spells)
    frame["ended_observed_at"] = pd.to_datetime(frame["ended_observed_at"], utc=True)
    return frame


def attach_adjudication(spells: pd.DataFrame, validation: dict[str, Any]) -> pd.DataFrame:
    by_revision = {}
    for change in validation["changes"]:
        revision = int(change["source_url"].rsplit("=", 1)[1])
        by_revision[revision] = change
    spells = spells.copy()
    spells["adjudication"] = spells.first_revision_id.map(
        lambda r: by_revision[r]["validation_status"] if r in by_revision else None
    )
    spells["independent_source"] = spells.first_revision_id.map(
        lambda r: by_revision[r]["independent_source"] if r in by_revision else None
    )
    spells["independent_event_date"] = spells.first_revision_id.map(
        lambda r: by_revision[r]["event_date"] if r in by_revision else None
    )
    spells["playcaller_name"] = None
    spells["playcaller_side"] = None
    spells["playcaller_status"] = "unknown"
    spells["playcaller_evidence"] = None
    spells["playcaller_usable_from"] = pd.Series(
        pd.NaT, index=spells.index, dtype="datetime64[us, UTC]"
    )
    for revision, (caller, status) in PLAYCALLER_CITATIONS.items():
        change = by_revision[revision]
        mask = spells.first_revision_id.eq(revision) & spells.role.isin(SIDE_BY_ROLE)
        if mask.sum() != 1:
            raise ValueError(f"Playcaller citation {revision} does not match one spell")
        spells.loc[mask, "playcaller_name"] = caller
        spells.loc[mask, "playcaller_side"] = spells.loc[mask, "role"].map(SIDE_BY_ROLE)
        spells.loc[mask, "playcaller_status"] = status
        spells.loc[mask, "playcaller_evidence"] = (
            change["independent_source"] + " | " + change["validation_note"]
        )
        spells.loc[mask, "playcaller_usable_from"] = spells.loc[mask, "first_observed_at"]
    return spells


def validate(
    spells: pd.DataFrame,
    observations: pd.DataFrame,
    validation: dict[str, Any],
    history: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["spells_missing_observation_timestamp"] = int(spells.first_observed_at.isna().sum())
    preseason_cutoff = spells.season.map(lambda s: pd.Timestamp(f"{s}-09-01T00:00:00Z"))
    baseline = spells.first_observation_basis.eq("preseason_sample_latest_revision_before_cutoff")
    result["preseason_spells_observed_after_cutoff"] = int(
        (baseline & (spells.first_observed_at > preseason_cutoff)).sum()
    )
    keys = history[["revision_id", "role", "person"]].drop_duplicates()
    mine = observations.loc[
        observations.state.isin(["filled", "interim"]), ["revision_id", "role", "person"]
    ].drop_duplicates()
    merged = keys.merge(mine, on=["revision_id", "role"], how="left", suffixes=("_ingest", "_mine"))
    result["ingest_rows"] = len(keys)
    result["ingest_rows_reproduced_same_person"] = int(
        merged.person_ingest.eq(merged.person_mine).sum()
    )
    counted = [c for c in validation["changes"] if c["validation_status"] in COUNTED_STATUSES]
    matched, detail = 0, []
    for change in counted:
        revision = int(change["source_url"].rsplit("=", 1)[1])
        row = spells.loc[spells.first_revision_id.eq(revision) & spells.role.eq(change["role"])]
        ok = len(row) == 1 and str(row.iloc[0].first_observed_at) == str(
            pd.Timestamp(change["revision_at"])
        )
        if ok:
            previous = spells.loc[
                spells.season.eq(change["season"])
                & spells.team.eq(change["team"])
                & spells.role.eq(change["role"])
                & spells.ended_revision_id.eq(revision)
            ]
            ok = len(previous) == 1
        matched += int(ok)
        detail.append(
            {
                "team": change["team"],
                "season": change["season"],
                "role": change["role"],
                "revision_id": revision,
                "matched": bool(ok),
            }
        )
    result["known_events"] = len(counted)
    result["known_events_matched"] = matched
    result["known_event_detail"] = detail
    inseason = spells.loc[
        spells.first_observation_basis.eq("inseason_revision_change")
        & spells.season.between(2022, 2025)
    ]
    filled_changes = inseason.loc[
        inseason.state.isin(["filled", "interim"]) & inseason.role.isin(["OC", "DC"])
    ]
    result["inseason_oc_dc_person_spells_2022_2025"] = len(filled_changes)
    result["of_which_adjudicated"] = int(filled_changes.adjudication.notna().sum())
    result["of_which_unadjudicated_revision_ids"] = [
        int(r) for r in filled_changes.loc[filled_changes.adjudication.isna(), "first_revision_id"]
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the dated staff-state table from cached Wikipedia staff revisions"
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "20260907T213814366437Z" / "change_validation.json",
    )
    args = parser.parse_args()
    source = args.source or max(
        p for p in ROOT.iterdir() if (p / "coordinator_history.parquet").exists()
    )
    ingest = load_ingest()
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    history = pd.read_parquet(source / "coordinator_history.parquet")
    observations = collect_observations(source, ingest)
    spells = attach_adjudication(build_spells(observations), validation)
    checks = validate(spells, observations, validation, history)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = OUT / stamp
    destination.mkdir(parents=True, exist_ok=False)
    spells.to_parquet(destination / "staff_state.parquet", index=False)
    by_state = spells.groupby(["season", "state"]).size().unstack(fill_value=0)
    manifest = {
        "source_snapshot": str(source),
        "validation_file": str(args.validation),
        "network_requests": 0,
        "spells": len(spells),
        "observations": len(observations),
        "state_counts_by_season": {
            int(s): {k: int(v) for k, v in r.items()} for s, r in by_state.iterrows()
        },
        "playcaller_status_counts": {
            k: int(v) for k, v in spells.playcaller_status.value_counts().items()
        },
        "validation": checks,
        "parquet_sha256": hashlib.sha256(
            (destination / "staff_state.parquet").read_bytes()
        ).hexdigest(),
    }
    write_stamped_artifact(manifest, destination / "manifest.json")
    print(destination)
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "validation"}, indent=1, default=str)
    )
    print(
        json.dumps(
            {k: v for k, v in checks.items() if k != "known_event_detail"}, indent=1, default=str
        )
    )


if __name__ == "__main__":
    main()
