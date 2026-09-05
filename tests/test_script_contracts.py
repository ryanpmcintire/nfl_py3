"""ENG-29: unit tests for the ``nfl_ats.script_contracts`` static scanner.

Each case writes a small synthetic script to ``tmp_path`` and scans it --
``scan_script`` never imports or executes its target, so these bodies can be
nonsense (no real imports need to resolve) as long as they parse.
"""

from __future__ import annotations

from pathlib import Path

from nfl_ats.script_contracts import STAMPED_LIBRARY_WRITERS, scan_library_writer, scan_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "script.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_declared_read_only_with_no_writes_passes(tmp_path: Path) -> None:
    source = """
READ_ONLY_SCRIPT = True


def main() -> None:
    print("hello")
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.declares_read_only is True
    assert contract.write_sites == ()
    assert contract.is_read_only_verified is True


def test_declared_read_only_with_an_artifacts_write_fails(tmp_path: Path) -> None:
    source = """
from pathlib import Path

READ_ONLY_SCRIPT = True
OUTPUT_DIR = Path("artifacts/some_screen")


def main() -> None:
    (OUTPUT_DIR / "results.json").write_text("{}")
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.declares_read_only is True
    assert contract.is_read_only_verified is False
    assert len(contract.gated_write_sites) == 1
    assert contract.gated_write_sites[0].classification == "artifacts"


def test_helper_call_passes_without_a_read_only_declaration(tmp_path: Path) -> None:
    source = """
from nfl_ats.provenance import write_experiment_artifact


def main() -> None:
    write_experiment_artifact(directory, "metadata.json", {}, command="demo", metrics={})
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.declares_read_only is False
    assert contract.calls_provenance_helper is True


def test_write_stamped_artifact_also_counts_as_the_helper(tmp_path: Path) -> None:
    source = """
from nfl_ats.provenance import write_stamped_artifact


def main() -> None:
    write_stamped_artifact({}, some_path)
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.calls_provenance_helper is True


def test_unknown_write_with_an_exception_entry_passes(tmp_path: Path) -> None:
    source = """
READ_ONLY_SCRIPT = True
READ_ONLY_EXCEPTIONS: dict[int, str] = {
    9: "destination is the off-device mirror drive, not artifacts/",
}


def main() -> None:
    some_mystery_path.write_text("data")
"""
    contract = scan_script(_write(tmp_path, source))
    write_sites = contract.write_sites
    assert len(write_sites) == 1
    assert write_sites[0].classification == "unknown"
    assert write_sites[0].line in contract.read_only_exceptions
    assert contract.is_read_only_verified is True


def test_unknown_write_without_an_exception_entry_fails(tmp_path: Path) -> None:
    source = """
READ_ONLY_SCRIPT = True


def main() -> None:
    some_mystery_path.write_text("data")
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.is_read_only_verified is False
    assert len(contract.gated_write_sites) == 1
    assert contract.gated_write_sites[0].classification == "unknown"


def test_json_dump_to_stdout_is_classified_stdout(tmp_path: Path) -> None:
    source = """
import json
import sys

READ_ONLY_SCRIPT = True


def main() -> None:
    json.dump({"ok": True}, sys.stdout)
"""
    contract = scan_script(_write(tmp_path, source))
    assert len(contract.write_sites) == 1
    site = contract.write_sites[0]
    assert site.kind == "json.dump("
    assert site.classification == "stdout"
    # stdout never gates a read-only claim.
    assert contract.is_read_only_verified is True


def test_registry_write_is_classified_registry_and_gates(tmp_path: Path) -> None:
    source = """
from pathlib import Path

READ_ONLY_SCRIPT = True
REGISTRY_ROOT = Path("registry") / "experiments"


def main() -> None:
    (REGISTRY_ROOT / "row.json").write_text("{}")
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.write_sites[0].classification == "registry"
    assert contract.is_read_only_verified is False


def test_cli_arg_destination_with_no_governed_default_is_tmp_or_arg(tmp_path: Path) -> None:
    source = """
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
"""
    contract = scan_script(_write(tmp_path, source))
    assert len(contract.write_sites) == 1
    assert contract.write_sites[0].classification == "tmp_or_arg"
    # tmp_or_arg never gates -- this script could self-certify read-only.
    assert contract.gated_write_sites == ()


def test_cli_arg_destination_with_a_governed_default_is_artifacts(tmp_path: Path) -> None:
    source = """
import argparse
from pathlib import Path

REPO = Path(".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "levers.json")
    args = parser.parse_args()
    args.out.write_text("{}")
"""
    contract = scan_script(_write(tmp_path, source))
    assert len(contract.write_sites) == 1
    assert contract.write_sites[0].classification == "artifacts"


def test_atomic_json_destination_is_the_second_positional_argument(tmp_path: Path) -> None:
    source = """
from nfl_ats.io import atomic_json

READ_ONLY_SCRIPT = True


def main() -> None:
    atomic_json({"a": 1}, some_scratch_path / "out.json")
"""
    contract = scan_script(_write(tmp_path, source))
    assert len(contract.write_sites) == 1
    assert contract.write_sites[0].kind == "atomic_json"
    # "some_scratch_path" resolves to nothing artifacts/registry-shaped, so
    # it is unknown, not a free pass -- distinct from the tmp_or_arg cases
    # above, which are recognised structurally (a parse_args() namespace).
    assert contract.write_sites[0].classification == "unknown"


def test_never_imports_or_executes_the_scanned_script(tmp_path: Path) -> None:
    """A script with import-time side effects (here: raising) must still
    scan cleanly -- scan_script only parses source with ast.
    """

    source = """
raise RuntimeError("this must never run")

READ_ONLY_SCRIPT = True


def main() -> None:
    pass
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.is_read_only_verified is True


def test_script_calling_an_imported_stamped_library_writer_is_recognised(tmp_path: Path) -> None:
    """ENG-38: a script whose own file has zero write sites, but which calls a
    function listed in STAMPED_LIBRARY_WRITERS, must be flagged
    ``calls_stamped_library_writer`` -- this is how e.g.
    ``record_bye_edge_fade_challenger.py`` (no write calls of its own; the
    real write happens inside the imported overlay function) resolves the
    provenance gate without a bare allowlist entry.
    """
    source = """
from nfl_ats.bye_edge_fade_overlay import record_bye_edge_fade_challenger_decisions


def main() -> None:
    record_bye_edge_fade_challenger_decisions()
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.write_sites == ()
    assert contract.calls_provenance_helper is False
    assert contract.calls_stamped_library_writer is True


def test_calling_an_unlisted_function_of_the_same_name_is_not_recognised(
    tmp_path: Path,
) -> None:
    """The allowlist match is on ``(module, function)``, not the bare function
    name -- a same-named function imported from an unrelated module must not
    be mistaken for the real, stamped one.
    """
    source = """
from some_other_module import record_bye_edge_fade_challenger_decisions


def main() -> None:
    record_bye_edge_fade_challenger_decisions()
"""
    contract = scan_script(_write(tmp_path, source))
    assert contract.calls_stamped_library_writer is False


def test_stamped_library_writers_really_stamp() -> None:
    """Every function listed in STAMPED_LIBRARY_WRITERS must itself call a
    provenance-stamping helper (write_experiment_artifact/
    write_stamped_artifact/stamp_sidecar) somewhere in its own body -- an
    AST-only check (never imports the library module), so the allowlist is a
    verified claim rather than a bare declaration a later edit to the library
    function could silently invalidate.
    """
    unstamped = [
        f"{module}.{function}"
        for module, function in sorted(STAMPED_LIBRARY_WRITERS)
        if not scan_library_writer(module, function, SRC_ROOT)
    ]
    assert not unstamped, (
        f"STAMPED_LIBRARY_WRITERS entries that do not actually call a "
        f"provenance-stamping helper: {unstamped}. Either wire in "
        "write_stamped_artifact/stamp_sidecar at that function's write site, "
        "or remove the entry -- a script delegating to it is not provenance-"
        "safe otherwise."
    )
