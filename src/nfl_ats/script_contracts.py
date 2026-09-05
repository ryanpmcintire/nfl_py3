"""ENG-29: static, AST-only classification of ``scripts/*.py`` write behaviour.

Backs the provenance gate in ``tests/test_experiment_registry.py``. That gate
used to be a hand-maintained ``_ALLOWLISTED_UNSTAMPED_SCRIPTS`` frozenset --
one entry (and a reason comment) per script that legitimately does not call
``write_experiment_artifact``. The list only ever grew, and the reason lived
in the test file, far from the code it described.

This module replaces the *judgment* half of that pattern (does this script
write into ``artifacts/`` or ``registry/``?) with a mechanical answer, so the
*declaration* half (``READ_ONLY_SCRIPT = True``, plus an optional
``READ_ONLY_EXCEPTIONS`` dict) can live beside the code it describes instead
of in a burn-down list.

Deliberately **never imports or executes the target script** -- it only
parses the source with :mod:`ast`. That is both a safety property (research
scripts are not vetted for import-time side effects) and a scope limitation
worth naming: a script that itself makes no write call, but calls into an
imported helper that writes on its behalf, is invisible to this scanner. The
project's older grep-based version of this same gate had the identical
single-file scope, so this is a continuation of an existing convention, not a
new gap.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

WriteClass = Literal["artifacts", "registry", "stdout", "tmp_or_arg", "unknown"]

_PROVENANCE_HELPERS = frozenset(
    {"write_experiment_artifact", "write_stamped_artifact", "stamp_sidecar"}
)

# ENG-38: a small, explicit allowlist of library functions (outside any single
# script's own file) that this AST-only scanner cannot see writing on a
# script's behalf -- e.g. ``record_bye_edge_fade_challenger.py`` has zero
# write sites of its own; the real write happens inside
# ``record_bye_edge_fade_challenger_decisions()`` in
# ``nfl_ats.bye_edge_fade_overlay``. Listing the delegating call here (as
# ``"module.qualified.name:function"``) lets the provenance gate treat a
# script that calls one of these exactly like a script that calls
# ``write_stamped_artifact``/``write_experiment_artifact`` directly, while
# ``tests/test_script_contracts.py::test_stamped_library_writers_really_stamp``
# keeps the claim honest with an AST check that the named function's own body
# actually calls a stamping helper.
STAMPED_LIBRARY_WRITERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("nfl_ats.bye_edge_fade_overlay", "record_bye_edge_fade_challenger_decisions"),
        (
            "nfl_ats.pace_mismatch_dog_tilt_overlay",
            "record_pace_mismatch_dog_tilt_challenger_decisions",
        ),
        (
            "nfl_ats.special_teams_return_tilt_overlay",
            "record_special_teams_return_tilt_challenger_decisions",
        ),
        (
            "nfl_ats.tank_zone_fade_tilt_overlay",
            "record_tank_zone_fade_tilt_challenger_decisions",
        ),
        (
            "nfl_ats.third_down_reversion_fade_overlay",
            "record_third_down_reversion_fade_challenger_decisions",
        ),
        (
            "nfl_ats.turnover_luck_rebound_tilt_overlay",
            "record_turnover_luck_rebound_tilt_challenger_decisions",
        ),
        ("nfl_ats.scheduled_lock", "execute_scheduled_lock"),
    }
)

_ATOMIC_HELPERS = frozenset(
    {"atomic_json", "atomic_text", "atomic_csv", "atomic_parquet", "atomic_bytes"}
)
_PANDAS_WRITE_METHODS = frozenset({"to_parquet", "to_csv", "to_json"})
_PATH_WRITE_METHODS = frozenset({"write_text", "write_bytes"})
_WRITE_MODE_CHARS = frozenset("wax")  # any of these in an open() mode string means "writes"

# Best-effort tokens used to classify a destination's unparsed source text.
# Order matters: registry beats artifacts beats stdout beats tmp/arg.
_TMP_OR_ARG_TOKENS = ("args.", "opts.", "parsed.", "tmp_path", "tempfile", "namedtemporaryfile")


@dataclass(frozen=True)
class WriteSite:
    """One call this scanner considers a potential filesystem write."""

    line: int
    kind: str
    classification: WriteClass
    detail: str


@dataclass(frozen=True)
class ScriptContract:
    """The scanner's verdict for one ``scripts/*.py`` file."""

    path: Path
    declares_read_only: bool
    calls_provenance_helper: bool
    write_sites: tuple[WriteSite, ...]
    read_only_exceptions: dict[int, str] = field(default_factory=dict)
    # ENG-38: True when the script calls a function listed in
    # STAMPED_LIBRARY_WRITERS (imported via a plain `from module import name`).
    # A script whose own write sites the scanner sees as none (the real write
    # happens inside the delegated-to library function) is otherwise
    # indistinguishable from a script that never writes at all; this field is
    # the gate's third acceptable resolution, alongside calls_provenance_helper
    # and is_read_only_verified.
    calls_stamped_library_writer: bool = False

    @property
    def gated_write_sites(self) -> tuple[WriteSite, ...]:
        """Write sites that count against a ``READ_ONLY_SCRIPT`` claim."""

        return tuple(
            site
            for site in self.write_sites
            if site.classification in ("artifacts", "registry")
            or (site.classification == "unknown" and site.line not in self.read_only_exceptions)
        )

    @property
    def is_read_only_verified(self) -> bool:
        """True if this script may legitimately claim ``READ_ONLY_SCRIPT = True``."""

        return self.declares_read_only and not self.gated_write_sites


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive; ast.unparse is total for parsed trees
        return "<unparsable>"


def _argparse_dest_name(call: ast.Call) -> str | None:
    """``parser.add_argument("--out-dir", ..., dest="foo")`` -> the namespace
    attribute name argparse will bind: an explicit ``dest=`` keyword if
    present, else the first ``--``-prefixed flag string, snake-cased.
    """

    for kw in call.keywords:
        if (
            kw.arg == "dest"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    for arg in call.args:
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ):
            return arg.value.lstrip("-").replace("-", "_")
    for arg in call.args:
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("-")
        ):
            return arg.value.lstrip("-").replace("-", "_")
    return None


def _argparse_arg_defaults(tree: ast.Module) -> dict[str, str]:
    """Map each ``add_argument`` destination to the unparsed text of its
    ``default=`` value (empty string if there is none, e.g. ``required=True``
    or an implicit ``None``) -- so a namespace attribute known to default
    into a governed tree (``default=REPO_ROOT / "artifacts" / "x"``) is not
    mistaken for a caller-supplied, opt-in destination.
    """

    defaults: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        dest = _argparse_dest_name(node)
        if dest is None:
            continue
        default_text = ""
        for kw in node.keywords:
            if kw.arg == "default" and not (
                isinstance(kw.value, ast.Constant) and kw.value.value is None
            ):
                default_text = _unparse(kw.value)
                names = {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
                resolved = [
                    bit for name in sorted(names) if (bit := _resolve_name_text(name, tree, set()))
                ]
                if resolved:
                    default_text += " " + " ".join(resolved)
        defaults[dest] = default_text
    return defaults


def _argparse_namespace_names(tree: ast.Module) -> frozenset[str]:
    """Names bound to ``parser.parse_args(...)`` -- this project spells the
    variable ``args``, ``arguments``, and ``parsed`` about equally often, so
    a fixed token list misses whichever ones aren't guessed. Any attribute
    access on one of these (``arguments.output``, ``args.out.parent``, ...)
    is a CLI-supplied destination, i.e. ``tmp_or_arg``, regardless of what
    the attribute happens to be named.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "parse_args"
        ):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return frozenset(names)


def _classify_literal(text: str) -> WriteClass:
    """Classify by literal content only -- no CLI-arg/tempfile inference.
    Used both standalone and as the base case other classifiers fall back on
    so a governed literal (``"registry"``/``"artifacts"``) always outranks a
    structural ``tmp_or_arg`` guess.
    """

    lowered = text.lower()
    if "registry" in lowered:
        return "registry"
    if "artifacts" in lowered:
        return "artifacts"
    if "stdout" in lowered:
        return "stdout"
    return "unknown"


def _classify_text(text: str, namespace_names: frozenset[str] = frozenset()) -> WriteClass:
    classification = _classify_literal(text)
    if classification != "unknown":
        return classification
    lowered = text.lower()
    if any(token in lowered for token in _TMP_OR_ARG_TOKENS):
        return "tmp_or_arg"
    if any(f"{name}." in text for name in namespace_names):
        return "tmp_or_arg"
    return "unknown"


def _enclosing_function_defaults_text(node: ast.expr, tree: ast.Module, target_name: str) -> str:
    """Best-effort fallback: a write call's destination is often a bare
    parameter name (``def report(path: Path = ARTIFACTS_ROOT / "x"): ...
    path.write_text(...)``); the literal hint lives in the default, not the
    call site. Only consulted when the call-site text alone is ambiguous.
    """

    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not (func.lineno <= node.lineno <= (func.end_lineno or func.lineno)):
            continue
        args = func.args
        params = [*args.posonlyargs, *args.args]
        defaults = list(args.defaults)
        paired = list(zip(params[len(params) - len(defaults) :], defaults, strict=True))
        for kw, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            if default is not None:
                paired.append((kw, default))
        for param, default in paired:
            if param.arg == target_name:
                return _unparse(default)
    return ""


def _assign_value_for_name(node: ast.AST, name: str) -> ast.expr | None:
    """If ``node`` is a plain or annotated assignment to ``name``, its RHS."""

    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets
    ):
        return node.value
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
        and node.value is not None
    ):
        return node.value
    return None


def _resolve_name_text(name: str, tree: ast.Module, seen: set[str], depth: int = 0) -> str:
    """Best-effort trace of every assignment to ``name`` anywhere in the
    module (any scope -- this scanner does not track scoping precisely,
    which only widens what it considers, never narrows), chasing further
    names found in each assignment's RHS up to a small depth. Handles the
    two dominant patterns in this codebase: a module-level constant
    (``OUTPUT_DIR = REPO_ROOT / "artifacts" / "x"``) referenced several hops
    from the call site, and a local rebinding of an argparse namespace
    attribute (``output: Path = arguments.output``).
    """

    if name in seen or depth > 4:
        return ""
    seen.add(name)
    parts: list[str] = []
    for node in ast.walk(tree):
        value = _assign_value_for_name(node, name)
        if value is None:
            continue
        parts.append(_unparse(value))
        for sub in ast.walk(value):
            if isinstance(sub, ast.Name) and sub.id not in seen:
                nested = _resolve_name_text(sub.id, tree, seen, depth + 1)
                if nested:
                    parts.append(nested)
    return " ".join(parts)


def _namespace_attr_classification(
    dest: ast.AST,
    tree: ast.Module,
    namespace_names: frozenset[str],
    arg_defaults: dict[str, str],
) -> WriteClass | None:
    """If ``dest`` references ``<namespace>.<attr>`` for a known
    ``parse_args()`` namespace, classify by that argument's *default*
    (``pool_levers.py``'s ``--out`` defaults straight into
    ``artifacts/pool_levers/``, so it is not a caller-supplied destination in
    the ``tmp_or_arg`` sense even though the call site only says
    ``args.out``) -- falling back to ``tmp_or_arg`` when the flag has no
    governed default (``required=True`` or ``default=None``), since the
    caller must then supply the destination explicitly. Returns ``None`` when
    ``dest`` does not reference a namespace attribute at all.
    """

    for node in ast.walk(dest):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in namespace_names
        ):
            default_text = arg_defaults.get(node.attr, "")
            if default_text:
                literal = _classify_literal(default_text)
                if literal != "unknown":
                    return literal
            return "tmp_or_arg"
    return None


def _classify_destination(
    dest: ast.AST | None,
    tree: ast.Module,
    namespace_names: frozenset[str],
    arg_defaults: dict[str, str] | None = None,
) -> tuple[WriteClass, str]:
    if dest is None:
        return "unknown", "<no destination arg>"
    text = _unparse(dest)
    literal = _classify_literal(text)
    if literal != "unknown":
        return literal, text

    namespace_classification = _namespace_attr_classification(
        dest, tree, namespace_names, arg_defaults or {}
    )
    if namespace_classification is not None:
        return namespace_classification, text

    classification = _classify_text(text, namespace_names)
    if classification != "unknown":
        return classification, text

    if isinstance(dest, ast.Name):
        fallback = _enclosing_function_defaults_text(dest, tree, dest.id)
        if fallback:
            fallback_class = _classify_text(fallback, namespace_names)
            if fallback_class != "unknown":
                return fallback_class, f"{text} (default={fallback})"

    names = {node.id for node in ast.walk(dest) if isinstance(node, ast.Name)}
    resolved_bits = [
        bit for name in sorted(names) if (bit := _resolve_name_text(name, tree, set()))
    ]
    if resolved_bits:
        combined_class = _classify_text(" ".join(resolved_bits), namespace_names)
        if combined_class != "unknown":
            return combined_class, f"{text} (resolved={' '.join(resolved_bits)[:80]})"

    return "unknown", text


def _open_mode_writes(call: ast.Call) -> bool:
    mode_node: ast.AST | None = None
    if len(call.args) >= 2:
        mode_node = call.args[1]
    else:
        for kw in call.keywords:
            if kw.arg == "mode":
                mode_node = kw.value
    if mode_node is None:
        return False  # default mode is "r"
    if not (isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str)):
        return False  # non-literal mode: cannot determine statically, don't guess
    return any(char in mode_node.value for char in _WRITE_MODE_CHARS)


def _resolve_dump_target(call: ast.Call, tree: ast.Module) -> ast.expr | None:
    """``json.dump(obj, fp)`` writes through a file-handle variable, not a
    path literal. Best-effort: if ``fp`` is a plain name assigned from
    ``open(path, ...)`` earlier in the same file, resolve to that path
    expression; otherwise fall back to the handle expression itself (still
    useful for the ``sys.stdout`` case).
    """

    if len(call.args) < 2:
        return None
    handle = call.args[1]
    if not isinstance(handle, ast.Name):
        return handle
    for candidate in ast.walk(tree):
        if (
            isinstance(candidate, ast.Assign)
            and isinstance(candidate.value, ast.Call)
            and isinstance(candidate.value.func, ast.Name)
            and candidate.value.func.id == "open"
            and len(candidate.targets) == 1
            and isinstance(candidate.targets[0], ast.Name)
            and candidate.targets[0].id == handle.id
        ):
            return candidate.value.args[0] if candidate.value.args else handle
    return handle


def _read_only_exceptions_dict_node(stmt: ast.stmt) -> ast.Dict | None:
    """``READ_ONLY_EXCEPTIONS = {...}`` or the annotated
    ``READ_ONLY_EXCEPTIONS: dict[int, str] = {...}`` -- both are common in
    this codebase's style, so both are recognised.
    """

    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target: ast.expr = stmt.targets[0]
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        target = stmt.target
        value = stmt.value
    else:
        return None
    if not (isinstance(target, ast.Name) and target.id == "READ_ONLY_EXCEPTIONS"):
        return None
    return value if isinstance(value, ast.Dict) else None


def _find_read_only_exceptions(tree: ast.Module) -> dict[int, str]:
    exceptions: dict[int, str] = {}
    for stmt in tree.body:
        dict_node = _read_only_exceptions_dict_node(stmt)
        if dict_node is None:
            continue
        for key_node, value_node in zip(dict_node.keys, dict_node.values, strict=True):
            if (
                isinstance(key_node, ast.Constant)
                and isinstance(key_node.value, int)
                and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, str)
            ):
                exceptions[key_node.value] = value_node.value
    return exceptions


def _declares_read_only(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target: ast.expr = stmt.targets[0]
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target = stmt.target
            value = stmt.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == "READ_ONLY_SCRIPT"
            and isinstance(value, ast.Constant)
            and value.value is True
        ):
            return True
    return False


def _calls_provenance_helper(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name in _PROVENANCE_HELPERS:
            return True
    return False


def _imported_origins(tree: ast.Module) -> dict[str, str]:
    """Map each locally-bound name to ``"module.name"`` for every
    ``from module import name [as alias]`` statement in the file -- used to
    resolve a bare call like ``record_bye_edge_fade_challenger_decisions(...)``
    back to the ``STAMPED_LIBRARY_WRITERS`` entry it came from.
    """

    origins: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                origins[local] = f"{node.module}.{alias.name}"
    return origins


def _calls_stamped_library_writer(tree: ast.Module) -> bool:
    targets = {f"{module}.{name}" for module, name in STAMPED_LIBRARY_WRITERS}
    origins = _imported_origins(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and origins.get(node.func.id) in targets
        ):
            return True
    return False


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _base_name(func: ast.AST) -> str | None:
    """For ``a.b(...)``, the name of ``a`` -- used to distinguish e.g.
    ``os.replace`` from an arbitrary object's ``.replace()`` method."""

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id
    return None


def _find_write_sites(tree: ast.Module) -> list[WriteSite]:
    sites: list[WriteSite] = []
    namespace_names = _argparse_namespace_names(tree)
    arg_defaults = _argparse_arg_defaults(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = _call_name(func)
        base = _base_name(func)
        line = node.lineno

        if isinstance(func, ast.Name) and func.id == "open":
            if _open_mode_writes(node):
                dest = node.args[0] if node.args else None
                classification, detail = _classify_destination(
                    dest, tree, namespace_names, arg_defaults
                )
                sites.append(WriteSite(line, "open", classification, detail))
            continue

        if name in _ATOMIC_HELPERS:
            # nfl_ats.io.atomic_*(payload, destination) -- destination is arg[1].
            dest = node.args[1] if len(node.args) >= 2 else None
            classification, detail = _classify_destination(
                dest, tree, namespace_names, arg_defaults
            )
            sites.append(WriteSite(line, name, classification, detail))
            continue

        if isinstance(func, ast.Attribute) and func.attr in _PANDAS_WRITE_METHODS:
            dest = node.args[0] if node.args else None
            classification, detail = _classify_destination(
                dest, tree, namespace_names, arg_defaults
            )
            sites.append(WriteSite(line, f".{func.attr}(", classification, detail))
            continue

        if isinstance(func, ast.Attribute) and func.attr in _PATH_WRITE_METHODS:
            classification, detail = _classify_destination(
                func.value, tree, namespace_names, arg_defaults
            )
            sites.append(WriteSite(line, f".{func.attr}(", classification, detail))
            continue

        if isinstance(func, ast.Attribute) and func.attr == "mkdir":
            classification, detail = _classify_destination(
                func.value, tree, namespace_names, arg_defaults
            )
            sites.append(WriteSite(line, ".mkdir(", classification, detail))
            continue

        if isinstance(func, ast.Attribute) and base == "json" and func.attr == "dump":
            dest = _resolve_dump_target(node, tree)
            classification, detail = _classify_destination(
                dest, tree, namespace_names, arg_defaults
            )
            sites.append(WriteSite(line, "json.dump(", classification, detail))
            continue

        if (
            isinstance(func, ast.Attribute)
            and base == "shutil"
            and (func.attr.startswith("copy") or func.attr == "move")
        ):
            dest = node.args[1] if len(node.args) >= 2 else (node.args[0] if node.args else None)
            classification, detail = _classify_destination(
                dest, tree, namespace_names, arg_defaults
            )
            sites.append(WriteSite(line, f"shutil.{func.attr}(", classification, detail))
            continue

        if isinstance(func, ast.Attribute) and base == "os" and func.attr in ("replace", "rename"):
            dest = node.args[1] if len(node.args) >= 2 else (node.args[0] if node.args else None)
            classification, detail = _classify_destination(
                dest, tree, namespace_names, arg_defaults
            )
            sites.append(WriteSite(line, f"os.{func.attr}(", classification, detail))
            continue

    return sites


def scan_script(path: Path) -> ScriptContract:
    """Parse ``path`` with :mod:`ast` and report its write footprint.

    Never imports or executes ``path`` -- static analysis only.
    """

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    return ScriptContract(
        path=path,
        declares_read_only=_declares_read_only(tree),
        calls_provenance_helper=_calls_provenance_helper(tree),
        write_sites=tuple(_find_write_sites(tree)),
        read_only_exceptions=_find_read_only_exceptions(tree),
        calls_stamped_library_writer=_calls_stamped_library_writer(tree),
    )


def scan_library_writer(module_name: str, function_name: str, src_root: Path) -> bool:
    """True if ``function_name`` inside ``src_root/<module_name path>.py``
    itself calls a provenance-stamping helper (``write_experiment_artifact``/
    ``write_stamped_artifact``/``stamp_sidecar``) somewhere in its own body.

    Backs ``test_script_contracts.py::test_stamped_library_writers_really_stamp``,
    which keeps every ``STAMPED_LIBRARY_WRITERS`` entry an independently
    verified claim rather than a bare declaration a later edit could
    silently invalidate. Same AST-only discipline as :func:`scan_script`:
    never imports ``module_name``. Returns ``False`` (never raises) if the
    module file or the named function cannot be found, so a typo'd entry
    fails the test loudly instead of erroring obscurely.
    """

    path = src_root.joinpath(*module_name.split(".")).with_suffix(".py")
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    func_node = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == function_name
        ),
        None,
    )
    if func_node is None:
        return False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and _call_name(node.func) in _PROVENANCE_HELPERS:
            return True
    return False
