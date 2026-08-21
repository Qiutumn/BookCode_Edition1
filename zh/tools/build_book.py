#!/usr/bin/env python3
"""Deterministic staged build orchestration for the Chinese edition.

Authored builders are imported only for their canonical ``cells`` value.  Every
notebook executes inside a private workspace copied into a fresh staging tree;
non-promoting commands never use a live chapter directory as their working
directory.  A stage carries a digest attestation that ``check`` verifies without
rebuilding it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tomllib
from types import ModuleType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
import uuid

import nbformat

try:
    from . import nb_tools
except ImportError:  # direct execution: python zh/tools/build_book.py
    import nb_tools  # type: ignore[no-redef]


ZH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ZH_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BOOK = ZH_ROOT / "book.toml"
ATTESTATION_NAME = ".zh-stage.json"
ATTESTATION_SCHEMA = 1
BOOK_FIELDS = {"schema_version", "book", "profiles", "units"}
UNIT_FIELDS = {
    "id", "order", "kind", "title", "status", "path", "builder", "manifest",
    "notebook", "org", "assets_dir", "required", "environment", "kernel",
    "timeout", "allow_errors",
}
PROFILE_FIELDS = {"execute", "timeout", "allow_errors", "units", "environment"}
MANIFEST_FIELDS = {
    "schema_version", "unit", "notebook", "execution", "assets",
    # Legacy translation manifests are accepted as a migration input.  New
    # manifests must use the canonical five tables above.
    "outputs", "expectations", "coverage", "licenses",
}
DISALLOWED_PATH_PARTS = {"solution", "solutions", "answer", "answers"}
ENVIRONMENT_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ConfigError(ValueError):
    """Configuration, stage, or safe-path validation failed."""


@dataclass(frozen=True)
class Profile:
    name: str
    execute: bool = False
    timeout: int = 300
    allow_errors: bool = False
    units: tuple[str, ...] = ()
    environments: tuple[str, ...] = ("core",)

    @property
    def environment(self) -> str:
        """Compatibility display value for older callers."""
        return "+".join(self.environments)


@dataclass(frozen=True)
class Unit:
    id: str
    order: int
    kind: str
    title: str
    status: str
    path: Path
    root: Path = ZH_ROOT
    builder: Path | None = None
    notebook: str | None = None
    org: str | None = None
    assets_dir: str = "generated"
    required: bool = False
    environments: tuple[str, ...] = ("core",)
    kernel: str = "python3"
    timeout: int | None = None
    allow_errors: bool | None = None
    manifest: Path | None = None

    @property
    def buildable(self) -> bool:
        return self.status == "available" and self.builder is not None

    @property
    def environment(self) -> str:
        """Compatibility display value for older callers."""
        return "+".join(self.environments)


@dataclass
class BuildResult:
    unit: Unit
    files: list[Path] = field(default_factory=list)
    executed: bool = False
    environments: tuple[str, ...] = ()


BuilderCache = MutableMapping[str, ModuleType]


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration does not exist: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a table: {path}")
    return data


def _check_fields(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{context}: unsupported fields: {', '.join(sorted(unknown))}")


def _strict_bool(value: Any, context: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be boolean")
    return value


def _strict_string(value: Any, context: str, *, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise ConfigError(f"{context} is required")
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _environment_groups(value: Any, context: str, *, default: tuple[str, ...] = ("core",)) -> tuple[str, ...]:
    if value is None:
        groups = list(default)
    elif isinstance(value, str):
        groups = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        groups = list(value)
    else:
        raise ConfigError(f"{context} must be an environment id or an array of ids")
    if not groups:
        raise ConfigError(f"{context} may not be empty")
    result: list[str] = []
    for group in groups:
        if not ENVIRONMENT_ID.fullmatch(group):
            raise ConfigError(f"{context}: invalid environment id {group!r}")
        if group not in result:
            result.append(group)
    return tuple(result)


def _is_solution_part(part: str) -> bool:
    token = part.casefold().replace("-", "_")
    return (
        token in DISALLOWED_PATH_PARTS
        or token.startswith("solution_")
        or token.startswith("answer_")
    )


def resolve_safe_path(
    base: Path,
    value: str | os.PathLike[str],
    *,
    must_exist: bool = False,
    file_only: bool = False,
) -> Path:
    """Resolve a repository-relative path and reject solution/escape paths."""

    raw = Path(value)
    if raw.is_absolute():
        raise ConfigError(f"absolute paths are not allowed in manifests: {value}")
    if any(_is_solution_part(part) for part in raw.parts):
        raise ConfigError(f"solution/answer paths are not valid build inputs: {value}")
    if ".ipynb_checkpoints" in raw.parts:
        raise ConfigError(f"checkpoint paths are not valid build inputs: {value}")
    resolved_base = base.resolve()
    resolved = (resolved_base / raw).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ConfigError(f"path escapes {resolved_base}: {value}") from exc
    if must_exist and not resolved.exists():
        raise ConfigError(f"required path does not exist: {resolved}")
    if file_only and resolved.exists() and not resolved.is_file():
        raise ConfigError(f"expected a file: {resolved}")
    return resolved


def _relative_file_name(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty relative filename")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ConfigError(f"{context} must be a filename without directories")
    if _is_solution_part(path.name):
        raise ConfigError(f"{context} may not name a solution artifact")
    return value


def _manifest_values(path: Path, expected_id: str) -> dict[str, Any]:
    data = _load_toml(path)
    _check_fields(data, MANIFEST_FIELDS, str(path))
    if data.get("schema_version") != 1:
        raise ConfigError(f"{path}: schema_version must be 1")
    unit = data.get("unit", {})
    notebook = data.get("notebook", {})
    outputs = data.get("outputs", {})
    execution = data.get("execution", {})
    assets = data.get("assets", {})
    for name, table in (
        ("unit", unit), ("notebook", notebook), ("outputs", outputs),
        ("execution", execution), ("assets", assets),
    ):
        if not isinstance(table, Mapping):
            raise ConfigError(f"{path}: [{name}] must be a table")
    # Canonical schema. Legacy [outputs] is read only as a migration adapter.
    _check_fields(unit, {"id", "title", "builder", "status", "environment", "source"}, f"{path} [unit]")
    _check_fields(notebook, {"file", "org_file"}, f"{path} [notebook]")
    _check_fields(outputs, {"notebook", "org"}, f"{path} [outputs]")
    _check_fields(execution, {"cwd", "kernel", "timeout", "allow_errors"}, f"{path} [execution]")
    _check_fields(assets, {"directory"}, f"{path} [assets]")
    manifest_id = unit.get("id")
    if manifest_id != expected_id:
        raise ConfigError(
            f"{path}: manifest unit id {manifest_id!r} does not match {expected_id!r}"
        )
    return {
        "title": unit.get("title"),
        "builder": unit.get("builder"),
        "status": unit.get("status"),
        "environment": unit.get("environment"),
        "notebook": notebook.get("file", outputs.get("notebook")),
        "org": notebook.get("org_file", outputs.get("org")),
        "path": execution.get("cwd"),
        "kernel": execution.get("kernel"),
        "timeout": execution.get("timeout"),
        "allow_errors": execution.get("allow_errors"),
        "assets_dir": assets.get("directory"),
    }


def re_full_unit_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value))


def _validate_environment_declarations(book_root: Path, profiles: Mapping[str, Profile], units: Sequence[Unit]) -> None:
    environment_root = book_root / "environments"
    if not environment_root.is_dir():
        return  # standalone fixture books need not ship the repository environments
    declared = {
        group
        for profile in profiles.values()
        for group in profile.environments
    } | {
        group
        for unit in units
        for group in unit.environments
    }
    unknown = [
        group for group in sorted(declared)
        if not (environment_root / f"{group}.lock.txt").is_file()
        and not (environment_root / f"{group}.in").is_file()
    ]
    if unknown:
        raise ConfigError(
            "unknown environment group(s): " + ", ".join(unknown)
        )


def load_book(path: Path = DEFAULT_BOOK) -> tuple[dict[str, Any], dict[str, Profile], list[Unit]]:
    """Load and strictly validate the canonical book and unit schemas."""

    path = path.resolve()
    data = _load_toml(path)
    _check_fields(data, BOOK_FIELDS, str(path))
    if data.get("schema_version") != 1:
        raise ConfigError(f"{path}: schema_version must be 1")
    book = data.get("book", {})
    if not isinstance(book, Mapping):
        raise ConfigError(f"{path}: [book] must be a table")
    _check_fields(book, {"title", "language", "source_license", "code_license"}, f"{path} [book]")

    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ConfigError(f"{path}: [profiles] must define at least one profile")
    profiles: dict[str, Profile] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(name, str) or not re_full_unit_id(name):
            raise ConfigError(f"invalid profile id: {name!r}")
        if not isinstance(raw, Mapping):
            raise ConfigError(f"profile {name}: expected a table")
        _check_fields(raw, PROFILE_FIELDS, f"profile {name}")
        timeout = raw.get("timeout", 300)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
            raise ConfigError(f"profile {name}: timeout must be a positive integer")
        raw_profile_units = raw.get("units", [])
        if not isinstance(raw_profile_units, list) or not all(isinstance(item, str) for item in raw_profile_units):
            raise ConfigError(f"profile {name}: units must be an array of ids")
        profiles[name] = Profile(
            name=name,
            execute=_strict_bool(raw.get("execute"), f"profile {name} execute"),
            timeout=timeout,
            allow_errors=_strict_bool(raw.get("allow_errors"), f"profile {name} allow_errors"),
            units=tuple(raw_profile_units),
            environments=_environment_groups(raw.get("environment"), f"profile {name} environment"),
        )

    raw_units = data.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ConfigError(f"{path}: [[units]] must contain at least one unit")
    units: list[Unit] = []
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, Mapping):
            raise ConfigError(f"unit {index}: expected a table")
        _check_fields(raw, UNIT_FIELDS, f"unit {index}")
        unit_id = raw.get("id")
        order = raw.get("order")
        if not isinstance(unit_id, str) or not re_full_unit_id(unit_id):
            raise ConfigError(f"unit {index}: invalid id {unit_id!r}")
        if unit_id in seen_ids:
            raise ConfigError(f"duplicate unit id: {unit_id}")
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise ConfigError(f"unit {unit_id}: order must be a positive integer")
        if order in seen_orders:
            raise ConfigError(f"duplicate unit order: {order}")
        seen_ids.add(unit_id)
        seen_orders.add(order)

        merged = dict(raw)
        manifest_path: Path | None = None
        if raw.get("manifest") is not None:
            if not isinstance(raw["manifest"], str):
                raise ConfigError(f"unit {unit_id}: manifest must be a relative path")
            manifest_path = resolve_safe_path(path.parent, raw["manifest"], must_exist=True, file_only=True)
            for key, value in _manifest_values(manifest_path, unit_id).items():
                if value is None:
                    continue
                if key in merged and merged[key] != value:
                    raise ConfigError(f"unit {unit_id}: {key} conflicts with {manifest_path}")
                merged[key] = value

        status = _strict_string(merged.get("status"), f"unit {unit_id} status", default="planned")
        if status not in {"available", "planned"}:
            raise ConfigError(f"unit {unit_id}: status must be available or planned")
        unit_path_value = merged.get("path", f"planned/{unit_id}")
        if not isinstance(unit_path_value, str):
            raise ConfigError(f"unit {unit_id}: path must be relative")
        unit_path = resolve_safe_path(path.parent, unit_path_value)
        unit_relative = unit_path.relative_to(path.parent)

        builder: Path | None = None
        builder_value = merged.get("builder")
        if builder_value is not None:
            if not isinstance(builder_value, str):
                raise ConfigError(f"unit {unit_id}: builder must be relative")
            builder = resolve_safe_path(path.parent, builder_value, file_only=True)
        if status == "available":
            if builder is None:
                raise ConfigError(f"unit {unit_id}: available units require a builder")
            if not builder.is_file():
                raise ConfigError(f"unit {unit_id}: builder does not exist: {builder}")
            try:
                builder.relative_to(unit_path)
            except ValueError as exc:
                raise ConfigError(
                    f"unit {unit_id}: builder must live inside the unit source directory"
                ) from exc

        timeout = merged.get("timeout")
        if timeout is not None and (not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1):
            raise ConfigError(f"unit {unit_id}: timeout must be a positive integer")
        allow_errors_raw = merged.get("allow_errors")
        allow_errors = None if allow_errors_raw is None else _strict_bool(
            allow_errors_raw, f"unit {unit_id} allow_errors"
        )
        required = _strict_bool(merged.get("required"), f"unit {unit_id} required")
        kind = _strict_string(merged.get("kind"), f"unit {unit_id} kind", default="chapter")
        title = _strict_string(merged.get("title"), f"unit {unit_id} title", default=unit_id)
        kernel = _strict_string(merged.get("kernel"), f"unit {unit_id} kernel", default="python3")

        units.append(
            Unit(
                id=unit_id,
                order=order,
                kind=kind,
                title=title,
                status=status,
                path=unit_relative,
                root=path.parent,
                builder=builder,
                notebook=_relative_file_name(merged.get("notebook"), f"unit {unit_id} notebook"),
                org=_relative_file_name(merged.get("org"), f"unit {unit_id} org"),
                assets_dir=_relative_file_name(merged.get("assets_dir", "generated"), f"unit {unit_id} assets_dir") or "generated",
                required=required,
                environments=_environment_groups(merged.get("environment"), f"unit {unit_id} environment"),
                kernel=kernel,
                timeout=timeout,
                allow_errors=allow_errors,
                manifest=manifest_path,
            )
        )

    units.sort(key=lambda item: item.order)
    unit_ids = {unit.id for unit in units}
    for profile in profiles.values():
        unknown = set(profile.units) - unit_ids
        if unknown:
            raise ConfigError(f"profile {profile.name}: unknown units: {', '.join(sorted(unknown))}")
    _validate_environment_declarations(path.parent, profiles, units)
    return dict(book), profiles, units


def load_builder(unit: Unit, cache: BuilderCache | None = None) -> ModuleType:
    """Import a builder once under a private name so ``__main__`` never runs."""

    if cache is not None and unit.id in cache:
        return cache[unit.id]
    if unit.builder is None:
        raise ConfigError(f"unit {unit.id} has no builder")
    if any(_is_solution_part(part) for part in unit.builder.parts):
        raise ConfigError(f"unit {unit.id}: solution builders are forbidden")
    module_name = f"_zh_builder_{unit.id.replace('-', '_')}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, unit.builder)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot load builder: {unit.builder}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"failed to load builder {unit.builder}: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
    if not hasattr(module, "cells"):
        raise ConfigError(f"builder {unit.builder} does not define cells")
    try:
        nb_tools.validate_cells(module.cells)
    except Exception as exc:
        raise ConfigError(f"builder {unit.builder} has invalid cells: {exc}") from exc
    if cache is not None:
        cache[unit.id] = module
    return module


def validate_static(
    book_path: Path = DEFAULT_BOOK,
    selected_ids: Sequence[str] = (),
    *,
    builder_cache: BuilderCache | None = None,
) -> tuple[dict[str, Any], dict[str, Profile], list[Unit]]:
    """Validate configuration and canonical cells in disposable source copies."""

    book, profiles, units = load_book(book_path)
    selected = set(selected_ids)
    known = {unit.id for unit in units}
    unknown = selected - known
    if unknown:
        raise ConfigError(f"unknown selected units: {', '.join(sorted(unknown))}")
    candidates: list[Unit] = []
    for unit in units:
        if unit.buildable and (not selected or unit.id in selected):
            if unit.notebook is None or unit.org is None:
                raise ConfigError(f"unit {unit.id}: buildable units require notebook and org names")
            candidates.append(unit)
        elif unit.required and not unit.buildable:
            raise ConfigError(f"required unit {unit.id} is not buildable")

    # Builder top-level code is authored Python and may read or write relative to
    # ``__file__``. Import only disposable copies so even static validation cannot
    # mutate a live chapter directory.
    scratch_parent_value = os.environ.get("CLAUDE_JOB_DIR")
    scratch_parent = Path(scratch_parent_value) / "tmp" if scratch_parent_value else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zh-validate-", dir=scratch_parent) as directory:
        isolated_root = Path(directory)
        for unit in candidates:
            _copy_authority_context(unit, isolated_root)
            workspace = _copy_workspace(unit, isolated_root)
            load_builder(_unit_in_workspace(unit, workspace), builder_cache)
    return book, profiles, units


def select_units(units: Sequence[Unit], profile: Profile, requested: Sequence[str]) -> list[Unit]:
    ids = tuple(requested) or profile.units
    if ids:
        by_id = {unit.id: unit for unit in units}
        try:
            selected = [by_id[unit_id] for unit_id in ids]
        except KeyError as exc:
            raise ConfigError(f"unknown selected unit: {exc.args[0]}") from exc
    else:
        selected = [unit for unit in units if unit.buildable]
    unavailable = [unit.id for unit in selected if not unit.buildable]
    if unavailable:
        raise ConfigError("selected units are planned and have no builder: " + ", ".join(unavailable))
    return selected


def effective_environments(profile: Profile, unit: Unit) -> tuple[str, ...]:
    groups: list[str] = []
    for group in (*profile.environments, *unit.environments):
        if group not in groups:
            groups.append(group)
    return tuple(groups)


def environment_lock_paths(profile: Profile, unit: Unit) -> tuple[Path, ...]:
    paths: list[Path] = []
    unresolved: list[str] = []
    for group in effective_environments(profile, unit):
        lock = unit.root / "environments" / f"{group}.lock.txt"
        if lock.is_file():
            paths.append(lock)
        elif (unit.root / "environments" / f"{group}.in").is_file():
            unresolved.append(group)
        elif (unit.root / "environments").is_dir():
            raise ConfigError(f"unit {unit.id}: unknown environment group {group!r}")
    if unresolved:
        raise ConfigError(
            f"unit {unit.id}: unresolved environment group(s) cannot execute: "
            + ", ".join(unresolved)
        )
    return tuple(paths)


def profile_matrix(profile: Profile, units: Sequence[Unit], requested: Sequence[str] = ()) -> dict[str, Any]:
    include = []
    for unit in select_units(units, profile, requested):
        locks = environment_lock_paths(profile, unit) if profile.execute else ()
        include.append(
            {
                "unit": unit.id,
                "environment": "+".join(effective_environments(profile, unit)),
                "locks": ",".join(str(path.relative_to(unit.root.parent)) for path in locks),
            }
        )
    return {"include": include}


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _validate_stage_location(stage: Path, units: Sequence[Unit]) -> None:
    stage = stage.resolve()
    forbidden: set[Path] = set()
    for unit in units:
        forbidden.add((unit.root / unit.path).resolve())
        if unit.builder is not None:
            forbidden.add(unit.builder.resolve())
        if unit.notebook:
            forbidden.add((unit.root / unit.path / unit.notebook).resolve())
        if unit.org:
            forbidden.add((unit.root / unit.path / unit.org).resolve())
    conflicts = [path for path in sorted(forbidden) if _paths_overlap(stage, path)]
    if conflicts:
        raise ConfigError(
            f"staging path overlaps live source/output path: {stage} <-> {conflicts[0]}"
        )


def _make_stage(requested: Path | None, units: Sequence[Unit], book_root: Path) -> Path:
    if requested is not None:
        requested = requested.expanduser().resolve()
        _validate_stage_location(requested, units)
        if requested.exists():
            raise ConfigError(
                f"staging directory already exists; choose a fresh path: {requested}"
            )
        requested.mkdir(parents=True, exist_ok=False)
        return requested
    build_root = book_root.parent / ".zh-build"
    build_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="staging-", dir=build_root))
    _validate_stage_location(stage, units)
    return stage


def _open_stage(requested: Path | None, units: Sequence[Unit]) -> Path:
    if requested is None:
        raise ConfigError("check requires --staging-dir pointing to an existing stage")
    stage = requested.expanduser().resolve()
    _validate_stage_location(stage, units)
    if not stage.is_dir():
        raise ConfigError(f"staging directory does not exist: {stage}")
    return stage


def _ignored_workspace_entries(unit: Unit):
    root_names = {name for name in (unit.notebook, unit.org, unit.assets_dir) if name}

    def ignore(path: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {"__pycache__", ".ipynb_checkpoints", ".build"}}
        if Path(path).resolve() == (unit.root / unit.path).resolve():
            ignored.update(name for name in names if name in root_names)
        return ignored

    return ignore


def _copy_workspace(unit: Unit, stage_root: Path) -> Path:
    source = (unit.root / unit.path).resolve()
    if not source.is_dir():
        raise ConfigError(f"unit {unit.id}: source directory does not exist: {source}")
    workspace = stage_root / ".workspace" / unit.id
    if workspace.exists():
        raise ConfigError(f"unit {unit.id}: workspace collision: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, workspace, ignore=_ignored_workspace_entries(unit))
    return workspace


def _copy_authority_context(unit: Unit, scratch_root: Path) -> None:
    """Copy repository-level authority files referenced by an isolated builder.

    Chapter workspaces already contain chapter-local data. Builders that verify
    root-level Markdown/notebook provenance derive the repository root from
    ``__file__``; mirror only the named authority files rather than copying or
    symlinking the whole repository.
    """

    if unit.builder is None:
        return
    builder_source = unit.builder.read_text(encoding="utf-8")
    repository_root = unit.root.parent
    candidates: list[Path] = [repository_root / "bibtex.bib"]
    chapter_match = re.fullmatch(r"chapter-(\d+)", unit.id)
    if chapter_match:
        number = int(chapter_match.group(1))
        stem = f"chp_{number:02d}"
        candidates.extend(
            [
                repository_root / "markdown" / f"{stem}.md",
                repository_root / "notebooks_updated" / f"{stem}.ipynb",
                repository_root / "notebooks" / f"{stem}.ipynb",
            ]
        )
        # Some chapter builders display or load a shared repository-root data
        # or figure asset (e.g. a source dataset or illustration) rather than
        # shipping a localized copy. Mirror only files the builder explicitly
        # names, so isolated execution never gains blanket access to the live
        # repository root.
        for directory in (repository_root / "data", repository_root / "markdown" / "figures"):
            if directory.is_dir():
                candidates.extend(entry for entry in directory.iterdir() if entry.is_file())
    for source in candidates:
        if not source.is_file() or source.name not in builder_source:
            continue
        relative = source.relative_to(repository_root)
        destination = scratch_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)


def _unit_in_workspace(unit: Unit, workspace: Path) -> Unit:
    if unit.builder is None:
        raise ConfigError(f"unit {unit.id} has no builder")
    source_root = (unit.root / unit.path).resolve()
    try:
        builder_relative = unit.builder.resolve().relative_to(source_root)
    except ValueError as exc:
        raise ConfigError(
            f"unit {unit.id}: builder must live inside the unit source directory"
        ) from exc
    isolated_builder = workspace / builder_relative
    if not isolated_builder.is_file():
        raise ConfigError(f"unit {unit.id}: isolated builder is missing: {isolated_builder}")
    return replace(unit, builder=isolated_builder)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[Path, str]:
    result: dict[Path, str] = {}
    if not root.is_dir():
        return result
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        result[path.relative_to(root)] = _file_digest(path)
    return result


@contextmanager
def _profile_environment(profile_name: str) -> Iterable[None]:
    previous = os.environ.get("BMCP_EXECUTION_PROFILE")
    os.environ["BMCP_EXECUTION_PROFILE"] = profile_name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("BMCP_EXECUTION_PROFILE", None)
        else:
            os.environ["BMCP_EXECUTION_PROFILE"] = previous


def _execute_notebook(
    notebook_path: Path,
    cwd: Path,
    *,
    kernel: str,
    timeout: int,
    allow_errors: bool,
    profile_name: str,
) -> nbformat.NotebookNode:
    try:
        from nbclient import NotebookClient
    except ImportError as exc:
        raise ConfigError("execution requires nbclient; install the selected environment") from exc
    notebook = nb_tools.read_notebook(notebook_path)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel,
        allow_errors=allow_errors,
        record_timing=True,
    )
    try:
        with _profile_environment(profile_name):
            client.execute(cwd=str(cwd))
    except Exception as exc:
        raise ConfigError(f"execution failed for {notebook_path.name}: {exc}") from exc
    nbformat.write(notebook, str(notebook_path))
    nb_tools.validate_notebook(notebook, require_executed=True, allow_errors=allow_errors)
    return notebook


def _collect_output_files(records: Mapping[str, Sequence[Mapping[str, Any]]], unit_stage: Path) -> list[Path]:
    files: list[Path] = []
    for cell_records in records.values():
        for record in cell_records:
            path_value = record.get("path")
            if not isinstance(path_value, str):
                continue
            candidate = (unit_stage / path_value).resolve()
            try:
                candidate.relative_to(unit_stage.resolve())
            except ValueError as exc:
                raise ConfigError(f"extracted output escapes unit stage: {path_value}") from exc
            if candidate.is_file() and candidate not in files:
                files.append(candidate)
    return files


def _copy_workspace_changes(
    workspace: Path,
    before: Mapping[Path, str],
    unit_stage: Path,
) -> list[Path]:
    changed: list[Path] = []
    after = _tree_snapshot(workspace)
    for relative, digest in after.items():
        if before.get(relative) == digest:
            continue
        source = workspace / relative
        destination = unit_stage / relative
        if destination.exists():
            raise ConfigError(f"workspace output collides with staged artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        changed.append(destination)
    return changed


def build_unit(
    unit: Unit,
    stage_root: Path,
    profile: Profile,
    *,
    execute: bool | None = None,
    builder_cache: BuilderCache | None = None,
) -> BuildResult:
    should_execute = profile.execute if execute is None else execute
    if should_execute:
        environment_lock_paths(profile, unit)
    _copy_authority_context(unit, stage_root)
    workspace = _copy_workspace(unit, stage_root)
    module = load_builder(_unit_in_workspace(unit, workspace), builder_cache)
    workspace_before = _tree_snapshot(workspace)
    unit_stage = stage_root / unit.path
    unit_stage.mkdir(parents=True, exist_ok=True)
    assert unit.notebook is not None and unit.org is not None
    notebook_path = unit_stage / unit.notebook
    org_path = unit_stage / unit.org
    notebook = nb_tools.write_ipynb(module.cells, notebook_path, kernelspec_name=unit.kernel)

    outputs: Mapping[str, Sequence[Mapping[str, Any]]] = {}
    if should_execute:
        # Profile timeout intentionally wins. Unit timeout is only retained as
        # migration metadata for profiles that may omit a timeout in the future.
        allow_errors = unit.allow_errors if unit.allow_errors is not None else profile.allow_errors
        notebook = _execute_notebook(
            notebook_path,
            workspace,
            kernel=unit.kernel,
            timeout=profile.timeout,
            allow_errors=allow_errors,
            profile_name=profile.name,
        )
        outputs = nb_tools.extract_outputs(notebook_path, unit_stage / unit.assets_dir, unit.id)

    nb_tools.write_org(
        module.cells,
        org_path,
        unit.title,
        outputs_by_cell=outputs,
        markdown_converter="internal",
    )
    allow_errors = unit.allow_errors if unit.allow_errors is not None else profile.allow_errors
    nb_tools.validate_notebook(notebook, require_executed=should_execute, allow_errors=allow_errors)
    files = [notebook_path, org_path]
    files.extend(_collect_output_files(outputs, unit_stage))
    if should_execute:
        files.extend(_copy_workspace_changes(workspace, workspace_before, unit_stage))
    unique_files = list(dict.fromkeys(files))
    return BuildResult(
        unit=unit,
        files=unique_files,
        executed=should_execute,
        environments=effective_environments(profile, unit),
    )


def check_stage(results: Sequence[BuildResult], stage_root: Path, profile: Profile) -> None:
    for result in results:
        unit = result.unit
        notebook_path = stage_root / unit.path / (unit.notebook or "")
        org_path = stage_root / unit.path / (unit.org or "")
        if not notebook_path.is_file() or not org_path.is_file():
            raise ConfigError(f"unit {unit.id}: staged notebook or Org output is missing")
        allow_errors = unit.allow_errors if unit.allow_errors is not None else profile.allow_errors
        nb_tools.validate_notebook(
            notebook_path,
            require_executed=result.executed,
            allow_errors=allow_errors,
        )
        org_text = org_path.read_text(encoding="utf-8")
        if not org_text.startswith("#+TITLE:") or "#+LANGUAGE: zh-CN" not in org_text:
            raise ConfigError(f"unit {unit.id}: staged Org metadata is invalid")
        for artifact in result.files:
            if not artifact.is_file():
                raise ConfigError(f"unit {unit.id}: staged artifact is missing: {artifact}")
            try:
                artifact.resolve().relative_to(stage_root.resolve())
            except ValueError as exc:
                raise ConfigError(f"unit {unit.id}: artifact escapes stage: {artifact}") from exc


def _attestation_payload(stage_root: Path, profile: Profile, results: Sequence[BuildResult]) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_SCHEMA,
        "profile": profile.name,
        "units": [
            {
                "id": result.unit.id,
                "executed": result.executed,
                "environments": list(result.environments),
                "files": [
                    {
                        "path": str(path.relative_to(stage_root)),
                        "sha256": _file_digest(path),
                        "size": path.stat().st_size,
                    }
                    for path in sorted(result.files)
                ],
            }
            for result in results
        ],
    }


def write_attestation(stage_root: Path, profile: Profile, results: Sequence[BuildResult]) -> Path:
    target = stage_root / ATTESTATION_NAME
    if target.exists():
        raise ConfigError(f"stage attestation already exists: {target}")
    target.write_text(
        json.dumps(_attestation_payload(stage_root, profile, results), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_attested_results(
    stage_root: Path,
    profile: Profile,
    units: Sequence[Unit],
    requested: Sequence[str],
) -> list[BuildResult]:
    attestation_path = stage_root / ATTESTATION_NAME
    try:
        payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"stage attestation is missing: {attestation_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"stage attestation is invalid JSON: {exc}") from exc
    if payload.get("schema_version") != ATTESTATION_SCHEMA:
        raise ConfigError("unsupported stage attestation schema")
    if payload.get("profile") != profile.name:
        raise ConfigError(
            f"stage profile {payload.get('profile')!r} does not match requested {profile.name!r}"
        )
    raw_units = payload.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ConfigError("stage attestation contains no units")
    by_id = {unit.id: unit for unit in units}
    attested_ids = [item.get("id") for item in raw_units if isinstance(item, Mapping)]
    if requested and list(requested) != attested_ids:
        raise ConfigError(
            "requested units do not exactly match attested stage units: "
            f"requested={list(requested)!r}, attested={attested_ids!r}"
        )
    results: list[BuildResult] = []
    for item in raw_units:
        if not isinstance(item, Mapping):
            raise ConfigError("invalid unit entry in stage attestation")
        unit_id = item.get("id")
        if unit_id not in by_id:
            raise ConfigError(f"attested unit is not in current book: {unit_id!r}")
        unit = by_id[unit_id]
        expected_env = effective_environments(profile, unit)
        if tuple(item.get("environments", ())) != expected_env:
            raise ConfigError(f"unit {unit.id}: attested environments no longer match configuration")
        raw_files = item.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ConfigError(f"unit {unit.id}: attestation contains no files")
        files: list[Path] = []
        for record in raw_files:
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                raise ConfigError(f"unit {unit.id}: invalid attested file record")
            candidate = (stage_root / record["path"]).resolve()
            try:
                candidate.relative_to(stage_root.resolve())
            except ValueError as exc:
                raise ConfigError(f"unit {unit.id}: attested path escapes stage") from exc
            if not candidate.is_file():
                raise ConfigError(f"unit {unit.id}: attested file is missing: {candidate}")
            if candidate.stat().st_size != record.get("size") or _file_digest(candidate) != record.get("sha256"):
                raise ConfigError(f"unit {unit.id}: staged file changed after attestation: {candidate}")
            files.append(candidate)
        results.append(
            BuildResult(
                unit=unit,
                files=files,
                executed=_strict_bool(item.get("executed"), f"attested unit {unit.id} executed"),
                environments=expected_env,
            )
        )
    check_stage(results, stage_root, profile)
    return results


def _promotion_map(results: Sequence[BuildResult], stage_root: Path) -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    for result in results:
        destination_root = result.unit.root.resolve()
        for source in result.files:
            relative = source.resolve().relative_to(stage_root.resolve())
            destination = (destination_root / relative).resolve()
            try:
                destination.relative_to(destination_root)
            except ValueError as exc:
                raise ConfigError(f"promotion path escapes configured root: {destination}") from exc
            if destination in mapping:
                raise ConfigError(f"multiple staged files target the same destination: {destination}")
            mapping[destination] = source
    return mapping


def _stale_generated_files(results: Sequence[BuildResult], mapping: Mapping[Path, Path]) -> list[Path]:
    managed_dirs: set[Path] = {
        (result.unit.root / result.unit.path / result.unit.assets_dir).resolve()
        for result in results
    }
    staged_destinations = set(mapping)
    stale: list[Path] = []
    for directory in managed_dirs:
        if not directory.is_dir():
            continue
        stale.extend(
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path.resolve() not in staged_destinations
        )
    return sorted(set(stale))


def atomic_promote(results: Sequence[BuildResult], stage_root: Path) -> list[Path]:
    """Transactionally replace attested files; never silently prune stale files.

    Stale managed outputs abort before the first write.  Removing those files is
    an explicit, separate user-authorized operation and is intentionally outside
    this function.
    """

    mapping = _promotion_map(results, stage_root)
    stale = _stale_generated_files(results, mapping)
    if stale:
        formatted = "\n  - ".join(str(path) for path in stale)
        raise ConfigError(
            "promotion refused because stale generated files require explicit pruning authorization:\n  - "
            + formatted
        )
    for destination, source in mapping.items():
        if not source.is_file():
            raise ConfigError(f"promotion source is missing: {source}")
        if destination.exists() and not destination.is_file():
            raise ConfigError(f"promotion destination is not a file: {destination}")

    prepared: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination, source in mapping.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.new-{uuid.uuid4().hex}")
            shutil.copy2(source, temporary)
            prepared[destination] = temporary
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
                shutil.copy2(destination, backup)
                backups[destination] = backup
        for destination, temporary in prepared.items():
            os.replace(temporary, destination)
            installed.append(destination)
    except Exception:
        for destination in reversed(installed):
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            elif destination.exists():
                destination.unlink()
        raise
    finally:
        for temporary in prepared.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    return installed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "matrix", "build", "check", "release"))
    parser.add_argument("--book", type=Path, default=DEFAULT_BOOK)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--unit", action="append", default=[], dest="units")
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override execution for build only; release always executes",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="release only: transactionally install the checked stage",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="compatibility no-op: stages are always retained for independent checking",
    )
    return parser


def _default_profile(command: str) -> str:
    if command == "release":
        return "release"
    if command == "matrix":
        return "smoke"
    return "static"


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        builder_cache: dict[str, ModuleType] = {}
        if args.command == "validate":
            if args.promote or args.staging_dir or args.execute is not None:
                raise ConfigError("validate does not accept staging, execution, or promotion options")
            _, profiles, units = validate_static(args.book, args.units, builder_cache=builder_cache)
            print(f"validated {len(units)} units and {len(profiles)} profiles")
            return 0

        _, profiles, units = load_book(args.book)
        profile_name = args.profile or _default_profile(args.command)
        if profile_name not in profiles:
            raise ConfigError(f"unknown profile: {profile_name}")
        profile = profiles[profile_name]

        if args.command == "matrix":
            if args.promote or args.staging_dir or args.execute is not None:
                raise ConfigError("matrix does not accept staging, execution, or promotion options")
            print(json.dumps(profile_matrix(profile, units, args.units), separators=(",", ":")))
            return 0

        selected = select_units(units, profile, args.units)

        if args.command == "check":
            if args.promote:
                raise ConfigError("check never promotes; use release --promote")
            if args.execute is not None:
                raise ConfigError("check validates an existing stage and does not accept --execute")
            stage_root = _open_stage(args.staging_dir, selected)
            results = load_attested_results(stage_root, profile, units, args.units)
            print(f"checked {len(results)} attested unit(s) in {stage_root}")
            return 0

        if args.command == "build" and args.promote:
            raise ConfigError("build never promotes; use release --promote")
        if args.command == "release":
            if profile.name != "release":
                raise ConfigError("release command requires --profile release")
            if args.execute is False:
                raise ConfigError("release cannot use --no-execute")
            execute = True
        else:
            execute = args.execute

        # Reject an unsafe/reused caller stage before importing any authored
        # builder. Then validate canonical cells from disposable source copies and
        # reuse the resulting in-memory modules during generation.
        stage_root = _make_stage(args.staging_dir, selected, args.book.resolve().parent)
        validate_static(args.book, [unit.id for unit in selected], builder_cache=builder_cache)
        results: list[BuildResult] = []
        for unit in selected:
            result = build_unit(
                unit,
                stage_root,
                profile,
                execute=execute,
                builder_cache=builder_cache,
            )
            results.append(result)
            mode = "executed" if result.executed else "generated"
            print(f"{unit.id}: {mode} in {stage_root / unit.path}")
        check_stage(results, stage_root, profile)
        write_attestation(stage_root, profile, results)
        print(f"attested {len(results)} staged unit(s) in {stage_root}")

        if args.command == "release" and args.promote:
            # Reload from the attestation so promotion cannot bypass the same
            # integrity path used by the standalone check command.
            attested = load_attested_results(stage_root, profile, units, args.units)
            promoted = atomic_promote(attested, stage_root)
            print(f"promoted {len(promoted)} generated file(s)")
        elif args.command == "release":
            print("promotion skipped (pass --promote to install release artifacts)")
        return 0
    except (
        ConfigError,
        nb_tools.CellValidationError,
        nb_tools.NotebookValidationError,
        nb_tools.UnsupportedMarkdownError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
