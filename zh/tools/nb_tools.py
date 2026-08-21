"""Notebook and Org build helpers for the Chinese edition.

Chapter builders may continue to use the original two-field cell dictionaries::

    {"type": "markdown", "source": "## 标题"}
    {"type": "code", "source": "print('你好')"}

Stable ``id`` and ``metadata`` fields are optional.  When an id is omitted, a
content-derived id is assigned; inserting an unrelated cell therefore does not
rename outputs from existing cells.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


CELL_TYPES = {"markdown", "code"}
CELL_FIELDS = {"type", "source", "id", "metadata"}
SUPPORTED_IMAGE_MIMES = {
    "image/png": ("png", True),
    "image/svg+xml": ("svg", False),
    "image/jpeg": ("jpg", True),
}
SUPPORTED_DATA_MIMES = (*SUPPORTED_IMAGE_MIMES, "text/plain")
_PANDOC_ENV = "ZH_PANDOC"
_PANDOC_VERSION_ENV = "ZH_PANDOC_VERSION"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAFE_ORG_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_RAW_LINE_PREFIX = "\x00RAWLINE\x00"
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")
_MATH_LABEL_PREFIX = "ZH_MATH_LABEL:"


class CellValidationError(ValueError):
    """An authored cell dictionary is malformed."""


class NotebookValidationError(ValueError):
    """A notebook is invalid for the Chinese-edition build."""


class UnsupportedMarkdownError(ValueError):
    """The internal converter cannot safely represent a Markdown construct."""


def _ensure_parent(path: os.PathLike[str] | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _source_text(source: Any) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        return "".join(source)
    raise CellValidationError("cell source must be a string")


def _content_cell_id(cell_type: str, source: str) -> str:
    digest = hashlib.sha256(f"{cell_type}\0{source}".encode("utf-8")).hexdigest()[:12]
    prefix = "md" if cell_type == "markdown" else "code"
    return f"{prefix}-{digest}"


def normalize_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy authored cells, assigning deterministic unique ids.

    The legacy two-field dictionaries remain valid.  Duplicate content receives
    a deterministic occurrence suffix (``-2``, ``-3``) rather than a raw cell
    index, while explicit ids must already be unique.
    """

    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise CellValidationError("cells must be a sequence of mappings")

    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    implicit_counts: dict[str, int] = {}
    for index, raw in enumerate(cells):
        if not isinstance(raw, Mapping):
            raise CellValidationError(f"cell {index}: expected a mapping")
        unknown = set(raw) - CELL_FIELDS
        if unknown:
            raise CellValidationError(
                f"cell {index}: unsupported fields: {', '.join(sorted(unknown))}"
            )
        cell_type = raw.get("type")
        if cell_type not in CELL_TYPES:
            raise CellValidationError(
                f"cell {index}: type must be one of {sorted(CELL_TYPES)}, got {cell_type!r}"
            )
        if "source" not in raw:
            raise CellValidationError(f"cell {index}: missing source")
        source = _source_text(raw["source"])
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CellValidationError(f"cell {index}: metadata must be a mapping")

        explicit_id = raw.get("id")
        if explicit_id is not None:
            if not isinstance(explicit_id, str) or not _SAFE_ID.fullmatch(explicit_id):
                raise CellValidationError(
                    f"cell {index}: id must match {_SAFE_ID.pattern!r}"
                )
            cell_id = explicit_id
        else:
            base_id = _content_cell_id(cell_type, source)
            occurrence = implicit_counts.get(base_id, 0) + 1
            implicit_counts[base_id] = occurrence
            suffix = "" if occurrence == 1 else f"-{occurrence}"
            cell_id = f"{base_id}{suffix}"

        if cell_id in used_ids:
            raise CellValidationError(f"cell {index}: duplicate id {cell_id!r}")
        used_ids.add(cell_id)
        normalized.append(
            {
                "type": cell_type,
                "source": source,
                "id": cell_id,
                "metadata": dict(metadata),
            }
        )
    return normalized


def validate_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate authored cells and return their normalized representation."""

    return normalize_cells(cells)


def cells_to_notebook(
    cells: Sequence[Mapping[str, Any]], kernelspec_name: str = "python3"
) -> nbformat.NotebookNode:
    """Build an in-memory notebook from authored cells."""

    normalized = normalize_cells(cells)
    nb = new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": kernelspec_name,
        },
        "language_info": {"name": "python", "version": "3"},
    }
    for cell in normalized:
        kwargs = {"id": cell["id"], "metadata": cell["metadata"]}
        if cell["type"] == "markdown":
            nb["cells"].append(new_markdown_cell(cell["source"], **kwargs))
        else:
            nb["cells"].append(new_code_cell(cell["source"], **kwargs))
    validate_notebook(nb)
    return nb


def write_ipynb(
    cells: Sequence[Mapping[str, Any]],
    path: os.PathLike[str] | str,
    kernelspec_name: str = "python3",
) -> nbformat.NotebookNode:
    """Write a valid v4 notebook and return the in-memory notebook."""

    nb = cells_to_notebook(cells, kernelspec_name=kernelspec_name)
    _ensure_parent(path)
    nbformat.write(nb, str(path))
    return nb


def _assign_missing_cell_ids(nb: nbformat.NotebookNode) -> None:
    """Assign deterministic in-memory IDs to legacy pre-nbformat-4.5 cells.

    Existing valid IDs are preserved.  Invalid or duplicate explicit IDs remain
    errors; only a genuinely missing/empty ID is migrated.
    """

    used = {
        cell.get("id")
        for cell in nb.cells
        if isinstance(cell.get("id"), str) and cell.get("id")
    }
    counts: dict[str, int] = {}
    for cell in nb.cells:
        if isinstance(cell.get("id"), str) and cell.get("id"):
            continue
        cell_type = str(cell.get("cell_type", "raw"))
        source = _source_text(cell.get("source", ""))
        base = _content_cell_id(cell_type, source)
        occurrence = counts.get(base, 0) + 1
        counts[base] = occurrence
        candidate = base if occurrence == 1 else f"{base}-{occurrence}"
        while candidate in used:
            occurrence += 1
            counts[base] = occurrence
            candidate = f"{base}-{occurrence}"
        cell["id"] = candidate
        used.add(candidate)


def read_notebook(path: os.PathLike[str] | str) -> nbformat.NotebookNode:
    with Path(path).open(encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)
    _assign_missing_cell_ids(nb)
    return nb


def validate_notebook(
    notebook: nbformat.NotebookNode | Mapping[str, Any] | os.PathLike[str] | str,
    *,
    require_executed: bool = False,
    allow_errors: bool = False,
) -> nbformat.NotebookNode:
    """Validate nbformat plus build-specific ids and execution invariants.

    ``require_executed`` requires every non-empty code cell to have a positive
    integer execution count.  Error outputs are rejected unless
    ``allow_errors`` is explicitly enabled.
    """

    if isinstance(notebook, (str, os.PathLike)):
        nb = read_notebook(notebook)
    elif isinstance(notebook, nbformat.NotebookNode):
        nb = notebook
    else:
        nb = nbformat.from_dict(dict(notebook))

    _assign_missing_cell_ids(nb)
    try:
        nbformat.validate(nb)
    except Exception as exc:  # nbformat exposes several validator backends
        raise NotebookValidationError(f"nbformat validation failed: {exc}") from exc

    ids: set[str] = set()
    counts: list[int] = []
    for index, cell in enumerate(nb.cells):
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not _SAFE_ID.fullmatch(cell_id):
            raise NotebookValidationError(f"cell {index}: missing or invalid stable id")
        if cell_id in ids:
            raise NotebookValidationError(f"cell {index}: duplicate id {cell_id!r}")
        ids.add(cell_id)

        if cell.cell_type != "code":
            continue
        count = cell.get("execution_count")
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
            raise NotebookValidationError(
                f"cell {cell_id}: execution_count must be null or a positive integer"
            )
        if require_executed and cell.source.strip():
            if count is None:
                raise NotebookValidationError(f"cell {cell_id}: code was not executed")
            counts.append(count)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error" and not allow_errors:
                name = output.get("ename", "Error")
                value = output.get("evalue", "")
                raise NotebookValidationError(f"cell {cell_id}: {name}: {value}".rstrip())

    if require_executed and counts:
        if len(set(counts)) != len(counts):
            raise NotebookValidationError(
                "executed code cells contain duplicate execution counts"
            )
        if counts != sorted(counts):
            raise NotebookValidationError(
                "execution counts are not monotonically increasing in notebook order"
            )
    return nb


def find_pandoc() -> Path | None:
    """Return an explicitly configured, version-pinned Pandoc executable.

    Host paths are intentionally never auto-discovered: the same source must not
    switch converters merely because one machine happens to have Pandoc.
    """

    configured = os.environ.get(_PANDOC_ENV)
    if not configured:
        return None
    candidate = Path(configured)
    if not candidate.is_absolute():
        raise ValueError(f"{_PANDOC_ENV} must be an absolute path")
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"configured Pandoc is not executable: {candidate}")
    expected = os.environ.get(_PANDOC_VERSION_ENV)
    if not expected:
        raise ValueError(
            f"{_PANDOC_VERSION_ENV} must pin the exact Pandoc version when {_PANDOC_ENV} is set"
        )
    completed = subprocess.run(
        [str(candidate), "--version"], capture_output=True, text=True, check=False
    )
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    actual = first_line.removeprefix("pandoc ").strip()
    if completed.returncode or actual != expected:
        raise RuntimeError(
            f"configured Pandoc version mismatch: expected {expected!r}, got {actual!r}"
        )
    return candidate


def _pandoc_md_to_org(text: str, executable: Path) -> str:
    command = [
        str(executable),
        "--from=gfm+tex_math_dollars+footnotes+fenced_code_attributes",
        "--to=org",
        "--wrap=none",
    ]
    completed = subprocess.run(
        command,
        input=text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Pandoc conversion failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    return completed.stdout.rstrip("\n")


def md_to_org(text: str, *, converter: str = "auto") -> str:
    """Convert controlled Markdown to Org.

    Production's ``auto`` mode is an alias for the deterministic internal
    converter and therefore never changes with host Pandoc availability.
    Explicit ``pandoc`` mode requires a pinned absolute executable. Unsupported
    internal syntax raises instead of being emitted in a corrupted form.
    """

    if converter not in {"auto", "internal", "pandoc"}:
        raise ValueError("converter must be 'auto', 'internal', or 'pandoc'")
    # ``auto`` is deliberately an alias for the deterministic internal path.
    # Pandoc is used only when the caller explicitly requests it and pins both
    # executable and exact version through the environment.
    if converter == "pandoc":
        pandoc = find_pandoc()
        if pandoc is None:
            raise FileNotFoundError(
                f"Pandoc mode requires {_PANDOC_ENV} and {_PANDOC_VERSION_ENV}"
            )
        return _pandoc_md_to_org(text, pandoc)
    return _internal_md_to_org(text)


def _unsupported_markdown(line: str, line_number: int) -> None:
    # These constructs require a full parser.  Raising is safer than producing
    # plausible-looking but semantically wrong Org.
    without_autolinks = re.sub(
        r"<(?:[A-Za-z][A-Za-z0-9+.-]*://[^<>\s]+|[^<>\s@]+@[^<>\s@]+)>",
        "",
        line,
    )
    # Inline math and code spans routinely contain bare `<`/`>` comparisons
    # (e.g. "$X_1<c_1$ 且 $X_2>c_2$"); strip them first so a `<` opening one
    # span and a `>` closing an unrelated later span cannot be misread as a
    # single HTML tag spanning both.
    without_math_and_code = re.sub(r"`[^`\n]*`", "", without_autolinks)
    without_math_and_code = re.sub(r"\$[^$\n]+\$", "", without_math_and_code)
    if re.search(r"<[/!?A-Za-z][^>]*>", without_math_and_code):
        raise UnsupportedMarkdownError(
            f"line {line_number}: raw HTML is unsupported without Pandoc"
        )
    # Recognized GFM pipe tables are lowered by ``_preprocess_gfm_tables`` before
    # this per-line pass runs; any ``|`` reaching here is ordinary text content.
    if re.match(r"^\s*[-*+]\s+\[[ xX]\]\s+", line):
        raise UnsupportedMarkdownError(
            f"line {line_number}: task lists are unsupported without Pandoc"
        )


def _normalize_multiline_inline_code(text: str) -> str:
    """Apply CommonMark whitespace normalization to soft-wrapped code spans.

    Protect complete single-line spans first. Otherwise a closing backtick from a
    MyST role can be mistaken for the opening of a multiline code span and collapse
    whole directives or paragraphs between two unrelated roles.
    """

    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"{len(protected) - 1}"

    text = re.sub(r"(?<!`)`(?!`)[^`\n]*?(?<!`)`(?!`)", protect, text)
    pattern = re.compile(r"(?<!`)`(?!`)([^`]*?\n[^`]*?)`(?!`)")
    while True:
        match = pattern.search(text)
        if match is None:
            break
        body = re.sub(r"[ \t]*\n[ \t]*", " ", match.group(1))
        text = text[: match.start()] + f"`{body}`" + text[match.end() :]
    return re.sub(
        r"(\d+)",
        lambda match: protected[int(match.group(1))],
        text,
    )


def _preprocess_setext_headings(lines: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(lines):
        if (
            index + 1 < len(lines)
            and lines[index].strip()
            and not lines[index].lstrip().startswith((">", "#", "```", "~~~"))
            and re.fullmatch(r"\s*(=+|-+)\s*", lines[index + 1])
        ):
            marker = lines[index + 1].strip()[0]
            result.append(("# " if marker == "=" else "## ") + lines[index].strip())
            index += 2
            continue
        result.append(lines[index])
        index += 1
    return result


def _preprocess_myst_math_fences(lines: list[str]) -> list[str]:
    """Lower fenced MyST math blocks without treating them as source code."""

    result: list[str] = []
    index = 0
    while index < len(lines):
        opening = re.fullmatch(r"\s*(`{3,}|~{3,})\{math\}\s*", lines[index])
        if opening is None:
            result.append(lines[index])
            index += 1
            continue
        marker = opening.group(1)
        end = index + 1
        closing = rf"\s*{re.escape(marker[0])}{{{len(marker)},}}\s*"
        while end < len(lines) and not re.fullmatch(closing, lines[end]):
            end += 1
        if end >= len(lines):
            raise UnsupportedMarkdownError(
                f"line {index + 1}: unterminated MyST math fence"
            )
        body = list(lines[index + 1 : end])
        options: dict[str, str] = {}
        while body and re.match(r"^:[A-Za-z0-9_-]+:\s*", body[0]):
            option = re.match(r"^:([A-Za-z0-9_-]+):\s*(.*)$", body.pop(0))
            assert option is not None
            options[option.group(1)] = option.group(2)
        unknown = set(options) - {"label", "name"}
        if unknown:
            raise UnsupportedMarkdownError(
                f"line {index + 1}: unsupported MyST math option(s): "
                + ", ".join(sorted(unknown))
            )
        label = options.get("label") or options.get("name")
        if label:
            if not _SAFE_ORG_NAME.fullmatch(label):
                raise UnsupportedMarkdownError(
                    f"line {index + 1}: invalid MyST math label {label!r}"
                )
            result.append(_MATH_LABEL_PREFIX + label)
        result.append("$$")
        result.extend(body)
        result.append("$$")
        index = end + 1
    return result


def _parse_list_table_rows(body: list[str]) -> list[list[str]]:
    """Parse an RST/MyST ``list-table`` body into rows of cell text.

    Each row begins with ``* -`` (optionally followed by its first cell's
    text on the same line); subsequent more-indented ``- `` lines are the
    row's remaining cells.
    """

    rows: list[list[str]] = []
    index = 0
    while index < len(body):
        line = body[index]
        if not line.strip():
            index += 1
            continue
        row_match = re.match(r"^\*\s*-\s?(.*)$", line)
        if row_match is None:
            raise UnsupportedMarkdownError(
                "list-table row must start with '* -'"
            )
        cells = [row_match.group(1).strip()]
        index += 1
        while index < len(body):
            cell_match = re.match(r"^\s+-\s?(.*)$", body[index])
            if cell_match is None:
                break
            cells.append(cell_match.group(1).strip())
            index += 1
        rows.append(cells)
    return rows


_FENCED_DIRECTIVE_NAMES = ("figure", "code-block", "list-table", "epigraph")


def _preprocess_myst_fenced_directives(lines: list[str]) -> list[str]:
    """Lower fenced MyST figure/code-block/list-table/epigraph blocks.

    These directives are written with the same triple-backtick fence as
    ordinary source code (``` ```{figure} path.png ``` ```), so without this
    pass the generic fenced-code handling below would mistake the directive
    name for a programming-language info string and silently turn a figure,
    code sample, table, or epigraph into a bogus, mislabeled source block.
    """

    result: list[str] = []
    index = 0
    while index < len(lines):
        opening = re.match(
            r"^\s*(`{3,}|~{3,})\{(" + "|".join(_FENCED_DIRECTIVE_NAMES) + r")\}\s*(.*)$",
            lines[index],
        )
        if opening is None:
            result.append(lines[index])
            index += 1
            continue
        marker, directive, argument = opening.groups()
        end = index + 1
        closing = rf"\s*{re.escape(marker[0])}{{{len(marker)},}}\s*"
        while end < len(lines) and not re.fullmatch(closing, lines[end]):
            end += 1
        if end >= len(lines):
            raise UnsupportedMarkdownError(
                f"line {index + 1}: unterminated MyST {directive!r} directive"
            )
        body = list(lines[index + 1 : end])
        options: dict[str, str] = {}
        while body and re.match(r"^:[A-Za-z0-9_-]+:\s*", body[0]):
            option = re.match(r"^:([A-Za-z0-9_-]+):\s*(.*)$", body.pop(0))
            assert option is not None
            options[option.group(1)] = option.group(2)
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()

        def require_label(allowed: set[str]) -> str | None:
            unknown = set(options) - allowed
            if unknown:
                raise UnsupportedMarkdownError(
                    f"line {index + 1}: unsupported MyST {directive!r} option(s): "
                    + ", ".join(sorted(unknown))
                )
            label = options.get("name") or options.get("label")
            if label and not _SAFE_ORG_NAME.fullmatch(label):
                raise UnsupportedMarkdownError(
                    f"line {index + 1}: invalid MyST {directive!r} label {label!r}"
                )
            return label

        rendered: list[str] = []
        if directive == "figure":
            label = require_label({"name", "label", "alt", "width", "height", "align"})
            target = argument.strip()
            if not target:
                raise UnsupportedMarkdownError(
                    f"line {index + 1}: MyST figure directive is missing an image path"
                )
            prefix = "" if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) else "file:"
            caption = _inline_md_to_org(
                " ".join(part.strip() for part in body if part.strip())
            )
            if label:
                rendered.extend([f"<<{label}>>", f"#+NAME: {label}"])
            if caption:
                rendered.append(f"#+CAPTION: {caption}")
            rendered.append(f"[[{prefix}{target}]]")
        elif directive == "code-block":
            label = require_label(
                {"name", "label", "caption", "lineno-start", "emphasize-lines", "linenos"}
            )
            language = argument.strip() or "text"
            if not re.fullmatch(r"[A-Za-z0-9_+.-]+", language):
                raise UnsupportedMarkdownError(
                    f"line {index + 1}: unsupported code-block language {language!r}"
                )
            if label:
                rendered.extend([f"<<{label}>>", f"#+NAME: {label}"])
            rendered.append(f"#+BEGIN_SRC {language}")
            rendered.extend(body)
            rendered.append("#+END_SRC")
        elif directive == "list-table":
            label = require_label({"name", "label", "header-rows", "width", "align"})
            caption = argument.strip()
            table_rows = _parse_list_table_rows(body)
            if label:
                rendered.append(f"<<{label}>>")
            if caption:
                rendered.append(f"#+CAPTION: {_inline_md_to_org(caption)}")
            header_rows = int(options.get("header-rows") or "0")
            for row_index, row_cells in enumerate(table_rows):
                rendered.append(
                    "|" + "|".join(f" {_inline_md_to_org(cell)} " for cell in row_cells) + "|"
                )
                if header_rows and row_index == header_rows - 1:
                    rendered.append(
                        "|" + "+".join("-" * 3 for _ in row_cells) + "|"
                    )
        else:  # epigraph
            if options:
                raise UnsupportedMarkdownError(
                    f"line {index + 1}: unsupported MyST epigraph option(s): "
                    + ", ".join(sorted(options))
                )
            rendered.append("#+BEGIN_QUOTE")
            for body_line in body:
                attribution = re.match(r"^-{2,3}\s*(.*)$", body_line)
                if attribution:
                    rendered.append(f"--- /{_inline_md_to_org(attribution.group(1))}/")
                elif body_line.strip():
                    rendered.append(_inline_md_to_org(body_line))
                else:
                    rendered.append("")
            rendered.append("#+END_QUOTE")

        result.extend(_RAW_LINE_PREFIX + line for line in rendered)
        index = end + 1
    return result


def _preprocess_myst_directives(lines: list[str]) -> list[str]:
    """Lower controlled MyST directives to deterministic Org-safe forms."""

    result: list[str] = []
    index = 0
    while index < len(lines):
        opening = re.match(r"^\s*(:{3,})\{([^}]+)\}\s*(.*)$", lines[index])
        if opening is None:
            result.append(lines[index])
            index += 1
            continue
        marker, directive, argument = opening.groups()
        end = index + 1
        while end < len(lines) and not re.fullmatch(
            rf"\s*{re.escape(marker)}\s*", lines[end]
        ):
            end += 1
        if end >= len(lines):
            raise UnsupportedMarkdownError(
                f"line {index + 1}: unterminated MyST {directive!r} directive"
            )
        body = lines[index + 1 : end]
        options: dict[str, str] = {}
        while body and re.match(r"^:[A-Za-z0-9_-]+:\s*", body[0]):
            option = re.match(r"^:([A-Za-z0-9_-]+):\s*(.*)$", body.pop(0))
            assert option is not None
            options[option.group(1)] = option.group(2)
        while body and not body[0].strip():
            body.pop(0)
        directive_key = directive.casefold()
        if directive_key == "math":
            unknown = set(options) - {"label", "name"}
            if unknown:
                raise UnsupportedMarkdownError(
                    f"line {index + 1}: unsupported MyST math option(s): "
                    + ", ".join(sorted(unknown))
                )
            label = options.get("label") or options.get("name") or argument.strip()
            if label:
                if not _SAFE_ORG_NAME.fullmatch(label):
                    raise UnsupportedMarkdownError(
                        f"line {index + 1}: invalid MyST math label {label!r}"
                    )
                result.append(_MATH_LABEL_PREFIX + label)
            result.append("$$")
            result.extend(body)
            result.append("$$")
            index = end + 1
            continue
        if directive_key not in {
            "admonition", "note", "warning", "tip", "important", "caution"
        }:
            raise UnsupportedMarkdownError(
                f"line {index + 1}: MyST directive {directive!r} is unsupported internally"
            )
        callout = options.get("class", directive if directive_key != "admonition" else "note")
        callout = re.split(r"\s+", callout.strip())[0].upper() or "NOTE"
        title = argument.strip() or directive.replace("-", " ").title()
        result.append(f"> [!{callout}] {title}")
        result.extend(">" if not item else f"> {item}" for item in body)
        index = end + 1
    return result


def _split_table_row(line: str) -> list[str]:
    """Split one GFM pipe-table row into its cell contents.

    Controlled table content has no literal unescaped ``|`` inside a cell, so a
    direct split is safe; content requiring cell-internal pipes needs Pandoc.
    """

    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _preprocess_gfm_tables(lines: list[str]) -> list[str]:
    """Lower controlled GFM pipe tables to pre-rendered Org table rows.

    Org's pipe-table syntax is a near match for GFM's, so each row is
    reformatted rather than rejected. Rendered rows are marked with
    ``_RAW_LINE_PREFIX`` so the main pass copies them through verbatim instead
    of re-parsing already-converted Org syntax as Markdown.
    """

    result: list[str] = []
    index = 0
    while index < len(lines):
        header = lines[index]
        separator = lines[index + 1] if index + 1 < len(lines) else ""
        if "|" not in header or not _TABLE_SEPARATOR.fullmatch(separator):
            result.append(header)
            index += 1
            continue
        header_cells = _split_table_row(header)
        column_count = len(header_cells)
        rendered = [
            "|" + "|".join(f" {_inline_md_to_org(cell)} " for cell in header_cells) + "|",
            "|" + "+".join("-" * 3 for _ in range(column_count)) + "|",
        ]
        row_index = index + 2
        while row_index < len(lines):
            row = lines[row_index]
            if not row.strip() or "|" not in row:
                break
            cells = _split_table_row(row)
            rendered.append(
                "|" + "|".join(f" {_inline_md_to_org(cell)} " for cell in cells) + "|"
            )
            row_index += 1
        result.extend(_RAW_LINE_PREFIX + line for line in rendered)
        index = row_index
    return result


def _internal_md_to_org(text: str) -> str:
    text = _normalize_multiline_inline_code(text)
    lines = _preprocess_setext_headings(text.splitlines())
    lines = _preprocess_myst_math_fences(lines)
    lines = _preprocess_myst_fenced_directives(lines)
    lines = _preprocess_myst_directives(lines)
    lines = _preprocess_gfm_tables(lines)
    out: list[str] = []
    in_quote = False
    quote_callout: str | None = None
    fence: tuple[str, int, str, bool] | None = None
    display_math = False
    footnote_continuation: str | None = None

    def close_quote() -> None:
        nonlocal in_quote, quote_callout
        if in_quote:
            out.append("#+END_QUOTE")
            in_quote = False
            quote_callout = None

    for line_number, line in enumerate(lines, 1):
        stripped = line.rstrip()

        if line.startswith(_RAW_LINE_PREFIX):
            close_quote()
            out.append(line[len(_RAW_LINE_PREFIX):])
            continue

        if fence is not None:
            marker, start, block_type, quoted = fence
            fence_line = stripped
            if quoted:
                quoted_line = re.match(r"^>\s?(.*)$", stripped)
                if quoted_line is None:
                    raise UnsupportedMarkdownError(
                        f"line {line_number}: quoted fenced block lost its quote prefix"
                    )
                fence_line = quoted_line.group(1)
            closing_fence = rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*$"
            if re.match(closing_fence, fence_line):
                out.append(f"#+END_{block_type}")
                fence = None
            else:
                out.append(fence_line if quoted else line)
            continue

        fence_match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", stripped)
        if fence_match:
            close_quote()
            marker = fence_match.group(1)
            info = fence_match.group(2).strip()
            if marker[0] == "`" and "`" in info:
                raise UnsupportedMarkdownError(
                    f"line {line_number}: backticks are invalid in a backtick-fence info string"
                )
            if info:
                language = info.split()[0].strip("{}.")
                if not re.fullmatch(r"[A-Za-z0-9_+.-]+", language):
                    raise UnsupportedMarkdownError(
                        f"line {line_number}: unsupported fenced-block info string {info!r}"
                    )
                out.append(f"#+BEGIN_SRC {language}")
                block_type = "SRC"
            else:
                out.append("#+BEGIN_EXAMPLE")
                block_type = "EXAMPLE"
            fence = (marker, line_number, block_type, False)
            continue

        if stripped.startswith(_MATH_LABEL_PREFIX):
            close_quote()
            label = stripped.removeprefix(_MATH_LABEL_PREFIX)
            if not _SAFE_ORG_NAME.fullmatch(label):
                raise UnsupportedMarkdownError(
                    f"line {line_number}: invalid MyST math label {label!r}"
                )
            out.extend([f"<<{label}>>", f"#+NAME: {label}"])
            continue

        if stripped.strip() == "$$":
            close_quote()
            out.append("\\[" if not display_math else "\\]")
            display_math = not display_math
            continue
        if display_math:
            out.append(line)
            continue

        myst_anchor = re.fullmatch(r"\s*\(([A-Za-z0-9_.:-]+)\)=\s*", stripped)
        if myst_anchor:
            close_quote()
            out.append(f"<<{myst_anchor.group(1)}>>")
            continue

        # A raw, empty `<a id="...">` anchor is a narrow, well-defined HTML
        # subset used purely as a cross-reference target (equivalent to a
        # MyST `(label)=` anchor) -- not general raw HTML.
        html_anchor = re.fullmatch(
            r'\s*<a\s+id="([A-Za-z0-9_.:-]+)"\s*>\s*</a>\s*', stripped
        )
        if html_anchor:
            close_quote()
            out.append(f"<<{html_anchor.group(1)}>>")
            continue

        _unsupported_markdown(stripped, line_number)

        continuation = re.match(r"^(?: {2,}|\t)(.*)$", line)
        if footnote_continuation and continuation:
            out.append("  " + _inline_md_to_org(continuation.group(1)))
            continue
        footnote_continuation = None

        footnote = re.match(r"^\[\^([A-Za-z0-9_-]+)\]:\s*(.*)$", stripped)
        if footnote:
            close_quote()
            label, body = footnote.groups()
            out.append(f"[fn:{label}] {_inline_md_to_org(body)}")
            footnote_continuation = label
            continue

        heading = re.match(r"^(#{1,6})\s+(.*?)(?:\s+#+)?$", stripped)
        if heading:
            close_quote()
            out.append("*" * len(heading.group(1)) + " " + _inline_md_to_org(heading.group(2)))
            continue

        if re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,})\s*", stripped):
            close_quote()
            out.append("-----")
            continue

        quote = re.match(r"^>\s?(.*)$", stripped)
        if quote:
            body = quote.group(1)
            if body.startswith(">"):
                raise UnsupportedMarkdownError(
                    f"line {line_number}: nested blockquotes require Pandoc"
                )
            callout = re.match(r"^\[!([A-Za-z][A-Za-z0-9_-]*)\]\s*(.*)$", body)
            if not in_quote:
                label = callout.group(1).upper() if callout else None
                if label:
                    out.append(f"#+ATTR_ORG: :callout {label}")
                out.append("#+BEGIN_QUOTE")
                in_quote = True
                quote_callout = label

            quote_fence = re.match(r"^\s*(`{3,}|~{3,})(.*)$", body)
            if quote_fence:
                marker = quote_fence.group(1)
                info = quote_fence.group(2).strip()
                if marker[0] == "`" and "`" in info:
                    raise UnsupportedMarkdownError(
                        f"line {line_number}: backticks are invalid in a backtick-fence info string"
                    )
                if info:
                    language = info.split()[0].strip("{}.")
                    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", language):
                        raise UnsupportedMarkdownError(
                            f"line {line_number}: unsupported fenced-block info string {info!r}"
                        )
                    out.append(f"#+BEGIN_SRC {language}")
                    block_type = "SRC"
                else:
                    out.append("#+BEGIN_EXAMPLE")
                    block_type = "EXAMPLE"
                fence = (marker, line_number, block_type, True)
                continue

            if callout:
                label, rest = callout.groups()
                out.append(f"*{label.upper()}*" + (f" — {_inline_md_to_org(rest)}" if rest else ""))
            else:
                out.append(_inline_md_to_org(body))
            continue
        close_quote()

        out.append(_inline_md_to_org(stripped))

    close_quote()
    if fence is not None:
        raise UnsupportedMarkdownError(f"line {fence[1]}: unterminated fenced code block")
    if display_math:
        raise UnsupportedMarkdownError("unterminated display-math block")
    return "\n".join(out)


def _inline_md_to_org(text: str) -> str:
    stash: list[str] = []

    def stash_value(value: str) -> str:
        stash.append(value)
        return f"\x00{len(stash) - 1}\x00"

    # Preserve CommonMark backslash escapes before applying emphasis/link rules.
    text = re.sub(
        r"\\([\\`*{}_\[\]()#+.!<>$|~-])",
        lambda match: stash_value(match.group(1)),
        text,
    )

    def stash_code(match: re.Match[str]) -> str:
        value = match.group(2)
        if value.startswith(" ") and value.endswith(" ") and value.strip():
            value = value[1:-1]
        if "=" not in value:
            return stash_value(f"={value}=")
        if "~" not in value:
            return stash_value(f"~{value}~")
        raise UnsupportedMarkdownError(
            "inline code containing both '=' and '~' requires Pandoc"
        )

    def stash_image(match: re.Match[str]) -> str:
        target = match.group(2)
        description = match.group(1) or match.group(3) or ""
        prefix = "" if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) else "file:"
        return stash_value(f"[[{prefix}{target}][{description}]]")

    def stash_citation(match: re.Match[str]) -> str:
        mode, raw_keys = match.groups()
        keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        if not keys:
            raise UnsupportedMarkdownError("empty MyST citation role")
        citations = "; ".join(f"@{key}" for key in keys)
        style = "/t" if mode == "t" else ""
        return stash_value(f"[cite{style}:{citations}]")

    text = re.sub(r"\{cite:([pt])\}`([^`]+)`", stash_citation, text)
    text = re.sub(
        r"\{math\}`([^`]+)`",
        lambda match: stash_value(f"${match.group(1)}$"),
        text,
    )
    text = re.sub(
        r"\{(?:ref|numref|eq)\}`([^`]+)`",
        lambda match: stash_value(f"[[{match.group(1)}]]"),
        text,
    )

    # CommonMark autolinks are not raw HTML and can be represented losslessly.
    text = re.sub(
        r"<([A-Za-z][A-Za-z0-9+.-]*://[^<>\s]+)>",
        lambda match: stash_value(f"[[{match.group(1)}]]"),
        text,
    )
    text = re.sub(
        r"<([^<>\s@]+@[^<>\s@]+)>",
        lambda match: stash_value(
            f"[[mailto:{match.group(1)}][{match.group(1)}]]"
        ),
        text,
    )

    # Images before links; both support the controlled, non-nested form used by
    # the book. Nested brackets/parentheses need Pandoc and are rejected below.
    text = re.sub(
        r"!\[([^\]]*)\]\(([^()\s]+)(?:\s+[\"']([^\"']+)[\"'])?\)",
        stash_image,
        text,
    )
    text = re.sub(r"\$\$[^$\n]+\$\$", lambda match: stash_value(match.group(0)), text)
    text = re.sub(r"\$[^$\n]+\$", lambda match: stash_value(match.group(0)), text)
    text = re.sub(
        r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", stash_code, text
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^()\s]+)(?:\s+[\"']([^\"']+)[\"'])?\)",
        lambda match: stash_value(f"[[{match.group(2)}][{match.group(1)}]]"),
        text,
    )
    text = re.sub(
        r"\[\^([A-Za-z0-9_-]+)\]", lambda match: f"[fn:{match.group(1)}]", text
    )

    # Stash converted emphasis too, so Org's bold markers are not consumed by
    # the subsequent Markdown italic pass.
    text = re.sub(
        r"\*\*\*([^*\n]+)\*\*\*",
        lambda match: stash_value(f"*/{match.group(1)}/*"),
        text,
    )
    text = re.sub(
        r"___([^_\n]+)___",
        lambda match: stash_value(f"*/{match.group(1)}/*"),
        text,
    )
    text = re.sub(
        r"\*\*([^*\n]+)\*\*",
        lambda match: stash_value(f"*{match.group(1)}*"),
        text,
    )
    text = re.sub(
        r"__([^_\n]+)__",
        lambda match: stash_value(f"*{match.group(1)}*"),
        text,
    )
    text = re.sub(
        r"(?<!\w)_([^_\s][^_\n]*?)_(?!\w)",
        lambda match: stash_value(f"/{match.group(1)}/"),
        text,
    )
    text = re.sub(
        r"(?<!\*)\*([^*\s][^*\n]*?)\*(?!\*)",
        lambda match: stash_value(f"/{match.group(1)}/"),
        text,
    )

    if re.search(r"!?\[[^\]]*\]\([^)]*[()]", text):
        raise UnsupportedMarkdownError("nested Markdown links require Pandoc")
    if "`" in text:
        raise UnsupportedMarkdownError("unmatched or multi-backtick code span requires Pandoc")

    def resolve_placeholder(index: int, active: frozenset[int] = frozenset()) -> str:
        if index in active:
            raise UnsupportedMarkdownError("cyclic inline placeholder")
        if index < 0 or index >= len(stash):
            raise UnsupportedMarkdownError("invalid inline placeholder")
        return _PLACEHOLDER.sub(
            lambda nested: resolve_placeholder(
                int(nested.group(1)), active | {index}
            ),
            stash[index],
        )

    text = _PLACEHOLDER.sub(
        lambda match: resolve_placeholder(int(match.group(1))), text
    )
    if "\x00" in text:
        raise UnsupportedMarkdownError("unresolved inline placeholder")
    return text


def _slug(value: str, *, fallback: str = "output", limit: int = 36) -> str:
    """Return a readable token plus a digest of the exact case-sensitive input."""

    original = value.strip()
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", original).strip("-._").lower()
    readable = (readable or fallback)[:limit]
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def _cell_output_name(cell: Mapping[str, Any]) -> str | None:
    metadata = cell.get("metadata", {})
    if isinstance(metadata, Mapping):
        zh = metadata.get("zh", {})
        if isinstance(zh, Mapping) and isinstance(zh.get("output_name"), str):
            return _slug(zh["output_name"])
        if isinstance(metadata.get("name"), str):
            return _slug(metadata["name"])
    return None


def _join_data(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _preferred_mime(data: Mapping[str, Any], *, include_text: bool) -> str | None:
    # A Jupyter MIME bundle contains alternative representations of one value.
    # Prefer vector graphics, then raster graphics, and use text only when no
    # supported image exists.
    order = ("image/svg+xml", "image/png", "image/jpeg")
    if include_text:
        order += ("text/plain",)
    return next((mime for mime in order if mime in data), None)


def _extract_outputs_from_notebook(
    nb: nbformat.NotebookNode,
    out_dir: os.PathLike[str] | str,
    prefix: str,
    *,
    include_text: bool,
) -> dict[str, list[dict[str, Any]]]:
    destination = Path(out_dir)
    safe_prefix = _slug(prefix, fallback="notebook")
    result: dict[str, list[dict[str, Any]]] = {}
    planned: list[tuple[Path, str, str, bool, str]] = []
    planned_names: set[str] = set()

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell_id = str(cell.id)
        identity = _slug(cell_id)
        semantic = _cell_output_name(cell)
        stable_stem = identity if semantic is None else f"{identity}-{semantic}"
        artifacts: list[dict[str, Any]] = []
        image_count = 0
        text_count = 0
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                if include_text:
                    artifacts.append(
                        {
                            "kind": "text",
                            "mime": "text/plain",
                            "stream": output.get("name", "stdout"),
                            "text": _join_data(output.get("text", "")),
                        }
                    )
                continue
            if output_type == "error":
                if include_text:
                    traceback = output.get("traceback") or []
                    text = "\n".join(traceback) if traceback else (
                        f"{output.get('ename', 'Error')}: {output.get('evalue', '')}".rstrip()
                    )
                    artifacts.append(
                        {"kind": "error", "mime": "text/plain", "text": text}
                    )
                continue
            if output_type not in {"display_data", "execute_result"}:
                continue
            data = output.get("data", {})
            if not isinstance(data, Mapping):
                continue
            mime = _preferred_mime(data, include_text=include_text)
            if mime is None:
                continue
            payload = _join_data(data[mime])
            if mime in SUPPORTED_IMAGE_MIMES:
                image_count += 1
                extension, binary = SUPPORTED_IMAGE_MIMES[mime]
                role = "figure" if image_count == 1 else f"figure-{image_count}"
                filename = f"{safe_prefix}-{stable_stem}-{role}.{extension}"
                record = {
                    "kind": "image",
                    "mime": mime,
                    "path": str(Path(destination.name) / filename),
                    "name": role,
                }
            else:
                text_count += 1
                binary = False
                role = "result" if text_count == 1 else f"result-{text_count}"
                filename = f"{safe_prefix}-{stable_stem}-{role}.txt"
                record = {
                    "kind": "text",
                    "mime": mime,
                    "text": payload,
                    "path": str(Path(destination.name) / filename),
                    "name": role,
                }
            if filename in planned_names:
                raise NotebookValidationError(
                    f"output filename collision before write: {filename}"
                )
            planned_names.add(filename)
            planned.append((destination / filename, payload, mime, binary, cell_id))
            artifacts.append(record)
        if artifacts:
            result[cell_id] = artifacts

    existing = [target for target, *_ in planned if target.exists()]
    if existing:
        raise NotebookValidationError(
            "refusing to overwrite existing extracted output(s): "
            + ", ".join(str(path) for path in existing)
        )
    if planned:
        destination.mkdir(parents=True, exist_ok=True)
    for target, payload, mime, binary, cell_id in planned:
        if binary:
            try:
                compact_payload = re.sub(r"\s+", "", payload)
                target.write_bytes(base64.b64decode(compact_payload, validate=True))
            except Exception as exc:
                raise NotebookValidationError(
                    f"cell {cell_id}: invalid {mime} payload"
                ) from exc
        else:
            target.write_text(payload, encoding="utf-8")
    return result


def extract_outputs(
    executed_ipynb_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    prefix: str,
    *,
    include_text: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Extract one preferred representation from each Jupyter MIME bundle."""

    nb = validate_notebook(executed_ipynb_path, allow_errors=True)
    return _extract_outputs_from_notebook(
        nb, out_dir, prefix, include_text=include_text
    )


def _lookup_outputs(
    outputs_by_cell: Mapping[Any, Sequence[Any]], index: int, cell_id: str
) -> Sequence[Any]:
    if cell_id in outputs_by_cell:
        return outputs_by_cell[cell_id]
    return outputs_by_cell.get(index, ())


def _org_result_lines(records: Iterable[Any]) -> list[str]:
    lines: list[str] = []
    for record in records:
        if isinstance(record, (str, os.PathLike)):
            lines.append(f"[[file:{record}]]")
            continue
        if not isinstance(record, Mapping):
            raise TypeError(f"unsupported output record: {record!r}")
        kind = record.get("kind")
        if kind == "image":
            lines.append(f"[[file:{record['path']}]]")
            continue
        if kind in {"text", "error"}:
            label = "error" if kind == "error" else record.get("stream", "output")
            lines.extend([f"#+RESULTS: {label}", "#+BEGIN_EXAMPLE"])
            lines.extend(_join_data(record.get("text", "")).rstrip("\n").split("\n"))
            lines.append("#+END_EXAMPLE")
            continue
        raise TypeError(f"unsupported output kind: {kind!r}")
    return lines


def _org_block_names(cell: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return a primary Org block name and stable target aliases."""

    metadata = cell.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise CellValidationError(f"cell {cell.get('id')}: metadata must be a mapping")
    explicit = metadata.get("name")
    if explicit is None:
        primary = str(cell["id"])
    elif isinstance(explicit, str) and _SAFE_ORG_NAME.fullmatch(explicit):
        primary = explicit
    else:
        raise CellValidationError(
            f"cell {cell.get('id')}: metadata.name must match {_SAFE_ORG_NAME.pattern!r}"
        )
    raw_aliases = metadata.get("source_names", ())
    if isinstance(raw_aliases, str):
        raw_aliases = (raw_aliases,)
    if not isinstance(raw_aliases, Sequence) or isinstance(raw_aliases, (bytes, bytearray)):
        raise CellValidationError(
            f"cell {cell.get('id')}: metadata.source_names must be a string sequence"
        )
    aliases: list[str] = []
    for value in (str(cell["id"]), *raw_aliases):
        if not isinstance(value, str) or not _SAFE_ORG_NAME.fullmatch(value):
            raise CellValidationError(
                f"cell {cell.get('id')}: invalid Org source name {value!r}"
            )
        if value != primary and value not in aliases:
            aliases.append(value)
    return primary, tuple(aliases)


def write_org(
    cells: Sequence[Mapping[str, Any]],
    path: os.PathLike[str] | str,
    title: str,
    images_by_cell: Mapping[Any, Sequence[Any]] | None = None,
    *,
    outputs_by_cell: Mapping[Any, Sequence[Any]] | None = None,
    markdown_converter: str = "auto",
) -> None:
    """Write Org source, including extracted code results when provided.

    ``images_by_cell`` retains the legacy index-keyed API.  New callers should
    pass ``outputs_by_cell`` keyed by stable cell id as returned by
    :func:`extract_outputs`.
    """

    normalized = normalize_cells(cells)
    images_by_cell = images_by_cell or {}
    outputs_by_cell = outputs_by_cell or {}
    block_names: dict[str, tuple[str, tuple[str, ...]]] = {}
    claimed_names: dict[str, str] = {}
    for cell in normalized:
        if cell["type"] != "code":
            continue
        primary, aliases = _org_block_names(cell)
        for name in (primary, *aliases):
            previous = claimed_names.get(name)
            if previous is not None and previous != cell["id"]:
                raise CellValidationError(
                    f"Org source name {name!r} is shared by cells {previous!r} and {cell['id']!r}"
                )
            claimed_names[name] = cell["id"]
        block_names[cell["id"]] = (primary, aliases)
    lines = [
        f"#+TITLE: {title}",
        "#+LANGUAGE: zh-CN",
        "#+OPTIONS: toc:3 num:2",
        "#+STARTUP: showall",
        "",
    ]
    for index, cell in enumerate(normalized):
        if cell["type"] == "markdown":
            lines.append(md_to_org(cell["source"], converter=markdown_converter))
            lines.append("")
            continue

        primary, aliases = block_names[cell["id"]]
        lines.extend(f"<<{alias}>>" for alias in aliases)
        lines.extend(
            [
                f"#+NAME: {primary}",
                "#+BEGIN_SRC python",
                cell["source"].rstrip("\n"),
                "#+END_SRC",
            ]
        )
        records: list[Any] = []
        records.extend(_lookup_outputs(images_by_cell, index, cell["id"]))
        records.extend(_lookup_outputs(outputs_by_cell, index, cell["id"]))
        lines.extend(_org_result_lines(records))
        lines.append("")

    _ensure_parent(path)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def extract_images(
    executed_ipynb_path: os.PathLike[str] | str,
    out_dir: os.PathLike[str] | str,
    prefix: str,
) -> dict[int, list[str]]:
    """Compatibility wrapper returning index-keyed image paths.

    Filenames are nevertheless stable and semantic; they never consist only of
    the mutable notebook cell index.
    """

    nb = validate_notebook(executed_ipynb_path, allow_errors=True)
    outputs = _extract_outputs_from_notebook(
        nb, out_dir, prefix, include_text=False
    )
    result: dict[int, list[str]] = {}
    for index, cell in enumerate(nb.cells):
        images = [
            record["path"]
            for record in outputs.get(cell.id, ())
            if record.get("kind") == "image"
        ]
        if images:
            result[index] = images
    return result


def count_errors(executed_ipynb_path: os.PathLike[str] | str) -> int:
    nb = validate_notebook(executed_ipynb_path, allow_errors=True)
    return sum(
        1
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    )
